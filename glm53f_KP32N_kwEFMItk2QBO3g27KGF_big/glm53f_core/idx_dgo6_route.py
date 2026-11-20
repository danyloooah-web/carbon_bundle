"""idx_dgo6_route.py: r60 IDX-ALT (DGO6) -- OUR component on the v32 king family: the TP4 decode engine's
target-verify DSA indexer logits (paged MQA, next_n = 6) by OUR CuTe-DSL kernel DGO6 (idx_dgo6.py, the r50 FINALI
sidecar byte for byte) instead of DeepGEMM's legacy sm100_paged_mqa_logits<6, ...>. BITWISE, proved at boot and voted
over the TP group; the top-k stays stock. No native unit (CuTe DSL, compiled at prepare).

What runs without this component (SGLang 314bce7c + the v32 king): in every target-verify forward each of the 11 DSA
layers' indexer is served by idx_nextn_route (served() / _split_call()), which calls its module global
``deepgemm_paged_mqa_logits_native`` (SGLang's paged_mqa_logits.py wrapper, imported by name, looked up at call time)
with deep_gemm.fp8_paged_mqa_logits: q rows [B * 6, 32, 128] e4m3, the index-K pages [pages, 64, 1, 132], the head
weights [B * 6, 32] fp32, the per-row pooled lengths [B, 6] int32, the per-row pooled page table (one row a query row),
the DeepGEMM schedule over the lengths .clamp(min=1) at the indexer's SM count, max_len = table width x 64; the
wrapper passes q[:q_offset].view(B, 6, 32, 128), weights[:q_offset] and table[::6] with clean_logits=False. Then
indexer._topk_from_kpool_logits reads each row's columns [0, its pooled length) (the same lengths). Under the king's row
split (idx_nextn_route._split / idx_ownq_route) each TP4 rank scores its own whole requests.

What this component does: listed in components.ACTIVE right after idx_nextn_route (NODE "dsa"), its routed() sets that
module global, for one target-verify call of a claimed DSA module (inside CUDA-graph captures too), to ``_native``: the
stock wrapper's exact signature; for a call inside the proved envelope (fn is deep_gemm.fp8_paged_mqa_logits, next_n 6,
1 <= B <= the largest proved B, q_offset = 6 B, row-contiguous 32 x 128 e4m3 query rows, row-contiguous fp32 weights,
contiguous int32 [B, 6] lengths, one int32 table row a query row with unit column stride, max_len = width x 64,
contiguous uint8 64-token pages, one device, the compiled variant present) it returns DGO6's logits on the same views
(q[:q_offset].view(B, 6, 32, 128), weights[:q_offset]) with the request's context = its LAST row's pooled length (the
widest: the pooled write plan's (start + k + 1) // 4 and the expanded seqlens // 4 are monotone in k; the r50 line's
form), a contiguous copy of table[::6] and DeepGEMM's legacy metadata over those contexts [B, 1] at the compiled SM
count (the stock CuTe wrapper's native next_n form); the king's schedule argument is not used. The three prologue
tensors are built once a forward (idx_nextn_route._schedule's memo rule: the same batch object, a later layer, the same
capture state, the same lengths / table views). Every other call is the stock wrapper's, with the caller's arguments. The value replaced is put
back after the call (finally); if the name is not SGLang's function when a call starts (another route holds it), this
route leaves it alone for that call (chaining would bypass that route).

Exactness contract: DGO6 = SGLang 93a5b07's CuTe DSL FP8 paged-MQA kernel with ONE change, the fp32 epilogue's final
4-way combine in DeepGEMM legacy's association ((s0x + s1x) + (s0y + s1y)) * 0.5 (gen_idxdgo.py); every logit of a row
r's columns [0, len_r) is then the same fp32 word as DeepGEMM's (r45 K7 kit idxF45 on box9: every valid column of every
row, random + cancellation families, 96..128 requests; the r50 FINALI / r54 FINALM0s lifetimes: boot proof ON and the
census bitwise on every rank). Columns at or past a row's length are not read by the top-k (clean_logits=False:
DeepGEMM leaves them unspecified as well). Proof (fail closed to DeepGEMM, never to garbage), once a process at the FIRST
claimed DSA module's prepare (= that module's first call: the first eager forward through it; never inside a capture or
a scored window), every rank:
  * gates (rank-uniform facts, not voted): IDX_ALT, disaggregation mode "decode", a TP group and an attention-TP group
    of TP_WORLD ranks, the indexer on a CUDA device with 32 x 128 heads, idx_nextn_route's NEXT_N / BLOCK and its global
    still SGLang's function, deep_gemm's two entries, not capturing; otherwise this component is inert on the engine
    (prefill TP2, the audit's colocated TP2, anything else: the king's calls exactly, nothing compiled);
  * DGO6 compiled for (64-token pages, 32 heads, 128 dims, next_n 6, the indexer's SM count): the only compile;
  * PROOF_CASES on bundle-owned seeded operands (never the live pools): families rand / real (UE8M0 power-of-two page
    scales, mixed-sign weights of spread magnitudes) / probe (partial sums cancelling inside the 128-dot and across the
    head-weighted sum), lengths kpool (the pooled write rule (start + k + 1) // 4, 1 request in 4 short) / edge (0, 1,
    2, 63..65, 127..129, 255..257, width x 64 - 1, width x 64) / zero (requests of length 0), all monotone per request
    as live, B 1..144 requests (cases over the SM count skipped: DeepGEMM legacy itself needs B <= it, SGLang
    dsa_indexer.py:1485), tables 64..832 pages wide, unsplit and row-split (this rank's share of TP_WORLD x B
    requests, the king's row slices), uint8 pages (the DSA pool's index_k_with_scale_buffer_dtype): DGO6 through
    _native's own code == the stock wrapper on deep_gemm.fp8_paged_mqa_logits with the king's schedule form, as int32
    words on every row's valid columns; DGO6 twice == itself; the same shape, fp32, unit column stride and a row
    stride the fused top-k accepts;
  * case GRAPH_CASE also captured (thread-local mode, side stream: prologue + launch) and replayed into a NaN-filled
    output == DeepGEMM's eager logits (capture safety: no host sync, no compile, the TVM-FFI env stream);
  * every rank's record all-gathered over the TP group (CPU group): ON only when every rank passed with columns compared;
    any miss / exception anywhere -> OFF on every rank, and routed() never sets the name (the king's calls exactly).
Serving faults: a call outside the envelope is the stock wrapper's; a compile variant missing inside a capture
(DgoFresh) -> the stock wrapper in that graph; any exception from our prologue / launch -> the stock wrapper, and this
rank stops DGO6 for the process (rank-local is safe: both paths give the same bytes and neither takes a collective).
Census (WARNING; this logger keeps WARNING under the arena's log_level error), "idxalt: ..." lines:
  boot   "idxalt: boot proof rank R/4 (T s, compile C s, sms S): DGO6 == DeepGEMM legacy logits ... -> ON on every rank"
  OFF    "idxalt: OFF on every rank (boot proof: rank R: ...): DeepGEMM's logits from here"
  inert  "idxalt: inert on this engine (...): DeepGEMM's logits (the king's calls exactly)"
  served "idxalt: served N (Python-level launches: eager calls + graph captures; C inside captures)"
Switch: IDX_ALT False = this component does nothing (the base bundle's behaviour exactly)."""
from __future__ import annotations

import contextlib
import logging
import time

import deep_gemm
import torch
from sglang.kernels.ops.attention.dsa.paged_mqa_logits import deepgemm_paged_mqa_logits_native as _STOCK_NATIVE
from sglang.srt.distributed import parallel_state
from sglang.srt.runtime_context import get_disagg

from glm53f_core import idx_dgo6, idx_nextn_route

NODE = "dsa"
IDX_ALT = True               # False = this component does nothing (the base bundle's behaviour exactly)
NEXT_N, HEADS, DIM, BLOCK = 6, 32, 128, 64
ROLE, TP_WORLD = "decode", 4
SEED = 20261120              # the proof's generator seed
FP8 = torch.float8_e4m3fn
# The boot proof: (family, requests, lengths, table width in 64-token pages, row split). Peak scratch ~260 MB.
PROOF_CASES = (
    ("rand", 24, "kpool", 640, True),       # A96-like share: 96 requests / 4 ranks, 82K-164K-token contexts
    ("rand", 32, "kpool", 832, True),       # B128-like share, the widest table (53,248 pooled columns)
    ("probe", 32, "kpool", 640, False),     # cancellation
    ("real", 27, "kpool", 512, False),      # an unsplit call under the split floor (162 rows)
    ("real", 1, "kpool", 64, False),        # one request
    ("real", 14, "edge", 64, False),        # page / KV-split boundaries
    ("rand", 14, "edge", 96, True),
    ("rand", 8, "zero", 128, False),        # requests of length 0 beside live ones
    ("real", 16, "kpool", 256, True),
    ("rand", 64, "kpool", 160, False),
    ("probe", 128, "kpool", 128, False),
    ("rand", 144, "kpool", 128, False),     # the largest B served (DeepGEMM legacy itself needs B <= the SM count)
)
TIME_CASES, TIME_REPS = (0, 1), 7  # r60: the live-band shares timed at boot (DeepGEMM vs DGO6, CUDA events)
GRAPH_CASE = 0               # the PROOF_CASES entry also captured in a CUDA graph and replayed
LOG_AT = (1, 11, 110, 1100)  # served-count census points (and every 4096th)
_DEV_TYPES = ("cuda",)       # device types served (the CPU suite's seam)

_ST = {"decided": False, "on": None, "off": None, "bad": None, "sms": None, "key": None, "dev": None, "max_b": 0,
       "served": 0, "captured": 0, "votes": None}
_CALL: list = []             # (forward_batch, layer id) of the routed call in flight
_MEMO: dict = {}             # the per-forward prologue: batch, layer, capturing, views key, (ctx, schedule, table)
_SAID: set = set()
_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)


def _say(tag: str, msg: str) -> None:
    """One census line a kind of event."""
    if tag not in _SAID:
        _SAID.add(tag)
        _log.warning("%s", msg)


def _need(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(why)


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001 - no CUDA state: treated as a capture (never compile / prove there)
        return True


# ------------------------------------------------------------------------------------------------ the served call
def _prologue(lens_2d, table, memo: bool):
    """(ctx [B] int32 = each request's last (widest) row, DeepGEMM's legacy metadata over ctx[:, None] at the compiled
    SM count, a contiguous copy of table[::6]); memo: once a forward (idx_nextn_route._schedule's rule)."""
    key = (lens_2d.data_ptr(), tuple(lens_2d.shape), tuple(lens_2d.stride()), table.data_ptr(), tuple(table.shape),
           tuple(table.stride()))
    capturing = _capturing()
    batch, layer = _CALL[-1] if (memo and _CALL) else (None, None)
    held = _MEMO
    if (memo and batch is not None and layer is not None and held and held["batch"] is batch
            and layer > held["layer"] and held["capturing"] == capturing and held["key"] == key):
        held["layer"] = layer
        return held["p"]
    ctx = lens_2d[:, -1].contiguous()
    schedule = deep_gemm.get_paged_mqa_logits_metadata(ctx.unsqueeze(-1), BLOCK, _ST["sms"])
    tab = table[::NEXT_N].contiguous()
    p = (ctx, schedule, tab)
    if memo:
        _MEMO.clear()
        if batch is not None and layer is not None:
            _MEMO.update(batch=batch, layer=layer, capturing=capturing, key=key, p=p)
    return p


def _dgo(q_fp8, kv_cache_fp8, weights, ctx_lens_2d, block_tables, max_seq_len, q_offset, B, memo: bool):
    """DGO6's logits [B * 6, max_len] (fp32 view of a 256-aligned allocation) for the stock wrapper's arguments."""
    ctx, schedule, tab = _prologue(ctx_lens_2d, block_tables, memo)
    return idx_dgo6.dgo_logits(q_fp8[:q_offset].view(B, NEXT_N, HEADS, DIM), kv_cache_fp8, weights[:q_offset], ctx, tab,
                               schedule, max_seq_len, _ST["sms"])


def _envelope(q, kv, w, lens, table, max_len, q_offset, B):
    """None for operands of the proved class, else (kind, text). Host metadata only (no device sync)."""
    if not (type(B) is int and 1 <= B <= _ST["max_b"] and q_offset == B * NEXT_N):
        return "requests", f"{B} requests, q_offset {q_offset}: outside 1..{_ST['max_b']} x {NEXT_N} rows"
    if not (q.dtype == FP8 and q.dim() == 3 and tuple(q.shape[1:]) == (HEADS, DIM) and q.shape[0] >= q_offset
            and tuple(q.stride()) == (HEADS * DIM, DIM, 1)):
        return "query", f"query {q.dtype} {tuple(q.shape)} strides {tuple(q.stride())}"
    if not (kv.dim() == 4 and tuple(kv.shape[1:]) == (BLOCK, 1, DIM + 4) and kv.dtype == torch.uint8
            and kv.is_contiguous()):
        return "pages", f"index-K pages {kv.dtype} {tuple(kv.shape)}"
    if not (w.dtype == torch.float32 and w.dim() == 2 and w.shape[1] == HEADS and w.shape[0] >= q_offset
            and tuple(w.stride()) == (HEADS, 1)):
        return "weights", f"weights {w.dtype} {tuple(w.shape)} strides {tuple(w.stride())}"
    if not (lens.dtype == torch.int32 and tuple(lens.shape) == (B, NEXT_N) and lens.is_contiguous()):
        return "lengths", f"lengths {lens.dtype} {tuple(lens.shape)}"
    if not (table.dtype == torch.int32 and table.dim() == 2 and table.shape[0] == q_offset and table.stride(1) == 1
            and max_len == table.shape[1] * BLOCK):
        return "table", f"table {table.dtype} {tuple(table.shape)} for {q_offset} rows, max_len {max_len}"
    if not all(t.device == _ST["dev"] for t in (q, kv, w, lens, table)):
        return "device", "an operand on another device"
    return None


def _outside(fn, q, kv, w, lens, table, max_len, q_offset, B, next_n):
    """None for a call DGO6 serves, else (kind, text): on, deep_gemm's function, next_n 6, the proved operand class
    (_envelope), the compiled variant present."""
    if _ST["on"] is not True or _ST["bad"] is not None:
        return "off", "this component is not on (or this rank stopped DGO6)"
    if fn is not deep_gemm.fp8_paged_mqa_logits:
        return "fn", "the logits function is not deep_gemm.fp8_paged_mqa_logits"
    if next_n != NEXT_N:
        return "next_n", f"next_n {next_n}, not {NEXT_N}"
    why = _envelope(q, kv, w, lens, table, max_len, q_offset, B)
    if why is not None:
        return why
    if _ST["key"] not in idx_dgo6._DGO_COMPILED:
        return "variant", "the compiled variant is missing"
    return None


def _count(out, capturing: bool):
    _ST["served"] += 1
    _ST["captured"] += int(capturing)
    n = _ST["served"]
    if n in LOG_AT or n % 4096 == 0:
        _log.warning("idxalt: served %d (Python-level launches: eager calls + graph captures; %d inside captures)", n,
                     _ST["captured"])
    return out


def _native(fp8_paged_mqa_logits_fn, q_fp8, kv_cache_fp8, weights, ctx_lens_2d, block_tables, schedule_metadata,
            max_seq_len, *, q_offset, B, next_n):
    """idx_nextn_route.deepgemm_paged_mqa_logits_native while routed() is live (SGLang's signature exactly): DGO6 for a
    call inside the proved envelope (schedule_metadata unused: DGO6 runs on its own), the stock wrapper otherwise."""
    why = _outside(fp8_paged_mqa_logits_fn, q_fp8, kv_cache_fp8, weights, ctx_lens_2d, block_tables, max_seq_len,
                   q_offset, B, next_n)
    if why is None:
        capturing = _capturing()
        try:
            out = _dgo(q_fp8, kv_cache_fp8, weights, ctx_lens_2d, block_tables, max_seq_len, q_offset, B, True)
        except idx_dgo6.DgoFresh as e:
            _say("fresh", f"idxalt: {e} needed inside a CUDA-graph capture: DeepGEMM's logits in that graph")
        except Exception as e:  # noqa: BLE001 - rank-local stop (same bytes either way, no collective)
            _ST["bad"] = f"{type(e).__name__}: {str(e)[:300]}"
            _log.warning("idxalt: DGO6 raised on this rank (%s): DeepGEMM's logits on this rank from here (the same "
                         "bytes; neither path takes a collective)", _ST["bad"])
        else:
            return _count(out, capturing)
    elif why[0] != "off":
        _say("out " + why[0], f"idxalt: DeepGEMM serves a call: {why[1]}")
    return _STOCK_NATIVE(fp8_paged_mqa_logits_fn, q_fp8, kv_cache_fp8, weights, ctx_lens_2d, block_tables,
                         schedule_metadata, max_seq_len, q_offset=q_offset, B=B, next_n=next_n)


# ------------------------------------------------------------------------------------------------ the boot proof
def _f8(x):
    return x.clamp(-448, 448).to(FP8)


def _operands(fam: str, n: int, pages: int, gen, dev):
    """Seeded q [n, 32, 128] e4m3, index-K pages [pages, 64, 1, 132] uint8 (64 x 128 e4m3 values, then 64 fp32 scales:
    the pool's fused page layout) and head weights [n, 32, 1] fp32 (the indexer's form; the king squeezes dim 2)."""
    def rn(*shape):
        return torch.randn(*shape, generator=gen, device=dev)

    half = DIM // 2
    if fam == "probe":
        r = rn(pages, BLOCK, half) * 64
        kvd = _f8(torch.cat([r, r + rn(pages, BLOCK, half) * 2], dim=-1))
        base = rn(n, HEADS, half) * 32
        q = _f8(torch.cat([base, -base + rn(n, HEADS, half)], dim=-1))
        w = torch.tensor([1.0, -1.0], device=dev).repeat(HEADS // 2)[None, :] * (1 + 1e-3 * rn(n, HEADS))
        sc = torch.rand(pages, BLOCK, generator=gen, device=dev) * 0.05 + 0.005
        del r, base
    elif fam == "real":
        kvd = _f8(rn(pages, BLOCK, DIM) * 64)
        q = _f8(rn(n, HEADS, DIM) * 48)
        w = rn(n, HEADS) * torch.pow(2.0, torch.randint(-6, 1, (n, HEADS), generator=gen, device=dev).float())
        sc = torch.pow(2.0, torch.randint(-9, -2, (pages, BLOCK), generator=gen, device=dev).float())
    else:
        kvd = _f8(rn(pages, BLOCK, DIM) * 8)
        q = _f8(rn(n, HEADS, DIM) * 8)
        w = rn(n, HEADS)
        sc = torch.rand(pages, BLOCK, generator=gen, device=dev) * 0.05 + 0.005
    kv = torch.empty((pages, BLOCK * (DIM + 4)), dtype=torch.uint8, device=dev)
    kv[:, :BLOCK * DIM] = kvd.contiguous().view(torch.uint8).reshape(pages, BLOCK * DIM)
    kv[:, BLOCK * DIM:] = sc.to(torch.float32).contiguous().view(torch.uint8).reshape(pages, BLOCK * 4)
    return q.contiguous(), kv.view(pages, BLOCK, 1, DIM + 4), w.to(torch.float32).unsqueeze(-1).contiguous()


def _lens(form: str, b: int, width: int, gen, dev):
    """[b, 6] int32 per-row pooled lengths in [0, width], monotone in the row (as the pooled write plan's)."""
    hi = max(8, width * 4 - 4 * NEXT_N)
    start = torch.randint(hi // 2, hi, (b, 1), generator=gen, device=dev)
    if b > 1:
        start[::4] = torch.randint(1, max(2, hi // 8), (len(range(0, b, 4)), 1), generator=gen, device=dev)
    lens = (start + torch.arange(1, NEXT_N + 1, device=dev)[None, :]) // 4
    if form == "edge":
        pts = torch.tensor([0, 1, 2, 63, 64, 65, 127, 128, 129, 255, 256, 257, width - 1, width], device=dev)
        last = pts[torch.arange(b, device=dev) % pts.numel()].clamp(0, width)
        lens = (last[:, None] - torch.arange(NEXT_N - 1, -1, -1, device=dev)[None, :] // 3).clamp(min=0)
    elif form == "zero":
        lens[::2] = 0
    return lens.clamp(0, width).to(torch.int32).contiguous()


def _table(b: int, cols: int, pages: int, gen, dev):
    """The king's per-row pooled table form: [b * 6, cols] int32, a request's 6 rows equal, distinct random pages."""
    per = torch.stack([torch.randperm(pages, generator=gen, device=dev)[:cols] for _ in range(b)])
    return per.to(torch.int32).repeat_interleave(NEXT_N, dim=0).contiguous()


def _legacy(q, kv, w, lens_2d, table, max_len, n, b, sms):
    """The king's call (idx_nextn_route's statements): the stock wrapper on deep_gemm.fp8_paged_mqa_logits with the
    DeepGEMM schedule over lens.clamp(min=1) at the indexer's SM count."""
    schedule = deep_gemm.get_paged_mqa_logits_metadata(lens_2d.clamp(min=1), BLOCK, sms)
    return _STOCK_NATIVE(deep_gemm.fp8_paged_mqa_logits, q, kv, w, lens_2d, table, schedule, max_len, q_offset=n, B=b,
                         next_n=NEXT_N)


def _same(got, want, lens_2d):
    """(ok, valid columns): int32 words equal on every row's columns [0, its length); the logits' class (fp32, 2-D, the
    same shape, unit column stride, a row stride the fused top-k accepts: % 4 == 0)."""
    rows = lens_2d.reshape(-1).long()
    if (got.dtype != torch.float32 or want.dtype != torch.float32 or got.dim() != 2
            or tuple(got.shape) != tuple(want.shape) or got.shape[0] != rows.numel() or got.stride(1) != 1
            or got.stride(0) % 4 or bool((rows > got.shape[1]).any())):
        return False, 0
    valid = torch.arange(got.shape[1], device=got.device)[None, :] < rows[:, None]
    a = torch.where(valid, got, torch.zeros_like(got)).view(torch.int32)
    b = torch.where(valid, want, torch.zeros_like(want)).view(torch.int32)
    return bool(torch.equal(a, b)), int(valid.sum())


def _graph(args, ref, lens_2d, dev) -> bool:
    """_dgo's prologue + launch captured in a CUDA graph (thread-local mode, side stream), replayed into a NaN-filled
    output: == ref on every valid column. The graph is released after the read."""
    cur = torch.cuda.current_stream(dev)
    s = torch.cuda.Stream(device=dev)
    s.wait_stream(cur)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        g.capture_begin(capture_error_mode="thread_local")
        try:
            out = _dgo(*args, False)
        finally:
            g.capture_end()
    cur.wait_stream(s)
    out.fill_(float("nan"))
    g.replay()
    ok = _same(out, ref, lens_2d)[0]
    del out, g
    return ok


def _time_us(fn):
    """r60: median CUDA-event time (us) of fn() over TIME_REPS after 2 warm calls; None without CUDA (CPU suites)."""
    if not torch.cuda.is_available():
        return None
    fn()
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(TIME_REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    ts.sort()
    return round(ts[len(ts) // 2], 1)


def _case(i: int, case, rank: int, gen, dev, sms: int):
    """(ok, text, valid columns) of one PROOF_CASES entry."""
    fam, b, form, cols, split = case
    world = TP_WORLD if split else 1
    reqs, pages, n = b * world, cols + 16, b * NEXT_N
    q, kv, w3 = _operands(fam, reqs * NEXT_N, pages, gen, dev)
    lens_all = _lens(form, reqs, cols * BLOCK, gen, dev)
    table = _table(reqs, cols, pages, gen, dev)
    rows = slice(rank * n, (rank + 1) * n) if split else slice(0, n)
    q_r, w_r, tab_r = q[rows], w3.squeeze(2)[rows], table[rows]
    lens_2d = lens_all.reshape(-1)[rows].view(b, NEXT_N)
    max_len = cols * BLOCK
    why = _envelope(q_r, kv, w_r, lens_2d, tab_r, max_len, n, b)
    if why is not None:
        return False, f"the case is outside the served envelope ({why})", 0
    want = _legacy(q_r, kv, w_r, lens_2d, tab_r, max_len, n, b, sms)
    args = (q_r, kv, w_r, lens_2d, tab_r, max_len, n, b)
    got = _dgo(*args, False)
    again = _dgo(*args, False)
    ok1, ncol = _same(got, want, lens_2d)
    ok2 = _same(again, got, lens_2d)[0]
    del got, again
    okg = _graph(args, want, lens_2d, dev) if i == GRAPH_CASE else True
    if ok1 and ok2 and okg and i in TIME_CASES:  # r60: per-lever boot timing (the stderr keeps only boot lines)
        t_leg = _time_us(lambda: _legacy(q_r, kv, w_r, lens_2d, tab_r, max_len, n, b, sms))
        t_dgo = _time_us(lambda: _dgo(*args, False))
        _ST.setdefault("timing", []).append((i, b, t_leg, t_dgo))
    del want, q, kv, w3, table
    if not (ok1 and ok2 and okg):
        return False, (f"DGO6 {'==' if ok1 else '!='} DeepGEMM, deterministic {ok2}"
                       + ("" if i != GRAPH_CASE else f", graph == eager {okg}")), ncol
    return True, "", ncol


def _prove(sms: int, dev, rank: int) -> dict:
    """This rank's boot-proof record (never raises: a fault is a failed record, voted)."""
    rec = {"ok": False, "why": None, "cases": 0, "cols": 0, "jit_s": None, "s": None}
    t0 = time.monotonic()
    try:
        _need(not _capturing(), "the proof would run inside a CUDA-graph capture")
        key = idx_dgo6.dgo_key(BLOCK, HEADS, DIM, NEXT_N, sms)
        t1 = time.monotonic()
        idx_dgo6.dgo_compile(key)
        rec["jit_s"] = round(time.monotonic() - t1, 1)
        _need(key in idx_dgo6._DGO_COMPILED, "the compile left no kernel under its key")
        _need(0 <= GRAPH_CASE < len(PROOF_CASES) and PROOF_CASES[GRAPH_CASE][1] <= sms,
              "the graph case would be skipped (its B is over the SM count)")
        _ST["key"], _ST["sms"] = key, sms
        gen = torch.Generator(device=dev)
        gen.manual_seed(SEED)
        for i, case in enumerate(PROOF_CASES):
            if case[1] > sms:
                continue
            ok, why, ncol = _case(i, case, rank, gen, dev, sms)
            rec["cases"] += 1
            rec["cols"] += ncol
            if not ok:
                rec["why"] = f"case {i} {case}: {why}"
                break
        if rec["why"] is None and rec["cols"] <= 0:
            rec["why"] = "no valid column compared"
        rec["ok"] = rec["why"] is None
    except Exception as e:  # noqa: BLE001 - a JIT / launch / proof fault is a failed record (voted)
        rec["why"] = f"{type(e).__name__}: {str(e)[:300]}"
    rec["s"] = round(time.monotonic() - t0, 1)
    return rec


def _inert_why(node):
    """The reason this engine keeps the king's calls (rank-uniform facts: not voted), or None."""
    mode = get_disagg().disaggregation_mode
    if mode != ROLE:
        return f"disaggregation mode {mode!r}: DGO6 serves the {ROLE} engine's verify calls only"
    tp = parallel_state.get_tp_group()
    attn = idx_nextn_route._SPLIT.get("group")
    tw, aw = getattr(tp, "world_size", None), getattr(attn, "world_size", None)
    if tw != TP_WORLD or aw != TP_WORLD:
        return f"TP world {tw}, attention-TP world {aw}: DGO6 serves the TP{TP_WORLD} decode engine only"
    indexer = getattr(node, "indexer", None)
    weight = getattr(getattr(indexer, "wq_b", None), "weight", None)
    if indexer is None or not torch.is_tensor(weight) or weight.device.type not in _DEV_TYPES:
        return "no indexer on a CUDA device"
    sms = getattr(indexer, "sm_count", None)
    if (getattr(indexer, "n_heads", None), getattr(indexer, "head_dim", None)) != (HEADS, DIM) or type(sms) is not int \
            or sms < 1:
        return f"the indexer is not {HEADS} x {DIM} heads with an SM count ({sms!r})"
    if (idx_nextn_route.NEXT_N, idx_nextn_route.BLOCK) != (NEXT_N, BLOCK):
        return f"idx_nextn_route serves next_n {idx_nextn_route.NEXT_N} on {idx_nextn_route.BLOCK}-token pages"
    if idx_nextn_route.deepgemm_paged_mqa_logits_native is not _STOCK_NATIVE:
        return "idx_nextn_route.deepgemm_paged_mqa_logits_native is not SGLang's (another route holds it)"
    if not (callable(getattr(deep_gemm, "fp8_paged_mqa_logits", None))
            and callable(getattr(deep_gemm, "get_paged_mqa_logits_metadata", None))):
        return "deep_gemm lacks fp8_paged_mqa_logits / get_paged_mqa_logits_metadata"
    if _capturing():
        return "the first prepare runs inside a CUDA-graph capture"
    return None


def prepare(node) -> None:
    """Once a process (the first claimed DSA module's prepare): the gates, then DGO6's compile + the boot proof on
    every rank, voted over the TP group (see the docstring). Never raises."""
    if not IDX_ALT or _ST["decided"]:
        return
    _ST["decided"] = True
    try:
        why = _inert_why(node)
    except Exception as e:  # noqa: BLE001 - an unreadable fact (the same code on every rank): inert
        why = f"{type(e).__name__}: {str(e)[:200]}"
    if why is not None:
        _ST["on"], _ST["off"] = False, why
        _say("inert", f"idxalt: inert on this engine ({why}): DeepGEMM's logits (the king's calls exactly)")
        return
    indexer = node.indexer
    tp = parallel_state.get_tp_group()
    rank, sms = tp.rank_in_group, indexer.sm_count
    _ST["dev"] = indexer.wq_b.weight.device
    _ST["max_b"] = max(c[1] for c in PROOF_CASES if c[1] <= sms) if any(c[1] <= sms for c in PROOF_CASES) else 0
    rec = _prove(sms, _ST["dev"], rank)
    votes = tp.all_gather_object(dict(rec))
    _ST["votes"] = votes
    bad = [(r, v) for r, v in enumerate(votes) if not (isinstance(v, dict) and v.get("ok") is True)]
    if bad:
        r, v = bad[0]
        _ST["on"] = False
        _ST["off"] = f"boot proof: rank {r}: {v.get('why') if isinstance(v, dict) else v}"
        _log.warning("idxalt: OFF on every rank (%s): DeepGEMM's logits from here (rank %d of %d: %s)", _ST["off"],
                     rank, tp.world_size, "pass" if rec["ok"] else "fail")
        return
    _ST["on"] = True
    jit = max(float(v.get("jit_s") or 0.0) for v in votes)
    _log.warning("idxalt: boot proof rank %d/%d (%.1f s, compile %.1f s, sms %d): DGO6 == DeepGEMM legacy logits as "
                 "int32 words on every valid column of %d cases (%d columns here; rand/real/probe x kpool/edge/zero, "
                 "B 1..%d, row split), deterministic, graph == eager -> ON on every rank "
                 "(slowest compile %.1f s; columns by rank %s); boot timing (case, B a rank, DeepGEMM us, DGO6 us) %s",
                 rank, tp.world_size, rec["s"], rec["jit_s"] or 0.0, sms, rec["cases"], rec["cols"], _ST["max_b"], jit,
                 [v.get("cols") for v in votes], _ST.get("timing"))


@contextlib.contextmanager
def routed(node, forward_batch):
    """idx_nextn_route.deepgemm_paged_mqa_logits_native = _native for one target-verify call of a claimed DSA module
    (in capture too) when the boot proof passed on every rank; the replaced value again after the call."""
    if (not IDX_ALT or _ST["on"] is not True or _ST["bad"] is not None
            or not forward_batch.forward_mode.is_target_verify()):
        yield
        return
    held = idx_nextn_route.deepgemm_paged_mqa_logits_native
    if held is not _STOCK_NATIVE:
        _say("held", "idxalt: idx_nextn_route.deepgemm_paged_mqa_logits_native is not SGLang's at a call (another "
                     "route holds the name): DeepGEMM's logits for it")
        yield
        return
    layer = getattr(node, "layer_id", None)
    _CALL.append((forward_batch, layer if type(layer) is int else None))
    idx_nextn_route.deepgemm_paged_mqa_logits_native = _native
    try:
        yield
    finally:
        idx_nextn_route.deepgemm_paged_mqa_logits_native = held
        _CALL.pop()
