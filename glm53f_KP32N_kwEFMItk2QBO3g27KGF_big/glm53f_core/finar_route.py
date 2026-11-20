"""r60 FINAR (ours, 2026-11-20): the TP4 decode engine's MoE exit at the live verify band with the king's A4 finalize
(moe_finalize_route: FlashInfer's unfinalized FP4 MoE pieces + the shared rows, moe_finalize_native.fuse) folded into
the PUSH READ of the MoE output's all-reduce: one launch of OUR finar kernel (finar_ar.cuh, the king's
dsa_pagetable_native unit's submodule "finar") computes A4's exact chain (fp32 slot-order fma.rn.ftz from +0, -1 slots
skipped, RN; add.rn of the shared row, RN) where the stock two-shot push all-reduce (CustomAllReduceV2 2shot_push, the
algorithm SGLang picks at 0.625 to 8 MiB on TP4 SM100) reads its partial, with stock's partition, rank order, single RN
and zero-word markers. The [T, 4096] finalized partial is never written nor re-read; A4's launch disappears.

Where (text of this image + the v32 king): A4 (moe_finalize_route) owes the finalize from its _moe until its _add, which
calls the module global moe_finalize_native.fuse; SGLang's forward_normal_dual_stream returns that output
(reduce_results False), the decoder layer's FFN exit completes its sum at once (the mHC FFN update is applied at exit,
so the sum is never deferred) through sum_post_experts_output -> the MoE TP group's all_reduce -> CustomAllReduceV2
2shot_push, then the mHC post (the king's mhc_fi_route). Nothing reads the output between A4's add and that sum.
How (decode engine only; inside CUDA-graph capture only): routed() swaps A4's module global moe_finalize_native for a
proxy whose fuse records the operands instead of launching (rows MIN_ROWS..MAX_ROWS, A4's layouts) and wraps the sum
groups' all_reduce (the king's mhc_ar_route pattern) so the sum of exactly that tensor is our launch into a fresh tensor;
anything else passes through untouched; a sum of another tensor first flushes the owed finalize (A4's own launch); a
layer call that returns with a finalize still owed raises (the sum went elsewhere: never served silently). Everything
is restored in a finally.
Boot (the first A4-served layer's prepare, eager, every rank in the same order): local facts voted on the CPU group
(decode engine, one 4-rank sum group set, every group's communicator picking 2shot_push at every served size in both
its graph and eager tables, the native's version / instances); a bundle-owned symmetric plane (torch symmetric memory,
the king's mhc_ar_route form); a SEEDED BOOT PROOF on every rank: FINAR == A4's moe_finalize_native.fuse -> the TP
group's stock all_reduce, byte for byte, at PROOF_ROWS (half-row owner chunks, -1 / all-skipped rows, +-0 corners,
subnormal / NaN / inf / overflow / cancellation), plus FINAR mode 0 == stock (transport) and back-to-back calls over both
parities with a skew drill; voted: any miss on any rank -> OFF everywhere (the king's chain exactly). Then a boot timing
line (graph replays: A4 + stock all-reduce vs FINAR), a diagnostic that never gates.
"""
from __future__ import annotations

import contextlib
import time
import types

import torch
from sglang.srt.models import deepseek_v2
from sglang.srt.runtime_context import get_disagg

from glm53f_core import moe_finalize_route as a4

NODE = "layer"
FINAR = True                         # False: inert (the king's chain everywhere)
FN_VERSION = 6001                    # finar_ar.cuh kVersion
HIDDEN, TOP_K, RANKS = 4096, 8, 4
MIN_ROWS, MAX_ROWS = 480, 768        # served rows (A4 serves 193..16384; stock 2shot_push covers 0.625..8 MiB on TP4)
FN_BLOCK, FN_TRIG = 1024, 0          # the kernel instance (threads a CTA, PDL trigger before / after the AG receive)
TIMEOUT_NS = 60 * 10 ** 9            # every poll loop traps past this (a loud device fault, never a hang)
PROOF_ROWS = (486, 528, 582, 672, 726, 768)   # 486 / 582 / 726: odd request counts (half-row owner chunks)
PROOF_B2B = 4                        # back-to-back calls over both parities, no host sync in between
SKEW_CYCLES = 1000000                # skew drill: rank r spins r x this many GPU cycles before the back-to-back calls
BOOT_TIMING = True                   # the boot timing line (diagnostic; never gates)
TIME_ROWS, TIME_EXITS, TIME_REPS = (576, 672), 8, 5
SEED = 20261120
SLOT_BYTES = 2048 * MAX_ROWS         # one (parity, kind, src) slot: a chunk of 128 T 16-byte vectors at T = MAX_ROWS
PLANE_BYTES = 2 * 2 * RANKS * SLOT_BYTES
_MISSING = object()
_LAYERS: set = set()
_state = {"decided": False, "on": False, "why": "", "groups": (), "native": None, "a4": None, "keep": None,
          "peers": None, "counter": None, "rank": 0, "none": None, "device": None, "said": set(), "tally": {}}
_CALL = {"live": False, "pending": None}


def _say(tag: str, detail: str) -> None:
    if tag not in _state["said"]:
        _state["said"].add(tag)
        print(f"[glm-finar] finar: {detail}", flush=True)


def _tally(route: str, describe=None) -> None:
    tally = _state["tally"]
    calls = tally[route] = tally.get(route, 0) + 1
    if calls & (calls - 1) == 0:
        print(f"[glm-finar] finar: {route} x{calls}" + (f": {describe()}" if describe is not None else ""), flush=True)


def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"finar: {why}")


def _load_native():
    import dsa_pagetable_native

    return dsa_pagetable_native.finar


def _groups():
    from sglang.srt.distributed.parallel_state import get_attn_tp_group, get_moe_tp_group, get_tp_group

    return get_tp_group(), get_attn_tp_group(), get_moe_tp_group()


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def _gather(group, obj) -> list:
    got = [None] * group.world_size
    torch.distributed.all_gather_object(got, obj, group=group.cpu_group)
    return got


def _two_shot(group, lo: int, hi: int):
    """None when the group's communicator picks plain 2shot_push (no multicast) for every [rows, 4096] BF16 sum of lo..hi
    rows in both its graph and its eager table; else why not."""
    ca = getattr(group, "ca_comm", None)
    if ca is None or getattr(ca, "disabled", True):
        return "no enabled custom all-reduce on the group"
    if type(ca).__name__ != "CustomAllReduceV2" or getattr(ca, "override_algo", None) is not None:
        return f"its custom all-reduce is {type(ca).__name__} (override {getattr(ca, 'override_algo', None)})"
    pick = getattr(ca, "_pick_config", None)
    if not callable(pick):
        return "CustomAllReduceV2 has no _pick_config"
    for rows in range(lo, hi + 1):
        for graph in (True, False):
            cfg = pick(rows * HIDDEN * 2, graph)
            name = getattr(getattr(cfg, "algo", None), "algo_name", None)
            if name != "2shot_push" or getattr(cfg, "use_multicast", True):
                return f"{rows} rows pick {name} ({'graph' if graph else 'eager'} table)"
    return None


def _local_facts(layer):
    """(ok, why, facts) on this rank; no collective."""
    if not FINAR:
        return False, "FINAR is False", {}
    if torch.cuda.is_current_stream_capturing():
        return False, "prepare ran under graph capture", {}
    unique = []
    for group in _groups():
        if all(group is not seen for seen in unique):
            unique.append(group)
    ranks = [list(g.ranks) for g in unique]
    if any(g.world_size != RANKS for g in unique) or any(r != ranks[0] for r in ranks):
        return False, f"the sum groups are not one {RANKS}-rank set (ranks {ranks})", {}
    for group in unique:
        why = _two_shot(group, MIN_ROWS, MAX_ROWS)
        if why is not None:
            return False, why, {}
    try:
        native = _load_native()
    except Exception as e:  # noqa: BLE001 - a missing or stale native unit: voted OFF
        return False, f"no finar native ({type(e).__name__}: {e})", {}
    if not (callable(getattr(native, "version", None)) and callable(getattr(native, "run", None))
            and callable(getattr(native, "info", None))):
        return False, "the finar native lacks version / info / run", {}
    if int(native.version()) != FN_VERSION:
        return False, f"the finar native is version {native.version()}, not {FN_VERSION}", {}
    facts = {}
    for mode in (0, 4):
        info = [int(v) for v in native.info(mode, FN_BLOCK, FN_TRIG)]
        facts[mode] = info
        if info[6] != 0 or info[1] < 1 or info[3] != 0:
            return False, f"instance mode {mode} block {FN_BLOCK} trig {FN_TRIG}: facts {info}", facts
    a4n = getattr(a4, "moe_finalize_native", None)
    if not callable(getattr(a4n, "fuse", None)):
        return False, "A4's moe_finalize_native has no fuse", facts
    return True, "", dict(facts, groups=len(unique), rank=unique[0].rank_in_group)


def _plane(group, device):
    import torch.distributed._symmetric_memory as symmetric_memory

    area = symmetric_memory.empty(PLANE_BYTES, dtype=torch.uint8, device=device)
    area.zero_()
    handle = symmetric_memory.rendezvous(area, group.device_group)
    _require(handle.rank == group.rank_in_group and handle.world_size == RANKS,
             f"a symmetric handle of rank {handle.rank} of {handle.world_size} on rank {group.rank_in_group}")
    peers = [int(p) for p in handle.buffer_ptrs]
    _require(len(peers) == RANKS and all(p and p % 16 == 0 for p in peers), f"plane bases {peers}")
    return area, handle, peers


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def _case(rank: int, rows: int, seed: int, device):
    """One rank's synthetic A4 operands in EP geometry (each of a token's 8 picks local with probability 1/4; slots in a
    shuffled order inside a GEMM2 buffer whose unused rows hold NaN), with corner rows: all slots -1, all 8 local,
    +-0 / subnormal / NaN / inf / huge weights and values, cancelling pairs, zero partial words."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) * 8 + rank)
    t = rows
    local = torch.rand((t, TOP_K), generator=gen) < 0.25
    local[0::17] = False                                # rows with every slot -1 (no local expert)
    local[5::23] = True                                 # rows with all 8 slots local (TP-like)
    owner = [min(4 * r // t, RANKS - 1) for r in range(t)]   # the all-reduce owner of row r's first vector
    coord = {}                                          # rows whose corner spans the ranks (same rows on every rank)
    for r in range(19, t, 43):
        coord[r] = "ftz" if r % 4 == rank else "zero"   # one rank's subnormal products, every other rank's +0
    for r in range(21, t, 43):
        coord[r] = "order" if (r + 1) % 4 == rank else "zero"   # slot order visible: 2^24 + 1 + 1 - 2^24
    for r in range(23, t, 43):
        coord[r] = "negzero"                            # no slot anywhere, -0 shared rows on every rank
    for r in range(25, t, 43):
        coord[r] = "zero" if owner[r] == rank else "lozero"   # the owner's +0 word against peers' (-0, x) words
    for r, kind in coord.items():
        local[r] = False
        if kind in ("ftz", "lozero"):
            local[r, 0] = True
        elif kind == "order":
            local[r, 0:4] = True
    count = int(local.sum())
    spare = 64
    order = torch.randperm(count + spare, generator=gen)[:count]
    idx = torch.full((t, TOP_K), -1, dtype=torch.int32)
    idx[local] = order.to(torch.int32)
    g = torch.randn((count + spare, HIDDEN), generator=gen) * 0.5
    used = torch.zeros(count + spare, dtype=torch.bool)
    used[order] = True
    g[~used] = float("nan")                             # never read: padding / dropped slots
    w = torch.rand((t, TOP_K), generator=gen) * 0.6
    s = torch.randn((t, HIDDEN), generator=gen) * 0.25
    tiny = 2.0 ** -130                                  # a bf16 subnormal (A4 flushes it: fma.rn.ftz)
    for r in range(3, t, 29):                           # weight corners on the row's local slots
        if r in coord:
            continue
        ks = [k for k in range(TOP_K) if int(idx[r, k]) >= 0]
        for k, v in zip(ks, (0.0, -0.0, tiny, -1.5)):
            w[r, k] = v
    for r in range(7, t, 31):                           # value corners in the row's first local slot
        if r in coord:
            continue
        ks = [k for k in range(TOP_K) if int(idx[r, k]) >= 0]
        if ks:
            q = int(idx[r, ks[0]])
            g[q, 0:8] = torch.tensor([0.0, -0.0, tiny, -tiny, 2.0 ** 100, -(2.0 ** 100), float("inf"), 1.0])
            w[r, ks[0]] = 2.0 ** 100 if r % 2 else 1.0  # overflow to inf on odd rows
            if len(ks) > 1:                             # an exactly cancelling pair
                g[int(idx[r, ks[1]]), 8:64] = -g[q, 8:64]
                w[r, ks[1]] = w[r, ks[0]]
    for r in range(11, t, 37):
        if r in coord:
            continue
        s[r, 0:4] = torch.tensor([float("nan"), tiny, -0.0, 0.0])
    for r in range(13, t, 41):                          # zero words: all slots -1 and +-0 shared rows
        if r not in coord:
            idx[r] = -1
            s[r] = -0.0 if (r + rank) % 2 else 0.0
    for r, kind in coord.items():
        q = [int(v) for v in idx[r] if int(v) >= 0]
        s[r] = -0.0 if kind in ("negzero", "lozero") else 0.0
        if kind == "ftz":                               # 2^-60 x 2^-70: a subnormal product (flushed by .ftz)
            g[q[0]] = 2.0 ** -70
            w[r, 0] = 2.0 ** -60
        elif kind == "order":                           # forward 0, reversed 2: the slot order shows
            for k, v in enumerate((2.0 ** 24, 1.0, 1.0, -(2.0 ** 24))):
                g[q[k]] = v
                w[r, k] = 1.0
        elif kind == "lozero":                          # low lanes -(2^-130) -> -0 under .ftz, high lanes 2^-60
            g[q[0], 0::2] = -(2.0 ** -70)
            g[q[0], 1::2] = 1.0
            w[r, 0] = 2.0 ** -60
    to = dict(device=device)
    return types.SimpleNamespace(g=_bf16(g).to(**to).contiguous(), idx=idx.reshape(-1).to(**to).contiguous(),
                                 w=_bf16(w).to(**to).contiguous(), s=_bf16(s).to(**to).contiguous(), rows=t)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape)
            and bool(torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))))


def _run(mode: int, out: torch.Tensor, *, x=None, case=None, pdl: bool = False) -> None:
    none = _state["none"]
    if mode == 0:
        args = (x, none, none, none, none)
    else:
        args = (none, case.g, case.idx, case.w, case.s)
    _state["native"].run(mode, FN_BLOCK, FN_TRIG, out, *args, _state["peers"], _state["counter"], SLOT_BYTES,
                         TIMEOUT_NS, _state["rank"], pdl)


def _stock(group, x: torch.Tensor) -> torch.Tensor:
    return type(group).all_reduce(group, x)


def _prove(group, device) -> tuple:
    """(ok, detail) of this rank's seeded proof; every rank makes the same launches and sums."""
    rank, a4n = _state["rank"], _state["a4"]
    flags, notes = [], []
    cases = {}
    for n, rows in enumerate(PROOF_ROWS):
        case = cases[rows] = _case(rank, rows, SEED + 101 * n, device)
        part = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        a4n.fuse(case.g, case.idx, case.w, case.s, part, False)
        ref = _stock(group, part)
        got4 = torch.empty_like(part)
        _run(4, got4, case=case, pdl=bool(n % 2))
        got0 = torch.empty_like(part)
        _run(0, got0, x=part, pdl=bool(n % 2))
        flags += [_same(got4, ref), _same(got0, ref)]
        notes.append(rows)
        case.ref = ref
    torch.cuda.synchronize(device)
    torch.cuda._sleep(int(rank * SKEW_CYCLES))
    b2b = []
    for k in range(PROOF_B2B):
        rows = PROOF_ROWS[k % len(PROOF_ROWS)]
        out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        _run(4, out, case=cases[rows], pdl=True)
        b2b.append((out, cases[rows].ref))
    flags += [_same(o, r) for o, r in b2b]
    torch.cuda.synchronize(device)
    ok = all(flags)
    return ok, f"{sum(flags)}/{len(flags)} byte-equal at rows {notes} + {PROOF_B2B} back-to-back"


def _timing(group, device) -> list:
    """Graph-replay timing of A4 + stock all-reduce vs FINAR on identical operands (diagnostic; uniform on every rank)."""
    rank, a4n, lines = _state["rank"], _state["a4"], []
    side = torch.cuda.Stream(device=device)
    for rows in TIME_ROWS:
        cases = [_case(rank, rows, SEED + 7000 + 31 * rows + k, device) for k in range(TIME_EXITS)]
        parts = [torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device) for _ in cases]

        def king():
            for c, o in zip(cases, parts):
                a4n.fuse(c.g, c.idx, c.w, c.s, o, True)
                _stock(group, o)

        def fused():
            for c, o in zip(cases, parts):
                _run(4, torch.empty_like(o), case=c, pdl=True)

        king()                                          # eager warm-ups first: the same launches and sums on every
        fused()                                         # rank whatever a capture below does (a capture runs none)
        torch.cuda.synchronize(device)
        graphs, ok = {}, True
        for name, fn in (("king", king), ("fused", fused)):
            try:
                graph = torch.cuda.CUDAGraph()
                side.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.graph(graph, stream=side, capture_error_mode="thread_local"):
                    fn()
                graphs[name] = graph
            except Exception as e:  # noqa: BLE001 - a diagnostic: voted, skipped everywhere
                ok = False
                lines.append(f"rows {rows}: {name} capture failed on rank {rank} ({type(e).__name__})")
        torch.cuda.synchronize(device)
        if not all(_gather(group, ok)):
            lines.append(f"rows {rows}: timing skipped (a rank's capture failed)")
            continue
        times = {name: [] for name in graphs}
        for _ in range(TIME_REPS):
            for name, graph in graphs.items():
                torch.distributed.barrier(group=group.cpu_group)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                graph.replay()
                e1.record()
                e1.synchronize()
                times[name].append(e0.elapsed_time(e1) * 1000.0 / TIME_EXITS)
        med = {name: sorted(v)[len(v) // 2] for name, v in times.items()}
        every = _gather(group, med)
        worst = {name: max(m[name] for m in every) for name in med}
        lines.append(f"rows {rows}: king {worst['king']:.1f} us vs fused {worst['fused']:.1f} us an exit (A4 + stock "
                     f"2shot_push vs FINAR, graph replay, slowest rank; d {worst['king'] - worst['fused']:+.1f} us)")
        del graphs
        torch.cuda.synchronize(device)
    return lines


def _decide(layer) -> None:
    """The one boot decision (every rank reaches it at the same layer: A4's eligibility and the engine are uniform)."""
    _state["decided"] = True
    if get_disagg().disaggregation_mode != "decode":
        _state["why"] = "not the decode engine"
        _say("inactive", f"inactive on this engine ({_state['why']}); A4 and stock's sum serve")
        return
    t0 = time.perf_counter()
    ok, why, facts = _local_facts(layer)
    unique = []
    for group in _groups():
        if all(group is not seen for seen in unique):
            unique.append(group)
    group = unique[0]
    if group.world_size != RANKS:
        _state["why"] = f"a {group.world_size}-rank group"
        _say("inactive", f"inactive on this engine ({_state['why']})")
        return
    votes = _gather(group, (ok, why))
    if not all(v[0] for v in votes):
        _state["why"] = "; ".join(f"rank {r}: {v[1]}" for r, v in enumerate(votes) if not v[0])
        _say("off", f"OFF on every rank before any launch: {_state['why']}")
        return
    device = _device()
    area, handle, peers = _plane(group, device)
    sms = int(facts[4][0])
    _state.update(groups=tuple(unique), native=_load_native(), a4=a4.moe_finalize_native, keep=(area, handle),
                  peers=peers, counter=torch.zeros(sms, dtype=torch.int32, device=device), rank=group.rank_in_group,
                  none=torch.empty(0, dtype=torch.bfloat16, device=device), device=device)
    torch.cuda.synchronize(device)
    group.barrier()
    try:
        proved, detail = _prove(group, device)
    except Exception as e:  # noqa: BLE001 - voted below (the same launches ran on every rank before any raise)
        proved, detail = False, f"{type(e).__name__}: {e}"
    votes = _gather(group, (proved, detail))
    if not all(v[0] for v in votes):
        _state["why"] = "; ".join(f"rank {r}: {v[1]}" for r, v in enumerate(votes) if not v[0])
        _say("off", f"boot proof MISS -> OFF on every rank (the king's chain serves): {_state['why']}")
        return
    _state["on"] = True
    _say("on", f"boot proof PASS on every rank ({votes[0][1]}); ON for {MIN_ROWS}-{MAX_ROWS} rows inside decode "
               f"graph capture, block {FN_BLOCK} trig {FN_TRIG}, plane {PLANE_BYTES / 2 ** 20:.0f} MiB a rank, "
               f"{len(unique)} group object(s), mode 4 facts {facts[4]}; {(time.perf_counter() - t0) * 1e3:.0f} ms")
    if BOOT_TIMING:
        for line in _timing(group, device):
            print(f"[glm-finar] finar: boot timing {line}", flush=True)


def prepare(layer) -> None:
    mlp = getattr(layer, "mlp", None)
    if not isinstance(mlp, deepseek_v2.DeepseekV2MoE) or id(layer) not in a4._LAYERS:
        return
    if not _state["decided"]:
        _decide(layer)
    if _state["on"]:
        _LAYERS.add(id(layer))


def _declined(gemm2, rows, weights, shared, out, pdl):
    if not _CALL["live"]:
        return "live", "a finalize outside this route's layer call"
    if not torch.cuda.is_current_stream_capturing():
        return "eager", "an eager call (FINAR serves decode graph captures only)"
    t = out.shape[0] if isinstance(out, torch.Tensor) and out.dim() == 2 else -1
    if not MIN_ROWS <= t <= MAX_ROWS:
        return f"rows {t > MAX_ROWS}", f"{t} rows, outside {MIN_ROWS} to {MAX_ROWS}"
    dev = _state["device"]

    def dense(x, dtype, shape):
        return (isinstance(x, torch.Tensor) and x.device == dev and x.dtype == dtype
                and tuple(x.shape) == shape and x.is_contiguous() and x.data_ptr() % 16 == 0)

    if not (dense(out, torch.bfloat16, (t, HIDDEN)) and dense(shared, torch.bfloat16, (t, HIDDEN))
            and isinstance(gemm2, torch.Tensor) and gemm2.dim() == 2 and gemm2.shape[0] >= 1
            and dense(gemm2, torch.bfloat16, (gemm2.shape[0], HIDDEN)) and dense(rows, torch.int32, (t * TOP_K,))
            and dense(weights, torch.bfloat16, (t, TOP_K)) and isinstance(pdl, bool)):
        return "layout", "operands not in A4's layout on the plane's device"
    return None


def _fuse(gemm2, rows, weights, shared, out, pdl):
    why = _declined(gemm2, rows, weights, shared, out, pdl)
    if why is not None:
        _say(why[0], f"A4's own launch serves a finalize: {why[1]}")
        return _state["a4"].fuse(gemm2, rows, weights, shared, out, pdl)
    _require(_CALL["pending"] is None, "a second finalize owed before the first one's sum")
    _CALL["pending"] = (out, gemm2, rows, weights, shared, pdl)
    return None


def _summing(before):

    def all_reduce(input_: torch.Tensor) -> torch.Tensor:
        held = _CALL["pending"]
        if held is None:
            return before(input_)
        out = held[0]
        _CALL["pending"] = None
        if not (isinstance(input_, torch.Tensor) and input_.data_ptr() == out.data_ptr() and input_.dtype == out.dtype
                and tuple(input_.shape) == tuple(out.shape) and input_.is_contiguous()):
            _state["a4"].fuse(*held[1:5], out, held[5])
            _tally("flushed", lambda: "a sum of another tensor came first: A4's launch ran, then that sum")
            return before(input_)
        y = torch.empty_like(out)
        case = types.SimpleNamespace(g=held[1], idx=held[2], w=held[3], s=held[4])
        _run(4, y, case=case, pdl=held[5])
        _tally("served", lambda: f"{out.shape[0]} rows, captured")
        return y

    return all_reduce


@contextlib.contextmanager
def routed(layer, forward_batch):
    if not _state["on"] or id(layer) not in _LAYERS:
        yield
        return
    _require(not _CALL["live"], "already routed (a layer's call inside another)")
    _require(a4.moe_finalize_native is _state["a4"], "A4's moe_finalize_native is not the module the proof ran")
    wrapped = []
    proxy = types.SimpleNamespace(fuse=_fuse, routed=_state["a4"].routed)
    _CALL.update(live=True, pending=None)
    owed = False
    a4.moe_finalize_native = proxy
    try:
        for group in _state["groups"]:
            current = group.all_reduce
            own = not (getattr(current, "__self__", None) is group
                       and getattr(current, "__func__", None) is type(group).all_reduce)
            wrapped.append((group, current if own else _MISSING))
            group.all_reduce = _summing(current)
        yield
        owed = _CALL["pending"] is not None
    finally:
        for group, kept in reversed(wrapped):
            if kept is _MISSING:
                del group.all_reduce
            else:
                group.all_reduce = kept
        a4.moe_finalize_native = _state["a4"]
        _CALL.update(live=False, pending=None)
    _require(not owed, "the layer call returned with A4's finalize still owed (its sum never came through the sum "
                       "groups this route wraps)")
