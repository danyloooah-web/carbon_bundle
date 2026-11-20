from __future__ import annotations
import contextlib
from glm53f_core import idx_nextn_route
NODE = "dsa"
def prepare(node) -> None:
    idx_nextn_route.prepare(node)
def _get_topk_paged(indexer, forward_batch, layer_id, q_fp8, weights, metadata):
    plan = metadata.attn_metadata.kpool_write_plan if forward_batch.forward_mode.is_draft_extend_v2() else None
    if plan is None or plan.num_draft_tokens != idx_nextn_route.NEXT_N:
        if plan is not None:
            idx_nextn_route._say("draft rows", f"stock's call serves a draft-extend step of {plan.num_draft_tokens} rows "
                                               "a request")
        return idx_nextn_route._STOCK(indexer, forward_batch, layer_id, q_fp8, weights, metadata)
    return idx_nextn_route.served(indexer, forward_batch, layer_id, q_fp8, weights, metadata, "draft-extend")


@contextlib.contextmanager
def routed(node, forward_batch):
    if not forward_batch.forward_mode.is_draft_extend_v2():
        yield
        return
    with idx_nextn_route.installed(idx_nextn_route._INDEXERS[id(node)], _get_topk_paged):
        yield
