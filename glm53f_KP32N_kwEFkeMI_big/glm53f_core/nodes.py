from __future__ import annotations
import contextlib
from sglang.srt.layers.linear import LinearBase
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA, DeepseekV2MoE
from sglang.srt.models.glm5_next import Glm5NextDecoderLayer, Glm5NextLinearAttention, Glm5NextModel
from glm53f_core import components
_ROUTES = {kind: tuple(c for c in components.ACTIVE if c.NODE == kind)
           for kind in ("dsa", "kda", "proj", "dmoe", "layer", "root", "logits", "embed")}
if _ROUTES["kda"]:
    _ROUTES["kda"] += _ROUTES["proj"]
    _ROUTES["proj"] = ()
def _prepared(module, kind: str, stock: type):
    if not isinstance(module, stock):
        raise TypeError(f"this bundle's {kind} node serves {stock.__name__}, got {type(module).__name__}")
    for component in _ROUTES[kind]:
        component.prepare(module)
    return module
def _forward(module, kind: str, forward_batch, args: tuple, kwargs: dict):
    with contextlib.ExitStack() as routes:
        for component in _ROUTES[kind]:
            routes.enter_context(component.routed(module, forward_batch))
        return type(module).forward(module, *args, **kwargs)
def prepare_dsa(module):
    return _prepared(module, "dsa", DeepseekV2AttentionMLA)
def forward_dsa(module, *args, **kwargs):
    return _forward(module, "dsa", kwargs["forward_batch"], args, kwargs)
def prepare_kda(module):
    return _prepared(module, "kda", Glm5NextLinearAttention)
def forward_kda(module, *args, **kwargs):
    return _forward(module, "kda", kwargs["forward_batch"], args, kwargs)


def prepare_proj(module):
    return _prepared(module, "proj", LinearBase)


def forward_proj(module, *args, **kwargs):
    
    return _forward(module, "proj", None, args, kwargs)


def prepare_dmoe(module):
    return _prepared(module, "dmoe", DeepseekV2MoE)

def forward_dmoe(module, *args, **kwargs):
    
    return _forward(module, "dmoe", None, args, kwargs)

def prepare_logits(module):

    from sglang.srt.layers.logits_processor import LogitsProcessor

    return _prepared(module, "logits", LogitsProcessor)


def forward_logits(module, *args, **kwargs):
    
    forward_batch = kwargs["logits_metadata"] if "logits_metadata" in kwargs else args[3]
    return _forward(module, "logits", forward_batch, args, kwargs)

def prepare_embed(module):
    # r60 RIDERS ERP (ours): model.embed_tokens served by the "embed" components (embed_repl_route), in the
    # prepare_logits / forward_logits pattern.
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    return _prepared(module, "embed", VocabParallelEmbedding)


def forward_embed(module, *args, **kwargs):
    return _forward(module, "embed", None, args, kwargs)





_BLOCK_ROUTES = _ROUTES["kda"] + _ROUTES["proj"]
_LAYERS: dict = {}
_STACKS: dict = {}

def prepare_layer(module):
    
    if type(module) is not Glm5NextDecoderLayer:
        raise TypeError(f"this bundle's layer node serves Glm5NextDecoderLayer, got {type(module).__name__}")
    if _ROUTES["root"]:
        raise RuntimeError("a whole-stack component cannot be served under a layer's node (make_bundles.py)")
    attention = module.self_attn
    if isinstance(attention, DeepseekV2AttentionMLA):
        components_of = _ROUTES["dsa"]
    elif isinstance(attention, Glm5NextLinearAttention):
        components_of = _BLOCK_ROUTES
    else:
        raise TypeError(f"the layer's attention module is {type(attention).__name__}, which this bundle does not know")
    for component in components_of:
        component.prepare(attention)
    for component in _ROUTES["layer"]:
        component.prepare(module)
    _LAYERS[id(module)] = (attention, components_of)
    return module


def forward_layer(module, *args, **kwargs):
    
    forward_batch = kwargs["forward_batch"] if "forward_batch" in kwargs else args[2]
    attention, components_of = _LAYERS[id(module)]
    with contextlib.ExitStack() as routes:
        for component in _ROUTES["layer"]:
            routes.enter_context(component.routed(module, forward_batch))
        for component in components_of:
            routes.enter_context(component.routed(attention, forward_batch))
        return Glm5NextDecoderLayer.forward(module, *args, **kwargs)

def prepare_model(module):
    
    if type(module) is not Glm5NextModel:
        raise TypeError(f"this bundle's root node serves Glm5NextModel, got {type(module).__name__}")
    if _ROUTES["dmoe"] or _ROUTES["layer"]:
        raise RuntimeError("a draft-layer or layer component cannot be served under the stack's node "
                           "(make_bundles.py)")
    attentions = [layer.self_attn for layer in module.layers]
    other = sorted({type(a).__name__ for a in attentions
                    if not isinstance(a, (DeepseekV2AttentionMLA, Glm5NextLinearAttention))})
    if other:
        raise TypeError(f"the stack holds attention modules this bundle does not know: {other}")
    first = {}
    for kind, stock, routes in (("dsa", DeepseekV2AttentionMLA, _ROUTES["dsa"]),
                                ("kda", Glm5NextLinearAttention, _BLOCK_ROUTES)):
        nodes = [attention for attention in attentions if isinstance(attention, stock)]
        if not nodes:
            raise RuntimeError(f"the stack holds no {stock.__name__}")
        for node in nodes:
            for component in routes:
                component.prepare(node)
        first[kind] = nodes[0]
    for component in _ROUTES["root"]:
        component.prepare(module)
    _STACKS[id(module)] = (first["dsa"], first["kda"])
    return module


def forward_model(module, *args, **kwargs):
    
    forward_batch = kwargs["forward_batch"] if "forward_batch" in kwargs else args[2]
    dsa, kda = _STACKS[id(module)]
    with contextlib.ExitStack() as routes:
        for node, components_of in ((module, _ROUTES["root"]), (dsa, _ROUTES["dsa"]), (kda, _BLOCK_ROUTES)):
            for component in components_of:
                routes.enter_context(component.routed(node, forward_batch))
        return Glm5NextModel.forward(module, *args, **kwargs)
