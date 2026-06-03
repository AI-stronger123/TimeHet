from .base import BaseGraphEncoder
from .graph_projector import GraphProjector

ENCODER_REGISTRY = {
    "graph_projector": GraphProjector,
}


def build_encoder(config: dict, llm_dim: int, vocab_size: int) -> BaseGraphEncoder:
    name = config["type"]
    return ENCODER_REGISTRY[name](
        graph_dim=config["graph_dim"],
        llm_dim=llm_dim,
        proj_dim=config.get("proj_dim", 768),
        vocab_size=vocab_size,
        **config.get("kwargs", {})
    )
