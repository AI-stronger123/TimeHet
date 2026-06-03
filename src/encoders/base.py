"""Graph Encoder 抽象基类。"""
import torch
import torch.nn as nn


class BaseGraphEncoder(nn.Module):
    def __init__(self, graph_dim: int, llm_dim: int, proj_dim: int, vocab_size: int, **kwargs):
        super().__init__()
        self.graph_dim = graph_dim
        self.llm_dim = llm_dim
        self.proj_dim = proj_dim
        self.vocab_size = vocab_size

    def forward(self, node_embs: torch.Tensor, word_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embs: [B, graph_dim]
            word_embeddings: [V, llm_dim]
        Returns:
            graph_tokens: [B, seq_len, llm_dim]
        """
        raise NotImplementedError
