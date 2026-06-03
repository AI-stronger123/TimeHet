"""Graph Projector：将图节点表示投影到 LLM 词嵌入空间。"""
import torch
import torch.nn as nn
from .base import BaseGraphEncoder


class TokenMixer(nn.Module):

    def __init__(self, graph_dim, vocab_size, llm_dim):
        super().__init__()
        self.up_projection = nn.Linear(graph_dim, llm_dim, bias=False)
        self.mapping = nn.Linear(vocab_size + 1, 1)

    def forward(self, h_graph, word_embeddings):
        """
        h_graph: [B, graph_dim]
        word_embeddings: [V, llm_dim]
        return: [B, 1, llm_dim]
        """
        h_graph = self.up_projection(h_graph)  # [B, llm_dim]
        dtype = word_embeddings.dtype
        w_graph = self.mapping.weight[0, 0].to(dtype)
        w_vocab = self.mapping.weight[0, 1:].to(dtype)
        bias = self.mapping.bias.to(dtype) if self.mapping.bias is not None else 0.0

        vocab_term = torch.matmul(w_vocab, word_embeddings)
        output = w_graph * h_graph + vocab_term + bias  # [B, llm_dim]
        return output.unsqueeze(1)  # [B, 1, llm_dim]


class GraphProjector(BaseGraphEncoder):
    def __init__(self, graph_dim: int, llm_dim: int, proj_dim: int, vocab_size: int, **kwargs):
        super().__init__(graph_dim, llm_dim, proj_dim, vocab_size, **kwargs)
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )
        self.mixer = TokenMixer(proj_dim, vocab_size, llm_dim)

    def forward(self, node_embs: torch.Tensor, word_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embs: [B, graph_dim]
            word_embeddings: [V, llm_dim]
        Returns:
            graph_tokens: [B, 1, llm_dim]
        """
        x = self.graph_proj(node_embs)  # [B, proj_dim]
        tokens = self.mixer(x, word_embeddings)  # [B, 1, llm_dim]
        return tokens
