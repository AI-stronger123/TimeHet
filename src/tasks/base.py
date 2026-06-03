"""Task 抽象基类 + Prompt 缓存。"""
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn


class BaseTask(ABC):
    def __init__(self, config: dict, llm_dim: int, num_classes: Optional[int] = None):
        self.config = config
        self.llm_dim = llm_dim
        self.num_classes = num_classes
        self._cached_prompt_embeds = None
        self._cached_prompt_device = None

    @abstractmethod
    def create_head(self) -> nn.Module:
        """创建任务头。"""
        ...

    @abstractmethod
    def build_prompt(self, **kwargs) -> str:
        """构建 prompt 模板。"""
        ...

    def _build_prompt_embeds(self, llm, tokenizer, prompt_text: str, device: str):
        """基类通用 prompt 缓存方法。"""
        parts = prompt_text.split("<graph>")
        embed_tokens = llm.get_input_embeddings()
        text_embeds = []
        for part in parts:
            if len(part) > 0:
                tokens = tokenizer(
                    part, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(device)
                text_embeds.append(embed_tokens(tokens).squeeze(0).detach())
            else:
                text_embeds.append(None)
        return text_embeds, parts

    def prepare_inputs(self, llm, tokenizer, prompt_text: str, graph_tokens: torch.Tensor, device: str):
        """
        将 graph_tokens 插入到 prompt 的 <graph> 位置。
        graph_tokens: [B, seq_len, llm_dim]
        """
        B = graph_tokens.size(0)
        if self._cached_prompt_embeds is None or self._cached_prompt_device != str(device):
            self._cached_prompt_embeds = self._build_prompt_embeds(llm, tokenizer, prompt_text, device)
            self._cached_prompt_device = str(device)
        text_embeds, parts = self._cached_prompt_embeds

        batch_inputs = []
        seq_len = graph_tokens.size(1)
        for b in range(B):
            seq = []
            text_idx = 0
            graph_idx = 0
            for i in range(len(parts)):
                if text_embeds[i] is not None:
                    seq.append(text_embeds[i])
                    text_idx += 1
                if i < len(parts) - 1:
                    # Insert graph tokens for this <graph> slot
                    graph_token = graph_tokens[b, graph_idx]
                    if graph_token.dim() == 1:
                        graph_token = graph_token.unsqueeze(0)
                    seq.append(graph_token)
                    graph_idx += 1
            batch_inputs.append(torch.cat(seq, dim=0))

        inputs_embeds = torch.stack(batch_inputs, dim=0)
        attention_mask = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=device)
        return inputs_embeds, attention_mask

    @abstractmethod
    def build_train_loader(self, dataset, config: dict):
        """构建训练 DataLoader。"""
        ...

    @abstractmethod
    def build_val_loader(self, dataset, config: dict):
        """构建验证 DataLoader。"""
        ...

    @abstractmethod
    def train_step(self, model, batch) -> Tuple[torch.Tensor, dict]:
        """返回 loss, metrics dict。"""
        ...

    @abstractmethod
    def eval_step(self, model, batch) -> dict:
        """返回 metrics dict。"""
        ...

    @abstractmethod
    def get_best_metric(self, metrics: dict) -> float:
        """从 metrics 中提取用于判断最优模型的指标值。"""
        ...
