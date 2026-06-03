"""GraphLLMModel：组装 TEMGH + Projector + LLM + Task Head。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLLMModel(nn.Module):
    def __init__(self, temgh, encoder, llm, tokenizer, task, device,
                 target_ntype: str, freeze_temgh: bool = True,
                 precomputed_feats: torch.Tensor = None):
        super().__init__()
        self.device = device
        self.temgh = temgh
        self.encoder = encoder
        self.llm = llm
        self.tokenizer = tokenizer
        self.task = task
        self.target_ntype = target_ntype
        self.freeze_temgh = freeze_temgh

        # Task head
        self.task_head = task.create_head().to(device)
        self.task_type = task.config.get("type", "link_prediction")

        # Precomputed feats fusion (for MAG CLS)
        self.precomputed_feats = None
        if precomputed_feats is not None:
            self.register_buffer("_pre_feats", precomputed_feats.to(device))
            feat_dim = precomputed_feats.size(1)
            graph_dim = encoder.graph_dim
            self.feat_proj = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, graph_dim)
            ).to(device)
            self.merge_proj = nn.Sequential(
                nn.Linear(graph_dim * 2, graph_dim),
                nn.LayerNorm(graph_dim),
                nn.GELU()
            ).to(device)
        else:
            self._pre_feats = None
            self.feat_proj = None
            self.merge_proj = None

        # TEMGH cache
        self._temgh_cache = None
        self._temgh_cache_id = None

        # Answer embeddings for alignment loss (link prediction)
        self._yes_embed = None
        self._no_embed = None

    def _get_temgh_output(self, graph, snapshot_times=None):
        """获取 TEMGH 输出，带缓存。"""
        graph_id = id(graph) if graph is not None else None
        if graph_id is not None and getattr(self, "_temgh_cache_id", None) == graph_id:
            return self._temgh_cache

        if self.freeze_temgh:
            with torch.no_grad():
                h = self.temgh(graph, self.target_ntype, snapshot_times=snapshot_times)
        else:
            h = self.temgh(graph, self.target_ntype, snapshot_times=snapshot_times)
        if not h.requires_grad:
            self._temgh_cache = h
            self._temgh_cache_id = graph_id
        return h

    def _fuse_precomputed(self, h_temgh, indices):
        if self._pre_feats is None:
            return h_temgh
        h_pre = self.feat_proj(self._pre_feats[indices])
        return self.merge_proj(torch.cat([h_temgh, h_pre], dim=-1))

    def _get_answer_embeddings(self):
        """缓存 'yes' 和 'no' 的 LLM token embedding。"""
        if self._yes_embed is None:
            embed = self.llm.get_input_embeddings()
            yes_tokens = self.tokenizer(
                "yes", return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)
            no_tokens = self.tokenizer(
                "no", return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)
            self._yes_embed = embed(yes_tokens).mean(dim=1).detach()  # [1, llm_dim]
            self._no_embed = embed(no_tokens).mean(dim=1).detach()    # [1, llm_dim]
        return self._yes_embed, self._no_embed

    def compute_alignment_loss(self, graph, indices, labels, snapshot_times=None):
        """
        计算对齐损失 L_align：让 graph token 靠近 answer token 的 embedding。
        正样本 → 'yes'，负样本 → 'no'。
        Args:
            indices: [E, 2] edge pairs
            labels: [E] 0/1 labels
        Returns:
            loss_align: scalar
        """
        h = self._get_temgh_output(graph, snapshot_times)

        E = indices.size(0)
        batch_nodes = indices.flatten()  # [E*2]
        node_embs = h[batch_nodes]
        node_embs = self._fuse_precomputed(node_embs, batch_nodes)

        word_embeddings = self.llm.get_input_embeddings().weight
        tokens = self.encoder(node_embs, word_embeddings)  # [E*2, 1, llm_dim]
        tokens = tokens.view(E, 2, -1)  # [E, 2, llm_dim]
        edge_tokens = tokens.mean(dim=1)  # [E, llm_dim]

        yes_embed, no_embed = self._get_answer_embeddings()
        # 统一 dtype，避免 Half / Float 混合导致 backward 失败
        target_dtype = edge_tokens.dtype
        yes_embed = yes_embed.to(target_dtype)
        no_embed = no_embed.to(target_dtype)

        pos_mask = labels == 1
        neg_mask = labels == 0

        loss = 0.0
        count = 0
        if pos_mask.any():
            loss += F.mse_loss(
                edge_tokens[pos_mask],
                yes_embed.expand_as(edge_tokens[pos_mask])
            )
            count += pos_mask.sum().item()
        if neg_mask.any():
            loss += F.mse_loss(
                edge_tokens[neg_mask],
                no_embed.expand_as(edge_tokens[neg_mask])
            )
            count += neg_mask.sum().item()

        return loss / max(count, 1)

    def forward(self, graph, indices, snapshot_times=None):
        """
        Args:
            graph: DGLGraph（Link 任务）或 None（Node CLS 复用缓存）
            indices: edge_pairs [E, 2]（Link）或 node_ids [B]（CLS）
        Returns:
            scores/logits from task_head
        """
        h = self._get_temgh_output(graph, snapshot_times)

        if self.task_type == "link_prediction":
            # Link prediction
            E = indices.size(0)
            batch_nodes = indices.flatten()  # [E*2]
            node_embs = h[batch_nodes]  # [E*2, graph_dim]
            node_embs = self._fuse_precomputed(node_embs, batch_nodes)
            word_embeddings = self.llm.get_input_embeddings().weight
            tokens = self.encoder(node_embs, word_embeddings)  # [E*2, 1, llm_dim]
            tokens = tokens.view(E, 2, -1)  # [E, 2, llm_dim]
            prompt_text = self.task.build_prompt(domain="author" if self.target_ntype == "author" else "wiki")
            inputs_embeds, attention_mask = self.task.prepare_inputs(
                self.llm, self.tokenizer, prompt_text, tokens, self.device
            )
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        # Align dtype with LLM (e.g., float16)
        inputs_embeds = inputs_embeds.to(self.llm.dtype)

        # LLM forward (skip lm_head)
        outputs = self.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False
        )
        last_hidden = outputs.last_hidden_state[:, -1, :]  # [B, llm_dim]
        out = self.task_head(last_hidden.float())
        return out.squeeze(-1) if out.dim() == 2 and out.size(1) == 1 else out
