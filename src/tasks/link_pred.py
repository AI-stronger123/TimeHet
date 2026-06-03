"""链接预测任务。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import dgl
from sklearn.metrics import roc_auc_score, average_precision_score

from .base import BaseTask


def _sample_edges(pos_g, neg_g, max_per_snap, device):
    pos_src, pos_dst = pos_g.edges()
    neg_src, neg_dst = neg_g.edges()
    pos_edges = torch.stack([pos_src, pos_dst], dim=1)
    neg_edges = torch.stack([neg_src, neg_dst], dim=1)
    n_pos = min(len(pos_edges), max_per_snap)
    n_neg = min(len(neg_edges), max_per_snap)
    perm_pos = torch.randperm(len(pos_edges))[:n_pos]
    perm_neg = torch.randperm(len(neg_edges))[:n_neg]
    pos_edges = pos_edges[perm_pos].to(device)
    neg_edges = neg_edges[perm_neg].to(device)
    all_edges = torch.cat([pos_edges, neg_edges], dim=0)
    all_labels = torch.cat([
        torch.ones(len(pos_edges), device=device),
        torch.zeros(len(neg_edges), device=device)
    ])
    perm = torch.randperm(len(all_edges))
    return all_edges[perm], all_labels[perm]


class LinkPredictionIterable:
    """链接预测训练迭代器：遍历 snapshot，做正负采样后合并。支持 snapshot 子采样。"""
    def __init__(self, feats, labels, ts_list, max_per_snap, device, batch_size,
                 max_snapshots=None):
        self.feats = feats
        self.labels = labels
        self.ts_list = ts_list
        self.max_per_snap = max_per_snap
        self.device = device
        self.batch_size = batch_size
        self.max_snapshots = max_snapshots

    def _pick_snapshots(self):
        n = len(self.feats)
        if self.max_snapshots is not None and self.max_snapshots < n:
            return torch.randperm(n)[:self.max_snapshots].tolist()
        return list(range(n))

    def __iter__(self):
        all_batches = []
        idxs = self._pick_snapshots()
        for i in idxs:
            G_feat = self.feats[i].to(self.device)
            pos_g = self.labels[i][0].to(self.device)
            neg_g = self.labels[i][1].to(self.device)
            ts = self.ts_list[i]
            edges, lbls = _sample_edges(pos_g, neg_g, self.max_per_snap, self.device)
            for start in range(0, len(edges), self.batch_size):
                end = min(start + self.batch_size, len(edges))
                all_batches.append((G_feat, edges[start:end], lbls[start:end], ts))
        perm = torch.randperm(len(all_batches))
        for i in perm:
            yield all_batches[i]

    def __len__(self):
        idxs = self._pick_snapshots()
        total = 0
        for i in idxs:
            _, (pos_g, neg_g), _ = self.feats[i], self.labels[i], self.ts_list[i]
            n = min(pos_g.num_edges(), self.max_per_snap) + min(neg_g.num_edges(), self.max_per_snap)
            total += (n + self.batch_size - 1) // self.batch_size
        return total


class LinkPredictionTask(BaseTask):
    def create_head(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.llm_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def build_prompt(self, **kwargs) -> str:
        domain = kwargs.get("domain", "author")
        override = self.config.get("prompt_override", None)
        if override:
            return override
        if domain == "wiki":
            return (
                "USER: Given two wiki page-centered subgraphs: <graph> and <graph>, "
                "we need to predict whether these two pages will be edited together in the next time period. "
                "Answer yes or no.\n"
                "ASSISTANT:"
            )
        else:
            return (
                "USER: Given two author-centered subgraphs: <graph> and <graph>, "
                "we need to predict whether these two authors will collaborate in the next time period. "
                "Answer yes or no.\n"
                "ASSISTANT:"
            )

    def _make_loader(self, dataset, config, split, max_key, max_snapshots_key=None):
        feats, labels, ts_list = dataset.load(split)
        return LinkPredictionIterable(
            feats, labels, ts_list,
            max_per_snap=config.get(max_key, 600),
            device=config.get("device", "cuda"),
            batch_size=config["batch_size"],
            max_snapshots=config.get(max_snapshots_key) if max_snapshots_key else None,
        )

    def build_train_loader(self, dataset, config: dict):
        return self._make_loader(dataset, config, "train",
                                 "max_train_per_snapshot", "max_train_snapshots_per_epoch")

    def build_val_loader(self, dataset, config: dict):
        return self._make_loader(dataset, config, "val", "max_val_per_snapshot")

    def build_test_loader(self, dataset, config: dict):
        return self._make_loader(dataset, config, "test", "max_test_per_snapshot")

    def train_step(self, model, batch) -> Tuple[torch.Tensor, dict]:
        G_feat, edges, lbls, ts = batch
        scores = model(G_feat, edges, snapshot_times=ts)
        loss_link = F.binary_cross_entropy_with_logits(scores, lbls)

        beta = self.config.get("beta", 0.0)
        if beta > 0:
            loss_align = model.compute_alignment_loss(G_feat, edges, lbls, snapshot_times=ts)
            loss = loss_link + beta * loss_align
        else:
            loss = loss_link
            loss_align = torch.tensor(0.0)

        return loss, {"loss_link": loss_link.item(), "loss_align": loss_align.item() if torch.is_tensor(loss_align) else 0.0}

    def eval_step(self, model, batch) -> dict:
        G_feat, edges, lbls, ts = batch
        scores = model(G_feat, edges, snapshot_times=ts).sigmoid().cpu()
        lbls = lbls.cpu()
        preds = (scores > 0.5).float()
        acc = (preds == lbls).float().mean().item()
        return {
            "_scores": scores,
            "_labels": lbls,
            "acc": acc,
        }

    def aggregate_metrics(self, all_metrics: List[dict]) -> dict:
        """聚合所有 batch 的 scores 和 labels，计算 AUC/AP/Acc。"""
        all_scores = torch.cat([m["_scores"] for m in all_metrics]).numpy()
        all_labels = torch.cat([m["_labels"] for m in all_metrics]).numpy()
        auc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)
        acc = ((all_scores > 0.5) == all_labels).mean()
        return {"auc": auc, "ap": ap, "acc": acc}

    def get_best_metric(self, metrics: dict) -> float:
        return metrics.get("auc", 0.0)
