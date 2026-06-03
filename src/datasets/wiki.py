import os
import numpy as np
import torch
import dgl
from typing import List, Tuple

from .base import BaseLinkDataset


def _build_wiki_snapshot(start, end, src_all, dst_all, edge_feat_all, num_nodes, feat_dim):
    src = torch.tensor(src_all[start:end], dtype=torch.long)
    dst = torch.tensor(dst_all[start:end], dtype=torch.long)
    efeat = torch.tensor(edge_feat_all[start:end], dtype=torch.float32)
    g = dgl.heterograph(
        {("node", "edit", "node"): (src, dst), ("node", "edit_rev", "node"): (dst, src)},
        num_nodes_dict={"node": num_nodes}
    )
    node_feat = torch.zeros(num_nodes, feat_dim)
    node_count = torch.zeros(num_nodes, 1)
    node_feat.index_add_(0, src, efeat)
    node_feat.index_add_(0, dst, efeat)
    ones = torch.ones(len(src), 1)
    node_count.index_add_(0, src, ones)
    node_count.index_add_(0, dst, ones)
    node_count[node_count == 0] = 1
    g.nodes["node"].data["feat"] = node_feat / node_count
    return g


def _sample_negative_edges_excluding_existing(pos_src, pos_dst, num_nodes):
    pos_src_np = pos_src.cpu().numpy()
    pos_dst_np = pos_dst.cpu().numpy()
    n = len(pos_src_np)
    existing = set(zip(pos_src_np.tolist(), pos_dst_np.tolist()))
    neg_src = pos_src_np.copy()
    neg_dst = np.zeros(n, dtype=np.int64)
    for i in range(n):
        s = int(neg_src[i])
        for _ in range(100):
            d = np.random.randint(0, num_nodes)
            if (s, d) not in existing:
                neg_dst[i] = d
                break
        else:
            neg_dst[i] = (s + 1) % num_nodes
    return torch.from_numpy(neg_src).long(), torch.from_numpy(neg_dst).long()


def _build_wiki_samples(processed_dir, time_window=3):
    data = np.load(os.path.join(processed_dir, "tgbl_wiki_full.npz"))
    window_ranges = np.load(os.path.join(processed_dir, "window_ranges.npy"))
    src_all = data["sources"]
    dst_all = data["destinations"]
    edge_feat_all = data["edge_feat"]
    all_ts = data["timestamps"]
    num_nodes = int(max(src_all.max(), dst_all.max()) + 1)
    feat_dim = edge_feat_all.shape[1]

    snapshots = [
        _build_wiki_snapshot(s, e, src_all, dst_all, edge_feat_all, num_nodes, feat_dim)
        for s, e in window_ranges
    ]

    feats, labels, ts_data = [], [], []
    t0 = all_ts[0]
    for i in range(len(snapshots) - time_window):
        input_snaps = snapshots[i:i + time_window]
        current_window_ts = [
            (all_ts[window_ranges[j][0]:window_ranges[j][1]].mean() - t0) / 86400.0
            for j in range(i, i + time_window)
        ]
        data_dict = {}
        for t, g in enumerate(input_snaps):
            for srctype, etype, dsttype in g.canonical_etypes:
                src, dst = g.edges(etype=(srctype, etype, dsttype))
                data_dict[(srctype, f"{etype}_t{t}", dsttype)] = (src, dst)
        big_graph = dgl.heterograph(data_dict, num_nodes_dict={"node": num_nodes})
        for t, g in enumerate(input_snaps):
            big_graph.nodes["node"].data[f"t{t}"] = g.nodes["node"].data["feat"].float()

        pos_g = snapshots[i + time_window][("node", "edit", "node")]
        pos_src, pos_dst = pos_g.edges()
        neg_src, neg_dst = _sample_negative_edges_excluding_existing(pos_src, pos_dst, num_nodes)
        neg_graph = dgl.heterograph({("node", "edit", "node"): (neg_src, neg_dst)}, {"node": num_nodes})

        feats.append(big_graph)
        labels.append((pos_g, neg_graph))
        ts_data.append(current_window_ts)

    return feats, labels, ts_data, feat_dim, num_nodes


class WikiLinkDataset(BaseLinkDataset):
    """Wiki 链接预测数据集。"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.processed_dir = config["processed_dir"]
        self.time_window = config.get("time_window", 3)
        self.train_ratio = config.get("train_ratio", 0.7)
        self.val_ratio = config.get("val_ratio", 0.15)
        self.device = config.get("device", "cuda")

        feats, labels, ts_data, feat_dim, num_nodes = _build_wiki_samples(self.processed_dir, self.time_window)
        self._feat_dim = feat_dim
        self._num_nodes = num_nodes

        n_total = len(feats)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        self.train_feats = feats[:n_train]
        self.train_labels = labels[:n_train]
        self.train_ts = ts_data[:n_train]
        self.val_feats = feats[n_train:n_train + n_val]
        self.val_labels = labels[n_train:n_train + n_val]
        self.val_ts = ts_data[n_train:n_train + n_val]
        self.test_feats = feats[n_train + n_val:]
        self.test_labels = labels[n_train + n_val:]
        self.test_ts = ts_data[n_train + n_val:]

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def target_ntype(self) -> str:
        return "node"

    def load(self, split: str) -> Tuple[List[dgl.DGLGraph], List[Tuple], List[List[float]]]:
        if split == "train":
            return self.train_feats, self.train_labels, self.train_ts
        elif split == "val":
            return self.val_feats, self.val_labels, self.val_ts
        elif split == "test":
            return self.test_feats, self.test_labels, self.test_ts
        else:
            raise ValueError(f"Unknown split: {split}")

