import os
import numpy as np
import torch
import dgl
from dgl.data.utils import load_graphs
from typing import List, Tuple
from .base import BaseLinkDataset


def _generate_APA(graph, device):
    """Compute author-author co-occurrence matrix via AP * PA^T."""
    AP = graph.adj(etype=('author', 'writes', 'paper')).to_dense()
    PA = AP.t()
    APA = torch.mm(AP.to(device), PA.to(device)).detach().cpu()
    APA[torch.eye(APA.shape[0]).bool()] = 0.5
    return APA

def _construct_htg_mag(glist, idx, time_window):
    sub_glist = glist[idx - time_window:idx]
    ID_dict = {}
    for ntype in glist[0].ntypes:
        ID_set = set()
        for g_s in sub_glist:
            tmp_set = set(g_s.ndata["_ID"][ntype].tolist())
            ID_set.update(tmp_set)
        ID_dict[ntype] = {ID: idx for idx, ID in enumerate(sorted(list(ID_set)))}

    hetero_dict = {}
    for (t, g_s) in enumerate(sub_glist):
        for srctype, etype, dsttype in g_s.canonical_etypes:
            src, dst = g_s.in_edges(g_s.nodes(dsttype), etype=etype)
            ID_src = g_s.ndata["_ID"][srctype]
            ID_dst = g_s.ndata["_ID"][dsttype]
            new_src = ID_src[src]
            new_dst = ID_dst[dst]
            new_new_src = [ID_dict[srctype][e.item()] for e in new_src]
            new_new_dst = [ID_dict[dsttype][e.item()] for e in new_dst]
            hetero_dict[(srctype, f"{etype}_t{t}", dsttype)] = (new_new_src, new_new_dst)
            hetero_dict[(dsttype, f"{etype}_r_t{t}", srctype)] = (new_new_dst, new_new_src)

    G_feat = dgl.heterograph(hetero_dict)
    for (t, g_s) in enumerate(sub_glist):
        for ntype in G_feat.ntypes:
            feat_dim = g_s.nodes[ntype].data["feat"].shape[1]
            G_feat.nodes[ntype].data[f"t{t}"] = torch.zeros(G_feat.num_nodes(ntype), feat_dim)
            node_id = g_s.ndata["_ID"][ntype]
            node_feat = g_s.nodes[ntype].data["feat"]
            for (id_, feat) in zip(node_id, node_feat):
                G_feat.nodes[ntype].data[f"t{t}"][ID_dict[ntype][id_.item()]] = feat

    snapshot_ts = [float(idx - time_window + t) for t in range(time_window)]
    return G_feat, snapshot_ts


def _construct_htg_label_mag(glist, idx, device):
    APA_cur = _generate_APA(glist[idx], device)
    APA_pre = _generate_APA(glist[idx - 1], device)
    APA_pre = (APA_pre > 0.5).float()
    APA_cur = (APA_cur > 0.5).float()
    APA_sub = APA_cur - APA_pre
    APA_add = APA_cur + APA_pre
    APA_add[torch.eye(APA_add.shape[0]).bool()] = 0.5

    indices_true = (APA_sub == 1).nonzero(as_tuple=True)
    indices_false = (APA_add == 0).nonzero(as_tuple=True)
    pos_src = indices_true[0]
    pos_dst = indices_true[1]
    size = int(pos_src.shape[0] * 0.1)
    pos_idx = torch.randperm(pos_src.shape[0])[:size]
    pos_src = pos_src[pos_idx]
    pos_dst = pos_dst[pos_idx]

    neg_src = indices_false[0]
    neg_dst = indices_false[1]
    neg_idx = torch.randperm(neg_src.shape[0])[:size]
    neg_src = neg_src[neg_idx]
    neg_dst = neg_dst[neg_idx]

    num_nodes = APA_cur.shape[0]
    return dgl.graph((pos_src, pos_dst), num_nodes=num_nodes), dgl.graph((neg_src, neg_dst), num_nodes=num_nodes)


class MAGLinkDataset(BaseLinkDataset):
    """MAG 链接预测数据集。"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.data_path = config["data_path"]
        self.time_window = config.get("time_window", 3)
        self.device = config.get("device", "cuda")
        self._feat_dim = None
        self._num_nodes = None
        self._load_all()

    def _load_all(self):
        glist, _ = load_graphs(self.data_path)
        from MODELS.utils import mp2vec_feat
        mp2vec_dir = self.config.get("mp2vec_dir", os.path.join(os.path.dirname(__file__), "../../../data/mp2vec"))
        glist = [mp2vec_feat(os.path.join(mp2vec_dir, f"g{i}.vector"), g) for (i, g) in enumerate(glist)]
        self.glist = glist

        # Determine feat_dim and num_nodes from first graph
        first_g = glist[0]
        self._feat_dim = first_g.nodes["author"].data["feat"].shape[1]
        self._num_nodes = first_g.num_nodes("author")

        self.train_feats, self.train_labels, self.train_ts = [], [], []
        self.val_feats, self.val_labels, self.val_ts = [], [], []
        self.test_feats, self.test_labels, self.test_ts = [], [], []

        for i in range(len(glist)):
            if i >= self.time_window:
                G_feat, snapshot_ts = _construct_htg_mag(glist, i, self.time_window)
                pos_label, neg_label = _construct_htg_label_mag(glist, i, self.device)
                if i == len(glist) - 1:
                    self.test_feats.append(G_feat)
                    self.test_labels.append((pos_label, neg_label))
                    self.test_ts.append(snapshot_ts)
                elif i == len(glist) - 2:
                    self.val_feats.append(G_feat)
                    self.val_labels.append((pos_label, neg_label))
                    self.val_ts.append(snapshot_ts)
                else:
                    self.train_feats.append(G_feat)
                    self.train_labels.append((pos_label, neg_label))
                    self.train_ts.append(snapshot_ts)

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def target_ntype(self) -> str:
        return "author"

    def load(self, split: str) -> Tuple[List[dgl.DGLGraph], List[Tuple], List[List[float]]]:
        if split == "train":
            return self.train_feats, self.train_labels, self.train_ts
        elif split == "val":
            return self.val_feats, self.val_labels, self.val_ts
        elif split == "test":
            return self.test_feats, self.test_labels, self.test_ts
        else:
            raise ValueError(f"Unknown split: {split}")

