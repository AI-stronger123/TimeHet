import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
import numpy as np

# Equations 4.1-4.3: TEMGH temporal context-aware encoder
class TEMGH_TimeEncoder(nn.Module):
    def __init__(self, n_hid, device):
        super(TEMGH_TimeEncoder, self).__init__()
        self.n_hid = n_hid
        self.device = device

        self.omega = nn.Parameter(torch.randn(n_hid))
        self.phi = nn.Parameter(torch.randn(n_hid))

        self.decay_lambda = nn.Parameter(torch.ones(1) * 0.01)
        self.w_delta = nn.Linear(1, n_hid)
        self.mlp_merge = nn.Linear(n_hid * 2, n_hid)

    def forward(self, t_now, t_prev, num_nodes):

        abs_enc = torch.sin(t_now * self.omega + self.phi)
 
        delta_t = torch.tensor([t_now - t_prev]).to(self.device).float()
        rel_enc = self.w_delta(torch.exp(-self.decay_lambda * delta_t))
 
        e_time = self.mlp_merge(torch.cat([abs_enc, rel_enc], dim=-1))
        return e_time.repeat(num_nodes, 1)

class TEMGHLayer(nn.Module):
    def __init__(self, graph, n_inp, n_hid, n_heads, timeframe, device, dropout):
        super(TEMGHLayer, self).__init__()
        self.n_hid = n_hid
        self.timeframe = timeframe
        self.device = device

        self.intra_rel_agg = nn.ModuleDict({
            etype: GATConv(n_inp, n_hid, n_heads, feat_drop=dropout, allow_zero_in_degree=True)
            for srctype, etype, dsttype in graph.canonical_etypes
        })
  
        self.attn_mlp = nn.Sequential(
            nn.Linear(n_hid * 3, n_hid),
            nn.LeakyReLU(0.2),
            nn.Linear(n_hid, 1)
        )

    def forward(self, graph, node_features, e_time_dict, s_prev_dict):
        new_features = {ntype: {} for ntype in graph.ntypes}
        for t_idx, ttype in enumerate(self.timeframe):
            for ntype in graph.ntypes:
                rel_res, rel_scores = [], []
                for stype, etype, dtype in graph.canonical_etypes:
                    if f"t{t_idx}" not in etype or dtype != ntype: continue

                    z_r = self.intra_rel_agg[etype](graph[stype, etype, dtype], 
                          (node_features[stype][ttype], node_features[dtype][ttype])).mean(1)

                    combined = torch.cat([z_r, s_prev_dict[dtype], e_time_dict[dtype]], dim=-1)
                    rel_res.append(z_r); rel_scores.append(self.attn_mlp(combined))
                
                if len(rel_res) > 0:
                    alpha = torch.softmax(torch.cat(rel_scores, dim=1), dim=1)
                    new_features[ntype][ttype] = sum(alpha[:, i:i+1] * rel_res[i] for i in range(len(rel_res)))
                else:
                    new_features[ntype][ttype] = s_prev_dict[ntype]
        return new_features


class TEMGH(nn.Module):
    def __init__(self, graph, n_inp, n_hid, n_layers, n_heads, time_window, norm, device, dropout=0.2):
        super(TEMGH, self).__init__()
        self.n_hid, self.device = n_hid, device
        self.timeframe = [f't{_}' for _ in range(time_window)]
        self.adaption_layer = nn.ModuleDict({ntype: nn.Linear(n_inp, n_hid) for ntype in graph.ntypes})
        

        self.time_encoder = TEMGH_TimeEncoder(n_hid, device)
        self.gnn_layers = nn.ModuleList([
            TEMGHLayer(graph, n_hid, n_hid, n_heads, self.timeframe, device, dropout)
            for _ in range(n_layers)
        ])
        

        self.gru_cells = nn.ModuleDict({ntype: nn.GRUCell(n_hid, n_hid) for ntype in graph.ntypes})

    def forward(self, graph, predict_type, snapshot_times=None):
        if snapshot_times is None: snapshot_times = [float(i) for i in range(len(self.timeframe))]
        inp_feat, s_v = {}, {}
        for ntype in graph.ntypes:
            inp_feat[ntype] = {ttype: self.adaption_layer[ntype](graph.nodes[ntype].data[ttype]) for ttype in self.timeframe}

            s_v[ntype] = torch.zeros(graph.num_nodes(ntype), self.n_hid).to(self.device)


        for i, ttype in enumerate(self.timeframe):
            t_now = snapshot_times[i]
            t_prev = snapshot_times[i-1] if i > 0 else t_now - (snapshot_times[i+1]-t_now if len(snapshot_times)>1 else 1.0)
            

            e_time_dict = {ntype: self.time_encoder(t_now, t_prev, graph.num_nodes(ntype)) for ntype in graph.ntypes}
            

            z_dict = inp_feat
            for layer in self.gnn_layers:
                z_dict = layer(graph, z_dict, e_time_dict, s_v)
            
            # State update (Equation 4.12)
            for ntype in graph.ntypes:
                s_v[ntype] = self.gru_cells[ntype](z_dict[ntype][ttype], s_v[ntype])
                
        return s_v[predict_type]

class LinkPredictor(nn.Module):
    def __init__(self, n_inp, n_classes):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(n_inp * 2, n_inp), nn.Linear(n_inp, n_classes)
        
    def forward(self, graph, node_feat):
        with graph.local_scope():
            graph.ndata['h'] = node_feat
            graph.apply_edges(lambda edges: {'score': self.fc2(F.relu(self.fc1(torch.cat([edges.src['h'], edges.dst['h']], 1))))})
            return graph.edata['score']


class NodePredictor(nn.Module):
    def __init__(self, n_inp: int, n_classes: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(n_inp, n_inp)
        self.fc2 = nn.Linear(n_inp, n_classes)

    def forward(self, node_feat: torch.Tensor):
        return self.fc2(F.relu(self.fc1(node_feat)))