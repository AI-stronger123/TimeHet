"""
    python run_regression.py
    python run_regression.py --hidden_dim 64 --epochs 200 --runs 5
"""
import argparse
import os
import random
import statistics
import sys

import dgl
from dgl.data.utils import load_graphs
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

from MODELS.TEMGH import TEMGH, NodePredictor


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class GraphPairDataset(Dataset):
  使用。"""
    def __init__(self, feats, labels):
        self.feats = feats
        self.labels = labels

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]


def load_covid_sequences(glist, time_window=14):
    feats = []
    labels = []

    for i in range(len(glist) - time_window):
        graphs_window = glist[i:i + time_window]
        target_graph = glist[i + time_window]

        data_dict = {}
        num_nodes_dict = {
            ntype: graphs_window[-1].num_nodes(ntype)
            for ntype in graphs_window[-1].ntypes
        }

        for t, g in enumerate(graphs_window):
            for srctype, etype, dsttype in g.canonical_etypes:
                src, dst = g.edges(etype=(srctype, etype, dsttype))
                new_etype = f"{etype}_t{t}"
                data_dict[(srctype, new_etype, dsttype)] = (src, dst)


        base_graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

 
        for ntype in base_graph.ntypes:
            for t, g in enumerate(graphs_window):
                base_graph.nodes[ntype].data[f"t{t}"] = g.nodes[ntype].data["feat"].float()

        feats.append(base_graph)
        labels.append(target_graph)


    n_total = len(feats)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    train_feats = feats[:n_train]
    train_labels = labels[:n_train]
    val_feats = feats[n_train:n_train + n_val]
    val_labels = labels[n_train:n_train + n_val]
    test_feats = feats[n_train + n_val:]
    test_labels = labels[n_train + n_val:]

    mean_std = {}
    for ntype in train_feats[0].ntypes:
        vals = []
        for g in train_feats:
            for t in range(time_window):
                vals.append(g.nodes[ntype].data[f"t{t}"])
        for g in train_labels:
            if "feat" in g.nodes[ntype].data:
                vals.append(g.nodes[ntype].data["feat"])
        vals = torch.cat(vals)
        mean_std[ntype] = (vals.mean().item(), vals.std().item() + 1e-8)


    def normalize_graphs(graph_list):
        for g in graph_list:
            for ntype in g.ntypes:
                mu, sigma = mean_std[ntype]
                for key in list(g.nodes[ntype].data.keys()):
                    g.nodes[ntype].data[key] = (g.nodes[ntype].data[key] - mu) / sigma

    normalize_graphs(train_feats)
    normalize_graphs(val_feats)
    normalize_graphs(test_feats)
    normalize_graphs(train_labels)
    normalize_graphs(val_labels)
    normalize_graphs(test_labels)

    return train_feats, train_labels, val_feats, val_labels, test_feats, test_labels, mean_std


@torch.no_grad()
def evaluate(model, loader, device, mean_std, predict_type="state"):
    model[0].eval()
    model[1].eval()

    mu, sigma = mean_std[predict_type]
    total_mae = 0.0
    total_mse = 0.0
    total_nodes = 0

    for g_feat, label in loader:
        g_feat = g_feat.to(device)
        label = label.to(device)

        h = model[0](g_feat, predict_type)
        pred = model[1](h)

        pred_orig = pred * sigma + mu
        label_orig = label * sigma + mu

        total_mae += F.l1_loss(pred_orig, label_orig, reduction='sum').item()
        total_mse += F.mse_loss(pred_orig, label_orig, reduction='sum').item()
        total_nodes += label.shape[0]

    mae = total_mae / total_nodes
    rmse = (total_mse / total_nodes) ** 0.5
    return mae, rmse


def main():
    parser = argparse.ArgumentParser(description="TEMGH + MLP for COVID Node Regression")
    parser.add_argument("--data_path", type=str, default="data/covid/covid_graphs.bin")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--time_window", type=int, default=14)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (based on val RMSE)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=1, help="每 N 个 epoch 评估一次 val/test")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/covid_reg")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "train.log")
    log_file = open(log_path, "w", encoding="utf-8")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = Tee(sys.stdout, log_file)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("Loading COVID graphs ...")
    glist, _ = load_graphs(args.data_path)

    print("Generating train / val / test sequences ...")
    train_feats, train_labels, val_feats, val_labels, test_feats, test_labels, mean_std = load_covid_sequences(
        glist, time_window=args.time_window
    )

    print(f"#train windows: {len(train_feats)}")
    print(f"#val windows:   {len(val_feats)}")
    print(f"#test windows:  {len(test_feats)}")
    print(f"State norm -> mean: {mean_std['state'][0]:.4f}, std: {mean_std['state'][1]:.4f}")

    sample_graph = train_feats[0]
    in_dim = sample_graph.nodes["state"].data["t0"].shape[1]

    if device.type != 'cpu':
        train_feats = [g.to(device) for g in train_feats]
        val_feats = [g.to(device) for g in val_feats]
        test_feats = [g.to(device) for g in test_feats]

    def collate_fn(batch):
        feats, labels = zip(*batch)
        batched_feat = dgl.batch(feats)
        label_feats = torch.cat(
            [g.nodes['state'].data['feat'].float() for g in labels], dim=0
        )
        return batched_feat, label_feats

    train_dataset = GraphPairDataset(train_feats, train_labels)
    val_dataset = GraphPairDataset(val_feats, val_labels)
    test_dataset = GraphPairDataset(test_feats, test_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )

    mae_runs = []
    rmse_runs = []

    for k in range(args.runs):
        print(f"\n{'='*60}")
        print(f"Run {k + 1} / {args.runs}")
        print(f"{'='*60}")
        set_seed(args.seed + k)

        encoder = TEMGH(
            graph=sample_graph,
            n_inp=in_dim,
            n_hid=args.hidden_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            time_window=args.time_window,
            norm=True,
            device=device,
            dropout=args.dropout,
        ).to(device)

        out_dim = train_labels[0].nodes["state"].data["feat"].shape[1]
        predictor = NodePredictor(args.hidden_dim, out_dim).to(device)

        model = [encoder, predictor]

        n_params = sum(p.numel() for p in list(encoder.parameters()) + list(predictor.parameters()) if p.requires_grad)
        print(f"# params: {n_params}")

        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(predictor.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        use_amp = device.type != 'cpu'
        scaler = GradScaler() if use_amp else None

        best_val_rmse = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, args.epochs + 1):
            encoder.train()
            predictor.train()

            train_loss_list = []

            for g_feat, label in train_loader:
                g_feat = g_feat.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                if use_amp:
                    with autocast():
                        h = encoder(g_feat, "state")
                        pred = predictor(h)
                        loss = F.smooth_l1_loss(pred, label, beta=0.5)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    h = encoder(g_feat, "state")
                    pred = predictor(h)
                    loss = F.smooth_l1_loss(pred, label, beta=0.5)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0
                    )
                    optimizer.step()

                train_loss_list.append(loss.item())

            scheduler.step()
            train_loss = float(np.mean(train_loss_list))

            if epoch % args.eval_every == 0:
                val_mae, val_rmse = evaluate(model, val_loader, device, mean_std)

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    patience_counter = 0

                    # Save best weights
                    os.makedirs(args.save_dir, exist_ok=True)
                    combined_state = {}
                    for key, v in encoder.state_dict().items():
                        combined_state[f"0.{key}"] = v
                    for key, v in predictor.state_dict().items():
                        combined_state[f"1.{key}"] = v
                    torch.save(combined_state, os.path.join(args.save_dir, "best.pt"))
                    torch.save(
                        {"mean": mean_std["state"][0], "std": mean_std["state"][1]},
                        os.path.join(args.save_dir, "norm_stats.pt")
                    )
                    print(
                        f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                        f"val_mae={val_mae:.4f} val_rmse={val_rmse:.4f} | [Saved best]"
                    )
                else:
                    patience_counter += 1
                    print(
                        f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                        f"val_mae={val_mae:.4f} val_rmse={val_rmse:.4f} | "
                        f"patience {patience_counter}/{args.patience}"
                    )

                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}")

        # Training finished, load best model and evaluate on test once
        best_ckpt = torch.load(os.path.join(args.save_dir, "best.pt"), map_location=device)
        encoder.load_state_dict({k.replace("0.", "", 1): v for k, v in best_ckpt.items() if k.startswith("0.")})
        predictor.load_state_dict({k.replace("1.", "", 1): v for k, v in best_ckpt.items() if k.startswith("1.")})
        test_mae, test_rmse = evaluate(model, test_loader, device, mean_std)
        print(f"\nRun {k + 1} Test MAE:  {test_mae:.4f}")
        print(f"Run {k + 1} Test RMSE: {test_rmse:.4f}")

        mae_runs.append(test_mae)
        rmse_runs.append(test_rmse)

    print(f"\n{'='*60}")
    print("Final Results")
    print(f"{'='*60}")
    if len(mae_runs) > 1:
        print(f"MAE:  {statistics.mean(mae_runs):.4f} ± {statistics.stdev(mae_runs):.4f}")
        print(f"RMSE: {statistics.mean(rmse_runs):.4f} ± {statistics.stdev(rmse_runs):.4f}")
    else:
        print(f"MAE:  {mae_runs[0]:.4f}")
        print(f"RMSE: {rmse_runs[0]:.4f}")
    print(f"{'='*60}")
    log_file.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
