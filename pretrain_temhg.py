import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_src = os.path.join(_PROJECT_ROOT, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

from utils.common import set_seed, ensure_dir
from datasets import build_dataset
from MODELS.TEMGH import TEMGH, LinkPredictor


class EarlyStopping:
    def __init__(self, patience=10, mode="max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        if self.mode == "max":
            improved = score > self.best_score
        else:
            improved = score < self.best_score
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def _eval_link(model, loader, target_ntype, device):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for G_feat, (pos_g, neg_g), ts in loader:
            G_feat = G_feat.to(device)
            pos_g = pos_g.to(device)
            neg_g = neg_g.to(device)
            h = model[0](G_feat, target_ntype, snapshot_times=ts if ts else None)
            pos_score = model[1](pos_g, h).squeeze().cpu().numpy()
            neg_score = model[1](neg_g, h).squeeze().cpu().numpy()
            all_scores.extend(pos_score.tolist())
            all_scores.extend(neg_score.tolist())
            all_labels.extend([1] * len(pos_score))
            all_labels.extend([0] * len(neg_score))
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, all_scores)
    return {"auc": auc}


def _train_link(model, train_loader, val_loader, target_ntype, cfg, device):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("lr", 5e-3)),
        weight_decay=float(cfg.get("weight_decay", 5e-4))
    )
    patience = int(cfg.get("patience", 10))
    epochs = int(cfg.get("epochs", 200))
    es = EarlyStopping(patience=patience, mode="max")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for G_feat, (pos_g, neg_g), ts in train_loader:
            G_feat = G_feat.to(device)
            pos_g = pos_g.to(device)
            neg_g = neg_g.to(device)
            h = model[0](G_feat, target_ntype, snapshot_times=ts if ts else None)
            pos_score = model[1](pos_g, h).squeeze()
            neg_score = model[1](neg_g, h).squeeze()
            scores = torch.cat([pos_score, neg_score])
            labels = torch.cat([
                torch.ones_like(pos_score),
                torch.zeros_like(neg_score)
            ])
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_m = _eval_link(model, val_loader, target_ntype, device)
        print(f"Pretrain {epoch}/{epochs}", flush=True)

        current_best = es.best_score
        if current_best is None or val_m["auc"] > current_best:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if es(val_m["auc"]):
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser(description="Pretrain TEMGH base model")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["mag_link", "wiki_link"],
                        help="Dataset name")
    parser.add_argument("--output", type=str, default=None,
                        help="Checkpoint save path (default: checkpoints/pretemgh/{dataset}.pt)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=70)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--device", type=str, default="cuda")

    # Model hyperparameters
    parser.add_argument("--n_hid", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--time_window", type=int, default=3)
    parser.add_argument("--norm", action="store_true", default=False,
                        help="Apply LayerNorm (default: False)")

    args = parser.parse_args()

    # Build dataset config
    if args.dataset == "mag_link":
        ds_config = {
            "name": "mag_link",
            "data_path": "./data/ogbn-mag/ogbn_graphs.bin",
            "mp2vec_dir": "./data/mp2vec",
            "time_window": args.time_window,
            "n_hid": args.n_hid,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "dropout": args.dropout,
            "norm": args.norm,
            "device": args.device,
        }
    elif args.dataset == "wiki_link":
        ds_config = {
            "name": "wiki_link",
            "processed_dir": "./data/wiki/tgb_processed",
            "time_window": args.time_window,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "n_hid": args.n_hid,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "dropout": args.dropout,
            "norm": args.norm,
            "device": args.device,
        }
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    training_cfg = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": args.device,
    }

    set_seed(args.seed)
    device = torch.device(args.device)

    dataset = build_dataset(ds_config)
    temgh_cfg = dataset.build_temgh_config()

    # Template graph
    if hasattr(dataset, "get_template_graph"):
        template_graph = dataset.get_template_graph(device)
    else:
        train_feats, _, _ = dataset.load("train")
        template_graph = train_feats[0].to(device)

    # Build model
    temgh = TEMGH(graph=template_graph, **temgh_cfg, device=device)
    predictor = LinkPredictor(n_inp=temgh_cfg["n_hid"], n_classes=1)
    model = nn.Sequential(temgh, predictor).to(device)

    train_loader = list(zip(*dataset.load("train")))
    val_loader = list(zip(*dataset.load("val")))
    test_loader = list(zip(*dataset.load("test")))

    model = _train_link(model, train_loader, val_loader, dataset.target_ntype, training_cfg, device)

    # Save checkpoint
    if args.output:
        out_path = args.output
    else:
        out_path = f"checkpoints/pretemgh/{args.dataset}.pt"
    ensure_dir(os.path.dirname(out_path))
    torch.save(model.state_dict(), out_path)
    print(f"\nCheckpoint saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
