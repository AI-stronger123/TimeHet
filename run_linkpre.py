"""Unified training & evaluation entry point for GraphLLM link prediction.
Usage:
    python run_linkpre.py --config experiments/mag_link/config.yaml
    python run_linkpre.py --config experiments/mag_link/config.yaml --mode test --resume outputs/xxx/best.pt
"""
import argparse
import os
import sys

# Resolve src/ relative to this script so it works regardless of CWD
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_src = os.path.join(_PROJECT_ROOT, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import torch
import torch.nn as nn

from utils.config import load_config
from datasets import build_dataset
from utils.common import ensure_dir, get_timestamp, set_seed
from utils.trainer import run_training, evaluate
from llm import load_llm
from encoders import build_encoder
from tasks import build_task
from graph_llm_model import GraphLLMModel
from MODELS.TEMGH import TEMGH, LinkPredictor


def main():
    parser = argparse.ArgumentParser(
        description="GraphLLM link prediction trainer / evaluator"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config (e.g. experiments/mag_link/config.yaml)"
    )
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test"],
        help="Run mode"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Checkpoint path to resume from"
    )
    parser.add_argument(
        "--num_runs", type=int, default=5,
        help="Number of test runs for mean+-std reporting"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed in config"
    )
    args, extra = parser.parse_known_args()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    cfg = load_config(args.config, extra)
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
        print(f"[Seed] Overridden to {args.seed}")
    set_seed(cfg.get("training", {}).get("seed", 666))
    device = torch.device(cfg.get("training", {}).get("device", "cuda"))

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = build_dataset(cfg["dataset"])

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    llm_model, tokenizer, llm_config = load_llm(cfg["llm"], device=device)

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------
    encoder_cfg = cfg.get("encoder", {})
    encoder = build_encoder(
        encoder_cfg,
        llm_dim=llm_config.hidden_size,
        vocab_size=llm_config.vocab_size
    )

    # ------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------
    task = build_task(cfg["task"], llm_dim=llm_config.hidden_size)

    # ------------------------------------------------------------------
    # TEMGH backbone
    # ------------------------------------------------------------------
    if hasattr(dataset, "get_template_graph"):
        template_graph = dataset.get_template_graph(device)
    else:
        train_feats, _, _ = dataset.load("train")
        template_graph = train_feats[0].to(device)

    temgh_cfg = dataset.build_temgh_config()
    temgh = TEMGH(graph=template_graph, **temgh_cfg, device=device)
    print(f"[Backbone] Using TEMGH (n_hid={temgh_cfg['n_hid']}, n_layers={temgh_cfg['n_layers']})")

    predictor = LinkPredictor(n_inp=temgh_cfg["n_hid"], n_classes=1)
    temgh_model = nn.Sequential(temgh, predictor).to(device)

    ckpt_path = cfg.get("temgh", {}).get("ckpt")
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)

        # 兼容旧版 checkpoint: 0.gnn_layer.xxx -> gnn_layers.0.xxx
        def _remap(k):
            if k.startswith("0.gnn_layer."):
                return k.replace("0.gnn_layer.", "gnn_layers.0.", 1)
            elif k.startswith("0."):
                return k.replace("0.", "", 1)
            return k

        temgh_sd = {_remap(k): v for k, v in ckpt.items() if k.startswith("0.")}
        missing, unexpected = temgh_model[0].load_state_dict(temgh_sd, strict=False)
        print(f"[TEMGH] Pretrained checkpoint loaded from {ckpt_path}")
        if missing:
            print(f"[TEMGH] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[TEMGH] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    else:
        if ckpt_path:
            print(f"[TEMGH] Checkpoint file not found: {ckpt_path}")
        else:
            print(f"[TEMGH] No checkpoint specified -- using random init.")

    # ------------------------------------------------------------------
    # Optional precomputed features (MAG CLS)
    # ------------------------------------------------------------------
    precomputed_feats = None
    if hasattr(dataset, "load_precomputed_feats"):
        precomputed_feats = dataset.load_precomputed_feats()

    # ------------------------------------------------------------------
    # Assemble GraphLLM model
    # ------------------------------------------------------------------
    model = GraphLLMModel(
        temgh=temgh,
        encoder=encoder,
        llm=llm_model,
        tokenizer=tokenizer,
        task=task,
        device=device,
        target_ntype=dataset.target_ntype,
        freeze_temgh=cfg.get("temgh", {}).get("freeze", True),
        precomputed_feats=precomputed_feats
    ).to(device)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    root_dir = cfg.get("output", {}).get("root_dir", "./outputs")
    root_dir = os.path.abspath(root_dir)
    exp_name = cfg.get("experiment_name", "exp")

    config_rel = os.path.relpath(args.config)
    parts = config_rel.replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[0] == "experiments":
        ds_task = parts[1]
        ds = ds_task.split("_")[0]
        task_name = "_".join(ds_task.split("_")[1:])
        beta = cfg.get("task", {}).get("beta", 0.0)
        beta_tag = f"_beta{beta}" if beta > 0 else ""
        output_dir = os.path.join(root_dir, ds, task_name, f"{get_timestamp()}_{exp_name}{beta_tag}")
    else:
        beta = cfg.get("task", {}).get("beta", 0.0)
        beta_tag = f"_beta{beta}" if beta > 0 else ""
        output_dir = os.path.join(root_dir, f"{get_timestamp()}_{exp_name}{beta_tag}")
    ensure_dir(output_dir)

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        assert ckpt["task_type"] == cfg["task"]["type"], (
            f"Checkpoint task {ckpt['task_type']} mismatch with config {cfg['task']['type']}"
        )
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if unexpected:
            print(f"[Resume] Ignored unexpected keys: {unexpected}")
        if missing:
            print(f"[Resume] Missing keys: {missing}")
        print(f"[Resume] Loaded from {args.resume}")

    # ------------------------------------------------------------------
    # Train or Test
    # ------------------------------------------------------------------
    if args.mode == "train":
        beta = cfg.get("task", {}).get("beta", 0.0)
        if beta > 0:
            print(f"[Task] beta={beta} (alignment loss enabled)")
        else:
            print(f"[Task] beta=0.0 (alignment loss disabled)")
        run_training(model, dataset, task, cfg, output_dir, device)

        best_ckpt = os.path.join(output_dir, "best.pt")
        if os.path.exists(best_ckpt):
            print("\n[Auto Test] Training finished, loading best checkpoint and running test...")
            ckpt = torch.load(best_ckpt, map_location=device)
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if unexpected:
                print(f"[Auto Test] Ignored unexpected keys: {unexpected}")
            if missing:
                print(f"[Auto Test] Missing keys: {missing}")
            test_loader = task.build_test_loader(dataset, cfg.get("training", {}))
            test_metrics = evaluate(model, test_loader, task, device)
            print("\n" + "=" * 60)
            print("Auto Test Results")
            print("=" * 60)
            for k, v in test_metrics.items():
                print(f"Test {k.capitalize()}: {v:.4f}")
            print("=" * 60)
        else:
            print("[Auto Test] best.pt not found, skipping test.")
    else:
        num_runs = args.num_runs
        all_runs = []
        for run in range(num_runs):
            set_seed(run)
            test_loader = task.build_test_loader(dataset, cfg.get("training", {}))
            test_metrics = evaluate(model, test_loader, task, device)
            all_runs.append(test_metrics)

        beta = cfg.get("task", {}).get("beta", 0.0)
        print("\n" + "=" * 60)
        print("Test Evaluation")
        if beta > 0:
            print(f"[Task] beta={beta} (alignment loss enabled)")
        print("=" * 60)
        if num_runs == 1:
            for k, v in all_runs[0].items():
                print(f"Test {k.capitalize()}: {v:.4f}")
        else:
            keys = list(all_runs[0].keys())
            for k in keys:
                values = [run[k] for run in all_runs]
                mean = sum(values) / len(values)
                std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
                print(f"Test {k.capitalize()}: {mean:.3f}+-{std:.3f}")
        print("=" * 60)



if __name__ == "__main__":
    main()
