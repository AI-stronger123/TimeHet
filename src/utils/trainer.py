import os
import time
import json
import torch
from typing import Dict, Any

from .common import ensure_dir


def train_epoch(model, train_loader, task, optimizer, device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        loss, _ = task.train_step(model, batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=task.config.get("grad_clip", 1.0)
        )
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {"train_loss": total_loss / max(num_batches, 1)}


@torch.no_grad()
def evaluate(model, val_loader, task, device) -> Dict[str, float]:
    model.eval()
    all_metrics = []
    for batch in val_loader:
        metrics = task.eval_step(model, batch)
        all_metrics.append(metrics)

    if not all_metrics:
        return {}

    # Allow task-specific aggregation (e.g., LinkPredictionTask needs AUC over all batches)
    if hasattr(task, "aggregate_metrics"):
        return task.aggregate_metrics(all_metrics)

    # Default: average per-batch metrics
    result = {}
    for k in all_metrics[0].keys():
        values = [m[k] for m in all_metrics if k in m]
        if values:
            result[k] = sum(values) / len(values)
    return result


def save_checkpoint(path: str, model, optimizer, epoch: int, best_metric: float,
                    task_type: str, config: Dict[str, Any], is_best: bool = False):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "task_type": task_type,
        "config": config,
    }
    if hasattr(model, "encoder"):
        ckpt["encoder_state_dict"] = model.encoder.state_dict()
    if hasattr(model, "task_head"):
        ckpt["head_state_dict"] = model.task_head.state_dict()
    torch.save(ckpt, path)


def run_training(model, dataset, task, config: Dict[str, Any], output_dir: str, device):
    ensure_dir(output_dir)
    log_path = os.path.join(output_dir, "train.log")
    metrics_jsonl = os.path.join(output_dir, "metrics.jsonl")

    # Save config snapshot
    import yaml
    config_path = os.path.join(output_dir, "config_snapshot.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # Build loaders
    train_loader = task.build_train_loader(dataset, config.get("training", {}))
    val_loader = task.build_val_loader(dataset, config.get("training", {}))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 5e-4)
    )

    epochs = config["training"]["epochs"]
    patience = config["training"].get("patience", 100)
    save_every = config.get("output", {}).get("save_every", 10)

    best_metric = -float("inf")
    patience_counter = 0
    total_start = time.time()

    log_file = open(log_path, "w", encoding="utf-8")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, task, optimizer, device)
        val_metrics = evaluate(model, val_loader, task, device)

        current = task.get_best_metric(val_metrics)
        is_best = current > best_metric
        if is_best:
            best_metric = current
            patience_counter = 0
            best_path = os.path.join(output_dir, "best.pt")
            save_checkpoint(best_path, model, optimizer, epoch, best_metric,
                            config["task"]["type"], config, is_best=True)
            msg_extra = f">>> Saved best model (Val metric: {current:.4f})"
        else:
            patience_counter += 1
            msg_extra = f"  (no improvement, patience {patience_counter}/{patience})"

        # Periodic save
        if save_every and epoch % save_every == 0:
            last_path = os.path.join(output_dir, "last.pt")
            save_checkpoint(last_path, model, optimizer, epoch, best_metric,
                            config["task"]["type"], config, is_best=False)

        elapsed = time.time() - t0
        total_elapsed = time.time() - total_start

        # Build log line
        parts = [f"Epoch {epoch:02d}/{epochs}", f"Loss: {train_metrics['train_loss']:.4f}"]
        for k, v in val_metrics.items():
            parts.append(f"Val {k.capitalize()}: {v:.4f}")
        parts.append(f"Time: {elapsed:.1f}s")
        if epoch == epochs or patience_counter >= patience:
            parts.append(f"Total: {total_elapsed:.1f}s")
        line = " | ".join(parts)

        print(line)
        print(msg_extra)
        log_file.write(line + "\n")
        log_file.write(msg_extra + "\n")
        log_file.flush()

        # JSONL
        record = {"epoch": epoch, **train_metrics}
        for k, v in val_metrics.items():
            record[f"val_{k}"] = v
        record["time_sec"] = elapsed
        record["is_best"] = is_best
        with open(metrics_jsonl, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(record) + "\n")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    log_file.close()
    print("Training completed.")
