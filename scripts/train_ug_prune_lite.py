#!/usr/bin/env python
"""Train the UG-Prune-Lite student selector from a teacher cache."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm_pruning.selectors.ug_prune_lite import UGPruneLiteSelector  # noqa: E402
from vlm_pruning.training.ug_prune_lite_dataset import UGPruneLiteCacheDataset, collate_ug_prune_lite  # noqa: E402
from vlm_pruning.training.ug_prune_lite_losses import UGPruneLiteLossWeights, ug_prune_lite_loss, ug_prune_lite_selection_metrics  # noqa: E402
from vlm_pruning.training.ug_prune_lite_model import UGPruneLiteModel, UGPruneLiteModelConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-cache", type=Path, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-loss-weight", type=float, default=1.5)
    parser.add_argument("--context-loss-weight", type=float, default=0.7)
    parser.add_argument("--keep-loss-weight", type=float, default=1.0)
    parser.add_argument("--gt-box-loss-weight", type=float, default=2.0)
    parser.add_argument("--diversity-loss-weight", type=float, default=0.1)
    parser.add_argument("--score-map-distill-weight", type=float, default=0.0)
    parser.add_argument("--score-map-distill-loss", choices=["kl", "mse"], default="kl")
    parser.add_argument("--score-map-temperature", type=float, default=2.0)
    parser.add_argument("--best-metric", default="selection/teacher_recall_at_58")
    parser.add_argument("--tie-break-metric", default="selection/gt_box_patch_coverage_at_58")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "offline"), choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "vlm-pruning"))
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def loss_weights_from_args(args: argparse.Namespace) -> UGPruneLiteLossWeights:
    return UGPruneLiteLossWeights(
        target=float(args.target_loss_weight),
        context=float(args.context_loss_weight),
        keep=float(args.keep_loss_weight),
        gt_box=float(args.gt_box_loss_weight),
        diversity=float(args.diversity_loss_weight),
        score_map_distill=float(args.score_map_distill_weight),
        score_map_temperature=float(args.score_map_temperature),
        score_map_loss=str(args.score_map_distill_loss),
    )


def move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def average_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    out: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            out[key] = sum(values) / float(len(values))
    return out


def evaluate(
    torch: Any,
    model: Any,
    dataset: UGPruneLiteCacheDataset,
    batch_size: int,
    device: Any,
    weights: UGPruneLiteLossWeights,
) -> dict[str, float]:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_ug_prune_lite(batch, torch),
    )
    metric_rows: list[dict[str, float]] = []
    selector = UGPruneLiteSelector(torch)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(batch["visual_features"], batch["query_embedding"])
            _loss, metrics = ug_prune_lite_loss(torch, outputs, batch, weights)
            metrics.update(ug_prune_lite_selection_metrics(torch, outputs, batch, selector))
            metric_rows.append(metrics)
    return average_rows(metric_rows)


def prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def metric_value(metrics: dict[str, float], key: str) -> float | None:
    if key in metrics:
        return float(metrics[key])
    val_key = f"val/{key}"
    if val_key in metrics:
        return float(metrics[val_key])
    return None


def save_checkpoint(
    torch: Any,
    path: Path,
    *,
    model: Any,
    model_config: UGPruneLiteModelConfig,
    config: dict[str, Any],
    metrics: dict[str, Any],
    epoch: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config.as_dict(),
            "selector_config": {"budget": 58, "grid": 24, "target_count": 29, "context_count": 17, "anchor_count": 12},
            "train_config": config,
            "metrics": metrics,
            "epoch": epoch,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.device == "cuda" and not torch.cuda.is_available() else args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = loss_weights_from_args(args)
    train_dataset = UGPruneLiteCacheDataset(torch, args.teacher_cache, limit=args.limit, device=device)
    feature_projection = train_dataset.metadata.get("feature_projection")
    val_dataset = (
        UGPruneLiteCacheDataset(torch, args.val_cache, limit=args.val_limit, device=device, feature_projection=feature_projection)
        if args.val_cache
        else None
    )
    first = train_dataset[0]
    model_config = UGPruneLiteModelConfig(
        visual_dim=int(first["visual_features"].shape[-1]),
        query_dim=int(first["query_embedding"].shape[-1]),
    )
    model = UGPruneLiteModel.build(torch, model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_ug_prune_lite(batch, torch),
    )
    config = {
        "method": "ug_prune_lite",
        "teacher_cache": str(args.teacher_cache),
        "val_cache": str(args.val_cache) if args.val_cache else None,
        "train/examples": len(train_dataset),
        "val/examples": len(val_dataset) if val_dataset is not None else 0,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "model": model_config.as_dict(),
        "loss_weights": weights.__dict__,
        "feature_projection": feature_projection,
        "best_metric": args.best_metric,
        "tie_break_metric": args.tie_break_metric,
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
        "save_epoch_checkpoints": args.save_epoch_checkpoints,
    }
    write_json(args.output_dir / "config.json", config)

    wandb_run = None
    if args.wandb and args.wandb_mode != "disabled":
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                mode=args.wandb_mode,
                name=args.output_dir.name,
                group="ug-prune-lite",
                job_type="train-scale",
                config=config,
            )
        except Exception as exc:
            print(f"W&B init failed; continuing locally. {type(exc).__name__}: {exc}")

    train_log_path = args.output_dir / "train_log.jsonl"
    validation_log_path = args.output_dir / "validation_log.jsonl"
    history: list[dict[str, float]] = []
    validation_history: list[dict[str, float]] = []
    selector = UGPruneLiteSelector(torch)
    best_score: float | None = None
    best_tie: float | None = None
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    epochs_without_improvement = 0

    with train_log_path.open("w", encoding="utf-8") as train_log, validation_log_path.open("w", encoding="utf-8") as val_log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_rows: list[dict[str, float]] = []
            for step, batch in enumerate(loader, start=1):
                batch = move_batch(batch, device)
                outputs = model(batch["visual_features"], batch["query_embedding"])
                loss, metrics = ug_prune_lite_loss(torch, outputs, batch, weights)
                metrics.update(ug_prune_lite_selection_metrics(torch, outputs, batch, selector))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                row = {"epoch": epoch, "step": step, **metrics}
                epoch_rows.append(metrics)
                history.append(row)
                train_log.write(json.dumps(row, ensure_ascii=True) + "\n")
                if wandb_run is not None:
                    wandb_run.log(row)
                print(f"epoch={epoch} step={step} loss={metrics['loss/total']:.4f}")

            train_epoch = prefixed("train_epoch", average_rows(epoch_rows))
            val_epoch: dict[str, float] = {}
            if val_dataset is not None:
                val_epoch = prefixed("val", evaluate(torch, model, val_dataset, args.batch_size, device, weights))
            epoch_summary = {"epoch": epoch, **train_epoch, **val_epoch}
            validation_history.append(epoch_summary)
            val_log.write(json.dumps(epoch_summary, ensure_ascii=True) + "\n")
            val_log.flush()
            if wandb_run is not None:
                wandb_run.log(epoch_summary)

            current_metrics = {"epoch": epoch, "train/latest": history[-1] if history else {}, "epoch_summary": epoch_summary}
            save_checkpoint(
                torch,
                args.output_dir / "checkpoint_last.pt",
                model=model,
                model_config=model_config,
                config=config,
                metrics=current_metrics,
                epoch=epoch,
            )
            if args.save_epoch_checkpoints:
                save_checkpoint(
                    torch,
                    args.output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                    model=model,
                    model_config=model_config,
                    config=config,
                    metrics=current_metrics,
                    epoch=epoch,
                )
            score = metric_value(epoch_summary, args.best_metric)
            tie = metric_value(epoch_summary, args.tie_break_metric) or 0.0
            if score is None:
                score = -float(metric_value(epoch_summary, "train_epoch/loss/total") or 0.0)
            improved = False
            if best_score is None or score > best_score + args.min_delta:
                improved = True
            elif best_score is not None and abs(score - best_score) <= args.min_delta and (best_tie is None or tie > best_tie):
                improved = True
            if improved:
                best_score = score
                best_tie = tie
                best_epoch = epoch
                best_metrics = current_metrics
                epochs_without_improvement = 0
                save_checkpoint(
                    torch,
                    args.output_dir / "checkpoint_best.pt",
                    model=model,
                    model_config=model_config,
                    config=config,
                    metrics=current_metrics,
                    epoch=epoch,
                )
                save_checkpoint(
                    torch,
                    args.output_dir / "checkpoint_best_selector.pt",
                    model=model,
                    model_config=model_config,
                    config=config,
                    metrics=current_metrics,
                    epoch=epoch,
                )
            else:
                epochs_without_improvement += 1
            print(f"epoch={epoch} best_epoch={best_epoch} best_score={best_score}")
            if val_dataset is not None and args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}; no improvement for {epochs_without_improvement} epochs.")
                break

    if not (args.output_dir / "checkpoint_best.pt").exists():
        shutil.copyfile(args.output_dir / "checkpoint_last.pt", args.output_dir / "checkpoint_best.pt")
    if not (args.output_dir / "checkpoint_best_selector.pt").exists():
        shutil.copyfile(args.output_dir / "checkpoint_best.pt", args.output_dir / "checkpoint_best_selector.pt")
    shutil.copyfile(args.output_dir / "checkpoint_best.pt", args.output_dir / "checkpoint.pt")
    final_metrics: dict[str, Any] = {
        "train/examples": len(train_dataset),
        "val/examples": len(val_dataset) if val_dataset is not None else 0,
        "train/steps": len(history),
        "train/initial_loss": history[0]["loss/total"] if history else None,
        "train/final_loss": history[-1]["loss/total"] if history else None,
        "best/epoch": best_epoch,
        "best/metric": args.best_metric,
        "best/value": best_score,
        "best/tie_break_metric": args.tie_break_metric,
        "best/tie_break_value": best_tie,
        "checkpoint/best": str(args.output_dir / "checkpoint_best.pt"),
        "checkpoint/best_selector": str(args.output_dir / "checkpoint_best_selector.pt"),
        "checkpoint/last": str(args.output_dir / "checkpoint_last.pt"),
        "checkpoint/compat": str(args.output_dir / "checkpoint.pt"),
    }
    if history:
        final_metrics.update({f"train/final/{key}": value for key, value in history[-1].items() if key not in {"epoch", "step"}})
    if validation_history:
        final_metrics.update({f"final_epoch/{key}": value for key, value in validation_history[-1].items() if key != "epoch"})
    if best_metrics:
        final_metrics.update({f"best_epoch/{key}": value for key, value in best_metrics["epoch_summary"].items() if key != "epoch"})
    write_json(args.output_dir / "metrics.json", final_metrics)
    if wandb_run is not None:
        wandb_run.log(final_metrics)
        wandb_run.finish()
    print(f"Saved UG-Prune-Lite checkpoints to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
