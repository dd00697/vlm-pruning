#!/usr/bin/env python
"""Evaluate UG-Prune-Lite selector quality without running LLaVA generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm_pruning.selectors.ug_prune_lite import UGPruneLiteSelector  # noqa: E402
from vlm_pruning.training.ug_prune_lite_dataset import UGPruneLiteCacheDataset  # noqa: E402
from vlm_pruning.training.ug_prune_lite_losses import selection_metrics_from_indices  # noqa: E402
from vlm_pruning.training.ug_prune_lite_model import UGPruneLiteModel, UGPruneLiteModelConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--collapse-bin-threshold", type=int, default=4)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "offline"), choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "vlm-pruning"))
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def checksum_indices(indices: list[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(str(int(index)).encode("ascii"))
        digest.update(b",")
    return digest.hexdigest()[:16]


def load_checkpoint_payload(torch: Any, checkpoint: Path, device: Any) -> dict[str, Any]:
    try:
        return torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint, map_location=device)


def load_student(torch: Any, checkpoint: Path, device: Any) -> Any:
    payload = load_checkpoint_payload(torch, checkpoint, device)
    config = UGPruneLiteModelConfig(**payload["model_config"])
    model = UGPruneLiteModel.build(torch, config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def checkpoint_feature_projection(torch: Any, checkpoint: Path, device: Any) -> dict[str, Any] | None:
    payload = load_checkpoint_payload(torch, checkpoint, device)
    return (payload.get("train_config") or {}).get("feature_projection")


def aggregate(rows: list[dict[str, Any]], collapse_threshold: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {"samples/evaluated": len(rows)}
    numeric_keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float))})
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            metrics[key] = sum(values) / float(len(values))
    metrics["selection/collapse_sample_fraction"] = (
        sum(1 for row in rows if float(row.get("selection/occupied_3x3_bins", 0.0)) < collapse_threshold) / float(len(rows))
        if rows
        else 0.0
    )
    metrics["selection/checksum_unique_count"] = len({row["selected_checksum"] for row in rows})
    return metrics


def main() -> int:
    args = parse_args()
    import torch

    device = torch.device("cpu" if args.device == "cuda" and not torch.cuda.is_available() else args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "method/name": "ug_prune_lite_selector_only",
        "teacher_cache": str(args.teacher_cache),
        "checkpoint": str(args.checkpoint),
        "limit": args.limit,
        "device": str(device),
        "collapse_bin_threshold": args.collapse_bin_threshold,
    }
    write_json(args.output_dir / "config.json", config)

    student = load_student(torch, args.checkpoint, device)
    feature_projection = checkpoint_feature_projection(torch, args.checkpoint, device)
    dataset = UGPruneLiteCacheDataset(
        torch,
        args.teacher_cache,
        limit=args.limit,
        device=device,
        feature_projection=feature_projection,
    )
    selector = UGPruneLiteSelector(torch)

    wandb_run = None
    if args.wandb and args.wandb_mode != "disabled":
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                mode=args.wandb_mode,
                name=args.output_dir.name,
                group="ug-prune-lite",
                job_type="selector-eval",
                config=config,
            )
        except Exception as exc:
            print(f"W&B init failed; continuing locally. {type(exc).__name__}: {exc}")

    rows: list[dict[str, Any]] = []
    with (args.output_dir / "per_sample_selector_results.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(len(dataset)):
            item = dataset[index]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                outputs = student(item["visual_features"].unsqueeze(0), item["query_embedding"].unsqueeze(0))
                selection = selector.select(
                    outputs["target_logits"][0],
                    outputs["context_logits"][0],
                    outputs["keep_logits"][0],
                    sample_id=item["sample_id"],
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - start
            metrics = selection_metrics_from_indices(
                selection.selected_indices,
                item["selected_indices"],
                item["role_labels"],
                item["box_labels"],
            )
            row = {
                "sample_id": item["sample_id"],
                "selector_latency_s": latency,
                "selected_checksum": checksum_indices(selection.selected_indices),
                "selected_indices": selection.selected_indices,
                **metrics,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            print(f"Evaluated selector {index + 1}/{len(dataset)}")

    metrics = aggregate(rows, args.collapse_bin_threshold)
    write_json(args.output_dir / "metrics.json", metrics)
    if wandb_run is not None:
        wandb_run.log(metrics)
        wandb_run.finish()
    print(f"Saved selector-only eval to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
