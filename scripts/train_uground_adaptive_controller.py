#!/usr/bin/env python
"""Train/evaluate a lightweight offline controller from UG-Prune oracle grids."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_baseline import BaselineError  # noqa: E402
from run_official_baseline import utc_now, write_json  # noqa: E402
from run_uground_adaptive_grid import evaluate_rule, result_metric  # noqa: E402
from vlm_pruning.controllers import FeatureSpec, extract_feature_dict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-grid-dir", required=True, type=Path)
    parser.add_argument("--dev-grid-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/controllers/uground_adaptive"))
    parser.add_argument("--feature-action", default="k10_b58_default_soft_average_query_max")
    parser.add_argument("--fallback-feature-action", default="k12_b58_default_soft_average_query_max")
    parser.add_argument("--min-oracle-acc-headroom", type=float, default=0.02)
    parser.add_argument("--min-oracle-miou-headroom", type=float, default=0.005)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def feature_rows(grid_dir: Path, feature_action: str, fallback_action: str) -> tuple[list[str], list[list[float]], list[str], dict[tuple[str, str], dict[str, Any]]]:
    action_rows = load_jsonl(grid_dir / "per_sample_action_results.jsonl")
    oracle_rows = load_jsonl(grid_dir / "oracle_per_sample.jsonl")
    label_by_sample = {str(row["sample_id"]): str(row["best_action"]) for row in oracle_rows}
    by_action_sample = {(str(row["action"]), str(row["sample_id"])): row for row in action_rows}
    spec = FeatureSpec()
    sample_ids: list[str] = []
    features: list[list[float]] = []
    labels: list[str] = []
    for sample_id, label in sorted(label_by_sample.items()):
        row = by_action_sample.get((feature_action, sample_id)) or by_action_sample.get((fallback_action, sample_id))
        if row is None:
            continue
        feature_dict = extract_feature_dict(row, spec)
        sample_ids.append(sample_id)
        features.append([feature_dict[name] for name in spec.names])
        labels.append(label)
    return sample_ids, features, labels, by_action_sample


def standardise(features: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    if not features:
        return [], [], []
    width = len(features[0])
    means = [sum(row[i] for row in features) / len(features) for i in range(width)]
    stds: list[float] = []
    for i, mean in enumerate(means):
        var = sum((row[i] - mean) ** 2 for row in features) / len(features)
        stds.append(max(math.sqrt(var), 1e-6))
    scaled = [[(row[i] - means[i]) / stds[i] for i in range(width)] for row in features]
    return scaled, means, stds


def apply_standardise(features: list[list[float]], means: list[float], stds: list[float]) -> list[list[float]]:
    return [[(row[i] - means[i]) / stds[i] for i in range(len(means))] for row in features]


def train_centroids(features: list[list[float]], labels: list[str]) -> dict[str, list[float]]:
    grouped: dict[str, list[list[float]]] = {}
    for feature, label in zip(features, labels):
        grouped.setdefault(label, []).append(feature)
    centroids: dict[str, list[float]] = {}
    for label, rows in grouped.items():
        centroids[label] = [sum(row[i] for row in rows) / len(rows) for i in range(len(rows[0]))]
    return centroids


def predict_one(feature: list[float], centroids: dict[str, list[float]]) -> tuple[str, float, dict[str, float]]:
    distances: dict[str, float] = {}
    for label, centroid in centroids.items():
        distances[label] = math.sqrt(sum((feature[i] - centroid[i]) ** 2 for i in range(len(feature))))
    best = min(distances, key=lambda label: (distances[label], label))
    sorted_distances = sorted(distances.values())
    margin = sorted_distances[1] - sorted_distances[0] if len(sorted_distances) > 1 else 0.0
    confidence = 1.0 / (1.0 + math.exp(-margin))
    scores = {label: -distance for label, distance in distances.items()}
    return best, confidence, scores


def evaluate(
    grid_dir: Path,
    *,
    feature_action: str,
    fallback_feature_action: str,
    means: list[float],
    stds: list[float],
    centroids: dict[str, list[float]],
) -> dict[str, Any]:
    sample_ids, features, labels, by_action_sample = feature_rows(grid_dir, feature_action, fallback_feature_action)
    scaled = apply_standardise(features, means, stds)
    predictions: list[dict[str, Any]] = []
    chosen: dict[str, dict[str, Any]] = {}
    for sample_id, feature, label in zip(sample_ids, scaled, labels):
        pred, confidence, scores = predict_one(feature, centroids)
        row = by_action_sample.get((pred, sample_id))
        if row is not None:
            chosen[sample_id] = row
        predictions.append(
            {
                "sample_id": sample_id,
                "target_action": label,
                "predicted_action": pred,
                "correct": pred == label,
                "confidence": confidence,
                "scores": scores,
            }
        )
    fixed_k12 = {
        sid: row
        for (action, sid), row in by_action_sample.items()
        if action == "k12_b58_default_soft_average_query_max" and sid in chosen
    }
    oracle = {
        sid: by_action_sample[(label, sid)]
        for sid, label in zip(sample_ids, labels)
        if (label, sid) in by_action_sample and sid in chosen
    }
    return {
        "grid_dir": str(grid_dir),
        "samples/evaluated": len(predictions),
        "label_accuracy": sum(1 for row in predictions if row["correct"]) / len(predictions) if predictions else None,
        "prediction_action_counts": _counts(row["predicted_action"] for row in predictions),
        "target_action_counts": _counts(labels),
        "policy_summary": evaluate_rule(chosen),
        "fixed_k12_summary": evaluate_rule(fixed_k12) if len(fixed_k12) == len(chosen) and chosen else None,
        "oracle_summary": evaluate_rule(oracle) if len(oracle) == len(chosen) and chosen else None,
        "predictions": predictions,
    }


def _counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out


def passes_oracle_gate(grid_dir: Path, args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    summary = load_json(grid_dir / "oracle_summary.json")
    headroom = summary.get("headroom_vs_k12_b58") or {}
    acc_delta = result_metric(headroom, "acc_at_0_5_delta")
    miou_delta = result_metric(headroom, "mean_iou_delta")
    passed = acc_delta >= args.min_oracle_acc_headroom or miou_delta >= args.min_oracle_miou_headroom
    return passed, {"acc_delta": acc_delta, "mean_iou_delta": miou_delta, "oracle_summary": summary}


def run(args: argparse.Namespace) -> int:
    gate_passed, gate = passes_oracle_gate(args.train_grid_dir, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not gate_passed and not args.force:
        write_json(
            args.output_dir / "controller_skipped.json",
            {
                "created_at_utc": utc_now(),
                "status": "skipped",
                "reason": "oracle_headroom_gate_failed",
                "gate": gate,
                "train_grid_dir": str(args.train_grid_dir),
            },
        )
        print("Skipped controller training: oracle headroom gate failed. Use --force for diagnostic training.")
        return 0

    sample_ids, features, labels, _ = feature_rows(args.train_grid_dir, args.feature_action, args.fallback_feature_action)
    if not features:
        raise BaselineError("No training features found. Check --feature-action and grid outputs.")
    scaled, means, stds = standardise(features)
    centroids = train_centroids(scaled, labels)
    model = {
        "created_at_utc": utc_now(),
        "controller_type": "standardized_nearest_centroid",
        "feature_action": args.feature_action,
        "fallback_feature_action": args.fallback_feature_action,
        "feature_names": list(FeatureSpec().names),
        "means": means,
        "stds": stds,
        "centroids": centroids,
        "train_samples": len(sample_ids),
        "train_action_counts": _counts(labels),
        "oracle_gate": gate,
    }
    write_json(args.output_dir / "controller.json", model)
    train_eval = evaluate(
        args.train_grid_dir,
        feature_action=args.feature_action,
        fallback_feature_action=args.fallback_feature_action,
        means=means,
        stds=stds,
        centroids=centroids,
    )
    write_json(args.output_dir / "train_eval.json", train_eval)
    if args.dev_grid_dir:
        dev_eval = evaluate(
            args.dev_grid_dir,
            feature_action=args.feature_action,
            fallback_feature_action=args.fallback_feature_action,
            means=means,
            stds=stds,
            centroids=centroids,
        )
        write_json(args.output_dir / "dev_eval.json", dev_eval)
    print(f"Saved UGround-style controller to {args.output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (BaselineError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
