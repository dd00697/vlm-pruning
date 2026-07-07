#!/usr/bin/env python
"""Select UG-Prune-Lite checkpoints using fixed allowed-dev generation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--top-k-selector-checkpoints", type=int, default=3)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="refcoco")
    parser.add_argument("--task", default="grounding", choices=["grounding"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--model-path", "--model", dest="model_path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--reserved-manifest", type=Path, default=Path("data/manifests/official_baselines/refcoco_val.jsonl"))
    parser.add_argument("--reserved-limit", type=int, default=500)
    parser.add_argument("--allow-reserved-final-eval", action="store_true")
    parser.add_argument("--wandb-project", default="vlm-pruning")
    parser.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_jsonl(path: Path, *, start_index: int = 0, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index < start_index:
                continue
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def source_row(row: dict[str, Any]) -> int | None:
    meta = row.get("metadata")
    if isinstance(meta, dict) and isinstance(meta.get("source_row"), int):
        return int(meta["source_row"])
    if isinstance(row.get("source_row"), int):
        return int(row["source_row"])
    return None


def manifest_checksum(rows: list[dict[str, Any]]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row.get("sample_id", "")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except FileNotFoundError:
        return False


def reserved_guard(args: argparse.Namespace) -> dict[str, Any]:
    eval_rows = read_jsonl(args.manifest, start_index=args.start_index, limit=args.limit)
    reserved_rows = read_jsonl(args.reserved_manifest, limit=args.reserved_limit) if args.reserved_manifest.exists() else []
    reserved_ids = {str(row.get("sample_id")) for row in reserved_rows}
    violations: list[dict[str, Any]] = []
    for offset, row in enumerate(eval_rows):
        row_source = source_row(row)
        sample_id = str(row.get("sample_id", ""))
        if sample_id in reserved_ids or (row_source is not None and 0 <= row_source < args.reserved_limit):
            violations.append(
                {
                    "eval_offset": offset,
                    "manifest_row": args.start_index + offset,
                    "sample_id": sample_id,
                    "source_row": row_source,
                }
            )
    if same_path(args.manifest, args.reserved_manifest):
        end_index = args.start_index + len(eval_rows) - 1
        if eval_rows and args.start_index <= args.reserved_limit - 1 and end_index >= 0:
            violations.append(
                {
                    "manifest_row_range": [args.start_index, end_index],
                    "reason": "evaluation range overlaps reserved manifest rows",
                }
            )
    guard = {
        "allow_reserved_final_eval": bool(args.allow_reserved_final_eval),
        "manifest": str(args.manifest),
        "start_index": args.start_index,
        "limit": args.limit,
        "evaluated_rows": len(eval_rows),
        "evaluated_sample_checksum": manifest_checksum(eval_rows),
        "reserved_manifest": str(args.reserved_manifest),
        "reserved_limit": args.reserved_limit,
        "reserved_rows_loaded": len(reserved_rows),
        "reserved_violation_count": len(violations),
        "reserved_violations": violations[:10],
    }
    if violations and not args.allow_reserved_final_eval:
        raise SystemExit(
            "Refusing generation-aware checkpoint selection on reserved final rows. "
            f"First violations: {json.dumps(violations[:3], sort_keys=True)}"
        )
    return guard


def validation_rows(train_dir: Path) -> list[dict[str, Any]]:
    path = train_dir / "validation_log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metric(row: dict[str, Any] | None, key: str) -> float | None:
    if not row:
        return None
    value = row.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "value"


def checkpoint_epoch(path: Path, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> int | None:
    match = re.match(r"checkpoint_epoch_(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1))
    if path.name in {"checkpoint_best.pt", "checkpoint_best_selector.pt", "checkpoint.pt"}:
        value = metrics.get("best/epoch")
        return int(value) if isinstance(value, int) else None
    if path.name == "checkpoint_last.pt" and rows:
        value = rows[-1].get("epoch")
        return int(value) if isinstance(value, int) else None
    return None


def candidate_record(path: Path, reason: str, metrics: dict[str, Any], rows_by_epoch: dict[int, dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    epoch = checkpoint_epoch(path, metrics, rows)
    val_row = rows_by_epoch.get(epoch) if epoch is not None else None
    return {
        "label": path.stem,
        "checkpoint": str(path),
        "reason": reason,
        "epoch": epoch,
        "selector_teacher_recall_at_58": metric(val_row, "val/selection/teacher_recall_at_58"),
        "selector_gt_box_patch_coverage_at_58": metric(val_row, "val/selection/gt_box_patch_coverage_at_58"),
        "selector_val_loss": metric(val_row, "val/loss/total"),
    }


def discover_candidates(train_dir: Path, top_k: int) -> list[dict[str, Any]]:
    metrics = read_json(train_dir / "metrics.json")
    rows = validation_rows(train_dir)
    rows_by_epoch = {int(row["epoch"]): row for row in rows if isinstance(row.get("epoch"), int)}
    by_path: dict[Path, dict[str, Any]] = {}

    def add(path: Path, reason: str) -> None:
        if path.exists() and path not in by_path:
            by_path[path] = candidate_record(path, reason, metrics, rows_by_epoch, rows)

    for name in ("checkpoint_best_selector.pt", "checkpoint_best.pt", "checkpoint_last.pt"):
        add(train_dir / name, name.replace(".pt", ""))
    if not (train_dir / "checkpoint_best.pt").exists():
        add(train_dir / "checkpoint.pt", "checkpoint_compat")

    epoch_paths = sorted(train_dir.glob("checkpoint_epoch_*.pt"))
    scored: list[tuple[float, Path]] = []
    for path in epoch_paths:
        epoch = checkpoint_epoch(path, metrics, rows)
        row = rows_by_epoch.get(epoch) if epoch is not None else None
        scored.append((metric(row, "val/selection/teacher_recall_at_58") or -float("inf"), path))
    for _score, path in sorted(scored, key=lambda item: item[0], reverse=True)[: max(0, top_k)]:
        add(path, "top_selector_teacher_recall")

    if rows:
        best_teacher = max(rows, key=lambda row: metric(row, "val/selection/teacher_recall_at_58") or -float("inf"))
        best_gt = max(rows, key=lambda row: metric(row, "val/selection/gt_box_patch_coverage_at_58") or -float("inf"))
        best_loss = min(rows, key=lambda row: metric(row, "val/loss/total") if metric(row, "val/loss/total") is not None else float("inf"))
        for row, reason in [(best_teacher, "best_teacher_recall"), (best_gt, "best_gt_coverage"), (best_loss, "lowest_val_loss")]:
            epoch = int(row["epoch"])
            add(train_dir / f"checkpoint_epoch_{epoch:03d}.pt", reason)

    candidates = dedupe_candidates_by_epoch(list(by_path.values()))
    candidates.sort(
        key=lambda item: (
            item["selector_teacher_recall_at_58"] if item["selector_teacher_recall_at_58"] is not None else -float("inf"),
            item["selector_gt_box_patch_coverage_at_58"] if item["selector_gt_box_patch_coverage_at_58"] is not None else -float("inf"),
            -(item["selector_val_loss"] if item["selector_val_loss"] is not None else float("inf")),
            item["label"],
        ),
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"No checkpoints found in {train_dir}")
    return candidates


def candidate_preference(candidate: dict[str, Any]) -> int:
    label = str(candidate.get("label", ""))
    if label == "checkpoint_best_selector":
        return 0
    if label == "checkpoint_best":
        return 1
    if label == "checkpoint_last":
        return 2
    if label.startswith("checkpoint_epoch_"):
        return 3
    return 4


def dedupe_candidates_by_epoch(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int | str], dict[str, Any]] = {}
    for candidate in sorted(candidates, key=candidate_preference):
        epoch = candidate.get("epoch")
        key = ("epoch", int(epoch)) if isinstance(epoch, int) else ("path", str(candidate.get("checkpoint")))
        if key not in by_key:
            by_key[key] = candidate
    return list(by_key.values())


def generation_command(args: argparse.Namespace, checkpoint: Path, run_name: str, output_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_ug_prune_lite.py",
        "--dataset",
        args.dataset,
        "--task",
        args.task,
        "--manifest",
        str(args.manifest),
        "--checkpoint",
        str(checkpoint),
        "--output-root",
        str(output_root),
        "--run-name",
        run_name,
        "--start-index",
        str(args.start_index),
        "--limit",
        str(args.limit),
        "--precision",
        args.precision,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.model_path:
        cmd.extend(["--model-path", args.model_path])
    if args.wandb_mode != "disabled":
        cmd.extend(["--wandb", "--wandb-project", args.wandb_project, "--wandb-mode", args.wandb_mode])
    return cmd


def load_generation_metrics(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "metrics.json")


def run_candidate(args: argparse.Namespace, candidate: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(candidate["checkpoint"])
    run_name = safe_slug(f"{args.train_dir.name}_{candidate['label']}_dev{args.limit}_gen_select")
    output_root = args.output_dir / "runs"
    run_dir = output_root / run_name
    cmd = generation_command(args, checkpoint, run_name, output_root)
    result = {**candidate, "run_name": run_name, "run_dir": str(run_dir), "command": cmd}
    if args.dry_run:
        result.update({"status": "dry_run", "generation_metrics": {}})
        return result
    metrics_path = run_dir / "metrics.json"
    if args.reuse_existing and not args.force and metrics_path.exists():
        metrics = load_generation_metrics(run_dir)
        result.update({"status": "reused", "generation_metrics": metrics})
        return result
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    result.update({"status": "completed", "generation_metrics": load_generation_metrics(run_dir)})
    return result


def generation_value(result: dict[str, Any], key: str) -> float | None:
    value = (result.get("generation_metrics") or {}).get(key)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def sanity_pass(result: dict[str, Any]) -> bool:
    metrics = result.get("generation_metrics") or {}
    if not metrics:
        return bool(result.get("status") == "dry_run")
    exact = metrics.get("selection/exact_budget_rate")
    dup = metrics.get("selection/duplicate_count_mean")
    out = metrics.get("selection/out_of_range_count_mean")
    parsed = metrics.get("quality/parsed_box_rate")
    valid = metrics.get("quality/valid_output_rate")
    return (
        (exact is None or float(exact) >= 1.0)
        and (dup is None or float(dup) == 0.0)
        and (out is None or float(out) == 0.0)
        and (parsed is None or float(parsed) >= 1.0)
        and (valid is None or float(valid) >= 1.0)
    )


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for result in results:
        metrics = result.get("generation_metrics") or {}
        result["rank_primary_acc_at_0_5"] = metrics.get("quality/acc_at_0_5")
        result["rank_secondary_mean_iou"] = metrics.get("quality/mean_iou")
        result["rank_tertiary_mean_latency_s"] = metrics.get("efficiency/mean_latency_s")
        result["sanity_pass"] = sanity_pass(result)
    return sorted(
        results,
        key=lambda item: (
            1 if item.get("sanity_pass") else 0,
            generation_value(item, "quality/acc_at_0_5") if generation_value(item, "quality/acc_at_0_5") is not None else -float("inf"),
            generation_value(item, "quality/mean_iou") if generation_value(item, "quality/mean_iou") is not None else -float("inf"),
            -(generation_value(item, "efficiency/mean_latency_s") if generation_value(item, "efficiency/mean_latency_s") is not None else float("inf")),
        ),
        reverse=True,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "label",
        "checkpoint",
        "reason",
        "epoch",
        "status",
        "sanity_pass",
        "selector_teacher_recall_at_58",
        "selector_gt_box_patch_coverage_at_58",
        "selector_val_loss",
        "quality/acc_at_0_5",
        "quality/mean_iou",
        "efficiency/mean_latency_s",
        "efficiency/pruning_selection_latency_s",
        "selection/exact_budget_rate",
        "selection/duplicate_count_mean",
        "selection/out_of_range_count_mean",
        "run_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            metrics = row.get("generation_metrics") or {}
            flat = {field: row.get(field) for field in fields}
            flat["rank"] = rank
            for field in fields:
                if field in metrics:
                    flat[field] = metrics[field]
            writer.writerow(flat)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    guard = reserved_guard(args)
    candidates = discover_candidates(args.train_dir, args.top_k_selector_checkpoints)
    results = [run_candidate(args, candidate) for candidate in candidates]
    ranked = rank_results(results)
    best = ranked[0]
    write_json(
        args.output_dir / "generation_model_selection.json",
        {
            "config": {
                "train_dir": str(args.train_dir),
                "manifest": str(args.manifest),
                "start_index": args.start_index,
                "limit": args.limit,
                "top_k_selector_checkpoints": args.top_k_selector_checkpoints,
                "device": args.device,
                "precision": args.precision,
                "dry_run": args.dry_run,
                "ranking": ["quality/acc_at_0_5", "quality/mean_iou", "efficiency/mean_latency_s"],
            },
            "reserved_guard": guard,
            "candidates": results,
            "ranked": ranked,
            "best": best,
        },
    )
    write_csv(args.output_dir / "candidate_generation_results.csv", ranked)
    (args.output_dir / "best_generation_checkpoint.txt").write_text(str(best["checkpoint"]) + "\n", encoding="utf-8")
    print(f"Best generation checkpoint: {best['checkpoint']}")
    print(f"Wrote generation-aware selection to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
