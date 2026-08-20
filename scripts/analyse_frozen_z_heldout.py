#!/usr/bin/env python3
"""Release the frozen outcome embargo, score both splits, and apply the preregistered rule."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for value in (SCRIPT_DIR, REPO_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from frozen_z_heldout_common import (
    BOOTSTRAP,
    CAMPAIGN,
    MANIFESTS,
    THRESHOLDS,
    HeldoutError,
    atomic_json,
    canonical_bytes,
    now,
    read_json,
    read_jsonl,
    sha256,
    verify_manifest,
)
from run_baseline import box_iou
from validate_progressive_oracle_run import parse_box, validate_run


REPO = Path(__file__).resolve().parents[1]


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False, mode="w", encoding="utf-8") as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def score_row(row: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    scored = dict(row)
    predicted, raw, reason = parse_box(
        str(row.get("raw_model_output") or ""), int(manifest["width"]), int(manifest["height"])
    )
    iou = box_iou(predicted, manifest["gt_box"]) if predicted is not None else 0.0
    scored.update({
        "raw_parsed_box": raw, "predicted_box": predicted, "parse_failure_reason": reason,
        "valid": predicted is not None, "invalid": predicted is None,
        "box_iou": iou, "score": iou, "success_iou_0_5": bool(iou >= 0.5),
        "outcome_embargoed": False,
    })
    return scored


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "samples": count,
        "parsed_box_rate": sum(bool(row["valid"]) for row in rows) / count,
        "acc_at_0_5": sum(bool(row["success_iou_0_5"]) for row in rows) / count,
        "mean_iou": sum(float(row["box_iou"]) for row in rows) / count,
        "average_visual_tokens": sum(float(row["candidate_average_visual_tokens"]) for row in rows) / count,
        "token_layer_work": sum(sum(int(value) for value in row["realized_visual_tokens_by_layer"]) for row in rows) / count,
    }


def paired_bootstrap(z_rows: list[dict[str, Any]], full_rows: list[dict[str, Any]], split_index: int) -> dict[str, Any]:
    if [row["sample_id"] for row in z_rows] != [row["sample_id"] for row in full_rows]:
        raise HeldoutError("Paired inputs are not in identical manifest order")
    z_acc = np.asarray([bool(row["success_iou_0_5"]) for row in z_rows], dtype=np.float64)
    f_acc = np.asarray([bool(row["success_iou_0_5"]) for row in full_rows], dtype=np.float64)
    z_iou = np.asarray([float(row["box_iou"]) for row in z_rows], dtype=np.float64)
    f_iou = np.asarray([float(row["box_iou"]) for row in full_rows], dtype=np.float64)
    rng = np.random.default_rng(int(BOOTSTRAP["seed"]) + split_index)
    samples: dict[str, list[np.ndarray]] = {"acc_at_0_5": [], "mean_iou": []}
    remaining = int(BOOTSTRAP["resamples"])
    while remaining:
        batch = min(200, remaining)
        indices = rng.integers(0, len(z_rows), size=(batch, len(z_rows)))
        samples["acc_at_0_5"].append((z_acc[indices] - f_acc[indices]).mean(axis=1))
        samples["mean_iou"].append((z_iou[indices] - f_iou[indices]).mean(axis=1))
        remaining -= batch
    output: dict[str, Any] = {}
    for name, values in samples.items():
        distribution = np.concatenate(values)
        point = float((z_acc - f_acc).mean()) if name == "acc_at_0_5" else float((z_iou - f_iou).mean())
        output[name] = {
            "direction": "Z_M448_0_minus_Full", "point_delta": point,
            "lower_95": float(np.quantile(distribution, 0.025)),
            "upper_95": float(np.quantile(distribution, 0.975)),
            "resamples": int(BOOTSTRAP["resamples"]), "seed": int(BOOTSTRAP["seed"]) + split_index,
        }
    return output


def finalize_z(split: str, manifest: list[dict[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    validate_run(run_dir, Path(MANIFESTS[split]["path"]), expected_rows=len(manifest))
    seal = read_json(run_dir / "SEALED_RAW.json")
    if seal.get("quality_metrics_present") is not False:
        raise HeldoutError(f"{split} raw seal violated the outcome embargo")
    raw = read_jsonl(run_dir / "predictions.raw.jsonl")
    scored = [score_row(row, source) for row, source in zip(raw, manifest, strict=True)]
    scored_path = run_dir / "predictions.scored.jsonl"
    atomic_jsonl(scored_path, scored)
    cell_metrics = metrics(scored)
    atomic_json(run_dir / "metrics.json", cell_metrics)
    integrity = {
        "schema_version": 1, "status": "pass", "validated_at_utc": now(), "rows": len(scored),
        "manifest_sha256": MANIFESTS[split]["sha256"], "manifest_order_unique": True,
        "raw_predictions_sha256": sha256(run_dir / "predictions.raw.jsonl"),
        "scored_predictions_sha256": sha256(scored_path), "raw_outputs_independently_parsed_and_scored": True,
        "mechanical_integrity_sha256": sha256(run_dir / "integrity_raw.json"),
    }
    atomic_json(run_dir / "integrity_report.json", integrity)
    completion = {
        "status": "complete", "completed_at_utc": now(), "rows": len(scored),
        "manifest_sha256": MANIFESTS[split]["sha256"], "config_sha256": sha256(run_dir / "config.json"),
        "environment_sha256": sha256(run_dir / "environment.json"),
        "environment_epoch_sha256": {
            path.name: sha256(path) for path in sorted(run_dir.glob("environment*.json"))
        },
        "raw_predictions_sha256": sha256(run_dir / "predictions.raw.jsonl"),
        "scored_predictions_sha256": sha256(scored_path), "metrics_sha256": sha256(run_dir / "metrics.json"),
        "integrity_sha256": sha256(run_dir / "integrity_report.json"),
        "raw_seal_sha256": sha256(run_dir / "SEALED_RAW.json"),
    }
    atomic_json(run_dir / "COMPLETED.json", completion)
    return scored


def full_rows(split: str, registry: dict[str, Any], manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cell = registry["cells"][f"Full_{split}"]
    if cell["status"] != "complete_rerun":
        raise HeldoutError(f"Strict campaign expected a rerun Full control for {split}")
    run_dir = REPO / cell["run_dir"]
    validate_run(run_dir, Path(MANIFESTS[split]["path"]), expected_rows=len(manifest))
    return read_jsonl(run_dir / "predictions.jsonl")


def main() -> int:
    release_path = CAMPAIGN / "OUTCOME_EMBARGO_RELEASE.json"
    if not release_path.is_file() or read_json(release_path).get("status") != "outcome_embargo_released":
        raise HeldoutError("Both candidate raw artifacts are not sealed; analysis remains prohibited")
    registry = read_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json")
    if registry["phase_status"].get("quality") != "raw_sealed_both_splits":
        raise HeldoutError("Registry has not released both candidate splits")
    all_metrics: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    gate_results: dict[str, Any] = {}
    for split_index, split in enumerate(("testA", "testB")):
        manifest = verify_manifest(split)
        z_dir = REPO / registry["cells"][f"Z_M448_0_{split}"]["run_dir"]
        z = finalize_z(split, manifest, z_dir)
        full = full_rows(split, registry, manifest)
        all_metrics[split] = {"Full": metrics(full), "Z_M448_0": metrics(z)}
        contrasts[split] = paired_bootstrap(z, full, split_index)
        z_metrics, f_metrics = all_metrics[split]["Z_M448_0"], all_metrics[split]["Full"]
        gates = {
            "parsed_box_rate": z_metrics["parsed_box_rate"] >= THRESHOLDS["minimum_valid_parsed_box_rate"],
            "accuracy_loss": f_metrics["acc_at_0_5"] - z_metrics["acc_at_0_5"] <= THRESHOLDS["maximum_accuracy_loss_absolute"],
            "miou_loss": f_metrics["mean_iou"] - z_metrics["mean_iou"] <= THRESHOLDS["maximum_miou_loss_absolute"],
        }
        gate_results[split] = {"passed": all(gates.values()), "gates": gates}
    passes = [gate_results[split]["passed"] for split in ("testA", "testB")]
    if all(passes):
        status = "HELDOUT_CONFIRMED_BOTH_SPLITS"
    elif any(passes):
        status = "HELDOUT_MIXED_SPLIT_RESULT"
    else:
        status = "HELDOUT_NOT_CONFIRMED"
    analysis = {
        "schema_version": 1, "status": status, "analyzed_at_utc": now(), "metrics": all_metrics,
        "paired_contrasts": contrasts, "gate_results": gate_results, "thresholds": THRESHOLDS,
        "bootstrap": BOOTSTRAP, "splits_pooled": False,
    }
    analysis_root = CAMPAIGN / "analysis"
    atomic_json(analysis_root / "HELDOUT_RESULTS.json", analysis)
    atomic_json(analysis_root / "PAIRED_CONTRASTS.json", {"schema_version": 1, "contrasts": contrasts, "bootstrap": BOOTSTRAP})
    atomic_json(analysis_root / "HELDOUT_DECISION.json", {"schema_version": 1, "status": status, "gate_results": gate_results, "thresholds": THRESHOLDS})
    csv_path = analysis_root / "HELDOUT_RESULTS.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "method", "samples", "parsed_box_rate", "acc_at_0_5", "mean_iou", "average_visual_tokens", "token_layer_work"])
        for split in ("testA", "testB"):
            for method in ("Full", "Z_M448_0"):
                row = all_metrics[split][method]
                writer.writerow([split, method, row["samples"], row["parsed_box_rate"], row["acc_at_0_5"], row["mean_iou"], row["average_visual_tokens"], row["token_layer_work"]])
    registry["phase_status"]["analysis"] = "pass"
    registry["phase_status"]["quality"] = "complete"
    registry["heldout_status"] = status
    for split in ("testA", "testB"):
        registry["cells"][f"Z_M448_0_{split}"]["status"] = "complete"
    atomic_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json", registry)
    report = [
        "# Frozen Z_M448_0 held-out confirmation", "",
        f"Outcome embargo released only after both raw candidate splits were sealed. Final status: `{status}`.", "",
        "| Split | Method | Parsed | Acc@0.5 | mIoU | Avg visual tokens | Work |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("testA", "testB"):
        for method in ("Full", "Z_M448_0"):
            row = all_metrics[split][method]
            report.append(f"| {split} | {method} | {row['parsed_box_rate']:.4f} | {row['acc_at_0_5']:.4f} | {row['mean_iou']:.4f} | {row['average_visual_tokens']:.0f} | {row['token_layer_work']:.0f} |")
    report.extend(["", "The split-wise preregistered gates were applied without pooling or candidate reselection. Held-out quality does not constitute a new timing claim.", ""])
    (CAMPAIGN / "RESEARCH_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    state = read_json(CAMPAIGN / "CONTROLLER_STATE.json")
    state.update({"phase": "dissertation", "updated_at_utc": now(), "next_action": "build held-out dissertation assets and QA", "heldout_status": status})
    atomic_json(CAMPAIGN / "CONTROLLER_STATE.json", state)
    print(json.dumps({"status": status, "gate_results": gate_results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HeldoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
