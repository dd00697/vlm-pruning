#!/usr/bin/env python3
"""Independently validate sealed Full/Z rows and run the one frozen post-hoc analysis."""

from __future__ import annotations

import csv
import io
import json
import math
import sys
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
for value in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.frozen_z_error_analysis_common import (  # noqa: E402
    CAMPAIGN,
    INPUT_CAMPAIGN,
    QUALITY_LEDGER_SHA256,
    REPO,
    ErrorAnalysisFailure,
    atomic_json,
    atomic_text,
    now,
    read_json,
    record_file,
    require,
    sha256,
)
from scripts.run_baseline import box_iou  # noqa: E402
from scripts.validate_progressive_oracle_run import parse_box, validate_run  # noqa: E402


ANALYSIS = CAMPAIGN / "analysis"
FIGURES = ANALYSIS / "figures"
CELL_PATHS = {
    "Full_testA": "quality/full_testa_attempt1",
    "Z_M448_0_testA": "quality/z_m448_0_testa_attempt1",
    "Full_testB": "quality/full_testb_attempt1",
    "Z_M448_0_testB": "quality/z_m448_0_testb_attempt1",
}
ALLOWED_ANALYSIS_FILES = {
    "INPUT_AUDIT.json",
    "ERROR_ANALYSIS.json",
    "ERROR_ANALYSIS_TRANSITIONS.csv",
    "ERROR_ANALYSIS_THRESHOLD_CURVE.csv",
    "ERROR_ANALYSIS_SUMMARIES.csv",
    "ANALYSIS_INVENTORY.json",
    "figures/paired_localization_threshold_crossing.pdf",
}
MANIFESTS = {
    "testA": (REPO / "data/manifests/final_evaluation/refcoco_testA_unique.jsonl", 5657,
              "01332dfc67c59c9077f14325dc11cb301b9a01a55eccdcc815f92f3fe587fe23"),
    "testB": (REPO / "data/manifests/final_evaluation/refcoco_testB_unique.jsonl", 5095,
              "0f0fd8ff86faf8ac48446c27e71c350b14aed2a8d86d6a59b4f7599def2d40c9"),
}
EXPECTED_TRACE = {
    "Full": [576] * 32,
    "Z_M448_0": [576] * 4 + [448] * 8 + [48] * 8 + [0] * 12,
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            require(line.endswith("\n"), f"JSONL row lacks final newline: {path}:{index}")
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL row is not an object: {path}:{index}")
            yield value


def close(actual: Any, expected: float, label: str, tolerance: float = 1e-12) -> None:
    require(isinstance(actual, (int, float)) and math.isfinite(float(actual)), f"Non-numeric {label}")
    require(abs(float(actual) - expected) <= tolerance, f"Numeric mismatch {label}: {actual} != {expected}")


def verify_ledger_file(path: Path, expected: dict[str, Any], label: str) -> None:
    require(path.is_file(), f"Ledger file missing: {label}")
    require(set(expected) == {"bytes", "sha256"}, f"Ledger schema differs: {label}")
    require(path.stat().st_size == int(expected["bytes"]), f"Ledger bytes differ: {label}")
    require(sha256(path) == expected["sha256"], f"Ledger hash differs: {label}")


def require_exact_pairing(full_ids: list[str], z_ids: list[str], expected_ids: list[str], split: str) -> None:
    """Fail closed unless both methods have the exact frozen manifest identity and order."""
    require(full_ids == expected_ids, f"Full pairing differs: {split}")
    require(z_ids == expected_ids, f"Z pairing differs: {split}")
    require(len(expected_ids) == len(set(expected_ids)), f"Manifest IDs are not unique: {split}")


def require_expected_transitions(actual: dict[str, int], expected: dict[str, int], split: str) -> None:
    require(actual == expected, f"Frozen transition counts differ: {split}/{actual}")


def require_allowed_output_inventory(paths: Iterable[str]) -> None:
    observed = set(paths)
    require(observed == ALLOWED_ANALYSIS_FILES, f"Analysis output inventory differs: {sorted(observed)}")
    forbidden = ("subgroup", "qualitative", "example", "category", "object_size", "expression_length")
    require(not any(token in path.lower() for path in observed for token in forbidden),
            "Forbidden exploratory analysis output is present")


def verify_v2_lock() -> dict[str, Any]:
    lock = read_json(CAMPAIGN / "PRE_ANALYSIS_LOCK_V3.json")
    require(lock.get("status") == "locked" and lock.get("candidate_analysis_outputs_present") is False,
            "V2 pre-analysis lock differs")
    require((CAMPAIGN / "LOCK_SUPERSESSION.json").is_file(), "Lock supersession record is missing")
    for name, expected in lock.get("locks", {}).items():
        verify_ledger_file(CAMPAIGN / name, expected, f"pre-analysis-lock/{name}")
    source_lock = read_json(CAMPAIGN / "SOURCE_LOCK_V3.json")
    for name, expected in source_lock.get("sources", {}).items():
        verify_ledger_file(REPO / name, expected, f"source-lock/{name}")
    return lock


def validate_cell(cell_id: str, ledger_cell: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    method = "Z_M448_0" if cell_id.startswith("Z_M448_0") else "Full"
    split = "testA" if cell_id.endswith("testA") else "testB"
    relative = CELL_PATHS[cell_id]
    run_dir = INPUT_CAMPAIGN / relative
    require(ledger_cell.get("path") == str(run_dir.relative_to(REPO)), f"Ledger path differs: {cell_id}")
    files = ledger_cell.get("files")
    require(isinstance(files, dict), f"Ledger files missing: {cell_id}")
    for name, expected in files.items():
        verify_ledger_file(run_dir / name, expected, f"{cell_id}/{name}")
    manifest_path, expected_rows, manifest_sha = MANIFESTS[split]
    require(sha256(manifest_path) == manifest_sha, f"Manifest hash differs: {split}")
    validation = validate_run(run_dir, manifest_path, expected_rows=expected_rows)
    require(validation.get("status") == "pass", f"Independent runner validation failed: {cell_id}")
    prediction_path = run_dir / ("predictions.scored.jsonl" if method == "Z_M448_0" else "predictions.jsonl")
    trace = EXPECTED_TRACE[method]
    work = sum(trace)
    ids: list[str] = []
    ious: list[float] = []
    for index, triplet in enumerate(zip_longest(iter_jsonl(manifest_path), iter_jsonl(prediction_path))):
        source, row = triplet
        require(source is not None and row is not None, f"Row count differs: {cell_id}/{index}")
        require(row.get("sample_id") == source.get("sample_id"), f"ID differs: {cell_id}/{index}")
        require(row.get("manifest_index") == index, f"Manifest index differs: {cell_id}/{index}")
        for key in ("dataset", "split", "query", "image_id", "image_path", "width", "height", "gt_box"):
            require(row.get(key) == source.get(key), f"Manifest metadata differs: {cell_id}/{index}/{key}")
        require(row.get("method") == method, f"Method differs: {cell_id}/{index}")
        require(row.get("realized_visual_tokens_by_layer") == trace, f"Trace differs: {cell_id}/{index}")
        require(sum(int(value) for value in row["realized_visual_tokens_by_layer"]) == work,
                f"Work differs: {cell_id}/{index}")
        predicted, raw_box, reason = parse_box(str(row.get("raw_model_output") or ""), int(source["width"]), int(source["height"]))
        recomputed_iou = box_iou(predicted, source["gt_box"]) if predicted is not None else 0.0
        require(row.get("parse_failure_reason") == reason, f"Parse reason differs: {cell_id}/{index}")
        require(bool(row.get("valid")) is (predicted is not None), f"Validity differs: {cell_id}/{index}")
        require(bool(row.get("invalid")) is (predicted is None), f"Invalidity differs: {cell_id}/{index}")
        require(bool(row.get("success_iou_0_5")) is bool(recomputed_iou >= 0.5),
                f"Correctness differs: {cell_id}/{index}")
        close(row.get("box_iou"), recomputed_iou, f"IoU {cell_id}/{index}")
        if predicted is not None:
            require(row.get("predicted_box") == predicted, f"Predicted box differs: {cell_id}/{index}")
        if raw_box is not None:
            require(row.get("raw_parsed_box") == raw_box, f"Raw box differs: {cell_id}/{index}")
        ids.append(str(row["sample_id"]))
        ious.append(recomputed_iou)
    require(len(ids) == expected_rows and len(set(ids)) == expected_rows, f"Ordered unique rows differ: {cell_id}")
    return np.asarray(ious, dtype=np.float64), ids, {
        "status": "pass", "rows": expected_rows, "run_dir": str(run_dir.relative_to(REPO)),
        "prediction_file": prediction_path.name, "prediction_sha256": sha256(prediction_path),
        "manifest_sha256": manifest_sha, "trace": trace, "token_layer_work": work,
    }


def transition_counts(full: np.ndarray, z: np.ndarray, threshold: float = 0.5) -> dict[str, int]:
    require(full.shape == z.shape and full.ndim == 1, "Paired IoU arrays differ")
    fc, zc = full >= threshold, z >= threshold
    return {
        "both_correct": int(np.sum(fc & zc)),
        "full_only": int(np.sum(fc & ~zc)),
        "z_only": int(np.sum(~fc & zc)),
        "both_incorrect": int(np.sum(~fc & ~zc)),
    }


def paired_bootstrap(full: np.ndarray, z: np.ndarray, thresholds: list[float], seed: int,
                     resamples: int = 10000, batch: int = 125) -> tuple[dict[float, dict[str, float]], dict[str, dict[str, float]]]:
    require(full.shape == z.shape and full.ndim == 1 and len(full) > 0, "Bootstrap input differs")
    rng = np.random.default_rng(seed)
    differences = {threshold: (full >= threshold).astype(np.int8) - (z >= threshold).astype(np.int8)
                   for threshold in thresholds}
    states = {
        "both_correct": ((full >= 0.5) & (z >= 0.5)).astype(np.int8),
        "full_only": ((full >= 0.5) & (z < 0.5)).astype(np.int8),
        "z_only": ((full < 0.5) & (z >= 0.5)).astype(np.int8),
        "both_incorrect": ((full < 0.5) & (z < 0.5)).astype(np.int8),
    }
    sampled_differences = {threshold: np.empty(resamples, dtype=np.float64) for threshold in thresholds}
    sampled_states = {state: np.empty(resamples, dtype=np.float64) for state in states}
    cursor = 0
    while cursor < resamples:
        size = min(batch, resamples - cursor)
        indices = rng.integers(0, len(full), size=(size, len(full)), endpoint=False)
        for threshold, values in differences.items():
            sampled_differences[threshold][cursor:cursor + size] = values[indices].mean(axis=1)
        for state, values in states.items():
            sampled_states[state][cursor:cursor + size] = values[indices].mean(axis=1)
        cursor += size
    curve = {}
    for threshold, samples in sampled_differences.items():
        point = float(differences[threshold].mean())
        low, high = np.quantile(samples, [0.025, 0.975], method="linear")
        curve[threshold] = {"point": point, "lower_95": float(low), "upper_95": float(high)}
    transitions = {}
    for state, samples in sampled_states.items():
        point = float(states[state].mean())
        low, high = np.quantile(samples, [0.025, 0.975], method="linear")
        transitions[state] = {"rate": point, "lower_95": float(low), "upper_95": float(high)}
    return curve, transitions


def iou_summaries(full: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    delta = z - full
    states = {
        "both_correct": (full >= 0.5) & (z >= 0.5),
        "full_only": (full >= 0.5) & (z < 0.5),
        "z_only": (full < 0.5) & (z >= 0.5),
        "both_incorrect": (full < 0.5) & (z < 0.5),
    }
    return {
        "direction": "Z_M448_0_minus_Full",
        "mean": float(np.mean(delta)), "median": float(np.median(delta)),
        "q1": float(np.quantile(delta, 0.25, method="linear")),
        "q3": float(np.quantile(delta, 0.75, method="linear")),
        "abs_delta_le_0_05": float(np.mean(np.abs(delta) <= 0.05 + 1e-12)),
        "delta_lt_minus_0_10": float(np.mean(delta < -0.10 - 1e-12)),
        "delta_lt_minus_0_25": float(np.mean(delta < -0.25 - 1e-12)),
        "delta_gt_0_10": float(np.mean(delta > 0.10 + 1e-12)),
        "transition_group_medians": {
            state: (float(np.median(delta[mask])) if np.any(mask) else None)
            for state, mask in states.items()
        },
    }


def csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def build_figure(results: dict[str, Any], thresholds: list[float]) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True, constrained_layout=True)
    for axis, split in zip(axes, ("testA", "testB"), strict=True):
        curve = results[split]["threshold_curve"]
        point = np.asarray([100 * curve[f"{threshold:.2f}"]["point"] for threshold in thresholds])
        low = np.asarray([100 * curve[f"{threshold:.2f}"]["lower_95"] for threshold in thresholds])
        high = np.asarray([100 * curve[f"{threshold:.2f}"]["upper_95"] for threshold in thresholds])
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.axvline(0.5, color="#777777", linestyle="--", linewidth=1, label="Primary IoU=0.5")
        axis.fill_between(thresholds, low, high, color="#0072B2", alpha=0.18, label="Paired 95% interval")
        axis.plot(thresholds, point, marker="o", color="#0072B2", linewidth=1.6, label="Full minus Z")
        axis.set_title(split)
        axis.set_xlabel("IoU correctness threshold")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Accuracy difference (percentage points)")
    axes[1].legend(loc="upper left", fontsize=8)
    path = FIGURES / "paired_localization_threshold_crossing.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    require(CAMPAIGN.is_dir(), "Campaign is not initialized")
    verify_v2_lock()
    protocol = read_json(CAMPAIGN / "FROZEN_PROTOCOL.json")
    require(sha256(INPUT_CAMPAIGN / "QUALITY_ARTIFACT_LEDGER.json") == QUALITY_LEDGER_SHA256,
            "Quality artifact ledger changed")
    ledger = read_json(INPUT_CAMPAIGN / "QUALITY_ARTIFACT_LEDGER.json")
    require(ledger.get("status") == "pass", "Quality ledger is not accepted")
    ledger_cells = ledger.get("cells")
    require(isinstance(ledger_cells, dict) and set(ledger_cells) == set(CELL_PATHS), "Quality cell inventory differs")

    arrays: dict[str, np.ndarray] = {}
    ids: dict[str, list[str]] = {}
    input_cells: dict[str, Any] = {}
    for cell_id in ("Full_testA", "Z_M448_0_testA", "Full_testB", "Z_M448_0_testB"):
        arrays[cell_id], ids[cell_id], input_cells[cell_id] = validate_cell(cell_id, ledger_cells[cell_id])
    for split in ("testA", "testB"):
        manifest_ids = [str(row["sample_id"]) for row in iter_jsonl(MANIFESTS[split][0])]
        require_exact_pairing(ids[f"Full_{split}"], ids[f"Z_M448_0_{split}"], manifest_ids, split)
    atomic_json(ANALYSIS / "INPUT_AUDIT.json", {
        "schema_version": 1, "status": "pass", "completed_at_utc": now(),
        "quality_artifact_ledger_sha256": QUALITY_LEDGER_SHA256, "cells": input_cells,
    })

    thresholds = [float(value) for value in protocol["threshold_grid"]]
    results: dict[str, Any] = {}
    transition_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for split in ("testA", "testB"):
        full, z = arrays[f"Full_{split}"], arrays[f"Z_M448_0_{split}"]
        counts = transition_counts(full, z)
        require_expected_transitions(counts, protocol["expected_transitions"][split], split)
        curve, transition_intervals = paired_bootstrap(
            full, z, thresholds, int(protocol["bootstrap"]["child_seeds"][split]),
            int(protocol["bootstrap"]["resamples"]),
        )
        summaries = iou_summaries(full, z)
        results[split] = {
            "samples": len(full), "transitions": counts, "transition_rates": transition_intervals,
            "threshold_curve": {f"{threshold:.2f}": curve[threshold] for threshold in thresholds},
            "iou_delta_summaries": summaries,
        }
        for state in ("both_correct", "full_only", "z_only", "both_incorrect"):
            transition_rows.append({"split": split, "state": state, "count": counts[state], **transition_intervals[state]})
        for threshold in thresholds:
            curve_rows.append({"split": split, "iou_threshold": f"{threshold:.2f}", **curve[threshold]})
        for metric in ("mean", "median", "q1", "q3", "abs_delta_le_0_05", "delta_lt_minus_0_10",
                       "delta_lt_minus_0_25", "delta_gt_0_10"):
            summary_rows.append({"split": split, "metric": metric, "value": summaries[metric]})
        for state, value in summaries["transition_group_medians"].items():
            summary_rows.append({"split": split, "metric": f"median_delta_{state}", "value": value})

    analysis_value = {
        "schema_version": 1, "status": "complete", "analysis_type": "post_hoc_explanatory_quality_only",
        "name": "paired localization threshold-crossing analysis", "heldout_status_unchanged": "HELDOUT_MIXED_SPLIT_RESULT",
        "primary_threshold": 0.5, "threshold_grid": thresholds, "bootstrap": protocol["bootstrap"],
        "delta_directions": {"accuracy": "Full_minus_Z_M448_0", "iou": "Z_M448_0_minus_Full"},
        "results": results,
        "limitations": [
            "Post-hoc descriptive analysis; it does not alter the frozen held-out decision.",
            "The complete Z schedule is compared as a bundle; no individual stage receives a causal attribution.",
            "No semantic subgroup, category search, or hand-selected qualitative example is included.",
        ],
    }
    atomic_json(ANALYSIS / "ERROR_ANALYSIS.json", analysis_value)
    atomic_text(ANALYSIS / "ERROR_ANALYSIS_TRANSITIONS.csv", csv_text(
        transition_rows, ["split", "state", "count", "rate", "lower_95", "upper_95"]))
    atomic_text(ANALYSIS / "ERROR_ANALYSIS_THRESHOLD_CURVE.csv", csv_text(
        curve_rows, ["split", "iou_threshold", "point", "lower_95", "upper_95"]))
    atomic_text(ANALYSIS / "ERROR_ANALYSIS_SUMMARIES.csv", csv_text(
        summary_rows, ["split", "metric", "value"]))
    figure = build_figure(results, thresholds)
    report = """# Frozen-Z paired localization threshold-crossing analysis

## Scope and status

This is a post-hoc, explanatory analysis of the already sealed Full and Z_M448_0 RefCOCO testA and testB rows. It does not reopen candidate selection, revise the preregistered thresholds, add a method, or create a held-out timing claim. The accepted held-out status remains `HELDOUT_MIXED_SPLIT_RESULT`: testB passed all three descriptive retention gates, while testA failed only the accuracy-retention gate.

## Paired transitions at the primary IoU threshold

At IoU 0.5, testA contains 2,135 examples correct under both methods, 748 correct only under Full, 526 correct only under Z_M448_0, and 2,248 incorrect under both. Thus 1,274 examples change correctness state, but the relevant net imbalance is 222 more Full-only than Z-only successes, exactly the 3.92 percentage-point accuracy difference. TestB contains 1,413 both-correct, 622 Full-only, 505 Z-only and 2,555 both-incorrect examples. Its 1,127 flips are similarly numerous, but the net imbalance is smaller at 117 examples, or 2.30 points. The mixed decision therefore reflects an asymmetry in the direction of paired threshold crossings rather than a large difference in the total amount of correctness churn between splits.

## Threshold sensitivity

The fixed threshold curve evaluates the same paired boxes from IoU 0.30 to 0.70. On testA, Z_M448_0 is approximately level with Full at the coarsest threshold, after which the Full advantage increases around the primary and stricter localization thresholds. TestB follows the same direction more weakly. Read together with the much smaller mean-IoU differences, this pattern is consistent with the complete Z schedule preserving coarse target localization better than precise box overlap. It is not evidence that the final zero-token stage, the earlier merge, or any other single operation caused the change; isolating a stage would require a separately authorized ablation.

## Interpretation and limitations

The analysis strengthens the dissertation by explaining why a small average-IoU loss can coexist with a larger Acc@0.5 loss: Acc@0.5 is a thresholded metric, and gains and losses on different examples can cancel in the mean while still producing an imbalanced number of threshold crossings. The result also cautions against describing Z_M448_0 as simply preserving or losing a fixed set of examples. It changes which examples clear the localization criterion in both directions.

This analysis was specified after the held-out outcome was known and is therefore descriptive, not confirmatory. The threshold grid was frozen before generating this derived output, splits remain separate, and no semantic categories or examples were searched for a favourable explanation. The correct conclusion remains bounded: the complete Z schedule offers substantially lower visual token-layer work, meets the frozen quality rule on testB, and misses the testA accuracy rule because Full-only successes exceed Z-only successes by a larger margin. It does not establish equivalence, population generalisation, causal stage attribution, or held-out acceleration.
"""
    atomic_text(CAMPAIGN / "RESEARCH_REPORT.md", report)
    inventory_files = [
        ANALYSIS / "INPUT_AUDIT.json", ANALYSIS / "ERROR_ANALYSIS.json",
        ANALYSIS / "ERROR_ANALYSIS_TRANSITIONS.csv", ANALYSIS / "ERROR_ANALYSIS_THRESHOLD_CURVE.csv",
        ANALYSIS / "ERROR_ANALYSIS_SUMMARIES.csv", figure, CAMPAIGN / "RESEARCH_REPORT.md",
    ]
    require_allowed_output_inventory(
        [str(path.relative_to(ANALYSIS)) for path in inventory_files if path != CAMPAIGN / "RESEARCH_REPORT.md"]
        + ["ANALYSIS_INVENTORY.json"]
    )
    atomic_json(ANALYSIS / "ANALYSIS_INVENTORY.json", {
        "schema_version": 1, "status": "complete", "completed_at_utc": now(),
        "files": {str(path.relative_to(CAMPAIGN)): record_file(path) for path in inventory_files},
        "logical_tables": 1, "logical_figures": 1, "forbidden_subgroups_present": False,
    })
    registry = read_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json")
    registry.update({"status": "analysis_complete", "phase": "claim_audit",
                     "analysis_inventory_sha256": sha256(ANALYSIS / "ANALYSIS_INVENTORY.json")})
    atomic_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json", registry)
    print(json.dumps({"status": "complete", "analysis": str(ANALYSIS / "ERROR_ANALYSIS.json"),
                      "figure": str(figure)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ErrorAnalysisFailure, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
