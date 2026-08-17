#!/usr/bin/env python
"""Apply frozen SAQR quality/timing gates and produce compact campaign analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for entry in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from run_saqr_campaign import (  # noqa: E402
    CAMPAIGN_ROOT,
    DISCOVERY_MANIFEST,
    QUALITY_ROOT,
    TRAIN_MANIFEST,
    atomic_json,
    read_jsonl,
    sha256_file,
)
from run_saqr_matched_timing import TIMING_ROOT  # noqa: E402


ANALYSIS_ROOT = CAMPAIGN_ROOT / "analysis"
K12_DISCOVERY = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815/analysis/rescored/K12_ORIGINAL.jsonl"
K12_TRAIN_LABELS = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815/gate_labels/labels_full3000_attempt1/labels.jsonl"
RESAMPLES = 10_000


class AnalysisError(RuntimeError):
    pass


def paired_interval(candidate: np.ndarray, reference: np.ndarray, *, seed: int) -> dict[str, Any]:
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise AnalysisError("Paired vectors do not align")
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(RESAMPLES, dtype=np.float64)
    for start in range(0, RESAMPLES, 500):
        count = min(500, RESAMPLES - start)
        indices = rng.integers(0, delta.size, size=(count, delta.size))
        draws[start : start + count] = delta[indices].mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "point_delta": float(delta.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "bootstrap_resamples": RESAMPLES,
        "paired_rows": int(delta.size),
        "seed": seed,
        "direction": "candidate_minus_reference",
    }


def vectors_from_saqr(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([float(row["correct"]) for row in rows]),
        np.asarray([float(row["iou"]) for row in rows]),
        np.asarray([float(row["parse_error"] is None) for row in rows]),
    )


def metrics(acc: np.ndarray, iou: np.ndarray, parsed: np.ndarray) -> dict[str, float]:
    return {
        "rows": int(acc.size),
        "parse_rate": float(parsed.mean()),
        "acc_at_0_5": float(acc.mean()),
        "miou": float(iou.mean()),
        "median_iou": float(np.median(iou)),
    }


def calibration_reference(manifest_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    label_rows = read_jsonl(K12_TRAIN_LABELS)
    by_id = {str(row["sample_id"]): row for row in label_rows}
    selected = []
    for row in manifest_rows:
        if row["training_partition"] != "calibration":
            continue
        label = by_id.get(str(row["sample_id"]))
        if label is None or label["training_partition"] != "calibration":
            raise AnalysisError("K12 calibration label pairing failed")
        selected.append(label["safe_prediction"])
    if len(selected) != 600:
        raise AnalysisError("K12 calibration reference does not contain 600 rows")
    return selected


def run_calibration_gate() -> int:
    candidate_path = QUALITY_ROOT / "calibration_saqr196_attempt1/predictions.jsonl"
    candidate = read_jsonl(candidate_path)
    manifest = read_jsonl(TRAIN_MANIFEST)
    expected_ids = [str(row["sample_id"]) for row in manifest if row["training_partition"] == "calibration"]
    if [row["sample_id"] for row in candidate] != expected_ids:
        raise AnalysisError("SAQR calibration rows do not match frozen calibration order")
    reference = calibration_reference(manifest)
    candidate_acc, candidate_iou, candidate_parsed = vectors_from_saqr(candidate)
    reference_acc = np.asarray([float(row["success_iou_0_5"]) for row in reference])
    reference_iou = np.asarray([float(row["box_iou"]) for row in reference])
    reference_parsed = np.asarray([float(row["predicted_box"] is not None) for row in reference])
    candidate_metrics = metrics(candidate_acc, candidate_iou, candidate_parsed)
    reference_metrics = metrics(reference_acc, reference_iou, reference_parsed)
    acc_loss = reference_metrics["acc_at_0_5"] - candidate_metrics["acc_at_0_5"]
    miou_loss = reference_metrics["miou"] - candidate_metrics["miou"]
    checks = {
        "parse_rate_gte_0_995": candidate_metrics["parse_rate"] >= 0.995,
        "acc_loss_lte_0_030": acc_loss <= 0.030,
        "miou_loss_lte_0_015": miou_loss <= 0.015,
        "artifact_integrity": True,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "gate": "CALIBRATION_FEASIBLE",
        "candidate": candidate_metrics,
        "reference_k12": reference_metrics,
        "acc_loss": acc_loss,
        "miou_loss": miou_loss,
        "checks": checks,
        "candidate_sha256": sha256_file(candidate_path),
        "reference_sha256": sha256_file(K12_TRAIN_LABELS),
        "next_state": (
            "RUN_DISCOVERY_QUALITY" if all(checks.values())
            else "SKIPPED_DISCOVERY_AND_TIMING_CALIBRATION_GATE_FAILED"
        ),
    }
    atomic_json(ANALYSIS_ROOT / "CALIBRATION_GATE.json", result)
    return 0


def discovery_reference() -> list[dict[str, Any]]:
    rows = read_jsonl(K12_DISCOVERY)
    manifest = read_jsonl(DISCOVERY_MANIFEST)
    if [row["sample_id"] for row in rows] != [str(row["sample_id"]) for row in manifest]:
        raise AnalysisError("Accepted K12 reference does not match frozen discovery order")
    return rows


def run_discovery_gate() -> int:
    calibration = json.loads((ANALYSIS_ROOT / "CALIBRATION_GATE.json").read_text())
    if calibration["status"] != "PASS":
        raise AnalysisError("Discovery analysis is forbidden after calibration failure")
    candidate_path = QUALITY_ROOT / "discovery_saqr196_attempt1/predictions.jsonl"
    no_query_path = QUALITY_ROOT / "discovery_saqr196_no_query_attempt1/predictions.jsonl"
    candidate = read_jsonl(candidate_path)
    no_query = read_jsonl(no_query_path)
    reference = discovery_reference()
    ids = [str(row["sample_id"]) for row in reference]
    if [row["sample_id"] for row in candidate] != ids or [row["sample_id"] for row in no_query] != ids:
        raise AnalysisError("Discovery/no-query pairing failed")
    candidate_acc, candidate_iou, candidate_parsed = vectors_from_saqr(candidate)
    no_query_acc, no_query_iou, no_query_parsed = vectors_from_saqr(no_query)
    reference_acc = np.asarray([float(row["success_iou_0_5"]) for row in reference])
    reference_iou = np.asarray([float(row["box_iou"]) for row in reference])
    reference_parsed = np.asarray([float(row["predicted_box"] is not None) for row in reference])
    candidate_metrics = metrics(candidate_acc, candidate_iou, candidate_parsed)
    reference_metrics = metrics(reference_acc, reference_iou, reference_parsed)
    no_query_metrics = metrics(no_query_acc, no_query_iou, no_query_parsed)
    acc_loss = reference_metrics["acc_at_0_5"] - candidate_metrics["acc_at_0_5"]
    miou_loss = reference_metrics["miou"] - candidate_metrics["miou"]
    feasible_checks = {
        "parse_rate_gte_0_995": candidate_metrics["parse_rate"] >= 0.995,
        "acc_loss_lte_0_020": acc_loss <= 0.020,
        "miou_loss_lte_0_010": miou_loss <= 0.010,
        "artifact_integrity": True,
    }
    strong_checks = {
        "acc_loss_lte_0_010": acc_loss <= 0.010,
        "miou_loss_lte_0_005": miou_loss <= 0.005,
    }
    movement = {
        "k12_correct_saqr_wrong": int(np.sum((reference_acc == 1) & (candidate_acc == 0))),
        "saqr_correct_k12_wrong": int(np.sum((reference_acc == 0) & (candidate_acc == 1))),
        "near_miss_saqr_iou_0_4_to_0_5": int(np.sum((candidate_iou >= 0.4) & (candidate_iou < 0.5))),
        "catastrophic_loss_delta_iou_lte_minus_0_5": int(np.sum((candidate_iou - reference_iou) <= -0.5)),
    }
    result = {
        "status": "PASS" if all(feasible_checks.values()) else "FAIL",
        "gate": "DISCOVERY_QUALITY_FEASIBLE",
        "strong_quality": all(strong_checks.values()) and all(feasible_checks.values()),
        "candidate": candidate_metrics,
        "reference_k12": reference_metrics,
        "no_query": no_query_metrics,
        "acc_loss": acc_loss,
        "miou_loss": miou_loss,
        "feasible_checks": feasible_checks,
        "strong_checks": strong_checks,
        "paired_candidate_minus_k12": {
            "acc_at_0_5": paired_interval(candidate_acc, reference_acc, seed=1101),
            "miou": paired_interval(candidate_iou, reference_iou, seed=1102),
        },
        "paired_candidate_minus_no_query": {
            "acc_at_0_5": paired_interval(candidate_acc, no_query_acc, seed=1201),
            "miou": paired_interval(candidate_iou, no_query_iou, seed=1202),
        },
        "discordant_and_error_movement": movement,
        "candidate_sha256": sha256_file(candidate_path),
        "no_query_sha256": sha256_file(no_query_path),
        "reference_sha256": sha256_file(K12_DISCOVERY),
        "next_state": (
            "RUN_MATCHED_TIMING" if all(feasible_checks.values())
            else "SKIPPED_MATCHED_TIMING_QUALITY_GATE_FAILED"
        ),
    }
    atomic_json(ANALYSIS_ROOT / "DISCOVERY_QUALITY_GATE.json", result)
    write_quality_csv(result)
    create_quality_figure(result)
    return 0


def write_quality_csv(result: dict[str, Any]) -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS_ROOT / "QUALITY_VS_WORK.csv"
    rows = [
        ("K12_ORIGINAL", result["reference_k12"]["acc_at_0_5"], result["reference_k12"]["miou"], 7872),
        ("E256_POOL", 0.402, 0.413749, 8192),
        ("SAQR196", result["candidate"]["acc_at_0_5"], result["candidate"]["miou"], 6272),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "acc_at_0_5", "miou", "visual_token_layer_work"])
        writer.writerows(rows)


def create_quality_figure(result: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    methods = ["K12", "E256_POOL", "SAQR196"]
    work = [7872, 8192, 6272]
    accuracy = [result["reference_k12"]["acc_at_0_5"], 0.402, result["candidate"]["acc_at_0_5"]]
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.scatter(work, accuracy, s=70)
    for method, x, y in zip(methods, work, accuracy, strict=True):
        axis.annotate(method, (x, y), xytext=(5, 5), textcoords="offset points")
    axis.set_xlabel("Visual token-layer work")
    axis.set_ylabel("RefCOCO Acc@0.5")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = ANALYSIS_ROOT / "quality_vs_token_layer_work.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)


def hierarchical_interval(candidate: np.ndarray, reference: np.ndarray, *, seed: int) -> dict[str, Any]:
    if candidate.shape != reference.shape or candidate.shape[0] != 3:
        raise AnalysisError("Hierarchical timing arrays must align as [3 repetitions, N rows]")
    delta = candidate - reference
    rng = np.random.default_rng(seed)
    draws = np.empty(RESAMPLES)
    repetitions, examples = delta.shape
    for draw in range(RESAMPLES):
        rep_indices = rng.integers(0, repetitions, size=repetitions)
        example_indices = rng.integers(0, examples, size=examples)
        draws[draw] = delta[rep_indices][:, example_indices].mean()
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "point_delta": float(delta.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "bootstrap_resamples": RESAMPLES,
        "repetitions": repetitions,
        "examples_per_repetition": examples,
        "direction": "candidate_minus_k12",
        "seed": seed,
    }


def load_timing_matrix(method: str, key: str) -> np.ndarray:
    matrices = []
    for repetition in (1, 2, 3):
        position = (1 if ORDER_FOR_ANALYSIS[repetition][0] == method else 2)
        path = TIMING_ROOT / "natural" / f"rep{repetition}_pos{position}_{method.lower()}_attempt1/timing.jsonl"
        rows = read_jsonl(path)
        if len(rows) != 1000:
            raise AnalysisError(f"Timing repetition is incomplete: {path}")
        matrices.append([float(row[key]) for row in rows])
    return np.asarray(matrices, dtype=np.float64)


ORDER_FOR_ANALYSIS = {
    1: ("K12_ORIGINAL", "SAQR196"),
    2: ("SAQR196", "K12_ORIGINAL"),
    3: ("K12_ORIGINAL", "SAQR196"),
}


def run_timing_analysis() -> int:
    discovery = json.loads((ANALYSIS_ROOT / "DISCOVERY_QUALITY_GATE.json").read_text())
    if discovery["status"] != "PASS":
        raise AnalysisError("Matched timing analysis is forbidden after discovery-quality failure")
    keys = (
        "end_to_end_latency_s",
        "time_to_first_token_s",
        "main_prefill_with_overhead_latency_s",
        "decode_latency_s",
        "prompt_kv_total_bytes",
        "final_kv_total_bytes",
        "generated_token_count",
        "peak_allocated_gpu_bytes",
    )
    summary: dict[str, Any] = {}
    intervals: dict[str, Any] = {}
    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, key in enumerate(keys):
        candidate = load_timing_matrix("SAQR196", key)
        reference = load_timing_matrix("K12_ORIGINAL", key)
        matrices[key] = (candidate, reference)
        summary[key] = {
            "saqr196_mean": float(candidate.mean()),
            "saqr196_median": float(np.median(candidate)),
            "k12_mean": float(reference.mean()),
            "k12_median": float(np.median(reference)),
            "candidate_minus_k12": float((candidate - reference).mean()),
            "candidate_over_k12": float(candidate.mean() / reference.mean()),
        }
        intervals[key] = hierarchical_interval(candidate, reference, seed=2100 + index)
    e2e = summary["end_to_end_latency_s"]
    prefill = summary["main_prefill_with_overhead_latency_s"]
    e2e_reduction = 1.0 - e2e["candidate_over_k12"]
    prefill_reduction = 1.0 - prefill["candidate_over_k12"]
    checks = {
        "mean_natural_e2e_lower": e2e["saqr196_mean"] < e2e["k12_mean"],
        "e2e_ci95_entirely_below_zero": intervals["end_to_end_latency_s"]["upper_95"] < 0.0,
        "magnitude_gate": e2e_reduction >= 0.05 or prefill_reduction >= 0.10,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "gate": "ACTUAL_REAL_ACCELERATION",
        "checks": checks,
        "e2e_reduction": e2e_reduction,
        "prefill_including_overhead_reduction": prefill_reduction,
        "metrics": summary,
        "hierarchical_paired_intervals": intervals,
    }
    atomic_json(ANALYSIS_ROOT / "TIMING_ACCELERATION_GATE.json", result)
    create_timing_figure(summary)
    return 0


def create_timing_figure(summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    components = ["time_to_first_token_s", "decode_latency_s", "end_to_end_latency_s"]
    labels = ["TTFT", "Decode", "Natural E2E"]
    k12 = [summary[key]["k12_mean"] for key in components]
    saqr = [summary[key]["saqr196_mean"] for key in components]
    x = np.arange(len(components))
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.bar(x - 0.18, k12, width=0.36, label="K12")
    axis.bar(x + 0.18, saqr, width=0.36, label="SAQR196")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Seconds per example")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(ANALYSIS_ROOT / "matched_timing_decomposition.png", dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["calibration-gate", "discovery-gate", "timing-gate"])
    args = parser.parse_args()
    if args.command == "calibration-gate":
        return run_calibration_gate()
    if args.command == "discovery-gate":
        return run_discovery_gate()
    return run_timing_analysis()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
