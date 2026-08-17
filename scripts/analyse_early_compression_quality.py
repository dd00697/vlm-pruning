#!/usr/bin/env python
"""Analyse E256 quality and apply the registered Phase-5 branch gates."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_early_compression_quality import atomic_write, canonical_bytes, utc_now  # noqa: E402
from validate_early_compression_quality import validate_run  # noqa: E402
from vlm_pruning.artifacts import sha256_file  # noqa: E402


CAMPAIGN_ROOT = REPO_ROOT / "experiments/hardware_aware_zero_tail_refcoco_val_20260816"
MANIFEST = (
    REPO_ROOT
    / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
)
E256_RUN = CAMPAIGN_ROOT / "early_compression/quality/e256_pool_attempt1"
K12_ROWS = (
    REPO_ROOT
    / "experiments/adaptive_efficiency_refcoco_val_20260815/analysis/rescored/K12_ORIGINAL.jsonl"
)
M448_ROWS = (
    REPO_ROOT
    / "experiments/adaptive_efficiency_refcoco_val_20260815/runs/m448_attempt1/predictions.jsonl"
)
OUTPUT_DIR = CAMPAIGN_ROOT / "early_compression/analysis"
EXPECTED_K12_SHA256 = "6a40fe8125fa4003fcdbe2ac33e73acd353755603d554af193fa833d525b53f2"
EXPECTED_M448_SHA256 = "52e375657ba259a8785dcea3804770070e57f16f5ccc682b78f5edd167c9574b"
RESAMPLES = 10_000


class EarlyAnalysisError(RuntimeError):
    """Raised when accepted quality evidence cannot be paired exactly."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 1000:
        raise EarlyAnalysisError(f"{path} does not contain 1,000 rows")
    identifiers = [str(row["sample_id"]) for row in rows]
    if len(set(identifiers)) != 1000:
        raise EarlyAnalysisError(f"{path} contains duplicate IDs")
    return rows


def paired_interval(candidate: np.ndarray, reference: np.ndarray, *, seed: int) -> dict[str, Any]:
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise EarlyAnalysisError("Paired vectors do not align")
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


def metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "acc_at_0_5": sum(bool(row["success_iou_0_5"]) for row in rows) / len(rows),
        "mean_iou": sum(float(row["box_iou"]) for row in rows) / len(rows),
        "parse_rate": sum(row.get("predicted_box") is not None for row in rows) / len(rows),
    }


def compare(candidate: list[dict[str, Any]], reference: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    candidate_ids = [str(row["sample_id"]) for row in candidate]
    reference_ids = [str(row["sample_id"]) for row in reference]
    if candidate_ids != reference_ids:
        raise EarlyAnalysisError("Reference rows are not aligned to E256")
    candidate_acc = np.asarray([bool(row["success_iou_0_5"]) for row in candidate], dtype=np.float64)
    reference_acc = np.asarray([bool(row["success_iou_0_5"]) for row in reference], dtype=np.float64)
    candidate_iou = np.asarray([float(row["box_iou"]) for row in candidate], dtype=np.float64)
    reference_iou = np.asarray([float(row["box_iou"]) for row in reference], dtype=np.float64)
    return {
        "acc_delta": paired_interval(candidate_acc, reference_acc, seed=seed),
        "miou_delta": paired_interval(candidate_iou, reference_iou, seed=seed + 1),
        "reference_correct_candidate_wrong": int(np.sum((reference_acc == 1) & (candidate_acc == 0))),
        "reference_wrong_candidate_correct": int(np.sum((reference_acc == 0) & (candidate_acc == 1))),
    }


def source_record() -> dict[str, Any]:
    path = Path("scripts/analyse_early_compression_quality.py")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(path)], cwd=REPO_ROOT, text=True
    ).strip()
    if status:
        raise EarlyAnalysisError("Analysis requires a clean committed source file")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    return {
        "git_commit": commit,
        "source_sha256": {str(path): sha256_file(REPO_ROOT / path)},
    }


def run() -> int:
    validation = validate_run(E256_RUN, MANIFEST, expected_rows=1000)
    if validation["status"] != "pass":
        raise EarlyAnalysisError("E256 did not independently validate")
    if sha256_file(K12_ROWS) != EXPECTED_K12_SHA256:
        raise EarlyAnalysisError("Accepted K12 reference hash changed")
    if sha256_file(M448_ROWS) != EXPECTED_M448_SHA256:
        raise EarlyAnalysisError("Accepted M448 reference hash changed")
    e256 = load_jsonl(E256_RUN / "predictions.jsonl")
    k12 = load_jsonl(K12_ROWS)
    m448 = load_jsonl(M448_ROWS)
    result_metrics = metrics(e256)
    k12_like = (
        result_metrics["acc_at_0_5"] >= 0.429
        and result_metrics["mean_iou"] >= 0.4165978636
        and result_metrics["parse_rate"] == 1.0
    )
    near_quality = (
        result_metrics["acc_at_0_5"] >= 0.419
        and result_metrics["mean_iou"] >= 0.4115978636
    )
    completion_path = E256_RUN / "COMPLETED.json"
    quality_results = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": utc_now(),
        "source": source_record(),
        "e256_completion_sha256": sha256_file(completion_path),
        "e256_independent_validation": validation["checks"],
        "reference_hashes": {
            "K12_ORIGINAL": EXPECTED_K12_SHA256,
            "M448": EXPECTED_M448_SHA256,
        },
        "metrics": {
            "E256_POOL": result_metrics,
            "K12_ORIGINAL": metrics(k12),
            "M448": metrics(m448),
        },
        "paired_bootstrap": {
            "resamples": RESAMPLES,
            "E256_POOL_minus_K12_ORIGINAL": compare(e256, k12, seed=20260817),
            "E256_POOL_minus_M448": compare(e256, m448, seed=20260819),
        },
        "quality_screen": {
            "thresholds": {
                "acc_at_0_5_min": 0.429,
                "mean_iou_min": 0.4165978636,
                "parse_rate_exact": 1.0,
            },
            "E256_POOL_pass": k12_like,
        },
        "lora_near_quality_screen": {
            "thresholds": {"acc_at_0_5_min": 0.419, "mean_iou_min": 0.4115978636},
            "E256_POOL_pass": near_quality,
            "acc_check": result_metrics["acc_at_0_5"] >= 0.419,
            "miou_check": result_metrics["mean_iou"] >= 0.4115978636,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    quality_path = OUTPUT_DIR / "QUALITY_RESULTS.json"
    atomic_write(quality_path, canonical_bytes(quality_results))
    branch = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": utc_now(),
        "quality_results_sha256": sha256_file(quality_path),
        "decisions": {
            "E196_POOL": {
                "decision": "SKIPPED_CONSERVATIVE_POOL_NOT_QUALITY_FEASIBLE",
                "trigger": "E256_POOL passes K12-like quality screen",
                "trigger_value": k12_like,
            },
            "E256_LORA": {
                "decision": "SKIPPED_LORA_NEAR_QUALITY_GATE_FAILED",
                "all_required_trigger_conditions": False,
                "integrity_pass": True,
                "acc_at_0_5_at_least_0_419": result_metrics["acc_at_0_5"] >= 0.419,
                "mean_iou_at_least_0_4115978636": result_metrics["mean_iou"] >= 0.4115978636,
                "profile_speed_condition": "NOT_EVALUATED_SHORT_CIRCUIT_NEAR_QUALITY_FAILED",
                "e256_not_already_quality_feasible": not k12_like,
                "training_launched": False,
            },
            "EARLY_COMPRESSION_MATCHED_TIMING": {
                "decision": "SKIPPED_NO_QUALITY_FEASIBLE_EARLY_CANDIDATE",
                "quality_feasible_candidates": [],
            },
            "EARLY_COMPRESSION_FIXED32": {
                "decision": "SKIPPED_NO_QUALITY_FEASIBLE_EARLY_CANDIDATE",
                "quality_feasible_candidates": [],
            },
        },
        "phase5_terminal": True,
        "phase5_result": "NO_QUALITY_FEASIBLE_EARLY_COMPRESSION_CANDIDATE",
    }
    branch_path = OUTPUT_DIR / "BRANCH_DECISIONS.json"
    atomic_write(branch_path, canonical_bytes(branch))
    completion = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "quality_results_sha256": sha256_file(quality_path),
        "branch_decisions_sha256": sha256_file(branch_path),
        "e256_completion_sha256": sha256_file(completion_path),
    }
    atomic_write(OUTPUT_DIR / "COMPLETED.json", canonical_bytes(completion))
    print(json.dumps({"quality": quality_results["metrics"], "branches": branch}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (EarlyAnalysisError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
