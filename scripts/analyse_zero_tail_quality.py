#!/usr/bin/env python
"""Validate and compare the mandatory hardware-campaign zero-tail quality cells."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO_ROOT / "experiments/hardware_aware_zero_tail_refcoco_val_20260816"
MANIFEST = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
K12 = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815/analysis/rescored/K12_ORIGINAL.jsonl"
M448 = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815/runs/m448_attempt1/predictions.jsonl"
RUNS = {
    "Z_K12_0": CAMPAIGN_ROOT / "zero_tail/runs/z_k12_0_attempt1",
    "Z_M448_0": CAMPAIGN_ROOT / "zero_tail/runs/z_m448_0_attempt1",
}
EXPECTED_HASHES = {
    "manifest": "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd",
    "k12": "6a40fe8125fa4003fcdbe2ac33e73acd353755603d554af193fa833d525b53f2",
    "m448": "52e375657ba259a8785dcea3804770070e57f16f5ccc682b78f5edd167c9574b",
}
QUALITY_GATE = {"acc_at_0_5": 0.429, "mean_iou": 0.4165978636, "parse_rate": 1.0}
BOOTSTRAPS = 10_000
SEED = 20260816


class ZeroTailAnalysisError(RuntimeError):
    """Raised when zero-tail quality evidence is incomplete or misaligned."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row.get("sample_id") or "") for row in rows]
    if len(rows) != 1000 or len(set(ids)) != 1000:
        raise ZeroTailAnalysisError(f"Expected 1,000 unique rows: {path}")
    return rows


def assert_aligned(groups: dict[str, list[dict[str, Any]]]) -> list[str]:
    manifest_ids = [str(row["sample_id"]) for row in read_rows(MANIFEST)]
    for name, rows in groups.items():
        if [str(row["sample_id"]) for row in rows] != manifest_ids:
            raise ZeroTailAnalysisError(f"{name} is not in frozen manifest order")
    return manifest_ids


def metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "acc_at_0_5": float(np.mean([bool(row["success_iou_0_5"]) for row in rows])),
        "mean_iou": float(np.mean([float(row["box_iou"]) for row in rows])),
        "parse_rate": float(np.mean([row.get("parse_failure_reason") is None for row in rows])),
    }


def interval(left: np.ndarray, right: np.ndarray, *, seed: int, scale: float) -> dict[str, Any]:
    delta = left - right
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    remaining = BOOTSTRAPS
    while remaining:
        count = min(500, remaining)
        indices = rng.integers(0, delta.size, size=(count, delta.size))
        draws.append(delta[indices].mean(axis=1))
        remaining -= count
    boot = np.concatenate(draws)
    return {
        "point_delta": float(delta.mean() * scale),
        "lower_95": float(np.quantile(boot, 0.025) * scale),
        "upper_95": float(np.quantile(boot, 0.975) * scale),
        "bootstrap_resamples": BOOTSTRAPS,
        "seed": seed,
        "unit": "percentage_points" if scale == 100.0 else "0_to_1",
    }


def paired(left: list[dict[str, Any]], right: list[dict[str, Any]], *, seed_offset: int) -> dict[str, Any]:
    left_acc = np.asarray([float(bool(row["success_iou_0_5"])) for row in left])
    right_acc = np.asarray([float(bool(row["success_iou_0_5"])) for row in right])
    left_iou = np.asarray([float(row["box_iou"]) for row in left])
    right_iou = np.asarray([float(row["box_iou"]) for row in right])
    return {
        "acc_at_0_5_percentage_points": interval(
            left_acc, right_acc, seed=SEED + seed_offset * 10, scale=100.0
        ),
        "mean_iou": interval(
            left_iou, right_iou, seed=SEED + seed_offset * 10 + 1, scale=1.0
        ),
        "discordant_correctness": {
            "left_correct_right_wrong": int(np.sum((left_acc == 1) & (right_acc == 0))),
            "left_wrong_right_correct": int(np.sum((left_acc == 0) & (right_acc == 1))),
        },
    }


def critical_failures(
    candidate: list[dict[str, Any]],
    k12: list[dict[str, Any]],
) -> dict[str, Any]:
    k12_correct_candidate_wrong: list[dict[str, Any]] = []
    candidate_correct_k12_wrong: list[dict[str, Any]] = []
    for current, reference in zip(candidate, k12, strict=True):
        entry = {
            "sample_id": current["sample_id"],
            "query": current["query"],
            "gt_box": current["gt_box"],
            "k12_output": reference["raw_model_output"],
            "k12_iou": reference["box_iou"],
            "candidate_output": current["raw_model_output"],
            "candidate_iou": current["box_iou"],
        }
        if bool(reference["success_iou_0_5"]) and not bool(current["success_iou_0_5"]):
            k12_correct_candidate_wrong.append(entry)
        if not bool(reference["success_iou_0_5"]) and bool(current["success_iou_0_5"]):
            candidate_correct_k12_wrong.append(entry)
    return {
        "k12_correct_candidate_wrong_count": len(k12_correct_candidate_wrong),
        "candidate_correct_k12_wrong_count": len(candidate_correct_k12_wrong),
        "k12_correct_candidate_wrong": k12_correct_candidate_wrong,
        "candidate_correct_k12_wrong": candidate_correct_k12_wrong,
    }


def main() -> int:
    for name, (path, expected) in {
        "manifest": (MANIFEST, EXPECTED_HASHES["manifest"]),
        "k12": (K12, EXPECTED_HASHES["k12"]),
        "m448": (M448, EXPECTED_HASHES["m448"]),
    }.items():
        if sha256(path) != expected:
            raise ZeroTailAnalysisError(f"Frozen {name} hash differs")
    from validate_progressive_oracle_run import validate_run

    independent = {
        cell_id: validate_run(run_dir, MANIFEST, expected_rows=1000, finalize=False)
        for cell_id, run_dir in RUNS.items()
    }
    groups = {
        "K12_ORIGINAL": read_rows(K12),
        "M448": read_rows(M448),
        **{
            cell_id: read_rows(run_dir / "predictions.jsonl")
            for cell_id, run_dir in RUNS.items()
        },
    }
    manifest_ids = assert_aligned(groups)
    result_metrics = {name: metrics(rows) for name, rows in groups.items()}
    gates: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    for candidate_index, cell_id in enumerate(RUNS, 1):
        current = result_metrics[cell_id]
        gates[cell_id] = {
            "passed": (
                current["acc_at_0_5"] >= QUALITY_GATE["acc_at_0_5"]
                and current["mean_iou"] >= QUALITY_GATE["mean_iou"]
                and current["parse_rate"] == QUALITY_GATE["parse_rate"]
            ),
            "thresholds": QUALITY_GATE,
        }
        for reference_index, reference in enumerate(("K12_ORIGINAL", "M448"), 1):
            contrasts[f"{cell_id}-minus-{reference}"] = paired(
                groups[cell_id],
                groups[reference],
                seed_offset=candidate_index * 10 + reference_index,
            )
        failures[cell_id] = critical_failures(groups[cell_id], groups["K12_ORIGINAL"])
    output_dir = CAMPAIGN_ROOT / "zero_tail/analysis"
    quality = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": utc_now(),
        "manifest_sha256": EXPECTED_HASHES["manifest"],
        "manifest_order_sha256": hashlib.sha256("\n".join(manifest_ids).encode()).hexdigest(),
        "metrics": result_metrics,
        "quality_gates": gates,
        "paired_contrasts": contrasts,
        "independent_validation": independent,
        "branch_decision": {
            "Z_M448_16": (
                "SKIPPED_ZERO_TAIL_ALREADY_QUALITY_FEASIBLE"
                if gates["Z_M448_0"]["passed"]
                else "REQUIRED_ZERO_TAIL_RECOVERY"
            )
        },
        "artifact_hashes": {
            cell_id: {
                "predictions": sha256(run_dir / "predictions.jsonl"),
                "metrics": sha256(run_dir / "metrics.json"),
                "integrity": sha256(run_dir / "integrity_report.json"),
                "completion": sha256(run_dir / "COMPLETED.json"),
            }
            for cell_id, run_dir in RUNS.items()
        },
    }
    write_json(output_dir / "QUALITY_RESULTS.json", quality)
    write_json(output_dir / "CRITICAL_FAILURES.json", failures)
    write_json(
        output_dir / "COMPLETED.json",
        {
            "status": "complete",
            "quality_results_sha256": sha256(output_dir / "QUALITY_RESULTS.json"),
            "critical_failures_sha256": sha256(output_dir / "CRITICAL_FAILURES.json"),
        },
    )
    print(json.dumps({"quality_gates": gates, "branch_decision": quality["branch_decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
