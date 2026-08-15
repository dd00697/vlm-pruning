#!/usr/bin/env python
"""Independently validate one frozen adaptive route-gate training artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm_pruning.artifacts import sha256_file  # noqa: E402
from vlm_pruning.llava_hooks.route_gate import FEATURE_DIMENSION, FEATURE_SCHEMA_SHA256  # noqa: E402
from vlm_pruning.llava_hooks.route_gate_model import (  # noqa: E402
    TRAINING_SPEC,
    TRAINING_SPEC_SHA256,
    build_gate_model,
    calibrate_threshold,
    normalization_sha256,
)


CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
DEFAULT_OUTPUT_ROOT = CAMPAIGN_ROOT / "gate_training"
DEFAULT_LABEL_RUN = CAMPAIGN_ROOT / "gate_labels/labels_full3000_attempt1"


class GateTrainingValidationError(RuntimeError):
    """Raised when a gate-training artifact violates the frozen protocol."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GateTrainingValidationError(f"{label}: expected {expected!r}, found {actual!r}")


def close(actual: float, expected: float, label: str, tolerance: float = 1e-7) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise GateTrainingValidationError(f"{label}: expected {expected}, found {actual}")


def verify_source(config: dict[str, Any]) -> None:
    commit = str(config.get("source/git_commit") or "")
    sources = config.get("source/sha256")
    if len(commit) != 40 or config.get("source/git_dirty") is not False or not isinstance(sources, dict):
        raise GateTrainingValidationError("Training source provenance is absent or dirty")
    for relative, expected in sources.items():
        payload = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=REPO_ROOT)
        equal(hashlib.sha256(payload).hexdigest(), expected, f"source hash {relative}")


def load_train_arrays(label_run: Path) -> dict[str, Any]:
    labels_path = label_run / "labels.jsonl"
    features = np.empty((3000, FEATURE_DIMENSION), dtype=np.float32)
    safe_iou = np.empty(3000, dtype=np.float64)
    aggressive_iou = np.empty(3000, dtype=np.float64)
    safe_valid = np.empty(3000, dtype=np.bool_)
    aggressive_valid = np.empty(3000, dtype=np.bool_)
    partitions: list[str] = []
    sample_ids: list[str] = []
    with labels_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 3000:
                raise GateTrainingValidationError("Train-label artifact has too many rows")
            row = json.loads(line)
            vector = np.asarray(row["feature_vector"], dtype=np.float32)
            if vector.shape != (FEATURE_DIMENSION,) or not np.isfinite(vector).all():
                raise GateTrainingValidationError(f"Malformed features at row {index}")
            equal(row.get("feature_schema_sha256"), FEATURE_SCHEMA_SHA256, f"feature schema row {index}")
            equal(hashlib.sha256(vector.tobytes()).hexdigest(), row.get("feature_vector_sha256"), f"feature hash row {index}")
            features[index] = vector
            safe_iou[index] = float(row["safe_prediction"]["box_iou"])
            aggressive_iou[index] = float(row["aggressive_prediction"]["box_iou"])
            safe_valid[index] = bool(row["safe_prediction"]["valid"])
            aggressive_valid[index] = bool(row["aggressive_prediction"]["valid"])
            partitions.append(str(row["training_partition"]))
            sample_ids.append(str(row["sample_id"]))
    if len(sample_ids) != 3000 or len(set(sample_ids)) != 3000:
        raise GateTrainingValidationError("Train-label artifact does not have 3,000 unique rows")
    counts = {name: partitions.count(name) for name in ("fit", "early_stop", "calibration")}
    equal(counts, TRAINING_SPEC["partitions"], "partition counts")
    return {
        "features": features,
        "safe_iou": safe_iou,
        "aggressive_iou": aggressive_iou,
        "safe_valid": safe_valid,
        "aggressive_valid": aggressive_valid,
        "partitions": np.asarray(partitions),
        "sample_ids": sample_ids,
    }


def validate_run(run_dir: Path, output_root: Path, label_run: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    history_path = run_dir / "training_history.json"
    calibration_path = run_dir / "calibration.json"
    predictions_path = run_dir / "calibration_predictions.jsonl"
    environment_path = run_dir / "environment.json"
    for path in (config_path, checkpoint_path, history_path, calibration_path, predictions_path, environment_path):
        if not path.is_file():
            raise GateTrainingValidationError(f"Missing training artifact: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verify_source(config)
    seed = int(config["seed"])
    if seed not in (0, 1, 2) or config.get("primary_seed") != (seed == 0):
        raise GateTrainingValidationError("Training seed registration differs")
    equal(config["training_spec_sha256"], TRAINING_SPEC_SHA256, "training spec hash")
    equal(config["feature_schema_sha256"], FEATURE_SCHEMA_SHA256, "feature schema hash")
    equal(config.get("validation_labels_used"), False, "validation-label leakage flag")
    equal(config["train_label_run"], str(label_run.resolve().relative_to(REPO_ROOT)), "train-label path")
    equal(config["train_labels_sha256"], sha256_file(label_run / "labels.jsonl"), "train-label hash")
    equal(config["train_label_config_sha256"], sha256_file(label_run / "config.json"), "train-label config hash")
    equal(config["train_label_integrity_sha256"], sha256_file(label_run / "integrity_report.json"), "train-label integrity hash")
    equal(config["train_label_completion_sha256"], sha256_file(label_run / "COMPLETED.json"), "train-label completion hash")

    frozen = json.loads((output_root / "FROZEN_GATE_TRAINING_SPEC.json").read_text(encoding="utf-8"))
    equal(frozen["status"], "frozen_before_seed_0", "spec freeze status")
    equal(frozen["training_spec"], TRAINING_SPEC, "frozen training spec")
    equal(frozen["training_spec_sha256"], TRAINING_SPEC_SHA256, "frozen spec hash")
    arrays = load_train_arrays(label_run)
    fit = arrays["features"][arrays["partitions"] == "fit"].astype(np.float64)
    expected_mean = fit.mean(axis=0).astype(np.float32)
    expected_std = np.maximum(fit.std(axis=0, ddof=0).astype(np.float32), np.float32(1e-6))
    mean_path = output_root / "normalization_mean.npy"
    std_path = output_root / "normalization_std.npy"
    normalization_path = output_root / "NORMALIZATION.json"
    mean = np.load(mean_path, allow_pickle=False)
    std = np.load(std_path, allow_pickle=False)
    if not np.array_equal(mean, expected_mean) or not np.array_equal(std, expected_std):
        raise GateTrainingValidationError("Normalization is not an exact fit-only recomputation")
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    expected_normalization_hash = normalization_sha256(mean, std)
    equal(normalization["fit_rows"], 2160, "normalization fit rows")
    equal(normalization["normalization_sha256"], expected_normalization_hash, "normalization hash")
    equal(normalization["mean_sha256"], sha256_file(mean_path), "normalization mean file hash")
    equal(normalization["standard_deviation_sha256"], sha256_file(std_path), "normalization std file hash")
    equal(config["normalization_sha256"], expected_normalization_hash, "config normalization hash")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    equal(checkpoint["seed"], seed, "checkpoint seed")
    equal(checkpoint["training_spec_sha256"], TRAINING_SPEC_SHA256, "checkpoint spec hash")
    equal(checkpoint["feature_schema_sha256"], FEATURE_SCHEMA_SHA256, "checkpoint feature hash")
    equal(checkpoint["normalization_sha256"], expected_normalization_hash, "checkpoint normalization hash")
    equal(checkpoint["config_sha256"], sha256_file(config_path), "checkpoint config hash")
    model = build_gate_model(torch).to(device="cpu", dtype=torch.float32)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    equal(sum(parameter.numel() for parameter in model.parameters()), 525570, "trainable parameter count")

    history = json.loads(history_path.read_text(encoding="utf-8"))
    rows = history["history"]
    if not 1 <= len(rows) <= 50 or int(history["epochs_trained"]) != len(rows):
        raise GateTrainingValidationError("Training history epoch count is invalid")
    losses = [float(row["early_stop_total_loss"]) for row in rows]
    expected_best_index = min(range(len(losses)), key=losses.__getitem__)
    equal(history["best_epoch"], expected_best_index + 1, "best early-stop epoch")
    equal(checkpoint["best_epoch"], history["best_epoch"], "checkpoint best epoch")
    close(history["best_early_stop_total_loss"], losses[expected_best_index], "best early-stop loss", tolerance=1e-12)
    if len(rows) < 50 and len(rows) - int(history["best_epoch"]) < 8:
        raise GateTrainingValidationError("Early stopping did not observe the frozen eight stale epochs")

    calibration_indices = np.flatnonzero(arrays["partitions"] == "calibration")
    normalized = ((arrays["features"][calibration_indices] - mean) / std).astype(np.float32)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(torch.from_numpy(normalized))[:, 0]).numpy().astype(np.float64)
    expected_calibration = calibrate_threshold(
        probabilities.tolist(),
        arrays["safe_iou"][calibration_indices].tolist(),
        arrays["aggressive_iou"][calibration_indices].tolist(),
        arrays["safe_valid"][calibration_indices].tolist(),
        arrays["aggressive_valid"][calibration_indices].tolist(),
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    for key in (
        "status",
        "successful_positive_aggressive_rate",
        "always_safe_fallback",
        "threshold",
        "chosen_metrics",
        "always_safe_metrics",
        "candidate_thresholds_evaluated",
        "feasible_positive_aggressive_thresholds",
        "threshold_evidence",
    ):
        equal(calibration[key], expected_calibration[key], f"calibration {key}")
    equal(calibration["checkpoint_sha256"], sha256_file(checkpoint_path), "calibration checkpoint hash")
    equal(calibration["rows"], 600, "calibration rows")

    prediction_rows = [
        json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    equal(len(prediction_rows), 600, "calibration prediction count")
    threshold = float(calibration["threshold"])
    for local_index, (row, source_index) in enumerate(zip(prediction_rows, calibration_indices, strict=True)):
        equal(row["sample_id"], arrays["sample_ids"][int(source_index)], f"calibration ID row {local_index}")
        equal(row["source_label_row"], int(source_index), f"calibration source row {local_index}")
        close(row["probability_safe"], probabilities[local_index], f"calibration probability row {local_index}")
        equal(row["threshold"], threshold, f"calibration threshold row {local_index}")
        expected_route = "K12_QUERY_TOPK" if probabilities[local_index] >= threshold else "M448"
        equal(row["route_choice"], expected_route, f"calibration route row {local_index}")

    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    equal(environment["trainable_parameters"], 525570, "environment parameter count")
    equal(environment["llava_parameters_trained"], 0, "frozen LLaVA parameter count")
    return {
        "status": "pass",
        "seed": seed,
        "checks": {
            "committed_clean_source": True,
            "train_only_labels_and_no_validation_leakage": True,
            "exact_fit_early_stop_calibration_partitions": True,
            "fit_only_normalization_recomputed": True,
            "frozen_architecture_and_parameter_limit": True,
            "checkpoint_load_and_metadata": True,
            "early_stopping_rule": True,
            "calibration_probabilities_recomputed": True,
            "constrained_threshold_recomputed": True,
        },
        "rows": {"fit": 2160, "early_stop": 240, "calibration": 600},
        "best_epoch": int(history["best_epoch"]),
        "epochs_trained": int(history["epochs_trained"]),
        "threshold": threshold,
        "calibration_status": calibration["status"],
        "calibration_metrics": calibration["chosen_metrics"],
        "training_spec_sha256": TRAINING_SPEC_SHA256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "normalization_sha256": expected_normalization_hash,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path),
        "history_sha256": sha256_file(history_path),
        "calibration_sha256": sha256_file(calibration_path),
        "calibration_predictions_sha256": sha256_file(predictions_path),
        "environment_sha256": sha256_file(environment_path),
    }


def validate_and_finalize(run_dir: Path, output_root: Path, label_run: Path) -> dict[str, Any]:
    report = validate_run(run_dir.resolve(), output_root.resolve(), label_run.resolve())
    integrity_path = run_dir / "integrity_report.json"
    completion_path = run_dir / "COMPLETED.json"
    if completion_path.is_file():
        stored_report = json.loads(integrity_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        equal(stored_report, report, "stored integrity report")
        equal(completion["integrity_report_sha256"], sha256_file(integrity_path), "completion integrity hash")
        for name, expected in completion["artifact_sha256"].items():
            equal(sha256_file(run_dir / name), expected, f"completion artifact hash {name}")
        return report
    atomic_write(integrity_path, canonical_bytes(report))
    artifact_names = (
        "config.json",
        "checkpoint.pt",
        "training_history.json",
        "calibration.json",
        "calibration_predictions.jsonl",
        "environment.json",
    )
    completion = {
        "status": "complete",
        "seed": report["seed"],
        "integrity_report_sha256": sha256_file(integrity_path),
        "artifact_sha256": {name: sha256_file(run_dir / name) for name in artifact_names},
    }
    atomic_write(completion_path, canonical_bytes(completion))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--label-run", type=Path, default=DEFAULT_LABEL_RUN)
    args = parser.parse_args()
    try:
        report = validate_and_finalize(args.run_dir, args.output_root, args.label_run)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (GateTrainingValidationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
