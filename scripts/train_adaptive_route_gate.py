#!/usr/bin/env python
"""Freeze and train the single registered RefCOCO adaptive route gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm_pruning.artifacts import sha256_file, source_provenance  # noqa: E402
from vlm_pruning.llava_hooks.route_gate import FEATURE_DIMENSION, FEATURE_SCHEMA_SHA256  # noqa: E402
from vlm_pruning.llava_hooks.route_gate_model import (  # noqa: E402
    TRAINING_SPEC,
    TRAINING_SPEC_SHA256,
    build_gate_model,
    calibrate_threshold,
    normalization_sha256,
)


CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
DEFAULT_LABEL_RUN = CAMPAIGN_ROOT / "gate_labels/labels_full3000_attempt1"
DEFAULT_OUTPUT_ROOT = CAMPAIGN_ROOT / "gate_training"
EXPECTED_MANIFEST_SHA256 = "8f2b567f27ccc05d1fef7b8368f1de7c454127e224c8f25f24944444d2b1d18f"
SOURCE_PATHS = (
    Path("scripts/train_adaptive_route_gate.py"),
    Path("scripts/validate_adaptive_route_gate_training.py"),
    Path("vlm_pruning/llava_hooks/route_gate.py"),
    Path("vlm_pruning/llava_hooks/route_gate_model.py"),
)


class GateTrainingError(RuntimeError):
    """Raised when route-gate training would violate the frozen protocol."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def json_checksum(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-run", type=Path, default=DEFAULT_LABEL_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--freeze-spec-only", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def checked_source() -> dict[str, Any]:
    provenance = source_provenance(REPO_ROOT, SOURCE_PATHS)
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *[str(path) for path in SOURCE_PATHS]],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if status or provenance["git_dirty"] is None:
        raise GateTrainingError("Gate training requires its exact executable source committed and clean")
    provenance["git_dirty"] = False
    provenance["code_paths_checked"] = [str(path) for path in SOURCE_PATHS]
    return provenance


def freeze_spec(output_root: Path, source: dict[str, Any]) -> dict[str, Any]:
    path = output_root / "FROZEN_GATE_TRAINING_SPEC.json"
    proposed = {
        "status": "frozen_before_seed_0",
        "frozen_at_utc": utc_now(),
        "training_spec": TRAINING_SPEC,
        "training_spec_sha256": TRAINING_SPEC_SHA256,
        "source/git_commit": source["git_commit"],
        "source/git_dirty": False,
        "source/sha256": source["source_sha256"],
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in ("status", "training_spec", "training_spec_sha256", "source/git_commit", "source/git_dirty", "source/sha256"):
            if existing.get(key) != proposed.get(key):
                raise GateTrainingError(f"Existing frozen gate spec conflicts at {key}")
        return existing
    atomic_write(path, canonical_bytes(proposed))
    return proposed


def load_labels(label_run: Path) -> dict[str, Any]:
    labels_path = label_run / "labels.jsonl"
    config_path = label_run / "config.json"
    integrity_path = label_run / "integrity_report.json"
    completion_path = label_run / "COMPLETED.json"
    for path in (labels_path, config_path, integrity_path, completion_path):
        if not path.is_file():
            raise GateTrainingError(f"Missing complete train-label artifact: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if integrity.get("status") != "pass" or int(integrity.get("rows", -1)) != 3000:
        raise GateTrainingError("The 3,000-row train-label artifact is not independently valid")
    if config.get("dataset/split") != "train" or config.get("dataset/manifest_checksum") != EXPECTED_MANIFEST_SHA256:
        raise GateTrainingError("Gate labels are not from the frozen RefCOCO-train manifest")
    features = np.empty((3000, FEATURE_DIMENSION), dtype=np.float32)
    safe_iou = np.empty(3000, dtype=np.float64)
    aggressive_iou = np.empty(3000, dtype=np.float64)
    safe_valid = np.empty(3000, dtype=np.bool_)
    aggressive_valid = np.empty(3000, dtype=np.bool_)
    sample_ids: list[str] = []
    partitions: list[str] = []
    with labels_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 3000:
                raise GateTrainingError("Gate label file contains more than 3,000 records")
            row = json.loads(line)
            vector = np.asarray(row["feature_vector"], dtype=np.float32)
            if vector.shape != (FEATURE_DIMENSION,) or not np.isfinite(vector).all():
                raise GateTrainingError(f"Malformed feature vector at label row {index}")
            if row.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256:
                raise GateTrainingError(f"Feature schema differs at label row {index}")
            if hashlib.sha256(vector.tobytes()).hexdigest() != row.get("feature_vector_sha256"):
                raise GateTrainingError(f"Feature vector hash differs at label row {index}")
            features[index] = vector
            safe_iou[index] = float(row["safe_prediction"]["box_iou"])
            aggressive_iou[index] = float(row["aggressive_prediction"]["box_iou"])
            safe_valid[index] = bool(row["safe_prediction"]["valid"])
            aggressive_valid[index] = bool(row["aggressive_prediction"]["valid"])
            sample_ids.append(str(row["sample_id"]))
            partitions.append(str(row["training_partition"]))
    if len(sample_ids) != 3000 or len(set(sample_ids)) != 3000:
        raise GateTrainingError("Gate labels do not contain 3,000 unique records")
    counts = {name: partitions.count(name) for name in ("fit", "early_stop", "calibration")}
    if counts != TRAINING_SPEC["partitions"]:
        raise GateTrainingError(f"Gate-label partitions differ from the frozen split: {counts}")
    return {
        "features": features,
        "safe_iou": safe_iou,
        "aggressive_iou": aggressive_iou,
        "safe_valid": safe_valid,
        "aggressive_valid": aggressive_valid,
        "sample_ids": sample_ids,
        "partitions": np.asarray(partitions),
        "labels_sha256": sha256_file(labels_path),
        "label_config_sha256": sha256_file(config_path),
        "label_integrity_sha256": sha256_file(integrity_path),
        "label_completion_sha256": sha256_file(completion_path),
        "label_run": str(label_run.resolve().relative_to(REPO_ROOT)),
    }


def fit_or_verify_normalization(output_root: Path, data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fit_mask = data["partitions"] == "fit"
    fit = data["features"][fit_mask].astype(np.float64)
    if fit.shape != (2160, FEATURE_DIMENSION):
        raise GateTrainingError("Normalization fit did not receive exactly 2,160 fit records")
    mean = fit.mean(axis=0).astype(np.float32)
    std = fit.std(axis=0, ddof=0).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    expected_hash = normalization_sha256(mean, std)
    mean_path = output_root / "normalization_mean.npy"
    std_path = output_root / "normalization_std.npy"
    manifest_path = output_root / "NORMALIZATION.json"
    if manifest_path.is_file() or mean_path.is_file() or std_path.is_file():
        if not (manifest_path.is_file() and mean_path.is_file() and std_path.is_file()):
            raise GateTrainingError("Normalization artifact is only partially present")
        stored_mean = np.load(mean_path, allow_pickle=False)
        stored_std = np.load(std_path, allow_pickle=False)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not np.array_equal(stored_mean, mean) or not np.array_equal(stored_std, std):
            raise GateTrainingError("Stored normalization differs from a fit-only recomputation")
        if manifest.get("normalization_sha256") != expected_hash:
            raise GateTrainingError("Stored normalization composite hash differs")
        return stored_mean, stored_std, manifest
    atomic_save_npy(mean_path, mean)
    atomic_save_npy(std_path, std)
    manifest = {
        "status": "fit_on_2160_train_records_only",
        "created_at_utc": utc_now(),
        "feature_dimension": FEATURE_DIMENSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "fit_rows": 2160,
        "labels_sha256": data["labels_sha256"],
        "mean_path": str(mean_path.relative_to(REPO_ROOT)),
        "mean_sha256": sha256_file(mean_path),
        "standard_deviation_path": str(std_path.relative_to(REPO_ROOT)),
        "standard_deviation_sha256": sha256_file(std_path),
        "normalization_sha256": expected_hash,
    }
    atomic_write(manifest_path, canonical_bytes(manifest))
    return mean, std, manifest


def targets(data: dict[str, Any], indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    safe_iou = data["safe_iou"][indices]
    aggressive_iou = data["aggressive_iou"][indices]
    classification = torch.from_numpy((safe_iou > aggressive_iou).astype(np.float32))
    regression = torch.from_numpy((safe_iou - aggressive_iou).astype(np.float32))
    critical = (safe_iou >= 0.5) & (aggressive_iou < 0.5)
    weights = torch.from_numpy(np.where(critical, 5.0, 1.0).astype(np.float32))
    return classification, regression, weights


def loss_parts(outputs: torch.Tensor, classification: torch.Tensor, regression: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs[:, 0], classification, reduction="none"
    )
    classification_loss = (classification_loss * weights).mean()
    regression_loss = torch.nn.functional.smooth_l1_loss(
        outputs[:, 1], regression, reduction="mean", beta=1.0
    )
    total = classification_loss + 0.5 * regression_loss
    return total, classification_loss, regression_loss


def evaluate_loss(model: Any, features: torch.Tensor, classification: torch.Tensor, regression: torch.Tensor, weights: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        total, cls, reg = loss_parts(model(features), classification, regression, weights)
    return {"total": float(total), "classification": float(cls), "regression": float(reg)}


def train_seed(
    seed: int,
    output_root: Path,
    data: dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    normalization: dict[str, Any],
    source: dict[str, Any],
) -> None:
    if seed not in (0, 1, 2):
        raise GateTrainingError(f"Unregistered gate seed: {seed}")
    run_dir = output_root / f"seed_{seed}"
    if run_dir.exists():
        raise GateTrainingError(f"Gate training run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    config = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "run_name": f"adaptive_route_gate_seed_{seed}",
        "seed": seed,
        "primary_seed": seed == 0,
        "training_spec_sha256": TRAINING_SPEC_SHA256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "normalization_sha256": normalization["normalization_sha256"],
        "train_label_run": data["label_run"],
        "train_labels_sha256": data["labels_sha256"],
        "train_label_config_sha256": data["label_config_sha256"],
        "train_label_integrity_sha256": data["label_integrity_sha256"],
        "train_label_completion_sha256": data["label_completion_sha256"],
        "validation_labels_used": False,
        "source/git_commit": source["git_commit"],
        "source/git_dirty": False,
        "source/sha256": source["source_sha256"],
        "command/semantic": {
            "program": "scripts/train_adaptive_route_gate.py",
            "seed": seed,
            "label_run": data["label_run"],
            "output_root": str(output_root.resolve().relative_to(REPO_ROOT)),
        },
    }
    config["command/sha256"] = json_checksum(config["command/semantic"])
    atomic_write(run_dir / "config.json", canonical_bytes(config))
    config_sha = sha256_file(run_dir / "config.json")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = build_gate_model(torch).to(device="cpu", dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0)
    normalized = ((data["features"] - mean) / std).astype(np.float32)
    fit_indices = np.flatnonzero(data["partitions"] == "fit")
    early_indices = np.flatnonzero(data["partitions"] == "early_stop")
    calibration_indices = np.flatnonzero(data["partitions"] == "calibration")
    fit_features = torch.from_numpy(normalized[fit_indices])
    early_features = torch.from_numpy(normalized[early_indices])
    fit_class, fit_regression, fit_weights = targets(data, fit_indices)
    early_class, early_regression, early_weights = targets(data, early_indices)
    permutation_generator = torch.Generator(device="cpu").manual_seed(seed + 1000)
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(1, 51):
        model.train()
        permutation = torch.randperm(len(fit_indices), generator=permutation_generator)
        sums = np.zeros(3, dtype=np.float64)
        rows_seen = 0
        learning_rate = float(optimizer.param_groups[0]["lr"])
        for start in range(0, len(fit_indices), 64):
            batch = permutation[start : start + 64]
            optimizer.zero_grad(set_to_none=True)
            total, cls, reg = loss_parts(
                model(fit_features[batch]),
                fit_class[batch],
                fit_regression[batch],
                fit_weights[batch],
            )
            total.backward()
            optimizer.step()
            count = int(batch.numel())
            sums += np.asarray([float(total.detach()), float(cls.detach()), float(reg.detach())]) * count
            rows_seen += count
        early = evaluate_loss(model, early_features, early_class, early_regression, early_weights)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "fit_total_loss": float(sums[0] / rows_seen),
                "fit_classification_loss": float(sums[1] / rows_seen),
                "fit_regression_loss": float(sums[2] / rows_seen),
                "early_stop_total_loss": early["total"],
                "early_stop_classification_loss": early["classification"],
                "early_stop_regression_loss": early["regression"],
            }
        )
        if early["total"] < best_loss:
            best_loss = early["total"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= 8:
            break
    if best_state is None:
        raise GateTrainingError("Gate training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    checkpoint = {
        "state_dict": best_state,
        "seed": seed,
        "best_epoch": best_epoch,
        "training_spec_sha256": TRAINING_SPEC_SHA256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "normalization_sha256": normalization["normalization_sha256"],
        "config_sha256": config_sha,
    }
    atomic_torch_save(run_dir / "checkpoint.pt", checkpoint)
    training_history = {
        "status": "early_stopped" if len(history) < 50 else "maximum_epochs_reached",
        "seed": seed,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_early_stop_total_loss": best_loss,
        "history": history,
    }
    atomic_write(run_dir / "training_history.json", canonical_bytes(training_history))

    calibration_features = torch.from_numpy(normalized[calibration_indices])
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(calibration_features)[:, 0]).numpy().astype(np.float64)
    calibration = calibrate_threshold(
        probabilities.tolist(),
        data["safe_iou"][calibration_indices].tolist(),
        data["aggressive_iou"][calibration_indices].tolist(),
        data["safe_valid"][calibration_indices].tolist(),
        data["aggressive_valid"][calibration_indices].tolist(),
    )
    calibration.update(
        {
            "seed": seed,
            "partition": "calibration",
            "rows": 600,
            "checkpoint_sha256": sha256_file(run_dir / "checkpoint.pt"),
            "training_spec_sha256": TRAINING_SPEC_SHA256,
            "normalization_sha256": normalization["normalization_sha256"],
        }
    )
    atomic_write(run_dir / "calibration.json", canonical_bytes(calibration))
    threshold = float(calibration["threshold"])
    prediction_rows = []
    for local_index, source_index in enumerate(calibration_indices):
        probability = float(probabilities[local_index])
        safe = probability >= threshold
        prediction_rows.append(
            {
                "sample_id": data["sample_ids"][int(source_index)],
                "source_label_row": int(source_index),
                "probability_safe": probability,
                "threshold": threshold,
                "route_choice": "K12_QUERY_TOPK" if safe else "M448",
                "safe_iou": float(data["safe_iou"][source_index]),
                "aggressive_iou": float(data["aggressive_iou"][source_index]),
                "safe_valid": bool(data["safe_valid"][source_index]),
                "aggressive_valid": bool(data["aggressive_valid"][source_index]),
            }
        )
    atomic_write(
        run_dir / "calibration_predictions.jsonl",
        b"".join(
            (json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
            for row in prediction_rows
        ),
    )
    environment = {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "trainable_parameters": sum(item.numel() for item in model.parameters()),
        "llava_parameters_trained": 0,
    }
    atomic_write(run_dir / "environment.json", canonical_bytes(environment))
    from scripts.validate_adaptive_route_gate_training import validate_and_finalize

    report = validate_and_finalize(run_dir, output_root, REPO_ROOT / data["label_run"])
    print(
        json.dumps(
            {
                "seed": seed,
                "status": "complete",
                "best_epoch": best_epoch,
                "threshold": threshold,
                "aggressive_rate": calibration["chosen_metrics"]["aggressive_rate"],
                "integrity_sha256": sha256_file(run_dir / "integrity_report.json"),
                "checkpoint_sha256": report["checkpoint_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    args = parse_args()
    try:
        output_root = args.output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        source = checked_source()
        freeze_spec(output_root, source)
        if args.freeze_spec_only:
            print(
                json.dumps(
                    {
                        "status": "frozen_before_seed_0",
                        "path": str((output_root / "FROZEN_GATE_TRAINING_SPEC.json").relative_to(REPO_ROOT)),
                        "training_spec_sha256": TRAINING_SPEC_SHA256,
                    },
                    indent=2,
                )
            )
            return 0
        if sorted(args.seeds) != sorted(set(args.seeds)) or any(seed not in (0, 1, 2) for seed in args.seeds):
            raise GateTrainingError("Seeds must be a unique subset of registered seeds 0, 1, and 2")
        data = load_labels(args.label_run.resolve())
        mean, std, normalization = fit_or_verify_normalization(output_root, data)
        for seed in args.seeds:
            train_seed(seed, output_root, data, mean, std, normalization, source)
        return 0
    except (GateTrainingError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
