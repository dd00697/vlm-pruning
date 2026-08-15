#!/usr/bin/env python
"""Independently validate a physical adaptive-route gate quality run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.init_progressive_oracle_campaign import box_iou, parse_box  # noqa: E402
from scripts.validate_progressive_oracle_run import (  # noqa: E402
    validate_cache_and_tokens,
    validate_stage_traces,
)
from vlm_pruning.artifacts import sha256_file  # noqa: E402
from vlm_pruning.llava_hooks.route_gate import (  # noqa: E402
    AGGRESSIVE_ROUTE,
    AGGRESSIVE_SCHEDULE,
    FEATURE_DIMENSION,
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_SHA256,
    GATE_BOUNDARY,
    SAFE_ROUTE,
    SAFE_SCHEDULE,
)
from vlm_pruning.llava_hooks.route_gate_model import (  # noqa: E402
    TRAINING_SPEC_SHA256,
    build_gate_model,
    normalization_sha256,
)


DEFAULT_MANIFEST = REPO_ROOT / (
    "experiments/progressive_oracle_refcoco_val_20260814/"
    "manifests/refcoco_val_discovery1000.jsonl"
)
CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
DEFAULT_TRAINING_ROOT = CAMPAIGN_ROOT / "gate_training"
EXPECTED_MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
REGISTERED_PROMPT = (
    'Identify the target described by this referring expression: "{query}".\n'
    "Return exactly one bounding box in normalized 0-1 image coordinates as "
    "[x_min, y_min, x_max, y_max]. Do not include any other text."
)


class AdaptiveGateValidationError(RuntimeError):
    """Raised when a dynamic gate artifact violates its frozen contract."""


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


def equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AdaptiveGateValidationError(f"{label}: expected {expected!r}, found {actual!r}")


def close(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise AdaptiveGateValidationError(f"{label}: expected {expected}, found {actual}")


def read_jsonl(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AdaptiveGateValidationError(f"Missing JSONL: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected_rows or not all(isinstance(row, dict) for row in rows):
        raise AdaptiveGateValidationError(f"Expected {expected_rows} JSON objects in {path}")
    return rows


def verify_source(config: dict[str, Any]) -> None:
    commit = str(config.get("source/git_commit") or "")
    sources = config.get("source/sha256")
    if len(commit) != 40 or config.get("source/git_dirty") is not False or not isinstance(sources, dict):
        raise AdaptiveGateValidationError("Dynamic-gate source provenance is absent or dirty")
    for relative, expected in sources.items():
        payload = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=REPO_ROOT)
        equal(hashlib.sha256(payload).hexdigest(), expected, f"source hash {relative}")


def route_config(route_name: str, seed: int) -> dict[str, Any]:
    if route_name == SAFE_ROUTE:
        schedule = SAFE_SCHEDULE
    elif route_name == AGGRESSIVE_ROUTE:
        schedule = AGGRESSIVE_SCHEDULE
    else:
        raise AdaptiveGateValidationError(f"Unregistered dynamic route: {route_name}")
    return {
        "schedule": [event.as_dict() for event in schedule],
        "pre_llm_selector": "none",
        "pre_llm_budget": None,
        "oracle_teacher_layer": None,
        "seed": seed,
    }


def load_gate(training_root: Path, seed: int) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    training_run = training_root / f"seed_{seed}"
    try:
        checkpoint = torch.load(training_run / "checkpoint.pt", map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(training_run / "checkpoint.pt", map_location="cpu")
    model = build_gate_model(torch).to(device="cpu", dtype=torch.float32)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    mean = np.load(training_root / "normalization_mean.npy", allow_pickle=False)
    std = np.load(training_root / "normalization_std.npy", allow_pickle=False)
    normalization = json.loads((training_root / "NORMALIZATION.json").read_text(encoding="utf-8"))
    equal(normalization_sha256(mean, std), normalization["normalization_sha256"], "normalization hash")
    return model, mean, std, normalization


def validate_prediction(
    row: dict[str, Any],
    source: dict[str, Any],
    *,
    config: dict[str, Any],
    config_sha: str,
    model: Any,
    mean: np.ndarray,
    std: np.ndarray,
    row_index: int,
) -> None:
    label = f"row {row_index}"
    for key in ("sample_id", "query", "image_id", "image_path", "gt_box", "width", "height"):
        equal(row.get(key), source.get(key), f"{label} {key}")
    equal(row.get("dataset"), "refcoco", f"{label} dataset")
    equal(row.get("split"), "val", f"{label} split")
    equal(row.get("config_sha256"), config_sha, f"{label} config hash")
    equal(row.get("source_git_commit"), config["source/git_commit"], f"{label} source commit")
    seed = int(config["seed"])
    equal(row.get("gate_seed"), seed, f"{label} gate seed")
    equal(row.get("gate_checkpoint_sha256"), config["gate/training"]["checkpoint_sha256"], f"{label} checkpoint")
    equal(row.get("gate_training_spec_sha256"), TRAINING_SPEC_SHA256, f"{label} spec")
    equal(row.get("gate_normalization_sha256"), config["gate/training"]["normalization_sha256"], f"{label} normalization")
    threshold = float(config["gate/training"]["threshold"])
    close(row.get("gate_threshold"), threshold, f"{label} threshold")

    vector = np.asarray(row.get("gate_feature_vector"), dtype=np.float32)
    if vector.shape != (FEATURE_DIMENSION,) or not np.isfinite(vector).all():
        raise AdaptiveGateValidationError(f"{label} has a malformed gate feature vector")
    vector_hash = hashlib.sha256(vector.tobytes()).hexdigest()
    equal(row.get("gate_feature_vector_sha256"), vector_hash, f"{label} feature vector hash")
    equal(row.get("gate/feature_vector_sha256"), vector_hash, f"{label} diagnostic feature hash")
    equal(row.get("gate/feature_schema_sha256"), FEATURE_SCHEMA_SHA256, f"{label} feature schema")
    equal(row.get("gate/feature_dimension"), FEATURE_DIMENSION, f"{label} feature dimension")
    with torch.inference_mode():
        outputs = model(torch.from_numpy(((vector - mean) / std).astype(np.float32)).unsqueeze(0))[0]
        probability = float(torch.sigmoid(outputs[0]).item())
    close(row.get("gate_probability_safe"), probability, f"{label} probability", tolerance=2e-5)
    close(row.get("gate/probability_safe"), row["gate_probability_safe"], f"{label} duplicate probability")
    close(row.get("gate_safe_logit"), float(outputs[0]), f"{label} safe logit", tolerance=2e-5)
    close(row.get("gate_predicted_iou_delta"), float(outputs[1]), f"{label} IoU delta", tolerance=2e-5)
    route_name = SAFE_ROUTE if float(row["gate_probability_safe"]) >= threshold else AGGRESSIVE_ROUTE
    independent_route = SAFE_ROUTE if probability >= threshold else AGGRESSIVE_ROUTE
    equal(independent_route, route_name, f"{label} independent route decision")
    equal(row.get("gate_route_choice"), route_name, f"{label} route choice")
    equal(row.get("gate/route_name"), route_name, f"{label} diagnostic route")
    equal(bool(row.get("gate/aggressive")), route_name == AGGRESSIVE_ROUTE, f"{label} aggressive flag")
    equal(row.get("gate_physical_routes_executed"), 1, f"{label} physical route count")
    equal(row.get("gate/shared_dense_prefix_boundary"), GATE_BOUNDARY, f"{label} gate boundary")
    equal(row.get("gate/shared_dense_prefix_executions"), 1, f"{label} shared prefix count")
    close(row.get("gate/threshold"), threshold, f"{label} diagnostic threshold")
    if float(row.get("gate/latency_s", -1)) < 0 or float(row.get("gate_model_latency_s", -1)) < 0:
        raise AdaptiveGateValidationError(f"{label} gate latency is invalid")
    validate_stage_traces(row, route_config(route_name, seed), row_index)
    validate_cache_and_tokens(row, route_config(route_name, seed), row_index)
    expected_tokens = 246.0 if route_name == SAFE_ROUTE else 214.0
    close(row.get("candidate_average_visual_tokens"), expected_tokens, f"{label} token work")

    predicted, raw, failure = parse_box(
        str(row.get("raw_model_output") or ""), int(source["width"]), int(source["height"])
    )
    iou = box_iou(predicted, list(source["gt_box"])) if predicted is not None else 0.0
    equal(row.get("parse_failure_reason"), failure, f"{label} parse failure")
    equal(bool(row.get("valid")), predicted is not None, f"{label} valid")
    equal(bool(row.get("invalid")), predicted is None, f"{label} invalid")
    equal(bool(row.get("success_iou_0_5")), iou >= 0.5, f"{label} success")
    close(row.get("box_iou"), iou, f"{label} IoU")
    if raw is not None:
        for actual, expected in zip(row.get("raw_parsed_box") or [], raw, strict=True):
            close(actual, expected, f"{label} raw box")
    if predicted is not None:
        for actual, expected in zip(row.get("predicted_box") or [], predicted, strict=True):
            close(actual, expected, f"{label} predicted box")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    routes = Counter(str(row["gate_route_choice"]) for row in rows)
    return {
        "samples/evaluated": count,
        "quality/acc_at_0_5": sum(bool(row["success_iou_0_5"]) for row in rows) / count,
        "quality/mean_iou": sum(float(row["box_iou"]) for row in rows) / count,
        "quality/parsed_box_rate": sum(bool(row["valid"]) for row in rows) / count,
        "tokens/candidate_average_visual_tokens": sum(float(row["candidate_average_visual_tokens"]) for row in rows) / count,
        "tokens/visual_token_layer_work_mean": sum(sum(row["realized_visual_tokens_by_layer"]) for row in rows) / count,
        "gate/safe_rows": routes[SAFE_ROUTE],
        "gate/aggressive_rows": routes[AGGRESSIVE_ROUTE],
        "gate/safe_rate": routes[SAFE_ROUTE] / count,
        "gate/aggressive_rate": routes[AGGRESSIVE_ROUTE] / count,
        "gate/latency_mean_s": sum(float(row["gate/latency_s"]) for row in rows) / count,
        "gate/latency_max_s": max(float(row["gate/latency_s"]) for row in rows),
        "gate/model_latency_mean_s": sum(float(row["gate_model_latency_s"]) for row in rows) / count,
        "efficiency/peak_allocated_mb": max(float(row["peak_allocated_mb"]) for row in rows),
        "efficiency/peak_reserved_mb": max(float(row["peak_reserved_mb"]) for row in rows),
    }


def validate_run(run_dir: Path, manifest_path: Path, training_root: Path, *, expected_rows: int) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    environment_path = run_dir / "environment.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verify_source(config)
    equal(config["dataset/manifest_sha256"], sha256_file(manifest_path), "manifest hash in config")
    equal(config["dataset/manifest_sha256"], EXPECTED_MANIFEST_SHA256, "frozen validation manifest")
    equal(config["dataset/rows"], expected_rows, "config row count")
    equal(config["dataset/name"], "refcoco", "dataset")
    equal(config["dataset/split"], "val", "split")
    equal(config["model/name"], "llava-hf/llava-1.5-7b-hf", "model")
    equal(config["model/precision"], "bf16", "precision")
    equal(config["model/decoding"], "deterministic_greedy", "decoding")
    equal(config["prompt/template"], REGISTERED_PROMPT, "prompt")
    equal(config["gate/boundary"], GATE_BOUNDARY, "gate boundary")
    equal(config["gate/feature_schema"], FEATURE_SCHEMA, "feature schema")
    equal(config["gate/feature_schema_sha256"], FEATURE_SCHEMA_SHA256, "feature schema hash")
    equal(config["gate/feature_dimension"], FEATURE_DIMENSION, "feature dimension")
    equal(config["safe_route"], {"name": SAFE_ROUTE, "schedule": [event.as_dict() for event in SAFE_SCHEDULE]}, "safe route")
    equal(config["aggressive_route"], {"name": AGGRESSIVE_ROUTE, "schedule": [event.as_dict() for event in AGGRESSIVE_SCHEDULE]}, "aggressive route")
    seed = int(config["seed"])
    if seed not in (0, 1, 2) or config["primary_seed"] != (seed == 0):
        raise AdaptiveGateValidationError("Unregistered dynamic gate seed")
    training_run = training_root.resolve() / f"seed_{seed}"
    training = config["gate/training"]
    equal(training["run_dir"], str(training_run.relative_to(REPO_ROOT)), "training run path")
    equal(training["training_spec_sha256"], TRAINING_SPEC_SHA256, "training spec hash")
    equal(training["checkpoint_sha256"], sha256_file(training_run / "checkpoint.pt"), "checkpoint hash")
    equal(training["config_sha256"], sha256_file(training_run / "config.json"), "training config hash")
    equal(training["integrity_sha256"], sha256_file(training_run / "integrity_report.json"), "training integrity hash")
    equal(training["completion_sha256"], sha256_file(training_run / "COMPLETED.json"), "training completion hash")
    model, mean, std, normalization = load_gate(training_root.resolve(), seed)
    equal(training["normalization_sha256"], normalization["normalization_sha256"], "training normalization hash")

    all_manifest = read_jsonl(manifest_path, 1000)
    manifest = all_manifest[:expected_rows]
    predictions = read_jsonl(run_dir / "predictions.jsonl", expected_rows)
    equal([row["sample_id"] for row in predictions], [row["sample_id"] for row in manifest], "manifest order")
    if len({str(row["sample_id"]) for row in predictions}) != expected_rows:
        raise AdaptiveGateValidationError("Dynamic gate prediction IDs are not unique")
    config_sha = sha256_file(config_path)
    for index, (row, source) in enumerate(zip(predictions, manifest, strict=True)):
        validate_prediction(
            row,
            source,
            config=config,
            config_sha=config_sha,
            model=model,
            mean=mean,
            std=std,
            row_index=index,
        )
    metrics = aggregate(predictions)
    return {
        "status": "pass",
        "validated_at_utc": utc_now(),
        "seed": seed,
        "rows": expected_rows,
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": config_sha,
        "environment_sha256": sha256_file(environment_path),
        "predictions_sha256": sha256_file(run_dir / "predictions.jsonl"),
        "checkpoint_sha256": training["checkpoint_sha256"],
        "training_spec_sha256": TRAINING_SPEC_SHA256,
        "normalization_sha256": training["normalization_sha256"],
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "metrics": metrics,
        "checks": {
            "manifest_order_and_unique_ids": True,
            "committed_source_and_training_provenance": True,
            "checkpoint_and_normalization_recomputed": True,
            "feature_vectors_and_probabilities_recomputed": True,
            "shared_dense_k2_prefix_once": True,
            "gate_evaluated_once_and_one_route_executed": True,
            "chosen_route_trace_cache_and_merge": True,
            "raw_outputs_independently_rescored": True,
            "gate_latency_and_memory_recorded": True,
        },
    }


def validate_and_finalize(run_dir: Path, manifest_path: Path, training_root: Path, *, expected_rows: int) -> dict[str, Any]:
    report = validate_run(run_dir, manifest_path, training_root, expected_rows=expected_rows)
    metrics_path = run_dir / "metrics.json"
    integrity_path = run_dir / "integrity_report.json"
    atomic_write(metrics_path, canonical_bytes(report["metrics"]))
    atomic_write(integrity_path, canonical_bytes(report))
    completion = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "seed": report["seed"],
        "rows": expected_rows,
        "manifest_sha256": report["manifest_sha256"],
        "config_sha256": report["config_sha256"],
        "environment_sha256": report["environment_sha256"],
        "predictions_sha256": report["predictions_sha256"],
        "metrics_sha256": sha256_file(metrics_path),
        "integrity_sha256": sha256_file(integrity_path),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "training_spec_sha256": TRAINING_SPEC_SHA256,
        "normalization_sha256": report["normalization_sha256"],
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
    }
    atomic_write(run_dir / "COMPLETED.json", canonical_bytes(completion))
    validate_run(run_dir, manifest_path, training_root, expected_rows=expected_rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    try:
        report = (
            validate_and_finalize(
                args.run_dir, args.manifest, args.training_root, expected_rows=args.expected_rows
            )
            if args.finalize
            else validate_run(
                args.run_dir, args.manifest, args.training_root, expected_rows=args.expected_rows
            )
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (
        AdaptiveGateValidationError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
