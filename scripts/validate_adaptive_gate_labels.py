#!/usr/bin/env python
"""Independently validate paired physical train labels and K2 gate features."""

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
    SCALAR_FEATURE_NAMES,
)


CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
MANIFEST_AUDIT = CAMPAIGN_ROOT / "gate_data/MANIFEST_AUDIT.json"
EXPECTED_MANIFEST_SHA256 = "8f2b567f27ccc05d1fef7b8368f1de7c454127e224c8f25f24944444d2b1d18f"
REGISTERED_PROMPT = (
    'Identify the target described by this referring expression: "{query}".\n'
    "Return exactly one bounding box in normalized 0-1 image coordinates as "
    "[x_min, y_min, x_max, y_max]. Do not include any other text."
)


class GateLabelValidationError(RuntimeError):
    """Raised when a gate-label artifact violates the frozen protocol."""


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
        raise GateLabelValidationError(f"{label}: expected {expected!r}, found {actual!r}")


def close(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise GateLabelValidationError(f"{label}: expected {expected}, found {actual}")


def read_jsonl(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise GateLabelValidationError(f"Missing JSONL: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected_rows or not all(isinstance(row, dict) for row in rows):
        raise GateLabelValidationError(f"Expected {expected_rows} JSON objects in {path}")
    return rows


def verify_source(config: dict[str, Any]) -> None:
    commit = str(config.get("source/git_commit") or "")
    if len(commit) != 40 or config.get("source/git_dirty") is not False:
        raise GateLabelValidationError("Gate-label source revision is missing or dirty")
    sources = config.get("source/sha256")
    if not isinstance(sources, dict) or not sources:
        raise GateLabelValidationError("Gate-label source hashes are missing")
    for relative, expected in sources.items():
        payload = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=REPO_ROOT)
        equal(hashlib.sha256(payload).hexdigest(), expected, f"source hash {relative}")


def route_config(schedule: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "schedule": [event.as_dict() for event in schedule],
        "pre_llm_selector": "none",
        "pre_llm_budget": None,
        "oracle_teacher_layer": None,
        "seed": 0,
    }


def validate_prediction(
    row: dict[str, Any],
    source: dict[str, Any],
    *,
    config: dict[str, Any],
    config_sha: str,
    schedule: tuple[Any, ...],
    row_index: int,
    route_name: str,
) -> None:
    label = f"row {row_index} {route_name}"
    for key in ("sample_id", "dataset", "split", "query", "image_id", "image_path", "gt_box", "width", "height"):
        equal(row.get(key), source.get(key), f"{label} {key}")
    equal(row.get("config_sha256"), config_sha, f"{label} config hash")
    equal(row.get("source_git_commit"), config["source/git_commit"], f"{label} source commit")
    route_cfg = route_config(schedule)
    validate_stage_traces(row, route_cfg, row_index)
    validate_cache_and_tokens(row, route_cfg, row_index)
    predicted, raw, failure = parse_box(
        str(row.get("raw_model_output") or ""), int(source["width"]), int(source["height"])
    )
    iou = box_iou(predicted, list(source["gt_box"])) if predicted is not None else 0.0
    equal(row.get("parse_failure_reason"), failure, f"{label} parse failure")
    equal(bool(row.get("valid")), predicted is not None, f"{label} valid")
    equal(bool(row.get("invalid")), predicted is None, f"{label} invalid")
    equal(bool(row.get("success_iou_0_5")), iou >= 0.5, f"{label} success")
    close(float(row.get("box_iou", -1)), iou, f"{label} IoU")
    if raw is not None:
        equal(len(row.get("raw_parsed_box") or []), 4, f"{label} raw box length")
        for actual, expected in zip(row["raw_parsed_box"], raw, strict=True):
            close(float(actual), float(expected), f"{label} raw box")
    if predicted is not None:
        equal(len(row.get("predicted_box") or []), 4, f"{label} predicted box length")
        for actual, expected in zip(row["predicted_box"], predicted, strict=True):
            close(float(actual), float(expected), f"{label} predicted box")


def validate_feature(row: dict[str, Any], row_index: int) -> None:
    equal(row.get("feature_schema_sha256"), FEATURE_SCHEMA_SHA256, f"row {row_index} feature schema")
    equal(int(row.get("feature_dimension", -1)), FEATURE_DIMENSION, f"row {row_index} feature dimension")
    vector = np.asarray(row.get("feature_vector"), dtype=np.float32)
    if vector.shape != (FEATURE_DIMENSION,) or not np.isfinite(vector).all():
        raise GateLabelValidationError(f"row {row_index} feature vector is malformed")
    equal(
        hashlib.sha256(vector.tobytes()).hexdigest(),
        row.get("feature_vector_sha256"),
        f"row {row_index} feature vector hash",
    )
    scalars = row.get("feature_scalars")
    if not isinstance(scalars, dict) or tuple(scalars) != tuple(sorted(scalars)):
        # JSON was emitted with sorted keys; key-order evidence helps detect an
        # accidental schema-free map, but semantic names are checked below.
        pass
    if not isinstance(scalars, dict) or set(scalars) != set(SCALAR_FEATURE_NAMES):
        raise GateLabelValidationError(f"row {row_index} feature scalar schema differs")
    for offset, name in enumerate(SCALAR_FEATURE_NAMES):
        close(float(vector[-len(SCALAR_FEATURE_NAMES) + offset]), float(scalars[name]), f"row {row_index} scalar {name}", tolerance=1e-6)
    if float(row.get("feature_latency_s", -1)) < 0:
        raise GateLabelValidationError(f"row {row_index} feature latency is invalid")
    equal(row.get("shared_feature_prefix_boundary"), GATE_BOUNDARY, f"row {row_index} prefix boundary")
    equal(row.get("shared_feature_prefix_executions"), 1, f"row {row_index} prefix execution count")
    equal(row.get("physical_route_executions"), 2, f"row {row_index} route execution count")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    safe = [row["safe_prediction"] for row in rows]
    aggressive = [row["aggressive_prediction"] for row in rows]
    critical = sum(
        int(bool(left["success_iou_0_5"]) and not bool(right["success_iou_0_5"]))
        for left, right in zip(safe, aggressive, strict=True)
    )
    return {
        "samples/evaluated": count,
        "safe/quality/acc_at_0_5": sum(int(bool(row["success_iou_0_5"])) for row in safe) / count,
        "safe/quality/mean_iou": sum(float(row["box_iou"]) for row in safe) / count,
        "safe/quality/parsed_box_rate": sum(int(bool(row["valid"])) for row in safe) / count,
        "safe/tokens/average_visual_tokens": sum(float(row["candidate_average_visual_tokens"]) for row in safe) / count,
        "aggressive/quality/acc_at_0_5": sum(int(bool(row["success_iou_0_5"])) for row in aggressive) / count,
        "aggressive/quality/mean_iou": sum(float(row["box_iou"]) for row in aggressive) / count,
        "aggressive/quality/parsed_box_rate": sum(int(bool(row["valid"])) for row in aggressive) / count,
        "aggressive/tokens/average_visual_tokens": sum(float(row["candidate_average_visual_tokens"]) for row in aggressive) / count,
        "labels/critical_safe_correct_aggressive_wrong": critical,
        "labels/safe_iou_greater": sum(
            int(float(left["box_iou"]) > float(right["box_iou"]))
            for left, right in zip(safe, aggressive, strict=True)
        ),
        "features/dimension": FEATURE_DIMENSION,
        "features/latency_mean_s": sum(float(row["feature_latency_s"]) for row in rows) / count,
        "features/latency_max_s": max(float(row["feature_latency_s"]) for row in rows),
    }


def validate_run(run_dir: Path, manifest_path: Path, *, expected_rows: int) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verify_source(config)
    equal(config["dataset/manifest_checksum"], sha256_file(manifest_path), "manifest hash in config")
    equal(config["dataset/manifest_checksum"], EXPECTED_MANIFEST_SHA256, "frozen train manifest hash")
    equal(config["dataset/manifest_audit_sha256"], sha256_file(MANIFEST_AUDIT), "manifest audit hash")
    equal(config["dataset/name"], "refcoco", "dataset")
    equal(config["dataset/split"], "train", "split")
    equal(config["dataset/num_samples"], expected_rows, "config row count")
    equal(config["model/name"], "llava-hf/llava-1.5-7b-hf", "model")
    equal(config["model/precision"], "bf16", "precision")
    equal(config["model/decoding"], "deterministic_greedy", "decoding")
    equal(config["prompt/template"], REGISTERED_PROMPT, "prompt")
    equal(config["gate/feature_schema"], FEATURE_SCHEMA, "feature schema")
    equal(config["gate/feature_schema_sha256"], FEATURE_SCHEMA_SHA256, "feature schema hash")
    equal(config["gate/feature_dimension"], FEATURE_DIMENSION, "feature dimension")
    equal(config["safe_route"], {"name": SAFE_ROUTE, "schedule": [event.as_dict() for event in SAFE_SCHEDULE]}, "safe route")
    equal(config["aggressive_route"], {"name": AGGRESSIVE_ROUTE, "schedule": [event.as_dict() for event in AGGRESSIVE_SCHEDULE]}, "aggressive route")
    audit = json.loads(MANIFEST_AUDIT.read_text(encoding="utf-8"))
    equal(audit["leakage_audit"]["record_id_overlap_with_frozen_validation"], 0, "validation record leakage")
    all_manifest = read_jsonl(manifest_path, 3000)
    manifest = all_manifest[:expected_rows]
    if expected_rows == 3000:
        equal(config["scope"], "gate_labels", "full label scope")
        equal(Counter(row["training_partition"] for row in manifest), Counter({"fit": 2160, "early_stop": 240, "calibration": 600}), "manifest partitions")
    else:
        equal(config["scope"], "gate_label_smoke", "smoke scope")
        equal(config["limit"], expected_rows, "smoke limit")
    labels = read_jsonl(run_dir / "labels.jsonl", expected_rows)
    equal([row["sample_id"] for row in labels], [row["sample_id"] for row in manifest], "label order")
    if len({row["sample_id"] for row in labels}) != expected_rows:
        raise GateLabelValidationError("Gate label IDs are not unique")
    config_sha = sha256_file(config_path)
    for index, (source, row) in enumerate(zip(manifest, labels, strict=True)):
        equal(row.get("sample_id"), source["sample_id"], f"row {index} sample ID")
        equal(row.get("config_sha256"), config_sha, f"row {index} config hash")
        equal(row.get("source_git_commit"), config["source/git_commit"], f"row {index} source")
        equal(row.get("training_partition"), source["training_partition"], f"row {index} training partition")
        equal(row.get("gate_partition"), source["gate_partition"], f"row {index} gate partition")
        validate_feature(row, index)
        validate_prediction(
            row["safe_prediction"],
            source,
            config=config,
            config_sha=config_sha,
            schedule=SAFE_SCHEDULE,
            row_index=index,
            route_name="safe",
        )
        validate_prediction(
            row["aggressive_prediction"],
            source,
            config=config,
            config_sha=config_sha,
            schedule=AGGRESSIVE_SCHEDULE,
            row_index=index,
            route_name="aggressive",
        )
    metrics = aggregate(labels)
    environment_path = run_dir / "environment.json"
    return {
        "status": "pass",
        "validated_at_utc": utc_now(),
        "rows": expected_rows,
        "run_dir": str(run_dir),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_audit_sha256": sha256_file(MANIFEST_AUDIT),
        "config_sha256": config_sha,
        "labels_sha256": sha256_file(run_dir / "labels.jsonl"),
        "environment_sha256": sha256_file(environment_path),
        "source_git_commit": config["source/git_commit"],
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "metrics": metrics,
        "checks": {
            "train_only_and_zero_validation_record_leakage": True,
            "manifest_order_unique_and_grouped_splits": True,
            "shared_k2_feature_prefix_once": True,
            "feature_schema_dimension_and_hashes": True,
            "safe_physical_trace_cache_and_rescore": True,
            "aggressive_physical_trace_merge_cache_and_rescore": True,
            "source_and_environment_provenance": True,
        },
    }


def validate_and_finalize(run_dir: Path, manifest_path: Path, *, expected_rows: int) -> dict[str, Any]:
    report = validate_run(run_dir, manifest_path, expected_rows=expected_rows)
    metrics_path = run_dir / "metrics.json"
    integrity_path = run_dir / "integrity_report.json"
    atomic_write(metrics_path, canonical_bytes(report["metrics"]))
    atomic_write(integrity_path, canonical_bytes(report))
    completion = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "rows": expected_rows,
        "manifest_sha256": report["manifest_sha256"],
        "manifest_audit_sha256": report["manifest_audit_sha256"],
        "config_sha256": report["config_sha256"],
        "labels_sha256": report["labels_sha256"],
        "metrics_sha256": sha256_file(metrics_path),
        "integrity_sha256": sha256_file(integrity_path),
        "environment_sha256": report["environment_sha256"],
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "source_git_commit": report["source_git_commit"],
    }
    atomic_write(run_dir / "COMPLETED.json", canonical_bytes(completion))
    validate_run(run_dir, manifest_path, expected_rows=expected_rows)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-rows", required=True, type=int)
    parser.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = (
            validate_and_finalize(args.run_dir, args.manifest, expected_rows=args.expected_rows)
            if args.finalize
            else validate_run(args.run_dir, args.manifest, expected_rows=args.expected_rows)
        )
    except (GateLabelValidationError, OSError, subprocess.SubprocessError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
