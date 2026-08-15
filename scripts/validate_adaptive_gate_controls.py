#!/usr/bin/env python
"""Independently validate the two analytical adaptive gate controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.init_progressive_oracle_campaign import box_iou, parse_box  # noqa: E402
from vlm_pruning.artifacts import sha256_file  # noqa: E402


CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
DEFAULT_MANIFEST = REPO_ROOT / (
    "experiments/progressive_oracle_refcoco_val_20260814/"
    "manifests/refcoco_val_discovery1000.jsonl"
)
DEFAULT_PRIMARY_RUN = CAMPAIGN_ROOT / "gate_validation_runs/seed0_full1000_attempt1"
RANDOM_SEED = 20260815


class GateControlValidationError(RuntimeError):
    """Raised when an analytical routing control is not exactly traceable."""


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
        raise GateControlValidationError(f"{label}: expected {expected!r}, found {actual!r}")


def close(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise GateControlValidationError(f"{label}: expected {expected}, found {actual}")


def read_jsonl(path: Path, expected: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected:
        raise GateControlValidationError(f"Expected {expected} rows in {path}, found {len(rows)}")
    return rows


def physical_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("rb") as handle:
        for index, raw_line in enumerate(handle):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            sample_id = str(row.get("sample_id"))
            if sample_id in result:
                raise GateControlValidationError(f"Duplicate source ID {sample_id} in {path}")
            result[sample_id] = {
                "row": row,
                "row_index_zero_based": index,
                "row_sha256": hashlib.sha256(raw_line).hexdigest(),
            }
    return result


def independently_rescore(
    row: dict[str, Any],
    label: str,
    *,
    width: int,
    height: int,
    gt_box: list[float],
) -> None:
    predicted, raw, failure = parse_box(
        str(row.get("raw_model_output") or ""), width, height
    )
    iou = box_iou(predicted, gt_box) if predicted is not None else 0.0
    equal(row.get("parse_failure_reason"), failure, f"{label} parse failure")
    equal(bool(row.get("valid")), predicted is not None, f"{label} valid")
    equal(bool(row.get("invalid")), predicted is None, f"{label} invalid")
    equal(bool(row.get("success_iou_0_5")), iou >= 0.5, f"{label} success")
    close(row.get("box_iou"), iou, f"{label} IoU")
    if raw is not None:
        equal(len(row.get("raw_parsed_box") or []), 4, f"{label} raw box length")
        for actual, expected in zip(row["raw_parsed_box"], raw, strict=True):
            close(actual, expected, f"{label} raw box")
    if predicted is not None:
        equal(len(row.get("predicted_box") or []), 4, f"{label} box length")
        for actual, expected in zip(row["predicted_box"], predicted, strict=True):
            close(actual, expected, f"{label} box")


def aggregate(rows: list[dict[str, Any]], privileged: bool) -> dict[str, Any]:
    count = len(rows)
    aggressive = sum(row["route_choice"] == "M448" for row in rows)
    return {
        "samples/evaluated": count,
        "quality/acc_at_0_5": sum(bool(row["success_iou_0_5"]) for row in rows) / count,
        "quality/mean_iou": sum(float(row["box_iou"]) for row in rows) / count,
        "quality/parsed_box_rate": sum(bool(row["valid"]) for row in rows) / count,
        "routing/aggressive_rows": aggressive,
        "routing/aggressive_rate": aggressive / count,
        "tokens/analytical_average_visual_tokens": None
        if privileged
        else sum(float(row["candidate_average_visual_tokens"]) for row in rows) / count,
        "efficiency_claim": not privileged,
        "physical_inference_executed": False,
    }


def validate_run(run_dir: Path, manifest_path: Path, primary_run: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    predictions_path = run_dir / "predictions.jsonl"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    equal(config["manifest_sha256"], sha256_file(manifest_path), "manifest hash")
    equal(config["rows"], 1000, "config rows")
    equal(config["physical_inference_executed"], False, "control inference flag")
    primary = config["primary_gate_contract"]
    equal(primary["path"], str(primary_run.resolve().relative_to(REPO_ROOT)), "primary gate path")
    equal(primary["predictions_sha256"], sha256_file(primary_run / "predictions.jsonl"), "primary predictions hash")
    equal(primary["integrity_sha256"], sha256_file(primary_run / "integrity_report.json"), "primary integrity hash")
    equal(primary["completion_sha256"], sha256_file(primary_run / "COMPLETED.json"), "primary completion hash")
    primary_integrity = json.loads((primary_run / "integrity_report.json").read_text(encoding="utf-8"))
    equal(primary_integrity["status"], "pass", "primary gate integrity")
    equal(primary_integrity["seed"], 0, "primary gate seed")
    aggressive_count = int(primary_integrity["metrics"]["gate/aggressive_rows"])
    equal(primary["aggressive_rows"], aggressive_count, "primary aggressive count")
    close(primary["aggressive_rate"], aggressive_count / 1000, "primary aggressive rate")

    sources: dict[str, tuple[Path, dict[str, dict[str, Any]], str]] = {}
    for side, route_name in (("safe", "K12_QUERY_TOPK"), ("aggressive", "M448")):
        contract = config["source_contract"][side]
        equal(contract["route_name"], route_name, f"{side} route name")
        path = REPO_ROOT / contract["path"]
        equal(contract["sha256"], sha256_file(path), f"{side} source file hash")
        sources[route_name] = (path, physical_index(path), contract["sha256"])

    manifest = read_jsonl(manifest_path, 1000)
    rows = read_jsonl(predictions_path, 1000)
    manifest_ids = [str(row["sample_id"]) for row in manifest]
    equal([str(row["sample_id"]) for row in rows], manifest_ids, "control manifest order")
    if len(set(manifest_ids)) != 1000:
        raise GateControlValidationError("Manifest IDs are not unique")
    control_name = str(config["control_name"])
    privileged = bool(config.get("privileged"))
    if control_name == "MATCHED_RATE_RANDOM_SEED0":
        equal(privileged, False, "random privilege flag")
        equal(config["random_seed"], RANDOM_SEED, "random seed")
        ranked = sorted(
            manifest_ids,
            key=lambda sample_id: hashlib.sha256(f"{RANDOM_SEED}\0{sample_id}".encode("utf-8")).hexdigest(),
        )
        random_aggressive = set(ranked[:aggressive_count])
    elif control_name == "PRIVILEGED_GROUND_TRUTH_ORACLE":
        equal(privileged, True, "oracle privilege flag")
        equal(config.get("efficiency_claim"), False, "oracle efficiency claim")
        random_aggressive = set()
    else:
        raise GateControlValidationError(f"Unregistered control: {control_name}")

    for index, row in enumerate(rows):
        sample_id = manifest_ids[index]
        manifest_row = manifest[index]
        equal(row["control_name"], control_name, f"row {index} control name")
        equal(row["analytical_mixture_no_llava_inference"], True, f"row {index} no-inference flag")
        equal(row["oracle_privileged_non_deployable"], privileged, f"row {index} privileged flag")
        safe_source = sources["K12_QUERY_TOPK"][1].get(sample_id)
        aggressive_source = sources["M448"][1].get(sample_id)
        if safe_source is None or aggressive_source is None:
            raise GateControlValidationError(f"Physical source missing row {sample_id}")
        width = int(manifest_row["width"])
        height = int(manifest_row["height"])
        gt_box = list(manifest_row["gt_box"])
        equal(safe_source["row"].get("gt_box"), gt_box, f"row {index} safe ground truth")
        equal(aggressive_source["row"].get("gt_box"), gt_box, f"row {index} aggressive ground truth")
        independently_rescore(
            safe_source["row"],
            f"row {index} safe source",
            width=width,
            height=height,
            gt_box=gt_box,
        )
        independently_rescore(
            aggressive_source["row"],
            f"row {index} aggressive source",
            width=width,
            height=height,
            gt_box=gt_box,
        )
        if control_name == "MATCHED_RATE_RANDOM_SEED0":
            expected_route = "M448" if sample_id in random_aggressive else "K12_QUERY_TOPK"
            expected_rank = hashlib.sha256(f"{RANDOM_SEED}\0{sample_id}".encode("utf-8")).hexdigest()
            equal(row["rule_evidence"]["rank_sha256"], expected_rank, f"row {index} random rank")
            equal(row["rule_evidence"]["selected_aggressive"], sample_id in random_aggressive, f"row {index} random choice")
        else:
            safe_iou = float(safe_source["row"]["box_iou"])
            aggressive_iou = float(aggressive_source["row"]["box_iou"])
            expected_route = "K12_QUERY_TOPK" if safe_iou > aggressive_iou else "M448"
            close(row["rule_evidence"]["safe_iou"], safe_iou, f"row {index} oracle safe IoU")
            close(row["rule_evidence"]["aggressive_iou"], aggressive_iou, f"row {index} oracle aggressive IoU")
        equal(row["route_choice"], expected_route, f"row {index} route")
        path, source_map, source_file_sha256 = sources[expected_route]
        source = source_map[sample_id]
        descriptor = row["source"]
        equal(descriptor["path"], str(path.relative_to(REPO_ROOT)), f"row {index} source path")
        equal(descriptor["file_sha256"], source_file_sha256, f"row {index} source file hash")
        equal(descriptor["row_index_zero_based"], source["row_index_zero_based"], f"row {index} source index")
        equal(descriptor["row_sha256"], source["row_sha256"], f"row {index} source row hash")
        equal(descriptor["row_hash_scope"], "exact_jsonl_line_bytes_including_newline", f"row {index} hash scope")
        for key in (
            "raw_model_output",
            "raw_parsed_box",
            "predicted_box",
            "gt_box",
            "valid",
            "invalid",
            "parse_failure_reason",
            "box_iou",
            "success_iou_0_5",
        ):
            equal(row[key], source["row"].get(key), f"row {index} copied {key}")
        equal(row["width"], width, f"row {index} manifest width")
        equal(row["height"], height, f"row {index} manifest height")
        expected_tokens = 246.0 if expected_route == "K12_QUERY_TOPK" else 214.0
        close(row["candidate_average_visual_tokens"], expected_tokens, f"row {index} token work")
    metrics = aggregate(rows, privileged)
    if control_name == "MATCHED_RATE_RANDOM_SEED0":
        equal(metrics["routing/aggressive_rows"], aggressive_count, "matched random aggressive count")
    return {
        "status": "pass",
        "validated_at_utc": utc_now(),
        "control_name": control_name,
        "rows": 1000,
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(config_path),
        "predictions_sha256": sha256_file(predictions_path),
        "metrics": metrics,
        "checks": {
            "no_llava_inference": True,
            "exact_manifest_order_and_unique_ids": True,
            "physical_source_file_and_row_hashes": True,
            "physical_sources_independently_rescored": True,
            "registered_route_choice_rule_recomputed": True,
            "matched_primary_aggressive_rate": control_name == "MATCHED_RATE_RANDOM_SEED0",
            "oracle_privilege_and_no_efficiency_claim": control_name == "PRIVILEGED_GROUND_TRUTH_ORACLE",
        },
    }


def validate_and_finalize(run_dir: Path, manifest_path: Path, primary_run: Path) -> dict[str, Any]:
    report = validate_run(run_dir, manifest_path, primary_run)
    metrics_path = run_dir / "metrics.json"
    integrity_path = run_dir / "integrity_report.json"
    atomic_write(metrics_path, canonical_bytes(report["metrics"]))
    atomic_write(integrity_path, canonical_bytes(report))
    completion = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "control_name": report["control_name"],
        "rows": 1000,
        "manifest_sha256": report["manifest_sha256"],
        "config_sha256": report["config_sha256"],
        "predictions_sha256": report["predictions_sha256"],
        "metrics_sha256": sha256_file(metrics_path),
        "integrity_sha256": sha256_file(integrity_path),
    }
    atomic_write(run_dir / "COMPLETED.json", canonical_bytes(completion))
    validate_run(run_dir, manifest_path, primary_run)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--primary-run", type=Path, default=DEFAULT_PRIMARY_RUN)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    try:
        report = (
            validate_and_finalize(args.run_dir, args.manifest, args.primary_run)
            if args.finalize
            else validate_run(args.run_dir, args.manifest, args.primary_run)
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (GateControlValidationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
