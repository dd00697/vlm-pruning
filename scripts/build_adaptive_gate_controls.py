#!/usr/bin/env python
"""Assemble the two registered no-inference adaptive route-gate controls."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from vlm_pruning.artifacts import sha256_file  # noqa: E402


CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
DEFAULT_MANIFEST = REPO_ROOT / (
    "experiments/progressive_oracle_refcoco_val_20260814/"
    "manifests/refcoco_val_discovery1000.jsonl"
)
DEFAULT_PRIMARY_RUN = CAMPAIGN_ROOT / "gate_validation_runs/seed0_full1000_attempt1"
DEFAULT_OUTPUT_ROOT = CAMPAIGN_ROOT / "gate_controls"
REGISTRY_PATH = CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"
M448_PATH = CAMPAIGN_ROOT / "runs/m448_attempt1/predictions.jsonl"
RANDOM_SEED = 20260815


class GateControlBuildError(RuntimeError):
    """Raised when analytical controls cannot be traced to physical sources."""


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


def read_jsonl(path: Path, expected: int | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if expected is not None and len(rows) != expected:
        raise GateControlBuildError(f"Expected {expected} rows in {path}, found {len(rows)}")
    return rows


def indexed_physical_rows(path: Path, wanted_ids: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("rb") as handle:
        for index, raw_line in enumerate(handle):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            sample_id = str(row.get("sample_id"))
            if sample_id not in wanted_ids:
                continue
            if sample_id in result:
                raise GateControlBuildError(f"Duplicate physical source ID {sample_id} in {path}")
            result[sample_id] = {
                "row": row,
                "row_index_zero_based": index,
                "row_sha256": hashlib.sha256(raw_line).hexdigest(),
            }
    missing = sorted(wanted_ids - set(result))
    if missing:
        raise GateControlBuildError(f"Physical source {path} is missing IDs: {missing[:3]}")
    return result


def source_descriptor(
    path: Path, route_name: str, indexed: dict[str, Any], file_sha256: str
) -> dict[str, Any]:
    return {
        "route_name": route_name,
        "path": str(path.resolve().relative_to(REPO_ROOT)),
        "file_sha256": file_sha256,
        "row_index_zero_based": indexed["row_index_zero_based"],
        "row_sha256": indexed["row_sha256"],
        "row_hash_scope": "exact_jsonl_line_bytes_including_newline",
    }


def control_row(
    sample_id: str,
    control_name: str,
    route_name: str,
    indexed: dict[str, Any],
    source_path: Path,
    source_file_sha256: str,
    *,
    privileged: bool,
    rule_evidence: dict[str, Any],
) -> dict[str, Any]:
    source = indexed["row"]
    return {
        "sample_id": sample_id,
        "control_name": control_name,
        "route_choice": route_name,
        "analytical_mixture_no_llava_inference": True,
        "oracle_privileged_non_deployable": privileged,
        "efficiency_claim": not privileged,
        "source": source_descriptor(source_path, route_name, indexed, source_file_sha256),
        "rule_evidence": rule_evidence,
        "raw_model_output": source.get("raw_model_output"),
        "raw_parsed_box": source.get("raw_parsed_box"),
        "predicted_box": source.get("predicted_box"),
        "gt_box": source.get("gt_box"),
        "width": source.get("width"),
        "height": source.get("height"),
        "valid": bool(source.get("valid")),
        "invalid": bool(source.get("invalid")),
        "parse_failure_reason": source.get("parse_failure_reason"),
        "box_iou": float(source.get("box_iou") or 0.0),
        "success_iou_0_5": bool(source.get("success_iou_0_5")),
        "candidate_average_visual_tokens": 246.0 if route_name == "K12_QUERY_TOPK" else 214.0,
    }


def aggregate(rows: list[dict[str, Any]], *, privileged: bool) -> dict[str, Any]:
    count = len(rows)
    aggressive = sum(row["route_choice"] == "M448" for row in rows)
    return {
        "samples/evaluated": count,
        "quality/acc_at_0_5": sum(row["success_iou_0_5"] for row in rows) / count,
        "quality/mean_iou": sum(row["box_iou"] for row in rows) / count,
        "quality/parsed_box_rate": sum(row["valid"] for row in rows) / count,
        "routing/aggressive_rows": aggressive,
        "routing/aggressive_rate": aggressive / count,
        "tokens/analytical_average_visual_tokens": None
        if privileged
        else sum(row["candidate_average_visual_tokens"] for row in rows) / count,
        "efficiency_claim": not privileged,
        "physical_inference_executed": False,
    }


def write_unvalidated_control(run_dir: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if run_dir.exists():
        raise GateControlBuildError(f"Control directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    atomic_write(run_dir / "config.json", canonical_bytes(config))
    atomic_write(
        run_dir / "predictions.jsonl",
        b"".join(
            (json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
            for row in rows
        ),
    )


def run(args: argparse.Namespace) -> int:
    manifest = read_jsonl(args.manifest, 1000)
    manifest_ids = [str(row["sample_id"]) for row in manifest]
    wanted_ids = set(manifest_ids)
    primary = read_jsonl(args.primary_run / "predictions.jsonl", 1000)
    if [str(row["sample_id"]) for row in primary] != manifest_ids:
        raise GateControlBuildError("Primary gate rows do not match the frozen manifest")
    primary_integrity = json.loads((args.primary_run / "integrity_report.json").read_text(encoding="utf-8"))
    if primary_integrity.get("status") != "pass" or int(primary_integrity.get("seed", -1)) != 0:
        raise GateControlBuildError("Primary seed-0 gate is not independently valid")
    aggressive_count = int(primary_integrity["metrics"]["gate/aggressive_rows"])

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    safe_path = REPO_ROOT / registry["references"]["K12_QUERY_TOPK"]["source_path"]
    if sha256_file(safe_path) != registry["references"]["K12_QUERY_TOPK"]["source_sha256"]:
        raise GateControlBuildError("Registered physical K12 source hash differs")
    m448_integrity = json.loads((M448_PATH.parent / "integrity_report.json").read_text(encoding="utf-8"))
    if m448_integrity.get("status") != "pass":
        raise GateControlBuildError("Physical M448 source is not independently valid")
    safe_rows = indexed_physical_rows(safe_path, wanted_ids)
    aggressive_rows = indexed_physical_rows(M448_PATH, wanted_ids)
    safe_file_sha256 = sha256_file(safe_path)
    aggressive_file_sha256 = sha256_file(M448_PATH)
    source_contract = {
        "safe": {
            "route_name": "K12_QUERY_TOPK",
            "path": str(safe_path.relative_to(REPO_ROOT)),
            "sha256": safe_file_sha256,
            "registered_rescore_sha256": registry["references"]["K12_QUERY_TOPK"]["rescored_sha256"],
        },
        "aggressive": {
            "route_name": "M448",
            "path": str(M448_PATH.relative_to(REPO_ROOT)),
            "sha256": aggressive_file_sha256,
            "integrity_sha256": sha256_file(M448_PATH.parent / "integrity_report.json"),
        },
    }
    primary_contract = {
        "path": str(args.primary_run.resolve().relative_to(REPO_ROOT)),
        "predictions_sha256": sha256_file(args.primary_run / "predictions.jsonl"),
        "integrity_sha256": sha256_file(args.primary_run / "integrity_report.json"),
        "completion_sha256": sha256_file(args.primary_run / "COMPLETED.json"),
        "aggressive_rows": aggressive_count,
        "aggressive_rate": aggressive_count / 1000,
    }

    ranked = sorted(
        manifest_ids,
        key=lambda sample_id: hashlib.sha256(f"{RANDOM_SEED}\0{sample_id}".encode("utf-8")).hexdigest(),
    )
    random_aggressive = set(ranked[:aggressive_count])
    random_rows = []
    oracle_rows = []
    for sample_id in manifest_ids:
        random_route = "M448" if sample_id in random_aggressive else "K12_QUERY_TOPK"
        random_indexed = aggressive_rows[sample_id] if random_route == "M448" else safe_rows[sample_id]
        random_source = M448_PATH if random_route == "M448" else safe_path
        random_rows.append(
            control_row(
                sample_id,
                "MATCHED_RATE_RANDOM_SEED0",
                random_route,
                random_indexed,
                random_source,
                aggressive_file_sha256 if random_route == "M448" else safe_file_sha256,
                privileged=False,
                rule_evidence={
                    "seed": RANDOM_SEED,
                    "rank_sha256": hashlib.sha256(f"{RANDOM_SEED}\0{sample_id}".encode("utf-8")).hexdigest(),
                    "selected_aggressive": sample_id in random_aggressive,
                },
            )
        )
        safe_iou = float(safe_rows[sample_id]["row"].get("box_iou") or 0.0)
        aggressive_iou = float(aggressive_rows[sample_id]["row"].get("box_iou") or 0.0)
        oracle_route = "K12_QUERY_TOPK" if safe_iou > aggressive_iou else "M448"
        oracle_indexed = safe_rows[sample_id] if oracle_route == "K12_QUERY_TOPK" else aggressive_rows[sample_id]
        oracle_source = safe_path if oracle_route == "K12_QUERY_TOPK" else M448_PATH
        oracle_rows.append(
            control_row(
                sample_id,
                "PRIVILEGED_GROUND_TRUTH_ORACLE",
                oracle_route,
                oracle_indexed,
                oracle_source,
                safe_file_sha256 if oracle_route == "K12_QUERY_TOPK" else aggressive_file_sha256,
                privileged=True,
                rule_evidence={
                    "safe_iou": safe_iou,
                    "aggressive_iou": aggressive_iou,
                    "rule": "higher_ground_truth_iou_then_aggressive_on_exact_tie",
                },
            )
        )

    common = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "dataset": "refcoco_val_frozen1000",
        "manifest_path": str(args.manifest.resolve().relative_to(REPO_ROOT)),
        "manifest_sha256": sha256_file(args.manifest),
        "rows": 1000,
        "physical_inference_executed": False,
        "source_contract": source_contract,
        "primary_gate_contract": primary_contract,
    }
    random_dir = args.output_root / "random_rate_seed0"
    oracle_dir = args.output_root / "privileged_oracle"
    write_unvalidated_control(
        random_dir,
        {
            **common,
            "control_name": "MATCHED_RATE_RANDOM_SEED0",
            "privileged": False,
            "random_seed": RANDOM_SEED,
            "selection_rule": "exact_primary_aggressive_count_by_smallest_sha256(seed_NUL_sample_id)",
        },
        random_rows,
    )
    write_unvalidated_control(
        oracle_dir,
        {
            **common,
            "control_name": "PRIVILEGED_GROUND_TRUTH_ORACLE",
            "privileged": True,
            "efficiency_claim": False,
            "selection_rule": "higher_ground_truth_iou_then_aggressive_on_exact_tie",
        },
        oracle_rows,
    )
    from scripts.validate_adaptive_gate_controls import validate_and_finalize

    random_report = validate_and_finalize(random_dir, args.manifest, args.primary_run)
    oracle_report = validate_and_finalize(oracle_dir, args.manifest, args.primary_run)
    print(json.dumps({"random": random_report["metrics"], "oracle": oracle_report["metrics"]}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--primary-run", type=Path, default=DEFAULT_PRIMARY_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    try:
        return run(args)
    except (GateControlBuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
