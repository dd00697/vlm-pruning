#!/usr/bin/env python
"""Independently validate a pinned official Reroute physical artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
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


EXPECTED_MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
EXPECTED_OFFICIAL_COMMIT = "9d422042076de678b2ad67c4a6660e4b97971a1e"
REGISTERED_PROMPT = (
    'Identify the target described by this referring expression: "{query}".\n'
    "Return exactly one bounding box in normalized 0-1 image coordinates as "
    "[x_min, y_min, x_max, y_max]. Do not include any other text."
)


class OfficialValidationError(RuntimeError):
    """Raised when official physical evidence violates its frozen contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def json_checksum(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_jsonl(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise OfficialValidationError(f"Missing JSONL: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected_rows or not all(isinstance(row, dict) for row in rows):
        raise OfficialValidationError(f"Expected {expected_rows} JSON objects in {path}")
    return rows


def equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise OfficialValidationError(f"{label}: expected {expected!r}, found {actual!r}")


def close(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise OfficialValidationError(f"{label}: expected {expected}, found {actual}")


def verify_source(config: dict[str, Any]) -> None:
    commit = str(config.get("source/git_commit") or "")
    if len(commit) != 40 or config.get("source/git_dirty") is not False:
        raise OfficialValidationError("Adapter source revision is missing or dirty")
    for relative, expected in config["source/sha256"].items():
        payload = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=REPO_ROOT)
        equal(hashlib.sha256(payload).hexdigest(), expected, f"adapter source {relative}")
    official = config["official/source"]
    equal(official["commit"], EXPECTED_OFFICIAL_COMMIT, "official source commit")
    lock = json.loads(
        (REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815/official_sources/source_lock.json").read_text(encoding="utf-8")
    )["sources"]["reroute"]
    for field in ("commit", "recursive_source_sha256"):
        equal(official[field], lock[field], f"official source lock {field}")
    equal(official["working_tree_diff_bytes"], 0, "official checkout diff bytes")
    equal(
        official["working_tree_diff_sha256"],
        hashlib.sha256(b"").hexdigest(),
        "official checkout empty diff hash",
    )
    equal(config["command/sha256"], json_checksum(config["command/semantic"]), "semantic command hash")


def validate_trace(row: dict[str, Any], contract: dict[str, Any], row_index: int) -> None:
    equal(row.get("realized_visual_tokens_by_layer"), contract["realized_visual_tokens_by_layer"], f"row {row_index} visual trace")
    close(
        float(row.get("candidate_average_visual_tokens", -1)),
        sum(contract["realized_visual_tokens_by_layer"]) / 32.0,
        f"row {row_index} average visual tokens",
    )
    traces = row.get("official_routing_trace")
    if contract["action"] == "none":
        equal(traces, [], f"row {row_index} full routing trace")
        return
    if not isinstance(traces, list) or len(traces) != len(contract["drop_layers"]):
        raise OfficialValidationError(f"row {row_index} routing-stage count is invalid")
    previous = list(range(576))
    for stage, (trace, layer, budget) in enumerate(
        zip(traces, contract["drop_layers"], contract["budgets"], strict=True)
    ):
        label = f"row {row_index} route stage {stage}"
        equal(int(trace.get("layer", -1)), layer, f"{label} layer")
        equal(trace.get("action"), contract["action"], f"{label} action")
        equal(int(trace.get("requested_budget", -1)), budget, f"{label} requested budget")
        selected = trace.get("selected_original_indices")
        local = trace.get("selected_local_indices")
        if (
            not isinstance(selected, list)
            or not isinstance(local, list)
            or len(selected) != budget
            or selected != sorted(selected)
            or len(set(selected)) != budget
        ):
            raise OfficialValidationError(f"{label} selected indices violate budget/order/uniqueness")
        if contract["action"] == "physical_delete":
            equal(trace.get("incoming_original_indices"), previous, f"{label} incoming subset")
            expected = [previous[index] for index in local]
            equal(selected, expected, f"{label} local-to-original mapping")
            previous = selected
        else:
            equal(trace.get("incoming_original_indices"), list(range(576)), f"{label} recoverable pool")
            equal(selected, local, f"{label} recoverable original mapping")
        checksum = hashlib.sha256(
            json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        equal(trace.get("selected_original_indices_checksum"), checksum, f"{label} checksum")


def validate_run(run_dir: Path, manifest_path: Path, *, expected_rows: int) -> dict[str, Any]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    verify_source(config)
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    equal(environment["official_checkout_commit"], EXPECTED_OFFICIAL_COMMIT, "environment official commit")
    equal(environment["semantic_command_sha256"], config["command/sha256"], "environment command hash")
    equal(config["dataset/manifest_checksum"], sha256_file(manifest_path), "manifest hash in config")
    equal(config["dataset/manifest_checksum"], EXPECTED_MANIFEST_SHA256, "frozen manifest hash")
    equal(config["dataset/num_samples"], expected_rows, "config row count")
    equal(config["model/name"], "llava-hf/llava-1.5-7b-hf", "model")
    equal(config["model/precision"], "bf16", "precision")
    equal(config["model/decoding"], "deterministic_greedy", "decoding")
    equal(config["prompt/template"], REGISTERED_PROMPT, "prompt")
    manifest_all = read_jsonl(manifest_path, 1000)
    manifest = manifest_all[:expected_rows]
    if expected_rows == 1000:
        equal(config["scope"], "official_discovery", "quality scope")
    else:
        equal(config["scope"], "official_parity_smoke", "smoke scope")
        equal(config["limit"], expected_rows, "smoke limit")
    predictions = read_jsonl(run_dir / "predictions.jsonl", expected_rows)
    equal([row["sample_id"] for row in predictions], [row["sample_id"] for row in manifest], "manifest order")
    contract = config["official/method_contract"]
    config_sha = sha256_file(run_dir / "config.json")
    for index, (source, row) in enumerate(zip(manifest, predictions, strict=True)):
        for key in ("sample_id", "query", "image_id", "image_path", "gt_box"):
            equal(row.get(key), source.get(key), f"row {index} {key}")
        equal(row.get("config_sha256"), config_sha, f"row {index} config hash")
        equal(row.get("official_source_commit"), EXPECTED_OFFICIAL_COMMIT, f"row {index} official commit")
        equal(row.get("official_source_recursive_sha256"), config["official/source"]["recursive_source_sha256"], f"row {index} official source hash")
        if contract["action"] != "none":
            modules = row.get("official_algorithm_modules")
            if not isinstance(modules, dict) or set(modules) != {"router", "patching", "dispatcher"}:
                raise OfficialValidationError(f"row {index} official module evidence is incomplete")
            if any("official_sources/checkouts/reroute/" not in value for value in modules.values()):
                raise OfficialValidationError(f"row {index} algorithm module escaped official source")
        validate_trace(row, contract, index)
        predicted, _raw, failure = parse_box(
            str(row.get("raw_model_output") or ""), int(source["width"]), int(source["height"])
        )
        iou = box_iou(predicted, list(source["gt_box"])) if predicted is not None else 0.0
        equal(row.get("parse_failure_reason"), failure, f"row {index} parse failure")
        equal(bool(row.get("valid")), predicted is not None, f"row {index} valid")
        equal(bool(row.get("success_iou_0_5")), iou >= 0.5, f"row {index} Acc")
        close(float(row.get("box_iou", -1)), iou, f"row {index} IoU")
    metrics = {
        "samples/evaluated": expected_rows,
        "quality/acc_at_0_5": sum(int(bool(row["success_iou_0_5"])) for row in predictions) / expected_rows,
        "quality/mean_iou": sum(float(row["box_iou"]) for row in predictions) / expected_rows,
        "quality/parsed_box_rate": sum(int(bool(row["valid"])) for row in predictions) / expected_rows,
        "tokens/candidate_average_visual_tokens": sum(float(row["candidate_average_visual_tokens"]) for row in predictions) / expected_rows,
        "efficiency/peak_allocated_mb": max(float(row["peak_allocated_mb"]) for row in predictions),
        "efficiency/peak_reserved_mb": max(float(row["peak_reserved_mb"]) for row in predictions),
    }
    return {
        "status": "pass",
        "validated_at_utc": utc_now(),
        "rows": expected_rows,
        "run_dir": str(run_dir),
        "config_sha256": config_sha,
        "predictions_sha256": sha256_file(run_dir / "predictions.jsonl"),
        "manifest_sha256": sha256_file(manifest_path),
        "official_source_commit": EXPECTED_OFFICIAL_COMMIT,
        "environment_sha256": sha256_file(run_dir / "environment.json"),
        "metrics": metrics,
        "checks": {
            "manifest_order": True,
            "source_native_modules": contract["action"] != "none",
            "official_config_trace": True,
            "exact_budgets_and_indices": True,
            "raw_outputs_rescored": True,
            "environment_and_command_hashes": True,
        },
    }


def validate_and_finalize(run_dir: Path, manifest_path: Path, *, expected_rows: int) -> dict[str, Any]:
    report = validate_run(run_dir, manifest_path, expected_rows=expected_rows)
    atomic_write(run_dir / "metrics.json", canonical_bytes(report["metrics"]))
    atomic_write(run_dir / "integrity_report.json", canonical_bytes(report))
    completion = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "rows": expected_rows,
        "manifest_sha256": report["manifest_sha256"],
        "config_sha256": report["config_sha256"],
        "predictions_sha256": report["predictions_sha256"],
        "metrics_sha256": sha256_file(run_dir / "metrics.json"),
        "integrity_sha256": sha256_file(run_dir / "integrity_report.json"),
        "environment_sha256": report["environment_sha256"],
        "official_source_commit": EXPECTED_OFFICIAL_COMMIT,
    }
    atomic_write(run_dir / "COMPLETED.json", canonical_bytes(completion))
    validate_run(run_dir, manifest_path, expected_rows=expected_rows)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
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
    except (OfficialValidationError, OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
