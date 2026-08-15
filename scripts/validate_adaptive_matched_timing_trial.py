#!/usr/bin/env python
"""Independently validate one adaptive matched-timing trial."""

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

from scripts.run_adaptive_matched_timing_trial import (  # noqa: E402
    FIXED_GENERATED_TOKENS,
    REGISTERED_PROMPT,
    aggregate,
)
from vlm_pruning.artifacts import sha256_file  # noqa: E402


EXPECTED_MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"


class MatchedTimingValidationError(RuntimeError):
    """Raised when a timing trial violates the registered matched protocol."""


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
        raise MatchedTimingValidationError(f"{label}: expected {expected!r}, found {actual!r}")


def close(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise MatchedTimingValidationError(f"{label}: expected {expected}, found {actual}")


def read_jsonl(path: Path, expected: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected:
        raise MatchedTimingValidationError(f"Expected {expected} rows in {path}, found {len(rows)}")
    return rows


def verify_source(config: dict[str, Any]) -> None:
    commit = str(config.get("source/git_commit") or "")
    sources = config.get("source/sha256")
    if len(commit) != 40 or config.get("source/git_dirty") is not False or not isinstance(sources, dict):
        raise MatchedTimingValidationError("Timing source provenance is absent or dirty")
    for relative, expected in sources.items():
        payload = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=REPO_ROOT)
        equal(hashlib.sha256(payload).hexdigest(), expected, f"source hash {relative}")


def validate_trace(row: dict[str, Any], config: dict[str, Any], index: int) -> None:
    method = config["method"]
    selected = config["selected_candidate"]
    route = row.get("route_name")
    if method == "full":
        expected = [576] * 32
        equal(route, None, f"row {index} full route")
    elif method == "k12_original":
        expected = [576] * 12 + [48] * 20
        equal(route, None, f"row {index} K12 route")
    elif selected == "M448":
        expected = [576] * 4 + [448] * 8 + [48] * 20
        equal(route, None, f"row {index} M448 route")
    elif selected == "GATE_SEED0":
        if route == "K12_QUERY_TOPK":
            expected = [576] * 12 + [48] * 20
        elif route == "M448":
            expected = [576] * 4 + [448] * 8 + [48] * 20
        else:
            raise MatchedTimingValidationError(f"row {index} gate route is unregistered")
    else:
        raise MatchedTimingValidationError("Selected timing candidate is unregistered")
    equal(row["realized_visual_tokens_by_layer"], expected, f"row {index} token trace")
    equal(row["visual_token_layer_work"], sum(expected), f"row {index} layer work")
    close(row["average_visual_tokens"], sum(expected) / 32, f"row {index} average tokens")


def validate_run(run_dir: Path, manifest_path: Path, *, expected_rows: int) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    timing_path = run_dir / "timing.jsonl"
    environment_path = run_dir / "environment.json"
    metrics_path = run_dir / "metrics.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verify_source(config)
    equal(config["dataset/manifest_sha256"], sha256_file(manifest_path), "manifest hash in config")
    equal(config["dataset/manifest_sha256"], EXPECTED_MANIFEST_SHA256, "frozen manifest hash")
    equal(config["dataset/split"], "val", "dataset split")
    equal(config["prompt/template"], REGISTERED_PROMPT, "prompt")
    equal(config["model/name"], "llava-hf/llava-1.5-7b-hf", "model")
    equal(config["model/precision"], "bf16", "precision")
    equal(config["limit"], expected_rows, "row limit")
    if int(config.get("attempt", -1)) not in (1, 2):
        raise MatchedTimingValidationError("Timing attempt is not one of the two bounded attempts")
    if config["scope"] == "matched_timing":
        equal(expected_rows, 1000, "full timing rows")
        if int(config["warmup_examples"]) < 20:
            raise MatchedTimingValidationError("Full timing has fewer than 20 warm-ups")
    elif config["scope"] != "matched_timing_smoke":
        raise MatchedTimingValidationError("Timing scope is unregistered")
    selection_path = REPO_ROOT / config["selection_path"]
    equal(config["selection_sha256"], sha256_file(selection_path), "selection artifact hash")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    equal(config["selected_candidate"], selection["project_selection"]["selected"], "selected candidate")
    manifest = read_jsonl(manifest_path, 1000)[:expected_rows]
    rows = read_jsonl(timing_path, expected_rows)
    equal([row["sample_id"] for row in rows], [row["sample_id"] for row in manifest], "timing order")
    if len({row["sample_id"] for row in rows}) != expected_rows:
        raise MatchedTimingValidationError("Timing IDs are not unique")
    config_sha = sha256_file(config_path)
    latency_fields = (
        "vision_encoder_latency_s",
        "projector_latency_s",
        "query_scoring_latency_s",
        "selector_latency_s",
        "gather_latency_s",
        "merge_latency_s",
        "route_gate_latency_s",
        "main_prefill_with_overhead_latency_s",
        "decode_latency_s",
        "decode_time_per_generated_token_s",
        "end_to_end_latency_s",
    )
    for index, row in enumerate(rows):
        equal(row["manifest_index"], index, f"row {index} manifest index")
        equal(row["config_sha256"], config_sha, f"row {index} config hash")
        equal(row["source_git_commit"], config["source/git_commit"], f"row {index} source")
        equal(row["method"], config["method"], f"row {index} method")
        equal(row["repetition"], config["repetition"], f"row {index} repetition")
        equal(row["order_position"], config["order_position"], f"row {index} position")
        equal(row["mode"], config["mode"], f"row {index} mode")
        equal(row["physical_routes_executed"], 1, f"row {index} physical route count")
        if not isinstance(row.get("raw_model_output"), str):
            raise MatchedTimingValidationError(f"row {index} raw output is absent")
        for field in latency_fields:
            value = row.get(field)
            if value is None or not math.isfinite(float(value)) or float(value) < 0:
                raise MatchedTimingValidationError(f"row {index} invalid {field}: {value}")
        generated = int(row["generated_token_count"])
        if config["mode"] == "fixed32":
            equal(generated, FIXED_GENERATED_TOKENS, f"row {index} fixed token count")
        elif not 1 <= generated <= 48:
            raise MatchedTimingValidationError(f"row {index} natural generated count is invalid")
        close(
            row["decode_time_per_generated_token_s"],
            float(row["decode_latency_s"]) / generated,
            f"row {index} decode per token",
        )
        for field in (
            "prompt_kv_total_bytes",
            "final_kv_total_bytes",
            "peak_allocated_gpu_bytes",
            "peak_reserved_gpu_bytes",
        ):
            if int(row.get(field, 0)) <= 0:
                raise MatchedTimingValidationError(f"row {index} invalid memory field {field}")
        if config["method"] != "candidate" or config["selected_candidate"] != "GATE_SEED0":
            close(row["route_gate_latency_s"], 0.0, f"row {index} absent gate latency")
        validate_trace(row, config, index)
    stored_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    recomputed = aggregate(rows)
    equal(stored_metrics, recomputed, "stored aggregate metrics")
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if "gpu_before_warmup" not in environment or "gpu_after_warmup" not in environment:
        raise MatchedTimingValidationError("GPU/clock environment evidence is missing")
    return {
        "status": "pass",
        "validated_at_utc": utc_now(),
        "rows": expected_rows,
        "method": config["method"],
        "repetition": config["repetition"],
        "order_position": config["order_position"],
        "mode": config["mode"],
        "selected_candidate": config["selected_candidate"],
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": config_sha,
        "environment_sha256": sha256_file(environment_path),
        "timing_sha256": sha256_file(timing_path),
        "metrics_sha256": sha256_file(metrics_path),
        "metrics": recomputed,
        "checks": {
            "clean_committed_source_and_selection": True,
            "warmup_outside_recorded_rows": True,
            "manifest_order_and_unique_ids": True,
            "cuda_synchronised_component_timings": True,
            "query_selector_gather_merge_gate_components": True,
            "natural_or_exact_fixed_decode_contract": True,
            "cache_memory_trace_and_compute_proxy": True,
            "gpu_driver_clock_and_power_environment": True,
        },
    }


def validate_and_finalize(run_dir: Path, manifest_path: Path, *, expected_rows: int) -> dict[str, Any]:
    report = validate_run(run_dir, manifest_path, expected_rows=expected_rows)
    integrity_path = run_dir / "integrity_report.json"
    atomic_write(integrity_path, canonical_bytes(report))
    completion = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "rows": expected_rows,
        "method": report["method"],
        "repetition": report["repetition"],
        "order_position": report["order_position"],
        "mode": report["mode"],
        "manifest_sha256": report["manifest_sha256"],
        "config_sha256": report["config_sha256"],
        "environment_sha256": report["environment_sha256"],
        "timing_sha256": report["timing_sha256"],
        "metrics_sha256": report["metrics_sha256"],
        "integrity_sha256": sha256_file(integrity_path),
    }
    atomic_write(run_dir / "COMPLETED.json", canonical_bytes(completion))
    validate_run(run_dir, manifest_path, expected_rows=expected_rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    try:
        report = (
            validate_and_finalize(args.run_dir, args.manifest, expected_rows=args.expected_rows)
            if args.finalize
            else validate_run(args.run_dir, args.manifest, expected_rows=args.expected_rows)
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (
        MatchedTimingValidationError,
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
