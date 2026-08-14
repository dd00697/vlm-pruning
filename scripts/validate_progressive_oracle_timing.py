#!/usr/bin/env python
"""Independently validate and finalize paired exploratory timing artifacts."""

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
for path in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_progressive_oracle_refcoco import REGISTERED_PROMPT  # noqa: E402
from run_progressive_oracle_timing import aggregate  # noqa: E402
from validate_progressive_oracle_run import (  # noqa: E402
    validate_cache_and_tokens,
    validate_stage_traces,
)
from vlm_pruning.artifacts import sha256_file  # noqa: E402
from vlm_pruning.llava_hooks import average_visual_tokens, realized_visual_trace, validate_schedule  # noqa: E402


EXPECTED_MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"


class TimingValidationError(RuntimeError):
    """Raised when a paired timing artifact violates the registered contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_jsonl(path: Path, expected: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise TimingValidationError(f"Missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise TimingValidationError(f"Blank line {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TimingValidationError(f"Non-object row {path}:{line_number}")
            rows.append(row)
    if len(rows) != expected:
        raise TimingValidationError(f"Expected {expected} rows in {path}, found {len(rows)}")
    return rows


def equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise TimingValidationError(f"{label}: expected {expected!r}, found {actual!r}")


def close(actual: Any, expected: Any, label: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise TimingValidationError(f"{label}: expected {expected!r}, found {actual!r}")


def verify_source(config: dict[str, Any]) -> None:
    commit = str(config.get("source/git_commit") or "")
    if len(commit) != 40 or config.get("source/git_dirty") is not False:
        raise TimingValidationError("Timing source revision is missing, abbreviated, or dirty")
    sources = config.get("source/sha256")
    if not isinstance(sources, dict) or not sources:
        raise TimingValidationError("Timing source hashes are absent")
    for relative, expected in sources.items():
        try:
            payload = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=REPO_ROOT)
        except subprocess.CalledProcessError as exc:
            raise TimingValidationError(f"Cannot recover {relative} at {commit}") from exc
        equal(hashlib.sha256(payload).hexdigest(), expected, f"source hash {relative}")


def quality_config(schedule: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schedule": schedule,
        "pre_llm_selector": "none",
        "pre_llm_budget": None,
        "oracle_teacher_layer": None,
        "seed": 0,
    }


def expected_segments(schedule: list[dict[str, Any]]) -> list[tuple[int, int]]:
    boundaries = [0] + [int(event["boundary"]) for event in schedule] + [32]
    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


def validate_variant(
    row: dict[str, Any],
    *,
    variant: str,
    schedule: list[dict[str, Any]],
    config_sha: str,
    source_commit: str,
    row_index: int,
) -> None:
    label = f"row {row_index} {variant}"
    equal(row.get("variant"), variant, f"{label} variant")
    equal(row.get("config_sha256"), config_sha, f"{label} config hash")
    equal(row.get("source_git_commit"), source_commit, f"{label} source commit")
    if not isinstance(row.get("raw_model_output"), str):
        raise TimingValidationError(f"{label} raw output is absent")
    generated = int(row.get("generated_token_count", 0))
    if not 1 <= generated <= 48:
        raise TimingValidationError(f"{label} generated-token count {generated} is invalid")
    required_nonnegative = (
        "vision_encoder_latency_s", "projector_latency_s", "llm_prefill_latency_s",
        "selector_total_latency_s", "decode_latency_s", "decode_time_per_generated_token_s",
        "end_to_end_latency_s", "prompt_kv_total_bytes", "final_kv_total_bytes",
        "peak_allocated_gpu_bytes", "peak_reserved_gpu_bytes",
    )
    for field in required_nonnegative:
        value = row.get(field)
        if value is None or not math.isfinite(float(value)) or float(value) < 0:
            raise TimingValidationError(f"{label} invalid timing/memory field {field}={value!r}")
    close(
        row["decode_time_per_generated_token_s"],
        float(row["decode_latency_s"]) / generated,
        f"{label} decode time/token",
    )
    stage_latencies = row.get("selector_stage_latencies")
    if not isinstance(stage_latencies, list) or len(stage_latencies) != len(schedule):
        raise TimingValidationError(f"{label} selector-stage timing is incomplete")
    close(
        row["selector_total_latency_s"],
        sum(float(stage["latency_s"]) for stage in stage_latencies),
        f"{label} selector total",
    )
    for index, (stage, event) in enumerate(zip(stage_latencies, schedule, strict=True)):
        equal(int(stage.get("stage_index", -1)), index, f"{label} selector stage index")
        equal(int(stage.get("boundary", -1)), int(event["boundary"]), f"{label} selector boundary")
        equal(stage.get("selector"), event["selector"], f"{label} selector name")
        if float(stage.get("latency_s", -1)) < 0:
            raise TimingValidationError(f"{label} negative selector timing")

    helper_config = quality_config(schedule)
    try:
        validate_stage_traces(row, helper_config, row_index)
        validate_cache_and_tokens(row, helper_config, row_index)
    except Exception as exc:
        raise TimingValidationError(f"{label} pruning/cache evidence failed: {exc}") from exc
    expected_trace = realized_visual_trace(validate_schedule(schedule))
    equal(row.get("realized_visual_tokens_by_layer"), expected_trace, f"{label} visual trace")
    close(row.get("candidate_average_visual_tokens"), average_visual_tokens(expected_trace), f"{label} average tokens")

    segments = expected_segments(schedule)
    for key, total_key in (("prompt_kv_by_segment", "prompt_kv_total_bytes"), ("final_kv_by_segment", "final_kv_total_bytes")):
        values = row.get(key)
        if not isinstance(values, list) or len(values) != len(segments):
            raise TimingValidationError(f"{label} {key} is incomplete")
        for index, (value, (start, end)) in enumerate(zip(values, segments, strict=True)):
            equal(int(value.get("segment_index", -1)), index, f"{label} {key} index")
            equal(int(value.get("start_layer", -1)), start, f"{label} {key} start")
            equal(int(value.get("end_layer", -1)), end, f"{label} {key} end")
            if int(value.get("prompt_length", -1)) <= 0 or int(value.get("kv_bytes", -1)) <= 0:
                raise TimingValidationError(f"{label} {key} contains invalid storage evidence")
        equal(sum(int(value["kv_bytes"]) for value in values), int(row[total_key]), f"{label} {key} total")


def validate_timing(run_dir: Path, manifest_path: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    timing_path = run_dir / "per_sample_timing.jsonl"
    environment_path = run_dir / "environment.json"
    if not config_path.is_file() or not environment_path.is_file():
        raise TimingValidationError("Timing config/environment is incomplete")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = read_jsonl(manifest_path, 1000)
    rows = read_jsonl(timing_path, 1000)
    exact = {
        "scope": "exploratory_timing",
        "matched_full_reference_internal": True,
        "pair_order": "alternating_full_first_by_zero_based_manifest_index",
        "timing_implementation": "paired-progressive-cuda-sync-v1",
        "model/name": "llava-hf/llava-1.5-7b-hf",
        "model/precision": "bf16",
        "model/decoding": "deterministic_greedy",
        "model/max_new_tokens": 48,
        "dataset/name": "refcoco",
        "dataset/split": "val",
        "dataset/num_samples": 1000,
        "dataset/manifest_checksum": EXPECTED_MANIFEST_SHA256,
        "prompt/template": REGISTERED_PROMPT,
        "position_mode": "gap_preserving",
        "original_visual_tokens": 576,
        "grid": 24,
        "full_schedule": [],
    }
    for key, expected in exact.items():
        equal(config.get(key), expected, f"config {key}")
    if int(config.get("warmup_examples_per_path", -1)) < 20:
        raise TimingValidationError("Timing warm-up is below 20 examples per path")
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise TimingValidationError("Timing manifest checksum differs from frozen discovery subset")
    manifest_name = str(config.get("dataset/manifest") or "").lower()
    if "testa" in manifest_name or "testb" in manifest_name:
        raise TimingValidationError("Forbidden testA/testB manifest in timing config")
    if any(str(row.get("dataset") or "").lower() != "refcoco" or str(row.get("split") or "").lower() != "val" for row in manifest):
        raise TimingValidationError("Timing manifest contains a forbidden dataset/split")
    verify_source(config)
    candidate_schedule = [event.as_dict() for event in validate_schedule(config.get("candidate_schedule") or [])]
    config_sha = sha256_file(config_path)
    manifest_ids = [str(row["sample_id"]) for row in manifest]
    row_ids = [str(row.get("sample_id") or "") for row in rows]
    equal(row_ids, manifest_ids, "timing manifest order")
    if len(set(row_ids)) != 1000:
        raise TimingValidationError("Timing sample IDs are not unique")
    for index, row in enumerate(rows):
        equal(int(row.get("manifest_index", -1)), index, f"row {index} manifest index")
        order = ["full", "candidate"] if index % 2 == 0 else ["candidate", "full"]
        equal(row.get("pair_order"), order, f"row {index} pair order")
        equal(row.get("config_sha256"), config_sha, f"row {index} config hash")
        equal(row.get("source_git_commit"), config["source/git_commit"], f"row {index} source")
        validate_variant(
            row["full"], variant="full", schedule=[], config_sha=config_sha,
            source_commit=config["source/git_commit"], row_index=index,
        )
        validate_variant(
            row["candidate"], variant="candidate", schedule=candidate_schedule, config_sha=config_sha,
            source_commit=config["source/git_commit"], row_index=index,
        )
    metrics = aggregate(rows)
    return {
        "status": "pass",
        "validated_at_utc": utc_now(),
        "rows": 1000,
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": config_sha,
        "timing_sha256": sha256_file(timing_path),
        "source_git_commit": config["source/git_commit"],
        "metrics": metrics,
        "checks": {
            "manifest_order_unique": True,
            "validation_only_no_test_splits": True,
            "source_and_protocol": True,
            "warmup_and_alternating_pair_order": True,
            "matched_full_reference": True,
            "cuda_synchronised_component_fields": True,
            "stage_and_segment_storage": True,
            "token_trace_positions_and_caches": True,
        },
        "issues": [],
    }


def validate_and_finalize_timing(run_dir: Path, manifest_path: Path) -> dict[str, Any]:
    report = validate_timing(run_dir, manifest_path)
    metrics_path = run_dir / "metrics.json"
    integrity_path = run_dir / "integrity_report.json"
    completion_path = run_dir / "COMPLETED.json"
    atomic_write(metrics_path, canonical_json_bytes(report["metrics"]))
    atomic_write(integrity_path, canonical_json_bytes(report))
    completion = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "rows": 1000,
        "manifest_sha256": report["manifest_sha256"],
        "source_git_commit": report["source_git_commit"],
        "config_sha256": sha256_file(run_dir / "config.json"),
        "timing_sha256": sha256_file(run_dir / "per_sample_timing.jsonl"),
        "metrics_sha256": sha256_file(metrics_path),
        "integrity_sha256": sha256_file(integrity_path),
        "environment_sha256": sha256_file(run_dir / "environment.json"),
    }
    atomic_write(completion_path, canonical_json_bytes(completion))
    validate_timing(run_dir, manifest_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    try:
        report = (
            validate_and_finalize_timing(args.run_dir, args.manifest)
            if args.finalize
            else validate_timing(args.run_dir, args.manifest)
        )
    except (TimingValidationError, OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
