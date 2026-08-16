#!/usr/bin/env python
"""Independently validate one hardware-aware final matched-timing trial."""

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
for entry in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from run_hardware_matched_timing_trial import (  # noqa: E402
    FIXED32_METHODS,
    FIXED32_ORDER_PLAN,
    FIXED_CONTINUATION_TOKEN_IDS,
    METHODS,
    ORDER_PLAN,
    REGISTERED_PROMPT,
    aggregate,
)
from vlm_pruning.artifacts import sha256_file  # noqa: E402


EXPECTED_MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
REFERENCE_SOURCES = {
    "K12_ORIGINAL": (
        Path("experiments/adaptive_efficiency_refcoco_val_20260815/matched_timing/natural/")
        / "rep1_pos2_k12_original_attempt1/timing.jsonl",
        "df01a26cceeac1d52b833b85a6101ce71780415dfe997d57069f551c9ca78d6e",
    ),
    "M448_OPT": (
        Path("experiments/adaptive_efficiency_refcoco_val_20260815/matched_timing/natural/")
        / "rep1_pos3_candidate_attempt1/timing.jsonl",
        "ba171c2febf75af3246fad262686c74b2f7b07cfa5ebb5db4c113b7de6e1e34e",
    ),
    "Z_K12_0": (
        Path("experiments/hardware_aware_zero_tail_refcoco_val_20260816/zero_tail/runs/")
        / "z_k12_0_attempt1/predictions.jsonl",
        "a50a8b7d3f9b602a579f04388ac970dd1138fa90a047c4890794e6e568c88cc7",
    ),
    "Z_M448_0": (
        Path("experiments/hardware_aware_zero_tail_refcoco_val_20260816/zero_tail/runs/")
        / "z_m448_0_attempt1/predictions.jsonl",
        "8f7ac62796e4ec6af703b198fc06c204c5c6774677a190cb1f19898b611e3f75",
    ),
}
M448_RETAINED_SOURCE = (
    Path("experiments/adaptive_efficiency_refcoco_val_20260815/runs/")
    / "m448_attempt1/predictions.jsonl",
    "52e375657ba259a8785dcea3804770070e57f16f5ccc682b78f5edd167c9574b",
)
EXPECTED_TRACES = {
    "K12_ORIGINAL": [576] * 12 + [48] * 20,
    "M448_OPT": [576] * 4 + [448] * 8 + [48] * 20,
    "Z_K12_0": [576] * 12 + [48] * 8 + [0] * 12,
    "Z_M448_0": [576] * 4 + [448] * 8 + [48] * 8 + [0] * 12,
}
EXPECTED_STAGES = {
    "K12_ORIGINAL": (),
    "M448_OPT": ((4, 448, "query_topk_merge"), (12, 48, "query_topk")),
    "Z_K12_0": ((12, 48, "k12_original"), (20, 0, "none")),
    "Z_M448_0": (
        (4, 448, "query_topk_merge"),
        (12, 48, "query_topk"),
        (20, 0, "none"),
    ),
}


class HardwareTimingValidationError(RuntimeError):
    """Raised when a final timing artifact violates its frozen contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


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
        raise HardwareTimingValidationError(
            f"{label}: expected {expected!r}, found {actual!r}"
        )


def close(actual: Any, expected: Any, label: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise HardwareTimingValidationError(
            f"{label}: expected {expected!r}, found {actual!r}"
        )


def read_jsonl(path: Path, expected_rows: int | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if expected_rows is not None and len(rows) != expected_rows:
        raise HardwareTimingValidationError(
            f"Expected {expected_rows} rows in {path}, found {len(rows)}"
        )
    return rows


def verify_source(config: dict[str, Any]) -> None:
    commit = str(config.get("source/git_commit") or "")
    sources = config.get("source/sha256")
    if len(commit) != 40 or config.get("source/git_dirty") is not False or not isinstance(sources, dict):
        raise HardwareTimingValidationError("Timing source provenance is absent or dirty")
    for relative, expected in sources.items():
        payload = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=REPO_ROOT)
        equal(hashlib.sha256(payload).hexdigest(), expected, f"source hash {relative}")


def validate_config(config: dict[str, Any], manifest_path: Path, expected_rows: int) -> None:
    verify_source(config)
    equal(config["dataset/manifest_sha256"], sha256_file(manifest_path), "manifest config hash")
    equal(config["dataset/manifest_sha256"], EXPECTED_MANIFEST_SHA256, "frozen manifest hash")
    equal(config["dataset/split"], "val", "dataset split")
    equal(config["prompt/template"], REGISTERED_PROMPT, "prompt")
    equal(config["model/name"], "llava-hf/llava-1.5-7b-hf", "model")
    equal(config["model/precision"], "bf16", "precision")
    equal(config["limit"], expected_rows, "row limit")
    if int(config.get("attempt", 0)) < 1:
        raise HardwareTimingValidationError("Attempt number must be positive")
    scope = config.get("scope")
    if scope == "final_matched_timing":
        equal(expected_rows, 1000, "full timing rows")
        equal(config["warmup_examples"], 20, "full timing warm-ups")
        equal(config["smoke"], False, "full timing smoke flag")
    elif scope == "final_timing_smoke":
        if not 1 <= expected_rows <= 5 or not 0 <= int(config["warmup_examples"]) <= 5:
            raise HardwareTimingValidationError("Smoke size is outside the registered bound")
        equal(config["smoke"], True, "smoke flag")
    else:
        raise HardwareTimingValidationError(f"Unregistered timing scope: {scope}")
    mode = config["mode"]
    plan = ORDER_PLAN if mode == "natural" else FIXED32_ORDER_PLAN
    if mode not in {"natural", "fixed32"}:
        raise HardwareTimingValidationError(f"Unregistered timing mode: {mode}")
    repetition = int(config["repetition"])
    position = int(config["order_position"])
    method = config["method"]
    equal(tuple(plan[repetition])[position - 1], method, "registered method order")
    if method not in METHODS:
        raise HardwareTimingValidationError(f"Unregistered timing method: {method}")
    if mode == "fixed32":
        if method not in FIXED32_METHODS:
            raise HardwareTimingValidationError("Fixed32 method is unregistered")
        equal(
            config["fixed_continuation_token_ids"],
            list(FIXED_CONTINUATION_TOKEN_IDS),
            "fixed continuation",
        )
        equal(config["fixed_generated_tokens"], len(FIXED_CONTINUATION_TOKEN_IDS), "fixed count")
    else:
        equal(config["fixed_continuation_token_ids"], None, "natural continuation")
    for path_key, hash_key in (
        ("m448_opt_selection", "m448_opt_selection_sha256"),
        ("quality_results", "quality_results_sha256"),
    ):
        path = REPO_ROOT / config[path_key]
        equal(sha256_file(path), config[hash_key], f"frozen dependency {path_key}")


def validate_stage_traces(row: dict[str, Any], method: str, row_index: int) -> None:
    stages = row["stage_traces"]
    expected = EXPECTED_STAGES[method]
    equal(len(stages), len(expected), f"row {row_index} stage count")
    incoming: list[int] | None = None
    for stage_index, (stage, (boundary, budget, selector)) in enumerate(zip(stages, expected)):
        equal(stage["stage_index"], stage_index, f"row {row_index} stage index")
        equal(stage["boundary"], boundary, f"row {row_index} stage boundary")
        equal(stage["selector"], selector, f"row {row_index} stage selector")
        equal(stage["requested_budget"], budget, f"row {row_index} requested budget")
        equal(stage["realized_budget"], budget, f"row {row_index} realized budget")
        outgoing = [int(value) for value in stage["outgoing_original_indices"]]
        equal(outgoing, sorted(set(outgoing)), f"row {row_index} raster/unique outgoing")
        equal(len(outgoing), budget, f"row {row_index} outgoing count")
        stage_incoming = [int(value) for value in stage["incoming_original_indices"]]
        if incoming is not None:
            equal(stage_incoming, incoming, f"row {row_index} chained stage indices")
        if not set(outgoing).issubset(stage_incoming):
            raise HardwareTimingValidationError(f"row {row_index} outgoing is not an incoming subset")
        if selector == "none":
            equal(stage.get("score_map_checksum"), None, f"row {row_index} zero-tail score")
            equal(stage.get("score_map_recomputed"), False, f"row {row_index} zero-tail recompute")
            equal(outgoing, [], f"row {row_index} zero-tail output")
        if selector == "query_topk_merge":
            close(stage["merge_input_mass"], 576.0, f"row {row_index} merge input mass", 1e-4)
            close(stage["merge_output_mass"], 576.0, f"row {row_index} merge output mass", 1e-4)
            equal(stage["merge_applied"], True, f"row {row_index} merge flag")
        incoming = outgoing


def validate_reference_semantics(
    rows: list[dict[str, Any]], config: dict[str, Any], expected_rows: int
) -> None:
    relative, expected_hash = REFERENCE_SOURCES[config["method"]]
    path = REPO_ROOT / relative
    equal(sha256_file(path), expected_hash, "semantic reference hash")
    references = read_jsonl(path)[:expected_rows]
    equal(len(references), expected_rows, "semantic reference row count")
    equal(
        [row["sample_id"] for row in rows],
        [row["sample_id"] for row in references],
        "semantic reference order",
    )
    for index, (row, reference) in enumerate(zip(rows, references)):
        equal(row["prompt_kv_total_bytes"], reference["prompt_kv_total_bytes"] if "prompt_kv_total_bytes" in reference else reference["cache/prompt_total_bytes"], f"row {index} prompt KV semantics")
        if config["mode"] == "natural":
            equal(row["raw_model_output"], reference["raw_model_output"], f"row {index} raw output")
            equal(row["generated_token_count"], reference["generated_token_count"], f"row {index} generated count")
            expected_final = reference.get("final_kv_total_bytes", reference.get("cache/final_total_bytes"))
            equal(row["final_kv_total_bytes"], expected_final, f"row {index} final KV semantics")
        if config["method"] != "K12_ORIGINAL":
            reference_retained = reference.get("final_retained_indices")
            if reference_retained is not None:
                equal(row["retained_indices"], reference_retained, f"row {index} retained indices")
    if config["method"] == "M448_OPT":
        retained_relative, retained_hash = M448_RETAINED_SOURCE
        retained_path = REPO_ROOT / retained_relative
        equal(sha256_file(retained_path), retained_hash, "M448 retained-index reference hash")
        retained_references = read_jsonl(retained_path)[:expected_rows]
        equal(
            [row["sample_id"] for row in rows],
            [row["sample_id"] for row in retained_references],
            "M448 retained-index reference order",
        )
        for index, (row, reference) in enumerate(zip(rows, retained_references)):
            equal(
                row["retained_indices"],
                reference["final_retained_indices"],
                f"row {index} M448 retained indices",
            )


def validate_run(run_dir: Path, manifest_path: Path, *, expected_rows: int) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    environment_path = run_dir / "environment.json"
    timing_path = run_dir / "timing.jsonl"
    metrics_path = run_dir / "metrics.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, manifest_path, expected_rows)
    manifest = read_jsonl(manifest_path, 1000)[:expected_rows]
    rows = read_jsonl(timing_path, expected_rows)
    equal([row["sample_id"] for row in rows], [row["sample_id"] for row in manifest], "manifest order")
    if len({row["sample_id"] for row in rows}) != expected_rows:
        raise HardwareTimingValidationError("Timing sample IDs are not unique")
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
        "time_to_first_token_s",
        "decode_latency_s",
        "decode_time_per_generated_token_s",
        "end_to_end_latency_s",
    )
    expected_trace = EXPECTED_TRACES[config["method"]]
    for index, row in enumerate(rows):
        equal(row["manifest_index"], index, f"row {index} manifest index")
        equal(row["config_sha256"], config_sha, f"row {index} config hash")
        equal(row["source_git_commit"], config["source/git_commit"], f"row {index} source")
        for field in ("method", "mode", "repetition", "order_position"):
            equal(row[field], config[field], f"row {index} {field}")
        equal(row["physical_routes_executed"], 1, f"row {index} physical routes")
        equal(row["realized_visual_tokens_by_layer"], expected_trace, f"row {index} token trace")
        equal(row["visual_token_layer_work"], sum(expected_trace), f"row {index} layer work")
        close(row["average_visual_tokens"], sum(expected_trace) / 32, f"row {index} average tokens")
        if not isinstance(row.get("raw_model_output"), str):
            raise HardwareTimingValidationError(f"row {index} raw output is absent")
        for field in latency_fields:
            value = row.get(field)
            if value is None or not math.isfinite(float(value)) or float(value) < 0:
                raise HardwareTimingValidationError(f"row {index} invalid {field}: {value}")
        generated = int(row["generated_token_count"])
        if config["mode"] == "fixed32":
            equal(generated, len(FIXED_CONTINUATION_TOKEN_IDS), f"row {index} fixed count")
            equal(row["decode_step_count"], len(FIXED_CONTINUATION_TOKEN_IDS), f"row {index} fixed steps")
            equal(row["fixed_continuation_token_ids"], list(FIXED_CONTINUATION_TOKEN_IDS), f"row {index} fixed input")
        else:
            if not 1 <= generated <= 48:
                raise HardwareTimingValidationError(f"row {index} natural generated count is invalid")
            equal(row["decode_step_count"], max(generated - 1, 1), f"row {index} decode steps")
            equal(row["fixed_continuation_token_ids"], None, f"row {index} natural fixed input")
        close(
            row["decode_time_per_generated_token_s"],
            float(row["decode_latency_s"]) / int(row["decode_step_count"]),
            f"row {index} decode per step",
        )
        close(row["route_gate_latency_s"], 0.0, f"row {index} route gate")
        for field in (
            "prompt_kv_total_bytes",
            "final_kv_total_bytes",
            "peak_allocated_gpu_bytes",
            "peak_reserved_gpu_bytes",
        ):
            if int(row.get(field, 0)) <= 0:
                raise HardwareTimingValidationError(f"row {index} invalid memory field {field}")
        if int(row["final_kv_total_bytes"]) <= int(row["prompt_kv_total_bytes"]):
            raise HardwareTimingValidationError(f"row {index} generated cache did not grow")
        validate_stage_traces(row, config["method"], index)
    validate_reference_semantics(rows, config, expected_rows)
    recomputed = aggregate(rows)
    recomputed["time_to_first_token_s"] = {
        "mean": sum(float(row["time_to_first_token_s"]) for row in rows) / len(rows)
    }
    stored_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    equal(stored_metrics, recomputed, "stored aggregate metrics")
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if "gpu_before_warmup" not in environment or "gpu_after_warmup" not in environment:
        raise HardwareTimingValidationError("GPU environment evidence is absent")
    profiler_hashes = None
    if config["torch_profiler_capture"]:
        profiler_path = run_dir / "torch_profiler_summary.json"
        trace_path = run_dir / "torch_profiler_trace.json"
        profiler = json.loads(profiler_path.read_text(encoding="utf-8"))
        if int(profiler.get("cuda_kernel_event_count", 0)) <= 0:
            raise HardwareTimingValidationError("Profiler CUDA kernel count is absent")
        equal(profiler["trace_sha256"], sha256_file(trace_path), "profiler trace hash")
        profiler_hashes = {
            "summary_sha256": sha256_file(profiler_path),
            "trace_sha256": sha256_file(trace_path),
        }
    return {
        "status": "pass",
        "validated_at_utc": utc_now(),
        "rows": expected_rows,
        "method": config["method"],
        "mode": config["mode"],
        "repetition": config["repetition"],
        "order_position": config["order_position"],
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": config_sha,
        "environment_sha256": sha256_file(environment_path),
        "timing_sha256": sha256_file(timing_path),
        "metrics_sha256": sha256_file(metrics_path),
        "profiler_hashes": profiler_hashes,
        "metrics": recomputed,
        "checks": {
            "clean_committed_recorded_source": True,
            "frozen_manifest_order_and_unique_ids": True,
            "counterbalanced_order_and_warmups": True,
            "natural_semantics_or_identical_fixed_continuation": True,
            "exact_token_trace_and_zero_tail_no_score": True,
            "cache_growth_and_memory_evidence": True,
            "component_ttft_decode_and_end_to_end_timings": True,
            "profiler_kernel_counts_when_registered": True,
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
        "mode": report["mode"],
        "repetition": report["repetition"],
        "order_position": report["order_position"],
        "manifest_sha256": report["manifest_sha256"],
        "config_sha256": report["config_sha256"],
        "environment_sha256": report["environment_sha256"],
        "timing_sha256": report["timing_sha256"],
        "metrics_sha256": report["metrics_sha256"],
        "profiler_hashes": report["profiler_hashes"],
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
        HardwareTimingValidationError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
        KeyError,
        IndexError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
