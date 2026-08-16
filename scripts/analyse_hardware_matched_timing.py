#!/usr/bin/env python
"""Validate and analyse final natural or fixed32 hardware timing trials."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for entry in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from run_hardware_matched_timing_trial import (  # noqa: E402
    CAMPAIGN_ROOT,
    FIXED32_METHODS,
    FIXED32_ORDER_PLAN,
    METHODS,
    ORDER_PLAN,
)
from validate_hardware_matched_timing_trial import validate_run  # noqa: E402
from vlm_pruning.artifacts import sha256_file, source_provenance  # noqa: E402


MANIFEST = REPO_ROOT / (
    "experiments/progressive_oracle_refcoco_val_20260814/manifests/"
    "refcoco_val_discovery1000.jsonl"
)
QUALITY_RESULTS = CAMPAIGN_ROOT / "zero_tail/analysis/QUALITY_RESULTS.json"
TIMING_ROOT = CAMPAIGN_ROOT / "matched_timing"
ANALYSIS_ROOT = TIMING_ROOT / "analysis"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260816
CONTRAST_FIELDS = (
    "vision_encoder_latency_s",
    "projector_latency_s",
    "query_scoring_latency_s",
    "selector_latency_s",
    "gather_latency_s",
    "merge_latency_s",
    "main_prefill_with_overhead_latency_s",
    "time_to_first_token_s",
    "decode_latency_s",
    "decode_time_per_generated_token_s",
    "end_to_end_latency_s",
    "prompt_kv_total_bytes",
    "final_kv_total_bytes",
)
SUMMARY_FIELDS = CONTRAST_FIELDS + (
    "generated_token_count",
    "visual_token_layer_work",
    "average_visual_tokens",
)
SOURCE_PATHS = (
    Path("scripts/analyse_hardware_matched_timing.py"),
    Path("scripts/run_hardware_matched_timing_trial.py"),
    Path("scripts/validate_hardware_matched_timing_trial.py"),
)


class HardwareTimingAnalysisError(RuntimeError):
    """Raised when final timing analysis is incomplete or inconsistent."""


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_completion(run_dir: Path) -> dict[str, Any]:
    completion = read_json(run_dir / "COMPLETED.json")
    if completion.get("status") != "complete":
        raise HardwareTimingAnalysisError(f"Incomplete timing sentinel: {run_dir}")
    for name, key in (
        ("config.json", "config_sha256"),
        ("environment.json", "environment_sha256"),
        ("timing.jsonl", "timing_sha256"),
        ("metrics.json", "metrics_sha256"),
        ("integrity_report.json", "integrity_sha256"),
    ):
        if sha256_file(run_dir / name) != completion[key]:
            raise HardwareTimingAnalysisError(f"Completion hash mismatch for {run_dir / name}")
    for name, expected in (completion.get("profiler_hashes") or {}).items():
        filename = "torch_profiler_summary.json" if name == "summary_sha256" else "torch_profiler_trace.json"
        if sha256_file(run_dir / filename) != expected:
            raise HardwareTimingAnalysisError(f"Profiler hash mismatch: {run_dir / filename}")
    return completion


def accepted_run(mode: str, repetition: int, position: int, method: str) -> Path:
    stem = f"rep{repetition}_pos{position}_{method.lower()}_attempt"
    candidates = sorted((TIMING_ROOT / mode).glob(f"{stem}*"))
    completed = [path for path in candidates if (path / "COMPLETED.json").exists()]
    if len(completed) != 1:
        raise HardwareTimingAnalysisError(
            f"Expected one completed run for {mode} rep{repetition} pos{position} {method}, found {len(completed)}"
        )
    for path in candidates:
        if path != completed[0] and not (path / "REJECTION.json").exists():
            raise HardwareTimingAnalysisError(f"Unresolved timing attempt: {path}")
    return completed[0]


def mean_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"rows": len(rows)}
    for field in SUMMARY_FIELDS:
        values = [float(row[field]) for row in rows]
        result[field] = {
            "mean": float(statistics.fmean(values)),
            "sample_sd": float(statistics.stdev(values)),
            "median": float(statistics.median(values)),
            "p95": float(np.quantile(np.asarray(values), 0.95)),
        }
    result["peak_allocated_gpu_bytes"] = max(int(row["peak_allocated_gpu_bytes"]) for row in rows)
    result["peak_reserved_gpu_bytes"] = max(int(row["peak_reserved_gpu_bytes"]) for row in rows)
    return result


def hierarchical_paired_interval(
    left: list[list[dict[str, Any]]],
    right: list[list[dict[str, Any]]],
    field: str,
    *,
    seed: int,
) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise HardwareTimingAnalysisError("Paired repetitions are not aligned")
    repetitions = len(left)
    rows = len(left[0])
    deltas = np.empty((repetitions, rows), dtype=np.float64)
    for repetition, (left_rows, right_rows) in enumerate(zip(left, right)):
        if len(left_rows) != rows or len(right_rows) != rows:
            raise HardwareTimingAnalysisError("Paired row counts differ")
        if [row["sample_id"] for row in left_rows] != [row["sample_id"] for row in right_rows]:
            raise HardwareTimingAnalysisError("Paired sample order differs")
        deltas[repetition] = [
            float(left_row[field]) - float(right_row[field])
            for left_row, right_row in zip(left_rows, right_rows)
        ]
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    chunk_size = 50
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk_size):
        count = min(chunk_size, BOOTSTRAP_RESAMPLES - start)
        repetition_indices = rng.integers(0, repetitions, size=(count, repetitions))
        row_indices = rng.integers(0, rows, size=(count, repetitions, rows))
        sampled = deltas[repetition_indices[:, :, None], row_indices]
        draws[start : start + count] = sampled.mean(axis=(1, 2))
    return {
        "point_delta": float(deltas.mean()),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "hierarchy": "repetitions_then_aligned_rows",
        "seed": seed,
        "direction": "candidate_minus_k12",
    }


def quality_metrics() -> dict[str, dict[str, float]]:
    quality = read_json(QUALITY_RESULTS)["metrics"]
    return {
        method: {
            "acc_at_0_5": float(quality[source]["acc_at_0_5"]),
            "mean_iou": float(quality[source]["mean_iou"]),
            "parse_rate": float(quality[source]["parse_rate"]),
        }
        for method, source in {
            "K12_ORIGINAL": "K12_ORIGINAL",
            "M448_OPT": "M448",
            "Z_K12_0": "Z_K12_0",
            "Z_M448_0": "Z_M448_0",
        }.items()
    }


def quality_feasible(metrics: dict[str, float]) -> bool:
    return (
        metrics["acc_at_0_5"] >= 0.429
        and metrics["mean_iou"] >= 0.4165978636
        and metrics["parse_rate"] == 1.0
    )


def load_mode(mode: str) -> tuple[dict[str, list[list[dict[str, Any]]]], list[dict[str, Any]]]:
    plan = ORDER_PLAN if mode == "natural" else FIXED32_ORDER_PLAN
    methods = METHODS if mode == "natural" else FIXED32_METHODS
    rows_by_method = {method: [] for method in methods}
    trials: list[dict[str, Any]] = []
    for repetition, order in plan.items():
        for position, method in enumerate(order, 1):
            run_dir = accepted_run(mode, repetition, position, method)
            completion = verify_completion(run_dir)
            report = validate_run(run_dir, MANIFEST, expected_rows=1000)
            rows = read_jsonl(run_dir / "timing.jsonl")
            rows_by_method[method].append(rows)
            profiler = None
            if (run_dir / "torch_profiler_summary.json").exists():
                profiler = read_json(run_dir / "torch_profiler_summary.json")
            trials.append(
                {
                    "repetition": repetition,
                    "order_position": position,
                    "method": method,
                    "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                    "completion_sha256": sha256_file(run_dir / "COMPLETED.json"),
                    "timing_sha256": completion["timing_sha256"],
                    "independent_validation_status": report["status"],
                    "profiler": profiler,
                }
            )
    return rows_by_method, trials


def source_record() -> dict[str, Any]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *[str(path) for path in SOURCE_PATHS]],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if status:
        raise HardwareTimingAnalysisError("Timing analysis requires clean committed source")
    source = source_provenance(REPO_ROOT, SOURCE_PATHS)
    source["git_dirty"] = False
    return source


def analyse_natural() -> dict[str, Any]:
    output_path = ANALYSIS_ROOT / "NATURAL_RESULTS.json"
    gate_path = ANALYSIS_ROOT / "STRICT_GATE.json"
    completion_path = ANALYSIS_ROOT / "NATURAL_COMPLETED.json"
    if any(path.exists() for path in (output_path, gate_path, completion_path)):
        raise HardwareTimingAnalysisError("Refusing to overwrite natural timing analysis")
    rows_by_method, trials = load_mode("natural")
    quality = quality_metrics()
    summaries = {
        method: mean_summary([row for repetition in rows_by_method[method] for row in repetition])
        for method in METHODS
    }
    contrasts: dict[str, Any] = {}
    for method_index, method in enumerate(METHODS[1:], 1):
        contrasts[method] = {
            field: hierarchical_paired_interval(
                rows_by_method[method],
                rows_by_method["K12_ORIGINAL"],
                field,
                seed=BOOTSTRAP_SEED + method_index * 100 + field_index,
            )
            for field_index, field in enumerate(CONTRAST_FIELDS)
        }
    profiler_counts = {
        trial["method"]: {
            "event_count": int(trial["profiler"]["event_count"]),
            "cuda_kernel_event_count": int(trial["profiler"]["cuda_kernel_event_count"]),
            "self_cpu_time_total_us": float(trial["profiler"]["self_cpu_time_total_us"]),
            "self_cuda_time_total_us": float(trial["profiler"]["self_cuda_time_total_us"]),
        }
        for trial in trials
        if trial["profiler"] is not None
    }
    source = source_record()
    results = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": utc_now(),
        "mode": "natural",
        "manifest_sha256": sha256_file(MANIFEST),
        "rows_per_trial": 1000,
        "repetitions_per_method": 5,
        "warmups_per_trial": 20,
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "hierarchy": "paired repetitions then aligned rows",
        },
        "source": source,
        "trials": trials,
        "quality": quality,
        "summaries": summaries,
        "candidate_minus_k12_contrasts": contrasts,
        "profiler_kernel_counts": profiler_counts,
    }
    atomic_write(output_path, canonical_bytes(results))
    decisions: dict[str, Any] = {}
    strict_winners: list[str] = []
    base = summaries["K12_ORIGINAL"]
    for method in METHODS[1:]:
        summary = summaries[method]
        contrast = contrasts[method]
        checks = {
            "quality_feasible": quality_feasible(quality[method]),
            "prompt_kv_lower": summary["prompt_kv_total_bytes"]["mean"]
            < base["prompt_kv_total_bytes"]["mean"],
            "mean_prefill_lower": summary["main_prefill_with_overhead_latency_s"]["mean"]
            < base["main_prefill_with_overhead_latency_s"]["mean"],
            "mean_end_to_end_lower": summary["end_to_end_latency_s"]["mean"]
            < base["end_to_end_latency_s"]["mean"],
            "prefill_ci_excludes_zero_faster": contrast[
                "main_prefill_with_overhead_latency_s"
            ]["upper_95"] < 0.0,
            "end_to_end_ci_excludes_zero_faster": contrast["end_to_end_latency_s"][
                "upper_95"
            ] < 0.0,
        }
        strict = all(checks.values())
        if strict:
            strict_winners.append(method)
        decisions[method] = {
            "checks": checks,
            "strict_actual_efficiency_win": strict,
            "classification": (
                "STRICT_ACTUAL_EFFICIENCY_WIN"
                if strict
                else "MEMORY_OR_COMPUTE_PROXY_ONLY"
                if checks["quality_feasible"] and checks["prompt_kv_lower"]
                else "NOT_AN_EFFICIENCY_QUALITY_WIN"
            ),
        }
    selected = None
    if strict_winners:
        minimum_e2e = min(summaries[method]["end_to_end_latency_s"]["mean"] for method in strict_winners)
        tied = [
            method
            for method in strict_winners
            if summaries[method]["end_to_end_latency_s"]["mean"] <= minimum_e2e * 1.005
        ]
        selected = min(
            tied,
            key=lambda method: (
                summaries[method]["main_prefill_with_overhead_latency_s"]["mean"],
                summaries[method]["prompt_kv_total_bytes"]["mean"],
                -quality[method]["acc_at_0_5"],
                -quality[method]["mean_iou"],
                summaries[method]["visual_token_layer_work"]["mean"],
            ),
        )
    gate = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": utc_now(),
        "strict_gate_definition": (
            "quality feasible; prompt KV, mean prefill and mean natural E2E lower than K12; "
            "paired hierarchical 95% prefill and E2E intervals both below zero"
        ),
        "decisions": decisions,
        "strict_winners": strict_winners,
        "selected_decoder_internal_method": selected,
        "phase5_decision": (
            "SKIPPED_DECODER_INTERNAL_STRICT_WIN"
            if selected is not None
            else "TRIGGERED_NO_DECODER_INTERNAL_STRICT_WIN"
        ),
        "natural_results_sha256": sha256_file(output_path),
    }
    atomic_write(gate_path, canonical_bytes(gate))
    completion = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "natural_results_sha256": sha256_file(output_path),
        "strict_gate_sha256": sha256_file(gate_path),
        "trials": len(trials),
        "rows": sum(len(repetition) for method in rows_by_method.values() for repetition in method),
        "phase5_decision": gate["phase5_decision"],
    }
    atomic_write(completion_path, canonical_bytes(completion))
    return gate


def analyse_fixed32() -> dict[str, Any]:
    output_path = ANALYSIS_ROOT / "FIXED32_RESULTS.json"
    completion_path = ANALYSIS_ROOT / "FIXED32_COMPLETED.json"
    if output_path.exists() or completion_path.exists():
        raise HardwareTimingAnalysisError("Refusing to overwrite fixed32 timing analysis")
    rows_by_method, trials = load_mode("fixed32")
    summaries = {
        method: mean_summary([row for repetition in rows_by_method[method] for row in repetition])
        for method in FIXED32_METHODS
    }
    contrasts = {
        method: {
            field: hierarchical_paired_interval(
                rows_by_method[method],
                rows_by_method["K12_ORIGINAL"],
                field,
                seed=BOOTSTRAP_SEED + method_index * 100 + field_index,
            )
            for field_index, field in enumerate(CONTRAST_FIELDS)
        }
        for method_index, method in enumerate(FIXED32_METHODS[1:], 1)
    }
    results = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": utc_now(),
        "mode": "fixed32",
        "role": "secondary_decode_diagnostic_not_quality_or_primary_deployment_result",
        "identical_continuation_inputs": True,
        "manifest_sha256": sha256_file(MANIFEST),
        "rows_per_trial": 1000,
        "repetitions_per_method": 3,
        "warmups_per_trial": 20,
        "source": source_record(),
        "trials": trials,
        "summaries": summaries,
        "candidate_minus_k12_contrasts": contrasts,
    }
    atomic_write(output_path, canonical_bytes(results))
    completion = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "fixed32_results_sha256": sha256_file(output_path),
        "trials": len(trials),
        "rows": sum(len(repetition) for method in rows_by_method.values() for repetition in method),
        "identical_continuation_inputs": True,
    }
    atomic_write(completion_path, canonical_bytes(completion))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("natural", "fixed32"), required=True)
    args = parser.parse_args()
    try:
        result = analyse_natural() if args.mode == "natural" else analyse_fixed32()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        HardwareTimingAnalysisError,
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
