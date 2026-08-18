#!/usr/bin/env python
"""Validate and analyse repeated timing for all final-campaign software stacks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_final_val_timing_campaign import plans  # noqa: E402
from vlm_pruning.artifacts import sha256_file  # noqa: E402
from vlm_pruning.final_statistics import hierarchical_paired_bootstrap_interval  # noqa: E402


CAMPAIGN = REPO_ROOT / "experiments/final_val_efficiency_baselines_20260818"
ANALYSIS = CAMPAIGN / "analysis"
MANIFEST = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
ROWS = 1_000
RESAMPLES = 10_000
SEED = 20_260_818
STACKS = ("canonical", "fastv", "nuwa")

EXPECTED_TRACES = {
    "Full": [576] * 32,
    "K12": [576] * 12 + [48] * 20,
    "M448_OPT": [576] * 4 + [448] * 8 + [48] * 20,
    "Z_M448_0": [576] * 4 + [448] * 8 + [48] * 8 + [0] * 12,
    "FASTV_STACK_FULL": [576] * 32,
    "FASTV_MATCHED_199": [576] * 3 + [160] * 29,
    "NUWA_STACK_FULL": [576] * 32,
    "NUWA_NATIVE_192": [336] * 16 + [48] * 16,
}

FIELDS = (
    "vision_encoder_latency_s",
    "projector_latency_s",
    "query_scoring_latency_s",
    "selector_latency_s",
    "gather_latency_s",
    "merge_latency_s",
    "prefill_excluding_pruning_latency_s",
    "prefill_including_pruning_latency_s",
    "time_to_first_token_s",
    "decode_latency_s",
    "decode_time_per_generated_token_s",
    "end_to_end_latency_s",
    "throughput_examples_per_s",
    "generated_token_count",
    "prompt_kv_total_bytes",
    "final_kv_total_bytes",
    "peak_allocated_gpu_bytes",
    "peak_reserved_gpu_bytes",
    "visual_token_layer_work",
    "average_visual_tokens",
)


class TimingAnalysisError(RuntimeError):
    """Raised when repeated timing evidence violates the frozen protocol."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("natural", "fixed32"), required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise TimingAnalysisError(f"Missing timing file: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != ROWS or any(not isinstance(row, dict) for row in rows):
        raise TimingAnalysisError(f"Timing file must contain exactly {ROWS} object rows: {path}")
    return rows


def atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise TimingAnalysisError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def normalize(row: dict[str, Any]) -> dict[str, float | None]:
    def first(*names: str) -> float | None:
        for name in names:
            value = row.get(name)
            if value is not None:
                return float(value)
        return None

    end_to_end = first("end_to_end_latency_s")
    return {
        "vision_encoder_latency_s": first("vision_encoder_latency_s"),
        "projector_latency_s": first("projector_latency_s"),
        "query_scoring_latency_s": first("query_scoring_latency_s"),
        "selector_latency_s": first("selector_latency_s"),
        "gather_latency_s": first("gather_latency_s"),
        "merge_latency_s": first("merge_latency_s"),
        "prefill_excluding_pruning_latency_s": first(
            "main_prefill_excluding_pruning_latency_s",
            "llm_prefill_excluding_selector_latency_s",
        ),
        "prefill_including_pruning_latency_s": first(
            "main_prefill_with_overhead_latency_s",
            "llm_prefill_with_overhead_latency_s",
        ),
        "time_to_first_token_s": first("time_to_first_token_s"),
        "decode_latency_s": first("decode_latency_s"),
        "decode_time_per_generated_token_s": first("decode_time_per_generated_token_s"),
        "end_to_end_latency_s": end_to_end,
        "throughput_examples_per_s": first("throughput_examples_per_s") or (
            1.0 / end_to_end if end_to_end else None
        ),
        "generated_token_count": first("generated_token_count"),
        "prompt_kv_total_bytes": first("prompt_kv_total_bytes"),
        "final_kv_total_bytes": first("final_kv_total_bytes"),
        "peak_allocated_gpu_bytes": first("peak_allocated_gpu_bytes"),
        "peak_reserved_gpu_bytes": first("peak_reserved_gpu_bytes"),
        "visual_token_layer_work": first("visual_token_layer_work"),
        "average_visual_tokens": first("average_visual_tokens"),
    }


def validate_ledger(stack: str, mode: str, expected_jobs: int) -> dict[str, Any]:
    directory = "canonical" if stack == "canonical" else f"{stack}_native"
    path = CAMPAIGN / f"timing/{directory}/{mode}/RUN_LEDGER.json"
    if not path.is_file():
        raise TimingAnalysisError(f"Missing serial timing ledger: {path}")
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("status") != "complete" or len(ledger.get("jobs", [])) != expected_jobs:
        raise TimingAnalysisError(f"Timing ledger is incomplete: {path}")
    intervals = []
    for index, row in enumerate(ledger["jobs"]):
        if row.get("status") not in {
            "complete",
            "complete_after_resume_validation",
            "validated_preexisting_complete",
        }:
            raise TimingAnalysisError(f"Timing ledger contains a non-executed job at {index}")
        if row.get("gpu_compute_pids_before") or row.get("gpu_compute_pids_after"):
            raise TimingAnalysisError(f"Competing/remaining GPU process recorded at timing job {index}")
        started = datetime.fromisoformat(str(row["started_at_utc"]).replace("Z", "+00:00"))
        finished = datetime.fromisoformat(str(row["finished_at_utc"]).replace("Z", "+00:00"))
        if finished <= started:
            raise TimingAnalysisError(f"Non-positive timing job interval at {index}")
        intervals.append((started, finished))
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise TimingAnalysisError("Timing ledger contains overlapping GPU job intervals")
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}


def validate_trace(method: str, trace: list[int]) -> None:
    if method == "FEATHER_LLAVA_MAIN":
        valid = (
            len(trace) == 32
            and trace[:8] == [576] * 8
            and len(set(trace[8:16])) == 1
            and len(set(trace[16:])) == 1
            and 576 >= trace[8] >= trace[16] >= 0
        )
        if not valid:
            raise TimingAnalysisError(f"FEATHER timing trace differs: {trace}")
    elif trace != EXPECTED_TRACES[method]:
        raise TimingAnalysisError(f"{method} timing trace differs: {trace}")


def load_stack(
    stack: str,
    mode: str,
    manifest_ids: list[str],
) -> tuple[dict[str, list[list[dict[str, float | None]]]], list[dict[str, Any]], dict[str, Any]]:
    jobs = plans(stack, mode)
    ledger = validate_ledger(stack, mode, len(jobs))
    by_method: dict[str, list[list[dict[str, float | None]]]] = {}
    inventory = []
    for repetition, position, method, run_dir, _ in jobs:
        completion_path = run_dir / "COMPLETED.json"
        config_path = run_dir / "config.json"
        timing_path = run_dir / ("predictions.jsonl" if stack == "nuwa" else "timing.jsonl")
        if not completion_path.is_file() or not config_path.is_file():
            raise TimingAnalysisError(f"Incomplete timing directory: {run_dir}")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if str(completion.get("status") or "").lower() not in {"pass", "complete"}:
            raise TimingAnalysisError(f"Rejected timing completion: {run_dir}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            config.get("method") != method
            or int(config.get("repetition", -1)) != repetition
            or int(config.get("order_position", -1)) != position
        ):
            raise TimingAnalysisError(f"Timing configuration/order differs: {run_dir}")
        recorded_mode = config.get("generation_mode") if stack == "nuwa" else config.get("mode")
        if recorded_mode != mode:
            raise TimingAnalysisError(f"Timing generation mode differs: {run_dir}")
        config_sha = sha256_file(config_path)
        rows = read_jsonl(timing_path)
        if [str(row.get("sample_id")) for row in rows] != manifest_ids:
            raise TimingAnalysisError(f"Timing manifest order differs: {run_dir}")
        if [int(row.get("manifest_index", -1)) for row in rows] != list(range(ROWS)):
            raise TimingAnalysisError(f"Timing manifest indices differ: {run_dir}")
        if any(row.get("config_sha256") != config_sha for row in rows):
            raise TimingAnalysisError(f"Timing row config hash differs: {run_dir}")
        normalized = []
        for row_index, row in enumerate(rows):
            if row.get("method") != method:
                raise TimingAnalysisError(f"Timing row method differs at {run_dir}:{row_index + 1}")
            trace = [int(value) for value in row.get("realized_visual_tokens_by_layer") or []]
            validate_trace(method, trace)
            normalized.append(normalize(row))
        by_method.setdefault(method, []).append(normalized)
        inventory.append(
            {
                "stack": stack,
                "mode": mode,
                "method": method,
                "repetition": repetition,
                "order_position": position,
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "rows": ROWS,
                "timing_sha256": sha256_file(timing_path),
                "config_sha256": config_sha,
                "validation_status": "pass",
            }
        )
    expected_repetitions = 5 if mode == "natural" else 3
    if any(len(repetitions) != expected_repetitions for repetitions in by_method.values()):
        raise TimingAnalysisError(f"{stack} does not contain the required repetitions")
    return by_method, inventory, ledger


def distribution(values: list[float]) -> dict[str, float | int]:
    ordered = np.asarray(values, dtype=np.float64)
    return {
        "available_rows": int(ordered.size),
        "mean": float(ordered.mean()),
        "std": float(ordered.std(ddof=1)) if ordered.size > 1 else 0.0,
        "median": float(np.median(ordered)),
        "p05": float(np.quantile(ordered, 0.05)),
        "p95": float(np.quantile(ordered, 0.95)),
    }


def summarize(stack: str, methods: dict[str, list[list[dict[str, float | None]]]]) -> list[dict[str, Any]]:
    output = []
    for method, repetitions in methods.items():
        row: dict[str, Any] = {
            "stack": stack,
            "method": method,
            "repetitions": len(repetitions),
            "rows_per_repetition": ROWS,
        }
        for field in FIELDS:
            values = [
                float(sample[field])
                for repetition in repetitions
                for sample in repetition
                if sample[field] is not None
            ]
            row[field] = distribution(values) if values else {"available_rows": 0}
        output.append(row)
    return output


def contrasts(
    stacks: dict[str, dict[str, list[list[dict[str, float | None]]]]]
) -> list[dict[str, Any]]:
    pairs = {
        "canonical": (
            ("K12", "Full"),
            ("M448_OPT", "Full"),
            ("Z_M448_0", "Full"),
            ("FEATHER_LLAVA_MAIN", "Full"),
            ("M448_OPT", "K12"),
            ("Z_M448_0", "K12"),
        ),
        "fastv": (("FASTV_MATCHED_199", "FASTV_STACK_FULL"),),
        "nuwa": (("NUWA_NATIVE_192", "NUWA_STACK_FULL"),),
    }
    output = []
    for stack, stack_pairs in pairs.items():
        if stack not in stacks:
            continue
        for left, right in stack_pairs:
            for field in (
                "end_to_end_latency_s",
                "prefill_including_pruning_latency_s",
                "time_to_first_token_s",
                "decode_latency_s",
                "generated_token_count",
            ):
                left_rows = stacks[stack][left]
                right_rows = stacks[stack][right]
                if any(
                    sample[field] is None
                    for repetition in (*left_rows, *right_rows)
                    for sample in repetition
                ):
                    continue
                left_values = [
                    [float(sample[field]) for sample in repetition]  # type: ignore[arg-type]
                    for repetition in left_rows
                ]
                right_values = [
                    [float(sample[field]) for sample in repetition]  # type: ignore[arg-type]
                    for repetition in right_rows
                ]
                interval = hierarchical_paired_bootstrap_interval(
                    left_values,
                    right_values,
                    resamples=RESAMPLES,
                    seed=SEED,
                )
                output.append(
                    {
                        "stack": stack,
                        "contrast": f"{left}-minus-{right}",
                        "left": left,
                        "right": right,
                        "metric": field,
                        **interval,
                    }
                )
    return output


def fixed32_trigger(
    summaries: list[dict[str, Any]],
    interval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {(row["stack"], row["method"]): row for row in summaries}
    reference = {
        "canonical": "Full",
        "fastv": "FASTV_STACK_FULL",
        "nuwa": "NUWA_STACK_FULL",
    }
    output = {}
    for stack, full in reference.items():
        full_tokens = float(summary[(stack, full)]["generated_token_count"]["mean"])
        full_e2e = float(summary[(stack, full)]["end_to_end_latency_s"]["mean"])
        reasons = []
        methods = [method for current_stack, method in summary if current_stack == stack and method != full]
        for method in methods:
            token_mean = float(summary[(stack, method)]["generated_token_count"]["mean"])
            token_fraction = abs(token_mean - full_tokens) / full_tokens
            if token_fraction > 0.02:
                reasons.append(
                    {
                        "method": method,
                        "trigger": "mean_generated_token_difference_fraction_gt_0_02",
                        "observed_fraction": token_fraction,
                    }
                )
            contrast = next(
                (
                    row for row in interval_rows
                    if row["stack"] == stack
                    and row["left"] == method
                    and row["right"] == full
                    and row["metric"] == "end_to_end_latency_s"
                ),
                None,
            )
            if contrast is not None:
                e2e_fraction = abs(float(contrast["point_delta"])) / full_e2e
                if e2e_fraction <= 0.02:
                    reasons.append(
                        {
                            "method": method,
                            "trigger": "natural_e2e_contrast_within_plus_minus_0_02",
                            "observed_fraction": e2e_fraction,
                        }
                    )
        output[stack] = {"required": bool(reasons), "reasons": reasons}
    return {
        "status": "complete",
        "generated_token_threshold_fraction": 0.02,
        "near_zero_e2e_threshold_fraction": 0.02,
        "fixed32_repetitions_if_required": 3,
        "stacks": output,
    }


def analysis_stacks(mode: str) -> tuple[str, ...]:
    """Return exactly the software stacks registered for this analysis mode."""

    if mode == "natural":
        return STACKS
    trigger_path = ANALYSIS / "FIXED32_TRIGGER.json"
    if not trigger_path.is_file():
        raise TimingAnalysisError("Fixed32 analysis requires FIXED32_TRIGGER.json")
    trigger = json.loads(trigger_path.read_text(encoding="utf-8"))
    if trigger.get("status") != "complete":
        raise TimingAnalysisError("Fixed32 trigger artifact is incomplete")
    unknown = set(trigger.get("stacks", {})) - set(STACKS)
    if unknown:
        raise TimingAnalysisError(f"Fixed32 trigger contains unknown stacks: {sorted(unknown)}")
    required = tuple(
        stack
        for stack in STACKS
        if trigger.get("stacks", {}).get(stack, {}).get("required") is True
    )
    if not required:
        raise TimingAnalysisError("Fixed32 analysis was requested but no stack was triggered")
    return required


def main() -> int:
    args = parse_args()
    if sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise TimingAnalysisError("Frozen timing manifest hash differs")
    manifest = read_jsonl(MANIFEST)
    manifest_ids = [str(row["sample_id"]) for row in manifest]
    if len(set(manifest_ids)) != ROWS:
        raise TimingAnalysisError("Frozen timing manifest IDs are not unique")
    stack_data = {}
    inventory = []
    ledgers = {}
    summaries = []
    for stack in analysis_stacks(args.mode):
        stack_data[stack], stack_inventory, ledgers[stack] = load_stack(
            stack, args.mode, manifest_ids
        )
        inventory.extend(stack_inventory)
        summaries.extend(summarize(stack, stack_data[stack]))
    interval_rows = contrasts(stack_data)
    stem = "NATURAL" if args.mode == "natural" else "FIXED32"
    atomic_json(ANALYSIS / f"TIMING_{stem}_SUMMARY.json", summaries)
    atomic_json(ANALYSIS / f"TIMING_{stem}_CONTRASTS.json", interval_rows)
    atomic_csv(ANALYSIS / f"TIMING_{stem}_CONTRASTS.csv", interval_rows)
    atomic_json(
        ANALYSIS / f"TIMING_{stem}_INVENTORY.json",
        {"status": "complete", "mode": args.mode, "runs": inventory, "ledgers": ledgers},
    )
    if args.mode == "natural":
        atomic_json(ANALYSIS / "FIXED32_TRIGGER.json", fixed32_trigger(summaries, interval_rows))
    print(json.dumps({"status": "complete", "mode": args.mode, "summaries": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
