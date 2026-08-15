#!/usr/bin/env python
"""Aggregate matched timing, apply triggers, and finalize the campaign decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
STATE_PATH = CAMPAIGN_ROOT / "MATCHED_TIMING_STATE.json"
NATURAL_PATH = CAMPAIGN_ROOT / "analysis/NATURAL_MATCHED_TIMING_RESULTS.json"
FINAL_PATH = CAMPAIGN_ROOT / "analysis/MATCHED_TIMING_RESULTS.json"
SELECTION_PATH = CAMPAIGN_ROOT / "analysis/GATE_RESULTS_AND_SELECTION.json"
REGISTRY_PATH = CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"
CONTROLLER_PATH = CAMPAIGN_ROOT / "CONTROLLER_STATE.json"
REPORT_PATH = CAMPAIGN_ROOT / "RESEARCH_REPORT.md"
METHODS = ("full", "k12_original", "candidate")
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260815


class MatchedTimingAnalysisError(RuntimeError):
    """Raised when serial matched-timing evidence is incomplete or inconsistent."""


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MatchedTimingAnalysisError(f"Expected JSON object: {path}")
    return value


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def collect(mode: str, repetitions: int) -> tuple[dict[str, list[list[dict[str, Any]]]], list[dict[str, Any]]]:
    state = read_json(STATE_PATH)
    values: dict[str, list[list[dict[str, Any]]]] = {method: [] for method in METHODS}
    artifacts = []
    for repetition in range(1, repetitions + 1):
        matching = [
            record
            for key, record in state["runs"].items()
            if key.startswith(f"{mode}_rep{repetition}_") and record.get("status") == "accepted"
        ]
        if len(matching) != 3:
            raise MatchedTimingAnalysisError(f"{mode} repetition {repetition} does not have three accepted trials")
        if {record["position"] for record in matching} != {1, 2, 3} or {record["method"] for record in matching} != set(METHODS):
            raise MatchedTimingAnalysisError(f"{mode} repetition {repetition} is not a complete registered order")
        for record in matching:
            run_dir = REPO_ROOT / record["accepted_run_dir"]
            integrity = read_json(run_dir / "integrity_report.json")
            completion = read_json(run_dir / "COMPLETED.json")
            if integrity.get("status") != "pass" or completion.get("status") != "complete":
                raise MatchedTimingAnalysisError(f"Timing artifact is not independently valid: {run_dir}")
            rows = [
                json.loads(line)
                for line in (run_dir / "timing.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(rows) != 1000:
                raise MatchedTimingAnalysisError(f"Timing artifact is not 1,000 rows: {run_dir}")
            values[record["method"]].append(rows)
            artifacts.append(
                {
                    "method": record["method"],
                    "repetition": repetition,
                    "order_position": record["position"],
                    "run_dir": relative(run_dir),
                    "timing_sha256": sha256_file(run_dir / "timing.jsonl"),
                    "integrity_sha256": sha256_file(run_dir / "integrity_report.json"),
                    "completion_sha256": sha256_file(run_dir / "COMPLETED.json"),
                }
            )
    for method in METHODS:
        values[method].sort(key=lambda rows: int(rows[0]["repetition"]))
    return values, artifacts


def summarize_method(repetitions: list[list[dict[str, Any]]]) -> dict[str, Any]:
    fields = (
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
        "generated_token_count",
        "prompt_kv_total_bytes",
        "final_kv_total_bytes",
        "visual_token_layer_work",
        "average_visual_tokens",
    )
    result: dict[str, Any] = {"repetitions": len(repetitions), "rows_per_repetition": 1000}
    for field in fields:
        rep_means = [float(np.mean([float(row[field]) for row in rows])) for rows in repetitions]
        result[field] = {
            "mean": float(np.mean(rep_means)),
            "repetition_means": rep_means,
            "repetition_sample_sd": float(np.std(rep_means, ddof=1)) if len(rep_means) > 1 else 0.0,
        }
    result["peak_allocated_gpu_bytes"] = max(
        int(row["peak_allocated_gpu_bytes"]) for rows in repetitions for row in rows
    )
    result["peak_reserved_gpu_bytes"] = max(
        int(row["peak_reserved_gpu_bytes"]) for rows in repetitions for row in rows
    )
    return result


def hierarchical_interval(
    left: list[list[dict[str, Any]]],
    right: list[list[dict[str, Any]]],
    field: str,
    *,
    seed: int,
) -> dict[str, Any]:
    if len(left) != len(right):
        raise MatchedTimingAnalysisError("Hierarchical contrast repetitions differ")
    delta = np.asarray(
        [
            [float(lrow[field]) - float(rrow[field]) for lrow, rrow in zip(lrows, rrows, strict=True)]
            for lrows, rrows in zip(left, right, strict=True)
        ],
        dtype=np.float64,
    )
    repetitions = delta.shape[0]
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, BOOTSTRAPS, 250):
        count = min(250, BOOTSTRAPS - start)
        rep_indices = rng.integers(0, repetitions, size=(count, repetitions))
        sample_indices = rng.integers(0, 1000, size=(count, 1000))
        selected = delta[rep_indices[:, :, None], sample_indices[:, None, :]]
        samples.append(selected.mean(axis=(1, 2)))
    boot = np.concatenate(samples)
    point = float(delta.mean())
    right_mean = float(
        np.mean([[float(row[field]) for row in repetition] for repetition in right])
    )
    return {
        "field": field,
        "point_delta": point,
        "point_ratio": None if right_mean == 0 else 1.0 + point / right_mean,
        "lower_95": float(np.quantile(boot, 0.025)),
        "upper_95": float(np.quantile(boot, 0.975)),
        "bootstrap_resamples": BOOTSTRAPS,
        "seed": seed,
        "hierarchy": "resample_repetitions_then_shared_example_indices",
    }


def analyse_mode(mode: str, repetitions: int) -> dict[str, Any]:
    rows, artifacts = collect(mode, repetitions)
    summaries = {method: summarize_method(rows[method]) for method in METHODS}
    contrasts = {}
    offset = 0
    for left, right in (("candidate", "k12_original"), ("candidate", "full"), ("k12_original", "full")):
        name = f"{left}-minus-{right}"
        contrasts[name] = {}
        for field in (
            "main_prefill_with_overhead_latency_s",
            "end_to_end_latency_s",
            "decode_latency_s",
            "decode_time_per_generated_token_s",
            "generated_token_count",
            "prompt_kv_total_bytes",
            "final_kv_total_bytes",
        ):
            contrasts[name][field] = hierarchical_interval(
                rows[left], rows[right], field, seed=BOOTSTRAP_SEED + offset
            )
            offset += 1
    return {
        "mode": mode,
        "repetitions": repetitions,
        "methods": summaries,
        "hierarchical_paired_contrasts": contrasts,
        "artifacts": artifacts,
    }


def fixed_trigger(natural: dict[str, Any]) -> dict[str, Any]:
    generated = {
        method: natural["methods"][method]["generated_token_count"]["mean"] for method in METHODS
    }
    generated_range = max(generated.values()) - min(generated.values())
    candidate = natural["methods"]["candidate"]["end_to_end_latency_s"]["mean"]
    full = natural["methods"]["full"]["end_to_end_latency_s"]["mean"]
    relative_change = candidate / full - 1.0
    return {
        "triggered": generated_range > 0.25 or abs(relative_change) <= 0.02,
        "generated_token_means": generated,
        "generated_token_mean_range": generated_range,
        "generated_token_range_trigger": generated_range > 0.25,
        "candidate_vs_full_end_to_end_relative_change": relative_change,
        "within_plus_minus_2_percent_trigger": abs(relative_change) <= 0.02,
        "fixed_generated_tokens_if_triggered": 32,
    }


def timing_gate(natural: dict[str, Any]) -> dict[str, Any]:
    candidate = natural["methods"]["candidate"]
    k12 = natural["methods"]["k12_original"]
    prefill_ratio = (
        candidate["main_prefill_with_overhead_latency_s"]["mean"]
        / k12["main_prefill_with_overhead_latency_s"]["mean"]
    )
    end_ratio = candidate["end_to_end_latency_s"]["mean"] / k12["end_to_end_latency_s"]["mean"]
    prompt_less = candidate["prompt_kv_total_bytes"]["mean"] < k12["prompt_kv_total_bytes"]["mean"]
    passed = prefill_ratio <= 0.95 and end_ratio <= 0.98 and prompt_less
    interval = natural["hierarchical_paired_contrasts"]["candidate-minus-k12_original"]
    return {
        "passed": passed,
        "prefill_ratio": prefill_ratio,
        "prefill_at_most_0_95": prefill_ratio <= 0.95,
        "end_to_end_ratio": end_ratio,
        "end_to_end_at_most_0_98": end_ratio <= 0.98,
        "candidate_prompt_kv_bytes": candidate["prompt_kv_total_bytes"]["mean"],
        "k12_prompt_kv_bytes": k12["prompt_kv_total_bytes"]["mean"],
        "prompt_kv_strictly_less": prompt_less,
        "prefill_interval_excludes_no_change": interval["main_prefill_with_overhead_latency_s"]["upper_95"] < 0,
        "end_to_end_interval_excludes_no_change": interval["end_to_end_latency_s"]["upper_95"] < 0,
    }


def append_once(items: list[dict[str, Any]], key: str, value: dict[str, Any]) -> None:
    if any(item.get(key) == value[key] for item in items):
        return
    items.append(value)


def finalize_campaign(natural: dict[str, Any], trigger: dict[str, Any], fixed: dict[str, Any] | None) -> dict[str, Any]:
    selection = read_json(SELECTION_PATH)
    gate = timing_gate(natural)
    route = selection["route_gate_decision"]
    selected = selection["project_selection"]["selected"]
    if route["router_label"] == "LIGHTWEIGHT_ROUTER_MAIN_METHOD_JUSTIFIED" and selected == "GATE_SEED0":
        outcome = "C — Learned adaptive-routing success"
    else:
        outcome = "B — Information-preservation success"
    confirmation = gate["passed"]
    result = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "selected_project_candidate": selected,
        "natural": natural,
        "fixed_length_trigger": trigger,
        "fixed_length": fixed,
        "timing_gate": gate,
        "primary_outcome": outcome,
        "router_label": route["router_label"],
        "future_8811_confirmation_recommended_not_launched": confirmation,
        "future_confirmation_scope": [selected, "K12_ORIGINAL"] if confirmation else [],
    }
    atomic_write(FINAL_PATH, canonical_bytes(result))

    registry = read_json(REGISTRY_PATH)
    controller = read_json(CONTROLLER_PATH)
    registry["phases"]["timing"] = {
        "status": "complete",
        "analysis_path": relative(FINAL_PATH),
        "analysis_sha256": sha256_file(FINAL_PATH),
        "selected_candidate": selected,
        "timing_gate_passed": gate["passed"],
        "fixed_length_status": "complete" if fixed is not None else "skipped_not_triggered",
    }
    registry["timing_results"] = {
        "selected_candidate": selected,
        "timing_gate": gate,
        "fixed_length_trigger": trigger,
    }
    append_once(
        registry["decisions"],
        "decision",
        {
            "at_utc": utc_now(),
            "decision": "complete_matched_timing",
            "selected_candidate": selected,
            "timing_gate_passed": gate["passed"],
            "primary_outcome": outcome,
            "future_8811_confirmation_recommended_not_launched": confirmation,
            "evidence_path": relative(FINAL_PATH),
            "evidence_sha256": sha256_file(FINAL_PATH),
        },
    )
    registry["updated_at_utc"] = utc_now()
    controller["state"] = "MATCHED_TIMING_COMPLETE_FINAL_AUDIT_PENDING"
    controller["next_required_action"] = "reconcile_final_report_registry_and_completion_audit"
    append_once(
        controller["decision_history"],
        "phase",
        {
            "phase": "timing",
            "status": "complete",
            "selected_candidate": selected,
            "timing_gate_passed": gate["passed"],
            "evidence_path": relative(FINAL_PATH),
            "evidence_sha256": sha256_file(FINAL_PATH),
        },
    )
    controller["updated_at_utc"] = utc_now()

    natural_methods = natural["methods"]
    candidate_contrast = natural["hierarchical_paired_contrasts"]["candidate-minus-k12_original"]
    timing_lines = [
        "## Conditional matched timing",
        "",
        f"Timing ran serially on the frozen 1,000 rows with no overlapping CUDA process: five natural-greedy repetitions of Full, original K12/B48 more-context, and {selected} in the exact registered orders.",
        "",
        "| Method | Prefill incl. overhead (s) | End-to-end (s) | Generated tokens | Prompt KV (MiB) | Layer work |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, label in (("full", "Full"), ("k12_original", "K12 original"), ("candidate", selected)):
        item = natural_methods[method]
        timing_lines.append(
            f"| {label} | {item['main_prefill_with_overhead_latency_s']['mean']:.6f} | {item['end_to_end_latency_s']['mean']:.6f} | {item['generated_token_count']['mean']:.3f} | {item['prompt_kv_total_bytes']['mean'] / 1024**2:.2f} | {item['visual_token_layer_work']['mean']:.1f} |"
        )
    prefill_ci = candidate_contrast["main_prefill_with_overhead_latency_s"]
    end_ci = candidate_contrast["end_to_end_latency_s"]
    timing_lines.extend(
        [
            "",
            f"Candidate minus K12 prefill is {prefill_ci['point_delta']:+.6f}s (95% hierarchical paired CI [{prefill_ci['lower_95']:+.6f}, {prefill_ci['upper_95']:+.6f}]); end-to-end is {end_ci['point_delta']:+.6f}s (CI [{end_ci['lower_95']:+.6f}, {end_ci['upper_95']:+.6f}]). The registered timing gate {'passes' if gate['passed'] else 'fails'}: prefill ratio {gate['prefill_ratio']:.3f}, end-to-end ratio {gate['end_to_end_ratio']:.3f}, prompt-KV-less={gate['prompt_kv_strictly_less']}.",
            "",
            (
                "The fixed-32-token control was triggered and completed in three balanced repetitions; it is timing-only and contributes no quality evidence."
                if fixed is not None
                else f"The fixed-length control was skipped by the exact trigger: generated-token mean range {trigger['generated_token_mean_range']:.3f}, candidate-vs-Full end-to-end change {trigger['candidate_vs_full_end_to_end_relative_change']:+.3%}."
            ),
            "",
            f"Machine-readable timing evidence: `{relative(FINAL_PATH)}`.",
        ]
    )
    report = REPORT_PATH.read_text(encoding="utf-8")
    report, count = re.subn(
        r"## Conditional matched timing\n\n.*?(?=\n## Failures and rejected attempts)",
        "\n".join(timing_lines),
        report,
        flags=re.DOTALL,
    )
    if count != 1:
        raise MatchedTimingAnalysisError("Could not replace timing report section")
    recommendation = "\n".join(
        [
            "## Final recommendation",
            "",
            f"Primary outcome: **{outcome}**.",
            "",
            f"Router decision: `{route['router_label']}`.",
            "",
            (
                f"The selected deployable project candidate is **{selected}**. It passes both the K12-like quality screen and matched timing gate. Recommend—but do not launch—one later 8,811-example confirmation of {selected} with K12 original as the sole essential comparator."
                if confirmation
                else f"The selected deployable project candidate is **{selected}**. It passes the K12-like quality screen but not the complete matched timing gate, so no 8,811-example confirmation is recommended yet."
            ),
            "",
            "Scientific decision: the discovery subset supports information preservation by M448 as the dissertation-safe quality result; learned routing is elevated only to the level stated by the router decision above. No held-out, transfer, dissertation-editing, legacy-goal, or 8,811-example job was launched.",
        ]
    )
    report, count = re.subn(
        r"## Final recommendation\n\n.*\Z",
        recommendation + "\n",
        report,
        flags=re.DOTALL,
    )
    if count != 1:
        raise MatchedTimingAnalysisError("Could not replace final recommendation")
    atomic_write(REPORT_PATH, report.encode("utf-8"))
    atomic_write(REGISTRY_PATH, canonical_bytes(registry))
    atomic_write(CONTROLLER_PATH, canonical_bytes(controller))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--natural", action="store_true")
    group.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    natural = analyse_mode("natural", 5)
    trigger = fixed_trigger(natural)
    natural_result = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "natural": natural,
        "fixed_length_trigger": trigger,
        "timing_gate": timing_gate(natural),
    }
    atomic_write(NATURAL_PATH, canonical_bytes(natural_result))
    if args.natural and trigger["triggered"]:
        state = read_json(STATE_PATH)
        state.setdefault("modes", {})["fixed32"] = {
            "status": "triggered_pending",
            "trigger_evidence_path": relative(NATURAL_PATH),
            "trigger_evidence_sha256": sha256_file(NATURAL_PATH),
        }
        state["updated_at_utc"] = utc_now()
        atomic_write(STATE_PATH, canonical_bytes(state))
        print(json.dumps(natural_result, indent=2, sort_keys=True))
        return 0
    if args.finalize and not trigger["triggered"]:
        raise MatchedTimingAnalysisError("Fixed finalization requested although fixed timing was not triggered")
    fixed = analyse_mode("fixed32", 3) if trigger["triggered"] else None
    result = finalize_campaign(natural, trigger, fixed)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MatchedTimingAnalysisError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
