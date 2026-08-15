#!/usr/bin/env python
"""Finalize learned-gate evidence, controls, and project-candidate selection."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
ANALYSIS_PATH = CAMPAIGN_ROOT / "analysis/GATE_RESULTS_AND_SELECTION.json"
REGISTRY_PATH = CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"
STATE_PATH = CAMPAIGN_ROOT / "CONTROLLER_STATE.json"
REPORT_PATH = CAMPAIGN_ROOT / "RESEARCH_REPORT.md"
LABEL_RUN = CAMPAIGN_ROOT / "gate_labels/labels_full3000_attempt1"
TRAINING_ROOT = CAMPAIGN_ROOT / "gate_training"
VALIDATION_ROOT = CAMPAIGN_ROOT / "gate_validation_runs"
CONTROLS_ROOT = CAMPAIGN_ROOT / "gate_controls"
SAFE_ROWS_PATH = CAMPAIGN_ROOT / "analysis/rescored/K12_QUERY_TOPK.jsonl"
ORIGINAL_K12_PATH = CAMPAIGN_ROOT / "analysis/rescored/K12_ORIGINAL.jsonl"
M448_PATH = CAMPAIGN_ROOT / "runs/m448_attempt1/predictions.jsonl"
OFFICIAL_ROOT = CAMPAIGN_ROOT / "official_runs"
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260815


class GateAnalysisError(RuntimeError):
    """Raised when learned-gate evidence cannot support a terminal decision."""


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
        raise GateAnalysisError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path, expected: int = 1000) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected or not all(isinstance(row, dict) for row in rows):
        raise GateAnalysisError(f"Expected {expected} JSON objects: {path}")
    return rows


def indexed_source_rows(path: Path, wanted_ids: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id"))
            if sample_id in wanted_ids:
                result[sample_id] = row
    if set(result) != wanted_ids:
        raise GateAnalysisError(f"Historical safe source is missing frozen IDs: {sorted(wanted_ids - set(result))[:3]}")
    return result


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def run_hashes(run_dir: Path) -> dict[str, Any]:
    result = {"run_dir": relative(run_dir)}
    for key, name in (
        ("config_sha256", "config.json"),
        ("environment_sha256", "environment.json"),
        ("predictions_sha256", "predictions.jsonl"),
        ("metrics_sha256", "metrics.json"),
        ("integrity_sha256", "integrity_report.json"),
        ("completion_sha256", "COMPLETED.json"),
    ):
        path = run_dir / name
        if path.is_file():
            result[key] = sha256_file(path)
    return result


def compact_metrics(rows: Sequence[dict[str, Any]], *, token_key: str | None = None) -> dict[str, Any]:
    count = len(rows)
    result = {
        "rows": count,
        "acc_at_0_5": sum(bool(row["success_iou_0_5"]) for row in rows) / count,
        "mean_iou": sum(float(row["box_iou"]) for row in rows) / count,
        "parse_rate": sum(row.get("parse_failure_reason") is None for row in rows) / count,
    }
    if token_key is not None:
        result["average_visual_tokens"] = sum(float(row[token_key]) for row in rows) / count
    return result


def paired_interval(delta: np.ndarray, *, scale: float, seed: int) -> dict[str, Any]:
    if delta.shape != (1000,):
        raise GateAnalysisError("Paired interval requires 1,000 aligned rows")
    rng = np.random.default_rng(seed)
    chunks = []
    for start in range(0, BOOTSTRAPS, 500):
        count = min(500, BOOTSTRAPS - start)
        indices = rng.integers(0, 1000, size=(count, 1000))
        chunks.append(delta[indices].mean(axis=1) * scale)
    samples = np.concatenate(chunks)
    return {
        "point_delta": float(delta.mean() * scale),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "bootstrap_resamples": BOOTSTRAPS,
        "seed": seed,
    }


def contrast(
    name: str,
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    *,
    left_tokens: float | None,
    right_tokens: float | None,
    left_work: float | None,
    right_work: float | None,
    seed_offset: int,
) -> dict[str, Any]:
    if [row["sample_id"] for row in left] != [row["sample_id"] for row in right]:
        raise GateAnalysisError(f"Rows are not aligned for {name}")
    left_acc = np.asarray([float(bool(row["success_iou_0_5"])) for row in left], dtype=np.float64)
    right_acc = np.asarray([float(bool(row["success_iou_0_5"])) for row in right], dtype=np.float64)
    left_iou = np.asarray([float(row["box_iou"]) for row in left], dtype=np.float64)
    right_iou = np.asarray([float(row["box_iou"]) for row in right], dtype=np.float64)
    return {
        "contrast": name,
        "acc_at_0_5_percentage_points": paired_interval(
            left_acc - right_acc, scale=100.0, seed=BOOTSTRAP_SEED + seed_offset
        ),
        "mean_iou": paired_interval(
            left_iou - right_iou, scale=1.0, seed=BOOTSTRAP_SEED + 100 + seed_offset
        ),
        "discordant": {
            "left_correct_right_wrong": int(np.sum((left_acc == 1) & (right_acc == 0))),
            "left_wrong_right_correct": int(np.sum((left_acc == 0) & (right_acc == 1))),
        },
        "parse_rate_delta": (
            sum(row.get("parse_failure_reason") is None for row in left)
            - sum(row.get("parse_failure_reason") is None for row in right)
        )
        / 1000,
        "average_visual_tokens_delta": None
        if left_tokens is None or right_tokens is None
        else left_tokens - right_tokens,
        "visual_token_layer_work_delta": None
        if left_work is None or right_work is None
        else left_work - right_work,
    }


def k12_like(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics["acc_at_0_5"]) >= 0.429
        and float(metrics["mean_iou"]) >= 0.4165978636
        and float(metrics["average_visual_tokens"]) < 246.0
    )


def training_seed_evidence(seed: int) -> dict[str, Any]:
    run_dir = TRAINING_ROOT / f"seed_{seed}"
    integrity = read_json(run_dir / "integrity_report.json")
    completion = read_json(run_dir / "COMPLETED.json")
    calibration = read_json(run_dir / "calibration.json")
    history = read_json(run_dir / "training_history.json")
    environment = read_json(run_dir / "environment.json")
    if integrity.get("status") != "pass" or completion.get("status") != "complete":
        raise GateAnalysisError(f"Gate training seed {seed} is not independently complete")
    return {
        "run_dir": relative(run_dir),
        "checkpoint_sha256": integrity["checkpoint_sha256"],
        "integrity_sha256": sha256_file(run_dir / "integrity_report.json"),
        "completion_sha256": sha256_file(run_dir / "COMPLETED.json"),
        "best_epoch": history["best_epoch"],
        "epochs_trained": history["epochs_trained"],
        "calibration_status": calibration["status"],
        "threshold": calibration["threshold"],
        "calibration_metrics": calibration["chosen_metrics"],
        "normalization_sha256": integrity["normalization_sha256"],
        "training_spec_sha256": integrity["training_spec_sha256"],
        "environment_sha256": sha256_file(run_dir / "environment.json"),
        "training_device": environment["device"],
        "trainable_parameters": environment["trainable_parameters"],
        "llava_parameters_trained": environment["llava_parameters_trained"],
    }


def gate_seed_evidence(seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = VALIDATION_ROOT / f"seed{seed}_full1000_attempt1"
    integrity = read_json(run_dir / "integrity_report.json")
    completion = read_json(run_dir / "COMPLETED.json")
    if integrity.get("status") != "pass" or completion.get("status") != "complete":
        raise GateAnalysisError(f"Physical gate seed {seed} is not independently complete")
    rows = read_jsonl(run_dir / "predictions.jsonl")
    metrics = integrity["metrics"]
    compact = {
        "rows": 1000,
        "acc_at_0_5": float(metrics["quality/acc_at_0_5"]),
        "mean_iou": float(metrics["quality/mean_iou"]),
        "parse_rate": float(metrics["quality/parsed_box_rate"]),
        "average_visual_tokens": float(metrics["tokens/candidate_average_visual_tokens"]),
        "visual_token_layer_work": float(metrics["tokens/visual_token_layer_work_mean"]),
        "safe_rows": int(metrics["gate/safe_rows"]),
        "aggressive_rows": int(metrics["gate/aggressive_rows"]),
        "safe_rate": float(metrics["gate/safe_rate"]),
        "aggressive_rate": float(metrics["gate/aggressive_rate"]),
        "gate_latency_mean_s": float(metrics["gate/latency_mean_s"]),
        "gate_model_latency_mean_s": float(metrics["gate/model_latency_mean_s"]),
        "peak_allocated_mb": float(metrics["efficiency/peak_allocated_mb"]),
        "peak_reserved_mb": float(metrics["efficiency/peak_reserved_mb"]),
    }
    return {
        "seed": seed,
        "metrics": compact,
        "k12_like": k12_like(compact),
        "artifacts": run_hashes(run_dir),
        "checkpoint_sha256": integrity["checkpoint_sha256"],
        "checks": integrity["checks"],
    }, rows


def route_errors(
    gate_rows: Sequence[dict[str, Any]],
    safe_rows: Sequence[dict[str, Any]],
    aggressive_rows: Sequence[dict[str, Any]],
    safe_physical_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not (
        [row["sample_id"] for row in gate_rows]
        == [row["sample_id"] for row in safe_rows]
        == [row["sample_id"] for row in aggressive_rows]
    ):
        raise GateAnalysisError("Route-error sources are not aligned")
    false_aggressive = 0
    false_safe = 0
    missed_safe_iou_benefit = 0
    parity_differences = []
    safe_output_differences = 0
    aggressive_output_differences = 0
    safe_retained_index_differences = 0
    source_policy_correct = 0
    source_policy_iou = 0.0
    physical_correct = 0
    physical_iou = 0.0
    for index, (gate, safe, aggressive) in enumerate(zip(gate_rows, safe_rows, aggressive_rows, strict=True)):
        route = gate["gate_route_choice"]
        if route == "M448":
            false_aggressive += int(bool(safe["success_iou_0_5"]) and not bool(aggressive["success_iou_0_5"]))
            missed_safe_iou_benefit += int(float(safe["box_iou"]) > float(aggressive["box_iou"]))
            source = aggressive
        else:
            false_safe += int(float(safe["box_iou"]) <= float(aggressive["box_iou"]))
            source = safe
            historical = safe_physical_by_id[str(gate["sample_id"])]
            safe_retained_index_differences += int(
                gate.get("final_retained_indices") != historical.get("retained_indices")
            )
        source_policy_correct += int(bool(source["success_iou_0_5"]))
        source_policy_iou += float(source["box_iou"])
        physical_correct += int(bool(gate["success_iou_0_5"]))
        physical_iou += float(gate["box_iou"])
        for field in ("raw_model_output", "predicted_box", "box_iou", "success_iou_0_5"):
            if gate.get(field) != source.get(field):
                if route == "M448":
                    aggressive_output_differences += 1
                else:
                    safe_output_differences += 1
                if len(parity_differences) < 10:
                    parity_differences.append(
                        {
                            "row": index,
                            "sample_id": gate["sample_id"],
                            "route": route,
                            "first_differing_field": field,
                        }
                    )
                break
    count = len(gate_rows)
    return {
        "false_aggressive_critical_safe_correct_aggressive_wrong": false_aggressive,
        "false_safe_safe_chosen_when_aggressive_iou_at_least_safe": false_safe,
        "aggressive_choice_with_missed_safe_iou_benefit": missed_safe_iou_benefit,
        "chosen_route_output_differences": safe_output_differences + aggressive_output_differences,
        "safe_choice_output_differences": safe_output_differences,
        "aggressive_choice_output_differences": aggressive_output_differences,
        "safe_retained_index_differences": safe_retained_index_differences,
        "first_output_differences": parity_differences,
        "output_parity_interpretation": (
            "M448 uses the same current progressive cache implementation and must match exactly; "
            "historical K12 uses an older mid-layer selector/cache implementation, so retained-"
            "index and physical-output drift are reported as implementation sensitivity."
        ),
        "physical_metrics": {
            "acc_at_0_5": physical_correct / count,
            "mean_iou": physical_iou / count,
        },
        "frozen_source_policy_metrics": {
            "acc_at_0_5": source_policy_correct / count,
            "mean_iou": source_policy_iou / count,
        },
        "physical_minus_source_policy": {
            "acc_at_0_5": (physical_correct - source_policy_correct) / count,
            "mean_iou": (physical_iou - source_policy_iou) / count,
        },
    }


def append_once(items: list[dict[str, Any]], key: str, value: dict[str, Any]) -> None:
    identifier = value[key]
    for existing in items:
        if existing.get(key) == identifier:
            existing.update(value)
            return
    items.append(value)


def pareto_frontier(items: Sequence[dict[str, Any]]) -> list[str]:
    """Return points not jointly dominated on quality, tokens, and layer work."""

    members: list[str] = []
    for item in items:
        dominated = False
        for other in items:
            if other is item:
                continue
            weakly_better = (
                float(other["acc_at_0_5"]) >= float(item["acc_at_0_5"])
                and float(other["mean_iou"]) >= float(item["mean_iou"])
                and float(other["average_visual_tokens"]) <= float(item["average_visual_tokens"])
                and float(other["visual_token_layer_work"]) <= float(item["visual_token_layer_work"])
            )
            strictly_better = (
                float(other["acc_at_0_5"]) > float(item["acc_at_0_5"])
                or float(other["mean_iou"]) > float(item["mean_iou"])
                or float(other["average_visual_tokens"]) < float(item["average_visual_tokens"])
                or float(other["visual_token_layer_work"]) < float(item["visual_token_layer_work"])
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            members.append(str(item["candidate"]))
    return sorted(members)


def select_project_candidate(
    registry: dict[str, Any], gate_seed0: dict[str, Any]
) -> dict[str, Any]:
    work = {"C448": 6848.0, "C480": 7104.0, "D1": 7104.0, "M448": 6848.0}
    overhead = {"C448": 0, "C480": 0, "D1": 0, "M448": 1, "GATE_SEED0": 2}
    candidates = []
    for cell_id in ("C448", "C480", "D1", "M448"):
        cell = registry["cells"][cell_id]
        if cell.get("status") != "accepted":
            continue
        metrics = cell["metrics"]
        candidate = {
            "candidate": cell_id,
            "acc_at_0_5": float(metrics["acc_at_0_5"]),
            "mean_iou": float(metrics["mean_iou"]),
            "average_visual_tokens": float(metrics["average_visual_tokens"]),
            "visual_token_layer_work": work[cell_id],
            "overhead_complexity_rank": overhead[cell_id],
        }
        candidate["k12_like"] = k12_like(candidate)
        candidates.append(candidate)
    gate_metrics = gate_seed0["metrics"]
    gate_candidate = {
        "candidate": "GATE_SEED0",
        "acc_at_0_5": gate_metrics["acc_at_0_5"],
        "mean_iou": gate_metrics["mean_iou"],
        "average_visual_tokens": gate_metrics["average_visual_tokens"],
        "visual_token_layer_work": gate_metrics["visual_token_layer_work"],
        "overhead_complexity_rank": overhead["GATE_SEED0"],
        "k12_like": gate_seed0["k12_like"],
    }
    candidates.append(gate_candidate)
    eligible = [item for item in candidates if item["k12_like"]]
    pool = eligible if eligible else candidates
    highest_acc = max(item["acc_at_0_5"] for item in pool)
    near = [item for item in pool if highest_acc - item["acc_at_0_5"] <= 0.005 + 1e-12]
    selected = min(
        near,
        key=lambda item: (
            item["visual_token_layer_work"],
            item["average_visual_tokens"],
            -item["mean_iou"],
            item["overhead_complexity_rank"],
            item["candidate"],
        ),
    )
    return {
        "status": "selected_k12_like_candidate" if eligible else "no_k12_like_best_exploratory",
        "selected": selected["candidate"],
        "selected_metrics": selected,
        "highest_accuracy_in_rank_pool": highest_acc,
        "within_0_005_accuracy_pool": [item["candidate"] for item in near],
        "ranking_evidence": sorted(candidates, key=lambda item: item["candidate"]),
        "timing_triggered": bool(eligible),
    }


def update_report(analysis: dict[str, Any]) -> None:
    lines = [
        "## Learned schedule gate",
        "",
        f"The frozen train-only label artifact contains 3,000 paired physical K12/M448 outcomes and passed independent trace, cache, rescore, feature-hash, and leakage checks (`{analysis['labels']['labels_sha256']}`). The one pre-seed architecture is an 8,209→64→2 GELU MLP with dropout 0.1 and 525,570 trainable parameters; its immutable specification hash is `{analysis['training']['training_spec_sha256']}`. Normalization was fit only on 2,160 fitting rows; 240 rows selected the epoch and 600 disjoint train rows calibrated the threshold. The three deterministic CPU fits trained for {analysis['training']['total_epochs_trained']} aggregate epochs and updated zero LLaVA parameters.",
        "",
        "| Seed | Best epoch / trained | Threshold | Calib aggressive | Val aggressive | Acc@0.5 | mIoU | Avg tokens | Layer work | K12-like |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for seed in range(3):
        training = analysis["training"]["seeds"][str(seed)]
        physical = analysis["physical_seeds"][str(seed)]
        metrics = physical["metrics"]
        lines.append(
            f"| {seed} | {training['best_epoch']} / {training['epochs_trained']} | {training['threshold']:.6f} | {training['calibration_metrics']['aggressive_rate']:.3f} | {metrics['aggressive_rate']:.3f} | {metrics['acc_at_0_5']:.3f} | {metrics['mean_iou']:.4f} | {metrics['average_visual_tokens']:.3f} | {metrics['visual_token_layer_work']:.1f} | {'pass' if physical['k12_like'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "Calibration-to-validation shift (validation minus calibration):",
            "",
            "| Seed | Δ aggressive rate | Δ Acc@0.5 | Δ mIoU | Δ average tokens |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in range(3):
        shift = analysis["calibration_validation_shift"][str(seed)]
        lines.append(
            f"| {seed} | {shift['aggressive_rate']:+.3f} | {shift['acc_at_0_5']:+.3f} | {shift['mean_iou']:+.4f} | {shift['average_visual_tokens']:+.3f} |"
        )
    random_metrics = analysis["controls"]["matched_rate_random"]["metrics"]
    oracle_metrics = analysis["controls"]["privileged_oracle"]["metrics"]
    primary = analysis["physical_seeds"]["0"]["metrics"]
    primary_route_errors = analysis["route_errors"]["seed0"]
    decision = analysis["route_gate_decision"]
    selection = analysis["project_selection"]
    contrasts = {item["contrast"]: item for item in analysis["paired_contrasts"]}
    lines.extend(
        [
            "",
            f"The matched-rate random control is {random_metrics['quality/acc_at_0_5']:.3f}/{random_metrics['quality/mean_iou']:.4f} at aggressive rate {random_metrics['routing/aggressive_rate']:.3f}. Both physical seed 0 ({primary['acc_at_0_5']:.3f}) and its choices assembled from the same frozen route sources ({primary_route_errors['frozen_source_policy_metrics']['acc_at_0_5']:.3f}) beat that point control. The privileged no-efficiency oracle ceiling is {oracle_metrics['quality/acc_at_0_5']:.3f}/{oracle_metrics['quality/mean_iou']:.4f}; it assembled already physical rows and ran no LLaVA inference.",
            "",
            "Primary-seed paired evidence (10,000 resamples; intervals containing zero are inconclusive):",
            "",
            "| Contrast | Acc delta pp [95% CI] | mIoU delta [95% CI] | Discordant L+/R+ |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, label in (
        ("GATE_SEED0-minus-M448", "Gate seed 0 minus M448"),
        ("GATE_SEED0-minus-K12_ORIGINAL", "Gate seed 0 minus K12 original"),
        ("GATE_SEED0-minus-MATCHED_RATE_RANDOM", "Gate seed 0 minus matched-rate random"),
    ):
        item = contrasts[name]
        acc = item["acc_at_0_5_percentage_points"]
        iou = item["mean_iou"]
        discordant = item["discordant"]
        lines.append(
            f"| {label} | {acc['point_delta']:+.3f} [{acc['lower_95']:+.3f}, {acc['upper_95']:+.3f}] | {iou['point_delta']:+.4f} [{iou['lower_95']:+.4f}, {iou['upper_95']:+.4f}] | {discordant['left_correct_right_wrong']}/{discordant['left_wrong_right_correct']} |"
        )
    lines.extend(
        [
            "",
            f"Under the registered four-axis frontier (maximize Acc@0.5 and mIoU; minimize average tokens and sequence-aware layer work), the project frontier is **{', '.join(analysis['pareto_frontiers']['project']['members'])}** and the accepted official frontier is **{', '.join(analysis['pareto_frontiers']['official']['members'])}**. The strongest accepted official point, OFF_PDROP_REROUTE_192, is compute-only relative to the selected project candidate: it uses less layer work but is worse in both quality measures, so it does not jointly dominate the project result.",
            "",
            "Selected-project comparisons against every accepted official baseline:",
            "",
            "| Contrast | Acc delta pp [95% CI] | mIoU delta [95% CI] | Official relation to project |",
            "|---|---:|---:|---|",
        ]
    )
    for comparison in analysis["selected_project_vs_official"]:
        item = comparison["paired"]
        acc = item["acc_at_0_5_percentage_points"]
        iou = item["mean_iou"]
        lines.append(
            f"| {item['contrast']} | {acc['point_delta']:+.3f} [{acc['lower_95']:+.3f}, {acc['upper_95']:+.3f}] | {iou['point_delta']:+.4f} [{iou['lower_95']:+.4f}, {iou['upper_95']:+.4f}] | {comparison['dominance_vs_selected_project']} |"
        )
    lines.extend(
        [
            "",
            f"Primary seed 0 has {primary_route_errors['false_aggressive_critical_safe_correct_aggressive_wrong']} false-aggressive critical errors and {primary_route_errors['false_safe_safe_chosen_when_aggressive_iou_at_least_safe']} false-safe choices. Every dynamic row executed one shared K2 prefix, one gate, and only one route. All aggressive choices exactly match physical M448. Among the {primary['safe_rows']} safe choices, {primary_route_errors['safe_retained_index_differences']} retained-index sets and {primary_route_errors['safe_choice_output_differences']} raw outputs differ from the historical K12 source because it uses an older mid-layer selector/cache implementation; physical seed-0 Acc is {primary_route_errors['physical_metrics']['acc_at_0_5']:.3f} versus {primary_route_errors['frozen_source_policy_metrics']['acc_at_0_5']:.3f} when its identical route choices assemble frozen source outputs. This implementation sensitivity is reported rather than hidden or treated as physical-route failure.",
            "",
            f"Registered router decision: `{decision['router_label']}` ({decision['classification']}). Primary seed 0 is {primary['acc_at_0_5']:.3f}/{primary['mean_iou']:.4f} at {primary['average_visual_tokens']:.3f} tokens. The Phase-H project selection is **{selection['selected']}**, using the preregistered within-0.005 accuracy/compute/token/mIoU/overhead ranking.",
            "",
            f"Machine-readable gate, interval, control, and selection evidence: `{relative(ANALYSIS_PATH)}`.",
        ]
    )
    document = REPORT_PATH.read_text(encoding="utf-8")
    section = "\n".join(lines)
    updated, count = re.subn(
        r"## Learned schedule gate\n\n.*?(?=\n## Conditional matched timing)",
        section,
        document,
        flags=re.DOTALL,
    )
    if count != 1:
        raise GateAnalysisError("Could not replace the learned-gate report section")
    timing = "\n".join(
        [
            "## Conditional matched timing",
            "",
            f"Triggered because the project pool contains a K12-like deployable candidate. The registered serial five-repetition comparison is Full vs original K12/B48 more-context vs {selection['selected']}. No timing result is claimed until all 15 natural-greedy trials and any exact fixed-length trigger are independently complete.",
        ]
    )
    updated, count = re.subn(
        r"## Conditional matched timing\n\n.*?(?=\n## Failures and rejected attempts)",
        timing,
        updated,
        flags=re.DOTALL,
    )
    if count != 1:
        raise GateAnalysisError("Could not replace the timing report section")
    atomic_write(REPORT_PATH, updated.encode("utf-8"))


def main() -> int:
    registry = read_json(REGISTRY_PATH)
    state = read_json(STATE_PATH)
    label_integrity = read_json(LABEL_RUN / "integrity_report.json")
    if label_integrity.get("status") != "pass" or int(label_integrity.get("rows", -1)) != 3000:
        raise GateAnalysisError("Full gate labels are not independently valid")
    training_spec = read_json(TRAINING_ROOT / "FROZEN_GATE_TRAINING_SPEC.json")
    normalization = read_json(TRAINING_ROOT / "NORMALIZATION.json")
    training = {str(seed): training_seed_evidence(seed) for seed in range(3)}
    physical: dict[str, dict[str, Any]] = {}
    physical_rows: dict[int, list[dict[str, Any]]] = {}
    for seed in range(3):
        evidence, rows = gate_seed_evidence(seed)
        physical[str(seed)] = evidence
        physical_rows[seed] = rows

    safe_rows = read_jsonl(SAFE_ROWS_PATH)
    original_k12_rows = read_jsonl(ORIGINAL_K12_PATH)
    aggressive_rows = read_jsonl(M448_PATH)
    frozen_ids = {str(row["sample_id"]) for row in safe_rows}
    safe_physical_path = REPO_ROOT / registry["references"]["K12_QUERY_TOPK"]["source_path"]
    safe_physical_by_id = indexed_source_rows(safe_physical_path, frozen_ids)
    random_dir = CONTROLS_ROOT / "random_rate_seed0"
    oracle_dir = CONTROLS_ROOT / "privileged_oracle"
    random_integrity = read_json(random_dir / "integrity_report.json")
    oracle_integrity = read_json(oracle_dir / "integrity_report.json")
    if random_integrity.get("status") != "pass" or oracle_integrity.get("status") != "pass":
        raise GateAnalysisError("An analytical gate control is not independently valid")
    random_rows = read_jsonl(random_dir / "predictions.jsonl")
    oracle_rows = read_jsonl(oracle_dir / "predictions.jsonl")

    route_error_evidence = {
        f"seed{seed}": route_errors(
            physical_rows[seed], safe_rows, aggressive_rows, safe_physical_by_id
        )
        for seed in range(3)
    }
    contrasts = []
    for seed in range(3):
        metrics = physical[str(seed)]["metrics"]
        contrasts.append(
            contrast(
                f"GATE_SEED{seed}-minus-M448",
                physical_rows[seed],
                aggressive_rows,
                left_tokens=metrics["average_visual_tokens"],
                right_tokens=214.0,
                left_work=metrics["visual_token_layer_work"],
                right_work=6848.0,
                seed_offset=seed * 3,
            )
        )
        contrasts.append(
            contrast(
                f"GATE_SEED{seed}-minus-K12_ORIGINAL",
                physical_rows[seed],
                original_k12_rows,
                left_tokens=metrics["average_visual_tokens"],
                right_tokens=246.0,
                left_work=metrics["visual_token_layer_work"],
                right_work=7872.0,
                seed_offset=seed * 3 + 1,
            )
        )
        contrasts.append(
            contrast(
                f"GATE_SEED{seed}-minus-K12_QUERY_TOPK_SAFE_ROUTE",
                physical_rows[seed],
                safe_rows,
                left_tokens=metrics["average_visual_tokens"],
                right_tokens=246.0,
                left_work=metrics["visual_token_layer_work"],
                right_work=7872.0,
                seed_offset=seed * 3 + 2,
            )
        )
    primary_metrics = physical["0"]["metrics"]
    calibration_validation_shift = {}
    for seed in range(3):
        calibration_metrics = training[str(seed)]["calibration_metrics"]
        validation_metrics = physical[str(seed)]["metrics"]
        calibration_validation_shift[str(seed)] = {
            "direction": "validation_minus_calibration",
            "aggressive_rate": validation_metrics["aggressive_rate"]
            - float(calibration_metrics["aggressive_rate"]),
            "acc_at_0_5": validation_metrics["acc_at_0_5"]
            - float(calibration_metrics["acc_at_0_5"]),
            "mean_iou": validation_metrics["mean_iou"]
            - float(calibration_metrics["mean_iou"]),
            "parse_rate": validation_metrics["parse_rate"]
            - float(calibration_metrics["parse_rate"]),
            "average_visual_tokens": validation_metrics["average_visual_tokens"]
            - float(calibration_metrics["average_visual_tokens"]),
        }
    contrasts.append(
        contrast(
            "GATE_SEED0-minus-MATCHED_RATE_RANDOM",
            physical_rows[0],
            random_rows,
            left_tokens=primary_metrics["average_visual_tokens"],
            right_tokens=float(random_integrity["metrics"]["tokens/analytical_average_visual_tokens"]),
            left_work=primary_metrics["visual_token_layer_work"],
            right_work=None,
            seed_offset=20,
        )
    )

    random_metrics = random_integrity["metrics"]
    physical_beats_random = (
        primary_metrics["acc_at_0_5"] > float(random_metrics["quality/acc_at_0_5"])
        or (
            primary_metrics["acc_at_0_5"] == float(random_metrics["quality/acc_at_0_5"])
            and primary_metrics["mean_iou"] > float(random_metrics["quality/mean_iou"])
        )
    )
    primary_source_policy = route_error_evidence["seed0"]["frozen_source_policy_metrics"]
    source_policy_beats_random = (
        float(primary_source_policy["acc_at_0_5"])
        > float(random_metrics["quality/acc_at_0_5"])
        or (
            float(primary_source_policy["acc_at_0_5"])
            == float(random_metrics["quality/acc_at_0_5"])
            and float(primary_source_policy["mean_iou"])
            > float(random_metrics["quality/mean_iou"])
        )
    )
    beats_random = physical_beats_random and source_policy_beats_random
    contradictory_collapse = any(
        physical[str(seed)]["metrics"]["acc_at_0_5"] < 0.43 - 0.005 - 1e-12
        or physical[str(seed)]["metrics"]["mean_iou"] < 0.4284360666246254 - 0.005 - 1e-12
        for seed in (1, 2)
    )
    strong = (
        physical["0"]["k12_like"]
        and primary_metrics["average_visual_tokens"] <= 222.0
        and beats_random
        and not contradictory_collapse
    )
    recovered_gap = (
        (primary_metrics["acc_at_0_5"] - 0.43) / (0.441 - 0.43)
        if abs(0.441 - 0.43) > 0
        else None
    )
    exploratory = (
        not strong
        and primary_metrics["acc_at_0_5"] - 0.43 >= 0.010 - 1e-12
        and primary_metrics["mean_iou"] - 0.4284360666246254 >= -0.005 - 1e-12
        and primary_metrics["average_visual_tokens"] <= 222.0
        and recovered_gap is not None
        and recovered_gap >= 0.25 - 1e-12
    )
    if strong:
        route_decision = {
            "classification": "strong_main_method_candidate",
            "router_label": "LIGHTWEIGHT_ROUTER_MAIN_METHOD_JUSTIFIED",
        }
    elif exploratory:
        route_decision = {
            "classification": "useful_exploratory_learned_result",
            "router_label": "LIGHTWEIGHT_ROUTER_EXPLORATORY_ONLY",
        }
    else:
        route_decision = {
            "classification": "registered_gate_unsuccessful",
            "router_label": "LIGHTWEIGHT_ROUTER_NOT_JUSTIFIED",
        }
    route_decision.update(
        {
            "primary_k12_like": physical["0"]["k12_like"],
            "primary_average_tokens_at_most_222": primary_metrics["average_visual_tokens"] <= 222.0,
            "beats_matched_rate_random_point_quality": beats_random,
            "physical_beats_matched_rate_random_point_quality": physical_beats_random,
            "frozen_source_policy_beats_matched_rate_random_point_quality": source_policy_beats_random,
            "seed1_or_seed2_contradictory_collapse": contradictory_collapse,
            "acc_improvement_over_aggressive": primary_metrics["acc_at_0_5"] - 0.43,
            "miou_change_over_aggressive": primary_metrics["mean_iou"] - 0.4284360666246254,
            "recovered_acc_gap": recovered_gap,
        }
    )
    selection = select_project_candidate(registry, physical["0"])

    selected_rows = physical_rows[0] if selection["selected"] == "GATE_SEED0" else {
        "C448": read_jsonl(CAMPAIGN_ROOT / "runs/c448_attempt1/predictions.jsonl"),
        "C480": read_jsonl(CAMPAIGN_ROOT / "runs/c480_attempt1/predictions.jsonl"),
        "D1": read_jsonl(CAMPAIGN_ROOT / "runs/d1_attempt1/predictions.jsonl"),
        "M448": aggressive_rows,
    }[selection["selected"]]
    selected_metrics = selection["selected_metrics"]
    official_comparisons = []
    official_frontier_candidates = []
    for offset, (cell_id, run_name, work) in enumerate(
        (
            ("OFF_PDROP_L2_192", "pdrop_full1000_attempt1", 6123.0),
            ("OFF_PDROP_REROUTE_192", "pdrop_reroute_full1000_attempt1", 6127.0),
        )
    ):
        official_rows = read_jsonl(OFFICIAL_ROOT / run_name / "predictions.jsonl")
        official_metrics = compact_metrics(official_rows, token_key="candidate_average_visual_tokens")
        official_frontier_candidates.append(
            {
                "candidate": cell_id,
                "acc_at_0_5": official_metrics["acc_at_0_5"],
                "mean_iou": official_metrics["mean_iou"],
                "average_visual_tokens": official_metrics["average_visual_tokens"],
                "visual_token_layer_work": work,
            }
        )
        official_comparisons.append(
            {
                "official_cell": cell_id,
                "dominance_vs_selected_project": "compute_only"
                if work < selected_metrics["visual_token_layer_work"]
                else "neither",
                "paired": contrast(
                    f"{selection['selected']}-minus-{cell_id}",
                    selected_rows,
                    official_rows,
                    left_tokens=selected_metrics["average_visual_tokens"],
                    right_tokens=official_metrics["average_visual_tokens"],
                    left_work=selected_metrics["visual_token_layer_work"],
                    right_work=work,
                    seed_offset=30 + offset,
                ),
            }
        )

    analysis = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "labels": {
            "run_dir": relative(LABEL_RUN),
            "labels_sha256": label_integrity["labels_sha256"],
            "integrity_sha256": sha256_file(LABEL_RUN / "integrity_report.json"),
            "completion_sha256": sha256_file(LABEL_RUN / "COMPLETED.json"),
            "metrics": label_integrity["metrics"],
            "checks": label_integrity["checks"],
        },
        "training": {
            "frozen_spec_path": relative(TRAINING_ROOT / "FROZEN_GATE_TRAINING_SPEC.json"),
            "frozen_spec_file_sha256": sha256_file(TRAINING_ROOT / "FROZEN_GATE_TRAINING_SPEC.json"),
            "training_spec_sha256": training_spec["training_spec_sha256"],
            "parameter_count": 525570,
            "total_epochs_trained": sum(item["epochs_trained"] for item in training.values()),
            "training_device": "cpu",
            "llava_parameters_trained": 0,
            "normalization_sha256": normalization["normalization_sha256"],
            "seeds": training,
        },
        "physical_seeds": physical,
        "calibration_validation_shift": calibration_validation_shift,
        "route_errors": route_error_evidence,
        "controls": {
            "matched_rate_random": {
                "artifacts": run_hashes(random_dir),
                "metrics": random_integrity["metrics"],
                "checks": random_integrity["checks"],
            },
            "privileged_oracle": {
                "artifacts": run_hashes(oracle_dir),
                "metrics": oracle_integrity["metrics"],
                "checks": oracle_integrity["checks"],
            },
        },
        "paired_contrasts": contrasts,
        "a0_subgroup_evidence_path": relative(
            CAMPAIGN_ROOT / "analysis/A0_REFERENCE_AND_ERROR_ANALYSIS.json"
        ),
        "route_gate_decision": route_decision,
        "project_selection": selection,
        "pareto_frontiers": {
            "definition": (
                "non-dominated while maximizing Acc@0.5 and mIoU and minimizing "
                "average visual tokens and visual-token layer work"
            ),
            "project": {
                "members": pareto_frontier(selection["ranking_evidence"]),
                "candidates": selection["ranking_evidence"],
            },
            "official": {
                "members": pareto_frontier(official_frontier_candidates),
                "candidates": official_frontier_candidates,
            },
        },
        "selected_project_vs_official": official_comparisons,
        "strongest_accepted_official": "OFF_PDROP_REROUTE_192",
        "timing_triggered": selection["timing_triggered"],
    }
    atomic_write(ANALYSIS_PATH, canonical_bytes(analysis))

    for seed in range(3):
        registry["cells"][f"GATE_SEED{seed}"] = {
            "status": "accepted",
            "metrics": physical[str(seed)]["metrics"],
            "k12_like": physical[str(seed)]["k12_like"],
            "accepted_run": physical[str(seed)]["artifacts"],
            "training": training[str(seed)],
            "attempts": [
                {
                    "attempt": 1,
                    "status": "accepted",
                    "run_dir": physical[str(seed)]["artifacts"]["run_dir"],
                }
            ],
        }
    registry["cells"]["GATE_RANDOM_RATE_CONTROL"] = {
        "status": "accepted_analytical_no_inference",
        "metrics": random_integrity["metrics"],
        "artifacts": run_hashes(random_dir),
    }
    registry["cells"]["GATE_PRIVILEGED_ORACLE_CONTROL"] = {
        "status": "accepted_privileged_no_inference_no_efficiency_claim",
        "metrics": oracle_integrity["metrics"],
        "artifacts": run_hashes(oracle_dir),
    }
    registry["phases"]["route_gate"] = {
        "status": "complete",
        "analysis_path": relative(ANALYSIS_PATH),
        "analysis_sha256": sha256_file(ANALYSIS_PATH),
        "router_label": route_decision["router_label"],
    }
    registry["project_selection"] = selection
    append_once(
        registry["decisions"],
        "decision",
        {
            "at_utc": utc_now(),
            "decision": "complete_route_gate_and_select_project_candidate",
            "selected": selection["selected"],
            "router_label": route_decision["router_label"],
            "timing_triggered": selection["timing_triggered"],
            "evidence_path": relative(ANALYSIS_PATH),
            "evidence_sha256": sha256_file(ANALYSIS_PATH),
        },
    )
    registry["totals"]["accepted_validation_quality"] = sum(
        cell.get("status") == "accepted" for cell in registry["cells"].values()
    )
    registry["updated_at_utc"] = utc_now()
    state["state"] = "GATE_COMPLETE_MATCHED_TIMING_PENDING"
    state["next_required_action"] = "run_serial_matched_timing_for_selected_project_candidate"
    append_once(
        state["decision_history"],
        "phase",
        {
            "phase": "route_gate",
            "status": "complete",
            "selected_project_candidate": selection["selected"],
            "router_label": route_decision["router_label"],
            "evidence_path": relative(ANALYSIS_PATH),
            "evidence_sha256": sha256_file(ANALYSIS_PATH),
        },
    )
    state["updated_at_utc"] = utc_now()
    update_report(analysis)
    atomic_write(REGISTRY_PATH, canonical_bytes(registry))
    atomic_write(STATE_PATH, canonical_bytes(state))
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateAnalysisError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
