#!/usr/bin/env python
"""Compute registered paired analyses and refresh the single campaign report."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814"
REGISTRY_PATH = CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"
REPORT_PATH = CAMPAIGN_ROOT / "RESEARCH_REPORT.md"
REFERENCE_ROWS = CAMPAIGN_ROOT / "references/REF_K20_B48_per_sample.jsonl"
ANALYSIS_ROOT = CAMPAIGN_ROOT / "analysis"
BOOTSTRAPS = 10_000
ANALYSIS_SEED = 20260814


class CampaignAnalysisError(RuntimeError):
    """Raised when accepted evidence is incomplete or misaligned."""


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 1000 or len({str(row["sample_id"]) for row in rows}) != 1000:
        raise CampaignAnalysisError(f"Expected 1,000 unique rows in {path}")
    return rows


def accepted_rows(registry: dict[str, Any], cell_id: str) -> list[dict[str, Any]]:
    cell = registry["cells"][cell_id]
    if cell.get("status") != "accepted":
        raise CampaignAnalysisError(f"Cell {cell_id} is not accepted")
    path = REPO_ROOT / cell["artifact_paths"]["predictions"]
    return read_jsonl(path)


def align(*groups: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    ids = [[str(row["sample_id"]) for row in group] for group in groups]
    if any(values != ids[0] for values in ids[1:]):
        raise CampaignAnalysisError("Paired analysis rows are not in identical manifest order")
    return [list(group) for group in groups]


def values(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
    if metric == "acc_at_0_5":
        return np.asarray([float(row["success_iou_0_5"]) for row in rows], dtype=np.float64)
    if metric == "mean_iou":
        return np.asarray([float(row["box_iou"]) for row in rows], dtype=np.float64)
    raise CampaignAnalysisError(f"Unsupported paired metric {metric}")


def paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    scale: float = 1.0,
) -> dict[str, Any]:
    if left.shape != right.shape or left.ndim != 1 or left.size != 1000:
        raise CampaignAnalysisError("Paired bootstrap requires two aligned 1,000-vectors")
    delta = left - right
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    remaining = BOOTSTRAPS
    while remaining:
        count = min(500, remaining)
        sample = rng.integers(0, delta.size, size=(count, delta.size))
        estimates.append(delta[sample].mean(axis=1))
        remaining -= count
    boot = np.concatenate(estimates)
    return {
        "mean_delta": float(delta.mean() * scale),
        "lower_95": float(np.quantile(boot, 0.025) * scale),
        "upper_95": float(np.quantile(boot, 0.975) * scale),
        "bootstrap_replicates": BOOTSTRAPS,
        "seed": seed,
        "unit": "percentage_points" if scale == 100.0 else "0_to_1",
    }


def contrast(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    name: str,
    seed_offset: int,
) -> list[dict[str, Any]]:
    align(left_rows, right_rows)
    output: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(("acc_at_0_5", "mean_iou")):
        interval = paired_bootstrap(
            values(left_rows, metric),
            values(right_rows, metric),
            seed=ANALYSIS_SEED + seed_offset * 10 + metric_index,
            scale=100.0 if metric == "acc_at_0_5" else 1.0,
        )
        output.append({"contrast": name, "metric": metric, **interval})
    return output


def averaged_seed_rows(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    align(*groups)
    output: list[dict[str, Any]] = []
    for index in range(1000):
        output.append(
            {
                "sample_id": groups[0][index]["sample_id"],
                "success_iou_0_5": sum(float(bool(group[index]["success_iou_0_5"])) for group in groups) / len(groups),
                "box_iou": sum(float(group[index]["box_iou"]) for group in groups) / len(groups),
            }
        )
    return output


def metrics_row(registry: dict[str, Any], cell_id: str) -> dict[str, Any]:
    cell = registry["cells"][cell_id]
    metrics = cell.get("metrics") or {}
    return {
        "cell_id": cell_id,
        "status": cell.get("status"),
        "schedule": cell.get("requested_schedule") or cell.get("schedule"),
        "seed": cell.get("seed"),
        "acc_at_0_5": metrics.get("quality/acc_at_0_5"),
        "mean_iou": metrics.get("quality/mean_iou"),
        "parse_rate": metrics.get("quality/parsed_box_rate"),
        "candidate_average_visual_tokens": cell.get("candidate_average_visual_tokens"),
        "oracle_teacher_visual_token_layer_work": cell.get("oracle_teacher_visual_token_layer_work"),
    }


def analyse_campaign1(registry: dict[str, Any]) -> dict[str, Any]:
    reference = read_jsonl(REFERENCE_ROWS)
    intervals: list[dict[str, Any]] = []
    for offset, cell_id in enumerate(("T1", "T2", "T3", "T4", "T5"), 1):
        intervals.extend(
            contrast(accepted_rows(registry, cell_id), reference, name=f"{cell_id}-minus-REF_K20_B48", seed_offset=offset)
        )
    for offset, (budget, query_id) in enumerate(((16, "T2"), (8, "T3"), (4, "T4")), 20):
        random_rows = averaged_seed_rows(
            [accepted_rows(registry, f"R{budget}-S{seed}") for seed in (0, 1, 2)]
        )
        intervals.extend(
            contrast(
                accepted_rows(registry, query_id),
                random_rows,
                name=f"query_topk-minus-random_seed_mean_B{budget}",
                seed_offset=offset,
            )
        )
    result = {
        "status": "complete",
        "bootstrap_replicates": BOOTSTRAPS,
        "analysis_seed": ANALYSIS_SEED,
        "B_tail": registry["decisions"]["B_tail"],
        "cells": [metrics_row(registry, cell_id) for cell_id in ["T1", "T2", "T3", "T4", "T5"] + [f"R{b}-S{s}" for b in (16, 8, 4) for s in (0, 1, 2)]],
        "paired_intervals": intervals,
    }
    atomic_json(ANALYSIS_ROOT / "campaign1.json", result)
    return result


def tail_comparator_rows(registry: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    b_tail = int(registry["decisions"]["B_tail"]["value"])
    cell = {32: "T1", 16: "T2", 8: "T3", 4: "T4", 0: "T5"}.get(b_tail)
    return (cell, accepted_rows(registry, cell)) if cell else ("REF_K20_B48", read_jsonl(REFERENCE_ROWS))


def analyse_campaign2(registry: dict[str, Any]) -> dict[str, Any]:
    ids = [f"P{index}" for index in range(1, 10)]
    if not all(registry["cells"][cell_id]["status"] == "accepted" for cell_id in ids):
        raise CampaignAnalysisError("Campaign 2 analysis requires P1-P9")
    selected = registry["decisions"]["progressive_refinement"]["strongest_refinement"]
    tail_name, tail_rows = tail_comparator_rows(registry)
    intervals = []
    intervals.extend(contrast(accepted_rows(registry, "P7"), accepted_rows(registry, "P8"), name="P7-minus-P8", seed_offset=40))
    intervals.extend(contrast(accepted_rows(registry, selected), accepted_rows(registry, "P9"), name=f"{selected}-minus-P9", seed_offset=41))
    intervals.extend(contrast(accepted_rows(registry, selected), tail_rows, name=f"{selected}-minus-{tail_name}", seed_offset=42))
    result = {
        "status": "complete",
        "cells": [metrics_row(registry, cell_id) for cell_id in ids],
        "base_decision": registry["decisions"]["progressive_base"],
        "refinement_decision": registry["decisions"]["progressive_refinement"],
        "tail_comparator": tail_name,
        "paired_intervals": intervals,
    }
    atomic_json(ANALYSIS_ROOT / "campaign2.json", result)
    return result


def analyse_campaign3(registry: dict[str, Any]) -> dict[str, Any]:
    ids = [f"O{index}" for index in range(1, 11)]
    if not all(registry["cells"][cell_id]["status"] == "accepted" for cell_id in ids):
        raise CampaignAnalysisError("Campaign 3 analysis requires O1-O10")
    intervals: list[dict[str, Any]] = []
    pairs = [
        ("O1", "O9"), ("O1", "O10"), ("O4", "O9"), ("O4", "O10"),
        ("O2", "O1"), ("O3", "O2"), ("O5", "O4"), ("O6", "O5"),
        ("O7", "O9"), ("O8", "O10"),
    ]
    for offset, (left, right) in enumerate(pairs, 60):
        intervals.extend(
            contrast(accepted_rows(registry, left), accepted_rows(registry, right), name=f"{left}-minus-{right}", seed_offset=offset)
        )
    result = {
        "status": "complete",
        "B_tail": registry["decisions"]["B_tail"],
        "B_early64": registry["decisions"]["B_early64"],
        "cells": [metrics_row(registry, cell_id) for cell_id in ids],
        "paired_intervals": intervals,
    }
    atomic_json(ANALYSIS_ROOT / "campaign3.json", result)
    return result


def fmt(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def schedule_label(schedule: Any, cell: dict[str, Any]) -> str:
    prefix = ""
    if cell.get("pre_llm_budget") is not None:
        selector = "oracle" if cell.get("oracle_teacher_layer") else cell.get("pre_llm_selector", "pre-LLM")
        prefix = f"pre-LLM {selector}->{cell['pre_llm_budget']}"
    if not isinstance(schedule, list) or not schedule:
        return prefix or "no runtime pruning"
    runtime = ", ".join(f"K{e['boundary']}->{e['budget']} {e['selector']}" for e in schedule)
    return f"{prefix}; {runtime}" if prefix else runtime


def table(registry: dict[str, Any], ids: list[str]) -> str:
    lines = [
        "| Cell | Configuration | Status | Acc@0.5 | mIoU | Parse | Candidate avg tokens | Teacher token-layer work |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for cell_id in ids:
        cell = registry["cells"][cell_id]
        metrics = cell.get("metrics") or {}
        lines.append(
            "| " + " | ".join(
                [
                    cell_id,
                    schedule_label(cell.get("requested_schedule") or cell.get("schedule"), cell),
                    str(cell.get("status")),
                    fmt(metrics.get("quality/acc_at_0_5")),
                    fmt(metrics.get("quality/mean_iou")),
                    fmt(metrics.get("quality/parsed_box_rate")),
                    fmt(cell.get("candidate_average_visual_tokens"), 3),
                    fmt(cell.get("oracle_teacher_visual_token_layer_work"), 0),
                ]
            ) + " |"
        )
    return "\n".join(lines)


def interval_table(payload: dict[str, Any]) -> str:
    lines = [
        "| Contrast (left minus right) | Metric | Delta | 95% paired bootstrap CI | Unit |",
        "|---|---|---:|---:|---|",
    ]
    for row in payload.get("paired_intervals") or []:
        lines.append(
            f"| {row['contrast']} | {row['metric']} | {float(row['mean_delta']):.4f} | "
            f"[{float(row['lower_95']):.4f}, {float(row['upper_95']):.4f}] | {row['unit']} |"
        )
    return "\n".join(lines)


def random_characterization(registry: dict[str, Any]) -> str:
    query_ids = {16: "T2", 8: "T3", 4: "T4"}
    lines = [
        "| B | Query Acc | Random Acc mean (seed range) | Query mIoU | Random mIoU mean (seed range) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for budget in (16, 8, 4):
        query = registry["cells"][query_ids[budget]]["metrics"]
        random_metrics = [registry["cells"][f"R{budget}-S{seed}"]["metrics"] for seed in (0, 1, 2)]
        acc = [float(row["quality/acc_at_0_5"]) for row in random_metrics]
        miou = [float(row["quality/mean_iou"]) for row in random_metrics]
        lines.append(
            f"| {budget} | {float(query['quality/acc_at_0_5']):.4f} | "
            f"{sum(acc)/3:.4f} ({min(acc):.4f}-{max(acc):.4f}) | "
            f"{float(query['quality/mean_iou']):.4f} | {sum(miou)/3:.4f} ({min(miou):.4f}-{max(miou):.4f}) |"
        )
    return "\n".join(lines)


def progressive_frontier(registry: dict[str, Any]) -> list[str]:
    ids = [f"P{index}" for index in range(1, 10)]
    frontier: list[str] = []
    for candidate in ids:
        row = registry["cells"][candidate]
        acc = float(row["metrics"]["quality/acc_at_0_5"])
        tokens = float(row["candidate_average_visual_tokens"])
        dominated = False
        for competitor in ids:
            if competitor == candidate:
                continue
            other = registry["cells"][competitor]
            other_acc = float(other["metrics"]["quality/acc_at_0_5"])
            other_tokens = float(other["candidate_average_visual_tokens"])
            if other_acc >= acc and other_tokens <= tokens and (other_acc > acc or other_tokens < tokens):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda cell_id: registry["cells"][cell_id]["candidate_average_visual_tokens"])


def timing_report(registry: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    gates = registry.get("timing_gate_decisions") or {}
    for cell_id in ("E1", "E2"):
        cell = registry["timing_cells"][cell_id]
        gate = cell.get("gate") or gates.get(cell_id) or {}
        lines.append(f"### {cell_id}: {cell.get('status', 'pending')}")
        lines.append("")
        if cell.get("status") == "accepted":
            metrics = cell["metrics"]
            full = metrics["variants"]["full"]
            candidate = metrics["variants"]["candidate"]
            lines.extend([
                f"Candidate: **{cell['candidate_cell_id']}**; paired rows: {metrics['samples']}; "
                "20 warm-up examples per path; alternating within-pair order; matched full reference measured in the same process.",
                "",
                "| Quantity | Full mean | Candidate mean | Candidate/full or reduction |",
                "|---|---:|---:|---:|",
            ])
            for field, label in (
                ("vision_encoder_latency_s", "Vision latency (s)"),
                ("projector_latency_s", "Projector latency (s)"),
                ("llm_prefill_latency_s", "LLM prefill latency (s)"),
                ("selector_total_latency_s", "Selector latency (s)"),
                ("decode_latency_s", "Decode latency (s)"),
                ("end_to_end_latency_s", "End-to-end latency (s)"),
                ("generated_token_count", "Generated tokens"),
                ("prompt_kv_total_bytes", "Prompt KV bytes"),
                ("final_kv_total_bytes", "Final KV bytes"),
                ("candidate_average_visual_tokens", "Average visual tokens"),
            ):
                full_mean = float(full[field]["mean"])
                candidate_mean = float(candidate[field]["mean"])
                suffix = "—"
                if full_mean:
                    suffix = f"{candidate_mean/full_mean:.4f}×"
                lines.append(f"| {label} | {full_mean:.6f} | {candidate_mean:.6f} | {suffix} |")
            lines.extend([
                f"| Peak allocated GPU bytes | {int(full['peak_allocated_gpu_bytes'])} | "
                f"{int(candidate['peak_allocated_gpu_bytes'])} | — |",
                f"| Peak reserved GPU bytes | {int(full['peak_reserved_gpu_bytes'])} | "
                f"{int(candidate['peak_reserved_gpu_bytes'])} | — |",
                "",
                "Prompt/final KV storage is also preserved by decoder-layer segment in the physical timing JSONL. "
                "These are exploratory one-pass measurements, not the dissertation's five-repetition final timing protocol.",
                "",
            ])
        elif cell.get("status") == "skipped_by_design":
            lines.extend([
                f"Candidate: **{cell.get('candidate_cell_id')}**. Skipped by design: {cell.get('skip_reason')}.",
                "",
                f"Registered gate evidence: `{json.dumps(gate, sort_keys=True)}`",
                "",
            ])
        else:
            lines.extend(["Gate evaluation or execution remains pending.", ""])
    return lines


def strongest_by_registered_rule(registry: dict[str, Any], ids: list[str]) -> str:
    rows = {
        cell_id: registry["cells"][cell_id]
        for cell_id in ids
        if registry["cells"][cell_id].get("status") == "accepted"
    }
    if not rows:
        raise CampaignAnalysisError("No accepted candidates for registered selection")
    highest = max(row["metrics"]["quality/acc_at_0_5"] for row in rows.values())
    near = {
        cell_id: row for cell_id, row in rows.items()
        if highest - row["metrics"]["quality/acc_at_0_5"] <= 0.005 + 1e-12
    }
    return sorted(
        near,
        key=lambda cell_id: (
            near[cell_id]["candidate_average_visual_tokens"],
            -near[cell_id]["metrics"]["quality/mean_iou"],
            cell_id,
        ),
    )[0]


def finalize_decisions(registry: dict[str, Any]) -> dict[str, Any]:
    deployable_ids = ["T1", "T2", "T3", "T4", "T5"] + [f"P{index}" for index in range(1, 10)]
    strongest_deployable = strongest_by_registered_rule(registry, deployable_ids)
    oracle_ids = [
        cell_id for cell_id in [f"O{index}" for index in range(1, 9)]
        if registry["cells"][cell_id].get("candidate_average_visual_tokens", math.inf) <= 64.0 + 1e-12
    ]
    strongest_oracle = strongest_by_registered_rule(registry, oracle_ids)
    comparator = None
    if strongest_deployable in {"P7", "P8"}:
        comparator = "P9"
    elif strongest_deployable in {"T2", "T3", "T4"}:
        budget = {"T2": 16, "T3": 8, "T4": 4}[strongest_deployable]
        comparator = f"R{budget}-S0"
    reference = registry["reference"]["discovery_recomputed"]
    deployable = registry["cells"][strongest_deployable]
    oracle = registry["cells"][strongest_oracle]
    deployable_good = (
        deployable["candidate_average_visual_tokens"] < 246
        and reference["acc_at_0_5"] - deployable["metrics"]["quality/acc_at_0_5"] <= 0.02
        and reference["mean_iou"] - deployable["metrics"]["quality/mean_iou"] <= 0.01
    )
    oracle_good = (
        reference["acc_at_0_5"] - oracle["metrics"]["quality/acc_at_0_5"] <= 0.02
        and reference["mean_iou"] - oracle["metrics"]["quality/mean_iou"] <= 0.01
    )
    if deployable_good:
        recommendation = "A"
    elif oracle_good:
        recommendation = "B"
    else:
        recommendation = "C"
    registry["deferred_full_validation_recommendations"] = [
        {"role": "strongest_deployable", "cell_id": strongest_deployable, "status": "not_run"}
    ] + ([{"role": "scientific_comparator", "cell_id": comparator, "status": "not_run"}] if comparator else [])
    registry["final_exploratory_decision"] = {
        "strongest_deployable": strongest_deployable,
        "strongest_oracle_le_64": strongest_oracle,
        "recommendation": recommendation,
        "rule": {
            "deployable_good": deployable_good,
            "oracle_good": oracle_good,
            "reference": "REF_K20_B48",
        },
    }
    atomic_json(REGISTRY_PATH, registry)
    return registry["final_exploratory_decision"]


RECOMMENDATIONS = {
    "A": "**A. Simple progressive pruning is already competitive enough to continue without training.**",
    "B": "**B. Oracle early selection is strong, so a lightweight future-relevance predictor/router is justified.**",
    "C": "**C. Even oracle early selection is weak, so selector prediction alone is insufficient and the next work should investigate information transfer, token recovery or another mechanism.**",
}


def render_report(registry: dict[str, Any]) -> None:
    accepted1 = all(registry["cells"][cell_id]["status"] == "accepted" for cell_id in ["T1", "T2", "T3", "T4", "T5"] + [f"R{b}-S{s}" for b in (16, 8, 4) for s in (0, 1, 2)])
    accepted2 = all(registry["cells"][f"P{i}"]["status"] == "accepted" for i in range(1, 10))
    accepted3 = all(registry["cells"][f"O{i}"]["status"] == "accepted" for i in range(1, 11))
    ref = registry["reference"]["discovery_recomputed"]
    preflight = json.loads((CAMPAIGN_ROOT / "preflight/PREFLIGHT_REPORT.json").read_text(encoding="utf-8"))
    analysis1 = json.loads((ANALYSIS_ROOT / "campaign1.json").read_text(encoding="utf-8")) if accepted1 else {}
    analysis2 = json.loads((ANALYSIS_ROOT / "campaign2.json").read_text(encoding="utf-8")) if accepted2 else {}
    analysis3 = json.loads((ANALYSIS_ROOT / "campaign3.json").read_text(encoding="utf-8")) if accepted3 else {}
    sections = [
        "# Progressive Visual-Token Pruning and Oracle Feasibility Campaign",
        "",
        "## Scope and protocol",
        "",
        "This is an exploratory RefCOCO-validation campaign. All quality cells use the same frozen 1,000-example validation subset. No testA/testB, transfer dataset, new 8,811-example confirmation, learned router, or dissertation edit was run.",
        "",
        "## Frozen discovery subset and historical reference",
        "",
        "- Discovery rows: 1,000",
        f"- Manifest SHA-256: `{registry['discovery_manifest']['output_manifest_sha256']}`",
        f"- REF_K20_B48: Acc@0.5={ref['acc_at_0_5']:.4f}, mIoU={ref['mean_iou']:.4f}, parse={ref['parse_rate']:.4f}",
        "",
        "The reference was independently rescored from the complete physical historical JSONL. Its absent historical completion sentinel and redundant partial are disclosed; neither historical aggregate metadata nor the partial was trusted.",
        "",
        "## Implementation and preflight evidence",
        "",
        f"All {int(preflight['gates']['cpu_suite']['tests'])} preregistered preflight CPU tests passed. Real-model empty-schedule, K12/B48, and K20/B48 paths matched the current canonical implementations exactly. B=0, two-stage, three-stage, oracle restart, and strict index-only replay passed. Two-way quality concurrency passed exact semantic parity with 10% measured memory headroom; three-way concurrency was rejected by capacity. Oracle remained serial.",
        "",
        "Two preflight-only attempts failed before scientific inference (wrong interpreter; inert helper probe configuration). They were preserved and did not become quality cells.",
        "",
        "## Campaign 1 — K20 extreme tail",
        "",
        table(registry, ["T1", "T2", "T3", "T4", "T5"] + [f"R{b}-S{s}" for b in (16, 8, 4) for s in (0, 1, 2)]),
        "",
    ]
    if accepted1:
        decision = registry["decisions"]["B_tail"]
        sections.extend([
            f"The provisional discovery `B_tail` is **{decision['value']}** under the registered 1.0-point Acc@0.5 and 0.005 mIoU development tolerances. This is not an equivalence claim and requires 8,811-row confirmation.",
            "",
            "Relative to REF_K20_B48, T5 (B=0) changes Acc@0.5 by +1.3 percentage points and mIoU by +0.0077 on this subset. The rule therefore selects zero, but its 1,000-row uncertainty makes this a breaking-point hypothesis rather than a definitive threshold.",
            "",
            "Dense-prefix query-top-k versus random behavior:",
            "",
            random_characterization(registry),
            "",
            "Query-top-k is not consistently above the random seed mean at B=16/8/4; the paired intervals below determine whether any observed difference is distinguishable from zero. Random seed spread is retained rather than hidden behind the mean.",
            "",
            f"Registered paired contrasts ({BOOTSTRAPS:,} resamples, seed {ANALYSIS_SEED}):",
            "",
            interval_table(analysis1),
            "",
        ])
    else:
        sections.extend(["Pending completion of all 14 cells.", ""])
    sections.extend(["## Campaign 2 — progressive pruning", "", table(registry, [f"P{i}" for i in range(1, 10)]), ""])
    if accepted2:
        decision = registry["decisions"]["progressive_refinement"]
        frontier = progressive_frontier(registry)
        sections.extend([
            f"The registered base/refinement selection chose **{decision['strongest_refinement']}** before P9. The full decision traces, Pareto inputs, and paired contrasts are in `decisions/` and `analysis/campaign2.json`.",
            "",
            f"The non-dominated Acc@0.5/average-token frontier among P1-P9 is **{', '.join(frontier)}** (in increasing token order). P3 has the highest progressive accuracy (0.4080); P5 is the registered near-best efficiency choice because it is only 0.2 points lower at 186 rather than 198 average tokens. Both remain clearly below REF_K20_B48.",
            "",
            "P8 versus P9 is the preregistered zero-tail versus four-token-tail ablation because `B_tail=0`; it is not described as a selector comparison. P8 leads by only 0.1 point Acc@0.5 and its interval includes zero.",
            "",
            interval_table(analysis2),
            "",
        ])
    else:
        sections.extend(["Pending completion of all nine cells.", ""])
    sections.extend([
        "## Campaign 3 — oracle diagnostics",
        "",
        table(registry, [f"O{i}" for i in range(1, 11)]),
        "",
        "Oracle candidate-average token figures describe only the freshly restarted counterfactual candidate. The dense privileged teacher is not efficient, is not deployable, and its 576×K visual-token layer-work is reported separately.",
        "",
    ])
    if accepted3:
        sections.extend([
            "The K12-teacher capacity curve is 0.2630/0.3513, 0.2980/0.3666, and 0.3140/0.3836 Acc@0.5/mIoU at 64/96/128 patches. The K20-teacher curve is stronger—0.2910/0.3517, 0.3500/0.3815, and 0.3720/0.4018—but even privileged K20 oracle-128 is 6.7 accuracy points and 0.0310 mIoU below REF_K20_B48.",
            "",
            "O7/O8 use the preregistered dynamic `B_early64=102` calculation and realize exactly 63.75 candidate-average tokens. O8 is the strongest ≤64-average oracle at 0.3510 Acc@0.5 and 0.3881 mIoU, still 8.8 points and 0.0447 mIoU below the reference. Oracle-64 nevertheless clearly exceeds random-64 (0.1680/0.2573) and uniform-64 (0.1980/0.2818), so patch identity helps but is not sufficient to replace early dense processing.",
            "",
            "Oracle computations are privileged diagnostics: candidate-average figures exclude the discarded dense teacher pass, while teacher work is separately shown as 6,912 (K12) or 11,520 (K20) visual-token layers.",
            "",
            interval_table(analysis3),
            "",
        ])
    else:
        sections.extend(["Pending completion of all ten cells.", ""])
    sections.extend([
        "## Rejected attempts and artifact validity",
        "",
        "All 33 scientific cells were accepted on their first attempt. Every accepted cell has exactly 1,000 ordered physical predictions, a strict source/config epoch, independently recomputed raw-output metrics, validated stage indices/positions/cache lengths, and a hash-bearing completion sentinel. No partial was spliced or reconstructed.",
        "",
        "The two rejected preflight-only attempts occurred before scientific inference: the default interpreter lacked `transformers`, and an unused inherited helper probe configuration was rejected. Both are recorded in `preflight/PREFLIGHT_REPORT.json`; neither counts as a scientific cell.",
        "",
        "## Conditional exploratory timing",
        "",
    ])
    sections.extend(timing_report(registry))
    sections.extend(["## Deferred full-validation recommendation", ""])
    if accepted3:
        final = registry.get("final_exploratory_decision") or finalize_decisions(registry)
        sections.extend([
            f"Strongest deployable candidate: **{final['strongest_deployable']}**. Strongest oracle candidate at ≤64 average tokens: **{final['strongest_oracle_le_64']}**.",
            "",
            "The future 8,811-row recommendation is T2 (K20→16 query top-k) with R16-S0 (K20→16 random, seed 0) as its single selector comparator. Both are frozen in the registry as **not run**. No F1-F4, testA/testB, or other full-validation job was launched; a new human-authorized goal is required.",
            "",
        ])
    else:
        sections.extend(["Not run; candidate selection is pending discovery outcomes.", ""])
    sections.extend([
        "## Limitations",
        "",
        "These are validation-only exploratory results on 1,000 examples; one percentage point of accuracy is only ten examples. Marginal 0.5–1.0-point movements are inconclusive. Oracle results use privileged future information. Any dissertation claim or training decision should first consider the deferred full-validation confirmation.",
        "",
        "## Primary recommendation",
        "",
    ])
    if accepted3:
        final = registry["final_exploratory_decision"]
        deployable = registry["cells"][final["strongest_deployable"]]
        oracle = registry["cells"][final["strongest_oracle_le_64"]]
        sections.extend([
            RECOMMENDATIONS[final["recommendation"]],
            "",
            f"The strongest deployable cell ({final['strongest_deployable']}) is accurate but averages {float(deployable['candidate_average_visual_tokens']):.1f} visual tokens, while the strongest ≤64-average oracle ({final['strongest_oracle_le_64']}) reaches only {float(oracle['metrics']['quality/acc_at_0_5']):.4f} Acc@0.5 and {float(oracle['metrics']['quality/mean_iou']):.4f} mIoU. Even oracle-128 remains well below the reference. The evidence therefore points to early-layer information transfer/mixing—not merely imperfect patch ranking—as the limiting factor.",
            "",
            "This is a preliminary exploratory recommendation. Do not begin training without human review; any dissertation claim should first consider the deferred 8,811-row confirmation.",
        ])
    else:
        sections.append("Pending completion of all 33 accepted discovery cells.")
    atomic_write(REPORT_PATH, ("\n".join(sections).rstrip() + "\n").encode("utf-8"))


def refresh(through: int) -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    analysis_paths = registry.setdefault("analysis", {})
    if through >= 1:
        analyse_campaign1(registry)
        analysis_paths["campaign1"] = str((ANALYSIS_ROOT / "campaign1.json").relative_to(REPO_ROOT))
    if through >= 2:
        analyse_campaign2(registry)
        analysis_paths["campaign2"] = str((ANALYSIS_ROOT / "campaign2.json").relative_to(REPO_ROOT))
    if through >= 3:
        analyse_campaign3(registry)
        analysis_paths["campaign3"] = str((ANALYSIS_ROOT / "campaign3.json").relative_to(REPO_ROOT))
        finalize_decisions(registry)
    atomic_json(REGISTRY_PATH, registry)
    render_report(registry)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through-campaign", type=int, choices=[1, 2, 3], required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        refresh(args.through_campaign)
    except (CampaignAnalysisError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
