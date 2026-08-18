#!/usr/bin/env python
"""Join audited artifacts, apply the frozen decision rule, and build the campaign report."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm_pruning.artifacts import sha256_file  # noqa: E402
from vlm_pruning.final_statistics import pareto_frontier, select_internal_candidate  # noqa: E402


CAMPAIGN = REPO_ROOT / "experiments/final_val_efficiency_baselines_20260818"
ANALYSIS = CAMPAIGN / "analysis"
FIGURES = ANALYSIS / "figures"
REPORT = CAMPAIGN / "RESEARCH_REPORT.md"
QUALITY_PATH = ANALYSIS / "QUALITY_METHOD_METRICS.json"
TIMING_PATH = ANALYSIS / "TIMING_NATURAL_SUMMARY.json"
TIMING_CONTRASTS_PATH = ANALYSIS / "TIMING_NATURAL_CONTRASTS.json"
ACCOUNTING_PATH = ANALYSIS / "FLOP_KV_ACCOUNTING.json"
DECISION_PATH = ANALYSIS / "FROZEN_FINAL_CANDIDATE.json"
PARETO_PATH = ANALYSIS / "PARETO_FRONTIERS.json"

DISPLAY = {
    "Full": "Full",
    "K20": "K20/B48",
    "K12": "K12/B48",
    "M448_OPT": "M448",
    "Z_M448_0": "Z-M448-0",
    "FEATHER_LLAVA_MAIN": "FEATHER (LLaVA adapter)",
    "FASTV_MATCHED_199": "FastV (official fork)",
    "FASTV_STACK_FULL": "FastV-stack Full",
    "NUWA_STACK_FULL": "Nüwa-stack Full",
    "NUWA_NATIVE_192": "Nüwa NT192/DIST280",
}


class FinalizationError(RuntimeError):
    """Raised when audited inputs cannot support final report generation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
    )


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise FinalizationError(f"Refusing to write empty CSV: {path}")
    fields = list(rows[0])
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FinalizationError(f"Missing prerequisite: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def mean(timing: dict[str, Any], field: str) -> float:
    value = timing[field]
    if int(value.get("available_rows", 0)) <= 0:
        raise FinalizationError(f"Missing timing field {field} for {timing['method']}")
    return float(value["mean"])


def method_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quality_rows = read_json(QUALITY_PATH)
    timing_rows = read_json(TIMING_PATH)
    accounting = read_json(ACCOUNTING_PATH)["methods"]
    quality = {row["method"]: row for row in quality_rows}
    timing = {(row["stack"], row["method"]): row for row in timing_rows}
    accounting_by = {row["method"]: row for row in accounting}
    records = []
    for method, quality_row in quality.items():
        stack = str(quality_row["stack"])
        timing_stack = {
            "canonical": "canonical",
            "fastv_native_fork": "fastv",
            "nuwa_native": "nuwa",
        }[stack]
        timing_row = timing.get((timing_stack, method))
        accounting_row = accounting_by.get(method)
        record = {
            "method": method,
            "display": DISPLAY[method],
            "stack": stack,
            "acc_at_0_5": float(quality_row["acc_at_0_5"]),
            "mean_iou": float(quality_row["mean_iou"]),
            "median_iou": float(quality_row["median_iou"]),
            "parse_rate": float(quality_row["parse_rate"]),
            "valid_output_rate": float(quality_row["valid_output_rate"]),
            "visual_token_layer_work": float(quality_row["mean_token_layer_work"]),
            "average_visual_tokens": float(quality_row["mean_average_visual_tokens"]),
            "final_visual_tokens": float(quality_row["mean_final_visual_tokens"]),
            "natural_end_to_end_s": mean(timing_row, "end_to_end_latency_s") if timing_row else None,
            "prefill_including_overhead_s": (
                mean(timing_row, "prefill_including_pruning_latency_s") if timing_row else None
            ),
            "time_to_first_token_s": mean(timing_row, "time_to_first_token_s") if timing_row else None,
            "prompt_kv_bytes": (
                float(accounting_row["mean_analytical_prompt_kv_bytes"])
                if accounting_row else None
            ),
            "final_kv_bytes": (
                float(accounting_row["mean_analytical_final_kv_bytes"])
                if accounting_row else None
            ),
            "modeled_total_prefill_flops": (
                float(accounting_row["mean_modeled_total_multimodal_prefill_flops"])
                if accounting_row else None
            ),
            "peak_allocated_gpu_bytes": (
                float(accounting_row["mean_peak_allocated_gpu_bytes"])
                if accounting_row else None
            ),
        }
        records.append(record)
    return records, {"quality": quality, "timing": timing, "accounting": accounting_by}


def apply_decision(source: dict[str, Any]) -> dict[str, Any]:
    quality = {
        method: {
            "parse_rate": float(source["quality"][method]["parse_rate"]),
            "acc_at_0_5": float(source["quality"][method]["acc_at_0_5"]),
            "mean_iou": float(source["quality"][method]["mean_iou"]),
        }
        for method in ("Full", "K12", "M448_OPT", "Z_M448_0")
    }
    timing_summary = source["timing"]
    contrasts = read_json(TIMING_CONTRASTS_PATH)
    upper = {
        row["left"]: float(row["upper_95"])
        for row in contrasts
        if row["stack"] == "canonical"
        and row["right"] == "Full"
        and row["metric"] == "end_to_end_latency_s"
        and row["left"] in {"K12", "M448_OPT", "Z_M448_0"}
    }
    timing = {
        "Full": {
            "end_to_end_latency_s": mean(timing_summary[("canonical", "Full")], "end_to_end_latency_s")
        }
    }
    for method in ("K12", "M448_OPT", "Z_M448_0"):
        timing[method] = {
            "end_to_end_latency_s": mean(
                timing_summary[("canonical", method)], "end_to_end_latency_s"
            ),
            "delta_vs_full_upper_95_s": upper[method],
        }
    decision = select_internal_candidate(
        quality,
        timing,
        {"K12": 7872, "M448_OPT": 6848, "Z_M448_0": 6272},
    )
    protocol = read_json(CAMPAIGN / "FROZEN_PROTOCOL.json")
    inventory = read_json(ANALYSIS / "TIMING_NATURAL_INVENTORY.json")
    decision.update(
        {
            "created_at_utc": utc_now(),
            "validation_only": True,
            "testA_testB_authorized": False,
            "checkpoint": protocol["checkpoint"],
            "quality_manifest": protocol["manifests"]["quality"],
            "timing_manifest": protocol["manifests"]["timing"],
            "prompt_sha256": hashlib.sha256(
                protocol["inference"]["prompt"].encode()
            ).hexdigest(),
            "quality_evidence_sha256": sha256_file(QUALITY_PATH),
            "timing_evidence_sha256": sha256_file(TIMING_PATH),
            "timing_contrasts_sha256": sha256_file(TIMING_CONTRASTS_PATH),
            "timing_run_config_sha256": {
                row["run_dir"]: row["config_sha256"]
                for row in inventory["runs"]
                if row["stack"] == "canonical"
            },
        }
    )
    atomic_json(DECISION_PATH, decision)
    return decision


def build_pareto(records: list[dict[str, Any]]) -> dict[str, Any]:
    canonical_quality = [row for row in records if row["stack"] == "canonical"]
    canonical_timed = [row for row in canonical_quality if row["natural_end_to_end_s"] is not None]
    axes = {}
    for quality_field in ("acc_at_0_5", "mean_iou"):
        for cost_field, eligible in (
            ("visual_token_layer_work", canonical_quality),
            ("modeled_total_prefill_flops", [row for row in canonical_quality if row["modeled_total_prefill_flops"] is not None]),
            ("natural_end_to_end_s", canonical_timed),
            ("prefill_including_overhead_s", canonical_timed),
            ("time_to_first_token_s", canonical_timed),
            ("prompt_kv_bytes", [row for row in canonical_quality if row["prompt_kv_bytes"] is not None]),
            ("final_kv_bytes", [row for row in canonical_quality if row["final_kv_bytes"] is not None]),
        ):
            name = f"{quality_field}_vs_{cost_field}"
            axes[name] = {
                "quality_axis": quality_field,
                "cost_axis": cost_field,
                "eligible_methods": [row["method"] for row in eligible],
                "frontier": pareto_frontier(
                    eligible, quality_field=quality_field, cost_field=cost_field
                ),
            }
    native = {}
    by = {row["method"]: row for row in records}
    for stack, full, pruned in (
        ("fastv", "FASTV_STACK_FULL", "FASTV_MATCHED_199"),
        ("nuwa", "NUWA_STACK_FULL", "NUWA_NATIVE_192"),
    ):
        native[stack] = {
            "full": full,
            "pruned": pruned,
            "natural_latency_ratio_pruned_over_full": (
                by[pruned]["natural_end_to_end_s"] / (
                    next(
                        timing["end_to_end_latency_s"]["mean"]
                        for timing in read_json(TIMING_PATH)
                        if timing["method"] == full
                    )
                )
            ),
            "quality_retention_available": stack == "nuwa",
            "acc_retention": (
                by[pruned]["acc_at_0_5"] / by[full]["acc_at_0_5"]
                if stack == "nuwa" else None
            ),
            "miou_retention": (
                by[pruned]["mean_iou"] / by[full]["mean_iou"]
                if stack == "nuwa" else None
            ),
        }
    output = {
        "status": "complete",
        "dominance_rule": "higher quality and lower cost, at least one strict",
        "cross_stack_raw_latency_ranking_prohibited": True,
        "axes": axes,
        "source_native_normalized": native,
    }
    atomic_json(PARETO_PATH, output)
    return output


def plots(records: list[dict[str, Any]], pareto: dict[str, Any]) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    canonical = [row for row in records if row["stack"] == "canonical"]
    fastv = [row for row in records if row["method"] == "FASTV_MATCHED_199"]
    colors = {
        "Full": "#333333", "K20": "#777777", "K12": "#0072B2",
        "M448_OPT": "#009E73", "Z_M448_0": "#D55E00",
        "FEATHER_LLAVA_MAIN": "#CC79A7", "FASTV_MATCHED_199": "#E69F00",
    }
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for axis, metric, label in zip(axes, ("acc_at_0_5", "mean_iou"), ("Acc@0.5", "Mean IoU")):
        for row in canonical + fastv:
            hollow = row["stack"] != "canonical"
            axis.scatter(
                row["visual_token_layer_work"], row[metric], s=62,
                facecolors="none" if hollow else colors[row["method"]],
                edgecolors=colors[row["method"]], linewidths=1.5, zorder=3,
            )
            axis.annotate(
                DISPLAY[row["method"]],
                (row["visual_token_layer_work"], row[metric]), xytext=(4, 5),
                textcoords="offset points", fontsize=7.5,
            )
        frontier_methods = pareto["axes"][f"{metric}_vs_visual_token_layer_work"]["frontier"]
        frontier = sorted(
            [row for row in canonical if row["method"] in frontier_methods],
            key=lambda row: row["visual_token_layer_work"],
        )
        axis.plot(
            [row["visual_token_layer_work"] for row in frontier],
            [row[metric] for row in frontier], color="#555555", linestyle="--", linewidth=1,
        )
        axis.set_xlabel("Visual token-layer work")
        axis.set_ylabel(label)
        axis.grid(alpha=0.22)
    path = FIGURES / "quality_vs_token_work.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    timed = [row for row in canonical if row["natural_end_to_end_s"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    for axis, metric, label in zip(axes, ("acc_at_0_5", "mean_iou"), ("Acc@0.5", "Mean IoU")):
        for row in timed:
            axis.scatter(row["natural_end_to_end_s"], row[metric], s=65, color=colors[row["method"]])
            axis.annotate(DISPLAY[row["method"]], (row["natural_end_to_end_s"], row[metric]), xytext=(4, 5), textcoords="offset points", fontsize=8)
        axis.set_xlabel("Natural end-to-end latency (s/example)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.22)
    path = FIGURES / "quality_vs_natural_latency.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    timing_by = {
        row["method"]: row for row in read_json(TIMING_PATH) if row["stack"] == "canonical"
    }
    methods = ["Full", "K12", "M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN"]
    components = (
        ("vision_encoder_latency_s", "Vision"),
        ("projector_latency_s", "Projector"),
        ("query_scoring_latency_s", "Scoring"),
        ("selector_latency_s", "Selection"),
        ("gather_latency_s", "Gather"),
        ("merge_latency_s", "Merge"),
        ("prefill_excluding_pruning_latency_s", "LLM prefill"),
        ("decode_latency_s", "Decode"),
    )
    fig, axis = plt.subplots(figsize=(9.2, 4.4), constrained_layout=True)
    bottom = np.zeros(len(methods))
    palette = plt.get_cmap("tab20").colors
    for index, (field, label) in enumerate(components):
        values = np.asarray([mean(timing_by[method], field) for method in methods])
        axis.bar([DISPLAY[method] for method in methods], values, bottom=bottom, label=label, color=palette[index])
        bottom += values
    residual = np.asarray([mean(timing_by[method], "end_to_end_latency_s") for method in methods]) - bottom
    axis.bar([DISPLAY[method] for method in methods], np.maximum(residual, 0), bottom=bottom, label="Other/dispatch", color="#BBBBBB")
    axis.set_ylabel("Seconds per example")
    axis.tick_params(axis="x", rotation=18)
    axis.legend(ncol=3, fontsize=7.5)
    axis.grid(axis="y", alpha=0.22)
    path = FIGURES / "canonical_timing_decomposition.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def build_report(
    records: list[dict[str, Any]],
    source: dict[str, Any],
    decision: dict[str, Any],
    pareto: dict[str, Any],
    figure_paths: list[Path],
) -> str:
    del pareto
    by = {row["method"]: row for row in records}
    quality_order = [
        "Full", "K20", "K12", "M448_OPT", "Z_M448_0",
        "FASTV_MATCHED_199", "FEATHER_LLAVA_MAIN",
    ]
    timing_methods = ["Full", "K12", "M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN"]
    quality_table = markdown_table(
        ["Method", "Stack", "Acc@0.5", "mIoU", "Avg visual", "Token work"],
        [
            [
                DISPLAY[method], by[method]["stack"], f"{by[method]['acc_at_0_5']:.4f}",
                f"{by[method]['mean_iou']:.4f}", f"{by[method]['average_visual_tokens']:.1f}",
                f"{by[method]['visual_token_layer_work']:.0f}",
            ]
            for method in quality_order
        ],
    )
    timing_table = markdown_table(
        [
            "Method", "E2E s", "Prefill s", "TTFT s", "Select s", "Gather/merge s",
            "Decode s", "Output tok.", "Example/s", "Peak alloc GiB", "Prompt KV MiB",
        ],
        [
            [
                DISPLAY[method], f"{by[method]['natural_end_to_end_s']:.4f}",
                f"{by[method]['prefill_including_overhead_s']:.4f}",
                f"{by[method]['time_to_first_token_s']:.4f}",
                f"{mean(source['timing'][('canonical', method)], 'selector_latency_s'):.4f}",
                f"{mean(source['timing'][('canonical', method)], 'gather_latency_s') + mean(source['timing'][('canonical', method)], 'merge_latency_s'):.4f}",
                f"{mean(source['timing'][('canonical', method)], 'decode_latency_s'):.4f}",
                f"{mean(source['timing'][('canonical', method)], 'generated_token_count'):.2f}",
                f"{mean(source['timing'][('canonical', method)], 'throughput_examples_per_s'):.3f}",
                f"{by[method]['peak_allocated_gpu_bytes'] / 2**30:.3f}",
                f"{by[method]['prompt_kv_bytes'] / 2**20:.2f}",
            ]
            for method in timing_methods
        ],
    )
    quality_effects = read_json(ANALYSIS / "QUALITY_PAIRED_CONTRASTS.json")
    quality_effect_table = markdown_table(
        ["Contrast", "Metric", "Difference", "Paired 95% interval"],
        [
            [
                str(row["contrast"]), str(row["metric"]), f"{float(row['point_delta']):+.4f}",
                f"[{float(row['lower_95']):+.4f}, {float(row['upper_95']):+.4f}]",
            ]
            for row in quality_effects
        ],
    )
    timing_effects = [
        row for row in read_json(TIMING_CONTRASTS_PATH)
        if row["metric"] == "end_to_end_latency_s"
    ]
    timing_effect_table = markdown_table(
        ["Stack", "Contrast", "Difference s", "Hierarchical paired 95% interval"],
        [
            [
                str(row["stack"]), str(row["contrast"]), f"{float(row['point_delta']):+.4f}",
                f"[{float(row['lower_95']):+.4f}, {float(row['upper_95']):+.4f}]",
            ]
            for row in timing_effects
        ],
    )
    accounting_order = ["Full", "K12", "M448_OPT", "Z_M448_0", "FASTV_MATCHED_199", "FEATHER_LLAVA_MAIN", "NUWA_NATIVE_192"]
    accounting_table = markdown_table(
        ["Method", "LLM prefill TFLOPs", "Total modeled TFLOPs", "Score GFLOPs", "Merge GFLOPs"],
        [
            [
                DISPLAY[method], f"{source['accounting'][method]['mean_llm_prefill_flops'] / 1e12:.3f}",
                f"{source['accounting'][method]['mean_modeled_total_multimodal_prefill_flops'] / 1e12:.3f}",
                f"{source['accounting'][method]['mean_score_flops'] / 1e9:.3f}",
                f"{source['accounting'][method]['mean_merge_flops'] / 1e9:.3f}",
            ]
            for method in accounting_order
        ],
    )
    native_table = markdown_table(
        ["Native stack", "Control Acc/mIoU", "Pruned Acc/mIoU", "Control E2E s", "Pruned E2E s"],
        [
            [
                "Nüwa", f"{by['NUWA_STACK_FULL']['acc_at_0_5']:.4f}/{by['NUWA_STACK_FULL']['mean_iou']:.4f}",
                f"{by['NUWA_NATIVE_192']['acc_at_0_5']:.4f}/{by['NUWA_NATIVE_192']['mean_iou']:.4f}",
                f"{next(mean(row, 'end_to_end_latency_s') for row in source['timing'].values() if row['method'] == 'NUWA_STACK_FULL'):.4f}",
                f"{by['NUWA_NATIVE_192']['natural_end_to_end_s']:.4f}",
            ],
            [
                "FastV", "quality control unavailable after parity failure",
                f"{by['FASTV_MATCHED_199']['acc_at_0_5']:.4f}/{by['FASTV_MATCHED_199']['mean_iou']:.4f}",
                f"{next(mean(row, 'end_to_end_latency_s') for row in source['timing'].values() if row['method'] == 'FASTV_STACK_FULL'):.4f}",
                f"{by['FASTV_MATCHED_199']['natural_end_to_end_s']:.4f}",
            ],
        ],
    )
    trigger = read_json(ANALYSIS / "FIXED32_TRIGGER.json")
    required = [stack for stack, value in trigger["stacks"].items() if value["required"]]
    fixed_table: list[str] = []
    if required:
        fixed_rows = read_json(ANALYSIS / "TIMING_FIXED32_SUMMARY.json")
        fixed_table = markdown_table(
            ["Stack", "Method", "E2E s", "Prefill s", "Decode s", "Output tokens"],
            [
                [
                    str(row["stack"]), DISPLAY[str(row["method"])],
                    f"{mean(row, 'end_to_end_latency_s'):.4f}",
                    f"{mean(row, 'prefill_including_pruning_latency_s'):.4f}",
                    f"{mean(row, 'decode_latency_s'):.4f}",
                    f"{mean(row, 'generated_token_count'):.1f}",
                ]
                for row in fixed_rows
            ],
        )
    selected = decision["selected_method"] or "none"
    decision_table = markdown_table(
        ["Method", "Quality gate", "Acceleration gate", "Token work", "Outcome"],
        [
            [
                DISPLAY[method],
                "pass" if decision["method_decisions"][method]["quality_gate_pass"] else "fail",
                "pass" if decision["method_decisions"][method]["acceleration_gate_pass"] else "fail",
                str(decision["method_decisions"][method]["visual_token_layer_work"]),
                "selected" if decision["selected_method"] == method else "--",
            ]
            for method in ("K12", "M448_OPT", "Z_M448_0")
        ],
    )
    lines = [
        "# Final RefCOCO-validation efficiency and external-baseline campaign",
        "",
        f"Status: **{decision['status']}**. Registered internal candidate: **{selected}**. This is a validation-only decision; RefCOCO testA/testB and transfer evaluation were not run or authorized.",
        "",
        "## Scope and provenance",
        "",
        "The campaign uses the frozen 8,811-example RefCOCO validation manifest for deterministic quality and the frozen ordered 1,000-example validation subset for five-repetition timing. Canonical methods use LLaVA-1.5-7B BF16. Official FastV runs in its pinned Hugging Face fork; Nüwa runs with the native LLaVA checkpoint/stack; FEATHER is explicitly a source-faithful LLaVA adapter. Raw cross-stack latency ranking is prohibited.",
        "",
        "## Full-validation quality and visual work",
        "",
        *quality_table,
        "",
        "FastV is an absolute, fork-labelled result because disabled-path output parity with canonical Full failed. Nüwa also failed exact canonical preprocessing/output parity and is therefore evaluated only against its native Full control below. FEATHER passed exact upstream selection fixtures at both pruning boundaries.",
        "",
        "### Paired full-validation effects",
        "",
        *quality_effect_table,
        "",
        "Intervals containing zero are inconclusive and are not interpreted as evidence of equivalence. FastV is excluded from canonical pairing; Nüwa is paired only with Nüwa-stack Full.",
        "",
        "## Canonical repeated timing and memory",
        "",
        *timing_table,
        "",
        "Natural-output end-to-end latency is primary. Prefill includes selection, merging, gathering and segmented-cache overhead; the component figure also exposes the LLM-only and dispatch residual. Lower token work is not relabelled as measured speed.",
        "",
        "### Primary repeated-timing effects",
        "",
        *timing_effect_table,
        "",
        "## FLOPs and pruning arithmetic",
        "",
        *accounting_table,
        "",
        "The modeled total counts CLIP ViT-L/14@336, the projector, decoder attention/MLP and explicit pruning arithmetic. Ranking comparisons and gather memory movement are element counts rather than FLOPs. FastV's official path disables KV caching, so its physical KV is zero/cache-disabled while analytical KV is reported as a workload counterfactual.",
        "",
        "## Source-native controls",
        "",
        *native_table,
        "",
        "Only within-stack deltas and normalized speed/retention are scientific comparisons for the source-native rows. Software, checkpoint and preprocessing differences prevent a raw cross-stack winner claim.",
        "",
        "## Registered decision",
        "",
        f"The frozen rule first required parse rate >=99.5%, Acc@0.5 loss <=3.0 percentage points and mIoU loss <=0.015 relative to Full. It then required a lower five-repetition natural E2E mean and a hierarchical paired 95% interval whose upper endpoint is below zero. Among methods passing both, minimum visual token-layer work wins. The resulting status is `{decision['status']}` and the selected method is `{selected}`.",
        "",
        *decision_table,
        "",
        "K12 remains the latency-first operating point if its measured latency is lowest; M448 transfers early information through mass-preserving merging; Z-M448-0 minimizes internal work and late KV by removing all visual tokens after K20. The decision file preserves every gate outcome rather than forcing a winner.",
        "",
        "## Interpretation and relation to prior evidence",
        "",
        "The measured selector/merge/cache/dispatch components explain why token count and FLOPs need not predict E2E latency. The earlier learned router routed almost all rows to M448, the privileged early oracle showed that token identity alone was insufficient at very low early budgets, E256 pre-LLM pooling failed the quality screen, and the trained SAQR proof of concept was slower while its no-query diagnostic was stronger. Those immutable negative results support the training-free staged method; none was rerun here.",
        "",
        "FastV illustrates early decoder pruning, while VisionZip motivates one-shot pre-LLM compression that can avoid repeated decoder interruption. FEATHER adds a RoPE-free score pass and adaptive two-stage union; its adaptation is not the upstream Prism checkpoint. Nüwa combines vision-side aggregation and a later text-guided drop in a separate native stack.",
        "",
        "## Limitations",
        "",
        "This is one checkpoint, one GPU, batch size one, RefCOCO validation, deterministic generation and no seed replication because inference is deterministic. Confidence intervals quantify paired example/repetition uncertainty, not model-training uncertainty. Timing is implementation- and hardware-specific. Modeled FLOPs omit non-arithmetic kernel and memory costs but list those operations separately. No held-out, transfer, training, learned-router or unregistered-budget experiment was performed.",
        "",
        "## Figures and evidence",
        "",
        *(f"- `{path.relative_to(CAMPAIGN)}`" for path in figure_paths),
        "- `analysis/QUALITY_RUN_INVENTORY.json` and `analysis/QUALITY_PAIRED_CONTRASTS.json`",
        "- `analysis/TIMING_NATURAL_INVENTORY.json` and `analysis/TIMING_NATURAL_CONTRASTS.json`",
        "- `analysis/FLOP_KV_ACCOUNTING.json`, `analysis/PARETO_FRONTIERS.json`, and `analysis/FROZEN_FINAL_CANDIDATE.json`",
        "",
    ]
    if required:
        lines.extend(
            [
                "## Fixed-32 diagnostic",
                "",
                f"The preregistered natural-output trigger required the three-repetition fixed-32 diagnostic for: {', '.join(required)}. These results diagnose output-length dependence and do not replace the natural-output primary endpoint.",
                "",
                *fixed_table,
                "",
            ]
        )
    return "\n".join(lines)


def update_state(decision: dict[str, Any]) -> None:
    registry_path = CAMPAIGN / "EXPERIMENT_REGISTRY.json"
    controller_path = CAMPAIGN / "CONTROLLER_STATE.json"
    registry = read_json(registry_path)
    registry["phase_status"].update(
        {
            "implementation": "pass",
            "smokes": "pass",
            "quality": "pass",
            "timing": "pass",
            "analysis": "pass",
        }
    )
    registry["status"] = "analysis_complete_dissertation_pending"
    for cell in registry["cells"].values():
        cell["status"] = "complete"
    atomic_json(registry_path, registry)
    controller = read_json(controller_path)
    controller.update(
        {
            "phase": "dissertation",
            "status": "active",
            "active_workers": [],
            "next_action": "integrate_validated_generated_artifacts_and_rebuild_pdf",
            "selected_method": decision["selected_method"],
            "updated_at_utc": utc_now(),
        }
    )
    atomic_json(controller_path, controller)


def evidence_ledger(figure_paths: list[Path]) -> None:
    claims = {
        "full_validation_quality": [QUALITY_PATH, ANALYSIS / "QUALITY_RUN_INVENTORY.json"],
        "paired_quality_uncertainty": [ANALYSIS / "QUALITY_PAIRED_CONTRASTS.json"],
        "natural_timing": [TIMING_PATH, TIMING_CONTRASTS_PATH, ANALYSIS / "TIMING_NATURAL_INVENTORY.json"],
        "flop_kv_memory": [ACCOUNTING_PATH],
        "candidate_decision": [DECISION_PATH],
        "pareto_frontiers": [PARETO_PATH],
        "parity_and_stack_separation": [ANALYSIS / "PARITY_AUDIT.json"],
        "figures": figure_paths,
    }
    trigger = read_json(ANALYSIS / "FIXED32_TRIGGER.json")
    claims["fixed32_trigger"] = [ANALYSIS / "FIXED32_TRIGGER.json"]
    if any(value["required"] for value in trigger["stacks"].values()):
        claims["fixed32_diagnostic"] = [
            ANALYSIS / "TIMING_FIXED32_SUMMARY.json",
            ANALYSIS / "TIMING_FIXED32_CONTRASTS.json",
            ANALYSIS / "TIMING_FIXED32_INVENTORY.json",
        ]
    atomic_json(
        ANALYSIS / "CLAIM_EVIDENCE_LEDGER.json",
        {
            "status": "complete",
            "claims": {
                claim: [
                    {
                        "path": str(path.relative_to(REPO_ROOT)),
                        "sha256": sha256_file(path),
                    }
                    for path in paths
                ]
                for claim, paths in claims.items()
            },
        },
    )


def main() -> int:
    trigger = read_json(ANALYSIS / "FIXED32_TRIGGER.json")
    if any(value["required"] for value in trigger["stacks"].values()):
        if not (ANALYSIS / "TIMING_FIXED32_SUMMARY.json").is_file():
            raise FinalizationError("Triggered fixed32 timing has not been analysed")
    records, source = method_records()
    decision = apply_decision(source)
    pareto = build_pareto(records)
    figure_paths = plots(records, pareto)
    atomic_json(ANALYSIS / "JOINED_METHOD_RECORDS.json", records)
    atomic_csv(ANALYSIS / "JOINED_METHOD_RECORDS.csv", records)
    atomic_write(REPORT, (build_report(records, source, decision, pareto, figure_paths) + "\n").encode())
    evidence_ledger(figure_paths)
    update_state(decision)
    print(json.dumps({"status": "analysis_complete", "decision": decision["status"], "selected": decision["selected_method"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
