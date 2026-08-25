#!/usr/bin/env python3
"""Build audited IVC-only tables and figures for later dissertation integration."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO_ROOT / "experiments/ivc_prune_refcoco_val_20260825"
ANALYSIS = CAMPAIGN / "analysis"


class AssetError(RuntimeError):
    pass


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_analysis(name: str, status: str) -> dict[str, Any]:
    path = ANALYSIS / name
    if not path.is_file():
        raise AssetError(f"Missing accepted analysis: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != status:
        raise AssetError(f"Analysis is not complete: {path}")
    return value


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}"


def quality_table(quality: dict[str, Any]) -> str:
    rows = []
    labels = {"IVC_STACK_FULL": "IVC-stack Full", "IVC_PRUNE": "IVC-Prune"}
    for method in ("IVC_STACK_FULL", "IVC_PRUNE"):
        value = quality["summaries"][method]
        rows.append(
            f"{labels[method]} & {pct(value['acc_at_0_5'])} & {float(value['mean_iou']):.4f} "
            f"& {float(value['mean_average_visual_tokens']):.1f} & "
            f"{float(value['mean_visual_token_layer_work']):,.0f} \\\\"
        )
    return "\n".join(
        [
            "% Generated only from audited IVC campaign analysis.",
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{Complete RefCOCO-validation results within the pinned IVC software and checkpoint stack.}",
            "\\label{tab:ivc-quality-work}",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Method & Acc@0.5 (\\%) & mIoU & Mean visual tokens & Token-layer work\\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def timing_table(timing: dict[str, Any]) -> str:
    rows = []
    labels = {"IVC_STACK_FULL": "IVC-stack Full", "IVC_PRUNE": "IVC-Prune"}
    for method in ("IVC_STACK_FULL", "IVC_PRUNE"):
        value = timing["summaries"][method]
        e2e = value["end_to_end_latency_s"]
        prefill = value["llm_prefill_with_overhead_latency_s"]["mean"]
        decode = value["decode_latency_s"]["mean"]
        selector = value["selector_latency_s"]["mean"]
        memory = value["peak_allocated_gpu_bytes"]["mean"] / 1024**3
        rows.append(
            f"{labels[method]} & {e2e['mean']:.4f}$\\pm${e2e['sd_across_repetition_means']:.4f} "
            f"& {prefill:.4f} & {decode:.4f} & {selector:.4f} & {memory:.2f} \\\\"
        )
    return "\n".join(
        [
            "% Generated only from audited IVC campaign analysis.",
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{Paired natural-generation timing within the pinned IVC stack. End-to-end values are means across five complete repetition means.}",
            "\\label{tab:ivc-timing}",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Method & End-to-end (s) & LLM prefill (s) & Decode (s) & Selector (s) & Peak allocated (GiB)\\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def figures(quality: dict[str, Any], timing: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["IVC-stack Full", "IVC-Prune"]
    methods = ["IVC_STACK_FULL", "IVC_PRUNE"]
    colors = ["#666666", "#0072B2"]
    quality_values = [quality["summaries"][method] for method in methods]
    fig, axis = plt.subplots(figsize=(5.7, 3.8))
    for label, color, value in zip(labels, colors, quality_values, strict=True):
        axis.scatter(
            float(value["mean_average_visual_tokens"]),
            100 * float(value["acc_at_0_5"]),
            color=color,
            s=65,
            label=label,
            zorder=3,
        )
    axis.set_xlabel("Mean visual tokens per decoder layer")
    axis.set_ylabel("RefCOCO validation Acc@0.5 (%)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "ivc_quality_work.pdf", bbox_inches="tight")
    plt.close(fig)

    full = timing["summaries"]["IVC_STACK_FULL"]["end_to_end_latency_s"]
    ivc = timing["summaries"]["IVC_PRUNE"]["end_to_end_latency_s"]
    fig, axis = plt.subplots(figsize=(5.7, 3.8))
    axis.bar(
        labels,
        [full["mean"], ivc["mean"]],
        yerr=[full["sd_across_repetition_means"], ivc["sd_across_repetition_means"]],
        color=colors,
        capsize=4,
    )
    axis.set_ylabel("End-to-end latency per request (s)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "ivc_timing.pdf", bbox_inches="tight")
    plt.close(fig)

    histogram = quality["summaries"]["IVC_PRUNE"]["retained_visual_token_histogram"]
    counts = sorted((int(tokens), int(rows)) for tokens, rows in histogram.items())
    fig, axis = plt.subplots(figsize=(5.7, 3.8))
    axis.bar(
        [tokens for tokens, _ in counts],
        [rows for _, rows in counts],
        width=0.9,
        color="#0072B2",
    )
    axis.set_xlabel("Retained visual tokens after the IVC boundary")
    axis.set_ylabel("Validation examples")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "ivc_retained_token_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        quality = read_analysis("quality_analysis.json", "QUALITY_ANALYSIS_COMPLETE")
        timing = read_analysis("timing_natural_analysis.json", "TIMING_NATURAL_ANALYSIS_COMPLETE")
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        atomic_text(output / "ivc_quality_work.tex", quality_table(quality))
        atomic_text(output / "ivc_timing.tex", timing_table(timing))
        figures(quality, timing, output)
        values = {
            "quality": quality["summaries"],
            "quality_uncertainty": quality["paired_uncertainty"],
            "timing": timing["summaries"],
            "timing_contrasts": timing["paired_contrasts"],
            "ivc_relative_e2e_change_percent": timing["ivc_relative_e2e_change_percent"],
        }
        atomic_text(output / "ivc_report_values.json", json.dumps(values, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "assets_complete", "output": str(output)}, indent=2))
        return 0
    except (AssetError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
