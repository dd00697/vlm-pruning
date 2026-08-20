#!/usr/bin/env python3
"""Build the single table/figure dissertation block for the frozen-Z error analysis."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
for value in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.frozen_z_error_analysis_common import (  # noqa: E402
    CAMPAIGN, DISSERTATION, ErrorAnalysisFailure, atomic_json, atomic_text, now, read_json, record_file,
    require, sha256,
)

ANALYSIS = CAMPAIGN / "analysis"
GENERATED = DISSERTATION / "chapters/generated_frozen_z_error_analysis"
DERIVED = DISSERTATION / "analysis/derived/frozen_z_error_analysis_report_audit"
FIGURES = DISSERTATION / "images/final_evaluation/frozen_z_error_analysis"


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}"


def build() -> dict[str, object]:
    inventory = read_json(ANALYSIS / "ANALYSIS_INVENTORY.json")
    require(inventory.get("status") == "complete", "Analysis inventory is incomplete")
    for relative, expected in inventory["files"].items():
        path = CAMPAIGN / relative
        require(record_file(path) == expected, f"Analysis inventory binding differs: {relative}")
    analysis = read_json(ANALYSIS / "ERROR_ANALYSIS.json")
    require(analysis.get("status") == "complete", "Error analysis is incomplete")
    require(analysis.get("heldout_status_unchanged") == "HELDOUT_MIXED_SPLIT_RESULT",
            "Held-out status changed")
    results = analysis["results"]
    GENERATED.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    macro_lines = ["% Generated from the V3-locked paired error analysis; do not hand-edit."]
    for split, suffix in (("testA", "TestA"), ("testB", "TestB")):
        row = results[split]
        counts = row["transitions"]
        curve = row["threshold_curve"]["0.50"]
        summary = row["iou_delta_summaries"]
        macro_lines.extend([
            rf"\newcommand{{\FrozenZError{suffix}BothCorrect}}{{{counts['both_correct']:,}}}",
            rf"\newcommand{{\FrozenZError{suffix}FullOnly}}{{{counts['full_only']:,}}}",
            rf"\newcommand{{\FrozenZError{suffix}ZOnly}}{{{counts['z_only']:,}}}",
            rf"\newcommand{{\FrozenZError{suffix}BothWrong}}{{{counts['both_incorrect']:,}}}",
            rf"\newcommand{{\FrozenZError{suffix}AccGap}}{{{pct(curve['point'])}}}",
            rf"\newcommand{{\FrozenZError{suffix}MeanIoUDelta}}{{{summary['mean']:+.4f}}}",
        ])
    macro_path = GENERATED / "macros.tex"
    atomic_text(macro_path, "\n".join(macro_lines) + "\n")

    lines = [
        r"% Generated from the V3-locked paired error analysis; do not hand-edit.",
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Post-hoc paired correctness transitions at the primary IoU threshold of 0.5. Shares and percentile intervals are computed separately within each frozen split from 10,000 paired manifest-row bootstrap resamples.}",
        r"\label{tab:frozen-z-error-transitions}",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"Split & Paired state & Count & Share (\%) & Paired 95\% interval (\%) \\", r"\midrule",
    ]
    labels = {
        "both_correct": "Both correct", "full_only": "Full only", "z_only": "Z-M448-0 only",
        "both_incorrect": "Both incorrect",
    }
    for split in ("testA", "testB"):
        row = results[split]
        for state in ("both_correct", "full_only", "z_only", "both_incorrect"):
            interval = row["transition_rates"][state]
            lines.append(
                f"{split} & {labels[state]} & {row['transitions'][state]:,} & {pct(interval['rate'])} & "
                f"[{pct(interval['lower_95'])}, {pct(interval['upper_95'])}] \\\\"
            )
        if split == "testA":
            lines.append(r"\addlinespace")
    lines.extend([
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
        r"At the frozen 0.5 threshold, many examples change correctness state in both directions. TestA",
        r"has 748 Full-only and 526 Z-M448-0-only successes: a net imbalance of 222 examples, equal to",
        r"a 3.92 percentage-point Full advantage. TestB has 622 Full-only and 505 Z-M448-0-only",
        r"successes, a smaller imbalance of 117 examples or 2.30 points. The mixed held-out decision",
        r"therefore comes from the direction of paired threshold crossings, not simply from the total",
        r"number of examples whose status changes.", "",
        r"\begin{figure}[H]", r"\centering",
        r"\includegraphics[width=0.98\textwidth]{final_evaluation/frozen_z_error_analysis/paired_localization_threshold_crossing.pdf}",
        r"\caption{Post-hoc threshold sensitivity for the same sealed boxes. Positive values favour Full; shading is the paired 95\% interval. The dashed line marks the preregistered 0.5 threshold.}",
        r"\label{fig:frozen-z-error-thresholds}", r"\end{figure}", "",
        r"The fixed threshold grid adds a localisation-precision diagnostic without creating new model",
        r"outputs. At IoU 0.30, the Full-minus-Z-M448-0 difference is $-0.87$ points on testA and",
        r"$+0.12$ on testB, with both intervals crossing zero. Around and above the primary threshold,",
        r"the Full advantage grows, reaching 5.82 points at IoU 0.60 on testA and 2.83 points on testB.",
        r"Mean IoU is only 0.0099 lower for Z-M448-0 on testA and 0.0042 lower on testB; moreover, about",
        r"36\% of paired boxes on each split differ by at most 0.05 IoU. Together these observations are",
        r"consistent with coarse target localisation being more stable than precise overlap under the",
        r"complete compressed schedule.", "",
        r"This analysis is explanatory rather than confirmatory: it was specified after the held-out",
        r"outcome was known. It leaves the mixed split status unchanged, does not pool splits, and does not",
        r"search semantic groups or hand-selected examples. Most importantly, Z-M448-0 is a three-boundary",
        r"schedule evaluated as one bundle. These data cannot attribute the crossings to its layer-4 merge,",
        r"layer-12 selection or layer-20 zero-tail transition separately; that would require a new frozen",
        r"ablation. The bounded conclusion is that the testA failure is explained descriptively by a larger",
        r"Full-only imbalance at precision-sensitive thresholds, not that any single stage caused it.",
    ])
    main_path = GENERATED / "main.tex"
    atomic_text(main_path, "\n".join(lines) + "\n")

    figure_source = ANALYSIS / "figures/paired_localization_threshold_crossing.pdf"
    figure_target = FIGURES / figure_source.name
    shutil.copyfile(figure_source, figure_target)
    copied: dict[str, object] = {}
    for name in ("INPUT_AUDIT.json", "ERROR_ANALYSIS.json", "ERROR_ANALYSIS_TRANSITIONS.csv",
                 "ERROR_ANALYSIS_THRESHOLD_CURVE.csv", "ERROR_ANALYSIS_SUMMARIES.csv", "ANALYSIS_INVENTORY.json"):
        source = ANALYSIS / name
        target = DERIVED / name
        shutil.copyfile(source, target)
        copied[name] = {"source": record_file(source), "derived": record_file(target)}
    report_target = DERIVED / "RESEARCH_REPORT.md"
    shutil.copyfile(CAMPAIGN / "RESEARCH_REPORT.md", report_target)
    copied["RESEARCH_REPORT.md"] = {
        "source": record_file(CAMPAIGN / "RESEARCH_REPORT.md"), "derived": record_file(report_target),
    }
    manifest = {
        "schema_version": 1, "status": "complete", "created_at_utc": now(),
        "heldout_status_unchanged": "HELDOUT_MIXED_SPLIT_RESULT",
        "logical_tables": 1, "logical_figures": 1,
        "generated": {
            "macros.tex": {"path": str(macro_path), **record_file(macro_path)},
            "main.tex": {"path": str(main_path), **record_file(main_path)},
            "figure": {"path": str(figure_target), **record_file(figure_target)},
        },
        "derived": copied,
        "source_analysis_sha256": sha256(ANALYSIS / "ERROR_ANALYSIS.json"),
    }
    atomic_json(DERIVED / "ASSET_MANIFEST.json", manifest)
    return manifest


if __name__ == "__main__":
    try:
        print(json.dumps(build(), indent=2, sort_keys=True))
    except (ErrorAnalysisFailure, OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
