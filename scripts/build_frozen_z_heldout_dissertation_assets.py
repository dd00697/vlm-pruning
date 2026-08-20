#!/usr/bin/env python3
"""Generate isolated dissertation assets from the sealed held-out analysis."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
for value in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from frozen_z_heldout_common import CAMPAIGN, HeldoutError, atomic_json, now, read_json, sha256


DRAFT = Path("/workspace/Dissertation_Draft")
GENERATED = DRAFT / "chapters/generated_frozen_z_heldout"
DERIVED = DRAFT / "analysis/derived/frozen_z_m448_0_refcoco_testab"
VISUAL_REPAIRS = DRAFT / "images/heldout_visual_repairs/final_evaluation/final_val_efficiency"
SOURCE_NAMES = ["HELDOUT_RESULTS.json", "PAIRED_CONTRASTS.json", "HELDOUT_DECISION.json", "SPLIT_HETEROGENEITY.json"]


def tex(value: object) -> str:
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def num(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def human_status(status: str) -> str:
    return {
        "HELDOUT_CONFIRMED_BOTH_SPLITS": "confirmed on both frozen splits under the preregistered descriptive rule",
        "HELDOUT_MIXED_SPLIT_RESULT": "a mixed split result under the preregistered descriptive rule",
        "HELDOUT_NOT_CONFIRMED": "not confirmed under the preregistered descriptive rule",
    }[status]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def build_visual_repairs() -> dict[str, dict[str, str]]:
    """Regenerate prior audited plots with the right-edge Full labels moved inward.

    The original final-validation assets remain byte-for-byte untouched.  The dissertation uses
    these campaign-specific display copies so its exhaustive visual QA can pass without changing
    the completed validation evidence.
    """
    import matplotlib.axes
    import finalize_final_val_efficiency as finalizer

    analysis = CAMPAIGN.parents[0] / "final_val_efficiency_baselines_20260818/analysis"
    records = json.loads((analysis / "JOINED_METHOD_RECORDS.json").read_text(encoding="utf-8"))
    pareto = read_json(analysis / "PARETO_FRONTIERS.json")
    original_annotate = matplotlib.axes.Axes.annotate

    def inward_full(axis, text, *args, **kwargs):
        if text == "Full":
            kwargs["xytext"] = (-23, 5)
        return original_annotate(axis, text, *args, **kwargs)

    matplotlib.axes.Axes.annotate = inward_full
    original_figures = finalizer.FIGURES
    try:
        finalizer.FIGURES = VISUAL_REPAIRS
        paths = finalizer.plots(records, pareto)
    finally:
        finalizer.FIGURES = original_figures
        matplotlib.axes.Axes.annotate = original_annotate
    return {path.name: {"path": str(path), "sha256": sha256(path)} for path in paths}


def build() -> dict[str, object]:
    analysis_root = CAMPAIGN / "analysis"
    sources = {name: analysis_root / name for name in SOURCE_NAMES}
    if any(not path.is_file() for path in sources.values()):
        raise HeldoutError("Held-out terminal analysis artifacts are incomplete")
    results = read_json(sources["HELDOUT_RESULTS.json"])
    decision = read_json(sources["HELDOUT_DECISION.json"])
    status = str(decision.get("status") or "")
    wording = human_status(status)
    metrics = results["metrics"]
    contrasts = results["paired_contrasts"]
    gates = results["gate_results"]
    heterogeneity = read_json(sources["SPLIT_HETEROGENEITY.json"])
    GENERATED.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    macros = [
        r"% Generated from sealed frozen-Z held-out artifacts. Do not edit.",
        rf"\newcommand{{\FrozenZHeldoutStatusRaw}}{{{tex(status)}}}",
        rf"\newcommand{{\FrozenZHeldoutStatusHuman}}{{{tex(wording)}}}",
    ]
    for split, suffix in (("testA", "TestA"), ("testB", "TestB")):
        split_pass = "Pass" if gates[split]["passed"] else "Fail"
        macros.append(rf"\newcommand{{\FrozenZHeldout{suffix}Gate}}{{{split_pass}}}")
        for method, method_suffix in (("Full", "Full"), ("Z_M448_0", "Z")):
            row = metrics[split][method]
            macros.extend([
                rf"\newcommand{{\FrozenZHeldout{suffix}{method_suffix}Acc}}{{{num(row['acc_at_0_5'])}}}",
                rf"\newcommand{{\FrozenZHeldout{suffix}{method_suffix}IoU}}{{{num(row['mean_iou'])}}}",
                rf"\newcommand{{\FrozenZHeldout{suffix}{method_suffix}Parsed}}{{{num(row['parsed_box_rate'])}}}",
                rf"\newcommand{{\FrozenZHeldout{suffix}{method_suffix}Samples}}{{{int(row['samples'])}}}",
                rf"\newcommand{{\FrozenZHeldout{suffix}{method_suffix}AverageTokens}}{{{int(round(row['average_visual_tokens']))}}}",
                rf"\newcommand{{\FrozenZHeldout{suffix}{method_suffix}Work}}{{{int(round(row['token_layer_work']))}}}",
            ])
        acc, iou = contrasts[split]["acc_at_0_5"], contrasts[split]["mean_iou"]
        gate = gates[split]["gates"]
        macros.extend([
            rf"\newcommand{{\FrozenZHeldout{suffix}AccDeltaPP}}{{{100*acc['point_delta']:+.2f}}}",
            rf"\newcommand{{\FrozenZHeldout{suffix}AccCILowPP}}{{{100*acc['lower_95']:+.2f}}}",
            rf"\newcommand{{\FrozenZHeldout{suffix}AccCIHighPP}}{{{100*acc['upper_95']:+.2f}}}",
            rf"\newcommand{{\FrozenZHeldout{suffix}IoUDelta}}{{{iou['point_delta']:+.4f}}}",
            rf"\newcommand{{\FrozenZHeldout{suffix}IoUCILow}}{{{iou['lower_95']:+.4f}}}",
            rf"\newcommand{{\FrozenZHeldout{suffix}IoUCIHigh}}{{{iou['upper_95']:+.4f}}}",
            rf"\newcommand{{\FrozenZHeldout{suffix}ParsedGate}}{{{'Pass' if gate['parsed_box_rate'] else 'Fail'}}}",
            rf"\newcommand{{\FrozenZHeldout{suffix}AccGate}}{{{'Pass' if gate['accuracy_loss'] else 'Fail'}}}",
            rf"\newcommand{{\FrozenZHeldout{suffix}IoUGate}}{{{'Pass' if gate['miou_loss'] else 'Fail'}}}",
        ])
    macros_path = GENERATED / "macros.tex"
    atomic_text(macros_path, "\n".join(macros) + "\n")
    failed: list[str] = []
    gate_labels = {"parsed_box_rate": "parsed-box rate", "accuracy_loss": "accuracy-loss", "miou_loss": "mIoU-loss"}
    for split in ("testA", "testB"):
        failed_names = [label for key, label in gate_labels.items() if not gates[split]["gates"][key]]
        if failed_names:
            failed.append(f"{split}: {', '.join(failed_names)}")
    gate_sentence = (
        "All three preregistered gates passed on both splits."
        if not failed
        else "Failed preregistered gates were " + "; ".join(failed) + "."
    )
    split_diff = heterogeneity["split_difference_in_z_minus_full"]
    lines = [
        r"% Generated from sealed frozen-Z held-out artifacts. Do not edit.",
        r"The candidate was frozen from validation before either held-out split was executed. Under the",
        rf"frozen descriptive rule, the outcome is \emph{{{tex(wording.replace(' under the preregistered descriptive rule', ''))}}}. No split pooling,",
        r"candidate reselection or held-out timing measurement is part of this result.",
        tex(gate_sentence),
        "",
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{One-shot held-out quality and visual-token work.}", r"\label{tab:frozen-z-heldout-quality}",
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Split & Method & $N$ & Parsed (\%) & Acc@0.5 (\%) & mIoU & Avg. tokens & Work \\", r"\midrule",
    ]
    for split in ("testA", "testB"):
        for method in ("Full", "Z_M448_0"):
            row = metrics[split][method]
            display = "Z-M448-0" if method == "Z_M448_0" else method
            lines.append(
                f"{split} & {display} & {row['samples']:,} & {100*row['parsed_box_rate']:.2f} & "
                f"{100*row['acc_at_0_5']:.2f} & {num(row['mean_iou'])} & "
                f"{row['average_visual_tokens']:.0f} & {row['token_layer_work']:,.0f} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Paired held-out contrasts and preregistered split-wise gates. Contrasts are Z-M448-0 minus Full.}",
        r"\label{tab:frozen-z-heldout-contrasts}",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrllll}", r"\toprule",
        r"Split & Acc. delta pp [95\% CI] & mIoU delta [95\% CI] & Parsed & Acc. loss & mIoU loss & Split result \\", r"\midrule"])
    for split in ("testA", "testB"):
        acc, iou = contrasts[split]["acc_at_0_5"], contrasts[split]["mean_iou"]
        g = gates[split]["gates"]
        lines.append(
            f"{split} & {100*acc['point_delta']:+.2f} [{100*acc['lower_95']:+.2f}, {100*acc['upper_95']:+.2f}] & "
            f"{iou['point_delta']:+.4f} [{iou['lower_95']:+.4f}, {iou['upper_95']:+.4f}] & "
            f"{'Pass' if g['parsed_box_rate'] else 'Fail'} & {'Pass' if g['accuracy_loss'] else 'Fail'} & "
            f"{'Pass' if g['miou_loss'] else 'Fail'} & {'Pass' if gates[split]['passed'] else 'Fail'} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}", "",
        r"Intervals use 10,000 deterministic paired manifest-row bootstrap resamples separately within each split.",
        f"Descriptive split heterogeneity (testA minus testB in the Z-M448-0-minus-Full contrast) is "
        f"{100*split_diff['acc_at_0_5']:+.2f} accuracy percentage points and {split_diff['mean_iou']:+.4f} mIoU; it is not pooled.",
        r"The thresholds are descriptive confirmation criteria, not an equivalence, non-inferiority or population-generalisation test.",
    ])
    main_path = GENERATED / "main.tex"
    atomic_text(main_path, "\n".join(lines) + "\n")
    copied: dict[str, dict[str, object]] = {}
    for name, source in sources.items():
        target = DERIVED / name
        atomic_copy(source, target)
        copied[name] = {"source_sha256": sha256(source), "derived_sha256": sha256(target)}
    claim_evidence = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": now(),
        "heldout_status": status,
        "claims": {
            "candidate_frozen_before_heldout": {
                "supported": True,
                "evidence": "PRE_INFERENCE_LOCK.json and frozen validation decision",
            },
            "splitwise_quality_confirmation": {
                "supported": True,
                "evidence": "HELDOUT_RESULTS.json, PAIRED_CONTRASTS.json, HELDOUT_DECISION.json",
                "testA_passed": bool(gates["testA"]["passed"]),
                "testB_passed": bool(gates["testB"]["passed"]),
            },
            "heldout_timing_claim": {"supported": False, "reason": "No held-out timing campaign was authorized or run."},
            "population_equivalence_claim": {"supported": False, "reason": "Thresholds are descriptive and split-wise."},
        },
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
    }
    claim_path = DERIVED / "CLAIM_EVIDENCE_LEDGER.json"
    atomic_json(claim_path, claim_evidence)
    visual_repairs = build_visual_repairs()
    manifest = {
        "schema_version": 1, "status": "complete", "created_at_utc": now(),
        "heldout_status": status, "sources": copied,
        "generated": {
            "macros.tex": {"path": str(macros_path), "sha256": sha256(macros_path)},
            "main.tex": {"path": str(main_path), "sha256": sha256(main_path)},
            "CLAIM_EVIDENCE_LEDGER.json": {"path": str(claim_path), "sha256": sha256(claim_path)},
        },
        "visual_repairs": visual_repairs,
    }
    atomic_json(DERIVED / "ASSET_MANIFEST.json", manifest)
    return manifest


if __name__ == "__main__":
    try:
        print(json.dumps(build(), indent=2, sort_keys=True))
    except (HeldoutError, OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
