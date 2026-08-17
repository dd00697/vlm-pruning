#!/usr/bin/env python
"""Resolve the registered SAQR branches and write the cumulative preliminary report."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for entry in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from run_saqr_campaign import CAMPAIGN_ROOT, atomic_json, sha256_file  # noqa: E402
from analyse_saqr_campaign import ANALYSIS_ROOT  # noqa: E402


class FinalizeError(RuntimeError):
    pass


def read_required(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FinalizeError(f"Required artifact is missing: {path}")
    return json.loads(path.read_text())


def classification() -> tuple[str, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    calibration = read_required(ANALYSIS_ROOT / "CALIBRATION_GATE.json")
    if calibration["status"] != "PASS":
        return "F_CALIBRATION_FAILURE", calibration, None, None
    discovery = read_required(ANALYSIS_ROOT / "DISCOVERY_QUALITY_GATE.json")
    if discovery["status"] != "PASS":
        return "E_NO_USEFUL_TRADEOFF", calibration, discovery, None
    timing = read_required(ANALYSIS_ROOT / "TIMING_ACCELERATION_GATE.json")
    if timing["status"] == "PASS":
        return (
            "A_REAL_ACCELERATION_STRONG_QUALITY"
            if discovery["strong_quality"]
            else "B_REAL_ACCELERATION_ACCEPTABLE_QUALITY",
            calibration,
            discovery,
            timing,
        )
    return "C_QUALITY_PRESERVED_NO_REAL_ACCELERATION", calibration, discovery, timing


def skip_artifacts(calibration: dict[str, Any], discovery: dict[str, Any] | None) -> None:
    if calibration["status"] != "PASS":
        atomic_json(
            ANALYSIS_ROOT / "DISCOVERY_AND_TIMING_SKIP.json",
            {
                "status": "SKIPPED_DISCOVERY_AND_TIMING_CALIBRATION_GATE_FAILED",
                "calibration_gate_sha256": sha256_file(ANALYSIS_ROOT / "CALIBRATION_GATE.json"),
            },
        )
    elif discovery is not None and discovery["status"] != "PASS":
        atomic_json(
            ANALYSIS_ROOT / "MATCHED_TIMING_SKIP.json",
            {
                "status": "SKIPPED_MATCHED_TIMING_QUALITY_GATE_FAILED",
                "discovery_gate_sha256": sha256_file(ANALYSIS_ROOT / "DISCOVERY_QUALITY_GATE.json"),
            },
        )


def architecture_figure() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    labels = [
        "Frozen CLIP\n576 patches",
        "Projector\nrank-8 LoRA",
        "SAQR\nquery + spatial merge",
        "196 fixed tokens",
        "Frozen LLM\none dense cache",
    ]
    figure, axis = plt.subplots(figsize=(10.0, 2.2))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 2)
    axis.axis("off")
    positions = [0.2, 2.2, 4.2, 6.2, 8.2]
    for index, (x, label) in enumerate(zip(positions, labels, strict=True)):
        box = FancyBboxPatch(
            (x, 0.55), 1.55, 0.9, boxstyle="round,pad=0.08", facecolor="#dceaf7", edgecolor="#315d7d"
        )
        axis.add_patch(box)
        axis.text(x + 0.775, 1.0, label, ha="center", va="center", fontsize=9)
        if index < len(positions) - 1:
            axis.add_patch(FancyArrowPatch((x + 1.57, 1.0), (positions[index + 1] - 0.04, 1.0), arrowstyle="->", mutation_scale=12))
    figure.tight_layout()
    figure.savefig(ANALYSIS_ROOT / "saqr_architecture.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def fmt(value: float) -> str:
    return f"{value:.6f}"


def build_report(
    terminal: str,
    calibration: dict[str, Any],
    discovery: dict[str, Any] | None,
    timing: dict[str, Any] | None,
    training: dict[str, Any],
) -> str:
    lines = [
        "# SAQR Quick Feasibility Campaign",
        "",
        f"Final preliminary classification: `{terminal}`.",
        "",
        "## Question and method",
        "",
        "This single-seed proof of concept tested whether one trained pre-LLM merger could replace "
        "decoder-internal selection/segmentation. SAQR projects every one of the 576 CLIP patches "
        "with the existing multimodal projector, combines deterministic spatial overlap with a "
        "query-conditioned mass-conserving assignment, and emits a fixed raster-ordered 14x14 grid. "
        "The frozen LLM then processes 196 visual tokens through all 32 layers using one ordinary "
        "contiguous cache. No selector, ranking, gather, or decoder boundary is used at inference.",
        "",
        "Training used seed 0 only: exactly two 16x16 alignment epochs and two 14x14 adaptation "
        "epochs. The vision tower and LLM were frozen; SAQR/query components, a training-only box "
        "head, and rank-8 projector LoRA were trainable. The cache and training used only the frozen "
        "3,000 RefCOCO-train rows (2,160 fit, 240 early-stop, 600 calibration).",
        "",
        "## Training and calibration",
        "",
        f"Stage A selected epoch {training['stages']['A']['selected_epoch']}; Stage B selected epoch "
        f"{training['stages']['B']['selected_epoch']}. Frozen-parameter checksums remained exact.",
        "",
        "| Calibration method | Parse rate | Acc@0.5 | mIoU |",
        "|---|---:|---:|---:|",
        f"| K12 reference | {fmt(calibration['reference_k12']['parse_rate'])} | {fmt(calibration['reference_k12']['acc_at_0_5'])} | {fmt(calibration['reference_k12']['miou'])} |",
        f"| SAQR196 | {fmt(calibration['candidate']['parse_rate'])} | {fmt(calibration['candidate']['acc_at_0_5'])} | {fmt(calibration['candidate']['miou'])} |",
        "",
        f"Calibration gate: **{calibration['status']}** (Acc loss {fmt(calibration['acc_loss'])}, "
        f"mIoU loss {fmt(calibration['miou_loss'])}).",
    ]
    if discovery is None:
        lines.extend([
            "",
            "The preregistered calibration gate failed, so the 1,000-row discovery and matched "
            "timing branches were skipped. No recovery training or alternative architecture was run.",
        ])
    else:
        lines.extend([
            "",
            "## Discovery quality",
            "",
            "| Method | Parse rate | Acc@0.5 | mIoU | Visual token-layer work |",
            "|---|---:|---:|---:|---:|",
            f"| K12 original | {fmt(discovery['reference_k12']['parse_rate'])} | {fmt(discovery['reference_k12']['acc_at_0_5'])} | {fmt(discovery['reference_k12']['miou'])} | 7,872 |",
            f"| E256_POOL (historical) | — | 0.402000 | 0.413749 | 8,192 |",
            f"| SAQR196 | {fmt(discovery['candidate']['parse_rate'])} | {fmt(discovery['candidate']['acc_at_0_5'])} | {fmt(discovery['candidate']['miou'])} | 6,272 |",
            f"| SAQR196 no query | {fmt(discovery['no_query']['parse_rate'])} | {fmt(discovery['no_query']['acc_at_0_5'])} | {fmt(discovery['no_query']['miou'])} | 6,272 |",
            "",
            f"Discovery quality gate: **{discovery['status']}**; strong-quality flag: "
            f"**{discovery['strong_quality']}**. SAQR uses 20.3% less visual token-layer work than "
            "K12. The no-query row is a same-checkpoint diagnostic, not a separately trained ablation. "
            "E256_POOL is an untrained pre-projector baseline and is not a one-factor causal comparison.",
        ])
        if timing is None:
            lines.extend([
                "",
                "Quality did not pass the preregistered discovery gate, so matched timing was skipped.",
            ])
        else:
            e2e = timing["metrics"]["end_to_end_latency_s"]
            prefill = timing["metrics"]["main_prefill_with_overhead_latency_s"]
            lines.extend([
                "",
                "## Matched physical efficiency",
                "",
                "| Method | Natural E2E (s) | Prefill incl. method overhead (s) | Prompt KV bytes |",
                "|---|---:|---:|---:|",
                f"| K12 original | {fmt(e2e['k12_mean'])} | {fmt(prefill['k12_mean'])} | {timing['metrics']['prompt_kv_total_bytes']['k12_mean']:.0f} |",
                f"| SAQR196 | {fmt(e2e['saqr196_mean'])} | {fmt(prefill['saqr196_mean'])} | {timing['metrics']['prompt_kv_total_bytes']['saqr196_mean']:.0f} |",
                "",
                f"Actual-acceleration gate: **{timing['status']}**. Mean E2E reduction was "
                f"{100.0 * timing['e2e_reduction']:.2f}% and prefill-including-overhead reduction was "
                f"{100.0 * timing['prefill_including_overhead_reduction']:.2f}%. The paired hierarchical "
                "interval—not token count alone—determines the speed claim.",
            ])
    lines.extend([
        "",
        "## Interpretation and limitations",
        "",
        "This experiment directly addresses the earlier selector/runtime problem: compression occurs "
        "once before decoder layer 1, while every patch contributes positive mass and the LLM follows "
        "a regular dense path. Whether that design actually solved the problem is represented by the "
        "classification and physical timing gate above; reduced work or KV alone is not called a speedup.",
        "",
        "The result is preliminary: one seed cannot establish training stability; task-specific "
        "RefCOCO training cannot establish task generality; the 1,000 validation rows are discovery "
        "rather than held-out confirmation; and the train/discovery manifests share 90 underlying "
        "COCO images despite zero sample/original-record ID overlap. SAQR is a trained follow-up and "
        "must remain distinct from the dissertation's training-free UG-Prune contribution.",
        "",
        "No extra seed, lower budget, architecture recovery, held-out split, transfer benchmark, "
        "fixed32 timing, or compiler/kernel search was run.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    terminal, calibration, discovery, timing = classification()
    training = read_required(CAMPAIGN_ROOT / "checkpoints/TRAINING_SUMMARY.json")
    if training["status"] != "COMPLETE":
        raise FinalizeError("Training summary is not complete")
    skip_artifacts(calibration, discovery)
    architecture_figure()
    report = build_report(terminal, calibration, discovery, timing, training)
    report_path = CAMPAIGN_ROOT / "RESEARCH_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    positive = terminal.startswith(("A_", "B_"))
    claims = {
        "classification": terminal,
        "single_seed_preliminary": True,
        "dissertation_integration_required": positive,
        "claims": {
            "training": "checkpoints/TRAINING_SUMMARY.json",
            "calibration": "analysis/CALIBRATION_GATE.json",
            "discovery": "analysis/DISCOVERY_QUALITY_GATE.json" if discovery else None,
            "timing": "analysis/TIMING_ACCELERATION_GATE.json" if timing else None,
        },
    }
    atomic_json(ANALYSIS_ROOT / "CLAIM_LEDGER.json", claims)
    registry = read_required(CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json")
    registry["cells"]["T_SAQR256_S0"] = "COMPLETE"
    registry["cells"]["T_SAQR196_S0"] = "COMPLETE"
    registry["cells"]["C_SAQR196_S0"] = "COMPLETE"
    if discovery is None:
        registry["cells"]["Q_SAQR196_S0"] = "SKIPPED_CALIBRATION_GATE_FAILED"
        registry["cells"]["Q_SAQR196_S0_NO_QUERY"] = "SKIPPED_CALIBRATION_GATE_FAILED"
        registry["cells"]["MATCHED_TIMING"] = "SKIPPED_CALIBRATION_GATE_FAILED"
    else:
        registry["cells"]["Q_SAQR196_S0"] = "COMPLETE"
        registry["cells"]["Q_SAQR196_S0_NO_QUERY"] = "COMPLETE"
        registry["cells"]["MATCHED_TIMING"] = "COMPLETE" if timing else "SKIPPED_QUALITY_GATE_FAILED"
    registry["status"] = "COMPLETE_PENDING_FINAL_AUDIT"
    registry["classification"] = terminal
    atomic_json(CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json", registry)
    controller = read_required(CAMPAIGN_ROOT / "CONTROLLER_STATE.json")
    controller.update({
        "status": "FINAL_AUDIT_PENDING",
        "phase": "PHASE_7_FINAL_AUDIT",
        "last_completed_phase": "PHASE_6_REPORTING",
        "classification": terminal,
        "workers": [],
    })
    atomic_json(CAMPAIGN_ROOT / "CONTROLLER_STATE.json", controller)
    print(terminal)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FinalizeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
