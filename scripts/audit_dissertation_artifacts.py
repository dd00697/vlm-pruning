#!/usr/bin/env python3
"""Inventory and reopen every raw experiment family cited by the dissertation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DISS = Path("/workspace/Dissertation_Draft")
OUT = ROOT / "experiments/final_selector_ablation_and_repair_20260811/prelaunch_artefact_inventory.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_pruning.artifacts import (  # noqa: E402
    ArtifactIntegrityError,
    read_jsonl_strict,
    sha256_file,
    validate_metric_agreement,
    validate_rows,
)


def development_paths() -> list[tuple[str, Path, str]]:
    official = ROOT / "experiments/runs_official_baselines"
    ug = ROOT / "experiments/runs_ug_prune_single_pass"
    entries: list[tuple[str, Path, str]] = [
        ("C1", official / "official_fullmatrix_refcoco_full_tokens576_seed0_bf16", "quality"),
        ("C2", official / "official_fullmatrix_refcoco_random_tokens48_seed0_bf16", "quality"),
        ("C2", official / "official_fullmatrix_refcoco_uniform_tokens48_seed0_bf16", "quality"),
        ("C3,C6,C11,C13", ug / "official_fullmatrix_refcoco_ug_k12_b48_more_context_bf16", "quality"),
        ("C3", ug / "ablation_fullmatrix_refcoco_ug_k12_b128_more_anchors_bf16", "quality"),
        ("C5", ug / "ablation_fullmatrix_refcoco_ug_k12_b48_more_context_query_pooled_bf16", "quality"),
        ("C6", ug / "ablation_fullmatrix_refcoco_ug_k12_b72_more_context_bf16", "quality"),
        ("C5,C6,C12,C13", ug / "official_fullmatrix_refcoco_ug_k12_b128_more_context_bf16", "quality"),
        ("C5,C12", ug / "ablation_fullmatrix_refcoco_ug_k12_b128_more_context_query_pooled_bf16", "quality"),
        ("C6", ug / "official_fullmatrix_refcoco_ug_k12_b192_more_context_bf16", "quality"),
        ("C7", ug / "ablation_fullmatrix_refcoco_ug_k12_b48_more_context_single_layer_bf16", "quality"),
        ("C4,C14", official / "latency_iso_refcoco_full_tokens576_limit1000_seed0_bf16", "incidental_latency"),
        ("C4", ug / "latency_iso_refcoco_ug_k12_b48_more_context_limit1000_bf16", "incidental_latency"),
        ("C8", ug / "ablation_fullmatrix_refcoco_ug_k4_b48_more_context_single_layer_bf16", "quality"),
        ("C8", ug / "ablation_fullmatrix_refcoco_ug_k8_b48_more_context_single_layer_bf16", "quality"),
        ("C11", ug / "strong_ablation_refcoco_position_compact_k12_b48_more_context_bf16", "quality"),
        ("C12", ug / "strong_ablation_refcoco_query_pooled_more_anchors_k12_b128_bf16", "quality"),
    ]
    for layer, budget in ((12,48),(16,48),(20,48),(8,128),(12,128),(16,128),(20,128)):
        entries.append(("C8,C9,C10,C13", ug / f"strong_ablation_refcoco_depth_l{layer}_b{budget}_single_layer_bf16", "quality"))
    for budget in (48, 128):
        entries.append(("C13", ug / f"strong_ablation_refcoco_fusion_8_12_b{budget}_more_context_bf16", "quality"))
    for layer in (8, 12, 16, 20):
        for budget in (48, 128):
            entries.append(("C14", ug / f"strong_ablation_latency_refcoco_depth_l{layer}_b{budget}_limit1000_bf16", "incidental_latency"))
    return entries


REPLACEMENTS = {
    "official_fullmatrix_refcoco_random_tokens48_seed0_bf16": "repair_val_prellm_random_b48_seed0",
    "ablation_fullmatrix_refcoco_ug_k12_b72_more_context_bf16": "repair_val_k12_b72_fused_more_context",
    "strong_ablation_refcoco_query_pooled_more_anchors_k12_b128_bf16": "repair_val_k12_b128_pooled_more_anchors",
    "strong_ablation_refcoco_depth_l12_b48_single_layer_bf16": "repair_val_k12_b48_single_layer_more_context",
    "strong_ablation_refcoco_depth_l16_b48_single_layer_bf16": "repair_val_k16_b48_single_layer_more_context",
}

FOLLOWUP_REPLACEMENTS = {
    "followup_refcoco_val_ug_k20_b48_compact_seed0": "clean_val_k20_b48_structured_compact",
    "followup_refcoco_testA_random_k20_b48_seed0": "clean_testA_k20_b48_random_seed0",
    "followup_refcoco_testA_uniform_k20_b48_seed0": "clean_testA_k20_b48_uniform_seed0",
    "followup_refcoco_testB_random_k20_b48_seed0": "clean_testB_k20_b48_random_seed0",
    "followup_refcoco_testB_uniform_k20_b48_seed0": "clean_testB_k20_b48_uniform_seed0",
    "followup_refcoco_testA_ug_k12_b48_depth_only_seed0": "clean_testA_k12_b48_structured_seed0",
    "followup_refcoco_testB_ug_k12_b48_depth_only_seed0": "clean_testB_k12_b48_structured_seed0",
}

SUPERSEDED_WITHOUT_RERUN = {
    "ablation_fullmatrix_refcoco_ug_k12_b48_more_context_single_layer_bf16":
        "duplicate of repair_val_k12_b48_single_layer_more_context",
}


def followup_paths() -> list[tuple[str, Path, str]]:
    base = ROOT / "experiments/followup_evaluation"
    relative = [
        "runs/refcoco/current_k20_gap/followup_refcoco_val_ug_k20_b48_gap_current_seed0",
        "runs/refcoco/current_k20_gap/followup_refcoco_testA_ug_k20_b48_gap_current_seed0",
        "runs/refcoco/current_k20_gap/followup_refcoco_testB_ug_k20_b48_gap_current_seed0",
        "runs/refcoco/compact_ablation/followup_refcoco_val_ug_k20_b48_compact_seed0",
        "runs/refcoco/controls_k20_b48/followup_refcoco_testA_random_k20_b48_seed0",
        "runs/refcoco/controls_k20_b48/followup_refcoco_testA_uniform_k20_b48_seed0",
        "runs/refcoco/controls_k20_b48/followup_refcoco_testB_random_k20_b48_seed0",
        "runs/refcoco/controls_k20_b48/followup_refcoco_testB_uniform_k20_b48_seed0",
        "runs/refcoco/controls_k12_b48/followup_refcoco_testA_random_k12_b48_seed0",
        "runs/refcoco/controls_k12_b48/followup_refcoco_testA_uniform_k12_b48_seed0",
        "runs/refcoco/controls_k12_b48/followup_refcoco_testB_random_k12_b48_seed0",
        "runs/refcoco/controls_k12_b48/followup_refcoco_testB_uniform_k12_b48_seed0",
        "runs/refcoco/k12_depth_only/followup_refcoco_testA_ug_k12_b48_depth_only_seed0",
        "runs/refcoco/k12_depth_only/followup_refcoco_testB_ug_k12_b48_depth_only_seed0",
    ]
    return [("followup-refcoco", base / item, "quality") for item in relative]


def audit_quality(claims: str, run_dir: Path, use: str) -> dict[str, Any]:
    row: dict[str, Any] = {"claims": claims, "run_path": str(run_dir), "report_use": use, "issues": []}
    required = ("config.json", "metrics.json", "run_environment.json", "per_sample_results.jsonl")
    absent = [name for name in required if not (run_dir / name).is_file()]
    if absent:
        row.update({"status": "replacement_required", "issues": [f"missing files: {absent}"]})
        return row
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        environment = json.loads((run_dir / "run_environment.json").read_text(encoding="utf-8"))
        predictions = read_jsonl_strict(run_dir / "per_sample_results.jsonl")
        manifest_path = Path(str(config.get("dataset/manifest")))
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        manifest = read_jsonl_strict(manifest_path)
        limit = config.get("limit")
        if limit is not None:
            manifest = manifest[: int(limit)]
        budget = int(config["tokens/target_visual_tokens"])
        result = validate_rows(predictions, manifest, budget=budget, original_tokens=int(config.get("tokens/original_visual_tokens", 576)), require_retained=True)
        aggregate = validate_metric_agreement(predictions, metrics, task=str(config.get("task", "grounding")))
        row.update({
            "config_exists": True, "metrics_exists": True, "environment_exists": True,
            "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
            "recorded_manifest_checksum": config.get("dataset/manifest_checksum"),
            "expected_rows": len(manifest), "physical_rows": len(predictions),
            "row_integrity": result, "aggregate_agreement": aggregate,
            "git_commit": environment.get("git_commit", config.get("source/git_commit")),
            "git_dirty": environment.get("git_dirty", config.get("source/git_dirty")),
            "source_hashes": config.get("source/sha256"),
            "config_sha256": sha256_file(run_dir / "config.json"),
            "metrics_sha256": sha256_file(run_dir / "metrics.json"),
            "predictions_sha256": sha256_file(run_dir / "per_sample_results.jsonl"),
        })
        if result["status"] != "pass":
            row["issues"].extend(result["issues"])
        if aggregate["status"] != "pass":
            row["issues"].append(f"aggregate mismatch: {aggregate['mismatches']}")
    except (OSError, ValueError, KeyError, ArtifactIntegrityError) as exc:
        row["issues"].append(str(exc))
    replacement = REPLACEMENTS.get(run_dir.name) or FOLLOWUP_REPLACEMENTS.get(run_dir.name)
    if replacement:
        row.update({"status": "replacement_required", "replacement_run_id": replacement})
    elif run_dir.name in SUPERSEDED_WITHOUT_RERUN:
        row.update({"status": "superseded", "supersession_reason": SUPERSEDED_WITHOUT_RERUN[run_dir.name]})
    elif row["issues"]:
        # Complete legacy development files predate the final unique-ID and
        # retained-index contract. They remain auditable historical evidence,
        # but revised headline claims must use clean replacements/final runs.
        row["status"] = "historical_only" if row.get("physical_rows") == row.get("expected_rows") else "replacement_required"
    elif bool(row.get("git_dirty")):
        row["status"] = "historical_only"
    else:
        row["status"] = "verified_current"
    return row


def audit_final_campaign() -> dict[str, Any]:
    completion = ROOT / "experiments/final_evaluation/FINAL_COMPLETION_REPORT.json"
    report = json.loads(completion.read_text(encoding="utf-8"))
    issues = []
    checked = []
    for run in report.get("runs", []):
        run_dir = Path(run["run_dir"])
        metrics = run_dir / "metrics.json"
        results_name = "per_sample_timing.jsonl" if run.get("group") in {"timing", "timing_smoke", "serial_timing"} else "per_sample_results.jsonl"
        results = run_dir / results_name
        actual_metrics = sha256_file(metrics) if metrics.is_file() else None
        actual_results = sha256_file(results) if results.is_file() else None
        ok = actual_metrics == run.get("metrics_sha256") and actual_results == run.get("results_sha256")
        if not ok:
            issues.append(run["id"])
        checked.append({"id": run["id"], "group": run["group"], "run_dir": str(run_dir), "hashes_match_completion_report": ok})
    return {"status": "verified_current" if report.get("status") == "complete" and not issues else "replacement_required", "completion_report_sha256": sha256_file(completion), "reopened_runs": checked, "issues": issues}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    entries = [audit_quality(*item) for item in [*development_paths(), *followup_paths()]]
    final = audit_final_campaign()
    unexpected = [row for row in entries if row["status"] == "replacement_required" and not row.get("replacement_run_id")]
    payload = {
        "status": "pass" if not unexpected and final["status"] == "verified_current" else "fail",
        "scope": ["development evidence builders and claim matrix", "August follow-up evidence builder", "registered final evaluation and timing completion report"],
        "entries": entries, "final_evaluation": final,
        "registered_replacements": sorted({row["replacement_run_id"] for row in entries if row.get("replacement_run_id")}),
        "audit_added_replacements": [{"run_path": row["run_path"], "issues": row["issues"]} for row in unexpected],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "entries": len(entries), "registered_replacements": len(payload["registered_replacements"]), "unexpected_replacements": len(unexpected), "output": str(args.output)}, indent=2))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
