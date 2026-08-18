#!/usr/bin/env python
"""Fail-closed terminal audit for the recent-methods full RefCOCO-val goal."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "experiments/recent_methods_full_refcoco_val_20260817"
DISSERTATION = Path("/workspace/Dissertation_Draft")
MANIFEST = REPO / "data/manifests/final_evaluation/refcoco_val_unique.jsonl"
FINAL_REPORT = CAMPAIGN / "FINAL_COMPLETION_REPORT.json"
PROTOCOL = CAMPAIGN / "FROZEN_PROTOCOL.json"
REGISTRY = CAMPAIGN / "EXPERIMENT_REGISTRY.json"
STATE = CAMPAIGN / "CONTROLLER_STATE.json"
ANALYSIS_MANIFEST = CAMPAIGN / "analysis/ANALYSIS_MANIFEST.json"
ASSET_MANIFEST = DISSERTATION / "analysis/derived/recent_methods_full_refcoco_val/ASSET_MANIFEST.json"
CHANGE_RECORD = CAMPAIGN / "DISSERTATION_CHANGE_RECORD.json"
PDF_QA = DISSERTATION / "analysis/derived/recent_methods_full_refcoco_val/PDF_QA.json"
TEST_REPORT = CAMPAIGN / "FINAL_TEST_REPORT.json"
ROOT_PDF = Path("/workspace/dissertation_final.pdf")
SOURCE_PDF = DISSERTATION / "Dissertation_Template.pdf"
ROWS = 8_811


class AuditError(RuntimeError):
    """Raised when any terminal requirement lacks passing evidence."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise AuditError(f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_hash_manifest(root: Path, manifest: dict[str, Any], label: str) -> dict[str, Any]:
    if manifest.get("status") != "complete":
        raise AuditError(f"{label} is not complete")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise AuditError(f"{label} has no file hashes")
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise AuditError(f"{label} artifact differs: {path}")
    return {"status": "pass", "files": len(files)}


def validate_physical_runs(protocol: dict[str, Any]) -> dict[str, Any]:
    from scripts.validate_progressive_oracle_run import validate_run

    run_root = CAMPAIGN / "runs"
    expected = sorted((CAMPAIGN / cell["full_output"]).resolve() for cell in protocol["cells"].values())
    actual = sorted(path.resolve() for path in run_root.iterdir() if path.is_dir())
    if actual != expected or len(actual) != 5:
        raise AuditError("Full-run directory set is not exactly the five registered cells")
    reports = {}
    for cell_id, cell in protocol["cells"].items():
        run_dir = CAMPAIGN / cell["full_output"]
        report = validate_run(run_dir, MANIFEST, expected_rows=ROWS, finalize=False)
        if report.get("status") != "pass" or int(report.get("rows", -1)) != ROWS:
            raise AuditError(f"Physical validator failed: {cell_id}")
        reports[cell_id] = {
            "run_dir": str(run_dir.relative_to(CAMPAIGN)),
            "config_sha256": report["config_sha256"],
            "predictions_sha256": report["predictions_sha256"],
            "source_git_commit": report["source_git_commit"],
            "metrics": report["metrics"],
        }
    return {"status": "pass", "cells": reports, "count": len(reports), "rows_per_cell": ROWS}


def validate_protected(protocol: dict[str, Any]) -> dict[str, Any]:
    evidence = {}
    for relative, expected in protocol["protected_sha256"].items():
        path = REPO / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise AuditError(f"Protected artifact differs: {relative}")
        evidence[relative] = expected
    return {"status": "pass", "sha256": evidence}


def validate_source_epoch(protocol: dict[str, Any]) -> dict[str, Any]:
    amendment_path = CAMPAIGN / "SOURCE_EPOCH_AMENDMENT.json"
    amendment = read_json(amendment_path)
    if amendment.get("initial_protocol_commit") != protocol.get("source", {}).get("git_commit"):
        raise AuditError("Bounded source amendment does not match initial frozen epoch")
    for relative, expected in amendment["scientific_source_unchanged"].items():
        if sha256_file(REPO / relative) != expected:
            raise AuditError(f"Scientific inference source changed after execution: {relative}")
    for relative, record in amendment["changed_files"].items():
        if sha256_file(REPO / relative) != record["execution_sha256"]:
            raise AuditError(f"Bounded controller source changed after execution: {relative}")
    cell_commits = amendment.get("cell_execution_commits", {})
    if set(cell_commits) != set(protocol["cells"]):
        raise AuditError("Bounded source amendment does not map every registered cell")
    return {
        "status": "pass",
        "initial_commit": amendment["initial_protocol_commit"],
        "execution_commit": amendment["execution_commit"],
        "amendment_sha256": sha256_file(amendment_path),
        "scientific_source_unchanged": True,
        "cell_execution_commits": cell_commits,
    }


def validate_analysis_consistency() -> dict[str, Any]:
    method_path = CAMPAIGN / "analysis/METHOD_METRICS.json"
    figure_path = DISSERTATION / "analysis/derived/recent_methods_full_refcoco_val/FIGURE_DATA.json"
    methods = read_json(method_path)
    figure = read_json(figure_path)
    by_method = {row["method"]: row for row in methods}
    if len(by_method) != 8 or len(figure) != 16:
        raise AuditError("Method or two-panel figure inventory has the wrong size")
    for row in figure:
        source = by_method[row["method"]]
        expected = source[row["panel"]] * (100.0 if row["quality_scale"] == "percent" else 1.0)
        if abs(float(row["quality"]) - float(expected)) > 1e-12:
            raise AuditError(f"Figure quality differs for {row['method']} {row['panel']}")
        if int(row["token_layer_work"]) != int(source["token_layer_work"]):
            raise AuditError(f"Figure work differs for {row['method']}")
    report = (CAMPAIGN / "RESEARCH_REPORT.md").read_text(encoding="utf-8")
    tex = (DISSERTATION / "chapters/generated_recent_methods_full_val.tex").read_text(encoding="utf-8")
    for method, row in by_method.items():
        if f"{row['acc_at_0_5']:.4f}" not in report or f"{row['mean_iou']:.4f}" not in report:
            raise AuditError(f"Campaign report omits exact metrics for {method}")
        if f"{row['acc_at_0_5'] * 100:.2f}" not in tex or f"{row['mean_iou']:.4f}" not in tex:
            raise AuditError(f"Generated TeX omits exact metrics for {method}")
    return {
        "status": "pass",
        "methods": len(by_method),
        "figure_points": len(figure),
        "method_metrics_sha256": sha256_file(method_path),
        "figure_data_sha256": sha256_file(figure_path),
    }


def validate_dissertation() -> dict[str, Any]:
    change = read_json(CHANGE_RECORD)
    qa = read_json(PDF_QA)
    if change.get("status") != "complete" or not Path(change.get("backup_directory", "")).is_dir():
        raise AuditError("Dissertation change/backup record is incomplete")
    if qa.get("status") != "pass" or not qa.get("visual_inspection", {}).get("status") == "pass":
        raise AuditError("PDF structural/text/visual QA is incomplete")
    if not SOURCE_PDF.is_file() or not ROOT_PDF.is_file():
        raise AuditError("Final dissertation PDF is missing")
    if sha256_file(SOURCE_PDF) != sha256_file(ROOT_PDF):
        raise AuditError("User-facing PDF differs from validated build PDF")
    final_text = subprocess.check_output(["pdftotext", str(ROOT_PDF), "-"], text=True)
    required = (
        "Post-hoc full RefCOCO-validation comparison",
        "Recent full-validation schedules",
        "analytical visual-token work",
        "Six tail minus zero tail",
    )
    missing = [value for value in required if value not in final_text]
    if missing:
        raise AuditError(f"Final PDF omits required recent-methods text: {missing}")
    return {
        "status": "pass",
        "source_pdf_sha256": sha256_file(SOURCE_PDF),
        "user_pdf_sha256": sha256_file(ROOT_PDF),
        "pdf_qa_sha256": sha256_file(PDF_QA),
        "change_record_sha256": sha256_file(CHANGE_RECORD),
    }


def validate_tests_and_git() -> dict[str, Any]:
    tests = read_json(TEST_REPORT)
    if tests.get("status") != "pass" or int(tests.get("tests", 0)) < 262 or int(tests.get("returncode", 1)) != 0:
        raise AuditError("Final complete repository test evidence is insufficient")
    output_path = CAMPAIGN / tests["output_path"]
    if not output_path.is_file() or sha256_file(output_path) != tests["output_sha256"]:
        raise AuditError("Final test output hash differs")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).splitlines()
    allowed = {"?? PRO_REASONING_RESEARCH_HANDOFF.md", "?? legacy_goal2.md"}
    unexpected = sorted(set(status) - allowed)
    if unexpected:
        raise AuditError(f"Unexpected Git changes remain: {unexpected}")
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=REPO, text=True, capture_output=True)
    if diff_check.returncode != 0:
        raise AuditError(f"git diff --check failed: {diff_check.stdout}{diff_check.stderr}")
    return {"status": "pass", "tests": tests["tests"], "allowed_user_untracked": sorted(allowed)}


def validate_no_forbidden_or_active_work(protocol: dict[str, Any]) -> dict[str, Any]:
    configs = []
    for path in CAMPAIGN.rglob("config.json"):
        value = read_json(path)
        split = str(value.get("dataset/split") or value.get("split") or "val").lower()
        if split not in {"val", ""}:
            raise AuditError(f"Forbidden split config under campaign: {path}")
        if value.get("pre_llm_selector") not in {None, "none"}:
            raise AuditError(f"Forbidden pre-LLM selector under campaign: {path}")
        configs.append(str(path.relative_to(CAMPAIGN)))
    forbidden_names = ("testa", "testb", "training", "matched_timing", "transfer", "external_baseline", "router")
    forbidden_paths = [
        str(path.relative_to(CAMPAIGN))
        for path in CAMPAIGN.rglob("*")
        if any(term in path.name.lower() for term in forbidden_names)
        and "analysis" not in path.parts
        and path.name not in {"IMMUTABLE_CONTEXT.json"}
    ]
    if forbidden_paths:
        raise AuditError(f"Forbidden campaign paths exist: {forbidden_paths[:5]}")
    process_lines = subprocess.check_output(["ps", "-eo", "pid,args"], text=True).splitlines()
    active = [
        line.strip()
        for line in process_lines
        if any(name in line for name in ("run_progressive_oracle_refcoco.py", "run_recent_methods_full_val_campaign.py"))
        and "ps -eo" not in line
    ]
    if active:
        raise AuditError(f"Experiment workers remain: {active}")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True
    ).strip()
    if gpu:
        raise AuditError(f"GPU compute process remains: {gpu}")
    free = shutil.disk_usage("/workspace").free
    if free < int(protocol["storage"]["minimum_runtime_free_bytes"]):
        raise AuditError("Final storage reserve is below the registered minimum")
    return {"status": "pass", "campaign_configs": len(configs), "active_workers": [], "gpu_compute": [], "free_bytes": free}


def main() -> int:
    if FINAL_REPORT.exists():
        raise AuditError("Terminal report already exists; refusing to overwrite it")
    protocol = read_json(PROTOCOL)
    registry = read_json(REGISTRY)
    if registry.get("phase_status", {}).get("analysis") != "pass" or registry.get("phase_status", {}).get("dissertation") != "pass":
        raise AuditError("Analysis and dissertation phases must pass before terminal audit")
    if sha256_file(MANIFEST) != protocol["manifest"]["sha256"]:
        raise AuditError("Frozen manifest hash differs")
    requirements = {
        "manifest": {
            "status": "pass",
            "path": str(MANIFEST.relative_to(REPO)),
            "sha256": sha256_file(MANIFEST),
            "rows": ROWS,
        },
        "protected_inputs": validate_protected(protocol),
        "source_epoch": validate_source_epoch(protocol),
        "exact_registered_physical_runs": validate_physical_runs(protocol),
        "campaign_analysis_manifest": verify_hash_manifest(CAMPAIGN, read_json(ANALYSIS_MANIFEST), "campaign analysis"),
        "dissertation_asset_manifest": verify_hash_manifest(DISSERTATION, read_json(ASSET_MANIFEST), "dissertation assets"),
        "cross_artifact_consistency": validate_analysis_consistency(),
        "dissertation_and_pdf": validate_dissertation(),
        "tests_and_git": validate_tests_and_git(),
        "forbidden_scope_and_quiescence": validate_no_forbidden_or_active_work(protocol),
    }
    report = {
        "schema_version": 1,
        "status": "GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED",
        "completed_at_utc": utc_now(),
        "campaign": protocol["campaign"],
        "requirements": requirements,
        "next_action": "human_review_only",
        "forbidden_next_actions_without_new_goal": [
            "timing",
            "testA",
            "testB",
            "transfer",
            "external_baselines",
            "new_budget",
            "final_method_selection",
        ],
    }
    atomic_json(FINAL_REPORT, report)
    registry = read_json(REGISTRY)
    registry["phase_status"]["final_audit"] = "pass"
    registry["status"] = "GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED"
    registry["final_completion_report_sha256"] = sha256_file(FINAL_REPORT)
    atomic_json(REGISTRY, registry)
    state = read_json(STATE)
    state.update(
        {
            "updated_at_utc": utc_now(),
            "phase": "complete",
            "next_action": "human_review_required",
            "active_workers": [],
            "status": "GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED",
        }
    )
    atomic_json(STATE, state)
    print(json.dumps({"status": report["status"], "report": str(FINAL_REPORT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
