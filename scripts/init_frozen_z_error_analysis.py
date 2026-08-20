#!/usr/bin/env python3
"""Freeze the CPU-only error-analysis protocol and snapshot the dissertation baseline."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
for value in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.frozen_z_error_analysis_common import (  # noqa: E402
    BASELINE_PDF_SHA256,
    CAMPAIGN,
    DISSERTATION,
    INPUT_CAMPAIGN,
    INPUT_FINAL_SHA256,
    QUALITY_LEDGER_SHA256,
    REPO,
    SOURCE_ANCESTOR,
    USER_PDF,
    ErrorAnalysisFailure,
    atomic_json,
    now,
    read_json,
    record_file,
    require,
    sha256,
)


SOURCE_FILES = (
    "scripts/frozen_z_error_analysis_common.py",
    "scripts/init_frozen_z_error_analysis.py",
    "scripts/analyse_frozen_z_error_analysis.py",
    "tests/test_frozen_z_error_analysis.py",
)
OBSOLETE_TARGETS = (
    "chapters/generated_final_error.tex",
    "chapters/generated_qualitative_panels.tex",
    "analysis/derived/final_evaluation/error_analysis.csv",
    "images/final_evaluation/box_centre_error_heatmap.pdf",
    "images/final_evaluation/box_centre_error_heatmap.png",
    "images/final_evaluation/qualitative",
    "images/final_evaluation/qualitative_sheet_01.png",
    "images/final_evaluation/qualitative_sheet_02.png",
    "images/final_evaluation/qualitative_sheet_03.png",
    "images/final_evaluation/qualitative_sheet_04.png",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def tree_records(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): record_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def active_dependencies() -> dict[str, dict[str, Any]]:
    fls = DISSERTATION / "Dissertation_Template.fls"
    require(fls.is_file(), "Baseline LaTeX recorder file is missing")
    ignored = {".aux", ".toc", ".out", ".lof", ".lot", ".bbl", ".blg", ".fls", ".log"}
    paths: set[Path] = {DISSERTATION / "Dissertation_Template.tex"}
    for line in fls.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("INPUT "):
            continue
        raw = Path(line[6:])
        path = raw if raw.is_absolute() else DISSERTATION / raw
        try:
            path = path.resolve()
            path.relative_to(DISSERTATION.resolve())
        except ValueError:
            continue
        if path.is_file() and path.suffix.lower() not in ignored:
            paths.add(path)
    return {str(path.relative_to(DISSERTATION)): record_file(path) for path in sorted(paths)}


def obsolete_records() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name in OBSOLETE_TARGETS:
        path = DISSERTATION / name
        if path.is_file():
            records[name] = {"kind": "file", **record_file(path)}
        elif path.is_dir():
            records[name] = {"kind": "directory", "files": tree_records(path)}
        else:
            records[name] = {"kind": "absent"}
    return records


def copy_baseline() -> tuple[Path, dict[str, Any]]:
    backup_parent = Path("/workspace/dissertation_backups")
    backup_parent.mkdir(parents=True, exist_ok=True)
    stamp = now().replace("-", "").replace(":", "")
    final = backup_parent / f"frozen_z_error_analysis_{stamp}"
    require(not final.exists(), f"Backup collision: {final}")
    temporary = Path(tempfile.mkdtemp(prefix=".frozen_z_error_analysis_", dir=backup_parent))
    try:
        shutil.copytree(DISSERTATION, temporary / "Dissertation_Draft", dirs_exist_ok=True)
        shutil.copy2(USER_PDF, temporary / "dissertation_final.pdf")
        os.replace(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    manifest = {
        "dissertation_tree": tree_records(final / "Dissertation_Draft"),
        "user_pdf": record_file(final / "dissertation_final.pdf"),
    }
    return final, manifest


def main() -> int:
    require(not CAMPAIGN.exists(), f"Campaign already exists: {CAMPAIGN}")
    require(sha256(INPUT_CAMPAIGN / "FINAL_COMPLETION_REPORT.json") == INPUT_FINAL_SHA256,
            "Input terminal report changed")
    require(sha256(INPUT_CAMPAIGN / "QUALITY_ARTIFACT_LEDGER.json") == QUALITY_LEDGER_SHA256,
            "Input quality ledger changed")
    require(sha256(DISSERTATION / "Dissertation_Template.pdf") == BASELINE_PDF_SHA256,
            "Dissertation baseline PDF changed")
    require(sha256(USER_PDF) == BASELINE_PDF_SHA256, "User PDF baseline changed")
    status = set(git("status", "--porcelain", "--untracked-files=all").splitlines())
    allowed = {"?? PRO_REASONING_RESEARCH_HANDOFF.md", "?? legacy_goal2.md"}
    require(status <= allowed, f"Unexpected repository state: {sorted(status)}")
    head = git("rev-parse", "HEAD")
    require(subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_ANCESTOR, head], cwd=REPO).returncode == 0,
            "Current source is not descended from the accepted baseline")
    for name in SOURCE_FILES:
        require((REPO / name).is_file(), f"Required analysis source missing: {name}")

    dependencies = active_dependencies()
    retired = obsolete_records()
    backup_root, backup_manifest = copy_baseline()
    CAMPAIGN.mkdir(parents=True)

    protocol = {
        "schema_version": 1,
        "status": "frozen",
        "goal": "paired localization threshold-crossing error analysis and dissertation evidence audit",
        "cpu_only": True,
        "input_campaign": str(INPUT_CAMPAIGN.relative_to(REPO)),
        "input_terminal_report_sha256": INPUT_FINAL_SHA256,
        "quality_artifact_ledger_sha256": QUALITY_LEDGER_SHA256,
        "heldout_status_immutable": "HELDOUT_MIXED_SPLIT_RESULT",
        "methods": ["Full", "Z_M448_0"],
        "splits": {"testA": 5657, "testB": 5095},
        "primary_iou_threshold": 0.5,
        "threshold_grid": [round(0.30 + 0.05 * index, 2) for index in range(9)],
        "bootstrap": {
            "resamples": 10000,
            "unit": "paired_manifest_row_within_split",
            "interval": "percentile_95",
            "quantile_method": "linear",
            "master_seed": 20260820,
            "child_seeds": {"testA": 20260820, "testB": 20260821},
        },
        "expected_transitions": {
            "testA": {"both_correct": 2135, "full_only": 748, "z_only": 526, "both_incorrect": 2248},
            "testB": {"both_correct": 1413, "full_only": 622, "z_only": 505, "both_incorrect": 2555},
        },
        "allowed_outputs": {
            "logical_tables": 1,
            "logical_figures": 1,
            "discussion_words_min": 400,
            "discussion_words_max": 600,
        },
        "forbidden": [
            "model_inference", "gpu_work", "timing", "tuning", "ablation", "new_method",
            "semantic_subgroups", "object_size_subgroups", "expression_length_subgroups",
            "keyword_subgroups", "position_subgroups", "qualitative_case_selection",
        ],
    }
    schema = {
        "schema_version": 1,
        "transition_states": ["both_correct", "full_only", "z_only", "both_incorrect"],
        "delta_direction": "Z_M448_0_minus_Full_for_iou; Full_minus_Z_M448_0_for_accuracy_curve",
        "iou_summaries": [
            "mean", "median", "q1", "q3", "abs_delta_le_0_05", "delta_lt_minus_0_10",
            "delta_lt_minus_0_25", "delta_gt_0_10", "transition_group_medians",
        ],
        "required_analysis_files": [
            "INPUT_AUDIT.json", "ERROR_ANALYSIS.json", "ERROR_ANALYSIS_TRANSITIONS.csv",
            "ERROR_ANALYSIS_THRESHOLD_CURVE.csv", "ERROR_ANALYSIS_SUMMARIES.csv",
            "ANALYSIS_INVENTORY.json", "figures/paired_localization_threshold_crossing.pdf",
        ],
    }
    commands = {
        "schema_version": 1,
        "commands": [
            "/workspace/.followup_venv/bin/python scripts/analyse_frozen_z_error_analysis.py",
            "/workspace/.followup_venv/bin/python scripts/audit_dissertation_result_claims.py",
            "/workspace/.followup_venv/bin/python scripts/build_frozen_z_error_dissertation_assets.py",
            "latexmk -pdf -interaction=nonstopmode -halt-on-error Dissertation_Template.tex",
            "/workspace/.followup_venv/bin/python scripts/audit_frozen_z_error_analysis_goal.py",
        ],
        "gpu_commands": [],
    }
    source_lock = {
        "schema_version": 1,
        "status": "locked",
        "git_head": head,
        "git_status_allowed": sorted(status),
        "sources": {name: record_file(REPO / name) for name in SOURCE_FILES},
        "inputs": {
            "quality_artifact_ledger": record_file(INPUT_CAMPAIGN / "QUALITY_ARTIFACT_LEDGER.json"),
            "input_terminal_report": record_file(INPUT_CAMPAIGN / "FINAL_COMPLETION_REPORT.json"),
            "heldout_results": record_file(INPUT_CAMPAIGN / "analysis/HELDOUT_RESULTS.json"),
            "heldout_decision": record_file(INPUT_CAMPAIGN / "analysis/HELDOUT_DECISION.json"),
        },
    }
    atomic_json(CAMPAIGN / "FROZEN_PROTOCOL.json", protocol)
    atomic_json(CAMPAIGN / "ANALYSIS_SCHEMA.json", schema)
    atomic_json(CAMPAIGN / "COMMANDS.json", commands)
    atomic_json(CAMPAIGN / "SOURCE_LOCK.json", source_lock)
    atomic_json(CAMPAIGN / "RETIREMENT_MANIFEST.json", {
        "schema_version": 1, "status": "baseline_recorded", "recorded_at_utc": now(),
        "targets": retired, "backup_root": str(backup_root),
    })
    atomic_json(CAMPAIGN / "DISSERTATION_CHANGE_RECORD.json", {
        "schema_version": 1, "status": "baseline_captured", "captured_at_utc": now(),
        "backup_root": str(backup_root), "baseline_pdf_sha256": BASELINE_PDF_SHA256,
        "active_dependencies": dependencies, "backup_manifest": backup_manifest,
        "obsolete_targets": retired,
    })
    atomic_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json", {
        "schema_version": 1, "status": "initialized", "phase": "analysis",
        "heldout_status": "HELDOUT_MIXED_SPLIT_RESULT", "cpu_only": True,
    })
    atomic_json(CAMPAIGN / "CONTROLLER_STATE.json", {
        "schema_version": 1, "status": "running", "phase": "analysis",
        "active_workers": [], "updated_at_utc": now(),
    })
    locks = {
        name: record_file(CAMPAIGN / name)
        for name in ("FROZEN_PROTOCOL.json", "ANALYSIS_SCHEMA.json", "COMMANDS.json", "SOURCE_LOCK.json")
    }
    atomic_json(CAMPAIGN / "PRE_ANALYSIS_LOCK.json", {
        "schema_version": 1, "status": "locked", "locked_at_utc": now(),
        "candidate_analysis_outputs_present": False, "locks": locks,
    })
    print(json.dumps({"status": "initialized", "campaign": str(CAMPAIGN), "backup": str(backup_root),
                      "active_dependencies": len(dependencies), "obsolete_targets": len(retired)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ErrorAnalysisFailure, OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
