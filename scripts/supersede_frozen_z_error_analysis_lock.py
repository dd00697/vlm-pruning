#!/usr/bin/env python3
"""Supersede the deficient initial analysis lock before any derived result exists."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
for value in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.frozen_z_error_analysis_common import (  # noqa: E402
    CAMPAIGN, DISSERTATION, INPUT_CAMPAIGN, REPO, ErrorAnalysisFailure, atomic_json, now, read_json,
    record_file, require,
)

SOURCE_FILES = (
    "scripts/frozen_z_error_analysis_common.py",
    "scripts/init_frozen_z_error_analysis.py",
    "scripts/supersede_frozen_z_error_analysis_lock.py",
    "scripts/analyse_frozen_z_error_analysis.py",
    "tests/test_frozen_z_error_analysis.py",
)
MANIFESTS = {
    "testA": "data/manifests/final_evaluation/refcoco_testA_unique.jsonl",
    "testB": "data/manifests/final_evaluation/refcoco_testB_unique.jsonl",
}
OBSOLETE_TARGETS = (
    "chapters/generated_final_error.tex",
    "chapters/generated_qualitative_panels.tex",
    "appendices/generated_qualitative_panels.tex",
    "appendices/generated_supplementary_tables.tex",
    "analysis/derived/final_evaluation/error_analysis.csv",
    "analysis/derived/final_evaluation/qualitative_selection_ledger.json",
    "analysis/derived/final_evaluation/claim_to_artifact_ledger.json",
    "images/final_evaluation/box_centre_error_heatmap.pdf",
    "images/final_evaluation/box_centre_error_heatmap.png",
    "images/final_evaluation/qualitative",
    "images/final_evaluation/qualitative_sheet_01.png",
    "images/final_evaluation/qualitative_sheet_02.png",
    "images/final_evaluation/qualitative_sheet_03.png",
    "images/final_evaluation/qualitative_sheet_04.png",
)
DERIVED_OUTPUTS = (
    "analysis/INPUT_AUDIT.json", "analysis/ERROR_ANALYSIS.json",
    "analysis/ERROR_ANALYSIS_TRANSITIONS.csv", "analysis/ERROR_ANALYSIS_THRESHOLD_CURVE.csv",
    "analysis/ERROR_ANALYSIS_SUMMARIES.csv", "analysis/ANALYSIS_INVENTORY.json",
    "analysis/figures/paired_localization_threshold_crossing.pdf", "RESEARCH_REPORT.md",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def path_record(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {"kind": "file", **record_file(path)}
    if path.is_dir():
        return {"kind": "directory", "files": {
            str(item.relative_to(path)): record_file(item) for item in sorted(path.rglob("*")) if item.is_file()
        }}
    return {"kind": "absent"}


def jsonl_identity(path: Path) -> dict[str, Any]:
    count = 0
    ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            require(line.endswith("\n"), f"JSONL lacks final newline: {path}")
            row = json.loads(line)
            require(isinstance(row, dict) and isinstance(row.get("sample_id"), str),
                    f"JSONL row identity differs: {path}/{count}")
            ids.append(row["sample_id"])
            count += 1
    require(count > 0 and len(set(ids)) == count, f"JSONL IDs differ: {path}")
    return {"rows": count, "unique_ids": len(set(ids)), "first_id": ids[0], "last_id": ids[-1]}


def main() -> int:
    require(CAMPAIGN.is_dir(), "Campaign is not initialized")
    require(not (CAMPAIGN / "PRE_ANALYSIS_LOCK_V2.json").exists(), "V2 lock already exists")
    present = [name for name in DERIVED_OUTPUTS if (CAMPAIGN / name).exists()]
    require(not present, f"Derived outputs exist before lock supersession: {present}")
    require(git("status", "--porcelain", "--untracked-files=all").splitlines() == [
        "?? PRO_REASONING_RESEARCH_HANDOFF.md", "?? legacy_goal2.md"
    ], "Repository must be at a clean committed source epoch")
    ledger = read_json(INPUT_CAMPAIGN / "QUALITY_ARTIFACT_LEDGER.json")
    cells: dict[str, Any] = {}
    for cell_id, cell in sorted(ledger["cells"].items()):
        run_dir = REPO / cell["path"]
        prediction_name = "predictions.scored.jsonl" if cell_id.startswith("Z_M448_0") else "predictions.jsonl"
        prediction_path = run_dir / prediction_name
        cells[cell_id] = {
            "path": cell["path"],
            "files": {name: record_file(run_dir / name) for name in sorted(cell["files"])},
            "accepted_prediction": {
                "path": str(prediction_path.relative_to(REPO)), **record_file(prediction_path),
                **jsonl_identity(prediction_path),
            },
        }
    source_lock = {
        "schema_version": 2, "status": "locked", "git_head": git("rev-parse", "HEAD"),
        "sources": {name: record_file(REPO / name) for name in SOURCE_FILES},
        "inputs": {
            "quality_artifact_ledger": record_file(INPUT_CAMPAIGN / "QUALITY_ARTIFACT_LEDGER.json"),
            "input_terminal_report": record_file(INPUT_CAMPAIGN / "FINAL_COMPLETION_REPORT.json"),
            "heldout_results": record_file(INPUT_CAMPAIGN / "analysis/HELDOUT_RESULTS.json"),
            "heldout_decision": record_file(INPUT_CAMPAIGN / "analysis/HELDOUT_DECISION.json"),
            "manifests": {
                split: {"path": path, **record_file(REPO / path), **jsonl_identity(REPO / path)}
                for split, path in MANIFESTS.items()
            },
            "accepted_cells": cells,
        },
    }
    schema = {
        "schema_version": 2, "status": "frozen", "row_order": ["testA", "testB"],
        "threshold_inclusivity": "correct iff IoU >= threshold",
        "threshold_order": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        "transition_order": ["both_correct", "full_only", "z_only", "both_incorrect"],
        "csv": {
            "ERROR_ANALYSIS_TRANSITIONS.csv": {
                "columns": ["split", "state", "count", "rate", "lower_95", "upper_95"],
                "types": ["string", "string", "integer", "fraction", "fraction", "fraction"],
                "units": ["split", "state", "examples", "share", "share", "share"],
            },
            "ERROR_ANALYSIS_THRESHOLD_CURVE.csv": {
                "columns": ["split", "iou_threshold", "point", "lower_95", "upper_95"],
                "types": ["string", "decimal_2", "fraction", "fraction", "fraction"],
                "units": ["split", "IoU", "Full_minus_Z_accuracy", "same", "same"],
            },
            "ERROR_ANALYSIS_SUMMARIES.csv": {
                "columns": ["split", "metric", "value"], "types": ["string", "string", "number_or_null"],
                "units": ["split", "metric", "IoU_or_fraction_as_named"],
            },
        },
        "json_directions": {"accuracy": "Full_minus_Z_M448_0", "iou": "Z_M448_0_minus_Full"},
        "bootstrap": {
            "unit": "paired_manifest_row_within_split", "split_pooling": False, "resamples": 10000,
            "child_seeds": {"testA": 20260820, "testB": 20260821},
            "interval": "percentile_95", "quantile_method": "numpy_linear",
        },
        "summary_boundaries": {
            "abs_delta_le_0_05": "inclusive <= 0.05",
            "delta_lt_minus_0_10": "strict < -0.10",
            "delta_lt_minus_0_25": "strict < -0.25",
            "delta_gt_0_10": "strict > 0.10",
            "quartiles": "numpy linear quantiles at 0.25 and 0.75",
            "empty_transition_group_median": "JSON null",
        },
        "exact_analysis_outputs": list(DERIVED_OUTPUTS[:-1]),
        "forbidden_output_tokens": [
            "subgroup", "qualitative", "example", "category", "object_size", "expression_length", "keyword",
        ],
    }
    commands = {
        "schema_version": 2, "status": "frozen", "gpu_commands": [],
        "commands": [
            "/workspace/.followup_venv/bin/python -m unittest tests.test_frozen_z_error_analysis -v",
            "/workspace/.followup_venv/bin/python scripts/analyse_frozen_z_error_analysis.py",
            "/workspace/.followup_venv/bin/python scripts/audit_dissertation_result_claims.py",
            "/workspace/.followup_venv/bin/python scripts/build_frozen_z_error_dissertation_assets.py",
            "latexmk -pdf -interaction=nonstopmode -halt-on-error Dissertation_Template.tex",
            "/workspace/.followup_venv/bin/python -m unittest discover -s tests -v",
            "/workspace/.followup_venv/bin/python scripts/audit_frozen_z_error_analysis_goal.py",
        ],
    }
    retirement = {
        "schema_version": 2, "status": "baseline_recorded", "recorded_at_utc": now(),
        "backup_root": read_json(CAMPAIGN / "DISSERTATION_CHANGE_RECORD.json")["backup_root"],
        "targets": {name: path_record(DISSERTATION / name) for name in OBSOLETE_TARGETS},
    }
    atomic_json(CAMPAIGN / "SOURCE_LOCK_V2.json", source_lock)
    atomic_json(CAMPAIGN / "ANALYSIS_SCHEMA_V2.json", schema)
    atomic_json(CAMPAIGN / "COMMANDS_V2.json", commands)
    atomic_json(CAMPAIGN / "RETIREMENT_MANIFEST_V2.json", retirement)
    old_names = ("SOURCE_LOCK.json", "ANALYSIS_SCHEMA.json", "COMMANDS.json", "PRE_ANALYSIS_LOCK.json",
                 "RETIREMENT_MANIFEST.json")
    replacement_names = ("SOURCE_LOCK_V2.json", "ANALYSIS_SCHEMA_V2.json", "COMMANDS_V2.json",
                         "RETIREMENT_MANIFEST_V2.json")
    supersession = {
        "schema_version": 1, "status": "superseded_before_analysis", "superseded_at_utc": now(),
        "reason": "Initial lock omitted complete accepted-input binding, exact output schema, test commands, and three retired derivatives.",
        "interrupted_attempts": 2, "candidate_analysis_outputs_present": False,
        "superseded": {name: record_file(CAMPAIGN / name) for name in old_names},
        "replacements": {name: record_file(CAMPAIGN / name) for name in replacement_names},
    }
    atomic_json(CAMPAIGN / "LOCK_SUPERSESSION.json", supersession)
    lock_names = ("FROZEN_PROTOCOL.json", *replacement_names, "LOCK_SUPERSESSION.json")
    atomic_json(CAMPAIGN / "PRE_ANALYSIS_LOCK_V2.json", {
        "schema_version": 2, "status": "locked", "locked_at_utc": now(),
        "candidate_analysis_outputs_present": False,
        "locks": {name: record_file(CAMPAIGN / name) for name in lock_names},
    })
    print(json.dumps({"status": "locked_v2", "inputs": len(cells), "retirement_targets": len(OBSOLETE_TARGETS)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ErrorAnalysisFailure, OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
