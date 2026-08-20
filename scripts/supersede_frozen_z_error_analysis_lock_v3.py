#!/usr/bin/env python3
"""Freeze the one-line V3 inventory-guard correction after preserving the failed V2 attempt."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
for value in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.frozen_z_error_analysis_common import (  # noqa: E402
    CAMPAIGN, REPO, ErrorAnalysisFailure, atomic_json, now, read_json, record_file, require,
)

SOURCE_FILES = (
    "scripts/frozen_z_error_analysis_common.py",
    "scripts/init_frozen_z_error_analysis.py",
    "scripts/supersede_frozen_z_error_analysis_lock.py",
    "scripts/supersede_frozen_z_error_analysis_lock_v3.py",
    "scripts/analyse_frozen_z_error_analysis.py",
    "tests/test_frozen_z_error_analysis.py",
)
ACTIVE_OUTPUTS = (
    "analysis", "RESEARCH_REPORT.md",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def tree_records(root: Path) -> dict[str, dict[str, object]]:
    return {
        str(path.relative_to(root)): record_file(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def main() -> int:
    require(not (CAMPAIGN / "PRE_ANALYSIS_LOCK_V3.json").exists(), "V3 lock already exists")
    require(not any((CAMPAIGN / name).exists() for name in ACTIVE_OUTPUTS), "Active analysis outputs still exist")
    interrupted = CAMPAIGN / "analysis_interrupted_v2_inventory_guard"
    interrupted_report = CAMPAIGN / "RESEARCH_REPORT.interrupted_v2.md"
    require(interrupted.is_dir() and interrupted_report.is_file(), "Preserved V2 attempt is missing")
    require(git("status", "--porcelain", "--untracked-files=all").splitlines() == [
        "?? PRO_REASONING_RESEARCH_HANDOFF.md", "?? legacy_goal2.md"
    ], "Repository must be at a clean committed source epoch")
    v2_source = read_json(CAMPAIGN / "SOURCE_LOCK_V2.json")
    source_lock = {
        **v2_source,
        "schema_version": 3,
        "git_head": git("rev-parse", "HEAD"),
        "sources": {name: record_file(REPO / name) for name in SOURCE_FILES},
        "v2_input_binding_sha256": record_file(CAMPAIGN / "SOURCE_LOCK_V2.json")["sha256"],
    }
    schema = read_json(CAMPAIGN / "ANALYSIS_SCHEMA_V2.json")
    schema.update({
        "schema_version": 3,
        "inventory_guard_semantics": "Guard compares the seven physical analysis products plus the pending inventory filename before the inventory is atomically written.",
    })
    atomic_json(CAMPAIGN / "SOURCE_LOCK_V3.json", source_lock)
    atomic_json(CAMPAIGN / "ANALYSIS_SCHEMA_V3.json", schema)
    supersession = {
        "schema_version": 1, "status": "superseded_before_accepted_analysis", "superseded_at_utc": now(),
        "reason": "V2 analysis finished computation but its pre-write inventory guard omitted the pending inventory filename.",
        "scientific_mismatch": False, "active_candidate_outputs_present": False,
        "v2_lock": record_file(CAMPAIGN / "PRE_ANALYSIS_LOCK_V2.json"),
        "v2_preserved_outputs": tree_records(interrupted),
        "v2_preserved_report": record_file(interrupted_report),
        "source_v3": record_file(CAMPAIGN / "SOURCE_LOCK_V3.json"),
        "schema_v3": record_file(CAMPAIGN / "ANALYSIS_SCHEMA_V3.json"),
    }
    atomic_json(CAMPAIGN / "LOCK_SUPERSESSION_V3.json", supersession)
    lock_names = (
        "FROZEN_PROTOCOL.json", "SOURCE_LOCK_V3.json", "ANALYSIS_SCHEMA_V3.json", "COMMANDS_V2.json",
        "RETIREMENT_MANIFEST_V2.json", "LOCK_SUPERSESSION.json", "LOCK_SUPERSESSION_V3.json",
    )
    atomic_json(CAMPAIGN / "PRE_ANALYSIS_LOCK_V3.json", {
        "schema_version": 3, "status": "locked", "locked_at_utc": now(),
        "candidate_analysis_outputs_present": False,
        "locks": {name: record_file(CAMPAIGN / name) for name in lock_names},
    })
    print(json.dumps({"status": "locked_v3", "preserved_v2_files": len(tree_records(interrupted))}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ErrorAnalysisFailure, OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
