#!/usr/bin/env python3
"""Move obsolete K20-era error-analysis derivatives into the campaign archive."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
for value in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.frozen_z_error_analysis_common import (  # noqa: E402
    CAMPAIGN, DISSERTATION, ErrorAnalysisFailure, atomic_json, now, read_json, record_file, require,
)

ARCHIVE = CAMPAIGN / "retired_old_error_analysis"


def tree_records(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): record_file(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def current_record(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {"kind": "file", **record_file(path)}
    if path.is_dir():
        return {"kind": "directory", "files": tree_records(path)}
    return {"kind": "absent"}


def main() -> int:
    completion = CAMPAIGN / "RETIREMENT_COMPLETION.json"
    require(not completion.exists() and not ARCHIVE.exists(), "Retirement already completed")
    baseline = read_json(CAMPAIGN / "RETIREMENT_MANIFEST_V2.json")
    targets = baseline["targets"]
    before: dict[str, Any] = {}
    for name, expected in targets.items():
        path = DISSERTATION / name
        observed = current_record(path)
        require(observed == expected, f"Obsolete target changed before retirement: {name}")
        before[name] = observed
    moved: dict[str, Any] = {}
    ARCHIVE.mkdir(parents=True)
    for name, expected in targets.items():
        if expected["kind"] == "absent":
            moved[name] = {"action": "already_absent"}
            continue
        source = DISSERTATION / name
        target = ARCHIVE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), f"Archive collision: {target}")
        os.replace(source, target)
        archived = current_record(target)
        require(archived == expected, f"Archived bytes differ: {name}")
        require(not source.exists(), f"Retired source remains: {name}")
        moved[name] = {"action": "moved", "archive_path": str(target), "record": archived}
    atomic_json(completion, {
        "schema_version": 1, "status": "complete", "completed_at_utc": now(),
        "method_provenance": "obsolete layer-20 48-token K20-era error/subgroup/qualitative analysis; not frozen Z-M448-0 evidence",
        "backup_root": baseline["backup_root"], "archive_root": str(ARCHIVE),
        "before": before, "actions": moved,
        "active_namespace_sources_absent": all(not (DISSERTATION / name).exists() for name in targets),
    })
    print(json.dumps({"status": "complete", "targets": len(targets), "moved": sum(v["action"] == "moved" for v in moved.values())}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ErrorAnalysisFailure, OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
