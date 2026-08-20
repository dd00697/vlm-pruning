#!/usr/bin/env python3
"""Finalize the campaign-specific dissertation change record after validated publication."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
for value in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.frozen_z_error_analysis_common import (  # noqa: E402
    BASELINE_PDF_SHA256, CAMPAIGN, DISSERTATION, USER_PDF, ErrorAnalysisFailure, atomic_json, now,
    read_json, record_file, require,
)


def tree_records(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): record_file(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def main() -> int:
    path = CAMPAIGN / "DISSERTATION_CHANGE_RECORD.json"
    value = read_json(path)
    require(value.get("status") == "baseline_captured", "Change record phase differs")
    backup = Path(value["backup_root"])
    require(backup.is_dir(), "Dissertation backup is missing")
    baseline = value["backup_manifest"]
    require(tree_records(backup / "Dissertation_Draft") == baseline["dissertation_tree"],
            "Dissertation backup tree changed")
    require(record_file(backup / "dissertation_final.pdf") == baseline["user_pdf"],
            "Backed-up user PDF changed")
    require(baseline["user_pdf"]["sha256"] == BASELINE_PDF_SHA256, "Baseline PDF hash differs")
    qa = read_json(CAMPAIGN / "PDF_QA.json")
    publication = read_json(CAMPAIGN / "PDF_PUBLICATION_RECORD.json")
    require(qa.get("status") == "pass" and publication.get("status") == "published", "PDF QA/publication differs")
    source_pdf = DISSERTATION / "Dissertation_Template.pdf"
    require(record_file(source_pdf) == record_file(USER_PDF) == publication["source_pdf"],
            "Published source/user PDF identity differs")
    current = tree_records(DISSERTATION)
    before = baseline["dissertation_tree"]
    changed = {name: current[name] for name in sorted(set(current) & set(before)) if current[name] != before[name]}
    new = {name: current[name] for name in sorted(set(current) - set(before))}
    removed = {name: before[name] for name in sorted(set(before) - set(current))}
    value.update({
        "status": "complete", "finalized_at_utc": now(), "final_pdf": record_file(source_pdf),
        "published_pdf": record_file(USER_PDF), "pdf_qa": record_file(CAMPAIGN / "PDF_QA.json"),
        "publication_record": record_file(CAMPAIGN / "PDF_PUBLICATION_RECORD.json"),
        "changed_files": changed, "new_files": new, "removed_files": removed,
        "human_review_required": bool(qa.get("human_review_required")),
        "human_metadata_placeholders": qa.get("human_metadata_placeholders", []),
    })
    atomic_json(path, value)
    print(json.dumps({"status": "complete", "changed": len(changed), "new": len(new), "removed": len(removed)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ErrorAnalysisFailure, OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
