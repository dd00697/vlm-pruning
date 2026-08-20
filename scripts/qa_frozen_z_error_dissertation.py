#!/usr/bin/env python3
"""Prepare, record and finalize all-page visual QA for the revised dissertation."""

from __future__ import annotations

import argparse
import json
import os
import re
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
    CAMPAIGN, DISSERTATION, USER_PDF, ErrorAnalysisFailure, atomic_json, atomic_text, now, read_json,
    record_file, require, sha256,
)

DERIVED = DISSERTATION / "analysis/derived/frozen_z_error_analysis_report_audit"
SOURCE_PDF = DISSERTATION / "Dissertation_Template.pdf"
LOG = DISSERTATION / "Dissertation_Template.log"
TEXT = CAMPAIGN / "PDF_TEXT.txt"
RENDERED = CAMPAIGN / "rendered_pages"
PREPARED = CAMPAIGN / "PDF_QA_PREPARED.json"
VISUAL = CAMPAIGN / "VISUAL_INSPECTION.json"
FINAL = CAMPAIGN / "PDF_QA.json"


def scan_log(text: str) -> list[str]:
    patterns = {
        "fatal": r"(?:^|\n)! ",
        "undefined_reference": r"LaTeX Warning: Reference .* undefined",
        "undefined_citation": r"LaTeX Warning: Citation .* undefined",
        "multiply_defined_label": r"multiply defined",
        "overfull_box": r"Overfull \\hbox",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, text, re.IGNORECASE)]


def marker_issues(text: str) -> list[str]:
    required = (
        "Paired localisation threshold-crossing analysis", "748 Full-only", "526 Z-M448-0-only",
        "622 Full-only", "505 Z-M448-0-only", "40.20%", "candidate selected from validation",
        "mixed split result",
    )
    forbidden = (
        "40.69%", "spatial and relational keyword groups", "FINAL_CANDIDATE_SELECTED_VALIDATION_ONLY",
        "testA/testB remain reserved", "testA/testB were not run",
    )
    issues = [f"missing:{value}" for value in required if value not in text]
    issues.extend(f"forbidden:{value}" for value in forbidden if value in text)
    return issues


def page_count() -> int:
    output = subprocess.check_output(["pdfinfo", str(SOURCE_PDF)], text=True)
    match = re.search(r"^Pages:\s+(\d+)$", output, re.MULTILINE)
    require(match is not None, "PDF page count is unavailable")
    return int(match.group(1))


def validate_visual_report(report: dict[str, Any], expected_pages: list[int], expected_hashes: dict[str, Any]) -> None:
    require(report.get("status") == "pass", "Visual report status differs")
    pages = report.get("pages")
    require(isinstance(pages, dict), "Visual page notes are missing")
    require(sorted(int(page) for page in pages) == expected_pages, "Visual page coverage differs")
    for page in expected_pages:
        row = pages[str(page)]
        require(row.get("status") == "pass", f"Visual page failed: {page}")
        notes = row.get("notes")
        require(isinstance(notes, str) and len(notes.strip()) >= 25, f"Visual note is too short: {page}")
        require(row.get("render_sha256") == expected_hashes[f"page_{page:03d}.png"]["sha256"],
                f"Visual render hash differs: {page}")


def prepare() -> int:
    require(not PREPARED.exists() and not VISUAL.exists() and not FINAL.exists(), "PDF QA already started")
    require(SOURCE_PDF.is_file() and LOG.is_file(), "Built dissertation PDF/log is missing")
    log_issues = scan_log(LOG.read_text(encoding="utf-8", errors="replace"))
    require(not log_issues, f"LaTeX log issues: {log_issues}")
    text = subprocess.check_output(["pdftotext", str(SOURCE_PDF), "-"], text=True)
    issues = marker_issues(text)
    require(not issues, f"PDF textual QA issues: {issues}")
    atomic_text(TEXT, text)
    pages = page_count()
    require(pages >= 76, f"Unexpectedly short dissertation: {pages}")
    require(not RENDERED.exists(), "Rendered-page directory already exists")
    RENDERED.mkdir(parents=True)
    subprocess.run([
        "pdftoppm", "-png", "-r", "180", str(SOURCE_PDF), str(RENDERED / "page")
    ], check=True)
    raw = sorted(RENDERED.glob("page-*.png"))
    require(len(raw) == pages, f"Rendered page count differs: {len(raw)} != {pages}")
    for index, path in enumerate(raw, start=1):
        target = RENDERED / f"page_{index:03d}.png"
        os.replace(path, target)
    hashes = {path.name: record_file(path) for path in sorted(RENDERED.glob("page_*.png"))}
    placeholders = [value for value in ("[Student Name]", "[Submission month and year]", "[Supervisor name(s)]", "[Year]") if value in text]
    value = {
        "schema_version": 1, "status": "prepared", "prepared_at_utc": now(),
        "source_pdf": record_file(SOURCE_PDF), "latex_log": record_file(LOG), "pdf_text": record_file(TEXT),
        "page_count": pages, "pages_to_render": list(range(1, pages + 1)), "render_dpi": 180,
        "rendered_pages": hashes, "log_issues": [], "text_issues": [],
        "human_metadata_placeholders": placeholders,
        "human_review_required": bool(placeholders),
    }
    atomic_json(PREPARED, value)
    print(json.dumps({"status": "prepared", "pages": pages, "human_metadata_placeholders": placeholders}, indent=2))
    return 0


def record_visual(report_path: Path) -> int:
    require(PREPARED.is_file() and not VISUAL.exists() and not FINAL.exists(), "Visual QA phase differs")
    prepared = read_json(PREPARED)
    report = read_json(report_path)
    validate_visual_report(report, prepared["pages_to_render"], prepared["rendered_pages"])
    value = {**report, "schema_version": 1, "recorded_at_utc": now(), "prepared_sha256": sha256(PREPARED)}
    atomic_json(VISUAL, value)
    print(json.dumps({"status": "recorded", "pages": len(report["pages"])}, indent=2))
    return 0


def finalize() -> int:
    require(PREPARED.is_file() and VISUAL.is_file() and not FINAL.exists(), "Final QA phase differs")
    prepared = read_json(PREPARED)
    require(record_file(SOURCE_PDF) == prepared["source_pdf"], "Source PDF changed after prepare")
    require(record_file(LOG) == prepared["latex_log"], "LaTeX log changed after prepare")
    for name, expected in prepared["rendered_pages"].items():
        require(record_file(RENDERED / name) == expected, f"Rendered page changed: {name}")
    visual = read_json(VISUAL)
    validate_visual_report(visual, prepared["pages_to_render"], prepared["rendered_pages"])
    value = {
        "schema_version": 1, "status": "pass", "finalized_at_utc": now(),
        "source_pdf": prepared["source_pdf"], "page_count": prepared["page_count"],
        "pages_inspected": prepared["pages_to_render"], "prepared_sha256": sha256(PREPARED),
        "visual_sha256": sha256(VISUAL), "human_review_required": prepared["human_review_required"],
        "human_metadata_placeholders": prepared["human_metadata_placeholders"],
    }
    atomic_json(FINAL, value)
    DERIVED.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FINAL, DERIVED / "PDF_QA.json")
    print(json.dumps({"status": "pass", "pages": prepared["page_count"]}, indent=2))
    return 0


def publish() -> int:
    require(FINAL.is_file(), "PDF QA is not final")
    qa = read_json(FINAL)
    require(qa.get("status") == "pass" and record_file(SOURCE_PDF) == qa["source_pdf"],
            "Validated source PDF differs")
    with tempfile.NamedTemporaryFile(dir=USER_PDF.parent, prefix=".dissertation_final.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(SOURCE_PDF.read_bytes())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, USER_PDF)
    require(record_file(USER_PDF) == record_file(SOURCE_PDF), "Published PDF differs")
    record = {
        "schema_version": 1, "status": "published", "published_at_utc": now(),
        "source_pdf": record_file(SOURCE_PDF), "user_pdf": record_file(USER_PDF),
        "pdf_qa_sha256": sha256(FINAL), "atomic_publication": True,
    }
    atomic_json(CAMPAIGN / "PDF_PUBLICATION_RECORD.json", record)
    atomic_json(DERIVED / "PDF_PUBLICATION_RECORD.json", record)
    print(json.dumps({"status": "published", "sha256": record["source_pdf"]["sha256"]}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    record = sub.add_parser("record-visual")
    record.add_argument("--report", type=Path, required=True)
    sub.add_parser("finalize")
    sub.add_parser("publish")
    args = parser.parse_args()
    return {"prepare": prepare, "record-visual": lambda: record_visual(args.report),
            "finalize": finalize, "publish": publish}[args.command]()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ErrorAnalysisFailure, OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
