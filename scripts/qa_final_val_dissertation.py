#!/usr/bin/env python
"""Prepare, record, and finalize structural/text/visual QA for the final dissertation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DISSERTATION = Path("/workspace/Dissertation_Draft")
PDF = DISSERTATION / "Dissertation_Template.pdf"
LOG = DISSERTATION / "Dissertation_Template.log"
DERIVED = DISSERTATION / "analysis/derived/final_val_efficiency_baselines"
PREPARED = DERIVED / "PDF_QA_PREPARED.json"
VISUAL = DERIVED / "VISUAL_INSPECTION.json"
FINAL = DERIVED / "PDF_QA.json"
RENDERED = DERIVED / "rendered_pages"
MARKERS = (
    "Final RefCOCO-validation efficiency",
    "Complete RefCOCO-validation quality and visual work",
    "Source-native Nüwa quality",
    "Five-repetition canonical-stack timing",
    "Canonical throughput, analytical KV storage",
    "Source-native repeated timing",
    "Trace-driven modeled FLOPs",
    "Selected paired 95% bootstrap intervals",
    "Preregistered internal candidate rule",
    "Complete validation Acc.@0.5 and mIoU",
    "Canonical-stack validation quality against",
    "Canonical timing decomposition",
)


class DissertationQAError(RuntimeError):
    """Raised when the rebuilt dissertation fails an objective QA check."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise DissertationQAError(f"Missing QA artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def page_count() -> int:
    output = subprocess.check_output(["pdfinfo", str(PDF)], text=True)
    match = re.search(r"^Pages:\s+(\d+)$", output, flags=re.MULTILINE)
    if not match:
        raise DissertationQAError("Could not determine final PDF page count")
    return int(match.group(1))


def prepare() -> dict[str, Any]:
    if not PDF.is_file() or not LOG.is_file():
        raise DissertationQAError("Build the dissertation before preparing PDF QA")
    pages = page_count()
    page_text = {
        page: subprocess.check_output(
            ["pdftotext", "-f", str(page), "-l", str(page), str(PDF), "-"], text=True
        )
        for page in range(1, pages + 1)
    }
    complete_text = "\n\f\n".join(page_text.values())
    marker_pages = {
        marker: [page for page, text in page_text.items() if marker in text]
        for marker in MARKERS
    }
    missing = [marker for marker, locations in marker_pages.items() if not locations]
    if missing:
        raise DissertationQAError(f"Required final PDF content is missing: {missing}")
    forbidden_active_claims = (
        "On untouched RefCOCO testA",
        "Held-out RefCOCO testA and testB",
        "Transfer to POPE and TextVQA",
        "Final Held-Out Evaluation",
    )
    present_forbidden = [text for text in forbidden_active_claims if text in complete_text]
    if present_forbidden:
        raise DissertationQAError(f"Obsolete frozen-result narrative remains active: {present_forbidden}")
    log = LOG.read_text(encoding="utf-8", errors="replace")
    broken = {
        "undefined_references": bool(re.search(r"There were undefined references|Reference [`'].+undefined", log, re.I)),
        "undefined_citations": bool(re.search(r"There were undefined citations|Citation [`'].+undefined", log, re.I)),
        "multiply_defined_labels": bool(re.search(r"multiply defined", log, re.I)),
        "fatal_error": bool(re.search(r"Fatal error occurred|Emergency stop", log, re.I)),
    }
    if any(broken.values()):
        raise DissertationQAError(f"LaTeX build has broken references/citations/labels: {broken}")
    pages_to_render = sorted({page for locations in marker_pages.values() for page in locations})
    RENDERED.mkdir(parents=True, exist_ok=True)
    rendered = {}
    for page in pages_to_render:
        prefix = RENDERED / f"page_{page:03d}"
        subprocess.run(
            [
                "pdftoppm", "-f", str(page), "-l", str(page), "-r", "180", "-png",
                "-singlefile", str(PDF), str(prefix),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        image = prefix.with_suffix(".png")
        rendered[str(page)] = {
            "path": str(image),
            "sha256": sha256_file(image),
        }
    value = {
        "status": "prepared_for_visual_inspection",
        "prepared_at_utc": utc_now(),
        "pdf_path": str(PDF),
        "pdf_sha256": sha256_file(PDF),
        "pages": pages,
        "marker_pages": marker_pages,
        "pages_to_render": pages_to_render,
        "rendered_pages": rendered,
        "latex_log_sha256": sha256_file(LOG),
        "broken_build_markers": broken,
        "overfull_box_count": len(re.findall(r"Overfull \\[hv]box", log)),
        "obsolete_active_claims": [],
    }
    atomic_json(PREPARED, value)
    return value


def record_visual(pages: list[int], notes: str) -> dict[str, Any]:
    prepared = read_json(PREPARED)
    expected = sorted(int(page) for page in prepared["pages_to_render"])
    if sorted(pages) != expected or len(pages) != len(set(pages)):
        raise DissertationQAError(f"Visual page record must equal {expected}")
    if len(notes.strip()) < 20:
        raise DissertationQAError("Visual inspection requires a substantive note")
    for page, record in prepared["rendered_pages"].items():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise DissertationQAError(f"Rendered page changed before visual record: {page}")
    value = {
        "status": "pass",
        "recorded_at_utc": utc_now(),
        "pages_to_render": expected,
        "inspected_pages": sorted(pages),
        "inspection_note": notes.strip(),
        "rendered_page_sha256": {
            page: record["sha256"] for page, record in prepared["rendered_pages"].items()
        },
    }
    atomic_json(VISUAL, value)
    return value


def finalize() -> dict[str, Any]:
    prepared = read_json(PREPARED)
    visual = read_json(VISUAL)
    if prepared.get("status") != "prepared_for_visual_inspection":
        raise DissertationQAError("Prepared QA is incomplete")
    if sha256_file(PDF) != prepared["pdf_sha256"]:
        raise DissertationQAError("PDF changed after the visual pages were rendered")
    if visual.get("status") != "pass":
        raise DissertationQAError("Visual inspection did not pass")
    if sorted(visual.get("inspected_pages", [])) != sorted(prepared["pages_to_render"]):
        raise DissertationQAError("Visual inspection does not cover every required page")
    for page, record in prepared["rendered_pages"].items():
        image = Path(record["path"])
        if not image.is_file() or sha256_file(image) != record["sha256"]:
            raise DissertationQAError(f"Rendered page changed: {page}")
    value = {
        "status": "pass",
        "completed_at_utc": utc_now(),
        "pdf_path": prepared["pdf_path"],
        "pdf_sha256": prepared["pdf_sha256"],
        "pages": prepared["pages"],
        "marker_pages": prepared["marker_pages"],
        "broken_build_markers": prepared["broken_build_markers"],
        "overfull_box_count": prepared["overfull_box_count"],
        "rendered_pages": prepared["rendered_pages"],
        "visual_inspection": visual,
        "prepared_qa_sha256": sha256_file(PREPARED),
        "visual_inspection_sha256": sha256_file(VISUAL),
    }
    atomic_json(FINAL, value)
    return value


def parse_pages(value: str) -> list[int]:
    try:
        return [int(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Pages must be comma-separated integers") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("prepare")
    record = subparsers.add_parser("record-visual")
    record.add_argument("--pages", type=parse_pages, required=True)
    record.add_argument("--notes", required=True)
    subparsers.add_parser("finalize")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare()
    elif args.action == "record-visual":
        result = record_visual(args.pages, args.notes)
    else:
        result = finalize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DissertationQAError, OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
