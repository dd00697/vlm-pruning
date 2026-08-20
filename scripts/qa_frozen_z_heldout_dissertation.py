#!/usr/bin/env python3
"""Prepare, record all-page visual inspection, and finalize held-out dissertation QA."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
for value in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from frozen_z_heldout_common import CAMPAIGN, HeldoutError, atomic_json, now, read_json, sha256


DRAFT = Path("/workspace/Dissertation_Draft")
PDF = DRAFT / "Dissertation_Template.pdf"
USER_PDF = Path("/workspace/dissertation_final.pdf")
LOG = DRAFT / "Dissertation_Template.log"
PREPARED = CAMPAIGN / "PDF_QA_PREPARED.json"
VISUAL = CAMPAIGN / "VISUAL_INSPECTION.json"
FINAL = CAMPAIGN / "PDF_QA.json"
DERIVED = DRAFT / "analysis/derived/frozen_z_m448_0_refcoco_testab"
RENDERED = CAMPAIGN / "pdf_qa/rendered_pages"
FORBIDDEN = (
    "testA/testB remain reserved", "testA/testB or transfer result is part of this conclusion",
    "testA/testB and transfer evaluation were not run", "later one-shot testA/testB",
    "A later one-shot held-out study remains future work", "after human review, execute one frozen RefCOCO testA/testB",
)
STATUS_TEXT = {
    "HELDOUT_CONFIRMED_BOTH_SPLITS": "confirmed on both frozen splits",
    "HELDOUT_MIXED_SPLIT_RESULT": "mixed split result",
    "HELDOUT_NOT_CONFIRMED": "not confirmed under",
}


def pages() -> int:
    output = subprocess.check_output(["pdfinfo", str(PDF)], text=True)
    match = re.search(r"^Pages:\s+(\d+)$", output, re.MULTILINE)
    if not match:
        raise HeldoutError("Could not determine PDF page count")
    return int(match.group(1))


def prepare() -> dict[str, Any]:
    if not PDF.is_file() or not LOG.is_file():
        raise HeldoutError("Build the dissertation before QA")
    change = read_json(CAMPAIGN / "DISSERTATION_CHANGE_RECORD.json")
    if change.get("status") != "baseline_captured":
        raise HeldoutError("Dissertation baseline was not captured before edits")
    decision = read_json(CAMPAIGN / "analysis/HELDOUT_DECISION.json")
    status = str(decision["status"])
    count = pages()
    page_text = {
        page: subprocess.check_output(["pdftotext", "-f", str(page), "-l", str(page), str(PDF), "-"], text=True)
        for page in range(1, count + 1)
    }
    text = "\n\f\n".join(page_text.values())
    normalized_text = " ".join(text.split())
    markers = (
        "One-shot held-out confirmation", "One-shot held-out quality and visual-token work",
        "Paired held-out contrasts and preregistered split-wise gates", STATUS_TEXT[status],
        "10,000 deterministic paired", "Z-M448-0 minus Full",
    )
    missing = [marker for marker in markers if " ".join(marker.split()) not in normalized_text]
    if missing:
        raise HeldoutError(f"Required held-out PDF content is missing: {missing}")
    forbidden = [phrase for phrase in FORBIDDEN if phrase.lower() in text.lower()]
    if forbidden:
        raise HeldoutError(f"Stale active held-out claims remain: {forbidden}")
    log = LOG.read_text(encoding="utf-8", errors="replace")
    broken = {
        "undefined_references": bool(re.search(r"There were undefined references|Reference [`'].+undefined", log, re.I)),
        "undefined_citations": bool(re.search(r"There were undefined citations|Citation [`'].+undefined", log, re.I)),
        "multiply_defined_labels": bool(re.search(r"multiply defined", log, re.I)),
        "fatal_error": bool(re.search(r"Fatal error occurred|Emergency stop", log, re.I)),
    }
    if any(broken.values()):
        raise HeldoutError(f"LaTeX build has broken markers: {broken}")
    RENDERED.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, Any] = {}
    for page in range(1, count + 1):
        prefix = RENDERED / f"page_{page:03d}"
        subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", "180", "-png", "-singlefile", str(PDF), str(prefix)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        image = prefix.with_suffix(".png")
        rendered[str(page)] = {"path": str(image), "sha256": sha256(image)}
    value = {
        "schema_version": 1, "status": "prepared_for_all_page_visual_inspection", "prepared_at_utc": now(),
        "heldout_status": status, "pdf_sha256": sha256(PDF), "pages": count,
        "pages_to_render": list(range(1, count + 1)), "rendered_pages": rendered,
        "latex_log_sha256": sha256(LOG), "broken_build_markers": broken,
        "overfull_box_count": len(re.findall(r"Overfull \\[hv]box", log)), "forbidden_active_claims": [],
    }
    atomic_json(PREPARED, value)
    return value


def record_visual(records_path: Path) -> dict[str, Any]:
    prepared = read_json(PREPARED)
    records = read_json(records_path)
    expected = [int(page) for page in prepared["pages_to_render"]]
    raw_pages = records.get("pages")
    if not isinstance(raw_pages, dict) or sorted(int(page) for page in raw_pages) != expected:
        raise HeldoutError("Visual records must cover every prepared PDF page exactly")
    normalized: dict[str, Any] = {}
    for page in expected:
        record = raw_pages.get(str(page))
        if not isinstance(record, dict) or record.get("status") != "pass" or len(str(record.get("note") or "").strip()) < 20:
            raise HeldoutError(f"Page {page} lacks a substantive PASS inspection")
        image = Path(prepared["rendered_pages"][str(page)]["path"])
        if sha256(image) != prepared["rendered_pages"][str(page)]["sha256"]:
            raise HeldoutError(f"Rendered page {page} changed")
        normalized[str(page)] = {"status": "pass", "note": str(record["note"]).strip(), "rendered_sha256": sha256(image)}
    value = {"schema_version": 1, "status": "pass", "recorded_at_utc": now(), "inspected_pages": expected, "pages": normalized}
    atomic_json(VISUAL, value)
    return value


def finalize() -> dict[str, Any]:
    prepared, visual = read_json(PREPARED), read_json(VISUAL)
    if visual.get("status") != "pass" or visual.get("inspected_pages") != prepared.get("pages_to_render"):
        raise HeldoutError("All-page visual inspection has not passed")
    if sha256(PDF) != prepared["pdf_sha256"]:
        raise HeldoutError("Draft PDF changed after rendering")
    if not USER_PDF.is_file() or sha256(USER_PDF) != prepared["pdf_sha256"]:
        raise HeldoutError("Published user PDF is not byte-identical to the visually inspected draft")
    for page, record in prepared["rendered_pages"].items():
        if sha256(Path(record["path"])) != record["sha256"]:
            raise HeldoutError(f"Rendered page changed after inspection: {page}")
    value = {
        "schema_version": 1, "status": "pass", "completed_at_utc": now(),
        "heldout_status": prepared["heldout_status"], "pdf_sha256": prepared["pdf_sha256"],
        "published_pdf_sha256": sha256(USER_PDF), "pages": prepared["pages"],
        "all_pages_visually_inspected": True, "prepared_sha256": sha256(PREPARED),
        "visual_inspection_sha256": sha256(VISUAL), "latex_log_sha256": prepared["latex_log_sha256"],
    }
    atomic_json(FINAL, value)
    DERIVED.mkdir(parents=True, exist_ok=True)
    atomic_json(DERIVED / "PDF_QA.json", value)
    publication = {
        "schema_version": 1,
        "status": "published_after_visual_qa",
        "published_at_utc": now(),
        "previous_user_pdf_sha256": "30e7e444735d915e240075a04b780b668da8960beed07cb0ffeeacccedeb7406",
        "source_pdf_sha256": prepared["pdf_sha256"],
        "published_pdf_sha256": sha256(USER_PDF),
        "byte_identical": True,
        "pdf_qa_sha256": sha256(FINAL),
    }
    atomic_json(CAMPAIGN / "PDF_PUBLICATION_RECORD.json", publication)
    atomic_json(DERIVED / "PDF_PUBLICATION_RECORD.json", publication)
    manifest_path = DERIVED / "ASSET_MANIFEST.json"
    manifest = read_json(manifest_path)
    manifest["postpublication"] = {
        "PDF_QA.json": {"sha256": sha256(DERIVED / "PDF_QA.json")},
        "PDF_PUBLICATION_RECORD.json": {"sha256": sha256(DERIVED / "PDF_PUBLICATION_RECORD.json")},
    }
    atomic_json(manifest_path, manifest)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    subs.add_parser("prepare")
    record = subs.add_parser("record-visual")
    record.add_argument("--records", type=Path, required=True)
    subs.add_parser("finalize")
    args = parser.parse_args()
    result = prepare() if args.action == "prepare" else record_visual(args.records) if args.action == "record-visual" else finalize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HeldoutError, OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
