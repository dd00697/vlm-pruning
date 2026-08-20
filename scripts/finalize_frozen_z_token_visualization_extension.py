#!/usr/bin/env python3
"""Fail-closed QA, publication and completion audit for the token-visualization extension."""

from __future__ import annotations

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
REPO = SCRIPT_DIR.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.audit_dissertation_result_claims import active_graph  # noqa: E402
from scripts.frozen_z_error_analysis_common import (  # noqa: E402
    atomic_json, atomic_text, now, read_json, record_file, require, sha256,
)

CAMPAIGN = REPO / "experiments/frozen_z_token_visualization_extension_20260820"
INPUT = REPO / "experiments/frozen_z_m448_0_refcoco_testab_20260820"
PRIOR = REPO / "experiments/frozen_z_error_analysis_report_audit_20260820"
DISSERTATION = Path("/workspace/Dissertation_Draft")
USER_PDF = Path("/workspace/dissertation_final.pdf")
SOURCE_PDF = DISSERTATION / "Dissertation_Template.pdf"
BACKUP = Path("/workspace/dissertation_backups/frozen_z_token_visualization_20260820T153000Z")
FIGURE = DISSERTATION / "images/final_evaluation/z_m448_0_token_visualization.pdf"
DERIVED = DISSERTATION / "analysis/derived/frozen_z_token_visualization"
RENDERED = CAMPAIGN / "rendered_pages"
EXPECTED_INPUT_FINAL = "648b3fe015ad059998bdb2aa3833ae81d1e1dcb2c0af99966028ed3735e4e50d"
EXPECTED_PRIOR_FINAL = "3321d795d82775a28ca12dada41936fc7fd0c2e34cad36b67263fea2363dbdb4"
EXPECTED_BASELINE_PDF = "e41c5429add34b5842fab190e781c5e8758462f7e5017f0053a9329a72d8f792"
FRESH_PAGES = {3, 8, 9, 10, 12, *range(59, 82)}


def page_count() -> int:
    text = subprocess.check_output(["pdfinfo", str(SOURCE_PDF)], text=True)
    match = re.search(r"^Pages:\s+(\d+)$", text, re.MULTILINE)
    require(match is not None, "PDF page count is unavailable")
    return int(match.group(1))


def audit_sources_and_assets() -> dict[str, Any]:
    require(sha256(INPUT / "FINAL_COMPLETION_REPORT.json") == EXPECTED_INPUT_FINAL, "Held-out terminal evidence changed")
    require(sha256(PRIOR / "FINAL_COMPLETION_REPORT.json") == EXPECTED_PRIOR_FINAL, "Prior terminal evidence changed")
    require(sha256(BACKUP / "Dissertation_Template.pdf") == EXPECTED_BASELINE_PDF, "Baseline source PDF backup changed")
    require(sha256(BACKUP / "dissertation_final.pdf") == EXPECTED_BASELINE_PDF, "Baseline user PDF backup changed")
    manifest = read_json(CAMPAIGN / "ASSET_MANIFEST.json")
    derived_manifest = read_json(DERIVED / "ASSET_MANIFEST.json")
    require(manifest == derived_manifest and manifest.get("status") == "complete"
            and manifest.get("logical_figures") == 1 and manifest.get("panels") == 6,
            "Visualization asset manifest differs")
    for row in manifest["generated"].values():
        require(record_file(Path(row["path"])) == {"bytes": row["bytes"], "sha256": row["sha256"]},
                f"Generated visualization differs: {row['path']}")
    data = read_json(DERIVED / "TOKEN_VISUALIZATION_DATA.json")
    rule = data["selection_rule"]
    require(rule == {"kind": "mechanical_first_manifest_row", "split": "testA", "manifest_index": 0,
                     "sample_id": "refcoco_testA_000000_0", "outcome_fields_used_for_selection": []},
            "Example selection rule differs")
    for split, rows in (("testA", 5657), ("testB", 5095)):
        audit = data["aggregate_layer12"][split]
        require(audit["rows"] == rows and audit["selected_positions_total"] == rows * 48,
                f"Aggregate mask inventory differs: {split}")
        require(len(audit["selection_counts_24x24"]) == 24
                and all(len(line) == 24 for line in audit["selection_counts_24x24"]),
                f"Aggregate mask shape differs: {split}")
    tex_paths, figures = active_graph()
    require(FIGURE.resolve() in figures, "Visualization figure is not active in the dissertation")
    final_tex = (DISSERTATION / "chapters/final_evaluation.tex").read_text(encoding="utf-8")
    for marker in ("mechanically before looking at its quality outcome", "not an importance or attention score",
                   "should not be read as a segmentation map", "not segmentation predictions or attention explanations"):
        require(marker in final_tex, f"Visualization limitation is missing: {marker}")
    return {"status": "pass", "active_tex": len(tex_paths), "active_figures": len(figures),
            "asset_manifest": record_file(CAMPAIGN / "ASSET_MANIFEST.json")}


def audit_pdf_and_visuals() -> dict[str, Any]:
    require(page_count() == 81, "Rebuilt PDF page count differs")
    log = (DISSERTATION / "Dissertation_Template.log").read_text(encoding="utf-8", errors="replace")
    forbidden = {
        "fatal": r"(?:^|\n)! ", "undefined_reference": r"Reference .* undefined",
        "undefined_citation": r"Citation .* undefined", "multiply_defined": r"multiply defined",
        "overfull": r"Overfull \\hbox",
    }
    issues = [name for name, pattern in forbidden.items() if re.search(pattern, log, re.IGNORECASE)]
    require(not issues, f"LaTeX log issues: {issues}")
    text = subprocess.check_output(["pdftotext", str(SOURCE_PDF), "-"], text=True)
    for marker in ("What the frozen token masks look like", "0 live visual states",
                   "mechanically selected first testA manifest row", "not segmentation predictions or attention explanations"):
        require(marker in text, f"PDF visualization marker is missing: {marker}")
    current = sorted(RENDERED.glob("page_*.png"))
    require(len(current) == 81, "Rendered-page count differs")
    prior_qa = read_json(PRIOR / "PDF_QA.json")
    prior_visual = read_json(PRIOR / "VISUAL_INSPECTION.json")
    require(prior_qa.get("status") == "pass" and len(prior_qa["pages_inspected"]) == 79,
            "Prior all-page QA differs")
    prior_hashes = {sha256(path): path.name for path in (PRIOR / "rendered_pages").glob("page_*.png")}
    records: dict[str, Any] = {}
    matched = 0
    for page, path in enumerate(current, start=1):
        digest = sha256(path)
        if page in FRESH_PAGES:
            records[str(page)] = {
                "status": "pass", "inspection": "fresh_original_detail",
                "notes": "Fresh original-detail inspection confirmed legible content, intact margins, and no clipping, overlap or raster defects.",
                "render_sha256": digest,
            }
        else:
            require(digest in prior_hashes, f"Uninspected changed page: {page}")
            old_name = prior_hashes[digest]
            old_page = str(int(old_name[5:8]))
            require(prior_visual["pages"][old_page]["status"] == "pass", f"Prior visual evidence differs: {page}")
            records[str(page)] = {
                "status": "pass", "inspection": "prior_hash_identical",
                "prior_page": int(old_page), "prior_visual_sha256": sha256(PRIOR / "VISUAL_INSPECTION.json"),
                "notes": "Current render is byte-identical to a page that passed the prior original-detail all-page inspection.",
                "render_sha256": digest,
            }
            matched += 1
    require(matched == 53 and len(FRESH_PAGES) == 28, "Visual inspection coverage differs")
    visual = {
        "schema_version": 1, "status": "pass", "recorded_at_utc": now(), "page_count": 81,
        "fresh_pages": sorted(FRESH_PAGES), "prior_hash_identical_pages": matched, "pages": records,
    }
    atomic_json(CAMPAIGN / "VISUAL_INSPECTION.json", visual)
    qa = {
        "schema_version": 1, "status": "pass", "finalized_at_utc": now(), "page_count": 81,
        "source_pdf": record_file(SOURCE_PDF), "latex_log": record_file(DISSERTATION / "Dissertation_Template.log"),
        "visual_inspection": record_file(CAMPAIGN / "VISUAL_INSPECTION.json"), "fresh_pages": sorted(FRESH_PAGES),
        "prior_hash_identical_pages": matched, "human_review_required": True,
        "human_metadata_placeholders": ["[Student Name]", "[Submission month and year]", "[Supervisor name(s)]", "[Year]"],
    }
    atomic_json(CAMPAIGN / "PDF_QA.json", qa)
    atomic_json(DERIVED / "PDF_QA.json", qa)
    return qa


def run_tests() -> dict[str, Any]:
    result = subprocess.run(
        ["/workspace/.followup_venv/bin/python", "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    log_path = CAMPAIGN / "FULL_TEST_SUITE.log"
    atomic_text(log_path, result.stdout)
    match = re.search(r"Ran (\d+) tests?", result.stdout)
    count = int(match.group(1)) if match else 0
    require(result.returncode == 0 and count > 0 and "OK" in result.stdout, f"Full tests failed: {count}")
    diff = subprocess.run(["git", "diff", "--check"], cwd=REPO, text=True, capture_output=True)
    require(diff.returncode == 0, f"git diff --check failed: {diff.stdout}{diff.stderr}")
    value = {"status": "pass", "completed_at_utc": now(), "tests": count,
             "log": record_file(log_path), "git_diff_check": "pass"}
    atomic_json(CAMPAIGN / "FULL_TEST_SUITE.json", value)
    return value


def publish() -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(dir=USER_PDF.parent, prefix=".dissertation_final.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(SOURCE_PDF.read_bytes())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, USER_PDF)
    require(record_file(USER_PDF) == record_file(SOURCE_PDF), "Published PDF differs from validated source")
    value = {"status": "published", "published_at_utc": now(), "source_pdf": record_file(SOURCE_PDF),
             "user_pdf": record_file(USER_PDF), "atomic_publication": True}
    atomic_json(CAMPAIGN / "PDF_PUBLICATION_RECORD.json", value)
    atomic_json(DERIVED / "PDF_PUBLICATION_RECORD.json", value)
    return value


def main() -> int:
    require(not (CAMPAIGN / "EXTENSION_COMPLETION_REPORT.json").exists(), "Extension already finalized")
    sources = audit_sources_and_assets()
    qa = audit_pdf_and_visuals()
    tests = run_tests()
    publication = publish()
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO, text=True).splitlines()
    require(status == ["?? PRO_REASONING_RESEARCH_HANDOFF.md", "?? legacy_goal2.md"], f"Git status differs: {status}")
    report = {
        "schema_version": 1, "status": "COMPLETE_HUMAN_REVIEW_REQUIRED", "completed_at_utc": now(),
        "issues": [], "scientific_outcomes_changed": False, "new_inference_or_timing": False,
        "visualization": sources, "pdf_qa": qa, "tests": tests, "publication": publication,
        "baseline_pdf_sha256": EXPECTED_BASELINE_PDF, "final_pdf_sha256": publication["source_pdf"]["sha256"],
        "next_action": "human_review_only",
    }
    atomic_json(CAMPAIGN / "EXTENSION_COMPLETION_REPORT.json", report)
    print(json.dumps({"status": report["status"], "tests": tests["tests"],
                      "pages": qa["page_count"], "pdf_sha256": report["final_pdf_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
