#!/usr/bin/env python3
"""Fail-closed terminal audit for the frozen-Z error-analysis and dissertation goal."""

from __future__ import annotations

import json
import os
import py_compile
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
for value in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.frozen_z_error_analysis_common import (  # noqa: E402
    CAMPAIGN, DISSERTATION, INPUT_CAMPAIGN, INPUT_FINAL_SHA256, QUALITY_LEDGER_SHA256, REPO,
    TERMINAL_STATUS, USER_PDF, ErrorAnalysisFailure, atomic_json, atomic_text, now, read_json, record_file, require,
    sha256,
)

GOAL_SHA256 = "704141d08c28d5dcac587f10b99a5037edaa4381ce50a41319247093c6528679"
AGENTS_SHA256 = "1e4853dbbf86ea987f9e66093a22155916aae97daa8c4d2284f170a31e604c19"
PROTECTED = {
    "PRO_REASONING_RESEARCH_HANDOFF.md": "c64abee8234d7cc17ce92b29f019f58ce963c2ff67e7b659e827db443a0b3aaa",
    "legacy_goal2.md": "8b6bbb28e08182e94f490f72a30f2972e7f9d1be6564ae733b5be32f50010023",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def verify_record(path: Path, expected: dict[str, Any], label: str) -> None:
    require(record_file(path) == expected, f"Artifact binding differs: {label}")


def audit_locks() -> dict[str, Any]:
    lock = read_json(CAMPAIGN / "PRE_ANALYSIS_LOCK_V3.json")
    require(lock.get("status") == "locked" and lock.get("candidate_analysis_outputs_present") is False,
            "V3 pre-analysis lock differs")
    for name, expected in lock["locks"].items():
        verify_record(CAMPAIGN / name, expected, name)
    require((CAMPAIGN / "analysis_interrupted_v2_inventory_guard").is_dir(), "Preserved V2 attempt missing")
    require((CAMPAIGN / "RESEARCH_REPORT.interrupted_v2.md").is_file(), "Preserved V2 report missing")
    return {"status": "pass", "pre_analysis_lock_sha256": sha256(CAMPAIGN / "PRE_ANALYSIS_LOCK_V3.json")}


def audit_analysis() -> dict[str, Any]:
    inventory = read_json(CAMPAIGN / "analysis/ANALYSIS_INVENTORY.json")
    require(inventory.get("status") == "complete" and inventory.get("logical_tables") == 1
            and inventory.get("logical_figures") == 1 and inventory.get("forbidden_subgroups_present") is False,
            "Analysis inventory differs")
    for relative, expected in inventory["files"].items():
        verify_record(CAMPAIGN / relative, expected, relative)
    analysis = read_json(CAMPAIGN / "analysis/ERROR_ANALYSIS.json")
    expected = {
        "testA": {"both_correct": 2135, "full_only": 748, "z_only": 526, "both_incorrect": 2248},
        "testB": {"both_correct": 1413, "full_only": 622, "z_only": 505, "both_incorrect": 2555},
    }
    require(analysis.get("heldout_status_unchanged") == "HELDOUT_MIXED_SPLIT_RESULT", "Held-out status changed")
    for split, counts in expected.items():
        require(analysis["results"][split]["transitions"] == counts, f"Transition counts differ: {split}")
        require(list(analysis["results"][split]["threshold_curve"]) == [
            "0.30", "0.35", "0.40", "0.45", "0.50", "0.55", "0.60", "0.65", "0.70"
        ], f"Threshold order differs: {split}")
    words = len((CAMPAIGN / "RESEARCH_REPORT.md").read_text(encoding="utf-8").split())
    require(400 <= words <= 600, f"Analysis discussion word count differs: {words}")
    return {"status": "pass", "report_words": words, "analysis_sha256": sha256(CAMPAIGN / "analysis/ERROR_ANALYSIS.json")}


def audit_retirement_claims_assets() -> dict[str, Any]:
    retirement = read_json(CAMPAIGN / "RETIREMENT_COMPLETION.json")
    require(retirement.get("status") == "complete" and retirement.get("active_namespace_sources_absent") is True,
            "Retirement completion differs")
    require(len(retirement["actions"]) == 14, "Retirement scope differs")
    for name, action in retirement["actions"].items():
        require(not (DISSERTATION / name).exists(), f"Retired active source remains: {name}")
        if action["action"] == "moved":
            require(Path(action["archive_path"]).exists(), f"Retirement archive missing: {name}")
    claims = read_json(CAMPAIGN / "analysis/RESULT_CLAIM_AUDIT.json")
    require(claims.get("status") == "pass" and claims.get("unsupported_claims") == []
            and claims.get("unresolved_discrepancies") == [] and claims.get("numeric_result_claims", 0) >= 100,
            "Result claim audit differs")
    ledger = read_json(CAMPAIGN / "analysis/CLAIM_EVIDENCE_LEDGER.json")
    require(ledger.get("status") == "complete" and ledger.get("unsupported_claims") == 0,
            "Claim evidence ledger differs")
    assets = read_json(DISSERTATION / "analysis/derived/frozen_z_error_analysis_report_audit/ASSET_MANIFEST.json")
    require(assets.get("status") == "complete" and assets.get("logical_tables") == 1
            and assets.get("logical_figures") == 1, "Error-analysis assets differ")
    for row in assets["generated"].values():
        verify_record(Path(row["path"]), {"bytes": row["bytes"], "sha256": row["sha256"]}, row["path"])
    return {"status": "pass", "claims": claims["numeric_result_claims"], "retired_targets": 14}


def audit_pdf_change_record() -> dict[str, Any]:
    qa = read_json(CAMPAIGN / "PDF_QA.json")
    publication = read_json(CAMPAIGN / "PDF_PUBLICATION_RECORD.json")
    change = read_json(CAMPAIGN / "DISSERTATION_CHANGE_RECORD.json")
    require(qa.get("status") == "pass" and qa.get("page_count") >= 76, "PDF QA differs")
    require(len(qa["pages_inspected"]) == qa["page_count"], "PDF page inspection coverage differs")
    require(publication.get("status") == "published" and publication["source_pdf"] == publication["user_pdf"],
            "PDF publication differs")
    require(change.get("status") == "complete", "Dissertation change record is incomplete")
    source = DISSERTATION / "Dissertation_Template.pdf"
    require(record_file(source) == record_file(USER_PDF) == publication["source_pdf"], "Source/user PDF differs")
    return {
        "status": "pass", "pages": qa["page_count"], "pdf_sha256": sha256(source),
        "human_review_required": qa["human_review_required"],
        "human_metadata_placeholders": qa["human_metadata_placeholders"],
    }


def run_tests_and_static_checks() -> dict[str, Any]:
    changed = git("diff", "545083000a72aeb7060136ba428b7a73e0b81001", "--name-only").splitlines()
    python_files = [REPO / name for name in changed if name.endswith(".py") and (REPO / name).is_file()]
    for path in python_files:
        py_compile.compile(str(path), doraise=True)
    result = subprocess.run(
        ["/workspace/.followup_venv/bin/python", "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    log_path = CAMPAIGN / "analysis/FINAL_TEST_SUITE.log"
    atomic_text(log_path, result.stdout)
    match = re.search(r"Ran (\d+) tests?", result.stdout)
    tests = int(match.group(1)) if match else 0
    require(result.returncode == 0 and tests > 0 and "OK" in result.stdout, f"Full unit suite failed: {tests}")
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=REPO, text=True, capture_output=True)
    require(diff_check.returncode == 0, f"git diff --check failed: {diff_check.stdout}{diff_check.stderr}")
    value = {
        "schema_version": 1, "status": "pass", "completed_at_utc": now(), "tests": tests,
        "returncode": result.returncode, "compiled_python_files": len(python_files),
        "log": record_file(log_path), "git_diff_check": "pass",
    }
    atomic_json(CAMPAIGN / "analysis/FINAL_TEST_SUITE.json", value)
    return value


def audit_terminal_environment() -> dict[str, Any]:
    require(sha256(REPO / "GOAL.md") == GOAL_SHA256, "GOAL.md changed")
    require(sha256(REPO / "AGENTS.md") == AGENTS_SHA256, "AGENTS.md changed")
    for name, expected in PROTECTED.items():
        require(sha256(REPO / name) == expected, f"Protected file changed: {name}")
    status = git("status", "--porcelain", "--untracked-files=all").splitlines()
    require(status == ["?? PRO_REASONING_RESEARCH_HANDOFF.md", "?? legacy_goal2.md"], f"Git status differs: {status}")
    process_text = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    forbidden = [line for line in process_text.splitlines() if any(token in line for token in (
        "analyse_frozen_z_error_analysis.py", "latexmk -pdf", "pdflatex", "build_frozen_z_error_dissertation_assets.py"
    ))]
    require(not forbidden, f"Campaign process remains: {forbidden}")
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True, capture_output=True)
    require(gpu.returncode == 0 and not gpu.stdout.strip(), f"GPU compute process remains: {gpu.stdout}")
    stat = os.statvfs("/workspace")
    free = stat.f_bavail * stat.f_frsize
    require(free >= 80 * 1024**3, f"Workspace reserve differs: {free}")
    return {"status": "pass", "git_head": git("rev-parse", "HEAD"), "free_bytes": free, "gpu_compute": []}


def main() -> int:
    final_path = CAMPAIGN / "FINAL_COMPLETION_REPORT.json"
    require(not final_path.exists(), "Final completion report already exists")
    require(sha256(INPUT_CAMPAIGN / "FINAL_COMPLETION_REPORT.json") == INPUT_FINAL_SHA256,
            "Input completion report changed")
    require(sha256(INPUT_CAMPAIGN / "QUALITY_ARTIFACT_LEDGER.json") == QUALITY_LEDGER_SHA256,
            "Input quality ledger changed")
    evidence = {
        "locks": audit_locks(), "analysis": audit_analysis(),
        "retirement_claims_assets": audit_retirement_claims_assets(),
        "pdf_and_change_record": audit_pdf_change_record(),
    }
    tests = run_tests_and_static_checks()
    environment = audit_terminal_environment()
    registry = read_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json")
    registry.update({"status": "pass", "phase": "complete", "terminal_audit": "pass", "updated_at_utc": now()})
    atomic_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json", registry)
    controller = read_json(CAMPAIGN / "CONTROLLER_STATE.json")
    controller.update({"status": "complete", "phase": "human_review_only", "active_workers": [], "updated_at_utc": now()})
    atomic_json(CAMPAIGN / "CONTROLLER_STATE.json", controller)
    report = {
        "schema_version": 1, "status": TERMINAL_STATUS, "terminal_state": TERMINAL_STATUS,
        "completed_at_utc": now(), "heldout_status": "HELDOUT_MIXED_SPLIT_RESULT",
        "objective": "paired localization threshold-crossing error analysis, exhaustive dissertation evidence audit, revision, PDF QA and publication",
        "issues": [], "evidence": evidence, "tests": tests, "environment": environment,
        "human_review_items": evidence["pdf_and_change_record"]["human_metadata_placeholders"],
        "artifacts": {
            "analysis": record_file(CAMPAIGN / "analysis/ERROR_ANALYSIS.json"),
            "claim_audit": record_file(CAMPAIGN / "analysis/RESULT_CLAIM_AUDIT.json"),
            "claim_ledger": record_file(CAMPAIGN / "analysis/CLAIM_EVIDENCE_LEDGER.json"),
            "retirement": record_file(CAMPAIGN / "RETIREMENT_COMPLETION.json"),
            "pdf_qa": record_file(CAMPAIGN / "PDF_QA.json"),
            "publication": record_file(CAMPAIGN / "PDF_PUBLICATION_RECORD.json"),
            "change_record": record_file(CAMPAIGN / "DISSERTATION_CHANGE_RECORD.json"),
            "registry": record_file(CAMPAIGN / "EXPERIMENT_REGISTRY.json"),
            "controller": record_file(CAMPAIGN / "CONTROLLER_STATE.json"),
            "source_pdf": record_file(DISSERTATION / "Dissertation_Template.pdf"),
            "user_pdf": record_file(USER_PDF),
        },
        "next_action": "human_review_only",
    }
    atomic_json(final_path, report)
    print(json.dumps({"status": TERMINAL_STATUS, "tests": tests["tests"], "pdf_sha256": report["artifacts"]["source_pdf"]["sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ErrorAnalysisFailure, OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError, py_compile.PyCompileError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
