#!/usr/bin/env python3
"""Independent final audit for the selector-ablation and repair goal."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "experiments/final_selector_ablation_and_repair_20260811"
DISS = Path("/workspace/Dissertation_Draft")
PDF = Path("/workspace/dissertation_final.pdf")
OUTPUT = CAMPAIGN / "FINAL_COMPLETION_REPORT.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_final_selector_campaign import cells, integrity, run_dir  # noqa: E402
from vlm_pruning.artifacts import sha256_file  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing required JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def run_check(command: list[str]) -> dict[str, Any]:
    process = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"command": command, "returncode": process.returncode, "status": "pass" if process.returncode == 0 else "fail", "output_tail": process.stdout[-8000:]}


def file_manifest(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-code-checks", action="store_true")
    args = parser.parse_args()
    issues: list[str] = []
    registry = read_json(CAMPAIGN / "campaign_registry.json")
    epoch = read_json(CAMPAIGN / "experiment_epoch.json")
    serial = read_json(CAMPAIGN / "serial_smoke_report.json")
    soak = read_json(CAMPAIGN / "gpu_soak_report.json")
    prelaunch = read_json(CAMPAIGN / "prelaunch_artefact_inventory.json")
    equivalence = read_json(CAMPAIGN / "repair_manifest_identity_equivalence.json")
    run_reports = []
    for cell in cells():
        report = integrity(cell, "full")
        run_reports.append({"run_id": cell.run_id, "family": cell.family, "expected_rows": cell.expected_rows, "run_dir": str(run_dir(cell, "full")), **report})
        if report["status"] != "pass":
            issues.append(f"run {cell.run_id}: {report.get('issues')}")
        else:
            config = read_json(run_dir(cell, "full") / "config.json")
            if config.get("source/git_commit") != epoch.get("git_commit"):
                issues.append(f"run {cell.run_id}: source revision differs from experiment epoch")
    for name, report in (("serial smoke", serial), ("GPU soak", soak), ("prelaunch inventory", prelaunch), ("repair manifest equivalence", equivalence)):
        if report.get("status") != "pass":
            issues.append(f"{name} did not pass")

    derived = DISS / "analysis/derived/final_selector_ablation_and_repair"
    analysis = read_json(derived / "analysis_manifest.json")
    analysis_hashes = file_manifest(derived)
    for relative, expected in analysis.get("files", {}).items():
        path = DISS / relative
        if not path.is_file() or sha256_file(path) != expected:
            issues.append(f"analysis hash mismatch: {relative}")
    ledger = derived / "artefact_supersession_ledger.json"
    ledger_rows = json.loads(ledger.read_text(encoding="utf-8")) if ledger.is_file() else []
    if len(ledger_rows) != 13:
        issues.append(f"supersession ledger has {len(ledger_rows)} rows, expected 13")

    code_checks = [] if args.skip_code_checks else [
        run_check([sys.executable, "-m", "py_compile", "scripts/run_baseline.py", "scripts/run_official_baseline.py", "scripts/run_ug_prune.py", "scripts/run_ug_prune_single_pass.py", "scripts/run_strong_dissertation_ablations.py", "scripts/run_final_selector_campaign.py", "scripts/audit_dissertation_artifacts.py", "scripts/analyse_final_selector_campaign.py", "scripts/final_selector_completion_audit.py"]),
        run_check([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]),
    ]
    if any(check["status"] != "pass" for check in code_checks):
        issues.append("one or more independent code checks failed")

    pdf_qa = read_json(DISS / "FINAL_SELECTOR_REPAIR_PDF_QA.json")
    report_record = read_json(DISS / "analysis/derived/final_selector_ablation_and_repair/report_update_record.json")
    if pdf_qa.get("status") != "pass":
        issues.append("PDF QA did not pass")
    if report_record.get("status") != "complete":
        issues.append("report update record is incomplete")
    if not PDF.is_file() or sha256_file(PDF) != pdf_qa.get("final_pdf_sha256"):
        issues.append("user-facing PDF does not match the QA record")
    backups = sorted(Path("/workspace").glob("dissertation_final.pre_selector_repair_*.pdf"))
    if not backups:
        issues.append("recoverable pre-revision PDF backup is missing")

    status_output = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if status_output:
        issues.append("repository worktree is dirty at final audit")
    if revision != epoch.get("git_commit"):
        issues.append("current revision differs from the registered experiment epoch")
    process_output = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    active = [line.strip() for line in process_output.splitlines() if any(token in line for token in ("run_ug_prune_single_pass.py", "run_official_baseline.py", "run_final_selector_campaign.py")) and "final_selector_completion_audit.py" not in line]
    if active:
        issues.append(f"required experiment processes still active: {active}")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"], text=True).strip()

    payload = {
        "status": "complete" if not issues else "incomplete", "audited_at_utc": utc_now(), "issues": issues,
        "clean_git_commit": revision, "git_status_porcelain": status_output,
        "campaign_registry_sha256": sha256_file(CAMPAIGN / "campaign_registry.json"),
        "source_epoch": epoch, "environment_gpu": gpu, "three_process_soak": soak,
        "run_count": len(run_reports), "runs": run_reports,
        "prelaunch_inventory": {"status": prelaunch["status"], "sha256": sha256_file(CAMPAIGN / "prelaunch_artefact_inventory.json")},
        "repair_manifest_equivalence": equivalence, "analysis": analysis,
        "analysis_files_sha256": analysis_hashes, "supersession_ledger_sha256": sha256_file(ledger),
        "mechanism_results": json.loads((derived / "mechanism_quality.json").read_text(encoding="utf-8")),
        "paired_intervals": json.loads((derived / "mechanism_paired_intervals.json").read_text(encoding="utf-8")),
        "code_checks": code_checks, "report_update": report_record, "pdf_qa": pdf_qa,
        "final_pdf_sha256": sha256_file(PDF) if PDF.is_file() else None,
        "prior_pdf_backups": [{"path": str(path), "sha256": sha256_file(path)} for path in backups],
        "active_experiment_processes": active,
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(json.dumps({"status": payload["status"], "issues": issues, "output": str(OUTPUT)}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
