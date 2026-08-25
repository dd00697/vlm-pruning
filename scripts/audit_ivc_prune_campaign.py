#!/usr/bin/env python3
"""Fail-closed terminal audit for the IVC-Prune experiment and report update."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO_ROOT / "experiments/ivc_prune_refcoco_val_20260825"


def atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load(path: Path, expected_status: str, issues: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        issues.append(f"missing:{path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"invalid_json:{path}:{type(exc).__name__}")
        return None
    if value.get("status") != expected_status:
        issues.append(f"status:{path}:{value.get('status')}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dissertation-root", type=Path)
    parser.add_argument("--pdf", type=Path)
    return parser.parse_args()


def audit(args: argparse.Namespace) -> int:
    issues: list[str] = []
    load(CAMPAIGN / "SOURCE_LOCK.json", "SOURCE_LOCKED", issues)
    load(
        CAMPAIGN / "PROTOCOL.json",
        "PROTOCOL_FROZEN_NO_ACCEPTED_GPU_RUNS_YET",
        issues,
    )
    load(CAMPAIGN / "analysis/quality_analysis.json", "QUALITY_ANALYSIS_COMPLETE", issues)
    timing = load(
        CAMPAIGN / "analysis/timing_natural_analysis.json", "TIMING_NATURAL_ANALYSIS_COMPLETE", issues
    )
    trigger = load(CAMPAIGN / "analysis/FIXED32_TRIGGER.json", None, issues)
    if trigger is not None and trigger.get("required") is True:
        load(
            CAMPAIGN / "analysis/timing_fixed32_analysis.json",
            "TIMING_FIXED32_ANALYSIS_COMPLETE",
            issues,
        )
    required_ledgers = ["smoke", "quality", "timing-natural"]
    if trigger is not None and trigger.get("required") is True:
        required_ledgers.append("timing-fixed32")
    for phase in required_ledgers:
        ledger = CAMPAIGN / f"ledgers/{phase}.json"
        value = load(ledger, "complete", issues)
        if value and len(value.get("jobs", [])) != int(value.get("expected_jobs", -1)):
            issues.append(f"ledger_job_count:{ledger}")
        if value:
            intervals = []
            for row in value.get("jobs", []):
                if row.get("gpu_compute_pids_before") or row.get("gpu_compute_pids_after"):
                    issues.append(f"competing_gpu_process:{ledger}")
                if row.get("started_at_utc") and row.get("finished_at_utc"):
                    intervals.append((str(row["started_at_utc"]), str(row["finished_at_utc"])))
            for previous, current in itertools.pairwise(intervals):
                if current[0] < previous[1]:
                    issues.append(f"overlapping_gpu_jobs:{ledger}")
    heldout_names = [path for path in CAMPAIGN.rglob("*") if "testa" in path.name.lower() or "testb" in path.name.lower()]
    if heldout_names:
        issues.append("forbidden_heldout_artifacts:" + ",".join(str(path) for path in heldout_names))
    if args.dissertation_root is not None or args.pdf is not None:
        if args.dissertation_root is None or args.pdf is None:
            issues.append("final_report_audit_requires_both_dissertation_root_and_pdf")
        else:
            dissertation = args.dissertation_root.resolve()
            pdf = args.pdf.resolve()
            if not pdf.is_file():
                issues.append(f"missing_pdf:{pdf}")
            active_text = "\n".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in dissertation.rglob("*.tex")
                if "generated" not in {part.lower() for part in path.parts}
            )
            forbidden_result_labels = (
                "FEATHER (LLaVA adapter)",
                "FEATHER minus Full",
                "retained-index-fixture-validated FEATHER",
            )
            for label in forbidden_result_labels:
                if label in active_text:
                    issues.append(f"obsolete_feather_result_label:{label}")
            required_terms = ("IVC-Prune", "IVC-stack Full")
            for term in required_terms:
                if term not in active_text:
                    issues.append(f"missing_ivc_report_term:{term}")
            if pdf.is_file():
                result = subprocess.run(
                    ["pdftotext", str(pdf), "-"], check=False, text=True, capture_output=True
                )
                if result.returncode:
                    issues.append("pdftotext_failed")
                else:
                    for label in forbidden_result_labels:
                        if label in result.stdout:
                            issues.append(f"obsolete_feather_pdf_label:{label}")
                    for term in required_terms:
                        if term not in result.stdout:
                            issues.append(f"missing_ivc_pdf_term:{term}")
    status = "IVC_CAMPAIGN_COMPLETE_HUMAN_REVIEW_REQUIRED" if not issues else "AUDIT_FAILED"
    report = {
        "schema_version": 1,
        "status": status,
        "issues": issues,
        "natural_timing_status": timing.get("status") if timing else None,
        "fixed32_required": trigger.get("required") if trigger else None,
        "scientific_boundary": (
            "IVC is a post-hoc external reference and does not alter Z_M448_0 selection or held-out results."
        ),
    }
    atomic_json(CAMPAIGN / "FINAL_COMPLETION_REPORT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not issues else 2


def main() -> int:
    try:
        return audit(parse_args())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
