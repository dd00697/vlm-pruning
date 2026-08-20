#!/usr/bin/env python3
"""Capture and finalize the campaign-specific dissertation change record."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
for value in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from frozen_z_heldout_common import CAMPAIGN, HeldoutError, atomic_json, now, read_json, sha256
from finalize_frozen_z_heldout import append_event, write_ledger, verify_chain


DRAFT = Path("/workspace/Dissertation_Draft")
USER_PDF = Path("/workspace/dissertation_final.pdf")
RECORD = CAMPAIGN / "DISSERTATION_CHANGE_RECORD.json"
BASELINE_PDF_SHA256 = "30e7e444735d915e240075a04b780b668da8960beed07cb0ffeeacccedeb7406"
BASELINE_FILES = [
    Path("Dissertation_Template.pdf"), Path("Dissertation_Template.tex"),
    Path("chapters/abstract.tex"), Path("chapters/introduction.tex"),
    Path("chapters/experimental_design.tex"), Path("chapters/final_evaluation.tex"),
    Path("chapters/discussion.tex"), Path("chapters/conclusions.tex"),
    Path("chapters/wordcount.tex"), Path("CLAIM_EVIDENCE.md"),
    Path("analysis/derived/README.md"),
]
FINAL_FILES = BASELINE_FILES[1:] + [
    Path("chapters/generated_frozen_z_heldout/macros.tex"),
    Path("chapters/generated_frozen_z_heldout/main.tex"),
    Path("analysis/derived/frozen_z_m448_0_refcoco_testab/HELDOUT_RESULTS.json"),
    Path("analysis/derived/frozen_z_m448_0_refcoco_testab/PAIRED_CONTRASTS.json"),
    Path("analysis/derived/frozen_z_m448_0_refcoco_testab/HELDOUT_DECISION.json"),
    Path("analysis/derived/frozen_z_m448_0_refcoco_testab/SPLIT_HETEROGENEITY.json"),
    Path("analysis/derived/frozen_z_m448_0_refcoco_testab/CLAIM_EVIDENCE_LEDGER.json"),
    Path("analysis/derived/frozen_z_m448_0_refcoco_testab/ASSET_MANIFEST.json"),
    Path("analysis/derived/frozen_z_m448_0_refcoco_testab/PDF_QA.json"),
    Path("analysis/derived/frozen_z_m448_0_refcoco_testab/PDF_PUBLICATION_RECORD.json"),
]


def record_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise HeldoutError(f"Required dissertation file is missing: {path}")
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def capture() -> dict[str, object]:
    if RECORD.exists():
        raise HeldoutError("Dissertation change record already exists")
    draft_pdf = DRAFT / "Dissertation_Template.pdf"
    if sha256(draft_pdf) != BASELINE_PDF_SHA256 or sha256(USER_PDF) != BASELINE_PDF_SHA256:
        raise HeldoutError("Dissertation baseline PDFs differ from the audited pre-campaign PDF")
    stamp = now().replace(":", "").replace("-", "")
    backup = Path("/workspace/dissertation_backups") / f"frozen_z_m448_0_heldout_{stamp}"
    if backup.exists():
        raise HeldoutError(f"Backup collision: {backup}")
    baseline: dict[str, object] = {}
    for relative in BASELINE_FILES:
        source = DRAFT / relative
        baseline[str(relative)] = record_file(source)
        target = backup / "Dissertation_Draft" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    user_target = backup / "dissertation_final.pdf"
    user_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(USER_PDF, user_target)
    payload = {
        "schema_version": 1, "status": "baseline_captured", "captured_at_utc": now(),
        "backup_root": str(backup), "baseline_pdf_sha256": BASELINE_PDF_SHA256,
        "baseline_files": baseline,
        "baseline_user_pdf": record_file(USER_PDF),
    }
    atomic_json(RECORD, payload)
    return payload


def finalize() -> dict[str, object]:
    value = read_json(RECORD)
    if value.get("status") != "baseline_captured":
        raise HeldoutError("Dissertation change record is not in baseline_captured state")
    backup = Path(str(value["backup_root"]))
    for relative, expected in value["baseline_files"].items():
        path = backup / "Dissertation_Draft" / relative
        if record_file(path) != expected:
            raise HeldoutError(f"Backup integrity differs: {relative}")
    if record_file(backup / "dissertation_final.pdf") != value["baseline_user_pdf"]:
        raise HeldoutError("Backed-up user PDF differs")
    qa_path = CAMPAIGN / "PDF_QA.json"
    if read_json(qa_path).get("status") != "pass":
        raise HeldoutError("Campaign PDF QA has not passed")
    draft_pdf = DRAFT / "Dissertation_Template.pdf"
    if sha256(draft_pdf) != sha256(USER_PDF):
        raise HeldoutError("Draft and published user PDFs are not byte-identical")
    final_files = {str(path): record_file(DRAFT / path) for path in FINAL_FILES}
    value.update({
        "status": "complete", "finalized_at_utc": now(), "final_files": final_files,
        "final_draft_pdf": record_file(draft_pdf), "final_user_pdf": record_file(USER_PDF),
        "pdf_qa_sha256": sha256(qa_path),
    })
    atomic_json(RECORD, value)
    ledger_path = CAMPAIGN / "DECISION_LEDGER.jsonl"
    rows = []
    with ledger_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    verify_chain(rows)
    append_event(
        rows,
        "dissertation_and_pdf_qa_complete",
        RECORD,
        "Outcome-faithful dissertation assets, prose, rebuilt PDF, all-page visual QA and byte-identical publication passed.",
    )
    write_ledger(ledger_path, rows)
    registry_path = CAMPAIGN / "EXPERIMENT_REGISTRY.json"
    registry = read_json(registry_path)
    registry["phase_status"]["dissertation"] = "pass"
    registry["dissertation_change_record_sha256"] = sha256(RECORD)
    registry["pdf_qa_sha256"] = sha256(qa_path)
    registry["decision_ledger_sha256"] = sha256(ledger_path)
    atomic_json(registry_path, registry)
    state_path = CAMPAIGN / "CONTROLLER_STATE.json"
    state = read_json(state_path)
    state.update({"phase": "terminal_audit", "updated_at_utc": now(), "active_workers": [], "next_action": "run full tests and terminal audit"})
    atomic_json(state_path, state)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["capture", "finalize"])
    args = parser.parse_args()
    result = capture() if args.action == "capture" else finalize()
    print(json.dumps({"status": result["status"], "record": str(RECORD)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HeldoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
