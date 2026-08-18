#!/usr/bin/env python
"""Capture and finalize recoverable dissertation provenance for the final-val campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "experiments/final_val_efficiency_baselines_20260818"
DISSERTATION = Path("/workspace/Dissertation_Draft")
ROOT_PDF = Path("/workspace/dissertation_final.pdf")
RECORD = CAMPAIGN / "DISSERTATION_CHANGE_RECORD.json"
REGISTRY = CAMPAIGN / "EXPERIMENT_REGISTRY.json"
BACKUP_PARENT = Path("/workspace/dissertation_backups")
BASELINE_FILES = (
    DISSERTATION / "Dissertation_Template.tex",
    DISSERTATION / "chapters/abstract.tex",
    DISSERTATION / "chapters/introduction.tex",
    DISSERTATION / "chapters/experimental_design.tex",
    DISSERTATION / "chapters/technical.tex",
    DISSERTATION / "chapters/technical2.tex",
    DISSERTATION / "chapters/final_evaluation.tex",
    DISSERTATION / "chapters/discussion.tex",
    DISSERTATION / "chapters/conclusions.tex",
    DISSERTATION / "chapters/futurework.tex",
    DISSERTATION / "chapters/wordcount.tex",
    DISSERTATION / "CLAIM_EVIDENCE.md",
    DISSERTATION / "analysis/derived/README.md",
    DISSERTATION / "Dissertation_Template.pdf",
    ROOT_PDF,
)
GENERATED_NAMES = (
    "quality_work.tex",
    "timing_resources.tex",
    "flops.tex",
    "uncertainty_decision.tex",
    "main.tex",
    "macros.tex",
)
FIGURE_NAMES = (
    "quality_vs_token_work.pdf",
    "quality_vs_natural_latency.pdf",
    "canonical_timing_decomposition.pdf",
)
DERIVED_NAMES = (
    "QUALITY_METHOD_METRICS.json",
    "QUALITY_PAIRED_CONTRASTS.json",
    "TIMING_NATURAL_SUMMARY.json",
    "TIMING_NATURAL_CONTRASTS.json",
    "FLOP_KV_ACCOUNTING.json",
    "PARETO_FRONTIERS.json",
    "FROZEN_FINAL_CANDIDATE.json",
    "CLAIM_EVIDENCE_LEDGER.json",
    "ASSET_MANIFEST.json",
    "PDF_QA.json",
)


class SnapshotError(RuntimeError):
    """Raised when dissertation provenance cannot be captured or finalized safely."""


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


def label(path: Path) -> str:
    if path.is_relative_to(DISSERTATION):
        return f"dissertation:{path.relative_to(DISSERTATION)}"
    return f"workspace:{path.relative_to('/workspace')}"


def backup_relative(path: Path) -> Path:
    if path.is_relative_to(DISSERTATION):
        return Path("Dissertation_Draft") / path.relative_to(DISSERTATION)
    return Path("workspace") / path.relative_to("/workspace")


def final_files() -> tuple[Path, ...]:
    generated = DISSERTATION / "chapters/generated_final_val_efficiency"
    figures = DISSERTATION / "images/final_evaluation/final_val_efficiency"
    derived = DISSERTATION / "analysis/derived/final_val_efficiency_baselines"
    return (
        *BASELINE_FILES,
        *(generated / name for name in GENERATED_NAMES),
        *(figures / name for name in FIGURE_NAMES),
        *(derived / name for name in DERIVED_NAMES),
    )


def capture() -> dict[str, Any]:
    if RECORD.exists():
        raise SnapshotError("Dissertation change record already exists; refusing to replace it")
    missing = [str(path) for path in BASELINE_FILES if not path.is_file()]
    if missing:
        raise SnapshotError(f"Baseline dissertation files are missing: {missing}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = BACKUP_PARENT / f"final_val_efficiency_{stamp}"
    if backup.exists():
        raise SnapshotError(f"Backup collision: {backup}")
    files = {}
    for source in BASELINE_FILES:
        destination = backup / backup_relative(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        original_hash = sha256_file(source)
        backup_hash = sha256_file(destination)
        if original_hash != backup_hash:
            raise SnapshotError(f"Backup hash differs: {source}")
        files[label(source)] = {
            "source_path": str(source),
            "source_sha256_before": original_hash,
            "backup_path": str(destination),
            "backup_sha256": backup_hash,
        }
    record = {
        "status": "baseline_captured",
        "captured_at_utc": utc_now(),
        "backup_directory": str(backup),
        "baseline_files": files,
        "final_files": {},
    }
    atomic_json(RECORD, record)
    return record


def finalize() -> dict[str, Any]:
    if not RECORD.is_file():
        raise SnapshotError("Dissertation baseline has not been captured")
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    if record.get("status") != "baseline_captured":
        raise SnapshotError("Dissertation change record is not ready to finalize")
    backup = Path(record["backup_directory"])
    if not backup.is_dir():
        raise SnapshotError("Dissertation backup directory is missing")
    for name, item in record["baseline_files"].items():
        backup_path = Path(item["backup_path"])
        if not backup_path.is_file() or sha256_file(backup_path) != item["backup_sha256"]:
            raise SnapshotError(f"Dissertation backup changed: {name}")
    required = final_files()
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SnapshotError(f"Final dissertation artifacts are missing: {missing}")
    record["status"] = "complete"
    record["finalized_at_utc"] = utc_now()
    record["final_files"] = {
        label(path): {"path": str(path), "sha256_after": sha256_file(path)}
        for path in required
    }
    atomic_json(RECORD, record)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry.get("phase_status", {}).get("analysis") != "pass":
        raise SnapshotError("Analysis must pass before dissertation finalization")
    registry["phase_status"]["dissertation"] = "pass"
    registry["status"] = "dissertation_complete_final_audit_pending"
    registry["dissertation"] = {
        "asset_manifest_sha256": sha256_file(
            DISSERTATION / "analysis/derived/final_val_efficiency_baselines/ASSET_MANIFEST.json"
        ),
        "change_record_sha256": sha256_file(RECORD),
        "pdf_qa_sha256": sha256_file(
            DISSERTATION / "analysis/derived/final_val_efficiency_baselines/PDF_QA.json"
        ),
        "source_pdf_sha256": sha256_file(DISSERTATION / "Dissertation_Template.pdf"),
        "user_pdf_sha256": sha256_file(ROOT_PDF),
    }
    atomic_json(REGISTRY, registry)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "finalize"))
    args = parser.parse_args()
    result = capture() if args.action == "capture" else finalize()
    print(json.dumps({"status": result["status"], "backup": result["backup_directory"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
