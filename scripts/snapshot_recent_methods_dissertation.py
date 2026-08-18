#!/usr/bin/env python
"""Capture and finalize recoverable dissertation change provenance for this goal."""

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
CAMPAIGN = REPO / "experiments/recent_methods_full_refcoco_val_20260817"
DISSERTATION = Path("/workspace/Dissertation_Draft")
RECORD = CAMPAIGN / "DISSERTATION_CHANGE_RECORD.json"
REGISTRY = CAMPAIGN / "EXPERIMENT_REGISTRY.json"
BACKUP_PARENT = Path("/workspace/dissertation_backups")
BASELINE_FILES = (
    DISSERTATION / "chapters/abstract.tex",
    DISSERTATION / "chapters/experimental_design.tex",
    DISSERTATION / "chapters/technical2.tex",
    DISSERTATION / "chapters/final_evaluation.tex",
    DISSERTATION / "chapters/discussion.tex",
    DISSERTATION / "chapters/conclusions.tex",
    DISSERTATION / "chapters/wordcount.tex",
    DISSERTATION / "CLAIM_EVIDENCE.md",
    DISSERTATION / "analysis/derived/README.md",
    DISSERTATION / "Dissertation_Template.pdf",
    Path("/workspace/dissertation_final.pdf"),
)
OPTIONAL_ABSENT_BASELINE = {Path("/workspace/dissertation_final.pdf")}
FINAL_FILES = (
    *BASELINE_FILES,
    DISSERTATION / "chapters/generated_recent_methods_full_val.tex",
    DISSERTATION / "analysis/build_recent_methods_full_val.py",
    DISSERTATION / "analysis/qa_recent_methods_full_val_pdf.py",
    DISSERTATION / "images/final_evaluation/recent_methods_full_val/quality_vs_token_layer_work.pdf",
    DISSERTATION / "images/final_evaluation/recent_methods_full_val/quality_vs_token_layer_work.png",
    DISSERTATION / "analysis/derived/recent_methods_full_refcoco_val/ASSET_MANIFEST.json",
    DISSERTATION / "analysis/derived/recent_methods_full_refcoco_val/PDF_QA.json",
)


class SnapshotError(RuntimeError):
    """Raised when dissertation provenance cannot be captured safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_bytes(value))
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


def capture() -> dict[str, Any]:
    if RECORD.exists():
        raise SnapshotError("Dissertation change record already exists; refusing to replace it")
    missing = [
        str(path) for path in BASELINE_FILES if not path.is_file() and path not in OPTIONAL_ABSENT_BASELINE
    ]
    if missing:
        raise SnapshotError(f"Baseline dissertation files are missing: {missing}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = BACKUP_PARENT / f"recent_methods_full_val_{stamp}"
    if backup.exists():
        raise SnapshotError(f"Backup collision: {backup}")
    files = {}
    for source in BASELINE_FILES:
        if not source.is_file():
            files[label(source)] = {
                "source_path": str(source),
                "present_before": False,
                "source_sha256_before": None,
                "backup_path": None,
                "backup_sha256": None,
            }
            continue
        destination = backup / backup_relative(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        original_hash = sha256_file(source)
        backup_hash = sha256_file(destination)
        if original_hash != backup_hash:
            raise SnapshotError(f"Backup hash differs: {source}")
        files[label(source)] = {
            "source_path": str(source),
            "present_before": True,
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
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    if record.get("status") != "baseline_captured":
        raise SnapshotError("Change record is not ready to finalize")
    backup = Path(record["backup_directory"])
    if not backup.is_dir():
        raise SnapshotError("Backup directory is missing")
    for name, item in record["baseline_files"].items():
        if not item.get("present_before", True):
            continue
        backup_path = Path(item["backup_path"])
        if not backup_path.is_file() or sha256_file(backup_path) != item["backup_sha256"]:
            raise SnapshotError(f"Baseline backup changed: {name}")
    missing = [str(path) for path in FINAL_FILES if not path.is_file()]
    if missing:
        raise SnapshotError(f"Final dissertation files are missing: {missing}")
    record["status"] = "complete"
    record["finalized_at_utc"] = utc_now()
    record["final_files"] = {
        label(path): {"path": str(path), "sha256_after": sha256_file(path)} for path in FINAL_FILES
    }
    atomic_json(RECORD, record)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry.get("phase_status", {}).get("analysis") != "pass":
        raise SnapshotError("Analysis must pass before dissertation finalization")
    registry["phase_status"]["dissertation"] = "pass"
    registry["status"] = "dissertation_complete"
    registry["dissertation"] = {
        "asset_manifest_sha256": sha256_file(
            DISSERTATION / "analysis/derived/recent_methods_full_refcoco_val/ASSET_MANIFEST.json"
        ),
        "change_record_sha256": sha256_file(RECORD),
        "pdf_qa_sha256": sha256_file(
            DISSERTATION / "analysis/derived/recent_methods_full_refcoco_val/PDF_QA.json"
        ),
        "source_pdf_sha256": sha256_file(DISSERTATION / "Dissertation_Template.pdf"),
        "user_pdf_sha256": sha256_file(Path("/workspace/dissertation_final.pdf")),
    }
    atomic_json(REGISTRY, registry)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "finalize"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = capture() if args.action == "capture" else finalize()
    print(json.dumps({"status": result["status"], "backup_directory": result["backup_directory"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
