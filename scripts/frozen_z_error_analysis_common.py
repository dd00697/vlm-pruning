#!/usr/bin/env python3
"""Shared constants and fail-closed helpers for the frozen-Z error-analysis goal."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "experiments/frozen_z_error_analysis_report_audit_20260820"
INPUT_CAMPAIGN = REPO / "experiments/frozen_z_m448_0_refcoco_testab_20260820"
DISSERTATION = Path("/workspace/Dissertation_Draft")
USER_PDF = Path("/workspace/dissertation_final.pdf")
BASELINE_PDF_SHA256 = "c33cbefba4a73a8ad6cca233b641834ff721249478ab32b5718b5d6040d2c9f7"
INPUT_FINAL_SHA256 = "648b3fe015ad059998bdb2aa3833ae81d1e1dcb2c0af99966028ed3735e4e50d"
QUALITY_LEDGER_SHA256 = "7d182ae55896dc8b5188f729bdeda16d2d61ac57306d30e0a35145b4074ff2db"
SOURCE_ANCESTOR = "545083000a72aeb7060136ba428b7a73e0b81001"
TERMINAL_STATUS = "GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED"


class ErrorAnalysisFailure(RuntimeError):
    """Raised when frozen evidence or a required invariant differs."""


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ErrorAnalysisFailure(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode())


def atomic_text(path: Path, value: str) -> None:
    atomic_bytes(path, value.encode())


def record_file(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Required file is missing: {path}")
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def relative(path: Path, root: Path = REPO) -> str:
    return str(path.resolve().relative_to(root.resolve()))
