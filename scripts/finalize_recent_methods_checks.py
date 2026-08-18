#!/usr/bin/env python
"""Run and record the terminal non-GPU source/test checks for the campaign."""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "experiments/recent_methods_full_refcoco_val_20260817"
OUTPUT = CAMPAIGN / "final_checks/unittest_output.txt"
REPORT = CAMPAIGN / "FINAL_TEST_REPORT.json"
PYTHON_FILES = (
    REPO / "scripts/analyse_recent_methods_full_val.py",
    REPO / "scripts/audit_recent_methods_full_val.py",
    REPO / "scripts/finalize_recent_methods_checks.py",
    REPO / "scripts/init_recent_methods_full_val_campaign.py",
    REPO / "scripts/run_progressive_oracle_refcoco.py",
    REPO / "scripts/run_recent_methods_full_val_campaign.py",
    REPO / "scripts/snapshot_recent_methods_dissertation.py",
    REPO / "scripts/validate_progressive_oracle_run.py",
    REPO / "tests/test_progressive_pruning.py",
    REPO / "tests/test_recent_methods_full_val_analysis.py",
    REPO / "tests/test_recent_methods_full_val_campaign.py",
    REPO / "vlm_pruning/llava_hooks/progressive_pruning.py",
    Path("/workspace/Dissertation_Draft/analysis/build_recent_methods_full_val.py"),
    Path("/workspace/Dissertation_Draft/analysis/qa_recent_methods_full_val_pdf.py"),
)


class CheckError(RuntimeError):
    """Raised when terminal source/test checks fail."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def main() -> int:
    if REPORT.exists():
        raise CheckError("Final test report already exists; refusing to overwrite it")
    compiled = {}
    for path in PYTHON_FILES:
        if not path.is_file():
            raise CheckError(f"Required Python source is missing: {path}")
        py_compile.compile(str(path), doraise=True)
        compiled[str(path)] = sha256_file(path)
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    process = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    atomic_write(OUTPUT, process.stdout.encode("utf-8"))
    matches = re.findall(r"^Ran (\d+) tests? in ", process.stdout, flags=re.MULTILINE)
    tests = int(matches[-1]) if matches else 0
    diff = subprocess.run(["git", "diff", "--check"], cwd=REPO, text=True, capture_output=True)
    status = "pass" if process.returncode == 0 and tests > 0 and diff.returncode == 0 else "fail"
    report = {
        "status": status,
        "created_at_utc": utc_now(),
        "python": sys.executable,
        "compiled_source_sha256": compiled,
        "command": command,
        "returncode": process.returncode,
        "tests": tests,
        "output_path": str(OUTPUT.relative_to(CAMPAIGN)),
        "output_sha256": sha256_file(OUTPUT),
        "git_diff_check_returncode": diff.returncode,
        "git_diff_check_output": diff.stdout + diff.stderr,
    }
    atomic_json(REPORT, report)
    if status != "pass":
        raise CheckError(f"Terminal checks failed; see {REPORT}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
