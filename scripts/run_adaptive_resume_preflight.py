#!/usr/bin/env python
"""Prove deterministic physical resume for the adaptive campaign runner."""

from __future__ import annotations

import json
import signal
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/workspace/.followup_venv/bin/python")
MANIFEST = REPO_ROOT / (
    "experiments/progressive_oracle_refcoco_val_20260814/"
    "manifests/refcoco_val_discovery1000.jsonl"
)
OUTPUT_ROOT = REPO_ROOT / (
    "experiments/adaptive_efficiency_refcoco_val_20260815/preflight/smoke_runs"
)
RUN_NAME = "resume_c448"
RUN_DIR = OUTPUT_ROOT / RUN_NAME
LOG_PATH = RUN_DIR.parent / "resume_c448_driver.log"


def row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def command(*, resume: bool) -> list[str]:
    result = [
        str(PYTHON),
        "scripts/run_progressive_oracle_refcoco.py",
        "--cell-id",
        "PREFLIGHT_RESUME_C448",
        "--manifest",
        str(MANIFEST),
        "--output-root",
        str(OUTPUT_ROOT),
        "--run-name",
        RUN_NAME,
        "--schedule",
        "4:448:query_topk,12:48:query_topk",
        "--seed",
        "0",
        "--limit",
        "2",
    ]
    if resume:
        result.append("--resume")
    return result


def main() -> int:
    if RUN_DIR.exists():
        raise RuntimeError(f"Resume preflight requires a fresh directory: {RUN_DIR}")
    partial_path = RUN_DIR / "predictions.partial.jsonl"
    with LOG_PATH.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command(resume=False),
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline and process.poll() is None:
            if row_count(partial_path) >= 1:
                process.send_signal(signal.SIGINT)
                break
            time.sleep(0.25)
        else:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            process.wait(timeout=30)
            raise RuntimeError("Did not observe one physical partial row before the deadline")
        interrupted_status = process.wait(timeout=30)
        interrupted_rows = row_count(partial_path)
        if interrupted_status != 130 or interrupted_rows != 1:
            raise RuntimeError(
                f"Expected status 130 with one partial row, got {interrupted_status}/{interrupted_rows}"
            )
        resumed = subprocess.run(
            command(resume=True),
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if resumed.returncode != 0:
        raise RuntimeError(f"Resume command failed with status {resumed.returncode}")
    predictions = RUN_DIR / "predictions.jsonl"
    if row_count(predictions) != 2 or not (RUN_DIR / "COMPLETED.json").is_file():
        raise RuntimeError("Resumed run did not finalize exactly two physical rows")
    evidence = {
        "status": "pass",
        "interrupted_exit_status": interrupted_status,
        "partial_rows_before_resume": interrupted_rows,
        "final_rows": row_count(predictions),
        "run_dir": str(RUN_DIR.relative_to(REPO_ROOT)),
        "log": str(LOG_PATH.relative_to(REPO_ROOT)),
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
