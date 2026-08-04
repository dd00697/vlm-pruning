#!/usr/bin/env python
"""Run and record the final Python compile and unit-test gates."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/final_evaluation/verification"
COMPILE_TARGETS = (
    "scripts/run_baseline.py",
    "scripts/run_official_baseline.py",
    "scripts/run_ug_prune.py",
    "scripts/run_ug_prune_single_pass.py",
    "scripts/register_final_protocol.py",
        "scripts/run_final_evaluation.py",
        "scripts/finalize_selection_provenance.py",
        "scripts/freeze_final_method.py",
    "scripts/run_final_postfreeze.py",
    "scripts/prepare_final_timing_subset.py",
    "scripts/run_final_timing_trial.py",
    "scripts/run_final_timing_campaign.py",
    "scripts/final_completion_audit.py",
    "vlm_pruning/prior_methods.py",
    "vlm_pruning/efficiency.py",
    "vlm_pruning/llava_hooks/mid_layer_pruning.py",
    "../Dissertation_Draft/analysis/build_final_evaluation.py",
    "../Dissertation_Draft/analysis/qa_final_pdf.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], log_name: str) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / log_name).write_text(completed.stdout, encoding="utf-8")
    return {
        "command": command,
        "returncode": completed.returncode,
        "log": str((OUTPUT / log_name).relative_to(ROOT)),
    }


def main() -> int:
    missing = [target for target in COMPILE_TARGETS if not (ROOT / target).exists()]
    compile_result = (
        {"returncode": 2, "missing_targets": missing}
        if missing
        else run(
            [sys.executable, "-m", "py_compile", *COMPILE_TARGETS],
            "py_compile.log",
        )
    )
    tests_result = run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        "unittest.log",
    )
    passed = (
        int(compile_result["returncode"]) == 0
        and int(tests_result["returncode"]) == 0
    )
    payload = {
        "status": "passed" if passed else "failed",
        "completed_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "compile": compile_result,
        "tests": tests_result,
        "source_sha256": {
            target: sha256(ROOT / target)
            for target in COMPILE_TARGETS
            if (ROOT / target).exists()
        },
    }
    output = OUTPUT / "final_code_checks.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
