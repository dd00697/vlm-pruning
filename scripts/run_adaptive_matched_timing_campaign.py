#!/usr/bin/env python
"""Run the registered adaptive matched-timing trials strictly serially."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
OUTPUT_ROOT = CAMPAIGN_ROOT / "matched_timing"
LOG_ROOT = CAMPAIGN_ROOT / "matched_timing_logs"
STATE_PATH = CAMPAIGN_ROOT / "MATCHED_TIMING_STATE.json"
MANIFEST = REPO_ROOT / (
    "experiments/progressive_oracle_refcoco_val_20260814/"
    "manifests/refcoco_val_discovery1000.jsonl"
)
SELECTION_PATH = CAMPAIGN_ROOT / "analysis/GATE_RESULTS_AND_SELECTION.json"
PYTHON = Path("/workspace/.followup_venv/bin/python")
RUNNER = REPO_ROOT / "scripts/run_adaptive_matched_timing_trial.py"
NATURAL_ORDERS = (
    ("full", "k12_original", "candidate"),
    ("k12_original", "candidate", "full"),
    ("candidate", "full", "k12_original"),
    ("full", "candidate", "k12_original"),
    ("k12_original", "full", "candidate"),
)
FIXED_ORDERS = NATURAL_ORDERS[:3]


class MatchedTimingCampaignError(RuntimeError):
    """Raised when the serial matched-timing campaign cannot proceed safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def gpu_processes() -> list[str]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def entries(mode: str) -> list[dict[str, Any]]:
    if mode == "smoke":
        orders = (NATURAL_ORDERS[0],)
    elif mode == "natural":
        orders = NATURAL_ORDERS
    elif mode == "fixed32":
        orders = FIXED_ORDERS
    else:
        raise MatchedTimingCampaignError(f"Unknown timing mode: {mode}")
    return [
        {
            "id": f"{mode}_rep{repetition}_pos{position}_{method}",
            "repetition": repetition,
            "position": position,
            "method": method,
        }
        for repetition, order in enumerate(orders, 1)
        for position, method in enumerate(order, 1)
    ]


def prerequisites(mode: str) -> None:
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if not selection.get("timing_triggered") or not selection["project_selection"].get("timing_triggered"):
        raise MatchedTimingCampaignError("Matched timing was not triggered by project selection")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.is_file() else {"runs": {}}
    if mode == "natural":
        smoke = [record for key, record in state["runs"].items() if key.startswith("smoke_")]
        if len(smoke) != 3 or any(record.get("status") != "accepted" for record in smoke):
            raise MatchedTimingCampaignError("Natural timing requires all three serial smoke trials")
    if mode == "fixed32":
        analysis_path = CAMPAIGN_ROOT / "analysis/NATURAL_MATCHED_TIMING_RESULTS.json"
        if not analysis_path.is_file():
            raise MatchedTimingCampaignError("Fixed timing requires natural timing analysis")
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        if not analysis.get("fixed_length_trigger", {}).get("triggered"):
            raise MatchedTimingCampaignError("Fixed timing was not triggered")


def run(args: argparse.Namespace) -> int:
    prerequisites(args.mode)
    state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.is_file() else {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "runs": {},
    }
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    smoke = args.mode == "smoke"
    trial_mode = "natural" if smoke else args.mode
    expected_rows = 2 if smoke else 1000
    warmups = 2 if smoke else 20
    for entry in entries(args.mode):
        record = state["runs"].setdefault(
            entry["id"],
            {
                **entry,
                "mode": trial_mode,
                "status": "pending",
                "attempts": [],
            },
        )
        if record.get("status") == "accepted":
            continue
        accepted = False
        for attempt in range(len(record["attempts"]) + 1, 3):
            active = gpu_processes()
            if active:
                raise MatchedTimingCampaignError(
                    "Matched timing requires exclusive CUDA access; active processes: " + "; ".join(active)
                )
            command = [
                str(PYTHON),
                str(RUNNER),
                "--method",
                entry["method"],
                "--repetition",
                str(entry["repetition"]),
                "--order-position",
                str(entry["position"]),
                "--attempt",
                str(attempt),
                "--mode",
                trial_mode,
                "--manifest",
                str(MANIFEST),
                "--output-root",
                str(OUTPUT_ROOT),
                "--limit",
                str(expected_rows),
                "--warmup-examples",
                str(warmups),
                *( ["--smoke"] if smoke else [] ),
            ]
            mode_dir = "smoke" if smoke else trial_mode
            run_dir = OUTPUT_ROOT / mode_dir / (
                f"rep{entry['repetition']}_pos{entry['position']}_{entry['method']}_attempt{attempt}"
            )
            log_path = LOG_ROOT / f"{entry['id']}_attempt{attempt}.log"
            attempt_record = {
                "attempt": attempt,
                "status": "running",
                "started_at_utc": utc_now(),
                "command": shlex.join(command),
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "log_path": str(log_path.relative_to(REPO_ROOT)),
            }
            record["attempts"].append(attempt_record)
            record["status"] = "running"
            state["updated_at_utc"] = utc_now()
            atomic_json(STATE_PATH, state)
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_HOME": "/workspace/cache/huggingface",
                }
            )
            with log_path.open("w", encoding="utf-8") as handle:
                process = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            attempt_record["finished_at_utc"] = utc_now()
            attempt_record["exit_status"] = int(process.returncode)
            if process.returncode != 0:
                attempt_record["status"] = "rejected_runner_exit"
                record["status"] = "retry_pending" if attempt < 2 else "rejected"
                atomic_json(STATE_PATH, state)
                continue
            from scripts.validate_adaptive_matched_timing_trial import validate_run

            try:
                report = validate_run(run_dir, MANIFEST, expected_rows=expected_rows)
            except Exception as exc:
                attempt_record["status"] = "rejected_independent_validation"
                attempt_record["validation_error"] = f"{type(exc).__name__}: {exc}"
                record["status"] = "retry_pending" if attempt < 2 else "rejected"
                atomic_json(STATE_PATH, state)
                continue
            attempt_record["status"] = "accepted"
            attempt_record["integrity_sha256"] = sha256_file(run_dir / "integrity_report.json")
            attempt_record["completion_sha256"] = sha256_file(run_dir / "COMPLETED.json")
            record["status"] = "accepted"
            record["accepted_attempt"] = attempt
            record["accepted_run_dir"] = str(run_dir.relative_to(REPO_ROOT))
            record["metrics"] = report["metrics"]
            state["updated_at_utc"] = utc_now()
            atomic_json(STATE_PATH, state)
            accepted = True
            break
        if not accepted and record.get("status") != "accepted":
            raise MatchedTimingCampaignError(f"Timing trial exhausted bounded attempts: {entry['id']}")
    mode_records = [state["runs"][entry["id"]] for entry in entries(args.mode)]
    if any(record.get("status") != "accepted" for record in mode_records):
        raise MatchedTimingCampaignError(f"Timing mode {args.mode} is not terminally accepted")
    state.setdefault("modes", {})[args.mode] = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "trials": len(mode_records),
        "order": NATURAL_ORDERS if args.mode == "natural" else FIXED_ORDERS if args.mode == "fixed32" else (NATURAL_ORDERS[0],),
    }
    atomic_json(STATE_PATH, state)
    if args.mode == "natural":
        subprocess.check_call([str(PYTHON), "-m", "scripts.analyse_adaptive_matched_timing", "--natural"], cwd=REPO_ROOT)
    elif args.mode == "fixed32":
        subprocess.check_call([str(PYTHON), "-m", "scripts.analyse_adaptive_matched_timing", "--finalize"], cwd=REPO_ROOT)
    print(json.dumps({"status": "complete", "mode": args.mode, "trials": len(mode_records)}, indent=2))
    return 0


def sha256_file(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "natural", "fixed32"], required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except (MatchedTimingCampaignError, OSError, subprocess.SubprocessError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
