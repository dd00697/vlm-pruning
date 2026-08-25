#!/usr/bin/env python3
"""Execute the frozen IVC-Prune campaign serially with an auditable GPU ledger."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm_pruning.ivc_campaign import (
    METHODS,
    IVCCampaignError,
    timing_order,
)

CAMPAIGN = REPO_ROOT / "experiments/ivc_prune_refcoco_val_20260825"
QUALITY_MANIFEST = REPO_ROOT / "data/manifests/final_evaluation/refcoco_val_unique.jsonl"
TIMING_MANIFEST = (
    REPO_ROOT
    / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("smoke", "quality", "timing-natural", "timing-fixed32"), required=True
    )
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--smoke-limit", type=int, default=2)
    return parser.parse_args()


def gpu_compute_pids() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    )
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]


def plans(phase: str, smoke_limit: int = 2) -> list[dict[str, Any]]:
    if phase == "smoke":
        if not 1 <= smoke_limit <= 20:
            raise IVCCampaignError("Smoke limit must be in [1, 20]")
        return [
            {
                "method": method,
                "mode": "smoke",
                "generation_mode": "natural",
                "manifest": TIMING_MANIFEST,
                "run_dir": CAMPAIGN / f"smoke/{method.lower()}",
                "limit": smoke_limit,
                "warmups": 0,
                "repetition": 1,
                "position": position,
            }
            for position, method in enumerate(METHODS, 1)
        ]
    if phase == "quality":
        return [
            {
                "method": method,
                "mode": "quality",
                "generation_mode": "natural",
                "manifest": QUALITY_MANIFEST,
                "run_dir": CAMPAIGN / f"quality/{method.lower()}",
                "limit": None,
                "warmups": 0,
                "repetition": 1,
                "position": position,
            }
            for position, method in enumerate(METHODS, 1)
        ]
    if phase not in {"timing-natural", "timing-fixed32"}:
        raise IVCCampaignError(f"Unregistered campaign phase: {phase}")
    generation_mode = "natural" if phase == "timing-natural" else "fixed32"
    repetitions = 5 if generation_mode == "natural" else 3
    output = []
    for repetition in range(1, repetitions + 1):
        for position, method in enumerate(timing_order(generation_mode, repetition), 1):
            output.append(
                {
                    "method": method,
                    "mode": "timing",
                    "generation_mode": generation_mode,
                    "manifest": TIMING_MANIFEST,
                    "run_dir": CAMPAIGN
                    / f"timing/{generation_mode}/rep{repetition}_pos{position}_{method.lower()}",
                    "limit": 1000,
                    "warmups": 20,
                    "repetition": repetition,
                    "position": position,
                }
            )
    return output


def command(args: argparse.Namespace, job: dict[str, Any]) -> list[str]:
    result = [
        str(args.python.resolve()),
        "scripts/run_ivc_prune_refcoco.py",
        "--method",
        job["method"],
        "--mode",
        job["mode"],
        "--generation-mode",
        job["generation_mode"],
        "--manifest",
        str(job["manifest"]),
        "--run-dir",
        str(job["run_dir"]),
        "--source-root",
        str(args.source_root.resolve()),
        "--model-snapshot",
        str(args.model_snapshot.resolve()),
        "--warmup-examples",
        str(job["warmups"]),
        "--repetition",
        str(job["repetition"]),
        "--order-position",
        str(job["position"]),
    ]
    if job["limit"] is not None:
        result.extend(["--limit", str(job["limit"])])
    return result


def execute(args: argparse.Namespace) -> int:
    protocol = CAMPAIGN / "PROTOCOL.json"
    if not protocol.is_file():
        raise IVCCampaignError("Campaign protocol must be frozen before GPU execution")
    if args.phase == "timing-fixed32":
        trigger_path = CAMPAIGN / "analysis/FIXED32_TRIGGER.json"
        if not trigger_path.is_file() or json.loads(trigger_path.read_text()).get("required") is not True:
            raise IVCCampaignError("Fixed32 is not authorized without a positive analysis trigger")
    jobs = plans(args.phase, args.smoke_limit)
    ledger_path = CAMPAIGN / f"ledgers/{args.phase}.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("phase") != args.phase or int(ledger.get("expected_jobs", -1)) != len(jobs):
            raise IVCCampaignError("Existing ledger differs from the frozen execution plan")
        ledger["status"] = "running"
    else:
        ledger = {
            "schema_version": 1,
            "phase": args.phase,
            "expected_jobs": len(jobs),
            "serial_execution": True,
            "jobs": [],
            "status": "running",
        }
    existing = {
        (row["method"], int(row["repetition"]), int(row["order_position"])): row
        for row in ledger["jobs"]
    }
    environment = dict(os.environ)
    ivc_transformers = args.source_root.resolve() / "transformers/src"
    environment.update(
        {
            "PYTHONPATH": str(ivc_transformers),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_VERBOSITY": "error",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONWARNINGS": "ignore",
        }
    )
    for job in jobs:
        key = (job["method"], job["repetition"], job["position"])
        run_dir: Path = job["run_dir"]
        completed = run_dir / "COMPLETED.json"
        record = existing.get(key)
        if completed.is_file():
            if json.loads(completed.read_text()).get("status") != "complete":
                raise IVCCampaignError(f"Rejected pre-existing completion: {completed}")
            if record is None:
                record = {
                    "method": job["method"],
                    "repetition": job["repetition"],
                    "order_position": job["position"],
                    "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                    "status": "validated_preexisting_complete",
                }
                ledger["jobs"].append(record)
                existing[key] = record
            atomic_json(ledger_path, ledger)
            continue
        if gpu_compute_pids():
            raise IVCCampaignError("Another GPU compute process is active before an IVC cell")
        cmd = command(args, job)
        if run_dir.exists():
            if not (run_dir / "predictions.partial.jsonl").is_file():
                raise IVCCampaignError(f"Existing run is not resumable: {run_dir}")
            cmd.append("--resume")
        log = CAMPAIGN / f"logs/{args.phase}/{job['method'].lower()}_rep{job['repetition']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        if record is None:
            record = {
                "method": job["method"],
                "repetition": job["repetition"],
                "order_position": job["position"],
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "command": cmd,
                "started_at_utc": utc_now(),
                "gpu_compute_pids_before": [],
                "status": "running",
            }
            ledger["jobs"].append(record)
            existing[key] = record
        else:
            record.setdefault("resume_started_at_utc", []).append(utc_now())
            record.update({"command": cmd, "status": "running"})
        atomic_json(ledger_path, ledger)
        with log.open("a", encoding="utf-8") as handle:
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record.update(
            {
                "finished_at_utc": utc_now(),
                "exit_code": result.returncode,
                "status": "complete" if result.returncode == 0 else "failed",
                "gpu_compute_pids_after": gpu_compute_pids(),
            }
        )
        atomic_json(ledger_path, ledger)
        if result.returncode or not completed.is_file() or record["gpu_compute_pids_after"]:
            raise IVCCampaignError(f"IVC campaign cell failed: {job['method']} rep{job['repetition']}")
    ledger.update({"status": "complete", "completed_at_utc": utc_now()})
    atomic_json(ledger_path, ledger)
    print(json.dumps({"status": "complete", "ledger": str(ledger_path)}, indent=2))
    return 0


def main() -> int:
    try:
        return execute(parse_args())
    except (IVCCampaignError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
