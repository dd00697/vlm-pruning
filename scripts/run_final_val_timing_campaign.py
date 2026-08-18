#!/usr/bin/env python
"""Execute one frozen final-campaign timing stack serially with an audited ledger."""

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
CAMPAIGN = REPO_ROOT / "experiments/final_val_efficiency_baselines_20260818"
TIMING_MANIFEST = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
CANONICAL_PYTHON = Path("/workspace/.followup_venv/bin/python")
FASTV_PYTHON = CAMPAIGN / "envs/fastv/bin/python"
NUWA_PYTHON = CAMPAIGN / "envs/nuwa/bin/python"
FASTV_TRANSFORMERS = REPO_ROOT / "third_party/FastV/src/FastV/llava-hf/transformers/src"
NUWA_SOURCE = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815/official_sources/checkouts/nuwa"
LLAVA_SOURCE = Path("/workspace/third_party_official/LLaVA")
HF_HOME = Path("/workspace/cache/huggingface")

CANONICAL_ORDER = {
    1: ("Full", "K12", "M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN"),
    2: ("K12", "M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN", "Full"),
    3: ("M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN", "Full", "K12"),
    4: ("Z_M448_0", "FEATHER_LLAVA_MAIN", "Full", "K12", "M448_OPT"),
    5: ("FEATHER_LLAVA_MAIN", "Full", "K12", "M448_OPT", "Z_M448_0"),
}
PAIRED_ORDER = {
    1: ("FULL", "PRUNED"),
    2: ("PRUNED", "FULL"),
    3: ("FULL", "PRUNED"),
    4: ("PRUNED", "FULL"),
    5: ("FULL", "PRUNED"),
}
CUDA_LIBRARY_PATHS = (
    CAMPAIGN / "envs/nuwa/lib/python3.11/site-packages/nvidia/cusparselt/lib",
    Path("/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib"),
    Path("/usr/local/lib/python3.11/dist-packages/nvidia/cuda_cupti/lib"),
    Path("/usr/local/lib/python3.11/dist-packages/nvidia/cuda_nvrtc/lib"),
    Path("/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib"),
    Path("/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib"),
    Path("/usr/local/lib/python3.11/dist-packages/nvidia/cufft/lib"),
    Path("/usr/local/lib/python3.11/dist-packages/nvidia/curand/lib"),
    Path("/usr/local/lib/python3.11/dist-packages/nvidia/cusolver/lib"),
    Path("/usr/local/lib/python3.11/dist-packages/nvidia/cusparse/lib"),
    Path("/usr/local/lib/python3.11/dist-packages/nvidia/nccl/lib"),
    Path("/usr/local/lib/python3.11/dist-packages/nvidia/nvjitlink/lib"),
    Path("/usr/local/lib/python3.11/dist-packages/nvidia/nvtx/lib"),
)


class TimingCampaignError(RuntimeError):
    """Raised when a serial timing campaign cannot preserve its frozen contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    parser.add_argument("--stack", choices=("canonical", "fastv", "nuwa"), required=True)
    parser.add_argument("--mode", choices=("natural", "fixed32"), required=True)
    return parser.parse_args()


def gpu_compute_pids() -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]


def plans(stack: str, mode: str) -> list[tuple[int, int, str, Path, list[str]]]:
    repetitions = 5 if mode == "natural" else 3
    jobs = []
    if stack == "canonical":
        for repetition in range(1, repetitions + 1):
            for position, method in enumerate(CANONICAL_ORDER[repetition], 1):
                run_dir = CAMPAIGN / "timing/canonical" / mode / (
                    f"rep{repetition}_pos{position}_{method.lower()}_attempt1"
                )
                command = [
                    str(CANONICAL_PYTHON),
                    "scripts/run_final_val_canonical_timing.py",
                    "--method", method,
                    "--mode", mode,
                    "--repetition", str(repetition),
                    "--order-position", str(position),
                    "--attempt", "1",
                ]
                jobs.append((repetition, position, method, run_dir, command))
        return jobs
    full, pruned = (
        ("FASTV_STACK_FULL", "FASTV_MATCHED_199")
        if stack == "fastv"
        else ("NUWA_STACK_FULL", "NUWA_NATIVE_192")
    )
    for repetition in range(1, repetitions + 1):
        methods = tuple(full if item == "FULL" else pruned for item in PAIRED_ORDER[repetition])
        for position, method in enumerate(methods, 1):
            if stack == "fastv":
                run_dir = CAMPAIGN / "timing/fastv_native" / mode / (
                    f"rep{repetition}_pos{position}_{method.lower()}_attempt1"
                )
                command = [
                    str(FASTV_PYTHON),
                    "scripts/run_final_val_fastv_timing.py",
                    "--method", method,
                    "--mode", mode,
                    "--repetition", str(repetition),
                    "--order-position", str(position),
                    "--attempt", "1",
                ]
            else:
                run_dir = CAMPAIGN / "timing/nuwa_native" / mode / (
                    f"rep{repetition}_pos{position}_{method.lower()}_attempt1"
                )
                command = [
                    str(NUWA_PYTHON),
                    "scripts/run_native_nuwa_refcoco.py",
                    "--method", method,
                    "--manifest", str(TIMING_MANIFEST.relative_to(REPO_ROOT)),
                    "--run-dir", str(run_dir.relative_to(REPO_ROOT)),
                    "--mode", "timing",
                    "--generation-mode", mode,
                    "--warmup-examples", "20",
                    "--repetition", str(repetition),
                    "--order-position", str(position),
                ]
            jobs.append((repetition, position, method, run_dir, command))
    return jobs


def environment(stack: str) -> dict[str, str]:
    result = dict(os.environ)
    result.update(
        {
            "HF_HOME": str(HF_HOME),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_VERBOSITY": "error",
            "PYTHONWARNINGS": "ignore",
        }
    )
    if stack == "fastv":
        result["PYTHONPATH"] = str(FASTV_TRANSFORMERS)
        result["VLM_LOCAL_MODEL_PATH"] = str(
            Path("/workspace/cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/")
            / "snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
        )
    elif stack == "nuwa":
        result["PYTHONPATH"] = f"{NUWA_SOURCE}:{LLAVA_SOURCE}"
        result["LD_LIBRARY_PATH"] = ":".join(str(path) for path in CUDA_LIBRARY_PATHS)
    return result


def execute(args: argparse.Namespace) -> int:
    jobs = plans(args.stack, args.mode)
    ledger_path = CAMPAIGN / f"timing/{args.stack}_native" / args.mode / "RUN_LEDGER.json"
    if args.stack == "canonical":
        ledger_path = CAMPAIGN / "timing/canonical" / args.mode / "RUN_LEDGER.json"
    if args.mode == "fixed32":
        trigger_path = CAMPAIGN / "analysis/FIXED32_TRIGGER.json"
        if not trigger_path.is_file():
            raise TimingCampaignError("Fixed32 requires the registered natural-timing trigger artifact")
        trigger = json.loads(trigger_path.read_text(encoding="utf-8"))
        if trigger.get("stacks", {}).get(args.stack, {}).get("required") is not True:
            raise TimingCampaignError(f"Fixed32 is not triggered for the {args.stack} stack")
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if (
            ledger.get("stack") != args.stack
            or ledger.get("mode") != args.mode
            or int(ledger.get("expected_jobs", -1)) != len(jobs)
        ):
            raise TimingCampaignError("Existing timing ledger differs from the frozen plan")
        ledger["status"] = "running"
    else:
        ledger = {
            "schema_version": 1,
            "stack": args.stack,
            "mode": args.mode,
            "expected_jobs": len(jobs),
            "serial_execution": True,
            "jobs": [],
            "status": "running",
        }
    existing = {
        (int(row["repetition"]), int(row["order_position"]), str(row["method"])): row
        for row in ledger["jobs"]
    }
    for repetition, position, method, run_dir, command in jobs:
        key = (repetition, position, method)
        record = existing.get(key)
        completed = run_dir / "COMPLETED.json"
        if completed.is_file():
            status = str(json.loads(completed.read_text()).get("status") or "").lower()
            if status not in {"pass", "complete"}:
                raise TimingCampaignError(f"Rejected pre-existing completion: {completed}")
            if record is None:
                record = {
                    "repetition": repetition,
                    "order_position": position,
                    "method": method,
                    "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                    "status": "validated_preexisting_complete",
                }
                ledger["jobs"].append(record)
                existing[key] = record
            elif record.get("status") not in {"complete", "validated_preexisting_complete"}:
                record["status"] = "complete_after_resume_validation"
            atomic_json(ledger_path, ledger)
            continue
        if gpu_compute_pids():
            raise TimingCampaignError("Another GPU compute process is active before a timing cell")
        if run_dir.exists():
            if not (run_dir / "timing.partial.jsonl").is_file() and not (
                run_dir / "predictions.partial.jsonl"
            ).is_file():
                raise TimingCampaignError(f"Existing run is not resumable: {run_dir}")
            command.append("--resume")
        log_path = ledger_path.parent / "logs" / f"rep{repetition}_pos{position}_{method.lower()}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if record is None:
            record = {
                "repetition": repetition,
                "order_position": position,
                "method": method,
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "command": command,
                "started_at_utc": utc_now(),
                "gpu_compute_pids_before": [],
                "status": "running",
            }
            ledger["jobs"].append(record)
            existing[key] = record
        else:
            record.setdefault("resume_started_at_utc", []).append(utc_now())
            record["command"] = command
            record["status"] = "running"
        atomic_json(ledger_path, ledger)
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment(args.stack),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record["finished_at_utc"] = utc_now()
        record["exit_code"] = result.returncode
        record["status"] = "complete" if result.returncode == 0 else "failed"
        record["gpu_compute_pids_after"] = gpu_compute_pids()
        atomic_json(ledger_path, ledger)
        if result.returncode != 0 or not completed.is_file():
            raise TimingCampaignError(f"Timing cell failed: {method} rep{repetition}")
        if record["gpu_compute_pids_after"]:
            raise TimingCampaignError("GPU worker remained after a timing cell")
    ledger["status"] = "complete"
    ledger["completed_at_utc"] = utc_now()
    atomic_json(ledger_path, ledger)
    print(json.dumps({"status": "complete", "ledger": str(ledger_path)}, indent=2))
    return 0


def main() -> int:
    try:
        return execute(parse_args())
    except (TimingCampaignError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
