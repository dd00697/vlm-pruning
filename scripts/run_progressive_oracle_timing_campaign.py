#!/usr/bin/env python
"""Evaluate E1/E2 gates and run required exploratory timing serially."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from analyse_progressive_oracle_campaign import strongest_by_registered_rule  # noqa: E402
from validate_progressive_oracle_timing import validate_timing  # noqa: E402


CAMPAIGN_ROOT = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814"
REGISTRY_PATH = CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"
MANIFEST = CAMPAIGN_ROOT / "manifests/refcoco_val_discovery1000.jsonl"
OUTPUT_ROOT = CAMPAIGN_ROOT / "timing"
LOG_ROOT = CAMPAIGN_ROOT / "timing_logs"
PYTHON = Path("/workspace/.followup_venv/bin/python")
RUNNER = REPO_ROOT / "scripts/run_progressive_oracle_timing.py"


class TimingCampaignError(RuntimeError):
    """Raised when timing gates or artifacts violate the campaign protocol."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_registry() -> dict[str, Any]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if len(registry.get("cells") or {}) != 33:
        raise TimingCampaignError("Registry does not contain exactly 33 quality cells")
    if sum(cell.get("status") == "accepted" for cell in registry["cells"].values()) != 33:
        raise TimingCampaignError("Timing gates require all 33 accepted quality cells")
    return registry


def update_totals(registry: dict[str, Any]) -> None:
    quality = sum(cell.get("status") == "accepted" for cell in registry["cells"].values())
    timing = sum(cell.get("status") == "accepted" for cell in registry["timing_cells"].values())
    if quality != 33 or timing > 2:
        raise TimingCampaignError(f"Cell cap violation: quality={quality}, timing={timing}")
    registry["totals"] = {"accepted_quality": quality, "accepted_timing": timing}
    registry["updated_at_utc"] = utc_now()


def save_registry(registry: dict[str, Any]) -> None:
    update_totals(registry)
    atomic_json(REGISTRY_PATH, registry)


def evaluate_gates(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    b_tail = int(registry["decisions"]["B_tail"]["value"])
    e1_candidate = {32: "T1", 16: "T2", 8: "T3", 4: "T4", 0: "T5"}.get(b_tail)
    if e1_candidate is None:
        raise TimingCampaignError(f"No registered tail-only candidate for B_tail={b_tail}")
    progressive = strongest_by_registered_rule(registry, [f"P{i}" for i in range(1, 10)])
    cell = registry["cells"][progressive]
    reference = registry["reference"]["discovery_recomputed"]
    acc_loss = float(reference["acc_at_0_5"]) - float(cell["metrics"]["quality/acc_at_0_5"])
    miou_loss = float(reference["mean_iou"]) - float(cell["metrics"]["quality/mean_iou"])
    validation_ok = all(bool(value) for value in (cell.get("validation") or {}).values())
    return {
        "E1": {
            "passed": b_tail < 48,
            "candidate_cell_id": e1_candidate,
            "B_tail": b_tail,
            "rule": "B_tail < 48",
        },
        "E2": {
            "passed": (
                float(cell["candidate_average_visual_tokens"]) < 246.0
                and acc_loss <= 0.02 + 1e-12
                and miou_loss <= 0.010 + 1e-12
                and validation_ok
                and float(cell["metrics"]["quality/parsed_box_rate"]) == 1.0
            ),
            "candidate_cell_id": progressive,
            "average_visual_tokens": float(cell["candidate_average_visual_tokens"]),
            "acc_at_0_5_loss": acc_loss,
            "mean_iou_loss": miou_loss,
            "validation_ok": validation_ok,
            "parsing_ok": float(cell["metrics"]["quality/parsed_box_rate"]) == 1.0,
            "thresholds": {"average_visual_tokens_lt": 246, "acc_loss_lte": 0.02, "miou_loss_lte": 0.010},
        },
    }


def gpu_processes() -> list[str]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def accepted_is_valid(cell: dict[str, Any]) -> bool:
    if cell.get("status") != "accepted":
        return False
    run_dir_value = (cell.get("artifact_paths") or {}).get("run_dir")
    if not run_dir_value:
        return False
    try:
        report = validate_timing(REPO_ROOT / run_dir_value, MANIFEST)
    except Exception:
        return False
    return report.get("status") == "pass" and (REPO_ROOT / run_dir_value / "COMPLETED.json").is_file()


def run_e1(registry: dict[str, Any], gate: dict[str, Any]) -> None:
    cell = registry["timing_cells"]["E1"]
    if cell.get("status") == "accepted":
        if not accepted_is_valid(cell):
            raise TimingCampaignError("Accepted E1 no longer validates")
        return
    if not gate["passed"]:
        cell.update({"status": "skipped_by_design", "gate": gate, "skip_reason": "E1 gate failed"})
        save_registry(registry)
        return
    candidate_id = str(gate["candidate_cell_id"])
    candidate = registry["cells"][candidate_id]
    resume = cell.get("status") == "running" and bool(cell.get("attempts"))
    if resume:
        attempt = cell["attempts"][-1]
        run_dir = REPO_ROOT / attempt["run_dir"]
        run_name = run_dir.name
        attempt_number = int(attempt["attempt"])
    else:
        attempt_number = len(cell.get("attempts") or []) + 1
        run_name = f"e1_attempt{attempt_number}"
        run_dir = OUTPUT_ROOT / run_name
    schedule = ",".join(
        f"{int(event['boundary'])}:{int(event['budget'])}:{event['selector']}"
        for event in candidate["requested_schedule"]
    )
    command = [
        str(PYTHON), str(RUNNER),
        "--cell-id", "E1",
        "--candidate-cell-id", candidate_id,
        "--candidate-schedule", schedule,
        "--manifest", str(MANIFEST),
        "--output-root", str(OUTPUT_ROOT),
        "--run-name", run_name,
        "--warmup-examples", "20",
    ]
    if resume:
        command.append("--resume")
        attempt.setdefault("resume_events", []).append({"resumed_at_utc": utc_now(), "command": shlex.join(command)})
    else:
        attempt = {
            "attempt": attempt_number,
            "status": "running",
            "started_at_utc": utc_now(),
            "finished_at_utc": None,
            "command": shlex.join(command),
            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "exit_code": None,
            "failure_reason": None,
        }
        cell.setdefault("attempts", []).append(attempt)
    cell.update({"status": "running", "gate": gate, "candidate_cell_id": candidate_id})
    save_registry(registry)
    active = gpu_processes()
    if active:
        raise TimingCampaignError("E1 requires exclusive CUDA access; active processes: " + "; ".join(active))
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{run_name}.log"
    with log_path.open("a" if resume else "w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
        attempt["pid"] = process.pid
        save_registry(registry)
        return_code = process.wait()
    attempt.update({
        "finished_at_utc": utc_now(),
        "exit_code": int(return_code),
        "stdout_log": str(log_path.relative_to(REPO_ROOT)),
    })
    if return_code != 0:
        attempt.update({"status": "rejected", "failure_reason": f"runner_exit_{return_code}"})
        cell.update({"status": "rejected", "failure_reason": attempt["failure_reason"]})
        save_registry(registry)
        raise TimingCampaignError(f"E1 runner failed with exit {return_code}")
    try:
        report = validate_timing(run_dir, MANIFEST)
        completion_path = run_dir / "COMPLETED.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except Exception as exc:
        attempt.update({"status": "rejected", "failure_reason": f"independent_validation:{type(exc).__name__}:{exc}"})
        cell.update({"status": "rejected", "failure_reason": attempt["failure_reason"]})
        save_registry(registry)
        raise TimingCampaignError("E1 independent validation failed") from exc
    attempt["status"] = "accepted"
    cell.update({
        "status": "accepted",
        "failure_reason": None,
        "accepted_attempt": attempt_number,
        "source_commit": config["source/git_commit"],
        "manifest_sha256": report["manifest_sha256"],
        "config_sha256": report["config_sha256"],
        "candidate_schedule": config["candidate_schedule"],
        "matched_full_reference_internal": True,
        "metrics": report["metrics"],
        "validation": report["checks"],
        "completion_hashes": completion,
        "artifact_paths": {
            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "timing": str((run_dir / "per_sample_timing.jsonl").relative_to(REPO_ROOT)),
            "config": str((run_dir / "config.json").relative_to(REPO_ROOT)),
            "metrics": str((run_dir / "metrics.json").relative_to(REPO_ROOT)),
            "integrity": str((run_dir / "integrity_report.json").relative_to(REPO_ROOT)),
            "environment": str((run_dir / "environment.json").relative_to(REPO_ROOT)),
            "completion": str(completion_path.relative_to(REPO_ROOT)),
        },
    })
    save_registry(registry)


def main() -> int:
    try:
        registry = load_registry()
        gates = evaluate_gates(registry)
        registry["timing_gate_decisions"] = gates
        e2 = registry["timing_cells"]["E2"]
        if gates["E2"]["passed"]:
            raise TimingCampaignError("E2 unexpectedly passed; a generic E2 runner must be explicitly reviewed")
        e2.update({
            "status": "skipped_by_design",
            "gate": gates["E2"],
            "candidate_cell_id": gates["E2"]["candidate_cell_id"],
            "skip_reason": "Registered progressive candidate exceeds the Acc@0.5 and mIoU loss tolerances",
        })
        save_registry(registry)
        run_e1(registry, gates["E1"])
        subprocess.check_call(
            [str(PYTHON), "scripts/analyse_progressive_oracle_campaign.py", "--through-campaign", "3"],
            cwd=REPO_ROOT,
        )
        print(json.dumps({"status": "complete", "gates": gates, "totals": load_registry()["totals"]}, indent=2))
        return 0
    except (TimingCampaignError, OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
