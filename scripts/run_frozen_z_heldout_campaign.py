#!/usr/bin/env python3
"""Run preregistered held-out smokes and quality cells with serial GPU enforcement."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for value in (SCRIPT_DIR, REPO_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from frozen_z_heldout_common import (
    CAMPAIGN,
    MANIFESTS,
    PYTHON,
    SOURCE_PATHS,
    Z_SCHEDULE,
    HeldoutError,
    atomic_json,
    now,
    read_json,
    sha256,
)
from validate_progressive_oracle_run import validate_run


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/run_progressive_oracle_refcoco.py"
LOCKED_ARTIFACTS = ["FROZEN_PROTOCOL.json", "ANALYSIS_SCHEMA.json", "DECISION_THRESHOLDS.json", "COMMANDS.json", "SOURCE_ENVIRONMENT_LOCK.json"]


def active_compute() -> list[str]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True,
    ).strip()
    return [line for line in output.splitlines() if line.strip()]


def guard() -> None:
    if not CAMPAIGN.is_dir():
        raise HeldoutError("Campaign is not initialized")
    lock = read_json(CAMPAIGN / "PRE_INFERENCE_LOCK.json")
    for name in LOCKED_ARTIFACTS:
        if sha256(CAMPAIGN / name) != lock["artifacts"][name]:
            raise HeldoutError(f"Frozen preregistration artifact changed: {name}")
    source = read_json(CAMPAIGN / "SOURCE_ENVIRONMENT_LOCK.json")["source"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if commit != source["git_commit"]:
        raise HeldoutError("Repository commit differs from preregistered source")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *SOURCE_PATHS], cwd=REPO, text=True).strip()
    if dirty:
        raise HeldoutError("Registered executable source is dirty")
    for path, expected in source["sha256"].items():
        if sha256(REPO / path) != expected:
            raise HeldoutError(f"Registered source hash differs: {path}")
    free = shutil.disk_usage("/workspace").free
    if free < 80 * 1024**3:
        raise HeldoutError(f"Runtime free-space reserve violated: {free}")
    campaign_bytes = sum(path.stat().st_size for path in CAMPAIGN.rglob("*") if path.is_file())
    if campaign_bytes > 20 * 1024**3:
        raise HeldoutError(f"Campaign size cap violated: {campaign_bytes}")


def command(method: str, split: str, run_dir: Path, *, limit: int | None) -> list[str]:
    if method not in {"Full", "Z_M448_0"} or split not in MANIFESTS:
        raise HeldoutError("Unregistered method/split")
    cell_id = f"{method}_{split}"
    result = [
        str(PYTHON), str(RUNNER), "--cell-id", cell_id,
        "--manifest", str(MANIFESTS[split]["path"]),
        "--output-root", str(run_dir.parent), "--run-name", run_dir.name,
        "--runtime-implementation", "original", "--seed", "0",
        "--registered-scope", f"heldout_{split}", "--resume", "--skip-existing",
    ]
    if method == "Z_M448_0":
        result.extend(["--schedule", Z_SCHEDULE, "--outcome-embargo"])
    if limit is not None:
        result.extend(["--limit", str(limit)])
    return result


def update_state(phase: str, next_action: str, workers: list[dict[str, Any]] | None = None) -> None:
    state = read_json(CAMPAIGN / "CONTROLLER_STATE.json")
    state.update({"phase": phase, "updated_at_utc": now(), "next_action": next_action, "active_workers": workers or []})
    atomic_json(CAMPAIGN / "CONTROLLER_STATE.json", state)


def execute_cell(method: str, split: str, run_dir: Path, *, limit: int | None) -> dict[str, Any]:
    guard()
    before = active_compute()
    if before:
        raise HeldoutError(f"GPU is not idle before serial cell: {before}")
    cmd = command(method, split, run_dir, limit=limit)
    logs = CAMPAIGN / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{run_dir.name}.log"
    started = now()
    update_state("gpu_inference", f"running_{run_dir.name}", [{"cell": f"{method}_{split}", "started_at_utc": started, "command": cmd}])
    with log_path.open("a", encoding="utf-8") as handle:
        environment = os.environ.copy()
        environment.update({
            "HF_HOME": "/workspace/cache/huggingface",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        })
        process = subprocess.run(
            cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, env=environment
        )
    after = active_compute()
    if process.returncode != 0:
        raise HeldoutError(f"Cell failed with exit {process.returncode}; see {log_path}")
    if after:
        raise HeldoutError(f"GPU process remained after cell: {after}")
    expected = int(limit if limit is not None else MANIFESTS[split]["rows"])
    report = validate_run(run_dir, Path(MANIFESTS[split]["path"]), expected_rows=expected)
    record = {
        "method": method, "split": split, "run_dir": str(run_dir.relative_to(REPO)),
        "limit": limit, "started_at_utc": started, "finished_at_utc": now(),
        "command": cmd, "exit_code": process.returncode, "pids_before": before, "pids_after": after,
        "config_sha256": report["config_sha256"], "predictions_sha256": report["predictions_sha256"],
        "raw_sealed": method == "Z_M448_0", "log_sha256": sha256(log_path),
    }
    return record


def run_smokes() -> dict[str, Any]:
    registry = read_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json")
    if registry["phase_status"].get("full_audit") != "pass":
        raise HeldoutError("Historical Full audit must finish before smokes")
    if registry["phase_status"].get("smokes") == "pass":
        raise HeldoutError("Registered smokes already passed")
    records = []
    for split in ("testA", "testB"):
        for method in ("Full", "Z_M448_0"):
            name = f"{method.lower()}_{split.lower()}_smoke2_attempt1"
            records.append(execute_cell(method, split, CAMPAIGN / "smokes" / name, limit=2))
    atomic_json(CAMPAIGN / "SMOKE_LEDGER.json", {"schema_version": 1, "status": "complete", "serial_execution": True, "records": records})
    registry = read_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json")
    registry["phase_status"]["smokes"] = "pass"
    atomic_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json", registry)
    update_state("quality", f"{PYTHON} scripts/run_frozen_z_heldout_campaign.py quality")
    return {"status": "smokes_passed", "cells": len(records)}


def run_quality() -> dict[str, Any]:
    registry = read_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json")
    if registry["phase_status"].get("smokes") != "pass":
        raise HeldoutError("Smokes must pass before quality")
    if registry["phase_status"].get("quality") == "raw_sealed_both_splits":
        raise HeldoutError("Quality phase already sealed both candidate splits")
    partial_ledger = CAMPAIGN / "QUALITY_RUN_LEDGER.partial.json"
    records: list[dict[str, Any]] = (
        list(read_json(partial_ledger).get("records", [])) if partial_ledger.is_file() else []
    )
    # Complete or register both controls before the first candidate output exists.
    for split in ("testA", "testB"):
        cell = registry["cells"][f"Full_{split}"]
        if cell["status"] == "rerun_required":
            run_dir = CAMPAIGN / "quality" / f"full_{split.lower()}_attempt1"
            record = execute_cell("Full", split, run_dir, limit=None)
            records.append(record)
            atomic_json(partial_ledger, {"schema_version": 1, "status": "running", "serial_execution": True, "records": records})
            cell.update({"status": "complete_rerun", "run_dir": record["run_dir"], "config_sha256": record["config_sha256"], "predictions_sha256": record["predictions_sha256"]})
            atomic_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json", registry)
        elif cell["status"] not in {"reuse_approved", "complete_rerun"}:
            raise HeldoutError(f"Full {split} has no frozen reuse/rerun decision")
    # One-shot candidates. TestA is sealed but never scored or displayed before testB is sealed.
    for split in ("testA", "testB"):
        key = f"Z_M448_0_{split}"
        if registry["cells"][key]["status"] == "raw_sealed":
            continue
        if registry["cells"][key]["status"] != "pending":
            raise HeldoutError(f"Candidate {split} is not in the sole pending attempt state")
        run_dir = CAMPAIGN / "quality" / f"z_m448_0_{split.lower()}_attempt1"
        record = execute_cell("Z_M448_0", split, run_dir, limit=None)
        records.append(record)
        atomic_json(partial_ledger, {"schema_version": 1, "status": "running", "serial_execution": True, "records": records})
        seal = read_json(run_dir / "SEALED_RAW.json")
        registry = read_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json")
        registry["cells"][key].update({"status": "raw_sealed", "run_dir": record["run_dir"], "config_sha256": record["config_sha256"], "raw_predictions_sha256": seal["raw_predictions_sha256"], "seal_sha256": sha256(run_dir / "SEALED_RAW.json")})
        atomic_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json", registry)
    registry = read_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json")
    if any(registry["cells"][f"Z_M448_0_{split}"]["status"] != "raw_sealed" for split in ("testA", "testB")):
        raise HeldoutError("Both candidate splits are not sealed")
    ledger_path = CAMPAIGN / "QUALITY_RUN_LEDGER.json"
    atomic_json(ledger_path, {"schema_version": 1, "status": "raw_sealed_both_splits", "serial_execution": True, "records": records})
    if partial_ledger.is_file():
        partial_ledger.unlink()
    release = {
        "schema_version": 1, "status": "outcome_embargo_released", "released_at_utc": now(),
        "condition": "both complete candidate raw artifacts independently validated and sealed",
        "candidate_seals": {
            split: sha256(REPO / registry["cells"][f"Z_M448_0_{split}"]["run_dir"] / "SEALED_RAW.json")
            for split in ("testA", "testB")
        },
        "quality_ledger_sha256": sha256(ledger_path),
    }
    atomic_json(CAMPAIGN / "OUTCOME_EMBARGO_RELEASE.json", release)
    registry["phase_status"]["quality"] = "raw_sealed_both_splits"
    registry["phase_status"]["analysis"] = "ready"
    atomic_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json", registry)
    state = read_json(CAMPAIGN / "CONTROLLER_STATE.json")
    state["candidate_outcome_embargo"] = False
    state.update({"phase": "analysis", "updated_at_utc": now(), "active_workers": [], "next_action": f"{PYTHON} scripts/analyse_frozen_z_heldout.py"})
    atomic_json(CAMPAIGN / "CONTROLLER_STATE.json", state)
    return {"status": "raw_sealed_both_splits", "candidate_outcomes_calculated": False, "candidate_outcomes_displayed": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["smokes", "quality"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    guard()
    result = run_smokes() if args.phase == "smokes" else run_quality()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HeldoutError, OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
