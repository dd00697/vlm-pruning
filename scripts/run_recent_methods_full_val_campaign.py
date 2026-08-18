#!/usr/bin/env python
"""Run the frozen smokes and full-validation cells for the recent-methods campaign."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scripts.init_recent_methods_full_val_campaign import (  # noqa: E402
    CAMPAIGN_LIMIT_BYTES,
    CAMPAIGN_ROOT,
    CELLS,
    HISTORICAL_PARITY,
    MANIFEST,
    MIN_RUNTIME_FREE_BYTES,
    ROWS,
)
from scripts.validate_progressive_oracle_run import validate_run  # noqa: E402


RUNNER = REPO_ROOT / "scripts/run_progressive_oracle_refcoco.py"
DISCOVERY_MANIFEST = (
    REPO_ROOT
    / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
)
REGISTRY_PATH = CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"
STATE_PATH = CAMPAIGN_ROOT / "CONTROLLER_STATE.json"


class ControllerError(RuntimeError):
    """Raised when campaign execution violates the frozen contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def campaign_bytes() -> int:
    return sum(path.stat().st_size for path in CAMPAIGN_ROOT.rglob("*") if path.is_file())


def storage_guard() -> dict[str, int]:
    usage = shutil.disk_usage("/workspace")
    used = campaign_bytes()
    if usage.free < MIN_RUNTIME_FREE_BYTES:
        raise ControllerError(f"Runtime free-space reserve violated: {usage.free}")
    if used > CAMPAIGN_LIMIT_BYTES:
        raise ControllerError(f"Campaign size limit violated: {used}")
    return {"free_bytes": usage.free, "campaign_bytes": used}


def active_compute_processes() -> list[str]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    return [line for line in output.splitlines() if line.strip()]


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.is_file() or not STATE_PATH.is_file():
        raise ControllerError("Campaign has not been initialized")
    registry = read_json(REGISTRY_PATH)
    if list(registry.get("cells", {})) != list(CELLS):
        raise ControllerError("Registry cells/order differ from the frozen protocol")
    return registry


def update_state(phase: str, next_action: str, *, workers: list[dict[str, Any]] | None = None) -> None:
    state = read_json(STATE_PATH)
    state.update(
        {
            "updated_at_utc": utc_now(),
            "phase": phase,
            "next_action": next_action,
            "active_workers": workers or [],
        }
    )
    atomic_json(STATE_PATH, state)


def completed_attempt_present(run_dir: Path) -> bool:
    """Return true only for an attempt that reached atomic terminal finalization."""
    return all((run_dir / name).is_file() for name in ("config.json", "predictions.jsonl", "COMPLETED.json"))


def runner_command(
    cell_id: str,
    *,
    manifest: Path,
    output_root: Path,
    run_name: str,
    limit: int | None,
    full_validation: bool,
) -> list[str]:
    if cell_id not in CELLS:
        raise ControllerError(f"Unregistered cell: {cell_id}")
    command = [
        sys.executable,
        str(RUNNER),
        "--cell-id",
        cell_id,
        "--manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--run-name",
        run_name,
        "--schedule",
        str(CELLS[cell_id]["schedule"]),
        "--runtime-implementation",
        str(CELLS[cell_id]["runtime_implementation"]),
        "--seed",
        "0",
        "--resume",
        "--skip-existing",
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if full_validation:
        command.extend(["--registered-scope", "full_validation"])
    return command


def run_one(
    cell_id: str,
    *,
    manifest: Path,
    output_root: Path,
    run_name: str,
    limit: int | None,
    full_validation: bool,
) -> dict[str, Any]:
    storage_guard()
    command = runner_command(
        cell_id,
        manifest=manifest,
        output_root=output_root,
        run_name=run_name,
        limit=limit,
        full_validation=full_validation,
    )
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    expected = ROWS if full_validation else int(limit or 0)
    return validate_run(output_root / run_name, manifest, expected_rows=expected)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def semantic_signature(row: dict[str, Any]) -> dict[str, Any]:
    traces = []
    for trace in row["stage_traces"]:
        traces.append(
            {
                key: value
                for key, value in trace.items()
                if key not in {"selection_latency_s", "merge_latency_s"}
            }
        )
    return {
        "sample_id": row["sample_id"],
        "raw_model_output": row["raw_model_output"],
        "predicted_box": row["predicted_box"],
        "box_iou": row["box_iou"],
        "success_iou_0_5": row["success_iou_0_5"],
        "generated_token_count": row["generated_token_count"],
        "final_retained_indices": row["final_retained_indices"],
        "realized_visual_tokens_by_layer": row["realized_visual_tokens_by_layer"],
        "stage_traces": traces,
    }


def compare_semantics(
    left: Path,
    right: Path,
    *,
    expected_rows: int,
    right_prefix: bool = False,
) -> None:
    left_rows = read_jsonl(left)
    right_rows = read_jsonl(right)
    right_count_ok = len(right_rows) >= expected_rows if right_prefix else len(right_rows) == expected_rows
    if len(left_rows) != expected_rows or not right_count_ok:
        raise ControllerError("Semantic parity inputs have the wrong row count")
    right_rows = right_rows[:expected_rows]
    for index, (left_row, right_row) in enumerate(zip(left_rows, right_rows, strict=True)):
        if semantic_signature(left_row) != semantic_signature(right_row):
            raise ControllerError(f"Semantic parity mismatch at row {index}")


def run_smokes(limit: int) -> dict[str, Any]:
    if limit not in {2, 20}:
        raise ControllerError("Only registered 2-row and 20-row smokes are allowed")
    registry = load_registry()
    if active_compute_processes():
        raise ControllerError("GPU compute process is already active before serial smokes")
    label = "smoke2" if limit == 2 else "smoke20"
    output_root = CAMPAIGN_ROOT / "smokes" / ("two" if limit == 2 else "twenty")
    reports: dict[str, Any] = {}
    update_state(label, f"running_{label}")
    for cell_id in CELLS:
        run_name = f"{cell_id.lower()}_attempt1"
        reports[cell_id] = run_one(
            cell_id,
            manifest=MANIFEST,
            output_root=output_root,
            run_name=run_name,
            limit=limit,
            full_validation=False,
        )
    registry["phase_status"][label] = "pass"
    registry.setdefault("smoke_evidence", {})[label] = {
        cell_id: {
            "run_dir": report["run_dir"],
            "config_sha256": report["config_sha256"],
            "predictions_sha256": report["predictions_sha256"],
            "metrics": report["metrics"],
        }
        for cell_id, report in reports.items()
    }
    atomic_json(REGISTRY_PATH, registry)
    update_state(label, "run_historical_parity" if limit == 20 else "run_smoke20")
    return reports


def run_historical_parity(limit: int = 20) -> dict[str, Any]:
    if limit != 20:
        raise ControllerError("Historical parity is frozen at 20 rows")
    registry = load_registry()
    if registry["phase_status"].get("smoke20") != "pass":
        raise ControllerError("20-row full-manifest smokes must pass before historical parity")
    output_root = CAMPAIGN_ROOT / "smokes/historical_parity"
    reports: dict[str, Any] = {}
    update_state("historical_parity", "running_historical_parity")
    for cell_id, historical_relative in HISTORICAL_PARITY.items():
        run_name = f"{cell_id.lower()}_attempt1"
        report = run_one(
            cell_id,
            manifest=DISCOVERY_MANIFEST,
            output_root=output_root,
            run_name=run_name,
            limit=limit,
            full_validation=False,
        )
        historical = REPO_ROOT / historical_relative / "predictions.jsonl"
        compare_semantics(
            output_root / run_name / "predictions.jsonl",
            historical,
            expected_rows=limit,
            right_prefix=True,
        )
        reports[cell_id] = report
    registry["phase_status"]["historical_parity"] = "pass"
    registry["historical_parity_smoke"] = {
        cell_id: {
            "run_dir": report["run_dir"],
            "config_sha256": report["config_sha256"],
            "predictions_sha256": report["predictions_sha256"],
            "historical_path": HISTORICAL_PARITY[cell_id],
            "semantic_parity": True,
        }
        for cell_id, report in reports.items()
    }
    atomic_json(REGISTRY_PATH, registry)
    update_state("historical_parity", "run_parallel_soak")
    return reports


def _process_progress(processes: dict[str, subprocess.Popen[Any]], paths: dict[str, Path]) -> None:
    while processes:
        time.sleep(30)
        status = []
        finished = []
        for cell_id, process in processes.items():
            partial = paths[cell_id] / "predictions.partial.jsonl"
            final = paths[cell_id] / "predictions.jsonl"
            path = final if final.is_file() else partial
            rows = 0
            if path.is_file():
                with path.open("rb") as handle:
                    rows = sum(1 for _ in handle)
            code = process.poll()
            status.append(f"{cell_id}:{rows}/{ROWS if 'runs/' in str(paths[cell_id]) else 20}")
            if code is not None:
                finished.append((cell_id, code))
        print("progress " + " ".join(status), flush=True)
        for cell_id, code in finished:
            process = processes.pop(cell_id)
            process.wait()
            if code != 0:
                raise ControllerError(f"Worker {cell_id} exited with code {code}")


def run_parallel_soak() -> dict[str, Any]:
    registry = load_registry()
    if registry["phase_status"].get("historical_parity") != "pass":
        raise ControllerError("Historical parity must pass before the two-worker soak")
    if active_compute_processes():
        raise ControllerError("GPU compute process is already active before the two-worker soak")
    cells = ["V1_K12_48_K20_6", "V4_M448"]
    output_root = CAMPAIGN_ROOT / "smokes/parallel_soak"
    processes: dict[str, subprocess.Popen[Any]] = {}
    handles = []
    paths: dict[str, Path] = {}
    update_state("parallel_soak", "running_parallel_soak")
    try:
        for cell_id in cells:
            run_name = f"{cell_id.lower()}_attempt1"
            run_dir = output_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            log_handle = (run_dir / "worker.log").open("a", encoding="utf-8")
            handles.append(log_handle)
            command = runner_command(
                cell_id,
                manifest=MANIFEST,
                output_root=output_root,
                run_name=run_name,
                limit=20,
                full_validation=False,
            )
            processes[cell_id] = subprocess.Popen(
                command, cwd=REPO_ROOT, stdout=log_handle, stderr=subprocess.STDOUT, text=True
            )
            paths[cell_id] = run_dir
        update_state(
            "parallel_soak",
            "running_parallel_soak",
            workers=[{"cell_id": cell, "pid": process.pid} for cell, process in processes.items()],
        )
        _process_progress(processes, paths)
    finally:
        for handle in handles:
            handle.close()
    reports = {
        cell_id: validate_run(paths[cell_id], MANIFEST, expected_rows=20) for cell_id in cells
    }
    for cell_id in cells:
        serial = CAMPAIGN_ROOT / "smokes/twenty" / f"{cell_id.lower()}_attempt1/predictions.jsonl"
        compare_semantics(paths[cell_id] / "predictions.jsonl", serial, expected_rows=20)
    storage = storage_guard()
    gpu_free = int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
        ).strip()
    )
    registry["phase_status"]["parallel_soak"] = "pass"
    registry["parallel_soak"] = {
        "cells": cells,
        "semantic_parity": True,
        "max_workers": 2,
        "post_soak_gpu_free_mib": gpu_free,
        "storage": storage,
        "reports": reports,
    }
    atomic_json(REGISTRY_PATH, registry)
    update_state("parallel_soak", "run_full_validation")
    return registry["parallel_soak"]


def run_full(max_workers: int) -> dict[str, Any]:
    if max_workers not in {1, 2}:
        raise ControllerError("Full validation permits one or two workers only")
    registry = load_registry()
    required = ["smoke2", "smoke20", "historical_parity"]
    if any(registry["phase_status"].get(key) != "pass" for key in required):
        raise ControllerError("All serial smokes and historical parity must pass first")
    if max_workers == 2 and registry["phase_status"].get("parallel_soak") != "pass":
        raise ControllerError("Two-worker full execution requires a passing parallel soak")
    if active_compute_processes():
        raise ControllerError("GPU compute process is already active before full validation")
    output_root = CAMPAIGN_ROOT / "runs"
    ordered = list(CELLS)
    update_state("full_validation", "running_full_validation")
    for start in range(0, len(ordered), max_workers):
        batch = ordered[start : start + max_workers]
        processes: dict[str, subprocess.Popen[Any]] = {}
        handles = []
        paths: dict[str, Path] = {}
        existing_reports: dict[str, dict[str, Any]] = {}
        try:
            storage_guard()
            for cell_id in batch:
                run_name = f"{cell_id.lower()}_attempt1"
                run_dir = output_root / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                paths[cell_id] = run_dir
                if completed_attempt_present(run_dir):
                    existing_reports[cell_id] = validate_run(
                        run_dir, MANIFEST, expected_rows=ROWS, finalize=False
                    )
                    continue
                log_handle = (run_dir / "worker.log").open("a", encoding="utf-8")
                handles.append(log_handle)
                command = runner_command(
                    cell_id,
                    manifest=MANIFEST,
                    output_root=output_root,
                    run_name=run_name,
                    limit=None,
                    full_validation=True,
                )
                processes[cell_id] = subprocess.Popen(
                    command, cwd=REPO_ROOT, stdout=log_handle, stderr=subprocess.STDOUT, text=True
                )
            update_state(
                "full_validation",
                "running_full_validation",
                workers=[{"cell_id": cell, "pid": process.pid} for cell, process in processes.items()],
            )
            _process_progress(processes, paths)
        finally:
            for handle in handles:
                handle.close()
        registry = load_registry()
        for cell_id in batch:
            report = existing_reports.get(cell_id) or validate_run(
                paths[cell_id], MANIFEST, expected_rows=ROWS
            )
            registry["cells"][cell_id]["status"] = "accepted"
            registry["cells"][cell_id]["accepted_attempt"] = str(paths[cell_id].relative_to(CAMPAIGN_ROOT))
            registry["cells"][cell_id]["validation"] = report
        atomic_json(REGISTRY_PATH, registry)
    registry = load_registry()
    if any(cell["status"] != "accepted" for cell in registry["cells"].values()):
        raise ControllerError("Not every registered full-validation cell is accepted")
    registry["phase_status"]["full_validation"] = "pass"
    registry["status"] = "full_validation_complete"
    registry["full_validation_completed_at_utc"] = utc_now()
    atomic_json(REGISTRY_PATH, registry)
    update_state("full_validation", "run_analysis")
    return registry


def mark_tests(tests: int) -> dict[str, Any]:
    if tests <= 0:
        raise ControllerError("Test count must be positive")
    registry = load_registry()
    registry["phase_status"]["tests"] = "pass"
    registry["tests"] = {"status": "pass", "tests": tests, "recorded_at_utc": utc_now()}
    atomic_json(REGISTRY_PATH, registry)
    update_state("tests", "run_smoke2")
    return registry["tests"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    tests = sub.add_parser("mark-tests")
    tests.add_argument("--tests", type=int, required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--limit", type=int, choices=[2, 20], required=True)
    sub.add_parser("historical-parity")
    sub.add_parser("parallel-soak")
    full = sub.add_parser("full")
    full.add_argument("--max-workers", type=int, choices=[1, 2], default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.action == "mark-tests":
            result = mark_tests(args.tests)
        elif args.action == "smoke":
            result = run_smokes(args.limit)
        elif args.action == "historical-parity":
            result = run_historical_parity()
        elif args.action == "parallel-soak":
            result = run_parallel_soak()
        else:
            result = run_full(args.max_workers)
    except (ControllerError, OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        update_state("error", f"human_review_or_exact_repair_required:{type(exc).__name__}")
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "pass", "action": args.action, "result": result}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
