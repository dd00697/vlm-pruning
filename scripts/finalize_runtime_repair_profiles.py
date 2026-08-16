#!/usr/bin/env python
"""Aggregate exact runtime-repair profiles and freeze M448_OPT."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for entry in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from run_hardware_runtime_profile import (  # noqa: E402
    CAMPAIGN_ROOT,
    MANIFEST,
    REPAIR_CELL_IDS,
    REPAIR_ORDER_PLAN,
    aggregate,
)
from validate_hardware_runtime_profile import read_json, read_jsonl, validate_run  # noqa: E402
from vlm_pruning.artifacts import sha256_file  # noqa: E402


class RepairProfileError(RuntimeError):
    """Raised when exact repair evidence is incomplete or inconsistent."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def accepted_trial(repetition: int, position: int, cell_id: str) -> Path:
    root = CAMPAIGN_ROOT / "runtime_repairs/profiling/trials"
    candidates = sorted(
        root.glob(f"rep{repetition}_pos{position}_{cell_id.lower()}_attempt*")
    )
    accepted = [path for path in candidates if (path / "COMPLETED.json").is_file()]
    if len(accepted) != 1:
        raise RepairProfileError(
            f"Expected one accepted repair trial for rep={repetition}, pos={position}, "
            f"cell={cell_id}; found {len(accepted)}"
        )
    for path in candidates:
        if path not in accepted and not (path / "REJECTION.json").is_file():
            raise RepairProfileError(f"Nonaccepted repair trial lacks rejection: {path}")
    return accepted[0]


def validate_completion(run_dir: Path) -> dict[str, Any]:
    completion = read_json(run_dir / "COMPLETED.json")
    for filename, key in (
        ("config.json", "config_sha256"),
        ("environment.json", "environment_sha256"),
        ("profiles.jsonl", "profiles_sha256"),
        ("metrics.json", "metrics_sha256"),
        ("integrity_report.json", "integrity_sha256"),
    ):
        if completion.get(key) != sha256_file(run_dir / filename):
            raise RepairProfileError(f"Repair completion hash differs: {run_dir / filename}")
    return completion


def run() -> int:
    output_dir = CAMPAIGN_ROOT / "runtime_repairs/aggregate"
    if output_dir.exists():
        raise RepairProfileError(f"Refusing to overwrite repair aggregate: {output_dir}")
    rows_by_cell: dict[str, list[dict[str, Any]]] = {
        cell_id: [] for cell_id in REPAIR_CELL_IDS
    }
    trials: list[dict[str, Any]] = []
    for repetition, order in REPAIR_ORDER_PLAN.items():
        for position, cell_id in enumerate(order, 1):
            run_dir = accepted_trial(repetition, position, cell_id)
            completion = validate_completion(run_dir)
            independent = validate_run(run_dir, MANIFEST)
            rows_by_cell[cell_id].extend(read_jsonl(run_dir / "profiles.jsonl"))
            trials.append(
                {
                    "repetition": repetition,
                    "order_position": position,
                    "cell_id": cell_id,
                    "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                    "completion_sha256": sha256_file(run_dir / "COMPLETED.json"),
                    "profiles_sha256": completion["profiles_sha256"],
                    "independent_validation": independent,
                }
            )
    results: dict[str, Any] = {}
    for cell_id, rows in rows_by_cell.items():
        if len(rows) != 500 or len({row["sample_id"] for row in rows}) != 100:
            raise RepairProfileError(f"{cell_id} does not have five aligned 100-row trials")
        if not all(row.get("semantic_match") is True for row in rows):
            raise RepairProfileError(f"{cell_id} contains semantic drift")
        results[cell_id] = aggregate(rows)

    base_path = CAMPAIGN_ROOT / "profiling/aggregate/PROFILE_RESULTS.json"
    base_hash = sha256_file(base_path)
    base = read_json(base_path)["P_M448"]["means"]
    feasible: list[str] = []
    decisions: dict[str, Any] = {}
    for cell_id, result in results.items():
        means = result["means"]
        prefill_ratio = float(means["prefill_latency_s"]) / float(base["prefill_latency_s"])
        e2e_ratio = float(means["end_to_end_latency_s"]) / float(base["end_to_end_latency_s"])
        accepted = prefill_ratio <= 1.005 and e2e_ratio <= 1.005
        decisions[cell_id] = {
            "prefill_ratio_to_original_m448": prefill_ratio,
            "end_to_end_ratio_to_original_m448": e2e_ratio,
            "within_registered_half_percent_guard": accepted,
        }
        if accepted:
            feasible.append(cell_id)
    selected = (
        min(feasible, key=lambda cell_id: results[cell_id]["means"]["end_to_end_latency_s"])
        if feasible
        else "P_M448"
    )
    selection = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "base_profile_results_path": str(base_path.relative_to(REPO_ROOT)),
        "base_profile_results_sha256": base_hash,
        "base_m448_means": base,
        "candidate_decisions": decisions,
        "m448_opt": selected,
        "m448_opt_is_original": selected == "P_M448",
        "selection_rule": (
            "lowest 100-row five-repetition natural end-to-end mean among exact repairs "
            "no more than 0.5% slower than original M448 in prefill or end-to-end"
        ),
    }
    output_dir.mkdir(parents=True)
    atomic_write(output_dir / "TRIALS.json", canonical_bytes({"trials": trials}))
    atomic_write(output_dir / "REPAIR_PROFILE_RESULTS.json", canonical_bytes(results))
    atomic_write(output_dir / "M448_OPT_SELECTION.json", canonical_bytes(selection))
    completion = {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "trials": 15,
        "rows_per_cell": 500,
        "trials_sha256": sha256_file(output_dir / "TRIALS.json"),
        "results_sha256": sha256_file(output_dir / "REPAIR_PROFILE_RESULTS.json"),
        "selection_sha256": sha256_file(output_dir / "M448_OPT_SELECTION.json"),
        "m448_opt": selected,
    }
    atomic_write(output_dir / "COMPLETED.json", canonical_bytes(completion))
    print(json.dumps({"status": "complete", "selection": selection}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    try:
        return run()
    except (RepairProfileError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
