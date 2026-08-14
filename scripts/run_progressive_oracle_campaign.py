#!/usr/bin/env python
"""Resumable controller for the 33-cell progressive/oracle discovery campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
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

from scripts.validate_progressive_oracle_run import validate_run  # noqa: E402
from vlm_pruning.artifacts import sha256_file  # noqa: E402


CAMPAIGN_ROOT = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814"
REGISTRY_PATH = CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"
MANIFEST = CAMPAIGN_ROOT / "manifests/refcoco_val_discovery1000.jsonl"
RUN_ROOT = CAMPAIGN_ROOT / "runs"
PREFLIGHT_PATH = CAMPAIGN_ROOT / "preflight/PREFLIGHT_REPORT.json"
PYTHON = Path("/workspace/.followup_venv/bin/python")
RUNNER = REPO_ROOT / "scripts/run_progressive_oracle_refcoco.py"
EXPECTED_MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"


class CampaignControllerError(RuntimeError):
    """Raised when controller state or a run violates the campaign contract."""


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
    if not REGISTRY_PATH.is_file():
        raise CampaignControllerError(f"Missing registry: {REGISTRY_PATH}")
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if len(registry.get("cells") or {}) != 33:
        raise CampaignControllerError("Registry does not contain exactly 33 quality cells")
    return registry


def update_totals(registry: dict[str, Any]) -> None:
    accepted_quality = sum(cell.get("status") == "accepted" for cell in registry["cells"].values())
    accepted_timing = sum(cell.get("status") == "accepted" for cell in registry["timing_cells"].values())
    if accepted_quality > 33 or accepted_timing > 2:
        raise CampaignControllerError("Accepted cell cap exceeded")
    registry["totals"] = {
        "accepted_quality": accepted_quality,
        "accepted_timing": accepted_timing,
    }
    registry["updated_at_utc"] = utc_now()


def save_registry(registry: dict[str, Any]) -> None:
    update_totals(registry)
    atomic_json(REGISTRY_PATH, registry)


def verify_preflight(*, require_oracle_parallel: bool = False) -> dict[str, Any]:
    if not PREFLIGHT_PATH.is_file():
        raise CampaignControllerError("Preflight report is missing")
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    if preflight.get("status") != "pass":
        raise CampaignControllerError("Preflight report did not pass")
    required = {
        "cpu_suite",
        "unpruned_parity",
        "k12_b48_parity",
        "k20_b48_parity",
        "b0_generation",
        "two_stage",
        "three_stage",
        "oracle_restart",
        "oracle_index_replay",
        "quality_parallel_soak",
    }
    missing = sorted(key for key in required if not preflight.get("gates", {}).get(key, {}).get("pass"))
    if missing:
        raise CampaignControllerError(f"Preflight gates are missing/failed: {missing}")
    if int(preflight.get("safe_quality_parallel_width", 0)) != 2:
        raise CampaignControllerError("Registered safe quality width is not exactly two")
    if require_oracle_parallel and not preflight.get("gates", {}).get("oracle_parallel_soak", {}).get("pass"):
        raise CampaignControllerError("Oracle two-way concurrency has not passed a semantic soak")
    return preflight


def schedule_arg(schedule: Any) -> str:
    if not isinstance(schedule, list):
        raise CampaignControllerError(f"Cell schedule is unresolved: {schedule!r}")
    return ",".join(
        f"{int(event['boundary'])}:{int(event['budget'])}:{event['selector']}" for event in schedule
    )


def next_attempt(cell: dict[str, Any]) -> int:
    return len(cell.get("attempts") or []) + 1


def build_command(cell_id: str, cell: dict[str, Any]) -> tuple[list[str], Path, int, bool]:
    resume = cell.get("status") == "running" and bool(cell.get("attempts"))
    if resume:
        attempt = int(cell["attempts"][-1]["attempt"])
        run_dir = REPO_ROOT / cell["attempts"][-1]["run_dir"]
        run_name = run_dir.name
    else:
        attempt = next_attempt(cell)
        run_name = f"{cell_id.lower().replace('-', '_')}_attempt{attempt}"
        run_dir = RUN_ROOT / run_name
    command = [
        str(PYTHON),
        str(RUNNER),
        "--cell-id", cell_id,
        "--manifest", str(MANIFEST),
        "--output-root", str(RUN_ROOT),
        "--run-name", run_name,
        "--schedule", schedule_arg(cell.get("schedule")),
        "--seed", str(int(cell.get("seed") or 0)),
    ]
    if cell_id.startswith("O"):
        if cell_id in {"O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8"}:
            command.extend(
                [
                    "--pre-llm-selector", "oracle",
                    "--pre-llm-budget", str(int(cell["pre_llm_budget"])),
                    "--oracle-teacher-layer", str(int(cell["oracle_teacher_layer"])),
                ]
            )
        elif cell_id == "O9":
            command.extend(["--pre-llm-selector", "random", "--pre-llm-budget", "64"])
        elif cell_id == "O10":
            command.extend(["--pre-llm-selector", "uniform", "--pre-llm-budget", "64"])
    if resume:
        command.append("--resume")
    return command, run_dir, attempt, resume


def accepted_run_is_valid(cell: dict[str, Any]) -> bool:
    if cell.get("status") != "accepted":
        return False
    run_dir_value = cell.get("artifact_paths", {}).get("run_dir")
    if not run_dir_value:
        return False
    run_dir = REPO_ROOT / run_dir_value
    try:
        report = validate_run(run_dir, MANIFEST, expected_rows=1000)
    except Exception:
        return False
    completion = run_dir / "COMPLETED.json"
    return report.get("status") == "pass" and completion.is_file()


def mark_running(
    registry: dict[str, Any], cell_id: str, command: list[str], run_dir: Path, attempt: int, *, resume: bool
) -> None:
    cell = registry["cells"][cell_id]
    if resume:
        record = cell["attempts"][-1]
        record.setdefault("resume_events", []).append({"resumed_at_utc": utc_now(), "command": shlex.join(command)})
        record["status"] = "running"
        cell["status"] = "running"
        save_registry(registry)
        return
    record = {
        "attempt": attempt,
        "status": "running",
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "command": shlex.join(command),
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "exit_code": None,
        "failure_reason": None,
    }
    cell.setdefault("attempts", []).append(record)
    cell["status"] = "running"
    save_registry(registry)


def pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
        if value <= 0:
            return False
        os.kill(value, 0)
        return True
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def finalize_attempt(
    registry: dict[str, Any], cell_id: str, process: subprocess.Popen[str], run_dir: Path, log_path: Path
) -> bool:
    cell = registry["cells"][cell_id]
    attempt = cell["attempts"][-1]
    attempt["finished_at_utc"] = utc_now()
    attempt["exit_code"] = int(process.returncode)
    attempt["stdout_log"] = str(log_path.relative_to(REPO_ROOT))
    if process.returncode != 0:
        attempt["status"] = "rejected"
        attempt["failure_reason"] = f"runner_exit_{process.returncode}"
        cell["status"] = "rejected"
        cell["failure_reason"] = attempt["failure_reason"]
        save_registry(registry)
        return False
    try:
        report = validate_run(run_dir, MANIFEST, expected_rows=1000)
        completion_path = run_dir / "COMPLETED.json"
        if not completion_path.is_file():
            raise CampaignControllerError("Runner exited zero without completion sentinel")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except Exception as exc:
        attempt["status"] = "rejected"
        attempt["failure_reason"] = f"independent_validation:{type(exc).__name__}:{exc}"
        cell["status"] = "rejected"
        cell["failure_reason"] = attempt["failure_reason"]
        save_registry(registry)
        return False
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    attempt["status"] = "accepted"
    cell.update(
        {
            "status": "accepted",
            "failure_reason": None,
            "accepted_attempt": attempt["attempt"],
            "source_commit": config["source/git_commit"],
            "manifest_sha256": report["manifest_sha256"],
            "config_sha256": report["config_sha256"],
            "requested_schedule": config["schedule"],
            "realized_visual_tokens_by_layer": json.loads(
                next((run_dir / "predictions.jsonl").open("r", encoding="utf-8"))
            )["realized_visual_tokens_by_layer"],
            "candidate_average_visual_tokens": report["metrics"]["tokens/candidate_average_visual_tokens"],
            "oracle_teacher_visual_token_layer_work": report["metrics"]["tokens/oracle_teacher_visual_token_layer_work_mean"],
            "metrics": report["metrics"],
            "validation": report["checks"],
            "completion_hashes": completion,
            "artifact_paths": {
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "predictions": str((run_dir / "predictions.jsonl").relative_to(REPO_ROOT)),
                "config": str((run_dir / "config.json").relative_to(REPO_ROOT)),
                "metrics": str((run_dir / "metrics.json").relative_to(REPO_ROOT)),
                "integrity": str((run_dir / "integrity_report.json").relative_to(REPO_ROOT)),
                "completion": str(completion_path.relative_to(REPO_ROOT)),
                "oracle_indices": (
                    str((run_dir / "oracle_indices.jsonl").relative_to(REPO_ROOT))
                    if (run_dir / "oracle_indices.jsonl").is_file()
                    else None
                ),
            },
        }
    )
    save_registry(registry)
    return True


def launch_cells(registry: dict[str, Any], cell_ids: list[str], *, parallel_width: int) -> None:
    if parallel_width not in {1, 2}:
        raise CampaignControllerError("Only measured widths one and two are allowed")
    queue: list[str] = []
    for cell_id in cell_ids:
        cell = registry["cells"][cell_id]
        if cell.get("status") == "accepted":
            if not accepted_run_is_valid(cell):
                raise CampaignControllerError(f"Accepted cell {cell_id} no longer validates")
            continue
        queue.append(cell_id)
    active: dict[str, tuple[subprocess.Popen[str], Path, Any, Path]] = {}
    stop_launching = False
    while queue or active:
        while queue and len(active) < parallel_width and not stop_launching:
            cell_id = queue.pop(0)
            command, run_dir, attempt, resume = build_command(cell_id, registry["cells"][cell_id])
            if resume and pid_alive(registry["cells"][cell_id]["attempts"][-1].get("pid")):
                raise CampaignControllerError(f"Refusing to duplicate active worker for {cell_id}")
            log_path = CAMPAIGN_ROOT / "controller_logs" / f"{run_dir.name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a" if resume else "w", encoding="utf-8")
            mark_running(registry, cell_id, command, run_dir, attempt, resume=resume)
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            registry["cells"][cell_id]["attempts"][-1]["pid"] = process.pid
            save_registry(registry)
            active[cell_id] = (process, run_dir, log_handle, log_path)
            print(f"Launched {cell_id} pid={process.pid} attempt={attempt}", flush=True)
        if not active:
            break
        time.sleep(2.0)
        for cell_id in list(active):
            process, run_dir, log_handle, log_path = active[cell_id]
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            del active[cell_id]
            accepted = finalize_attempt(registry, cell_id, process, run_dir, log_path)
            print(f"Finished {cell_id} accepted={accepted} exit={return_code}", flush=True)
            if not accepted:
                stop_launching = True
        if stop_launching and not active:
            raise CampaignControllerError("A cell failed; no further jobs were launched")


def near_best_select(metrics: dict[str, dict[str, float]], *, count: int = 2) -> tuple[list[str], list[dict[str, Any]]]:
    remaining = dict(metrics)
    selected: list[str] = []
    trace: list[dict[str, Any]] = []
    while remaining and len(selected) < count:
        highest = max(row["acc_at_0_5"] for row in remaining.values())
        near = {
            cell_id: row
            for cell_id, row in remaining.items()
            if highest - row["acc_at_0_5"] <= 0.005 + 1e-12
        }
        winner = sorted(
            near,
            key=lambda cell_id: (
                near[cell_id]["average_visual_tokens"],
                -near[cell_id]["mean_iou"],
                cell_id,
            ),
        )[0]
        trace.append(
            {
                "round": len(selected) + 1,
                "highest_acc_at_0_5": highest,
                "near_best_ids": sorted(near),
                "winner": winner,
                "metrics": remaining,
            }
        )
        selected.append(winner)
        del remaining[winner]
    return selected, trace


def decide_campaign1(registry: dict[str, Any]) -> None:
    if not all(registry["cells"][cell_id]["status"] == "accepted" for cell_id in campaign1_ids()):
        raise CampaignControllerError("Campaign 1 decision requires all 14 accepted cells")
    reference = registry["reference"]["discovery_recomputed"]
    mapping = [(0, "T5"), (4, "T4"), (8, "T3"), (16, "T2"), (32, "T1")]
    decision_rows: list[dict[str, Any]] = []
    selected = 48
    for budget, cell_id in mapping:
        metrics = registry["cells"][cell_id]["metrics"]
        acc_loss = reference["acc_at_0_5"] - metrics["quality/acc_at_0_5"]
        miou_loss = reference["mean_iou"] - metrics["quality/mean_iou"]
        passes = acc_loss <= 0.01 + 1e-12 and miou_loss <= 0.005 + 1e-12
        decision_rows.append(
            {
                "budget": budget,
                "cell_id": cell_id,
                "acc_loss": acc_loss,
                "miou_loss": miou_loss,
                "passes": passes,
            }
        )
        if passes and selected == 48:
            selected = budget
    early64 = math.floor((64 * 32 - 12 * selected) / 20)
    registry["decisions"]["B_tail"] = {
        "value": selected,
        "provisional": True,
        "threshold_rows": decision_rows,
        "reference_id": "REF_K20_B48",
    }
    registry["decisions"]["B_early64"] = {
        "value": early64,
        "formula": "floor((64*32 - 12*B_tail)/20)",
        "candidate_average": (20 * early64 + 12 * selected) / 32,
    }
    save_registry(registry)
    decision_path = CAMPAIGN_ROOT / "decisions/campaign1_tail.json"
    atomic_json(decision_path, registry["decisions"]["B_tail"])


def decide_progressive_base(registry: dict[str, Any]) -> list[str]:
    ids = [f"P{index}" for index in range(1, 7)]
    if not all(registry["cells"][cell_id]["status"] == "accepted" for cell_id in ids):
        raise CampaignControllerError("P1-P6 must all be accepted before selection")
    metrics = {
        cell_id: {
            "acc_at_0_5": registry["cells"][cell_id]["metrics"]["quality/acc_at_0_5"],
            "mean_iou": registry["cells"][cell_id]["metrics"]["quality/mean_iou"],
            "average_visual_tokens": registry["cells"][cell_id]["candidate_average_visual_tokens"],
        }
        for cell_id in ids
    }
    selected, trace = near_best_select(metrics, count=2)
    b_tail = int(registry["decisions"]["B_tail"]["value"])
    for output_id, base_id in zip(("P7", "P8"), selected, strict=True):
        first = registry["cells"][base_id]["schedule"][0]
        registry["cells"][output_id]["schedule"] = [
            dict(first),
            {"boundary": 12, "budget": 96, "selector": "query_topk"},
            {"boundary": 20, "budget": b_tail, "selector": "query_topk" if b_tail else "none"},
        ]
        registry["cells"][output_id]["refines"] = base_id
    decision = {"selected_base_ids": selected, "selection_trace": trace, "B_tail": b_tail}
    registry["decisions"]["progressive_base"] = decision
    save_registry(registry)
    atomic_json(CAMPAIGN_ROOT / "decisions/progressive_base.json", decision)
    return selected


def decide_progressive_refinement(registry: dict[str, Any]) -> str:
    if not all(registry["cells"][cell_id]["status"] == "accepted" for cell_id in ("P7", "P8")):
        raise CampaignControllerError("P7/P8 must be accepted before P9 selection")
    metrics = {
        cell_id: {
            "acc_at_0_5": registry["cells"][cell_id]["metrics"]["quality/acc_at_0_5"],
            "mean_iou": registry["cells"][cell_id]["metrics"]["quality/mean_iou"],
            "average_visual_tokens": registry["cells"][cell_id]["candidate_average_visual_tokens"],
        }
        for cell_id in ("P7", "P8")
    }
    selected, trace = near_best_select(metrics, count=1)
    strongest = selected[0]
    schedule = [dict(event) for event in registry["cells"][strongest]["schedule"]]
    b_tail = int(registry["decisions"]["B_tail"]["value"])
    if b_tail > 0:
        schedule[-1] = {"boundary": 20, "budget": b_tail, "selector": "random"}
        contrast = "query_topk_vs_random_after_same_progressive_prefix"
    else:
        schedule[-1] = {"boundary": 20, "budget": 4, "selector": "query_topk"}
        contrast = "zero_tail_vs_minimal_nonzero_tail"
    registry["cells"]["P9"]["schedule"] = schedule
    registry["cells"]["P9"]["compares_to"] = strongest
    decision = {"strongest_refinement": strongest, "selection_trace": trace, "P9_contrast": contrast}
    registry["decisions"]["progressive_refinement"] = decision
    save_registry(registry)
    atomic_json(CAMPAIGN_ROOT / "decisions/progressive_refinement.json", decision)
    return strongest


def configure_oracle_dynamic_cells(registry: dict[str, Any]) -> None:
    b_tail = int(registry["decisions"]["B_tail"]["value"])
    early64 = int(registry["decisions"]["B_early64"]["value"])
    selector = "query_topk" if b_tail else "none"
    for cell_id in ("O7", "O8"):
        registry["cells"][cell_id]["pre_llm_budget"] = early64
        registry["cells"][cell_id]["schedule"] = [
            {"boundary": 20, "budget": b_tail, "selector": selector}
        ]
    save_registry(registry)


def campaign1_ids() -> list[str]:
    return ["T1", "T2", "T3", "T4", "T5"] + [
        f"R{budget}-S{seed}" for budget in (16, 8, 4) for seed in (0, 1, 2)
    ]


def run_campaign1(registry: dict[str, Any]) -> None:
    verify_preflight()
    launch_cells(registry, campaign1_ids(), parallel_width=2)
    decide_campaign1(registry)
    from scripts.analyse_progressive_oracle_campaign import refresh
    refresh(1)


def run_campaign2(registry: dict[str, Any]) -> None:
    verify_preflight()
    if registry["decisions"].get("B_tail") is None:
        raise CampaignControllerError("Campaign 2 requires the registered Campaign 1 tail decision")
    launch_cells(registry, [f"P{index}" for index in range(1, 7)], parallel_width=2)
    decide_progressive_base(registry)
    launch_cells(registry, ["P7", "P8"], parallel_width=2)
    decide_progressive_refinement(registry)
    launch_cells(registry, ["P9"], parallel_width=1)
    from scripts.analyse_progressive_oracle_campaign import refresh
    refresh(2)


def run_campaign3(registry: dict[str, Any], *, oracle_parallel: bool) -> None:
    verify_preflight(require_oracle_parallel=oracle_parallel)
    if registry["decisions"].get("B_tail") is None:
        raise CampaignControllerError("Campaign 3 requires the registered Campaign 1 tail decision")
    configure_oracle_dynamic_cells(registry)
    width = 2 if oracle_parallel else 1
    launch_cells(registry, [f"O{index}" for index in range(1, 11)], parallel_width=width)
    from scripts.analyse_progressive_oracle_campaign import refresh
    refresh(3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=["1", "2", "3", "all"], required=True)
    parser.add_argument("--oracle-parallel", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if not PYTHON.is_file():
            raise CampaignControllerError(f"Pinned Python environment missing: {PYTHON}")
        if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA256:
            raise CampaignControllerError("Frozen discovery manifest hash changed")
        registry = load_registry()
        if args.campaign in {"1", "all"}:
            run_campaign1(registry)
        if args.campaign in {"2", "all"}:
            registry = load_registry()
            run_campaign2(registry)
        if args.campaign in {"3", "all"}:
            registry = load_registry()
            run_campaign3(registry, oracle_parallel=args.oracle_parallel)
    except (CampaignControllerError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
