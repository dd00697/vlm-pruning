#!/usr/bin/env python
"""Execute the registered adaptive fixed-method state machine end to end."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_progressive_oracle_run import validate_run  # noqa: E402


CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
RUN_ROOT = CAMPAIGN_ROOT / "runs"
LOG_ROOT = CAMPAIGN_ROOT / "logs"
REJECTED_ROOT = CAMPAIGN_ROOT / "rejected_runs"
ANALYSIS_ROOT = CAMPAIGN_ROOT / "analysis"
STATE_PATH = CAMPAIGN_ROOT / "CONTROLLER_STATE.json"
REGISTRY_PATH = CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"
REPORT_PATH = CAMPAIGN_ROOT / "RESEARCH_REPORT.md"
MANIFEST = REPO_ROOT / (
    "experiments/progressive_oracle_refcoco_val_20260814/"
    "manifests/refcoco_val_discovery1000.jsonl"
)
PYTHON = "/workspace/.followup_venv/bin/python"
EXPECTED_ROWS = 1000
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260815

SCHEDULES: dict[str, str] = {
    "C448": "4:448:query_topk,12:48:query_topk",
    "C480": "4:480:query_topk,12:48:query_topk",
    "D1": "8:384:query_topk,12:48:query_topk",
    "M448": "4:448:query_topk_merge,12:48:query_topk",
    "B1": "4:384:query_topk,12:96:query_topk,20:16:query_topk",
    "A1": "4:384:query_topk,12:96:query_topk,20:48:query_topk",
}
AVERAGES = {"C448": 214.0, "C480": 222.0, "D1": 222.0, "M448": 214.0, "B1": 198.0, "A1": 210.0}
HISTORICAL = {
    "P3": {"acc_at_0_5": 0.408, "mean_iou": 0.4246506509420894, "parse_rate": 1.0, "average_visual_tokens": 198.0, "boundaries": 2},
    "P5": {"acc_at_0_5": 0.406, "mean_iou": 0.41979018313550226, "parse_rate": 1.0, "average_visual_tokens": 186.0, "boundaries": 2},
    "P8": {"acc_at_0_5": 0.401, "mean_iou": 0.41791704893558784, "parse_rate": 1.0, "average_visual_tokens": 192.0, "boundaries": 3},
}
REFERENCE_PATHS = {
    "K12_ORIGINAL": ANALYSIS_ROOT / "rescored/K12_ORIGINAL.jsonl",
    "P3": ANALYSIS_ROOT / "rescored/P3.jsonl",
    "P5": ANALYSIS_ROOT / "rescored/P5.jsonl",
    "P8": ANALYSIS_ROOT / "rescored/P8.jsonl",
}


class FixedCampaignError(RuntimeError):
    """Raised when fixed-method execution or evidence violates the goal."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FixedCampaignError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != EXPECTED_ROWS or not all(isinstance(row, dict) for row in rows):
        raise FixedCampaignError(f"Expected {EXPECTED_ROWS} JSON objects: {path}")
    return rows


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def save_state(state: dict[str, Any], registry: dict[str, Any]) -> None:
    now = utc_now()
    state["updated_at_utc"] = now
    registry["updated_at_utc"] = now
    atomic_json(STATE_PATH, state)
    atomic_json(REGISTRY_PATH, registry)


def normalized_metrics(validation: dict[str, Any]) -> dict[str, Any]:
    metrics = validation["metrics"]
    return {
        "rows": int(metrics["samples/evaluated"]),
        "acc_at_0_5": float(metrics["quality/acc_at_0_5"]),
        "mean_iou": float(metrics["quality/mean_iou"]),
        "parse_rate": float(metrics["quality/parsed_box_rate"]),
        "average_visual_tokens": float(metrics["tokens/candidate_average_visual_tokens"]),
        "peak_allocated_mb": metrics["efficiency/peak_allocated_mb"],
        "peak_reserved_mb": metrics["efficiency/peak_reserved_mb"],
    }


def append_decision_once(registry: dict[str, Any], event: dict[str, Any]) -> None:
    decision = str(event.get("decision") or "")
    if not decision:
        raise FixedCampaignError("Decision event is missing its decision name")
    for item in registry["decisions"]:
        if str(item.get("decision") or "") != decision:
            continue
        previous = {key: value for key, value in item.items() if key != "at_utc"}
        proposed = {key: value for key, value in event.items() if key != "at_utc"}
        if previous != proposed:
            raise FixedCampaignError(f"Decision {decision} conflicts with its persisted evidence")
        return
    registry["decisions"].append(event)


def k12_like(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics["acc_at_0_5"]) >= 0.429
        and float(metrics["mean_iou"]) >= 0.4165978636
        and float(metrics["average_visual_tokens"]) < 246.0
    )


def promising(candidate: dict[str, Any], reference: dict[str, Any]) -> bool:
    return (
        float(candidate["acc_at_0_5"]) - float(reference["acc_at_0_5"]) >= 0.010 - 1e-12
        and float(candidate["mean_iou"]) - float(reference["mean_iou"]) >= -0.005 - 1e-12
    )


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return process.wait()


def record_rejected(cell_id: str, attempt: dict[str, Any]) -> None:
    destination = REJECTED_ROOT / f"{cell_id.lower()}_attempt{attempt['attempt']}" / "ATTEMPT.json"
    atomic_json(destination, attempt)


def active_runner_processes(run_name: str, *, proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            raw = (process_dir / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
        if not any(Path(part).name == "run_progressive_oracle_refcoco.py" for part in command):
            continue
        try:
            registered_name = command[command.index("--run-name") + 1]
        except (ValueError, IndexError):
            continue
        if registered_name == run_name:
            matches.append({"pid": int(process_dir.name), "command": command})
    return sorted(matches, key=lambda item: int(item["pid"]))


def validated_completed_attempt(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    validation = validate_run(run_dir, MANIFEST, expected_rows=EXPECTED_ROWS)
    metrics = normalized_metrics(validation)
    completion = read_json(run_dir / "COMPLETED.json")
    expected = {
        "status": "complete",
        "rows": EXPECTED_ROWS,
        "manifest_sha256": sha256_file(MANIFEST),
        "source_git_commit": validation["source_git_commit"],
        "config_sha256": sha256_file(run_dir / "config.json"),
        "predictions_sha256": sha256_file(run_dir / "predictions.jsonl"),
        "metrics_sha256": sha256_file(run_dir / "metrics.json"),
        "integrity_sha256": sha256_file(run_dir / "integrity_report.json"),
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise FixedCampaignError(
                f"Completed physical attempt has invalid {key}: {completion.get(key)!r} != {value!r}"
            )
    evidence = {
        "run_dir": relative(run_dir),
        "config_sha256": expected["config_sha256"],
        "predictions_sha256": expected["predictions_sha256"],
        "metrics_sha256": expected["metrics_sha256"],
        "integrity_sha256": expected["integrity_sha256"],
        "completion_sha256": sha256_file(run_dir / "COMPLETED.json"),
    }
    return metrics, evidence


def recover_stale_workers(
    state: dict[str, Any],
    registry: dict[str, Any],
    *,
    proc_root: Path = Path("/proc"),
) -> list[dict[str, Any]]:
    workers = list(state.get("active_workers") or [])
    if not workers:
        return []
    recovered: list[dict[str, Any]] = []
    for cell_id in workers:
        cell = registry.get("cells", {}).get(cell_id)
        if not isinstance(cell, dict) or not cell.get("attempts"):
            raise FixedCampaignError(f"Stale worker {cell_id} has no registered attempt")
        attempt = cell["attempts"][-1]
        run_dir = REPO_ROOT / attempt["run_dir"]
        run_name = run_dir.name
        live = active_runner_processes(run_name, proc_root=proc_root)
        if live:
            raise FixedCampaignError(f"Registered worker {cell_id} is still live: {live}")
        if not (run_dir / "COMPLETED.json").is_file():
            partial = run_dir / "predictions.partial.jsonl"
            suffix = "a resumable partial" if partial.is_file() else "no resumable partial"
            raise FixedCampaignError(
                f"Worker {cell_id} is absent but has no completed physical artifact ({suffix}); "
                "resume it under its immutable source epoch before restarting the controller"
            )
        metrics, evidence = validated_completed_attempt(run_dir)
        if metrics["average_visual_tokens"] != AVERAGES[cell_id]:
            raise FixedCampaignError(
                f"Recovered {cell_id} realized average {metrics['average_visual_tokens']} != {AVERAGES[cell_id]}"
            )
        previously_accepted = cell.get("status") == "accepted"
        recovery_time = utc_now()
        attempt.update(
            {
                "status": "accepted",
                "finished_at_utc": attempt.get("finished_at_utc")
                or read_json(run_dir / "COMPLETED.json").get("completed_at_utc")
                or recovery_time,
                "exit_status": 0,
                "recovered_at_utc": recovery_time,
                "recovery_reason": "registered process absent; completed attempt independently validated",
            }
        )
        resume_command = f"{attempt['command']} --resume"
        if resume_command not in attempt.setdefault("resume_commands", []):
            attempt["resume_commands"].append(resume_command)
        cell["status"] = "accepted"
        cell["metrics"] = metrics
        cell["accepted_run"] = evidence
        if not previously_accepted:
            registry["totals"]["accepted_validation_quality"] += 1
        event = {
            "at_utc": recovery_time,
            "decision": f"recover_completed_stale_worker_{cell_id}",
            "cell_id": cell_id,
            "process_scan": {"matching_processes": [], "proc_root": str(proc_root)},
            "evidence": evidence,
            "metrics": metrics,
        }
        append_decision_once(registry, event)
        recovered.append(event)
    state["active_workers"] = []
    state["state"] = "RECOVERED_COMPLETED_STALE_WORKERS"
    state["next_required_action"] = "continue_adaptive_fixed_state_machine"
    state["decision_history"].append(
        {
            "phase": "stale_worker_recovery",
            "status": "complete",
            "cells": [event["cell_id"] for event in recovered],
            "process_absence_proved": True,
            "evidence": [event["evidence"] for event in recovered],
        }
    )
    save_state(state, registry)
    return recovered


def run_cell(cell_id: str, state: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    cell = registry["cells"][cell_id]
    if cell.get("status") == "accepted":
        run_dir = REPO_ROOT / cell["accepted_run"]["run_dir"]
        return normalized_metrics(validate_run(run_dir, MANIFEST, expected_rows=EXPECTED_ROWS))
    if cell.get("status") == "skipped":
        raise FixedCampaignError(f"Attempted to execute skipped cell {cell_id}")
    cell["schedule"] = SCHEDULES[cell_id]
    cell["registered_average_visual_tokens"] = AVERAGES[cell_id]
    for _ in range(3):
        resume = False
        if cell.get("status") == "running" and cell.get("attempts"):
            attempt = cell["attempts"][-1]
            run_dir = REPO_ROOT / attempt["run_dir"]
            resume = (run_dir / "predictions.partial.jsonl").is_file()
            if not resume:
                attempt["status"] = "failed"
                attempt["failure"] = "running registry entry had no resumable manifest-prefix partial"
                attempt["finished_at_utc"] = utc_now()
                record_rejected(cell_id, attempt)
                cell["status"] = "pending"
        if cell.get("status") != "running":
            attempt_number = len(cell.get("attempts") or []) + 1
            run_name = f"{cell_id.lower()}_attempt{attempt_number}"
            run_dir = RUN_ROOT / run_name
            log_path = LOG_ROOT / f"{run_name}.log"
            command = [
                PYTHON,
                "scripts/run_progressive_oracle_refcoco.py",
                "--cell-id",
                cell_id,
                "--manifest",
                str(MANIFEST),
                "--output-root",
                str(RUN_ROOT),
                "--run-name",
                run_name,
                "--schedule",
                SCHEDULES[cell_id],
                "--seed",
                "0",
            ]
            attempt = {
                "attempt": attempt_number,
                "status": "running",
                "started_at_utc": utc_now(),
                "run_dir": relative(run_dir),
                "log_path": relative(log_path),
                "command": shlex.join(command),
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
                ).strip(),
            }
            cell.setdefault("attempts", []).append(attempt)
            cell["status"] = "running"
        else:
            attempt = cell["attempts"][-1]
            run_dir = REPO_ROOT / attempt["run_dir"]
            log_path = REPO_ROOT / attempt["log_path"]
            command = shlex.split(attempt["command"])
            if resume:
                command.append("--resume")
                attempt.setdefault("resume_commands", []).append(shlex.join(command))
        state["active_workers"] = [cell_id]
        state["state"] = f"RUNNING_{cell_id}"
        state["next_required_action"] = f"complete_and_validate_{cell_id}"
        save_state(state, registry)
        status = run_command(command, log_path)
        state["active_workers"] = []
        if status == 0:
            try:
                validation = validate_run(run_dir, MANIFEST, expected_rows=EXPECTED_ROWS)
                metrics = normalized_metrics(validation)
                if metrics["average_visual_tokens"] != AVERAGES[cell_id]:
                    raise FixedCampaignError(
                        f"{cell_id} realized average {metrics['average_visual_tokens']} != {AVERAGES[cell_id]}"
                    )
                attempt.update({"status": "accepted", "finished_at_utc": utc_now(), "exit_status": 0})
                cell["status"] = "accepted"
                cell["metrics"] = metrics
                cell["accepted_run"] = {
                    "run_dir": relative(run_dir),
                    "config_sha256": sha256_file(run_dir / "config.json"),
                    "predictions_sha256": sha256_file(run_dir / "predictions.jsonl"),
                    "metrics_sha256": sha256_file(run_dir / "metrics.json"),
                    "integrity_sha256": sha256_file(run_dir / "integrity_report.json"),
                    "completion_sha256": sha256_file(run_dir / "COMPLETED.json"),
                }
                registry["totals"]["accepted_validation_quality"] += 1
                append_decision_once(
                    registry,
                    {
                        "at_utc": utc_now(),
                        "decision": f"accept_{cell_id}",
                        "evidence": cell["accepted_run"],
                        "metrics": metrics,
                    },
                )
                save_state(state, registry)
                return metrics
            except Exception as exc:
                status = 2
                attempt["validation_failure"] = f"{type(exc).__name__}: {exc}"
        attempt.update({"status": "failed", "finished_at_utc": utc_now(), "exit_status": status})
        cell["status"] = "pending"
        record_rejected(cell_id, attempt)
        registry["failures"].append(
            {"cell_id": cell_id, "attempt": attempt["attempt"], "at_utc": utc_now(), "evidence": relative(REJECTED_ROOT / f"{cell_id.lower()}_attempt{attempt['attempt']}" / "ATTEMPT.json")}
        )
        save_state(state, registry)
    raise FixedCampaignError(f"{cell_id} failed three bounded attempts")


def skip(cell_id: str, reason: str, registry: dict[str, Any]) -> None:
    cell = registry["cells"][cell_id]
    if cell.get("status") == "accepted":
        return
    cell.update({"status": "skipped", "skip_reason": reason, "skipped_at_utc": utc_now()})


def candidate_metrics(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {name: dict(metrics) for name, metrics in HISTORICAL.items()}
    for cell_id in SCHEDULES:
        cell = registry["cells"][cell_id]
        if cell.get("status") == "accepted":
            result[cell_id] = {
                **cell["metrics"],
                "boundaries": len(SCHEDULES[cell_id].split(",")),
            }
    return result


def rank_candidates(candidates: dict[str, dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    if not candidates:
        raise FixedCampaignError("Candidate ranking received an empty pool")
    highest = max(float(metrics["acc_at_0_5"]) for metrics in candidates.values())
    near = {
        name: metrics
        for name, metrics in candidates.items()
        if highest - float(metrics["acc_at_0_5"]) <= 0.005 + 1e-12
    }
    ordered = sorted(
        near,
        key=lambda name: (
            float(near[name]["average_visual_tokens"]),
            -float(near[name]["mean_iou"]),
            int(near[name].get("boundaries", 99)),
            name,
        ),
    )
    trace = [
        {
            "candidate": name,
            "acc_at_0_5": candidates[name]["acc_at_0_5"],
            "mean_iou": candidates[name]["mean_iou"],
            "average_visual_tokens": candidates[name]["average_visual_tokens"],
            "within_0_005_of_highest_acc": name in near,
        }
        for name in sorted(candidates, key=lambda item: (-float(candidates[item]["acc_at_0_5"]), item))
    ]
    return ordered[0], trace


def route_pool(registry: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    candidates = candidate_metrics(registry)
    new_eligible = {
        name: metrics
        for name, metrics in candidates.items()
        if name in SCHEDULES
        and float(metrics["average_visual_tokens"]) <= 222.0
        and float(metrics["mean_iou"]) >= 0.4165978636
    }
    if not new_eligible:
        return "P5", [{"candidate": "P5", "reason": "registered_no_new_eligible_fallback"}]
    eligible = {
        name: metrics
        for name, metrics in candidates.items()
        if float(metrics["average_visual_tokens"]) <= 222.0
        and float(metrics["mean_iou"]) >= 0.4165978636
    }
    return rank_candidates(eligible)


def paired_bootstrap(left: np.ndarray, right: np.ndarray, *, seed: int, scale: float) -> dict[str, Any]:
    if left.shape != right.shape or left.shape != (EXPECTED_ROWS,):
        raise FixedCampaignError("Paired bootstrap arrays are not aligned to the frozen 1,000 rows")
    difference = left - right
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    for start in range(0, BOOTSTRAPS, 500):
        count = min(500, BOOTSTRAPS - start)
        indices = rng.integers(0, EXPECTED_ROWS, size=(count, EXPECTED_ROWS))
        means.append(difference[indices].mean(axis=1) * scale)
    samples = np.concatenate(means)
    return {
        "point_delta": float(difference.mean() * scale),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "bootstrap_resamples": BOOTSTRAPS,
        "seed": seed,
    }


def rows_for(name: str, registry: dict[str, Any]) -> list[dict[str, Any]]:
    if name in REFERENCE_PATHS:
        return read_jsonl(REFERENCE_PATHS[name])
    cell = registry["cells"][name]
    return read_jsonl(REPO_ROOT / cell["accepted_run"]["run_dir"] / "predictions.jsonl")


def contrast(left_name: str, right_name: str, registry: dict[str, Any], seed_offset: int) -> dict[str, Any]:
    left = rows_for(left_name, registry)
    right = rows_for(right_name, registry)
    left_ids = [str(row["sample_id"]) for row in left]
    if left_ids != [str(row["sample_id"]) for row in right]:
        raise FixedCampaignError(f"Contrast {left_name}-{right_name} is not sample-aligned")
    left_acc = np.asarray([float(bool(row["success_iou_0_5"])) for row in left])
    right_acc = np.asarray([float(bool(row["success_iou_0_5"])) for row in right])
    left_iou = np.asarray([float(row["box_iou"]) for row in left])
    right_iou = np.asarray([float(row["box_iou"]) for row in right])
    left_parse = np.asarray([float(bool(row.get("valid", True))) for row in left])
    right_parse = np.asarray([float(bool(row.get("valid", True))) for row in right])
    left_metrics = candidate_metrics(registry).get(left_name) or registry["references"][left_name]
    right_metrics = candidate_metrics(registry).get(right_name) or registry["references"][right_name]
    return {
        "contrast": f"{left_name}-minus-{right_name}",
        "left": left_name,
        "right": right_name,
        "acc_at_0_5_percentage_points": paired_bootstrap(
            left_acc, right_acc, seed=BOOTSTRAP_SEED + seed_offset, scale=100.0
        ),
        "mean_iou": paired_bootstrap(
            left_iou, right_iou, seed=BOOTSTRAP_SEED + 100 + seed_offset, scale=1.0
        ),
        "discordant": {
            "left_correct_right_wrong": int(np.sum((left_acc == 1) & (right_acc == 0))),
            "left_wrong_right_correct": int(np.sum((left_acc == 0) & (right_acc == 1))),
        },
        "parse_rate_delta": float(left_parse.mean() - right_parse.mean()),
        "average_visual_tokens_delta": float(left_metrics["average_visual_tokens"])
        - float(right_metrics["average_visual_tokens"]),
        "visual_token_layer_work_delta": 32.0
        * (float(left_metrics["average_visual_tokens"]) - float(right_metrics["average_visual_tokens"])),
    }


def analyze(registry: dict[str, Any], aggressive_route: str, ranking: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [name for name in SCHEDULES if registry["cells"][name].get("status") == "accepted"]
    pairs: list[tuple[str, str]] = [("C448", "P3")]
    if "C480" in accepted:
        pairs.append(("C480", "C448"))
    if "D1" in accepted and "C480" in accepted:
        pairs.extend([("D1", "C480"), ("D1", "P3")])
    if "M448" in accepted:
        pairs.append(("M448", "C448"))
    if "B1" in accepted:
        pairs.append(("B1", "P3"))
    if "A1" in accepted:
        pairs.append(("A1", "B1"))
    pairs.extend((name, "K12_ORIGINAL") for name in accepted)
    seen: set[tuple[str, str]] = set()
    contrasts = []
    for left, right in pairs:
        if (left, right) not in seen:
            contrasts.append(contrast(left, right, registry, len(contrasts)))
            seen.add((left, right))
    return {
        "status": "complete",
        "created_at_utc": utc_now(),
        "accepted_cells": accepted,
        "cell_metrics": {name: registry["cells"][name]["metrics"] for name in accepted},
        "skipped_cells": {
            name: registry["cells"][name].get("skip_reason")
            for name in SCHEDULES
            if registry["cells"][name].get("status") == "skipped"
        },
        "paired_contrasts": contrasts,
        "aggressive_route": aggressive_route,
        "aggressive_route_ranking": ranking,
        "a0_subgroup_evidence_path": relative(ANALYSIS_ROOT / "A0_REFERENCE_AND_ERROR_ANALYSIS.json"),
    }


def update_report(registry: dict[str, Any], analysis: dict[str, Any]) -> None:
    metrics = candidate_metrics(registry)
    lines = [
        "## Adaptive fixed-method experiments",
        "",
        "Every listed result is an independently finalized physical run on the same frozen 1,000-example validation subset. Branches marked skipped were not executed because their registered parent gate did not trigger.",
        "",
        "| Cell | Schedule | Status | Acc@0.5 | mIoU | Parse | Avg visual tokens | K12-like screen |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for name in SCHEDULES:
        cell = registry["cells"][name]
        if cell.get("status") == "accepted":
            item = cell["metrics"]
            lines.append(
                f"| {name} | `{SCHEDULES[name]}` | accepted | {item['acc_at_0_5']:.4f} | {item['mean_iou']:.4f} | {item['parse_rate']:.4f} | {item['average_visual_tokens']:.1f} | {'pass' if k12_like(item) else 'fail'} |"
            )
        else:
            lines.append(
                f"| {name} | `{SCHEDULES[name]}` | skipped: {cell.get('skip_reason', 'not triggered')} | — | — | — | {AVERAGES[name]:.1f} | — |"
            )
    lines.extend(["", "Registered branch decisions:", ""])
    for decision in registry["decisions"]:
        if str(decision.get("decision", "")).startswith(("fixed_", "trigger_", "select_")):
            lines.append(f"- `{decision['decision']}`: {decision.get('reason', decision.get('selected', ''))}")
    lines.extend(
        [
            "",
            f"The frozen aggressive route for gate-label generation is **{analysis['aggressive_route']}** ({metrics[analysis['aggressive_route']]['average_visual_tokens']:.1f} average visual tokens). It cannot be changed after route outcomes are generated.",
            "",
            "Important paired intervals (10,000 resamples; a confidence interval containing zero is inconclusive):",
            "",
            "| Contrast | Acc delta pp [95% CI] | mIoU delta [95% CI] | Discordant L+/R+ | Token delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in analysis["paired_contrasts"]:
        acc = item["acc_at_0_5_percentage_points"]
        iou = item["mean_iou"]
        discordant = item["discordant"]
        lines.append(
            f"| {item['contrast']} | {acc['point_delta']:.3f} [{acc['lower_95']:.3f}, {acc['upper_95']:.3f}] | {iou['point_delta']:.4f} [{iou['lower_95']:.4f}, {iou['upper_95']:.4f}] | {discordant['left_correct_right_wrong']}/{discordant['left_wrong_right_correct']} | {item['average_visual_tokens_delta']:+.1f} |"
        )
    document = REPORT_PATH.read_text(encoding="utf-8")
    replacement = "\n".join(lines)
    updated, count = re.subn(
        r"## Adaptive fixed-method experiments\n\n.*?(?=\n## Official source-native comparisons)",
        replacement,
        document,
        flags=re.DOTALL,
    )
    if count != 1:
        raise FixedCampaignError("Could not update fixed-method section in cumulative report")
    atomic_text(REPORT_PATH, updated)


def main() -> int:
    state = read_json(STATE_PATH)
    registry = read_json(REGISTRY_PATH)
    if registry["phases"]["preflight"].get("status") != "complete":
        raise FixedCampaignError("Fixed campaign cannot start before passing preflight")
    recover_stale_workers(state, registry)
    if state.get("active_workers"):
        raise FixedCampaignError(f"Controller has unresolved active workers: {state['active_workers']}")

    p3 = HISTORICAL["P3"]
    c448 = run_cell("C448", state, registry)
    if k12_like(c448):
        skip("C480", "C448 passed K12-like; hard expansion was not triggered", registry)
        skip("D1", "C448 passed K12-like; representation-depth control was not triggered", registry)
        append_decision_once(registry, {"at_utc": utc_now(), "decision": "fixed_C448_k12_like", "reason": "skip hard expansion and proceed to hard selection"})
    else:
        acc_delta = c448["acc_at_0_5"] - p3["acc_at_0_5"]
        miou_delta = c448["mean_iou"] - p3["mean_iou"]
        triggered: list[str]
        if acc_delta >= 0.010 - 1e-12 and miou_delta >= -0.005 - 1e-12:
            triggered = ["C480"]
            reason = "C448 improved P3 by >=0.010 Acc with the mIoU guard"
        elif acc_delta < 0.005 - 1e-12:
            triggered = ["D1"]
            reason = "C448 Acc improvement over P3 was <0.005"
        else:
            triggered = ["C480", "D1"]
            reason = "C448 result was ambiguous/discordant under the registered gates"
        append_decision_once(registry, {"at_utc": utc_now(), "decision": "trigger_hard_followup", "triggered": triggered, "reason": reason, "deltas": {"acc": acc_delta, "mean_iou": miou_delta}})
        first_metrics: dict[str, dict[str, Any]] = {}
        for name in triggered:
            first_metrics[name] = run_cell(name, state, registry)
        if len(triggered) == 1:
            name = triggered[0]
            other = "D1" if name == "C480" else "C480"
            item = first_metrics[name]
            if k12_like(item) or promising(item, c448):
                append_decision_once(registry, {"at_utc": utc_now(), "decision": "trigger_matched_hard_control", "triggered": other, "reason": f"{name} passed the matched-control completion rule"})
                run_cell(other, state, registry)
            else:
                skip(other, f"{name} did not pass the matched-control completion rule", registry)
        for name in ("C480", "D1"):
            if registry["cells"][name].get("status") == "pending":
                skip(name, "not triggered by the registered C448 state", registry)

    hard_pool = {
        name: metrics
        for name, metrics in candidate_metrics(registry).items()
        if name in {"P3", "P5", "P8", "C448", "C480", "D1"}
        and metrics["average_visual_tokens"] <= 222.0
        and metrics["mean_iou"] >= 0.4165978636
    }
    strongest_hard, hard_ranking = rank_candidates(hard_pool)
    accepted_hard_pass = any(
        k12_like(registry["cells"][name]["metrics"])
        for name in ("C448", "C480", "D1")
        if registry["cells"][name].get("status") == "accepted"
    )
    append_decision_once(registry, {"at_utc": utc_now(), "decision": "select_strongest_hard", "selected": strongest_hard, "ranking": hard_ranking, "any_new_hard_k12_like": accepted_hard_pass})

    if accepted_hard_pass:
        skip("M448", "an accepted hard-pruning cell passed K12-like", registry)
        skip("B1", "fixed branch terminated at a K12-like hard-pruning cell", registry)
        skip("A1", "fixed branch terminated at a K12-like hard-pruning cell", registry)
    else:
        m448 = run_cell("M448", state, registry)
        merge_evidence = k12_like(m448) or promising(m448, c448)
        append_decision_once(registry, {"at_utc": utc_now(), "decision": "fixed_merge_evidence", "reason": "K12-like or >=0.010 Acc over C448 with mIoU guard" if merge_evidence else "merge evidence gate failed", "passed": merge_evidence})
        if k12_like(m448):
            skip("B1", "M448 passed K12-like", registry)
            skip("A1", "M448 passed K12-like", registry)
        else:
            b1 = run_cell("B1", state, registry)
            if promising(b1, p3):
                append_decision_once(registry, {"at_utc": utc_now(), "decision": "trigger_A1", "reason": "B1 improved P3 by >=0.010 Acc with the mIoU guard"})
                run_cell("A1", state, registry)
            else:
                skip("A1", "B1 did not improve P3 by >=0.010 Acc with the mIoU guard", registry)

    aggressive_route, aggressive_ranking = route_pool(registry)
    registry["route_gate_contract"] = {
        "safe_route": {"name": "K12_QUERY_TOPK", "schedule": "12:48:query_topk", "average_visual_tokens": 246.0},
        "aggressive_route": aggressive_route,
        "aggressive_route_metrics": candidate_metrics(registry)[aggressive_route],
        "aggressive_route_schedule": SCHEDULES.get(aggressive_route, aggressive_route),
        "frozen_at_utc": utc_now(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "ranking": aggressive_ranking,
    }
    append_decision_once(registry, {"at_utc": utc_now(), "decision": "select_and_freeze_aggressive_route", "selected": aggressive_route, "ranking": aggressive_ranking})
    analysis = analyze(registry, aggressive_route, aggressive_ranking)
    analysis_path = ANALYSIS_ROOT / "FIXED_METHOD_RESULTS.json"
    atomic_json(analysis_path, analysis)
    registry["phases"]["fixed_methods"] = {
        "status": "complete",
        "analysis_path": relative(analysis_path),
        "analysis_sha256": sha256_file(analysis_path),
        "aggressive_route": aggressive_route,
    }
    state["state"] = "FIXED_METHODS_COMPLETE_OFFICIAL_AND_GATE_PENDING"
    state["next_required_action"] = "complete_official_source_branch_then_generate_route_labels"
    state["decision_history"].append({"phase": "fixed_methods", "status": "complete", "evidence_path": relative(analysis_path), "evidence_sha256": sha256_file(analysis_path), "aggressive_route": aggressive_route})
    update_report(registry, analysis)
    save_state(state, registry)
    print(json.dumps({"status": "complete", "aggressive_route": aggressive_route, "analysis": relative(analysis_path)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FixedCampaignError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
