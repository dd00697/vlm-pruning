#!/usr/bin/env python
"""Run the terminal consistency audit and close the adaptive campaign."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
REGISTRY_PATH = CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"
STATE_PATH = CAMPAIGN_ROOT / "CONTROLLER_STATE.json"
REPORT_PATH = CAMPAIGN_ROOT / "RESEARCH_REPORT.md"
FINAL_PATH = CAMPAIGN_ROOT / "FINAL_COMPLETION_REPORT.json"
TEST_LOG_PATH = CAMPAIGN_ROOT / "analysis/FINAL_TEST_SUITE.log"
TEST_REPORT_PATH = CAMPAIGN_ROOT / "analysis/FINAL_TEST_SUITE.json"
MANIFEST = REPO_ROOT / (
    "experiments/progressive_oracle_refcoco_val_20260814/"
    "manifests/refcoco_val_discovery1000.jsonl"
)
EXPECTED_MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
EXPECTED_MANIFEST_ROWS = 1000


class FinalAuditError(RuntimeError):
    """Raised when the campaign cannot honestly satisfy its stopping condition."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalAuditError(f"Expected JSON object: {path}")
    return value


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def is_campaign_report_markdown(path: Path) -> bool:
    """Exclude documentation belonging to pinned third-party source checkouts."""
    relative_parts = path.resolve().relative_to(CAMPAIGN_ROOT).parts
    return bool(relative_parts) and relative_parts[0] != "official_sources"


def run_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    atomic_write(TEST_LOG_PATH, process.stdout.encode("utf-8"))
    match = re.search(r"Ran (\d+) tests", process.stdout)
    count = int(match.group(1)) if match else -1
    report = {
        "status": "pass" if process.returncode == 0 and count >= 188 else "fail",
        "completed_at_utc": utc_now(),
        "command": command,
        "exit_status": process.returncode,
        "tests_run": count,
        "minimum_expected_tests": 188,
        "log_path": relative(TEST_LOG_PATH),
        "log_sha256": sha256_file(TEST_LOG_PATH),
    }
    atomic_write(TEST_REPORT_PATH, canonical_bytes(report))
    if report["status"] != "pass":
        raise FinalAuditError(f"Final repository suite failed or ran too few tests: {report}")
    return report


def refresh_report_test_count(count: int) -> None:
    document = REPORT_PATH.read_text(encoding="utf-8")
    updated, replacements = re.subn(
        r"The complete repository suite passed \d+ tests\.",
        f"The complete repository suite passed {count} tests.",
        document,
    )
    if replacements != 1:
        raise FinalAuditError("Could not update the unique repository-test statement in the report")
    atomic_write(REPORT_PATH, updated.encode("utf-8"))


def validate_completion_hashes() -> dict[str, Any]:
    mapping = {
        "config_sha256": "config.json",
        "environment_sha256": "environment.json",
        "predictions_sha256": "predictions.jsonl",
        "labels_sha256": "labels.jsonl",
        "metrics_sha256": "metrics.json",
        "integrity_sha256": "integrity_report.json",
        "integrity_report_sha256": "integrity_report.json",
        "timing_sha256": "timing.jsonl",
        "calibration_sha256": "calibration.json",
    }
    sentinels = sorted(CAMPAIGN_ROOT.rglob("COMPLETED.json"))
    checked = 0
    recognized_hashes = 0
    for sentinel_path in sentinels:
        completion = read_json(sentinel_path)
        if completion.get("status") != "complete":
            raise FinalAuditError(f"Non-complete sentinel: {sentinel_path}")
        run_dir = sentinel_path.parent
        for key, name in mapping.items():
            expected = completion.get(key)
            if expected is None:
                continue
            path = run_dir / name
            if not path.is_file() or sha256_file(path) != expected:
                raise FinalAuditError(f"Completion hash mismatch {sentinel_path}:{key}")
            recognized_hashes += 1
        artifact_hashes = completion.get("artifact_sha256") or {}
        if not isinstance(artifact_hashes, dict):
            raise FinalAuditError(f"Malformed artifact hash map: {sentinel_path}")
        for name, expected in artifact_hashes.items():
            path = run_dir / name
            if not path.is_file() or sha256_file(path) != expected:
                raise FinalAuditError(f"Completion artifact hash mismatch {sentinel_path}:{name}")
            recognized_hashes += 1
        checked += 1
    if checked < 30 or recognized_hashes < 100:
        raise FinalAuditError(
            f"Unexpectedly sparse completion audit: sentinels={checked}, hashes={recognized_hashes}"
        )
    return {"sentinels_checked": checked, "artifact_hashes_checked": recognized_hashes}


def active_campaign_processes() -> list[dict[str, Any]]:
    output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    script_names = (
        "run_adaptive_gate_labels.py",
        "run_adaptive_route_gate.py",
        "run_adaptive_matched_timing_trial.py",
        "run_adaptive_matched_timing_campaign.py",
    )
    own_pid = os.getpid()
    result = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        if int(pid_text) == own_pid:
            continue
        if any(name in command for name in script_names):
            result.append({"pid": int(pid_text), "command": command})
    return result


def active_gpu_processes() -> list[str]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def command_strings(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "command" and isinstance(item, str):
                result.append(item)
            result.extend(command_strings(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(command_strings(item))
    return result


def validate_registry(registry: dict[str, Any]) -> dict[str, Any]:
    required_phases = ("A0", "preflight", "fixed_methods", "official_methods", "route_gate", "timing")
    missing = [name for name in required_phases if registry.get("phases", {}).get(name, {}).get("status") != "complete"]
    if missing:
        raise FinalAuditError(f"Campaign phases are not complete: {missing}")
    accepted = ("C448", "C480", "D1", "M448", "OFF_PDROP_L2_192", "OFF_PDROP_REROUTE_192", "GATE_SEED0", "GATE_SEED1", "GATE_SEED2")
    for cell_id in accepted:
        if registry["cells"].get(cell_id, {}).get("status") != "accepted":
            raise FinalAuditError(f"Required accepted cell is not accepted: {cell_id}")
    terminal = {
        "B1": "skipped",
        "A1": "skipped",
        "OFF_NUWA_192": "skipped",
        "OFF_NUWA_REROUTE_192": "skipped",
    }
    for cell_id, status in terminal.items():
        if registry["cells"].get(cell_id, {}).get("status") != status:
            raise FinalAuditError(f"Required skipped cell is not terminal: {cell_id}")
    for cell_id in ("GATE_RANDOM_RATE_CONTROL", "GATE_PRIVILEGED_ORACLE_CONTROL"):
        if not str(registry["cells"].get(cell_id, {}).get("status", "")).startswith("accepted"):
            raise FinalAuditError(f"Analytical control is not accepted: {cell_id}")
    commands = command_strings(registry)
    forbidden_terms = (
        "--limit 8811",
        "testa",
        "testb",
        "legacy_goal",
        "heldout",
        "transfer",
        "dissertation",
    )
    forbidden = [
        command
        for command in commands
        if any(term in command.lower() for term in forbidden_terms)
    ]
    if forbidden:
        raise FinalAuditError(f"Forbidden command is registered as executed: {forbidden[:1]}")
    return {
        "required_phases_complete": list(required_phases),
        "accepted_quality_cells": list(accepted),
        "terminal_skipped_cells": terminal,
        "registered_commands_checked": len(commands),
        "forbidden_registered_commands": 0,
    }


def validate_cross_artifact_consistency(
    registry: dict[str, Any], controller: dict[str, Any], timing: dict[str, Any]
) -> dict[str, Any]:
    selected = timing["selected_project_candidate"]
    router = timing["router_label"]
    if registry.get("project_selection", {}).get("selected") != selected:
        raise FinalAuditError("Registry project selection differs from timing analysis")
    if registry.get("phases", {}).get("route_gate", {}).get("router_label") != router:
        raise FinalAuditError("Registry router decision differs from timing analysis")
    timing_registry = registry.get("timing_results", {})
    if timing_registry.get("selected_candidate") != selected:
        raise FinalAuditError("Registry timing candidate differs from timing analysis")
    if timing_registry.get("timing_gate", {}).get("passed") != timing["timing_gate"]["passed"]:
        raise FinalAuditError("Registry timing gate differs from timing analysis")
    if controller.get("state") != "MATCHED_TIMING_COMPLETE_FINAL_AUDIT_PENDING":
        raise FinalAuditError("Controller is not at the exact pre-audit terminal state")
    return {
        "selected_project_candidate": selected,
        "router_label": router,
        "timing_gate_passed": timing["timing_gate"]["passed"],
        "controller_pre_audit_state": controller["state"],
    }


def validate_gate_and_timing() -> dict[str, Any]:
    label_integrity = read_json(CAMPAIGN_ROOT / "gate_labels/labels_full3000_attempt1/integrity_report.json")
    if label_integrity.get("status") != "pass" or int(label_integrity.get("rows", -1)) != 3000:
        raise FinalAuditError("Gate labels are not a valid 3,000-row artifact")
    gate_analysis = read_json(CAMPAIGN_ROOT / "analysis/GATE_RESULTS_AND_SELECTION.json")
    if gate_analysis.get("status") != "complete" or set(gate_analysis.get("physical_seeds", {})) != {"0", "1", "2"}:
        raise FinalAuditError("Gate analysis does not report all three physical seeds")
    timing = read_json(CAMPAIGN_ROOT / "analysis/MATCHED_TIMING_RESULTS.json")
    if timing.get("status") != "complete":
        raise FinalAuditError("Matched timing is not complete")
    timing_state = read_json(CAMPAIGN_ROOT / "MATCHED_TIMING_STATE.json")
    natural = [record for key, record in timing_state["runs"].items() if key.startswith("natural_")]
    if len(natural) != 15 or any(record.get("status") != "accepted" for record in natural):
        raise FinalAuditError("Natural matched timing does not have 15 accepted serial trials")
    fixed_triggered = bool(timing["fixed_length_trigger"]["triggered"])
    fixed = [record for key, record in timing_state["runs"].items() if key.startswith("fixed32_")]
    if fixed_triggered and (len(fixed) != 9 or any(record.get("status") != "accepted" for record in fixed)):
        raise FinalAuditError("Triggered fixed-length timing does not have nine accepted trials")
    if not fixed_triggered and fixed:
        raise FinalAuditError("Fixed-length timing ran without its exact trigger")
    return {
        "label_rows": 3000,
        "label_manifest_sha256": label_integrity["manifest_sha256"],
        "zero_validation_record_leakage": label_integrity["checks"]["train_only_and_zero_validation_record_leakage"],
        "gate_training_seeds": [0, 1, 2],
        "gate_physical_seeds": [0, 1, 2],
        "router_label": gate_analysis["route_gate_decision"]["router_label"],
        "selected_project_candidate": timing["selected_project_candidate"],
        "natural_timing_trials": len(natural),
        "fixed_length_triggered": fixed_triggered,
        "fixed_length_trials": len(fixed),
        "timing_gate_passed": timing["timing_gate"]["passed"],
        "primary_outcome": timing["primary_outcome"],
        "future_8811_confirmation_recommended_not_launched": timing[
            "future_8811_confirmation_recommended_not_launched"
        ],
    }


def validate_report(report: str, timing: dict[str, Any]) -> dict[str, Any]:
    markdown = sorted(
        path for path in CAMPAIGN_ROOT.rglob("*.md") if is_campaign_report_markdown(path)
    )
    if markdown != [REPORT_PATH]:
        raise FinalAuditError(f"Campaign has more than its single cumulative Markdown report: {markdown}")
    required_headers = (
        "## Scope and frozen protocol",
        "## A0 — verified historical reference",
        "## Implementation and preflight",
        "## Adaptive fixed-method experiments",
        "## Official source-native comparisons",
        "## Learned schedule gate",
        "## Conditional matched timing",
        "## Failures and rejected attempts",
        "## Limitations",
        "## Final recommendation",
    )
    if any(header not in report for header in required_headers):
        raise FinalAuditError("Cumulative report is missing a required section")
    for phrase in ("Calibration-to-validation shift", "project frontier", "official frontier"):
        if phrase not in report:
            raise FinalAuditError(f"Cumulative report is missing required gate analysis: {phrase}")
    allowed_outcomes = (
        "A — Training-free success",
        "B — Information-preservation success",
        "C — Learned adaptive-routing success",
        "D — Learned routing is exploratory only",
        "E — Official method dominates",
        "F — Stop this direction",
    )
    stated_outcomes = [outcome for outcome in allowed_outcomes if outcome in report]
    if stated_outcomes != [timing["primary_outcome"]]:
        raise FinalAuditError(f"Final report does not state exactly one allowed primary outcome: {stated_outcomes}")
    for text in (
        timing["primary_outcome"],
        timing["router_label"],
        timing["selected_project_candidate"],
        "8,811",
    ):
        if str(text) not in report:
            raise FinalAuditError(f"Final report does not state required decision: {text}")
    return {
        "single_markdown_report": relative(REPORT_PATH),
        "required_sections": list(required_headers),
        "explicit_primary_outcome": timing["primary_outcome"],
        "explicit_router_label": timing["router_label"],
        "explicit_selected_candidate": timing["selected_project_candidate"],
        "allowed_primary_outcome": timing["primary_outcome"],
        "calibration_shift_and_pareto_frontiers": True,
    }


def main() -> int:
    tests = run_tests()
    refresh_report_test_count(int(tests["tests_run"]))
    manifest_rows = sum(1 for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip())
    if manifest_rows != EXPECTED_MANIFEST_ROWS or sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise FinalAuditError("Frozen discovery manifest differs")
    registry = read_json(REGISTRY_PATH)
    controller = read_json(STATE_PATH)
    timing = read_json(CAMPAIGN_ROOT / "analysis/MATCHED_TIMING_RESULTS.json")
    registry_evidence = validate_registry(registry)
    gate_timing = validate_gate_and_timing()
    consistency = validate_cross_artifact_consistency(registry, controller, timing)
    report_evidence = validate_report(REPORT_PATH.read_text(encoding="utf-8"), timing)
    completions = validate_completion_hashes()
    workers = active_campaign_processes()
    gpu = active_gpu_processes()
    if workers or gpu:
        raise FinalAuditError(f"Campaign cannot close with live workers/GPU processes: {workers}, {gpu}")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).splitlines()
    unexpected = [line for line in status if line.strip() != "?? PRO_REASONING_RESEARCH_HANDOFF.md"]
    if unexpected:
        raise FinalAuditError(f"Final source tree is not clean: {unexpected}")
    dissertation_changes = [line for line in status if "dissertation" in line.lower()]
    if dissertation_changes:
        raise FinalAuditError(f"Dissertation files changed: {dissertation_changes}")
    forbidden_paths = [
        relative(path)
        for path in CAMPAIGN_ROOT.rglob("*")
        if "8811" in path.name.lower() or "testa" in path.name.lower() or "testb" in path.name.lower()
    ]
    if forbidden_paths:
        raise FinalAuditError(f"Forbidden confirmation/held-out campaign paths exist: {forbidden_paths[:3]}")

    registry["phases"]["final_audit"] = {
        "status": "complete",
        "report_path": relative(FINAL_PATH),
    }
    registry["completion"] = {
        "status": "GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED",
        "selected_project_candidate": timing["selected_project_candidate"],
        "primary_outcome": timing["primary_outcome"],
        "router_label": timing["router_label"],
        "future_8811_confirmation_launched": False,
    }
    registry["updated_at_utc"] = utc_now()
    controller["state"] = "GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED"
    controller["next_required_action"] = "stop_for_human_review_without_launching_8811_confirmation"
    controller["active_workers"] = []
    controller["updated_at_utc"] = utc_now()
    atomic_write(REGISTRY_PATH, canonical_bytes(registry))
    atomic_write(STATE_PATH, canonical_bytes(controller))

    final_registry = read_json(REGISTRY_PATH)
    final_controller = read_json(STATE_PATH)
    if final_registry["completion"]["status"] != "GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED":
        raise FinalAuditError("Final registry completion state did not persist")
    if final_controller["state"] != "GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED":
        raise FinalAuditError("Final controller completion state did not persist")
    audit = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "stopping_condition": "GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED",
        "manifest": {
            "path": relative(MANIFEST),
            "rows": manifest_rows,
            "sha256": sha256_file(MANIFEST),
        },
        "tests": {**tests, "report_path": relative(TEST_REPORT_PATH), "report_sha256": sha256_file(TEST_REPORT_PATH)},
        "registry_evidence": registry_evidence,
        "cross_artifact_consistency": consistency,
        "gate_and_timing_evidence": gate_timing,
        "completion_sentinels": completions,
        "report_evidence": report_evidence,
        "official_source_branch": read_json(CAMPAIGN_ROOT / "analysis/OFFICIAL_METHOD_RESULTS.json")["status"],
        "a0_revalidation": read_json(CAMPAIGN_ROOT / "analysis/A0_COMPLETED.json")["status"],
        "preservation_and_scope": {
            "active_campaign_workers": workers,
            "active_gpu_processes": gpu,
            "git_status_only_user_handoff": status,
            "user_handoff_preserved": (REPO_ROOT / "PRO_REASONING_RESEARCH_HANDOFF.md").is_file(),
            "dissertation_changes": dissertation_changes,
            "forbidden_confirmation_or_heldout_paths": forbidden_paths,
            "full_8811_confirmation_launched": False,
            "heldout_or_transfer_job_launched": False,
            "dissertation_job_launched": False,
            "legacy_job_launched": False,
        },
        "decisions": {
            "selected_project_candidate": timing["selected_project_candidate"],
            "primary_outcome": timing["primary_outcome"],
            "router_label": timing["router_label"],
            "timing_gate_passed": timing["timing_gate"]["passed"],
            "future_8811_confirmation_recommended_not_launched": timing[
                "future_8811_confirmation_recommended_not_launched"
            ],
        },
        "final_artifact_hashes": {
            "registry_sha256": sha256_file(REGISTRY_PATH),
            "controller_state_sha256": sha256_file(STATE_PATH),
            "research_report_sha256": sha256_file(REPORT_PATH),
            "gate_analysis_sha256": sha256_file(CAMPAIGN_ROOT / "analysis/GATE_RESULTS_AND_SELECTION.json"),
            "timing_analysis_sha256": sha256_file(CAMPAIGN_ROOT / "analysis/MATCHED_TIMING_RESULTS.json"),
        },
    }
    atomic_write(FINAL_PATH, canonical_bytes(audit))
    reopened = read_json(FINAL_PATH)
    if reopened.get("status") != "complete":
        raise FinalAuditError("Final completion report failed atomic reopen")
    for key, path in (
        ("registry_sha256", REGISTRY_PATH),
        ("controller_state_sha256", STATE_PATH),
        ("research_report_sha256", REPORT_PATH),
    ):
        if reopened["final_artifact_hashes"][key] != sha256_file(path):
            raise FinalAuditError(f"Final completion report hash differs: {key}")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FinalAuditError, OSError, subprocess.SubprocessError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
