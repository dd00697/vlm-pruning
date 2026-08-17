#!/usr/bin/env python
"""Independent fail-closed terminal audit for the hardware-aware campaign."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT.parent
CAMPAIGN_ROOT = REPO_ROOT / "experiments/hardware_aware_zero_tail_refcoco_val_20260816"
REGISTRY_PATH = CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"
CONTROLLER_PATH = CAMPAIGN_ROOT / "CONTROLLER_STATE.json"
REPORT_PATH = CAMPAIGN_ROOT / "RESEARCH_REPORT.md"
FINAL_PATH = CAMPAIGN_ROOT / "FINAL_COMPLETION_REPORT.json"
TEST_LOG_PATH = CAMPAIGN_ROOT / "analysis/FINAL_TEST_SUITE.log"
TEST_REPORT_PATH = CAMPAIGN_ROOT / "analysis/FINAL_TEST_SUITE.json"
MANIFEST_PATH = REPO_ROOT / (
    "experiments/progressive_oracle_refcoco_val_20260814/"
    "manifests/refcoco_val_discovery1000.jsonl"
)
EXPECTED_MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
EXPECTED_HANDOFF_SHA256 = "c64abee8234d7cc17ce92b29f019f58ce963c2ff67e7b659e827db443a0b3aaa"
PRIMARY_OUTCOME = "D — Quality/memory improvement only; no real latency win"
PRE_AUDIT_STATE = "FINAL_AUDIT_PENDING"
TERMINAL_STATE = "GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED"
MINIMUM_TESTS = 235


class FinalAuditError(RuntimeError):
    """Raised when any stopping-condition requirement is not independently true."""


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
    if not path.is_file():
        raise FinalAuditError(f"Missing file for hash: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> tuple[str | None, int, int]:
    if not root.is_dir():
        return None, 0, 0
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative_path = path.relative_to(root).as_posix()
        file_hash = sha256_file(path)
        size = path.stat().st_size
        digest.update(f"{relative_path}\0{file_hash}\0{size}\n".encode())
        files += 1
        total_bytes += size
    return digest.hexdigest(), files, total_bytes


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FinalAuditError(f"Missing required JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalAuditError(f"Expected JSON object: {path}")
    return value


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalAuditError(message)


def validate_terminal_inputs(
    registry: dict[str, Any], controller: dict[str, Any], strict: dict[str, Any],
    branches: dict[str, Any], decision: dict[str, Any], frontiers: dict[str, Any], report: str
) -> dict[str, Any]:
    required_phases = (
        "phase0", "instrumentation", "runtime_attribution", "runtime_repairs",
        "zero_tail_quality", "decoder_timing", "early_compression", "analysis",
    )
    incomplete = [
        phase for phase in required_phases
        if registry.get("phases", {}).get(phase, {}).get("status") != "complete"
    ]
    require(not incomplete, f"Incomplete registry phases: {incomplete}")
    require(registry.get("phases", {}).get("final_audit", {}).get("status") == "pending", "Final audit phase is not pending")
    require(registry.get("status") == PRE_AUDIT_STATE, "Registry is not in exact pre-audit state")
    require(controller.get("state") == PRE_AUDIT_STATE, "Controller is not in exact pre-audit state")
    require(controller.get("current_phase") == "final_audit", "Controller phase is not final_audit")
    require(controller.get("active_workers") == [], "Controller reports active workers")
    require(registry.get("primary_outcome") == PRIMARY_OUTCOME, "Registry outcome differs")
    require(controller.get("primary_outcome") == PRIMARY_OUTCOME, "Controller outcome differs")
    require(decision.get("primary_outcome") == PRIMARY_OUTCOME, "Final decision outcome differs")
    require(decision.get("single_recommended_method") == "K12_ORIGINAL", "Recommendation differs")
    require(strict.get("strict_winners") == [], "Strict gate unexpectedly has a winner")
    require(strict.get("phase5_decision") == "TRIGGERED_NO_DECODER_INTERNAL_STRICT_WIN", "Phase 5 was not correctly triggered")
    require(branches.get("phase5_terminal") is True, "Phase 5 is not terminal")
    require(branches.get("phase5_result") == "NO_QUALITY_FEASIBLE_EARLY_COMPRESSION_CANDIDATE", "Phase 5 result differs")
    exact_branches = {
        "E196_POOL": "SKIPPED_CONSERVATIVE_POOL_NOT_QUALITY_FEASIBLE",
        "E256_LORA": "SKIPPED_LORA_NEAR_QUALITY_GATE_FAILED",
        "EARLY_COMPRESSION_MATCHED_TIMING": "SKIPPED_NO_QUALITY_FEASIBLE_EARLY_CANDIDATE",
        "EARLY_COMPRESSION_FIXED32": "SKIPPED_NO_QUALITY_FEASIBLE_EARLY_CANDIDATE",
    }
    for cell, expected in exact_branches.items():
        require(branches["decisions"][cell]["decision"] == expected, f"Branch decision differs: {cell}")
    required_frontiers = {
        "acc_vs_natural_end_to_end", "miou_vs_natural_end_to_end", "acc_vs_prefill",
        "miou_vs_prefill", "acc_vs_ttft", "miou_vs_ttft", "acc_vs_prompt_kv",
        "miou_vs_prompt_kv", "acc_vs_final_kv", "miou_vs_final_kv",
        "acc_vs_token_layer_work", "miou_vs_token_layer_work",
        "acc_vs_final_visual_tokens", "miou_vs_final_visual_tokens",
    }
    require(set(frontiers.get("frontiers", {})) == required_frontiers, "Final Pareto axes are incomplete")
    require(report.count(f"Campaign state before independent audit: `{PRE_AUDIT_STATE}`.") == 1, "Report pre-audit marker is missing or duplicated")
    require(report.count(PRIMARY_OUTCOME) == 1, "Report primary outcome is missing or duplicated")
    return {
        "required_phases_complete": list(required_phases),
        "phase5_terminal": True,
        "strict_winner_count": 0,
        "pareto_axes": sorted(required_frontiers),
        "primary_outcome": PRIMARY_OUTCOME,
        "recommendation": "K12_ORIGINAL",
    }


def validate_manifest() -> dict[str, Any]:
    require(sha256_file(MANIFEST_PATH) == EXPECTED_MANIFEST_SHA256, "Frozen manifest hash changed")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(MANIFEST_PATH.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FinalAuditError(f"Manifest line {line_number} is invalid") from exc
        require(isinstance(row, dict), f"Manifest line {line_number} is not an object")
        rows.append(row)
    ids = [str(row["sample_id"]) for row in rows]
    missing_images = [row["image_path"] for row in rows if not (REPO_ROOT / row["image_path"]).is_file()]
    require(len(rows) == 1000, f"Manifest has {len(rows)} rows")
    require(len(set(ids)) == 1000, "Manifest IDs are not unique")
    require(not missing_images, f"Manifest images are missing: {missing_images[:1]}")
    order_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    return {
        "path": relative(MANIFEST_PATH), "sha256": EXPECTED_MANIFEST_SHA256,
        "rows": len(rows), "unique_ids": len(set(ids)), "all_images_exist": True,
        "first_sample_id": ids[0], "last_sample_id": ids[-1],
        "sample_id_order_sha256": order_hash,
    }


def validate_hash(path: Path, expected: Any, label: str) -> int:
    if expected is None:
        return 0
    require(path.is_file(), f"Missing hashed artifact {label}: {path}")
    require(sha256_file(path) == expected, f"Artifact hash mismatch {label}: {path}")
    return 1


def validate_completion_hashes() -> dict[str, Any]:
    direct = {
        "config_sha256": "config.json",
        "environment_sha256": "environment.json",
        "integrity_sha256": "integrity_report.json",
        "metrics_sha256": "metrics.json",
        "predictions_sha256": "predictions.jsonl",
        "profiles_sha256": "profiles.jsonl",
        "timing_sha256": "timing.jsonl",
        "torch_profiler_summary_sha256": "torch_profiler_summary.json",
        "torch_profiler_trace_sha256": "torch_profiler_trace.json",
    }
    checked = 0
    sentinels = [
        path for path in sorted(CAMPAIGN_ROOT.rglob("COMPLETED.json"))
        if "rejected_runs" not in path.parts
    ]
    for sentinel in sentinels:
        completion = read_json(sentinel)
        require(completion.get("status") == "complete", f"Non-complete sentinel: {sentinel}")
        checked += 1
        for key, filename in direct.items():
            if key in completion:
                checked += validate_hash(sentinel.parent / filename, completion[key], f"{sentinel}:{key}")
        profiler = completion.get("profiler_hashes")
        if profiler:
            checked += validate_hash(sentinel.parent / "torch_profiler_summary.json", profiler.get("summary_sha256"), f"{sentinel}:profiler_summary")
            checked += validate_hash(sentinel.parent / "torch_profiler_trace.json", profiler.get("trace_sha256"), f"{sentinel}:profiler_trace")
        integrity_path = sentinel.parent / "integrity_report.json"
        if integrity_path.is_file():
            require(read_json(integrity_path).get("status") == "pass", f"Integrity report failed: {integrity_path}")

    aggregate_maps = {
        "profiling/aggregate/COMPLETED.json": {
            "attribution_sha256": "ATTRIBUTION.json", "profile_results_sha256": "PROFILE_RESULTS.json",
            "pytorch_profiles_sha256": "PYTORCH_PROFILES.json", "trials_sha256": "TRIALS.json",
        },
        "runtime_repairs/aggregate/COMPLETED.json": {
            "results_sha256": "REPAIR_PROFILE_RESULTS.json", "selection_sha256": "M448_OPT_SELECTION.json",
            "trials_sha256": "TRIALS.json",
        },
        "runtime_repairs/compiled/COMPLETED.json": {"attempt_results_sha256": "ATTEMPT_RESULTS.json"},
        "zero_tail/analysis/COMPLETED.json": {
            "quality_results_sha256": "QUALITY_RESULTS.json", "critical_failures_sha256": "CRITICAL_FAILURES.json",
        },
        "early_compression/analysis/COMPLETED.json": {
            "quality_results_sha256": "QUALITY_RESULTS.json", "branch_decisions_sha256": "BRANCH_DECISIONS.json",
        },
        "matched_timing/analysis/NATURAL_COMPLETED.json": {
            "natural_results_sha256": "NATURAL_RESULTS.json", "strict_gate_sha256": "STRICT_GATE.json",
        },
        "matched_timing/analysis/FIXED32_COMPLETED.json": {"fixed32_results_sha256": "FIXED32_RESULTS.json"},
        "analysis/FINAL_ANALYSIS_COMPLETED.json": {
            "frontiers_sha256": "FINAL_PARETO_FRONTIERS.json", "decision_sha256": "FINAL_DECISION.json",
        },
    }
    for completion_relative, mapping in aggregate_maps.items():
        completion_path = CAMPAIGN_ROOT / completion_relative
        completion = read_json(completion_path)
        require(completion.get("status") == "complete", f"Analysis completion failed: {completion_path}")
        for key, filename in mapping.items():
            checked += validate_hash(completion_path.parent / filename, completion[key], f"{completion_path}:{key}")
    require(len(sentinels) >= 90, f"Unexpectedly few completion sentinels: {len(sentinels)}")
    require(checked >= 400, f"Unexpectedly few completion/hash checks: {checked}")
    return {"completion_sentinels": len(sentinels), "sentinel_and_hash_checks": checked}


def validate_independent_evidence() -> dict[str, Any]:
    profile_trials = read_json(CAMPAIGN_ROOT / "profiling/aggregate/TRIALS.json")["trials"]
    repair_trials = read_json(CAMPAIGN_ROOT / "runtime_repairs/aggregate/TRIALS.json")["trials"]
    require(len(profile_trials) == 15, "Profiling trial count differs")
    require(len(repair_trials) == 15, "Repair trial count differs")
    for trial in profile_trials + repair_trials:
        validation = trial.get("independent_validation", {})
        require(validation.get("status") == "pass", "A profile trial lacks passing independent validation")
        require(validation.get("all_semantic_matches") is True, "A profile trial has semantic drift")
    zero = read_json(CAMPAIGN_ROOT / "zero_tail/analysis/QUALITY_RESULTS.json")
    for method in ("Z_K12_0", "Z_M448_0"):
        validation = zero["independent_validation"][method]
        require(validation.get("status") == "pass", f"Zero-tail validation failed: {method}")
        require(all(validation.get("checks", {}).values()), f"Zero-tail check failed: {method}")
    natural = read_json(CAMPAIGN_ROOT / "matched_timing/analysis/NATURAL_RESULTS.json")
    fixed = read_json(CAMPAIGN_ROOT / "matched_timing/analysis/FIXED32_RESULTS.json")
    require(len(natural.get("trials", [])) == 20, "Natural timing does not have 20 trials")
    require(len(fixed.get("trials", [])) == 9, "Fixed32 timing does not have 9 trials")
    require(all(row.get("independent_validation_status") == "pass" for row in natural["trials"]), "A natural trial failed validation")
    require(all(row.get("independent_validation_status") == "pass" for row in fixed["trials"]), "A fixed32 trial failed validation")
    early = read_json(CAMPAIGN_ROOT / "early_compression/analysis/QUALITY_RESULTS.json")
    require(all(early.get("e256_independent_validation", {}).values()), "E256 independent checks failed")
    require(early["quality_screen"]["E256_POOL_pass"] is False, "E256 quality decision differs")
    require(early["lora_near_quality_screen"]["E256_POOL_pass"] is False, "E256 LoRA trigger differs")
    return {
        "profile_trials": 15, "repair_trials": 15, "zero_tail_quality_runs": 2,
        "natural_trials": 20, "fixed32_trials": 9, "early_quality_runs": 1,
        "all_independently_validated": True,
    }


def command_strings(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if "command" in str(key).lower() and isinstance(item, str):
                result.append(item)
            result.extend(command_strings(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(command_strings(item))
    return result


def validate_scope_and_leakage(registry: dict[str, Any]) -> dict[str, Any]:
    forbidden_names = ("8811", "testa", "testb", "refcoco+", "refcoco_plus", "refcocog", "transfer")
    forbidden_paths = []
    for path in CAMPAIGN_ROOT.rglob("*"):
        relative_name = str(path.relative_to(CAMPAIGN_ROOT)).lower()
        if any(term in relative_name for term in forbidden_names):
            forbidden_paths.append(relative_name)
    require(not forbidden_paths, f"Forbidden-scope output paths exist: {forbidden_paths[:1]}")
    commands = command_strings(registry)
    for config_path in CAMPAIGN_ROOT.rglob("config.json"):
        config = read_json(config_path)
        commands.extend(command_strings(config))
        split = str(config.get("dataset/split", "")).lower()
        rows = config.get("dataset/rows")
        if split == "val" and rows is not None:
            require(int(rows) <= 1000, f"Validation config exceeds 1,000 rows: {config_path}")
        dataset = str(config.get("dataset/name", "refcoco")).lower()
        require(dataset in ("", "refcoco"), f"Unauthorized dataset in {config_path}: {dataset}")
    forbidden_commands = [
        command for command in commands
        if any(term in command.lower() for term in ("--limit 8811", "testa", "testb", "refcoco+", "refcocog", "transfer"))
    ]
    require(not forbidden_commands, f"Forbidden command registered: {forbidden_commands[:1]}")
    early_paths = [str(path.relative_to(CAMPAIGN_ROOT)).lower() for path in (CAMPAIGN_ROOT / "early_compression").rglob("*")]
    unauthorized_training = [path for path in early_paths if any(term in path for term in ("lora", "adapter", "training", "checkpoint"))]
    require(not unauthorized_training, f"Unauthorized training artifact exists: {unauthorized_training[:1]}")
    require(registry["cells"]["E256_LORA"]["attempts"] == [], "Registry reports a LoRA attempt")
    require(registry["cells"]["E256_LORA"]["status"] == "SKIPPED_LORA_NEAR_QUALITY_GATE_FAILED", "LoRA skip differs")
    markdown = [
        path for path in CAMPAIGN_ROOT.rglob("*.md")
        if "rejected_runs" not in path.parts
    ]
    require(markdown == [REPORT_PATH], f"Campaign must have exactly one Markdown report: {markdown}")
    return {
        "forbidden_output_paths": 0, "registered_commands_checked": len(commands),
        "unauthorized_training_artifacts": 0, "validation_configs_at_most_1000": True,
        "campaign_markdown_reports": [relative(path) for path in markdown],
    }


def validate_historical_and_read_only() -> dict[str, Any]:
    preflight = read_json(CAMPAIGN_ROOT / "PREFLIGHT_AUDIT.json")
    checked = 0
    historical = preflight["historical_evidence"]
    for section in ("adaptive", "progressive"):
        for path_text, expected in historical[section]["hashes"].items():
            checked += validate_hash(REPO_ROOT / path_text, expected, f"historical:{path_text}")
    m448 = historical["m448"]
    for filename, expected in m448["hashes"].items():
        checked += validate_hash(REPO_ROOT / m448["run_dir"] / filename, expected, f"historical:m448:{filename}")
    checked += validate_hash(REPO_ROOT / historical["k12_original"]["rescored_path"], historical["k12_original"]["hashes"]["rescored"], "historical:k12:rescored")
    checked += validate_hash(REPO_ROOT / historical["k12_original"]["source_path"], historical["k12_original"]["hashes"]["source"], "historical:k12:source")
    for trial in historical["timing"]["natural_artifacts"]:
        run_dir = REPO_ROOT / trial["run_dir"]
        checked += validate_hash(run_dir / "COMPLETED.json", trial["completion_sha256"], "historical:timing:completion")
        checked += validate_hash(run_dir / "integrity_report.json", trial["integrity_sha256"], "historical:timing:integrity")
        checked += validate_hash(run_dir / "timing.jsonl", trial["timing_sha256"], "historical:timing:rows")
    dissertation = preflight["dissertation_read_only_snapshot"]
    tree_hash, files, total_bytes = tree_sha256(WORKSPACE / "Dissertation_Draft")
    require(tree_hash == dissertation["source_tree_sha256"], "Dissertation source tree changed")
    require(files == dissertation["source_files"], "Dissertation source file count changed")
    require(total_bytes == dissertation["source_bytes"], "Dissertation source byte count changed")
    pdf_path = Path(dissertation["pdf_path"])
    require(pdf_path.is_file() == dissertation["pdf_exists"], "Dissertation PDF existence changed")
    if pdf_path.is_file():
        require(sha256_file(pdf_path) == dissertation["pdf_sha256"], "Dissertation PDF changed")
    checkpoint = preflight["checkpoint"]
    snapshot = Path(checkpoint["snapshot_path"])
    checked += validate_hash(snapshot / "config.json", checkpoint["config_sha256"], "checkpoint:config")
    checked += validate_hash(snapshot / "model.safetensors.index.json", checkpoint["model_index_sha256"], "checkpoint:index")
    require(all((snapshot / name).exists() for name in checkpoint["required_files_present"]), "Checkpoint files are missing")
    return {
        "historical_hashes_checked": checked,
        "dissertation_source_tree_sha256": tree_hash,
        "dissertation_files": files,
        "dissertation_bytes": total_bytes,
        "checkpoint_revision": checkpoint["snapshot_revision"],
        "historical_evidence_unchanged": True,
    }


def validate_git_and_processes() -> dict[str, Any]:
    handoff = REPO_ROOT / "PRO_REASONING_RESEARCH_HANDOFF.md"
    require(handoff.is_file(), "User handoff is missing")
    require(sha256_file(handoff) == EXPECTED_HANDOFF_SHA256, "User handoff changed")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO_ROOT, text=True
    ).splitlines()
    require(status == ["?? PRO_REASONING_RESEARCH_HANDOFF.md"], f"Unexpected Git status: {status}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    process_output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    runner_names = (
        "run_hardware_runtime_profile.py", "run_hardware_matched_timing_trial.py",
        "run_hardware_zero_tail_quality.py", "run_early_compression_quality.py",
        "run_runtime_repair", "run_zero_tail",
    )
    active_workers = []
    own_pid = os.getpid()
    for line in process_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        if int(pid_text) != own_pid and any(name in command for name in runner_names):
            active_workers.append({"pid": int(pid_text), "command": command})
    require(not active_workers, f"Campaign workers remain active: {active_workers}")
    try:
        gpu_output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FinalAuditError("Could not verify GPU process isolation") from exc
    gpu_processes = [line.strip() for line in gpu_output.splitlines() if line.strip()]
    require(not gpu_processes, f"GPU compute processes remain active: {gpu_processes}")
    campaign_bytes = sum(path.stat().st_size for path in CAMPAIGN_ROOT.rglob("*") if path.is_file())
    require(campaign_bytes <= 250 * 1024**3, "Campaign artifact budget exceeded")
    disk = shutil.disk_usage(WORKSPACE)
    return {
        "git_commit": revision, "git_status": status,
        "user_handoff_sha256": EXPECTED_HANDOFF_SHA256,
        "active_campaign_workers": active_workers, "active_gpu_processes": gpu_processes,
        "campaign_bytes": campaign_bytes, "campaign_cap_bytes": 250 * 1024**3,
        "workspace_free_bytes": disk.free,
    }


def run_code_checks() -> dict[str, Any]:
    compile_command = [
        sys.executable, "-m", "py_compile",
        "scripts/finalize_hardware_aware_campaign.py",
        "scripts/audit_hardware_aware_campaign.py",
        "tests/test_hardware_aware_finalization.py",
    ]
    compile_process = subprocess.run(
        compile_command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    require(compile_process.returncode == 0, f"Final source compile failed: {compile_process.stdout[-2000:]}")
    test_command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    test_process = subprocess.run(
        test_command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    match = re.search(r"Ran (\d+) tests", test_process.stdout)
    tests_run = int(match.group(1)) if match else -1
    require(test_process.returncode == 0, f"Final unit suite failed: {test_process.stdout[-4000:]}")
    require(tests_run >= MINIMUM_TESTS, f"Final unit suite ran only {tests_run} tests")
    diff_process = subprocess.run(
        ["git", "diff", "--check"], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    require(diff_process.returncode == 0, f"git diff --check failed: {diff_process.stdout}")
    return {
        "compile_command": compile_command, "compile_exit_status": compile_process.returncode,
        "test_command": test_command, "test_exit_status": test_process.returncode,
        "tests_run": tests_run, "minimum_expected_tests": MINIMUM_TESTS,
        "test_output": test_process.stdout,
        "git_diff_check_exit_status": diff_process.returncode,
    }


def validate_finalized_state(registry: dict[str, Any], controller: dict[str, Any], report: str) -> None:
    require(registry.get("status") == "complete", "Final registry status differs")
    require(registry.get("terminal_state") == TERMINAL_STATE, "Final registry terminal state differs")
    require(registry["phases"]["final_audit"]["status"] == "complete", "Final registry audit phase differs")
    require(controller.get("state") == TERMINAL_STATE, "Final controller terminal state differs")
    require(controller.get("current_phase") == "complete", "Final controller phase differs")
    require(controller.get("active_workers") == [], "Final controller reports workers")
    require(PRIMARY_OUTCOME in report, "Final report outcome differs")
    require(report.count(f"Campaign terminal state: `{TERMINAL_STATE}`.") == 1, "Final report state differs")


def run() -> int:
    if FINAL_PATH.exists():
        raise FinalAuditError("Final completion report already exists; accepted output is immutable")
    registry = read_json(REGISTRY_PATH)
    controller = read_json(CONTROLLER_PATH)
    strict = read_json(CAMPAIGN_ROOT / "matched_timing/analysis/STRICT_GATE.json")
    branches = read_json(CAMPAIGN_ROOT / "early_compression/analysis/BRANCH_DECISIONS.json")
    decision = read_json(CAMPAIGN_ROOT / "analysis/FINAL_DECISION.json")
    frontiers = read_json(CAMPAIGN_ROOT / "analysis/FINAL_PARETO_FRONTIERS.json")
    analysis_completion = read_json(CAMPAIGN_ROOT / "analysis/FINAL_ANALYSIS_COMPLETED.json")
    report = REPORT_PATH.read_text(encoding="utf-8")
    inputs = validate_terminal_inputs(registry, controller, strict, branches, decision, frontiers, report)
    require(analysis_completion["registry_sha256"] == sha256_file(REGISTRY_PATH), "Pre-audit registry hash differs")
    require(analysis_completion["controller_sha256"] == sha256_file(CONTROLLER_PATH), "Pre-audit controller hash differs")
    require(analysis_completion["report_sha256"] == sha256_file(REPORT_PATH), "Pre-audit report hash differs")
    manifest = validate_manifest()
    hashes = validate_completion_hashes()
    independent = validate_independent_evidence()
    scope = validate_scope_and_leakage(registry)
    historical = validate_historical_and_read_only()
    runtime = validate_git_and_processes()
    code = run_code_checks()

    atomic_write(TEST_LOG_PATH, code.pop("test_output").encode("utf-8"))
    code["test_log_path"] = relative(TEST_LOG_PATH)
    code["test_log_sha256"] = sha256_file(TEST_LOG_PATH)
    code["completed_at_utc"] = utc_now()
    code["status"] = "pass"
    atomic_write(TEST_REPORT_PATH, canonical_bytes(code))

    terminal_time = utc_now()
    report = report.replace(
        f"Campaign state before independent audit: `{PRE_AUDIT_STATE}`.",
        f"Campaign terminal state: `{TERMINAL_STATE}`. Independent final audit passed {code['tests_run']} repository tests, all artifact/scope/read-only checks, and confirmed no worker or GPU compute process remained.",
    )
    atomic_write(REPORT_PATH, report.encode("utf-8"))
    registry["phases"]["final_audit"] = {
        "status": "complete", "evidence_path": relative(FINAL_PATH),
        "completed_at_utc": terminal_time,
    }
    registry["status"] = "complete"
    registry["terminal_state"] = TERMINAL_STATE
    registry["updated_at_utc"] = terminal_time
    controller["current_phase"] = "complete"
    controller["state"] = TERMINAL_STATE
    controller["next_required_action"] = None
    controller["stop_reason"] = "All authorized campaign work is terminal; human review required"
    controller["updated_at_utc"] = terminal_time
    atomic_write(REGISTRY_PATH, canonical_bytes(registry))
    atomic_write(CONTROLLER_PATH, canonical_bytes(controller))
    final_report = REPORT_PATH.read_text(encoding="utf-8")
    validate_finalized_state(registry, controller, final_report)

    requirements = {
        "phase0_provenance_and_storage_audits": "pass",
        "instrumentation_independently_valid": "pass",
        "mandatory_k12_m448_replay_profiles": "pass",
        "triggered_repairs_terminal_and_m448_opt_frozen": "pass",
        "mandatory_zero_tail_quality_cells": "pass",
        "z_m448_16_rule_based_skip": "pass",
        "decoder_internal_natural_and_fixed32_timing": "pass",
        "early_compression_and_adaptation_branches_terminal": "pass",
        "all_statistical_analyses_and_pareto_frontiers": "pass",
        "exactly_one_cumulative_markdown_report": "pass",
        "registry_controller_report_and_branches_agree": "pass",
        "frozen_manifest_exactly_1000_and_unchanged": "pass",
        "no_forbidden_scope_or_unauthorized_training": "pass",
        "dissertation_and_historical_evidence_unchanged": "pass",
        "source_tests_compile_and_diff_check": "pass",
        "git_expected_and_user_handoff_untouched": "pass",
        "no_worker_or_gpu_process": "pass",
    }
    payload = {
        "schema_version": 1,
        "status": "complete",
        "audited_at_utc": terminal_time,
        "terminal_state": TERMINAL_STATE,
        "primary_outcome": PRIMARY_OUTCOME,
        "single_recommended_method": "K12_ORIGINAL",
        "issues": [],
        "requirements": requirements,
        "terminal_inputs": inputs,
        "manifest": manifest,
        "artifact_hash_audit": hashes,
        "independent_validation": independent,
        "scope_and_leakage": scope,
        "historical_and_read_only": historical,
        "runtime_and_repository": runtime,
        "code_checks": code,
        "artifact_sha256": {
            "EXPERIMENT_REGISTRY.json": sha256_file(REGISTRY_PATH),
            "CONTROLLER_STATE.json": sha256_file(CONTROLLER_PATH),
            "RESEARCH_REPORT.md": sha256_file(REPORT_PATH),
            "analysis/FINAL_PARETO_FRONTIERS.json": sha256_file(CAMPAIGN_ROOT / "analysis/FINAL_PARETO_FRONTIERS.json"),
            "analysis/FINAL_DECISION.json": sha256_file(CAMPAIGN_ROOT / "analysis/FINAL_DECISION.json"),
            "analysis/FINAL_ANALYSIS_COMPLETED.json": sha256_file(CAMPAIGN_ROOT / "analysis/FINAL_ANALYSIS_COMPLETED.json"),
            "analysis/FINAL_TEST_SUITE.json": sha256_file(TEST_REPORT_PATH),
            "analysis/FINAL_TEST_SUITE.log": sha256_file(TEST_LOG_PATH),
        },
    }
    atomic_write(FINAL_PATH, canonical_bytes(payload))
    print(json.dumps({"status": "complete", "terminal_state": TERMINAL_STATE, "tests_run": code["tests_run"], "output": relative(FINAL_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (FinalAuditError, OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
