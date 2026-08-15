#!/usr/bin/env python
"""Run the clean final tests and audit every progressive/oracle campaign requirement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from analyse_progressive_oracle_campaign import render_report  # noqa: E402
from validate_progressive_oracle_run import validate_run  # noqa: E402
from validate_progressive_oracle_timing import validate_timing  # noqa: E402
from vlm_pruning.artifacts import sha256_file  # noqa: E402


CAMPAIGN_ROOT = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814"
REGISTRY_PATH = CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"
REPORT_PATH = CAMPAIGN_ROOT / "RESEARCH_REPORT.md"
AUDIT_PATH = CAMPAIGN_ROOT / "FINAL_COMPLETION_REPORT.json"
TEST_LOG_PATH = CAMPAIGN_ROOT / "analysis/final_test_suite.txt"
MANIFEST_PATH = CAMPAIGN_ROOT / "manifests/refcoco_val_discovery1000.jsonl"
LOCK_PATH = CAMPAIGN_ROOT / "manifests/discovery_manifest_lock.json"
CANCELLATION_PATH = CAMPAIGN_ROOT / "cancellation_record.json"
PREFLIGHT_PATH = CAMPAIGN_ROOT / "preflight/PREFLIGHT_REPORT.json"
EXPECTED_MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
NAMESPACE = "progressive-oracle-refcoco-val-20260814-v1"

CAMPAIGN1_IDS = (
    "T1", "T2", "T3", "T4", "T5",
    "R16-S0", "R16-S1", "R16-S2",
    "R8-S0", "R8-S1", "R8-S2",
    "R4-S0", "R4-S1", "R4-S2",
)
CAMPAIGN2_IDS = tuple(f"P{index}" for index in range(1, 10))
CAMPAIGN3_IDS = tuple(f"O{index}" for index in range(1, 11))
EXPECTED_CELL_IDS = frozenset(CAMPAIGN1_IDS + CAMPAIGN2_IDS + CAMPAIGN3_IDS)
FORBIDDEN_ACCEPTED_TERMS = (
    "testa", "testb", "refcoco+", "pope", "textvqa", "mme", "gqa",
    "mmbench", "mmmu", "vqav2", "scienceqa", "mmvet",
)
RECOMMENDATIONS = (
    "A. Simple progressive pruning is already competitive enough to continue without training.",
    "B. Oracle early selection is strong, so a lightweight future-relevance predictor/router is justified.",
    "C. Even oracle early selection is weak, so selector prediction alone is insufficient and the next work should investigate information transfer, token recovery or another mechanism.",
)


class CompletionAuditError(RuntimeError):
    """Raised when one or more definition-of-done requirements fail."""


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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CompletionAuditError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def exact_metrics(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return actual == expected


def valid_completion(run_dir: Path, completion: dict[str, Any], *, timing: bool) -> bool:
    if completion.get("status") != "complete" or int(completion.get("rows", -1)) != 1000:
        return False
    if completion.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        return False
    file_keys = {
        "config_sha256": "config.json",
        "metrics_sha256": "metrics.json",
        "integrity_sha256": "integrity_report.json",
    }
    file_keys["timing_sha256" if timing else "predictions_sha256"] = (
        "per_sample_timing.jsonl" if timing else "predictions.jsonl"
    )
    if timing:
        file_keys["environment_sha256"] = "environment.json"
    for key, name in file_keys.items():
        path = run_dir / name
        if not path.is_file() or completion.get(key) != sha256_file(path):
            return False
    oracle_hash = completion.get("oracle_indices_sha256")
    oracle_path = run_dir / "oracle_indices.jsonl"
    return timing or (
        (oracle_hash == sha256_file(oracle_path)) if oracle_path.is_file() else oracle_hash is None
    )


def active_process_evidence() -> dict[str, Any]:
    output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    active: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            pid_text, command = stripped.split(maxsplit=1)
        except ValueError:
            continue
        if int(pid_text) in {os.getpid(), os.getppid()}:
            continue
        lower = command.lower()
        if "k12_b48_expanded_evaluation_20260813" in lower or any(
            name in lower for name in (
                "run_progressive_oracle_campaign.py",
                "run_progressive_oracle_timing_campaign.py",
                "run_progressive_oracle_refcoco.py",
                "run_progressive_oracle_timing.py",
            )
        ):
            active.append(stripped)
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        gpu = []
    return {"active_campaign_processes": active, "active_gpu_processes": [line for line in gpu if line.strip()]}


def frozen_manifest_evidence(lock: dict[str, Any]) -> dict[str, Any]:
    source_path = REPO_ROOT / str(lock["source_manifest"])
    source = read_jsonl(source_path)
    manifest = read_jsonl(MANIFEST_PATH)
    source_ids = [str(row["sample_id"]) for row in source]
    selected = sorted(
        source,
        key=lambda row: (
            hashlib.sha256((NAMESPACE + "\0" + str(row["sample_id"])).encode("utf-8")).hexdigest(),
            str(row["sample_id"]).encode("utf-8"),
        ),
    )[:1000]
    selected_ids = {str(row["sample_id"]) for row in selected}
    expected = [row for row in source if str(row["sample_id"]) in selected_ids]
    return {
        "source_rows": len(source),
        "source_unique_ids": len(set(source_ids)),
        "manifest_rows": len(manifest),
        "manifest_unique_ids": len({str(row["sample_id"]) for row in manifest}),
        "selection_exact": manifest == expected,
        "source_sha256": sha256_file(source_path),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "lock_status": lock.get("status"),
        "namespace": lock.get("sampling_namespace"),
    }


def run_clean_test_suite() -> dict[str, Any]:
    status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, text=True)
    if status.strip():
        raise CompletionAuditError("Final test suite must start from a clean tracked worktree")
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    TEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_LOG_PATH.write_text(completed.stdout, encoding="utf-8")
    match = re.search(r"Ran (\d+) tests?", completed.stdout)
    discovered = int(match.group(1)) if match else 0
    evidence = {
        "status": "pass" if completed.returncode == 0 and discovered > 0 else "fail",
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "tests": discovered,
        "started_from_clean_tracked_worktree": True,
        "completed_at_utc": utc_now(),
        "output_path": str(TEST_LOG_PATH.relative_to(REPO_ROOT)),
        "output_sha256": sha256_file(TEST_LOG_PATH),
    }
    if evidence["status"] != "pass":
        raise CompletionAuditError(f"Final test suite failed; see {TEST_LOG_PATH}")
    return evidence


def perform_audit(registry: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {"status": "pass" if passed else "fail", "evidence": evidence}

    process = active_process_evidence()
    cancellation = read_json(CANCELLATION_PATH)
    check(
        "previous_and_current_campaign_processes_stopped",
        not process["active_campaign_processes"] and not process["active_gpu_processes"]
        and cancellation.get("status") == "no_active_legacy_or_cuda_process"
        and not cancellation.get("terminated_processes"),
        {"current": process, "initial_cancellation_record": cancellation},
    )

    lock = read_json(LOCK_PATH)
    manifest = frozen_manifest_evidence(lock)
    first_start = min(
        attempt["started_at_utc"]
        for cell in registry["cells"].values()
        for attempt in cell.get("attempts") or []
    )
    check(
        "frozen_discovery_manifest",
        manifest["source_rows"] == 8811
        and manifest["source_unique_ids"] == 8811
        and manifest["manifest_rows"] == 1000
        and manifest["manifest_unique_ids"] == 1000
        and manifest["selection_exact"]
        and manifest["source_sha256"] == lock["source_manifest_sha256"]
        and manifest["manifest_sha256"] == EXPECTED_MANIFEST_SHA256 == lock["output_manifest_sha256"]
        and manifest["lock_status"] == "frozen"
        and manifest["namespace"] == NAMESPACE
        and lock["created_at_utc"] < first_start,
        {**manifest, "frozen_at_utc": lock["created_at_utc"], "first_quality_start_utc": first_start},
    )

    preflight = read_json(PREFLIGHT_PATH)
    required_gates = {
        "cpu_suite", "unpruned_parity", "k12_b48_parity", "k20_b48_parity",
        "b0_generation", "two_stage", "three_stage", "oracle_restart",
        "oracle_index_replay", "quality_parallel_soak",
    }
    check(
        "mandatory_preflight_and_gpu_smokes",
        preflight.get("status") == "pass"
        and required_gates <= set(preflight.get("gates") or {})
        and all(preflight["gates"][name].get("pass") is True for name in required_gates)
        and preflight.get("safe_quality_parallel_width") == 2
        and preflight.get("oracle_parallel_width") == 1,
        {"required_gates": sorted(required_gates), "report": str(PREFLIGHT_PATH.relative_to(REPO_ROOT)), "capacity": preflight.get("capacity")},
    )

    cells = registry.get("cells") or {}
    check(
        "exact_scientific_cell_inventory",
        set(cells) == EXPECTED_CELL_IDS
        and all(cell.get("status") == "accepted" for cell in cells.values())
        and sum(cells[cell_id].get("campaign") == 1 for cell_id in CAMPAIGN1_IDS) == 14
        and sum(cells[cell_id].get("campaign") == 2 for cell_id in CAMPAIGN2_IDS) == 9
        and sum(cells[cell_id].get("campaign") == 3 for cell_id in CAMPAIGN3_IDS) == 10,
        {"campaign1": len(CAMPAIGN1_IDS), "campaign2": len(CAMPAIGN2_IDS), "campaign3": len(CAMPAIGN3_IDS), "ids": sorted(cells)},
    )

    quality_validations: dict[str, Any] = {}
    forbidden_hits: list[str] = []
    for cell_id in sorted(cells):
        cell = cells[cell_id]
        try:
            run_dir = REPO_ROOT / cell["artifact_paths"]["run_dir"]
            result = validate_run(run_dir, MANIFEST_PATH, expected_rows=1000)
            completion = read_json(run_dir / "COMPLETED.json")
            config = read_json(run_dir / "config.json")
            attempts = cell.get("attempts") or []
            command_text = " ".join(str(attempt.get("command") or "") for attempt in attempts).lower()
            config_text = json.dumps(config, sort_keys=True).lower()
            hits = [term for term in FORBIDDEN_ACCEPTED_TERMS if term in command_text or term in config_text]
            forbidden_hits.extend(f"{cell_id}:{term}" for term in hits)
            valid = (
                result.get("status") == "pass"
                and exact_metrics(result["metrics"], cell["metrics"])
                and completion == cell.get("completion_hashes")
                and valid_completion(run_dir, completion, timing=False)
                and config.get("scope") == "discovery"
                and config.get("dataset/name") == "refcoco"
                and config.get("dataset/split") == "val"
                and config.get("dataset/num_samples") == 1000
            )
            quality_validations[cell_id] = {
                "status": "pass" if valid else "fail",
                "rows": result.get("rows"),
                "predictions_sha256": result.get("predictions_sha256"),
                "config_sha256": result.get("config_sha256"),
                "source_git_commit": result.get("source_git_commit"),
            }
        except Exception as exc:
            quality_validations[cell_id] = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
    check(
        "all_quality_artifacts_independently_revalidated",
        len(quality_validations) == 33 and all(row["status"] == "pass" for row in quality_validations.values()),
        quality_validations,
    )
    check("no_forbidden_dataset_in_accepted_quality_artifacts", not forbidden_hits, forbidden_hits)

    b_tail = int(registry["decisions"]["B_tail"]["value"])
    expected_early = (64 * 32 - 12 * b_tail) // 20
    oracle_evidence: dict[str, Any] = {}
    oracle_ok = int(registry["decisions"]["B_early64"]["value"]) == expected_early
    for cell_id in ("O7", "O8"):
        cell = cells[cell_id]
        oracle_evidence[cell_id] = {
            "pre_llm_budget": cell.get("pre_llm_budget"),
            "candidate_average_visual_tokens": cell.get("candidate_average_visual_tokens"),
            "oracle_teacher_visual_token_layer_work": cell.get("oracle_teacher_visual_token_layer_work"),
        }
        oracle_ok = oracle_ok and int(cell["pre_llm_budget"]) == expected_early
        oracle_ok = oracle_ok and float(cell["candidate_average_visual_tokens"]) <= 64.0
        oracle_ok = oracle_ok and int(cell["oracle_teacher_visual_token_layer_work"]) > 0
    oracle_ok = oracle_ok and all(
        int(cells[f"O{index}"]["oracle_teacher_visual_token_layer_work"]) > 0 for index in range(1, 9)
    )
    check(
        "oracle_early64_and_teacher_accounting",
        oracle_ok,
        {"B_tail": b_tail, "calculated_B_early64": expected_early, "registered_B_early64": registry["decisions"]["B_early64"], **oracle_evidence},
    )

    timing_cells = registry.get("timing_cells") or {}
    timing_evidence: dict[str, Any] = {}
    timing_ok = set(timing_cells) == {"E1", "E2"}
    for cell_id, cell in timing_cells.items():
        if cell.get("status") == "accepted":
            try:
                run_dir = REPO_ROOT / cell["artifact_paths"]["run_dir"]
                result = validate_timing(run_dir, MANIFEST_PATH)
                completion = read_json(run_dir / "COMPLETED.json")
                valid = (
                    result.get("status") == "pass"
                    and exact_metrics(result["metrics"], cell["metrics"])
                    and completion == cell.get("completion_hashes")
                    and valid_completion(run_dir, completion, timing=True)
                    and cell.get("gate", {}).get("passed") is True
                )
                timing_evidence[cell_id] = {"status": "pass" if valid else "fail", "rows": result.get("rows"), "timing_sha256": result.get("timing_sha256")}
                timing_ok = timing_ok and valid
            except Exception as exc:
                timing_evidence[cell_id] = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
                timing_ok = False
        elif cell.get("status") == "skipped_by_design":
            valid = cell.get("gate", {}).get("passed") is False and bool(cell.get("skip_reason"))
            timing_evidence[cell_id] = {"status": "pass" if valid else "fail", "gate": cell.get("gate"), "reason": cell.get("skip_reason")}
            timing_ok = timing_ok and valid
        else:
            timing_evidence[cell_id] = {"status": "fail", "registered_status": cell.get("status")}
            timing_ok = False
    check("conditional_timing_gates_and_artifacts", timing_ok, timing_evidence)

    totals = registry.get("totals") or {}
    check(
        "job_caps_and_no_full_validation",
        totals == {"accepted_quality": 33, "accepted_timing": 1}
        and sum(cell.get("status") == "accepted" for cell in cells.values()) == 33
        and sum(cell.get("status") == "accepted" for cell in timing_cells.values()) <= 2
        and not ({"F1", "F2", "F3", "F4"} & set(cells))
        and all(cell.get("scope") == "discovery" for cell in cells.values()),
        {"totals": totals, "full_validation_ids_present": sorted({"F1", "F2", "F3", "F4"} & set(cells))},
    )

    deferred = registry.get("deferred_full_validation_recommendations") or []
    check(
        "deferred_candidates_explicitly_not_run",
        deferred == [
            {"role": "strongest_deployable", "cell_id": "T2", "status": "not_run"},
            {"role": "scientific_comparator", "cell_id": "R16-S0", "status": "not_run"},
        ],
        deferred,
    )

    report = REPORT_PATH.read_text(encoding="utf-8")
    required_headings = (
        "## Scope and protocol", "## Frozen discovery subset and historical reference",
        "## Implementation and preflight evidence", "## Campaign 1 — K20 extreme tail",
        "## Campaign 2 — progressive pruning", "## Campaign 3 — oracle diagnostics",
        "## Rejected attempts and artifact validity", "## Conditional exploratory timing",
        "## Deferred full-validation recommendation", "## Limitations",
        "## Final validation and completion audit", "## Primary recommendation",
    )
    report_metric_rows = all(
        f"| {cell_id} |" in report
        and f"{float(cells[cell_id]['metrics']['quality/acc_at_0_5']):.4f}" in report
        and f"{float(cells[cell_id]['metrics']['quality/mean_iou']):.4f}" in report
        for cell_id in cells
    )
    recommendation_occurrences = {label: report.count(label) for label in RECOMMENDATIONS}
    expected_label = RECOMMENDATIONS[{"A": 0, "B": 1, "C": 2}[registry["final_exploratory_decision"]["recommendation"]]]
    markdown_files = sorted(str(path.relative_to(CAMPAIGN_ROOT)) for path in CAMPAIGN_ROOT.rglob("*.md"))
    check(
        "single_cumulative_report_matches_registry",
        all(heading in report for heading in required_headings)
        and report_metric_rows
        and sum(recommendation_occurrences.values()) == 1
        and recommendation_occurrences[expected_label] == 1
        and "**not run**" in report
        and "E1: accepted" in report
        and "E2: skipped_by_design" in report
        and markdown_files == ["RESEARCH_REPORT.md"],
        {"required_headings": list(required_headings), "recommendation_occurrences": recommendation_occurrences, "markdown_files": markdown_files},
    )

    final_validation = registry.get("final_validation") or {}
    test_log = REPO_ROOT / str(final_validation.get("output_path") or "missing")
    check(
        "final_clean_worktree_test_suite",
        final_validation.get("status") == "pass"
        and final_validation.get("exit_code") == 0
        and int(final_validation.get("tests", 0)) > 0
        and final_validation.get("started_from_clean_tracked_worktree") is True
        and test_log.is_file()
        and final_validation.get("output_sha256") == sha256_file(test_log),
        final_validation,
    )

    campaign_model_files = [
        str(path.relative_to(CAMPAIGN_ROOT))
        for path in CAMPAIGN_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}
    ]
    check(
        "no_learned_router_or_training",
        not campaign_model_files
        and all(cell.get("deployable") is not None for cell in cells.values())
        and registry.get("final_exploratory_decision", {}).get("recommendation") in {"A", "B", "C"},
        {"campaign_model_files": campaign_model_files, "recommendation": registry.get("final_exploratory_decision")},
    )

    failed = [name for name, row in checks.items() if row["status"] != "pass"]
    return {
        "schema_version": 1,
        "campaign_id": registry.get("campaign_id"),
        "audited_at_utc": utc_now(),
        "status": "pass" if not failed else "fail",
        "summary": {"checks": len(checks), "passed": len(checks) - len(failed), "failed": failed},
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-final-tests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        registry = read_json(REGISTRY_PATH)
        if args.run_final_tests:
            previous = registry.get("final_validation")
            if previous:
                rejected = dict(previous)
                rejected["accepted"] = False
                rejected["rejection_reason"] = (
                    "zero_tests_discovered" if int(previous.get("tests", 0)) == 0
                    else "superseded_by_explicit_rerun"
                )
                registry.setdefault("final_validation_attempts", []).append(rejected)
            registry["final_validation"] = run_clean_test_suite()
            registry["updated_at_utc"] = utc_now()
            atomic_json(REGISTRY_PATH, registry)
            render_report(registry)
        audit = perform_audit(registry)
        atomic_json(AUDIT_PATH, audit)
        print(json.dumps(audit["summary"], indent=2, sort_keys=True))
        if audit["status"] != "pass":
            raise CompletionAuditError("Completion audit failed: " + ", ".join(audit["summary"]["failed"]))
        return 0
    except (CompletionAuditError, OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
