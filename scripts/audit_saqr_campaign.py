#!/usr/bin/env python
"""Independent fail-closed terminal audit for the quick SAQR campaign."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for entry in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from run_saqr_campaign import (  # noqa: E402
    CACHE_ROOT,
    CAMPAIGN_ROOT,
    DISCOVERY_MANIFEST,
    DISCOVERY_SHA256,
    TRAIN_MANIFEST,
    TRAIN_SHA256,
    atomic_json,
    read_jsonl,
    sha256_file,
)


HANDOFF = REPO_ROOT / "PRO_REASONING_RESEARCH_HANDOFF.md"
HANDOFF_SHA256 = "c64abee8234d7cc17ce92b29f019f58ce963c2ff67e7b659e827db443a0b3aaa"
TERMINAL = {
    "A_REAL_ACCELERATION_STRONG_QUALITY",
    "B_REAL_ACCELERATION_ACCEPTABLE_QUALITY",
    "C_QUALITY_PRESERVED_NO_REAL_ACCELERATION",
    "D_FAST_BUT_QUALITY_LOSS",
    "E_NO_USEFUL_TRADEOFF",
    "F_CALIBRATION_FAILURE",
}


class AuditError(RuntimeError):
    pass


def required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AuditError(f"Required JSON is missing: {path}")
    return json.loads(path.read_text())


def require(condition: bool, message: str, evidence: dict[str, Any]) -> None:
    evidence[message] = "pass" if condition else "fail"
    if not condition:
        raise AuditError(message)


def main() -> int:
    evidence: dict[str, Any] = {}
    require(sha256_file(TRAIN_MANIFEST) == TRAIN_SHA256, "frozen_train_manifest_hash", evidence)
    require(sha256_file(DISCOVERY_MANIFEST) == DISCOVERY_SHA256, "frozen_discovery_manifest_hash", evidence)
    require(sha256_file(HANDOFF) == HANDOFF_SHA256, "protected_handoff_unchanged", evidence)
    protocol = required_json(CAMPAIGN_ROOT / "FROZEN_PROTOCOL.json")
    require(protocol["status"] == "FROZEN_BEFORE_NEW_RESULTS", "protocol_frozen_before_results", evidence)
    require(protocol["model"]["seed"] == 0, "only_seed_zero_registered", evidence)
    require(protocol["architecture"]["alignment_grid"] == [16, 16] and protocol["architecture"]["candidate_grid"] == [14, 14], "only_256_196_budgets_registered", evidence)

    cache_summary = required_json(CACHE_ROOT / "SUMMARY.json")
    cache_validation = required_json(CACHE_ROOT / "INDEPENDENT_VALIDATION.json")
    cache_index = read_jsonl(CACHE_ROOT / "INDEX.jsonl")
    require(cache_summary["status"] == "COMPLETE" and cache_summary["rows"] == 3000, "training_cache_complete_3000", evidence)
    require(cache_validation["status"] == "PASS" and cache_validation["rows"] == 3000, "training_cache_independently_validated", evidence)
    require(len(cache_index) == 3000 and len({row["sample_id"] for row in cache_index}) == 3000, "training_cache_unique_ordered_rows", evidence)
    require(cache_summary["partitions"] == {"fit": 2160, "early_stop": 240, "calibration": 600}, "training_partitions_exact", evidence)

    training = required_json(CAMPAIGN_ROOT / "checkpoints/TRAINING_SUMMARY.json")
    require(training["status"] == "COMPLETE" and training["seed"] == 0, "single_seed_training_complete", evidence)
    require(len(training["stages"]["A"]["epochs"]) == 2 and len(training["stages"]["B"]["epochs"]) == 2, "exact_two_plus_two_epochs", evidence)
    require(training["frozen_parameter_checksum_before"] == training["frozen_parameter_checksum_after"], "frozen_parameters_unchanged", evidence)
    require(all("lora_" in name or name.startswith(("conditioner.", "resampler.", "box_head.")) for name in training["trainable_parameters"]), "trainable_allowlist_exact", evidence)

    calibration = required_json(CAMPAIGN_ROOT / "analysis/CALIBRATION_GATE.json")
    registry = required_json(CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json")
    controller = required_json(CAMPAIGN_ROOT / "CONTROLLER_STATE.json")
    claims = required_json(CAMPAIGN_ROOT / "analysis/CLAIM_LEDGER.json")
    classification = str(registry.get("classification"))
    require(classification in TERMINAL, "terminal_classification_registered", evidence)
    require(controller.get("classification") == classification, "controller_registry_classification_agree", evidence)
    require(claims.get("classification") == classification, "claim_registry_classification_agree", evidence)
    require(registry["cells"]["T_SAQR256_S0"] == "COMPLETE" and registry["cells"]["T_SAQR196_S0"] == "COMPLETE" and registry["cells"]["C_SAQR196_S0"] == "COMPLETE", "mandatory_training_calibration_cells_complete", evidence)

    if calibration["status"] == "FAIL":
        skip = required_json(CAMPAIGN_ROOT / "analysis/DISCOVERY_AND_TIMING_SKIP.json")
        require(classification == "F_CALIBRATION_FAILURE", "calibration_failure_classification", evidence)
        require(skip["status"] == "SKIPPED_DISCOVERY_AND_TIMING_CALIBRATION_GATE_FAILED", "calibration_failure_exact_skip", evidence)
    else:
        discovery = required_json(CAMPAIGN_ROOT / "analysis/DISCOVERY_QUALITY_GATE.json")
        require(registry["cells"]["Q_SAQR196_S0"] == "COMPLETE" and registry["cells"]["Q_SAQR196_S0_NO_QUERY"] == "COMPLETE", "conditional_discovery_cells_complete", evidence)
        if discovery["status"] == "FAIL":
            skip = required_json(CAMPAIGN_ROOT / "analysis/MATCHED_TIMING_SKIP.json")
            require(skip["status"] == "SKIPPED_MATCHED_TIMING_QUALITY_GATE_FAILED", "quality_failure_exact_timing_skip", evidence)
            require(classification == "E_NO_USEFUL_TRADEOFF", "quality_failure_classification", evidence)
        else:
            timing = required_json(CAMPAIGN_ROOT / "analysis/TIMING_ACCELERATION_GATE.json")
            require(registry["cells"]["MATCHED_TIMING"] == "COMPLETE", "conditional_matched_timing_complete", evidence)
            timing_files = list((CAMPAIGN_ROOT / "matched_timing/natural").glob("*/timing.jsonl"))
            require(len(timing_files) == 6 and all(len(read_jsonl(path)) == 1000 for path in timing_files), "exact_three_by_two_timing_cells", evidence)
            if timing["status"] == "PASS":
                require(classification.startswith(("A_", "B_")), "positive_timing_classification", evidence)
            else:
                require(classification == "C_QUALITY_PRESERVED_NO_REAL_ACCELERATION", "no_acceleration_classification", evidence)

    forbidden_fragments = ("saqr144", "seed1", "seed2", "testa", "testb", "fixed32", "full8811")
    campaign_paths = [str(path.relative_to(CAMPAIGN_ROOT)).lower() for path in CAMPAIGN_ROOT.rglob("*")]
    require(not any(fragment in path for fragment in forbidden_fragments for path in campaign_paths), "no_forbidden_campaign_branch", evidence)
    source_validation = required_json(CAMPAIGN_ROOT / "FINAL_SOURCE_VALIDATION.json")
    require(source_validation.get("compile") == "PASS" and source_validation.get("unit_tests") == "PASS" and source_validation.get("git_diff_check") == "PASS", "final_source_validation_pass", evidence)
    report = CAMPAIGN_ROOT / "RESEARCH_REPORT.md"
    require(report.exists() and classification in report.read_text(), "cumulative_report_matches_classification", evidence)
    positive = classification.startswith(("A_", "B_"))
    if positive:
        integration = required_json(CAMPAIGN_ROOT / "DISSERTATION_INTEGRATION.json")
        require(integration.get("status") == "PASS", "positive_result_dissertation_integrated", evidence)
        require(Path("/workspace/dissertation_final.pdf").exists(), "positive_result_pdf_exists", evidence)
    else:
        require(not claims["dissertation_integration_required"], "negative_result_leaves_dissertation_unmodified", evidence)
    gpu_processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    ).strip()
    require(not gpu_processes, "no_cuda_worker_remains", evidence)
    require(controller.get("workers") == [], "controller_has_no_worker", evidence)

    controller.update({
        "status": "COMPLETE",
        "phase": "TERMINAL",
        "last_completed_phase": "PHASE_7_FINAL_AUDIT",
        "terminal_state": "GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED",
    })
    atomic_json(CAMPAIGN_ROOT / "CONTROLLER_STATE.json", controller)
    registry["status"] = "COMPLETE"
    atomic_json(CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json", registry)
    completion = {
        "status": "GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED",
        "classification": classification,
        "single_seed_preliminary": True,
        "requirements": evidence,
        "report": "RESEARCH_REPORT.md",
        "claim_ledger": "analysis/CLAIM_LEDGER.json",
        "registry_sha256": sha256_file(CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"),
        "controller_sha256": sha256_file(CAMPAIGN_ROOT / "CONTROLLER_STATE.json"),
        "protected_handoff_sha256": sha256_file(HANDOFF),
    }
    atomic_json(CAMPAIGN_ROOT / "FINAL_COMPLETION_REPORT.json", completion)
    print("GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
