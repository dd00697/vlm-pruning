#!/usr/bin/env python
"""Independently validate the corrected hardware-aware Phase 0 artifacts."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from scripts.init_hardware_aware_zero_tail_campaign import (
    CAMPAIGN_ROOT,
    CELLS,
    REPO_ROOT,
    VALIDATION_MANIFEST,
    VALIDATION_SHA256,
    atomic_json,
    canonical_coco_image_id,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
)


class PreflightValidationError(RuntimeError):
    """Raised when a Phase 0 artifact fails independent validation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightValidationError(message)


def validate_completion_hashes(completion: dict[str, Any]) -> dict[str, str]:
    targets = {
        "preflight_sha256": CAMPAIGN_ROOT / "PREFLIGHT_AUDIT.json",
        "storage_sha256": CAMPAIGN_ROOT / "STORAGE_BUDGET.json",
        "registry_sha256": CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json",
        "controller_sha256": CAMPAIGN_ROOT / "CONTROLLER_STATE.json",
        "report_sha256": CAMPAIGN_ROOT / "RESEARCH_REPORT.md",
    }
    actual: dict[str, str] = {}
    for key, path in targets.items():
        actual[key] = sha256_file(path)
        require(actual[key] == completion.get(key), f"Phase 0 completion hash mismatch: {key}")
    return actual


def validate_rejected_archive(audit: dict[str, Any]) -> dict[str, Any]:
    rejection_pointer = audit.get("rejected_phase0_attempt") or {}
    archive = REPO_ROOT / str(rejection_pointer.get("archive_path"))
    rejection_path = archive / "REJECTION.json"
    require(archive.is_dir(), "Rejected first Phase 0 attempt was not preserved")
    require(sha256_file(rejection_path) == rejection_pointer.get("rejection_sha256"), "Rejected-attempt sentinel hash mismatch")
    rejection = read_json(rejection_path)
    require(rejection.get("status") == "rejected_before_independent_acceptance", "Rejected-attempt status mismatch")
    original_hashes = rejection.get("original_hashes") or {}
    for original_name, expected_hash in original_hashes.items():
        archived_name = "RESEARCH_REPORT.md.rejected" if original_name == "RESEARCH_REPORT.md" else original_name
        require(sha256_file(archive / archived_name) == expected_hash, f"Rejected artifact changed: {original_name}")
    return {
        "archive_path": str(archive.relative_to(REPO_ROOT)),
        "rejection_sha256": sha256_file(rejection_path),
        "preserved_artifacts": len(original_hashes),
    }


def main() -> int:
    completion = read_json(CAMPAIGN_ROOT / "analysis/PHASE0_COMPLETED.json")
    audit = read_json(CAMPAIGN_ROOT / "PREFLIGHT_AUDIT.json")
    registry = read_json(CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json")
    controller = read_json(CAMPAIGN_ROOT / "CONTROLLER_STATE.json")
    storage = read_json(CAMPAIGN_ROOT / "STORAGE_BUDGET.json")

    require(completion.get("status") == "complete", "Phase 0 completion is not complete")
    require(completion.get("manifest_sha256") == VALIDATION_SHA256, "Completion manifest lock mismatch")
    hashes = validate_completion_hashes(completion)
    require(audit.get("status") == "pass", "Preflight audit did not pass")
    require((audit.get("validation_manifest") or {}).get("rows") == 1000, "Validation row count mismatch")
    require((audit.get("validation_manifest") or {}).get("sha256") == VALIDATION_SHA256, "Audit manifest hash mismatch")
    require(sha256_file(VALIDATION_MANIFEST) == VALIDATION_SHA256, "Physical validation manifest changed")
    require((audit.get("conditional_train_manifest") or {}).get("validation_sample_id_overlap") == 0, "Train/validation record leakage")
    require((audit.get("conditional_train_manifest") or {}).get("validation_underlying_coco_image_overlap") == 90, "Underlying COCO overlap disclosure mismatch")
    train_rows = read_jsonl(REPO_ROOT / str((audit.get("conditional_train_manifest") or {})["path"]))
    validation_rows = read_jsonl(VALIDATION_MANIFEST)
    physical_overlap = {canonical_coco_image_id(row) for row in train_rows} & {canonical_coco_image_id(row) for row in validation_rows}
    require(len(physical_overlap) == 90, "Physical underlying COCO overlap is not 90")

    historical = audit.get("historical_evidence") or {}
    require((historical.get("progressive") or {}).get("status") == "pass", "Progressive history is not passing")
    require((historical.get("adaptive") or {}).get("status") == "complete", "Adaptive history is not complete")
    require((historical.get("m448") or {}).get("ordering", {}).get("rows") == 1000, "M448 row evidence mismatch")
    require((historical.get("k12_original") or {}).get("ordering", {}).get("rows") == 1000, "K12 row evidence mismatch")
    require((historical.get("timing") or {}).get("rows") == 15000, "Raw timing evidence is not 15,000 rows")
    require(len((historical.get("timing") or {}).get("natural_artifacts") or []) == 15, "Raw timing evidence is not 15 artifacts")
    require((audit.get("environment") or {}).get("active_compute_processes") == [], "GPU process existed during preflight")
    require((audit.get("active_campaign_processes") or []) == [], "Campaign process existed during preflight")

    require(set(registry.get("cells") or {}) == set(CELLS), "Registered cell matrix differs")
    require((registry.get("phases") or {}).get("phase0", {}).get("status") == "complete", "Registry Phase 0 is not complete")
    require(all(cell.get("status") == "pending" for cell in (registry.get("cells") or {}).values()), "Experimental cell advanced during preflight")
    require(controller.get("state") == "PHASE0_COMPLETE_INSTRUMENTATION_PENDING", "Controller state mismatch")
    require(controller.get("active_workers") == [], "Controller has an active worker")
    require(storage.get("campaign_cap_gib") == 250 and storage.get("launch_floor_gib") == 100, "Storage policy mismatch")
    require(shutil.disk_usage(CAMPAIGN_ROOT).free >= 100 * 1024**3, "Current storage is below launch floor")

    markdown = sorted(path for path in CAMPAIGN_ROOT.rglob("*.md") if path.is_file())
    require(markdown == [CAMPAIGN_ROOT / "RESEARCH_REPORT.md"], f"Expected one cumulative Markdown report, found {markdown}")
    forbidden_names = []
    for path in CAMPAIGN_ROOT.rglob("*"):
        lowered = path.name.lower()
        if any(term in lowered for term in ("8811", "testa", "testb", "refcoco+", "refcocog", "transfer")):
            forbidden_names.append(str(path.relative_to(CAMPAIGN_ROOT)))
    require(not forbidden_names, f"Forbidden output path found: {forbidden_names}")

    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    require(status == ["?? PRO_REASONING_RESEARCH_HANDOFF.md"], f"Unexpected Git status: {status}")
    source_commit = str(completion.get("source_commit"))
    commit_check = subprocess.run(["git", "cat-file", "-e", f"{source_commit}^{{commit}}"], cwd=REPO_ROOT)
    require(commit_check.returncode == 0, "Phase 0 source commit is unavailable")
    rejected = validate_rejected_archive(audit)

    result = {
        "schema_version": 1,
        "status": "pass",
        "validated_at_utc": utc_now(),
        "phase0_source_commit": source_commit,
        "validator_source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True).stdout.strip(),
        "validated_hashes": hashes,
        "validation_rows": 1000,
        "train_rows": 3000,
        "train_validation_sample_id_overlap": 0,
        "underlying_coco_image_overlap_disclosed": len(physical_overlap),
        "historical_timing_rows": 15000,
        "historical_timing_artifacts": 15,
        "registered_cells": sorted(CELLS),
        "single_cumulative_markdown_report": str(markdown[0].relative_to(REPO_ROOT)),
        "rejected_attempt": rejected,
        "active_workers": [],
        "active_gpu_processes": [],
    }
    output = CAMPAIGN_ROOT / "analysis/PHASE0_INDEPENDENT_VALIDATION.json"
    if output.exists():
        raise PreflightValidationError(f"Refusing to overwrite independent validation: {output}")
    atomic_json(output, result)
    print(json.dumps({**result, "output_sha256": sha256_file(output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
