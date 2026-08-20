#!/usr/bin/env python3
"""Independently decide whether historical held-out Full controls may be reused."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for value in (SCRIPT_DIR, REPO_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from frozen_z_heldout_common import (
    CAMPAIGN,
    HISTORICAL_FULL,
    MANIFESTS,
    MODEL,
    PROMPT,
    HeldoutError,
    atomic_json,
    now,
    read_json,
    read_jsonl,
    sha256,
    verify_manifest,
)
from validate_progressive_oracle_run import parse_box
from run_baseline import box_iou


def audit_split(split: str) -> dict[str, Any]:
    root = HISTORICAL_FULL[split]
    manifest = verify_manifest(split)
    predictions_path = root / "per_sample_results.jsonl"
    config_path = root / "config.json"
    environment_path = root / "run_environment.json"
    metrics_path = root / "metrics.json"
    required = [config_path, environment_path, metrics_path, predictions_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"split": split, "decision": "rerun", "passed": False, "missing": missing}
    config = read_json(config_path)
    environment = read_json(environment_path)
    predictions = read_jsonl(predictions_path)
    checks: dict[str, Any] = {
        "row_count": len(predictions) == MANIFESTS[split]["rows"],
        "ordered_unique_ids": [str(row.get("sample_id") or "") for row in predictions]
        == [str(row["sample_id"]) for row in manifest]
        and len({str(row.get("sample_id") or "") for row in predictions}) == len(predictions),
        "config_model": config.get("model/name") == MODEL,
        "config_precision": config.get("model/precision") == "bf16",
        "config_decoding": config.get("model/decoding") == "deterministic_greedy",
        "config_max_new_tokens": config.get("model/max_new_tokens") == 48,
        "config_seed": config.get("method/seed") == 0,
        "config_split": str(config.get("dataset/split") or "").lower() == split.lower(),
        "config_prompt": config.get("prompt/template") == PROMPT,
        "config_position_mode": config.get("position_mode") == "gap_preserving",
        "config_full_576": config.get("method/name") == "full" and config.get("tokens/actual_visual_tokens") == 576,
        "full_manifest_sha256_recorded": config.get("dataset/manifest_checksum") == MANIFESTS[split]["sha256"],
        "atomic_completed_marker": (root / "COMPLETED.json").is_file(),
        "source_commit_full": isinstance(environment.get("git_commit"), str) and len(environment.get("git_commit", "")) == 40,
        "source_clean": environment.get("git_dirty") is False,
        "source_file_hashes_recorded": isinstance(environment.get("source_sha256"), dict) and bool(environment.get("source_sha256")),
        "checkpoint_revision_recorded": environment.get("model_revision") is not None,
        "row_config_sha256_bound": all(row.get("config_sha256") == sha256(config_path) for row in predictions),
    }
    rescored_acc = 0
    rescored_iou = 0.0
    semantic_rows = True
    if len(predictions) == len(manifest):
        for source, row in zip(manifest, predictions, strict=True):
            if any(row.get(key) != source.get(key) for key in ("dataset", "split", "query", "image_id", "image_path", "gt_box")):
                semantic_rows = False
                break
            predicted, _, reason = parse_box(str(row.get("raw_model_output") or ""), int(source["width"]), int(source["height"]))
            iou = box_iou(predicted, source["gt_box"]) if predicted is not None else 0.0
            if row.get("parse_failure_reason") != reason or not math.isclose(float(row.get("box_iou", -1)), iou, abs_tol=1e-9):
                semantic_rows = False
                break
            rescored_iou += iou
            rescored_acc += int(iou >= 0.5)
    else:
        semantic_rows = False
    checks["row_semantics_and_raw_rescore"] = semantic_rows
    immutable_files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in required}
    failed = sorted(name for name, passed in checks.items() if not passed)
    # Every gate is mandatory. In particular, an old dirty source without file hashes cannot be
    # reconstructed from a commit or successful process return code.
    passed = not failed
    return {
        "split": split,
        "historical_path": str(root),
        "audited_at_utc": now(),
        "passed": passed,
        "decision": "reuse" if passed else "rerun",
        "checks": checks,
        "failed_gates": failed,
        "immutable_files": immutable_files,
        "rescored_metrics_if_semantic": {
            "rows": len(predictions),
            "acc_at_0_5": rescored_acc / len(predictions) if semantic_rows else None,
            "mean_iou": rescored_iou / len(predictions) if semantic_rows else None,
        },
        "reason": None if passed else "Strict immutable provenance gate failed; successful historical execution is insufficient.",
    }


def main() -> int:
    if not (CAMPAIGN / "PRE_INFERENCE_LOCK.json").is_file():
        raise HeldoutError("Campaign is not preregistered")
    if any(CAMPAIGN.glob("quality/z_*/*predictions*")):
        raise HeldoutError("Candidate output exists before the Full reuse decision")
    reports = {split: audit_split(split) for split in ("testA", "testB")}
    output = {"schema_version": 1, "status": "complete", "audited_at_utc": now(), "splits": reports}
    atomic_json(CAMPAIGN / "FULL_REUSE_AUDIT.json", output)
    registry = read_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json")
    for split, report in reports.items():
        registry["cells"][f"Full_{split}"] = {
            "status": "reuse_approved" if report["passed"] else "rerun_required",
            "historical_path": report["historical_path"],
            "audit_sha256": sha256(CAMPAIGN / "FULL_REUSE_AUDIT.json"),
        }
    registry["phase_status"]["full_audit"] = "pass"
    atomic_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json", registry)
    state = read_json(CAMPAIGN / "CONTROLLER_STATE.json")
    state.update({"phase": "smokes", "updated_at_utc": now(), "next_action": f"{Path(sys.executable)} scripts/run_frozen_z_heldout_campaign.py smokes"})
    atomic_json(CAMPAIGN / "CONTROLLER_STATE.json", state)
    # Deliberately print only reuse decisions, never historical held-out metrics.
    print(json.dumps({split: report["decision"] for split, report in reports.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HeldoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
