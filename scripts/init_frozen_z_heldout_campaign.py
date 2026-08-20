#!/usr/bin/env python3
"""Atomically preregister the one-shot Z_M448_0 RefCOCO held-out campaign."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from frozen_z_heldout_common import (
    ACCEPTED_Z_CONFIG,
    ACCEPTED_Z_CONFIG_SHA256,
    BOOTSTRAP,
    CAMPAIGN,
    FROZEN_DECISION,
    FROZEN_DECISION_SHA256,
    FULL_TRACE,
    HeldoutError,
    MANIFESTS,
    MODEL,
    MODEL_CONFIG_SHA256,
    MODEL_INDEX_SHA256,
    MODEL_REVISION,
    MODEL_SHARD_SHA256,
    MODEL_SNAPSHOT,
    PROMPT,
    PYTHON,
    SOURCE_PATHS,
    THRESHOLDS,
    Z_SCHEDULE,
    Z_TRACE,
    atomic_json,
    now,
    sha256,
    verify_manifest,
)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parents[1], text=True).strip()


def source_lock() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    missing = [path for path in SOURCE_PATHS if not (repo / path).is_file()]
    if missing:
        raise HeldoutError(f"Registered source paths are missing: {missing}")
    dirty = git_output("status", "--porcelain", "--", *SOURCE_PATHS)
    if dirty:
        raise HeldoutError("Held-out executable sources must be committed and clean before locking")
    return {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": False,
        "sha256": {path: sha256(repo / path) for path in SOURCE_PATHS},
    }


def checkpoint_lock() -> dict[str, Any]:
    if not MODEL_SNAPSHOT.is_dir():
        raise HeldoutError(f"Checkpoint snapshot is absent: {MODEL_SNAPSHOT}")
    reference = MODEL_SNAPSHOT.parent.parent / "refs/main"
    if reference.read_text(encoding="utf-8").strip() != MODEL_REVISION:
        raise HeldoutError("Checkpoint refs/main differs from the frozen revision")
    index_path = MODEL_SNAPSHOT / "model.safetensors.index.json"
    if sha256(MODEL_SNAPSHOT / "config.json") != MODEL_CONFIG_SHA256:
        raise HeldoutError("Checkpoint config hash differs")
    if sha256(index_path) != MODEL_INDEX_SHA256:
        raise HeldoutError("Checkpoint index hash differs")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    records = []
    for shard in shards:
        path = MODEL_SNAPSHOT / shard
        if not path.is_file():
            raise HeldoutError(f"Checkpoint shard is absent: {path}")
        physical_sha = sha256(path)
        if physical_sha != MODEL_SHARD_SHA256.get(shard):
            raise HeldoutError(f"Checkpoint shard hash differs: {shard}")
        records.append({"name": shard, "bytes": path.stat().st_size, "sha256": physical_sha})
    return {
        "name": MODEL,
        "revision": MODEL_REVISION,
        "snapshot": str(MODEL_SNAPSHOT),
        "config_sha256": MODEL_CONFIG_SHA256,
        "index_sha256": MODEL_INDEX_SHA256,
        "shards": records,
    }


def initialize() -> dict[str, Any]:
    if CAMPAIGN.exists():
        raise HeldoutError(f"Campaign already exists: {CAMPAIGN}")
    if not PYTHON.is_file():
        raise HeldoutError(f"Required Python environment is absent: {PYTHON}")
    if sha256(FROZEN_DECISION) != FROZEN_DECISION_SHA256:
        raise HeldoutError("Frozen candidate decision hash differs")
    decision = json.loads(FROZEN_DECISION.read_text(encoding="utf-8"))
    if decision.get("selected_method") != "Z_M448_0":
        raise HeldoutError("Frozen candidate artifact does not select Z_M448_0")
    if sha256(ACCEPTED_Z_CONFIG) != ACCEPTED_Z_CONFIG_SHA256:
        raise HeldoutError("Accepted Z validation config hash differs")
    accepted = json.loads(ACCEPTED_Z_CONFIG.read_text(encoding="utf-8"))
    if accepted.get("schedule_string") != Z_SCHEDULE or accepted.get("model/max_new_tokens") != 48:
        raise HeldoutError("Accepted Z validation semantics differ")
    manifests: dict[str, Any] = {}
    for split in ("testA", "testB"):
        rows = verify_manifest(split)
        spec = MANIFESTS[split]
        manifests[split] = {
            "path": str(Path(spec["path"]).relative_to(Path(__file__).resolve().parents[1])),
            "sha256": spec["sha256"],
            "rows": spec["rows"],
            "unique_ids": len({str(row["sample_id"]) for row in rows}),
            "first_id": rows[0]["sample_id"],
            "last_id": rows[-1]["sample_id"],
        }
    free = shutil.disk_usage("/workspace").free
    if free < 80 * 1024**3:
        raise HeldoutError(f"Free storage below 80 GiB: {free}")
    source = source_lock()
    checkpoint = checkpoint_lock()
    created = now()
    commands = {
        "audit_full": f"{PYTHON} scripts/audit_frozen_z_full_controls.py",
        "smoke": f"{PYTHON} scripts/run_frozen_z_heldout_campaign.py smokes",
        "quality": f"{PYTHON} scripts/run_frozen_z_heldout_campaign.py quality",
        "analysis": f"{PYTHON} scripts/analyse_frozen_z_heldout.py",
        "terminal_audit": f"{PYTHON} scripts/audit_frozen_z_heldout.py",
    }
    protocol = {
        "schema_version": 1,
        "campaign": CAMPAIGN.name,
        "created_at_utc": created,
        "scope": "one_shot_refcoco_testA_testB_full_vs_frozen_Z_M448_0_only",
        "manifests": manifests,
        "checkpoint": checkpoint,
        "inference": {
            "precision": "bf16", "batch_size": 1, "gpu_count": 1, "image_size": [336, 336],
            "initial_visual_tokens": 576, "layers": 32, "prompt": PROMPT,
            "decoding": "deterministic_greedy", "max_new_tokens": 48, "seed": 0,
            "position_mode": "gap_preserving",
        },
        "methods": {
            "Full": {"schedule": "", "trace": FULL_TRACE, "work": sum(FULL_TRACE), "average": 576.0},
            "Z_M448_0": {"schedule": Z_SCHEDULE, "trace": Z_TRACE, "work": sum(Z_TRACE), "average": 196.0},
        },
        "exact_logical_cells": ["Full_testA", "Z_M448_0_testA", "Full_testB", "Z_M448_0_testB"],
        "historical_full_policy": "reuse_only_after_strict_audit_else_rerun_split",
        "candidate_attempts": {"testA": 1, "testB": 1},
        "outcome_embargo": "seal_both_candidate_splits_before_any_metric_analysis_or_display",
        "bootstrap": BOOTSTRAP,
        "thresholds": THRESHOLDS,
        "statuses": ["HELDOUT_CONFIRMED_BOTH_SPLITS", "HELDOUT_MIXED_SPLIT_RESULT", "HELDOUT_NOT_CONFIRMED"],
        "forbidden": ["external_methods", "other_internal_methods", "new_validation", "timing", "transfer", "training", "tuning", "alternate_seed_prompt_parser_checkpoint_schedule"],
        "frozen_inputs": {
            "candidate_decision": str(FROZEN_DECISION.relative_to(Path(__file__).resolve().parents[1])),
            "candidate_decision_sha256": FROZEN_DECISION_SHA256,
            "accepted_z_config": str(ACCEPTED_Z_CONFIG.relative_to(Path(__file__).resolve().parents[1])),
            "accepted_z_config_sha256": ACCEPTED_Z_CONFIG_SHA256,
        },
        "source": source,
        "commands": commands,
        "storage": {"free_bytes_at_lock": free, "minimum_runtime_free_bytes": 80 * 1024**3, "campaign_cap_bytes": 20 * 1024**3},
    }
    analysis_schema = {
        "schema_version": 1, "locked_at_utc": created, "split_unit": ["testA", "testB"],
        "metrics": ["parsed_box_rate", "acc_at_0_5", "mean_iou"],
        "contrast_direction": "Z_M448_0_minus_Full",
        "paired_statistics": ["acc_at_0_5", "mean_iou"], "bootstrap": BOOTSTRAP,
        "pooling": "prohibited", "decision_thresholds": THRESHOLDS,
    }
    registry = {
        "schema_version": 1, "campaign": CAMPAIGN.name, "status": "active", "created_at_utc": created,
        "phase_status": {"pre_inference_lock": "pass", "full_audit": "pending", "smokes": "pending", "quality": "pending", "analysis": "embargoed", "dissertation": "pending", "terminal_audit": "pending"},
        "cells": {
            "Full_testA": {"status": "historical_audit_pending"},
            "Z_M448_0_testA": {"status": "pending", "attempt_limit": 1},
            "Full_testB": {"status": "historical_audit_pending"},
            "Z_M448_0_testB": {"status": "pending", "attempt_limit": 1},
        },
    }
    state = {"schema_version": 1, "status": "active", "phase": "full_audit", "updated_at_utc": created, "active_workers": [], "candidate_outcome_embargo": True, "next_action": commands["audit_full"]}
    environment = {
        "locked_at_utc": created, "python_executable": str(PYTHON), "python": sys.version,
        "platform": platform.platform(), "source": source, "checkpoint": checkpoint,
        "inference_environment": {
            "HF_HOME": "/workspace/cache/huggingface",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    }
    CAMPAIGN.mkdir(parents=True, exist_ok=False)
    atomic_json(CAMPAIGN / "FROZEN_PROTOCOL.json", protocol)
    atomic_json(CAMPAIGN / "ANALYSIS_SCHEMA.json", analysis_schema)
    atomic_json(CAMPAIGN / "DECISION_THRESHOLDS.json", THRESHOLDS)
    atomic_json(CAMPAIGN / "COMMANDS.json", commands)
    atomic_json(CAMPAIGN / "SOURCE_ENVIRONMENT_LOCK.json", environment)
    atomic_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json", registry)
    atomic_json(CAMPAIGN / "CONTROLLER_STATE.json", state)
    report = (
        "# Frozen Z_M448_0 held-out confirmation\n\n"
        f"Preregistered {created}. No held-out candidate outcome existed or was inspected at lock time. "
        "The campaign contains only canonical Full and frozen Z_M448_0 on testA/testB.\n"
    )
    (CAMPAIGN / "RESEARCH_REPORT.md").write_text(report, encoding="utf-8")
    locked = ["FROZEN_PROTOCOL.json", "ANALYSIS_SCHEMA.json", "DECISION_THRESHOLDS.json", "COMMANDS.json", "SOURCE_ENVIRONMENT_LOCK.json"]
    atomic_json(CAMPAIGN / "PRE_INFERENCE_LOCK.json", {
        "schema_version": 1, "status": "locked", "locked_at_utc": created,
        "artifacts": {name: sha256(CAMPAIGN / name) for name in locked},
        "candidate_outputs_present": False,
    })
    return {"status": "initialized_and_locked", "campaign": str(CAMPAIGN), "source_commit": source["git_commit"], "manifests": manifests}


if __name__ == "__main__":
    try:
        print(json.dumps(initialize(), indent=2, sort_keys=True))
    except HeldoutError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
