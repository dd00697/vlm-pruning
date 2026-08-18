#!/usr/bin/env python3
"""Initialize the frozen 2026-08-18 RefCOCO-val efficiency campaign."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "experiments/final_val_efficiency_baselines_20260818"
FULL_MANIFEST = REPO / "data/manifests/final_evaluation/refcoco_val_unique.jsonl"
TIMING_MANIFEST = REPO / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
FULL_SHA = "f26d7b954189e86f55b50f9506bb670d3a0cf847d3152779af6c08dbf094e3d7"
TIMING_SHA = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
MODEL_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
MODEL_SNAPSHOT = Path("/workspace/cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots") / MODEL_REVISION
PROMPT = (
    'Identify the target described by this referring expression: "{query}".\n'
    "Return exactly one bounding box in normalized 0-1 image coordinates as "
    "[x_min, y_min, x_max, y_max]. Do not include any other text."
)
OFFICIAL = {
    "fastv": {
        "path": REPO / "third_party/FastV",
        "commit": "d1659729b5bf1be225e99ee15783deeea80f63b1",
        "mode": "source_native_only",
    },
    "feather": {
        "path": Path("/workspace/third_party_official/FEATHER"),
        "commit": "c2a09b2765967601054c7b1fd513ac3e94ee4fc9",
        "mode": "source_faithful_llava_adapter_with_exact_parity_gate",
    },
    "nuwa": {
        "path": REPO / "experiments/adaptive_efficiency_refcoco_val_20260815/official_sources/checkouts/nuwa",
        "commit": "175b7d57a7890b9818dd26d82f2d91d9868b0d80",
        "mode": "source_native_only_or_audited_skip",
    },
}
EVIDENCE = {
    "Full": "experiments/runs_official_baselines/official_fullmatrix_refcoco_full_tokens576_seed0_bf16",
    "K20": "experiments/final_evaluation/runs/ug/final_selection_refcoco_val_k20_b48_query_max_more_anchors_bf16",
    "K12": "experiments/final_selector_ablation_and_repair_20260811/runs/repair_val_k12_b48_single_layer_more_context",
    "M448": "experiments/recent_methods_full_refcoco_val_20260817/runs/v4_m448_attempt1",
    "Z_M448_0": "experiments/recent_methods_full_refcoco_val_20260817/runs/v5_z_m448_0_attempt1",
}
INTERNAL = {
    "Full": {"trace": [576] * 32, "work": 18432, "average": 576.0, "final": 576},
    "K12": {"trace": [576] * 12 + [48] * 20, "work": 7872, "average": 246.0, "final": 48},
    "M448_OPT": {"trace": [576] * 4 + [448] * 8 + [48] * 20, "work": 6848, "average": 214.0, "final": 48},
    "Z_M448_0": {"trace": [576] * 4 + [448] * 8 + [48] * 8 + [0] * 12, "work": 6272, "average": 196.0, "final": 0},
}
INTERNAL_TIMING_ORDER = {
    "1": ["Full", "K12", "M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN"],
    "2": ["K12", "M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN", "Full"],
    "3": ["M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN", "Full", "K12"],
    "4": ["Z_M448_0", "FEATHER_LLAVA_MAIN", "Full", "K12", "M448_OPT"],
    "5": ["FEATHER_LLAVA_MAIN", "Full", "K12", "M448_OPT", "Z_M448_0"],
}
FASTV_TIMING_ORDER = {
    "1": ["FASTV_STACK_FULL", "FASTV_MATCHED_199"],
    "2": ["FASTV_MATCHED_199", "FASTV_STACK_FULL"],
    "3": ["FASTV_STACK_FULL", "FASTV_MATCHED_199"],
    "4": ["FASTV_MATCHED_199", "FASTV_STACK_FULL"],
    "5": ["FASTV_STACK_FULL", "FASTV_MATCHED_199"],
}


class InitError(RuntimeError):
    """A frozen prerequisite differs from the registered campaign."""


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def manifest_record(path: Path, expected_sha: str, expected_rows: int) -> dict[str, Any]:
    if sha256(path) != expected_sha:
        raise InitError(f"Manifest hash mismatch: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise InitError(f"Blank manifest row {number}: {path}")
            rows.append(json.loads(line))
    if len(rows) != expected_rows:
        raise InitError(f"Expected {expected_rows} rows in {path}, found {len(rows)}")
    ids = [str(row.get("sample_id") or "") for row in rows]
    if len(set(ids)) != expected_rows or any(not value for value in ids):
        raise InitError(f"Manifest IDs are not unique and non-empty: {path}")
    if any(str(row.get("dataset") or "").lower() != "refcoco" for row in rows):
        raise InitError(f"Non-RefCOCO row in {path}")
    if any(str(row.get("split") or "").lower() != "val" for row in rows):
        raise InitError(f"Non-validation row in {path}")
    missing = [str(row["image_path"]) for row in rows if not (REPO / str(row["image_path"])).is_file()]
    if missing:
        raise InitError(f"Missing manifest image: {missing[0]}")
    return {
        "path": str(path.relative_to(REPO)), "sha256": expected_sha, "rows": expected_rows,
        "unique_ids": len(set(ids)), "first_id": ids[0], "last_id": ids[-1],
        "unique_images": len({str(row["image_path"]) for row in rows}), "all_images_present": True,
    }


def recursive_source_hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file() and ".git" not in candidate.parts):
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(item)))
        count += 1
    return digest.hexdigest(), count


def official_record(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    path = Path(spec["path"])
    if not path.is_dir():
        raise InitError(f"Missing official checkout: {path}")
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(path), "status", "--short"], text=True).strip()
    if commit != spec["commit"] or status:
        raise InitError(f"Official {name} source lock differs: commit={commit}, dirty={bool(status)}")
    tree_hash, files = recursive_source_hash(path)
    return {
        "path": str(path), "commit": commit, "clean": True, "recursive_sha256": tree_hash,
        "files": files, "execution_mode": spec["mode"],
    }


def evidence_record(relative: str) -> dict[str, Any]:
    root = REPO / relative
    prediction = root / "predictions.jsonl"
    if not prediction.is_file():
        prediction = root / "per_sample_results.jsonl"
    required = [root / "config.json", root / "metrics.json", prediction]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise InitError(f"Historical evidence is incomplete: {missing}")
    return {
        "path": relative,
        "config_sha256": sha256(root / "config.json"),
        "metrics_sha256": sha256(root / "metrics.json"),
        "predictions": str(prediction.relative_to(REPO)),
        "predictions_sha256": sha256(prediction),
    }


def initialize() -> dict[str, Any]:
    if CAMPAIGN.exists():
        raise InitError(f"Campaign already exists: {CAMPAIGN}")
    free = shutil.disk_usage("/workspace").free
    if free < 120 * 1024**3:
        raise InitError(f"Initial free storage below 120 GiB: {free}")
    full = manifest_record(FULL_MANIFEST, FULL_SHA, 8811)
    timing = manifest_record(TIMING_MANIFEST, TIMING_SHA, 1000)
    if not MODEL_SNAPSHOT.is_dir():
        raise InitError(f"Canonical model snapshot is missing: {MODEL_SNAPSHOT}")
    refs_main = MODEL_SNAPSHOT.parent.parent / "refs/main"
    if not refs_main.is_file() or refs_main.read_text().strip() != MODEL_REVISION:
        raise InitError("Canonical checkpoint revision differs")
    official = {name: official_record(name, spec) for name, spec in OFFICIAL.items()}
    historical = {name: evidence_record(path) for name, path in EVIDENCE.items()}
    protected_names = ["AGENTS.md", "GOAL.md", "legacy_goal.md", "legacy_goal2.md", "PRO_REASONING_RESEARCH_HANDOFF.md"]
    protected = {name: sha256(REPO / name) for name in protected_names}
    created = now()
    protocol = {
        "schema_version": 1,
        "campaign": CAMPAIGN.name,
        "created_at_utc": created,
        "scope": "RefCOCO_val_only_no_heldout_no_training",
        "manifests": {"quality": full, "timing": timing},
        "checkpoint": {"name": "llava-hf/llava-1.5-7b-hf", "revision": MODEL_REVISION, "snapshot": str(MODEL_SNAPSHOT)},
        "inference": {"precision": "bf16", "batch_size": 1, "gpu_count": 1, "image_size": [336, 336], "initial_visual_tokens": 576, "layers": 32, "prompt": PROMPT, "decoding": "greedy", "max_new_tokens": 48, "seed": 0},
        "official_sources": official,
        "historical_evidence": historical,
        "internal_methods": INTERNAL,
        "external_quality_cells": {
            "E1_FASTV_MATCHED_199": {"status": "pending", "source_mode": "native", "fastv_k": 3, "target_tokens": 160, "trace": [576] * 3 + [160] * 29, "work": 6368, "average": 199.0},
            "E2_FEATHER_LLAVA_MAIN": {"status": "pending", "source_mode": "parity_proven_adapter", "boundaries": [8, 16], "upstream_reduction_ratio": 0.7, "keep_ratio": 0.3, "predefined_mask": "no_rope_s=3", "adaptive": True},
            "N0_NUWA_STACK_FULL": {"status": "pending_native_or_skip"},
            "E3_NUWA_NATIVE_192": {"status": "pending_native_or_skip", "NT": 192, "DIST": 280, "trace": [336] * 16 + [48] * 16, "work": 6144, "average": 192.0},
        },
        "timing": {"repetitions": 5, "warmup_examples": 20, "bootstrap_resamples": 10000, "bootstrap_seed": 20260818, "canonical_order": INTERNAL_TIMING_ORDER, "fastv_native_order": FASTV_TIMING_ORDER, "nuwa_native_order": FASTV_TIMING_ORDER, "primary": "natural_end_to_end_latency", "fixed32_trigger": {"mean_generated_token_difference_fraction": 0.02, "near_zero_e2e_fraction": 0.02, "repetitions": 3}},
        "selection_rule": {"eligible": ["K12", "M448_OPT", "Z_M448_0"], "minimum_parse_rate": 0.995, "maximum_acc_drop_pp": 3.0, "maximum_miou_drop": 0.015, "requires_mean_e2e_below_full": True, "requires_upper_paired_ci_below_zero": True, "objective": "minimum_visual_token_layer_work", "tie_breaks": ["lower_e2e", "higher_miou", "higher_acc"]},
        "exclusions": ["RefCOCO_testA", "RefCOCO_testB", "transfer", "training", "learned_router", "unregistered_budget"],
        "storage": {"initial_free_bytes": free, "minimum_initial_bytes": 120 * 1024**3, "minimum_runtime_bytes": 80 * 1024**3, "campaign_cap_bytes": 35 * 1024**3},
        "protected_inputs": protected,
    }
    registry = {
        "schema_version": 1, "campaign": CAMPAIGN.name, "status": "active", "created_at_utc": created,
        "phase_status": {"preflight": "pass", "implementation": "pending", "smokes": "pending", "quality": "pending", "timing": "pending", "analysis": "pending", "dissertation": "pending", "final_audit": "pending"},
        "cells": protocol["external_quality_cells"], "audited_skips": {},
    }
    state = {"schema_version": 1, "status": "active", "phase": "implementation", "updated_at_utc": created, "active_workers": [], "next_action": "compile_and_run_full_unit_suite"}
    report = (
        "# Final validation efficiency and baseline campaign\n\n"
        f"Initialized {created}. This report is cumulative and validation-only. No RefCOCO testA/testB, transfer evaluation, training or learned routing is authorized.\n\n"
        "## Frozen design\n\n"
        "The campaign compares Full, K12, M448_OPT and Z_M448_0 with a parity-proven FEATHER LLaVA adapter; official FastV and official Nüwa are isolated native-stack comparisons. FastV and Nüwa may not be reimplemented.\n"
    )
    atomic_json(CAMPAIGN / "FROZEN_PROTOCOL.json", protocol)
    atomic_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json", registry)
    atomic_json(CAMPAIGN / "CONTROLLER_STATE.json", state)
    (CAMPAIGN / "RESEARCH_REPORT.md").write_text(report, encoding="utf-8")
    return {"status": "initialized", "campaign": str(CAMPAIGN), "full_rows": 8811, "timing_rows": 1000, "official_sources": {name: value["commit"] for name, value in official.items()}}


if __name__ == "__main__":
    print(json.dumps(initialize(), indent=2, sort_keys=True))
