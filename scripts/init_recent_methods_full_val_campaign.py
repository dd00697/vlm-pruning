#!/usr/bin/env python
"""Initialize and freeze the validation-only recent-methods campaign."""

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


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CAMPAIGN_ROOT = REPO_ROOT / "experiments/recent_methods_full_refcoco_val_20260817"
MANIFEST = REPO_ROOT / "data/manifests/final_evaluation/refcoco_val_unique.jsonl"
MANIFEST_SHA256 = "f26d7b954189e86f55b50f9506bb670d3a0cf847d3152779af6c08dbf094e3d7"
ROWS = 8811
MODEL_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
MODEL_SNAPSHOT = (
    Path("/workspace/cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots")
    / MODEL_REVISION
)
MIN_INITIAL_FREE_BYTES = 100 * 1024**3
CAMPAIGN_LIMIT_BYTES = 20 * 1024**3
MIN_RUNTIME_FREE_BYTES = 60 * 1024**3
PROMPT = (
    'Identify the target described by this referring expression: "{query}".\n'
    "Return exactly one bounding box in normalized 0-1 image coordinates as "
    "[x_min, y_min, x_max, y_max]. Do not include any other text."
)

CELLS: dict[str, dict[str, Any]] = {
    "V1_K12_48_K20_6": {
        "schedule": "12:48:k12_original,20:6:query_topk",
        "events": [
            {"boundary": 12, "budget": 48, "selector": "k12_original"},
            {"boundary": 20, "budget": 6, "selector": "query_topk"},
        ],
        "trace": [576] * 12 + [48] * 8 + [6] * 12,
        "token_layer_work": 7368,
        "average_visual_tokens": 230.25,
        "final_visual_tokens": 6,
        "runtime_implementation": "original",
    },
    "V2_Z_K12_0": {
        "schedule": "12:48:k12_original,20:0:none",
        "events": [
            {"boundary": 12, "budget": 48, "selector": "k12_original"},
            {"boundary": 20, "budget": 0, "selector": "none"},
        ],
        "trace": [576] * 12 + [48] * 8 + [0] * 12,
        "token_layer_work": 7296,
        "average_visual_tokens": 228.0,
        "final_visual_tokens": 0,
        "runtime_implementation": "original",
    },
    "V3_C448": {
        "schedule": "4:448:query_topk,12:48:query_topk",
        "events": [
            {"boundary": 4, "budget": 448, "selector": "query_topk"},
            {"boundary": 12, "budget": 48, "selector": "query_topk"},
        ],
        "trace": [576] * 4 + [448] * 8 + [48] * 20,
        "token_layer_work": 6848,
        "average_visual_tokens": 214.0,
        "final_visual_tokens": 48,
        "runtime_implementation": "original",
    },
    "V4_M448": {
        "schedule": "4:448:query_topk_merge,12:48:query_topk",
        "events": [
            {"boundary": 4, "budget": 448, "selector": "query_topk_merge"},
            {"boundary": 12, "budget": 48, "selector": "query_topk"},
        ],
        "trace": [576] * 4 + [448] * 8 + [48] * 20,
        "token_layer_work": 6848,
        "average_visual_tokens": 214.0,
        "final_visual_tokens": 48,
        "runtime_implementation": "original",
    },
    "V5_Z_M448_0": {
        "schedule": "4:448:query_topk_merge,12:48:query_topk,20:0:none",
        "events": [
            {"boundary": 4, "budget": 448, "selector": "query_topk_merge"},
            {"boundary": 12, "budget": 48, "selector": "query_topk"},
            {"boundary": 20, "budget": 0, "selector": "none"},
        ],
        "trace": [576] * 4 + [448] * 8 + [48] * 8 + [0] * 12,
        "token_layer_work": 6272,
        "average_visual_tokens": 196.0,
        "final_visual_tokens": 0,
        "runtime_implementation": "original",
    },
}

HISTORICAL_PARITY = {
    "V2_Z_K12_0": "experiments/hardware_aware_zero_tail_refcoco_val_20260816/zero_tail/runs/z_k12_0_attempt1",
    "V3_C448": "experiments/adaptive_efficiency_refcoco_val_20260815/runs/c448_attempt1",
    "V4_M448": "experiments/adaptive_efficiency_refcoco_val_20260815/runs/m448_attempt1",
    "V5_Z_M448_0": "experiments/hardware_aware_zero_tail_refcoco_val_20260816/zero_tail/runs/z_m448_0_attempt1",
}

COMPARATORS = {
    "Full": "experiments/runs_official_baselines/official_fullmatrix_refcoco_full_tokens576_seed0_bf16",
    "K12": "experiments/final_selector_ablation_and_repair_20260811/runs/repair_val_k12_b48_single_layer_more_context",
    "K20": "experiments/final_evaluation/runs/ug/final_selection_refcoco_val_k20_b48_query_max_more_anchors_bf16",
    "K12_QUERY_TOPK": "experiments/final_selector_ablation_and_repair_20260811/runs/mech_val_k12_b48_query_topk",
    "K12_STRUCTURED": "experiments/final_selector_ablation_and_repair_20260811/runs/mech_val_k12_b48_structured_tca",
}

PROTECTED = ("legacy_goal.md", "legacy_goal2.md", "PRO_REASONING_RESEARCH_HANDOFF.md")
SOURCE_PATHS = (
    "scripts/run_progressive_oracle_refcoco.py",
    "scripts/validate_progressive_oracle_run.py",
    "scripts/init_recent_methods_full_val_campaign.py",
    "scripts/run_recent_methods_full_val_campaign.py",
    "vlm_pruning/llava_hooks/progressive_pruning.py",
    "tests/test_progressive_pruning.py",
    "tests/test_recent_methods_full_val_campaign.py",
)


class InitializationError(RuntimeError):
    """Raised when the campaign cannot be frozen safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def read_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise InitializationError("Frozen full-validation manifest hash mismatch")
    rows: list[dict[str, Any]] = []
    with MANIFEST.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise InitializationError(f"Blank manifest line {line_number}")
            row = json.loads(line)
            rows.append(row)
    if len(rows) != ROWS:
        raise InitializationError(f"Expected {ROWS} manifest rows, found {len(rows)}")
    ids = [str(row.get("sample_id") or "") for row in rows]
    if len(set(ids)) != ROWS or any(not value for value in ids):
        raise InitializationError("Manifest sample IDs are empty or non-unique")
    if any(str(row.get("dataset") or "").lower() != "refcoco" for row in rows):
        raise InitializationError("Manifest contains a non-RefCOCO row")
    if any(str(row.get("split") or "").lower() != "val" for row in rows):
        raise InitializationError("Manifest contains a non-validation row")
    required = {"sample_id", "query", "image_id", "image_path", "width", "height", "gt_box"}
    if any(not required.issubset(row) for row in rows):
        raise InitializationError("Manifest row is missing a required field")
    unique_images = sorted({str(row["image_path"]) for row in rows})
    missing_images = [path for path in unique_images if not (REPO_ROOT / path).is_file()]
    if missing_images:
        raise InitializationError(f"Missing {len(missing_images)} manifest images; first={missing_images[0]}")
    return rows, {
        "path": str(MANIFEST.relative_to(REPO_ROOT)),
        "sha256": MANIFEST_SHA256,
        "rows": ROWS,
        "unique_ids": len(set(ids)),
        "unique_images": len(unique_images),
        "all_images_present": True,
        "first_id": ids[0],
        "last_id": ids[-1],
    }


def validate_checkpoint() -> dict[str, Any]:
    ref = MODEL_SNAPSHOT.parent.parent / "refs/main"
    if not ref.is_file() or ref.read_text(encoding="utf-8").strip() != MODEL_REVISION:
        raise InitializationError("Cached model ref is not the pinned revision")
    if not MODEL_SNAPSHOT.is_dir():
        raise InitializationError(f"Pinned model snapshot is missing: {MODEL_SNAPSHOT}")
    required = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    missing = [name for name in required if not (MODEL_SNAPSHOT / name).is_file()]
    if missing:
        raise InitializationError(f"Pinned model snapshot is incomplete: {missing}")
    return {
        "name": "llava-hf/llava-1.5-7b-hf",
        "revision": MODEL_REVISION,
        "snapshot": str(MODEL_SNAPSHOT),
        "index_sha256": sha256_file(MODEL_SNAPSHOT / "model.safetensors.index.json"),
        "shards": {
            name: {"bytes": (MODEL_SNAPSHOT / name).stat().st_size}
            for name in required if name.endswith(".safetensors")
        },
    }


def validate_sources() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    missing = [path for path in SOURCE_PATHS if not (REPO_ROOT / path).is_file()]
    if missing:
        raise InitializationError(f"Required campaign source is missing: {missing}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *SOURCE_PATHS], cwd=REPO_ROOT, text=True
    ).strip()
    if status:
        raise InitializationError(f"Executable campaign source is dirty:\n{status}")
    return {
        "git_commit": commit,
        "git_dirty": False,
        "sha256": {path: sha256_file(REPO_ROOT / path) for path in SOURCE_PATHS},
    }


def validate_existing_paths() -> dict[str, Any]:
    result: dict[str, Any] = {"historical_parity": {}, "comparators": {}}
    for label, relative in HISTORICAL_PARITY.items():
        root = REPO_ROOT / relative
        required = [root / "config.json", root / "predictions.jsonl", root / "COMPLETED.json"]
        if any(not path.is_file() for path in required):
            raise InitializationError(f"Historical parity evidence is incomplete for {label}")
        result["historical_parity"][label] = {
            "path": relative,
            "config_sha256": sha256_file(root / "config.json"),
            "predictions_sha256": sha256_file(root / "predictions.jsonl"),
            "completion_sha256": sha256_file(root / "COMPLETED.json"),
        }
    for label, relative in COMPARATORS.items():
        root = REPO_ROOT / relative
        predictions = root / "per_sample_results.jsonl"
        required = [root / "config.json", predictions, root / "metrics.json"]
        if any(not path.is_file() for path in required):
            raise InitializationError(f"Comparator evidence is incomplete for {label}")
        result["comparators"][label] = {
            "path": relative,
            "config_sha256": sha256_file(root / "config.json"),
            "predictions_sha256": sha256_file(predictions),
            "metrics_sha256": sha256_file(root / "metrics.json"),
        }
    return result


def initialize() -> dict[str, Any]:
    if CAMPAIGN_ROOT.exists():
        raise InitializationError(f"Campaign root already exists: {CAMPAIGN_ROOT}")
    usage = shutil.disk_usage("/workspace")
    if usage.free < MIN_INITIAL_FREE_BYTES:
        raise InitializationError(f"Initial free space is below 100 GiB: {usage.free}")
    _, manifest = read_manifest()
    checkpoint = validate_checkpoint()
    sources = validate_sources()
    evidence = validate_existing_paths()
    protected = {path: sha256_file(REPO_ROOT / path) for path in PROTECTED}
    created = utc_now()
    cell_records = {
        cell_id: {
            **cell,
            "cell_id": cell_id,
            "seed": 0,
            "full_run_name": f"{cell_id.lower()}_attempt1",
            "full_output": f"runs/{cell_id.lower()}_attempt1",
            "smoke2_output": f"smokes/two/{cell_id.lower()}_attempt1",
            "smoke20_output": f"smokes/twenty/{cell_id.lower()}_attempt1",
            "status": "pending",
        }
        for cell_id, cell in CELLS.items()
    }
    protocol = {
        "schema_version": 1,
        "campaign": "recent_methods_full_refcoco_val_20260817",
        "created_at_utc": created,
        "scope": "refcoco_val_full_validation_only",
        "manifest": manifest,
        "model": checkpoint,
        "prompt": PROMPT,
        "precision": "bf16",
        "batch_size": 1,
        "image_size": [336, 336],
        "initial_visual_tokens": 576,
        "layers": 32,
        "position_mode": "gap_preserving",
        "decoding": "deterministic_greedy",
        "max_new_tokens": 48,
        "inference_seed": 0,
        "cells": cell_records,
        "comparators": COMPARATORS,
        "historical_parity": HISTORICAL_PARITY,
        "analysis": {
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 20260817,
            "metrics": ["acc_at_0_5", "mean_iou", "median_iou", "parse_rate"],
            "contrasts": [
                ["V1_K12_48_K20_6", "V2_Z_K12_0"],
                ["V1_K12_48_K20_6", "K12"],
                ["V2_Z_K12_0", "K12"],
                ["V4_M448", "V3_C448"],
                ["V5_Z_M448_0", "V4_M448"],
                ["V4_M448", "K12"],
                ["V5_Z_M448_0", "K12"],
                ["V1_K12_48_K20_6", "Full"],
                ["V2_Z_K12_0", "Full"],
                ["V3_C448", "Full"],
                ["V4_M448", "Full"],
                ["V5_Z_M448_0", "Full"],
                ["V1_K12_48_K20_6", "K20"],
                ["V2_Z_K12_0", "K20"],
                ["V3_C448", "K20"],
                ["V4_M448", "K20"],
                ["V5_Z_M448_0", "K20"],
            ],
            "main_table_order": [
                "Full", "K20", "K12", "V1_K12_48_K20_6", "V2_Z_K12_0",
                "V3_C448", "V4_M448", "V5_Z_M448_0",
            ],
            "required_figure": "two_panel_quality_vs_token_layer_work",
        },
        "forbidden": {
            "splits": ["testA", "testB"],
            "activities": [
                "training", "matched_timing", "external_baselines", "transfer",
                "learned_router", "budget_search", "random_controls",
            ],
        },
        "storage": {
            "campaign_limit_bytes": CAMPAIGN_LIMIT_BYTES,
            "minimum_runtime_free_bytes": MIN_RUNTIME_FREE_BYTES,
            "initial_free_bytes": usage.free,
        },
        "protected_sha256": protected,
        "source": sources,
        "report_schema": {
            "mandatory_table_methods": [
                "Full", "K12", "K20", "V1_K12_48_K20_6", "V2_Z_K12_0",
                "V3_C448", "V4_M448", "V5_Z_M448_0",
            ],
            "mandatory_columns": [
                "acc_at_0_5", "mean_iou", "final_visual_tokens",
                "average_visual_tokens", "token_layer_work",
            ],
            "mandatory_topics": [
                "six_vs_zero_tail", "merge_vs_deletion", "zero_tail_196",
                "token_proxy_vs_latency", "router_oracle_e256_saqr", "post_hoc_status",
            ],
        },
    }
    registry = {
        "schema_version": 1,
        "campaign": protocol["campaign"],
        "created_at_utc": created,
        "status": "initialized",
        "manifest": manifest,
        "source": sources,
        "cells": cell_records,
        "comparators": evidence["comparators"],
        "historical_parity": evidence["historical_parity"],
        "protected_sha256": protected,
        "phase_status": {
            "preflight": "pass",
            "tests": "pending",
            "smoke2": "pending",
            "smoke20": "pending",
            "historical_parity": "pending",
            "parallel_soak": "pending",
            "full_validation": "pending",
            "analysis": "pending",
            "dissertation": "pending",
            "final_audit": "pending",
        },
    }
    state = {
        "schema_version": 1,
        "campaign": protocol["campaign"],
        "updated_at_utc": created,
        "phase": "initialized",
        "terminal": False,
        "next_action": "run_complete_cpu_suite",
        "active_workers": [],
    }
    preflight = {
        "status": "pass",
        "created_at_utc": created,
        "manifest": manifest,
        "checkpoint": checkpoint,
        "storage": protocol["storage"],
        "source": sources,
        "evidence": evidence,
        "protected_sha256": protected,
    }
    report = (
        "# Recent Methods Full RefCOCO-Validation Campaign\n\n"
        "Status: initialized; no physical result has been run.\n\n"
        "The frozen protocol registers exactly five deterministic 8,811-row RefCOCO-validation "
        "cells. Test splits, timing, training, transfer tasks, external baselines, learned routing, "
        "and unregistered budgets are forbidden.\n"
    )
    atomic_write(CAMPAIGN_ROOT / "FROZEN_PROTOCOL.json", canonical_bytes(protocol))
    atomic_write(CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json", canonical_bytes(registry))
    atomic_write(CAMPAIGN_ROOT / "CONTROLLER_STATE.json", canonical_bytes(state))
    atomic_write(CAMPAIGN_ROOT / "PREFLIGHT_REPORT.json", canonical_bytes(preflight))
    atomic_write(CAMPAIGN_ROOT / "RESEARCH_REPORT.md", report.encode("utf-8"))
    return preflight


def main() -> int:
    try:
        report = initialize()
    except (InitializationError, OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps({"status": report["status"], "campaign_root": str(CAMPAIGN_ROOT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
