#!/usr/bin/env python
"""Validate the historical K20/B48 reference and initialise campaign metadata.

This module is deliberately independent of the historical run's own integrity
report.  It reads the physical prediction JSONL, verifies the frozen protocol,
recomputes every box/IoU from raw text, and writes only new campaign-local
reference and registry artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814"
FULL_MANIFEST = REPO_ROOT / "data/manifests/final_evaluation/refcoco_val_unique.jsonl"
DISCOVERY_MANIFEST = CAMPAIGN_ROOT / "manifests/refcoco_val_discovery1000.jsonl"
DISCOVERY_LOCK = CAMPAIGN_ROOT / "manifests/discovery_manifest_lock.json"
HISTORICAL_RUN = (
    REPO_ROOT
    / "experiments/final_selector_ablation_and_repair_20260811/runs/mech_val_k20_b48_query_topk"
)
REFERENCE_DIR = CAMPAIGN_ROOT / "references"
REFERENCE_REPORT = REFERENCE_DIR / "REF_K20_B48.json"
REFERENCE_ROWS = REFERENCE_DIR / "REF_K20_B48_per_sample.jsonl"
REGISTRY_PATH = CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"
REPORT_PATH = CAMPAIGN_ROOT / "RESEARCH_REPORT.md"

EXPECTED_FULL_ROWS = 8811
EXPECTED_DISCOVERY_ROWS = 1000
EXPECTED_FULL_MANIFEST_SHA256 = "f26d7b954189e86f55b50f9506bb670d3a0cf847d3152779af6c08dbf094e3d7"
EXPECTED_PREDICTIONS_SHA256 = "95332c4a110ef1624c4fedc9e6eb9b28f3f51e5f50023d065e172920aeb06f9e"
EXPECTED_PROMPT = (
    'Identify the target described by this referring expression: "{query}".\n'
    "Return exactly one bounding box in normalized 0-1 image coordinates as "
    "[x_min, y_min, x_max, y_max]. Do not include any other text."
)


class CampaignInitialisationError(RuntimeError):
    """Raised when reference evidence or campaign metadata is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def canonical_jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8") for row in rows
    )


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_jsonl_strict(path: Path, *, expected_rows: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CampaignInitialisationError(f"Missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise CampaignInitialisationError(f"Blank line at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CampaignInitialisationError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise CampaignInitialisationError(f"Non-object row at {path}:{line_number}")
            rows.append(row)
    if expected_rows is not None and len(rows) != expected_rows:
        raise CampaignInitialisationError(f"Expected {expected_rows} rows in {path}, found {len(rows)}")
    return rows


def parse_box(text: str, width: int, height: int) -> tuple[list[float] | None, list[float] | None, str | None]:
    """Registered normalized-0-1 parser, kept local for independent rescoring."""
    import re

    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))
    if len(matches) < 4:
        return None, None, "fewer_than_four_numbers"
    try:
        raw = [float(value) for value in matches[:4]]
    except ValueError:
        return None, None, "non_numeric_box"
    if not all(math.isfinite(value) for value in raw):
        return None, raw, "non_finite_box"
    x1, y1, x2, y2 = raw
    if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
        return None, raw, "outside_normalized_0_1"
    predicted = [x1 * width, y1 * height, x2 * width, y2 * height]
    if predicted[2] <= predicted[0] or predicted[3] <= predicted[1]:
        return None, raw, "degenerate_box"
    return predicted, raw, None


def box_iou(a: list[float], b: list[float]) -> float:
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def indices_checksum(indices: list[int]) -> str:
    return sha256_bytes(json.dumps(indices, separators=(",", ":")).encode("utf-8"))[:16]


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise CampaignInitialisationError(f"{label}: expected {expected!r}, found {actual!r}")


def _assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise CampaignInitialisationError(f"{label}: expected {expected!r}, found {actual!r}")


def validate_source_revision(repo_root: Path, config: dict[str, Any]) -> dict[str, str]:
    commit = str(config.get("source/git_commit") or "")
    if len(commit) != 40 or bool(config.get("source/git_dirty")):
        raise CampaignInitialisationError("Historical source revision is absent, abbreviated, or dirty")
    recorded = config.get("source/sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise CampaignInitialisationError("Historical source hashes are missing")
    verified: dict[str, str] = {}
    for relative, expected in sorted(recorded.items()):
        try:
            payload = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=repo_root)
        except subprocess.CalledProcessError as exc:
            raise CampaignInitialisationError(f"Cannot retrieve historical source {relative}") from exc
        actual = sha256_bytes(payload)
        _assert_equal(actual, expected, f"historical source hash {relative}")
        verified[str(relative)] = actual
    return verified


def validate_protocol(config: dict[str, Any], full_manifest: Path) -> dict[str, Any]:
    checks = {
        "dataset/manifest_checksum": EXPECTED_FULL_MANIFEST_SHA256,
        "dataset/name": "refcoco",
        "dataset/num_samples": EXPECTED_FULL_ROWS,
        "dataset/split": "val",
        "method/seed": 0,
        "model/decoding": "deterministic_greedy",
        "model/max_new_tokens": 48,
        "model/name": "llava-hf/llava-1.5-7b-hf",
        "model/precision": "bf16",
        "position_mode": "gap_preserving",
        "prompt/template": EXPECTED_PROMPT,
        "pruning_implementation": "mid_layer_single_pass",
        "pruning_location": "after_layer_20",
        "run/is_limited": False,
        "selection/method": "ug",
        "selection/strategy": "query_topk",
        "tokens/actual_visual_tokens": 48,
        "tokens/original_visual_tokens": 576,
        "tokens/target_visual_tokens": 48,
        "ug/decode_mode": "layer_aware_cache",
        "ug/diagnostics_mode": "benchmark",
        "ug/fusion_mode": "single_layer",
        "ug/map_source": "query_max",
        "ug/position_mode": "gap_preserving",
        "ug/probe_layers": "20",
        "ug/prune_after_layer": 20,
    }
    for key, expected in checks.items():
        _assert_equal(config.get(key), expected, f"config {key}")
    _assert_equal(sha256_file(full_manifest), EXPECTED_FULL_MANIFEST_SHA256, "full manifest hash")
    return checks


def validate_reference(
    *,
    repo_root: Path = REPO_ROOT,
    historical_run: Path = HISTORICAL_RUN,
    full_manifest: Path = FULL_MANIFEST,
    discovery_manifest: Path = DISCOVERY_MANIFEST,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_path = historical_run / "config.json"
    prediction_path = historical_run / "per_sample_results.jsonl"
    if not config_path.is_file() or not prediction_path.is_file():
        raise CampaignInitialisationError("Historical config or physical predictions are missing")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol_checks = validate_protocol(config, full_manifest)
    source_hashes = validate_source_revision(repo_root, config)
    prediction_sha = sha256_file(prediction_path)
    _assert_equal(prediction_sha, EXPECTED_PREDICTIONS_SHA256, "historical prediction hash")

    manifests = read_jsonl_strict(full_manifest, expected_rows=EXPECTED_FULL_ROWS)
    discovery = read_jsonl_strict(discovery_manifest, expected_rows=EXPECTED_DISCOVERY_ROWS)
    manifest_ids = [str(row["sample_id"]) for row in manifests]
    discovery_ids = [str(row["sample_id"]) for row in discovery]
    if len(set(manifest_ids)) != EXPECTED_FULL_ROWS or len(set(discovery_ids)) != EXPECTED_DISCOVERY_ROWS:
        raise CampaignInitialisationError("Manifest IDs are not unique")
    if not set(discovery_ids).issubset(set(manifest_ids)):
        raise CampaignInitialisationError("Discovery IDs are not a subset of the full manifest")

    predictions = read_jsonl_strict(prediction_path, expected_rows=EXPECTED_FULL_ROWS)
    prediction_ids = [str(row.get("sample_id") or "") for row in predictions]
    _assert_equal(prediction_ids, manifest_ids, "historical prediction order")
    if len(set(prediction_ids)) != EXPECTED_FULL_ROWS:
        raise CampaignInitialisationError("Historical prediction IDs are not unique")

    discovery_set = set(discovery_ids)
    subset_by_id: dict[str, dict[str, Any]] = {}
    full_iou_sum = 0.0
    full_success_sum = 0
    parse_count = 0
    for index, (manifest, prediction) in enumerate(zip(manifests, predictions, strict=True)):
        sample_id = str(manifest["sample_id"])
        for key in ("dataset", "split", "query", "image_id", "image_path", "gt_box"):
            _assert_equal(prediction.get(key), manifest.get(key), f"row {index} {key}")
        _assert_equal(int(prediction.get("original_visual_tokens", -1)), 576, f"row {index} original budget")
        for key in (
            "actual_llm_visual_tokens",
            "actual_retained_visual_tokens",
            "expected_retained_visual_tokens",
            "target_visual_tokens",
            "tokens/actual_visual_tokens",
        ):
            _assert_equal(int(prediction.get(key, -1)), 48, f"row {index} {key}")
        _assert_equal(prediction.get("selector_strategy"), "query_topk", f"row {index} selector")
        _assert_equal(prediction.get("position_ids_mode"), "gap_preserving", f"row {index} positions")
        _assert_equal(prediction.get("decode_mode"), "layer_aware_cache", f"row {index} decode")
        retained = prediction.get("retained_indices")
        if not isinstance(retained, list) or retained != sorted(retained) or len(retained) != 48:
            raise CampaignInitialisationError(f"row {index} retained indices are malformed")
        if len(set(retained)) != 48 or any(type(value) is not int or value < 0 or value >= 576 for value in retained):
            raise CampaignInitialisationError(f"row {index} retained indices fail uniqueness/range")
        _assert_equal(prediction.get("retained_indices_checksum"), indices_checksum(retained), f"row {index} index checksum")
        original_length = int(prediction.get("original_input_sequence_length", -1))
        compacted_length = int(prediction.get("compacted_input_sequence_length", -1))
        _assert_equal(compacted_length, original_length - 528, f"row {index} compacted length")
        generated_cached = int(prediction.get("mid_layer/cache_generated_tokens", -1))
        _assert_equal(
            int(prediction.get("mid_layer/early_cache_seq_len", -1)),
            original_length + generated_cached,
            f"row {index} early cache length",
        )
        _assert_equal(
            int(prediction.get("mid_layer/late_cache_seq_len", -1)),
            compacted_length + generated_cached,
            f"row {index} late cache length",
        )
        _assert_equal(
            int(prediction.get("mid_layer/early_attention_mask_len", -1)),
            original_length + generated_cached,
            f"row {index} early mask length",
        )
        _assert_equal(
            int(prediction.get("mid_layer/late_attention_mask_len", -1)),
            compacted_length + generated_cached,
            f"row {index} late mask length",
        )

        predicted_box, raw_box, parse_reason = parse_box(
            str(prediction.get("raw_model_output") or ""), int(manifest["width"]), int(manifest["height"])
        )
        parsed = predicted_box is not None
        iou = box_iou(predicted_box, list(manifest["gt_box"])) if predicted_box is not None else 0.0
        success = bool(iou >= 0.5)
        _assert_equal(prediction.get("parse_failure_reason"), parse_reason, f"row {index} parse reason")
        _assert_equal(bool(prediction.get("invalid")), not parsed, f"row {index} invalid")
        _assert_equal(bool(prediction.get("success_iou_0_5")), success, f"row {index} success")
        _assert_close(float(prediction.get("box_iou", -1.0)), iou, f"row {index} IoU")
        if raw_box is not None:
            for component, stored in zip(raw_box, prediction.get("raw_parsed_box") or [], strict=True):
                _assert_close(float(stored), component, f"row {index} raw box")
        if predicted_box is not None:
            for component, stored in zip(predicted_box, prediction.get("predicted_box") or [], strict=True):
                _assert_close(float(stored), component, f"row {index} predicted box")
        full_iou_sum += iou
        full_success_sum += int(success)
        parse_count += int(parsed)
        if sample_id in discovery_set:
            subset_by_id[sample_id] = {
                "sample_id": sample_id,
                "raw_model_output": str(prediction.get("raw_model_output") or ""),
                "raw_parsed_box": raw_box,
                "predicted_box": predicted_box,
                "gt_box": manifest["gt_box"],
                "box_iou": iou,
                "success_iou_0_5": success,
                "parsed": parsed,
            }

    subset = [subset_by_id[sample_id] for sample_id in discovery_ids]
    if len(subset) != EXPECTED_DISCOVERY_ROWS:
        raise CampaignInitialisationError("Historical predictions do not cover every discovery row")
    subset_iou = sum(float(row["box_iou"]) for row in subset)
    subset_success = sum(int(bool(row["success_iou_0_5"])) for row in subset)
    subset_parsed = sum(int(bool(row["parsed"])) for row in subset)
    lock = json.loads(DISCOVERY_LOCK.read_text(encoding="utf-8"))
    _assert_equal(sha256_file(discovery_manifest), lock.get("output_manifest_sha256"), "discovery manifest lock")
    report = {
        "reference_id": "REF_K20_B48",
        "status": "pass",
        "validated_at_utc": utc_now(),
        "historical_run": str(historical_run.relative_to(repo_root)),
        "historical_config_sha256": sha256_file(config_path),
        "historical_predictions_path": str(prediction_path.relative_to(repo_root)),
        "historical_predictions_sha256": prediction_sha,
        "historical_completion_sentinel_present": (historical_run / "COMPLETED.json").is_file(),
        "historical_partial_present": (historical_run / "per_sample_results.partial.jsonl").is_file(),
        "historical_partial_sha256": (
            sha256_file(historical_run / "per_sample_results.partial.jsonl")
            if (historical_run / "per_sample_results.partial.jsonl").is_file()
            else None
        ),
        "acceptance_basis": "independent physical JSONL validation; historical integrity metadata was not trusted",
        "protocol_checks": protocol_checks,
        "verified_historical_source_sha256": source_hashes,
        "full_manifest_sha256": sha256_file(full_manifest),
        "full_rows": EXPECTED_FULL_ROWS,
        "full_recomputed": {
            "acc_at_0_5": full_success_sum / EXPECTED_FULL_ROWS,
            "mean_iou": full_iou_sum / EXPECTED_FULL_ROWS,
            "parse_rate": parse_count / EXPECTED_FULL_ROWS,
        },
        "discovery_manifest_sha256": sha256_file(discovery_manifest),
        "discovery_rows": EXPECTED_DISCOVERY_ROWS,
        "discovery_recomputed": {
            "acc_at_0_5": subset_success / EXPECTED_DISCOVERY_ROWS,
            "mean_iou": subset_iou / EXPECTED_DISCOVERY_ROWS,
            "parse_rate": subset_parsed / EXPECTED_DISCOVERY_ROWS,
        },
    }
    return report, subset


def registered_cells() -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    for cell_id, budget in (("T1", 32), ("T2", 16), ("T3", 8), ("T4", 4), ("T5", 0)):
        cells[cell_id] = {
            "campaign": 1,
            "scope": "discovery",
            "deployable": True,
            "schedule": [{"boundary": 20, "budget": budget, "selector": "query_topk" if budget else "none"}],
            "seed": 0,
            "status": "pending",
            "attempts": [],
        }
    for budget in (16, 8, 4):
        for seed in (0, 1, 2):
            cell_id = f"R{budget}-S{seed}"
            cells[cell_id] = {
                "campaign": 1,
                "scope": "discovery",
                "deployable": True,
                "schedule": [{"boundary": 20, "budget": budget, "selector": "random"}],
                "seed": seed,
                "status": "pending",
                "attempts": [],
            }
    schedules = {
        "P1": [(8, 256), (12, 48)],
        "P2": [(8, 192), (12, 48)],
        "P3": [(4, 384), (12, 48)],
        "P4": [(4, 256), (12, 48)],
        "P5": [(2, 384), (12, 48)],
        "P6": [(2, 288), (12, 48)],
    }
    for cell_id, schedule in schedules.items():
        cells[cell_id] = {
            "campaign": 2,
            "scope": "discovery",
            "deployable": True,
            "schedule": [{"boundary": layer, "budget": budget, "selector": "query_topk"} for layer, budget in schedule],
            "seed": 0,
            "status": "pending",
            "attempts": [],
        }
    for cell_id in ("P7", "P8", "P9"):
        cells[cell_id] = {
            "campaign": 2,
            "scope": "discovery",
            "deployable": True,
            "schedule": "determined_by_registered_campaign_decision",
            "seed": 0,
            "status": "pending",
            "attempts": [],
        }
    for cell_id, teacher, budget in (
        ("O1", 12, 64), ("O2", 12, 96), ("O3", 12, 128),
        ("O4", 20, 64), ("O5", 20, 96), ("O6", 20, 128),
    ):
        cells[cell_id] = {
            "campaign": 3,
            "scope": "discovery",
            "deployable": False,
            "oracle_teacher_layer": teacher,
            "pre_llm_budget": budget,
            "schedule": [],
            "seed": 0,
            "status": "pending",
            "attempts": [],
        }
    for cell_id, teacher in (("O7", 12), ("O8", 20)):
        cells[cell_id] = {
            "campaign": 3,
            "scope": "discovery",
            "deployable": False,
            "oracle_teacher_layer": teacher,
            "pre_llm_budget": "B_early64",
            "schedule": "K20_to_B_tail",
            "seed": 0,
            "status": "pending",
            "attempts": [],
        }
    cells["O9"] = {
        "campaign": 3, "scope": "discovery", "deployable": True,
        "pre_llm_budget": 64, "pre_llm_selector": "random", "schedule": [], "seed": 0,
        "status": "pending", "attempts": [],
    }
    cells["O10"] = {
        "campaign": 3, "scope": "discovery", "deployable": True,
        "pre_llm_budget": 64, "pre_llm_selector": "uniform", "schedule": [], "seed": 0,
        "status": "pending", "attempts": [],
    }
    if len(cells) != 33:
        raise AssertionError(f"Registered {len(cells)} quality cells, expected 33")
    return cells


def initialise_registry(reference: dict[str, Any], *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    lock = json.loads(DISCOVERY_LOCK.read_text(encoding="utf-8"))
    registry = {
        "schema_version": 1,
        "campaign_id": "progressive_oracle_refcoco_val_20260814",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "source_commit_at_initialisation": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(),
        "scope": {
            "dataset": "refcoco",
            "split": "val",
            "discovery_only": True,
            "new_full_validation_runs_authorized": 0,
            "quality_cell_cap": 33,
            "timing_cell_cap": 2,
        },
        "discovery_manifest": lock,
        "reference": reference,
        "decisions": {
            "B_tail": None,
            "B_early64": None,
            "progressive_base": None,
            "progressive_refinement": None,
        },
        "cells": registered_cells(),
        "timing_cells": {
            "E1": {"status": "pending", "scope": "timing", "attempts": []},
            "E2": {"status": "pending", "scope": "timing", "attempts": []},
        },
        "deferred_full_validation_recommendations": [],
        "totals": {"accepted_quality": 0, "accepted_timing": 0},
    }
    return registry


def initial_report(reference: dict[str, Any]) -> str:
    metrics = reference["discovery_recomputed"]
    return f"""# Progressive Visual-Token Pruning and Oracle Feasibility Campaign

## Scope and protocol

This is an exploratory RefCOCO-validation campaign. Every quality cell uses the
same frozen 1,000-example validation subset. No RefCOCO testA/testB, transfer
dataset, new 8,811-example confirmation, learned router, or dissertation edit is
part of this campaign.

## Frozen discovery subset

- Rows: 1,000
- Manifest SHA-256: `{reference['discovery_manifest_sha256']}`
- Sampling rule and provenance: `manifests/discovery_manifest_lock.json`

## Historical reference

`REF_K20_B48` was independently recomputed from the complete physical historical
prediction JSONL, rather than trusted from its aggregate metrics file.

| Reference | Rows | Acc@0.5 | mIoU | Parse rate |
|---|---:|---:|---:|---:|
| REF_K20_B48 (frozen subset) | 1,000 | {metrics['acc_at_0_5']:.4f} | {metrics['mean_iou']:.4f} | {metrics['parse_rate']:.4f} |

The historical run has no `COMPLETED.json` sentinel; it is accepted as read-only
reference evidence only because this campaign independently validated its exact
physical JSONL, configuration, source hashes, row order, budgets, caches, parser,
and metrics. Its redundant partial file is not used.

## Implementation and preflight evidence

Pending.

## Campaign 1 — K20 extreme tail

Pending.

## Campaign 2 — progressive pruning

Pending.

## Campaign 3 — oracle diagnostics

Pending. Oracle candidate-average token figures will exclude the dense teacher
pass and will report teacher visual-token layer-work separately.

## Conditional exploratory timing

Pending registered gates.

## Deferred full-validation recommendation

Not run. Candidate selection is pending discovery outcomes.

## Limitations

These are validation-only exploratory results on 1,000 examples. One percentage
point of Acc@0.5 is only ten examples. Marginal differences are inconclusive and
all dissertation claims require separately authorized full-validation evidence.

## Primary recommendation

Pending completion of all accepted discovery evidence.
"""


def initialise_campaign() -> dict[str, Any]:
    report, subset = validate_reference()
    reference_rows_bytes = canonical_jsonl_bytes(subset)
    report = dict(report)
    report["campaign_reference_rows_path"] = str(REFERENCE_ROWS.relative_to(REPO_ROOT))
    report["campaign_reference_rows_sha256"] = sha256_bytes(reference_rows_bytes)
    atomic_write(REFERENCE_ROWS, reference_rows_bytes)
    atomic_write(REFERENCE_REPORT, canonical_json_bytes(report))
    registry = initialise_registry(report)
    if REGISTRY_PATH.exists() or REPORT_PATH.exists():
        raise CampaignInitialisationError("Registry/report already exists; refusing to overwrite campaign state")
    atomic_write(REGISTRY_PATH, canonical_json_bytes(registry))
    atomic_write(REPORT_PATH, initial_report(report).encode("utf-8"))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report, subset = validate_reference()
        if not args.validate_only:
            if REGISTRY_PATH.exists() or REPORT_PATH.exists():
                raise CampaignInitialisationError("Registry/report already exists; refusing to overwrite campaign state")
            rows_bytes = canonical_jsonl_bytes(subset)
            report["campaign_reference_rows_path"] = str(REFERENCE_ROWS.relative_to(REPO_ROOT))
            report["campaign_reference_rows_sha256"] = sha256_bytes(rows_bytes)
            atomic_write(REFERENCE_ROWS, rows_bytes)
            atomic_write(REFERENCE_REPORT, canonical_json_bytes(report))
            atomic_write(REGISTRY_PATH, canonical_json_bytes(initialise_registry(report)))
            atomic_write(REPORT_PATH, initial_report(report).encode("utf-8"))
    except (CampaignInitialisationError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
