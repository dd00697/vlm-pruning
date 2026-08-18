#!/usr/bin/env python
"""Independently rescore and analyse the frozen recent-methods full-val campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CAMPAIGN_ROOT = REPO_ROOT / "experiments/recent_methods_full_refcoco_val_20260817"
ANALYSIS_ROOT = CAMPAIGN_ROOT / "analysis"
PROTOCOL_PATH = CAMPAIGN_ROOT / "FROZEN_PROTOCOL.json"
REGISTRY_PATH = CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"
REPORT_PATH = CAMPAIGN_ROOT / "RESEARCH_REPORT.md"
SOURCE_AMENDMENT_PATH = CAMPAIGN_ROOT / "SOURCE_EPOCH_AMENDMENT.json"
MANIFEST_PATH = REPO_ROOT / "data/manifests/final_evaluation/refcoco_val_unique.jsonl"
ROWS = 8_811
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20_260_817

DISPLAY = {
    "Full": "Full",
    "K20": "K20/B48",
    "K12": "K12/B48",
    "V1_K12_48_K20_6": "K12->48, K20->6",
    "V2_Z_K12_0": "K12->48, K20->0",
    "V3_C448": "C448",
    "V4_M448": "M448",
    "V5_Z_M448_0": "Z_M448_0",
}

COMPARATOR_ACCOUNTING = {
    "Full": {"trace": [576] * 32, "token_layer_work": 18_432, "average_visual_tokens": 576.0, "final_visual_tokens": 576},
    "K20": {"trace": [576] * 20 + [48] * 12, "token_layer_work": 12_096, "average_visual_tokens": 378.0, "final_visual_tokens": 48},
    "K12": {"trace": [576] * 12 + [48] * 20, "token_layer_work": 7_872, "average_visual_tokens": 246.0, "final_visual_tokens": 48},
}


class AnalysisError(RuntimeError):
    """Raised when evidence cannot support a fail-closed analysis."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_bytes(value))


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise AnalysisError(f"Refusing to write empty CSV: {path}")
    fields = list(values[0])
    if any(list(row) != fields for row in values):
        raise AnalysisError(f"Inconsistent CSV fields: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AnalysisError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path, expected: int = ROWS) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AnalysisError(f"Missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise AnalysisError(f"Blank JSONL line: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AnalysisError(f"Non-object JSONL row: {path}:{line_number}")
            rows.append(value)
    if len(rows) != expected:
        raise AnalysisError(f"Expected {expected} rows in {path}, found {len(rows)}")
    return rows


def close(actual: Any, expected: Any, label: str, tolerance: float = 1e-10) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise AnalysisError(f"{label}: expected {expected!r}, found {actual!r}")


def strict_parse(text: str, width: int, height: int) -> tuple[list[float] | None, list[float] | None, str | None]:
    """Independent copy of the frozen progressive runner's normalized-0-1 parser."""
    if not text or not text.strip():
        return None, None, "empty_response"
    candidates: list[str] = []
    for pattern in (r"<box>\s*([^<]+?)\s*</box>", r"\[([^\]]+)\]", r"\(([^\)]+)\)"):
        candidates.extend(re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL))
    if not candidates:
        candidates = [text]
    for candidate in candidates:
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", candidate)
        if len(numbers) < 4:
            continue
        raw = [float(numbers[index]) for index in range(4)]
        if raw[2] <= raw[0] or raw[3] <= raw[1]:
            return None, raw, "invalid_coordinate_order"
        pixels = [raw[0] * width, raw[1] * height, raw[2] * width, raw[3] * height]
        if pixels[2] <= 0 or pixels[3] <= 0 or pixels[0] >= width or pixels[1] >= height:
            return None, raw, "box_outside_image"
        clipped = [
            max(0.0, min(float(width), pixels[0])),
            max(0.0, min(float(height), pixels[1])),
            max(0.0, min(float(width), pixels[2])),
            max(0.0, min(float(height), pixels[3])),
        ]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return None, raw, "empty_box_after_clipping"
        return clipped, raw, None
    return None, None, "no_parseable_box"


def historical_parse(text: str, width: int, height: int) -> tuple[list[float] | None, list[float] | None, str | None]:
    """Independent copy of the evaluator used by the frozen Full/K12/K20 runs."""
    if not text or not text.strip():
        return None, None, "empty_response"
    candidates: list[str] = []
    for pattern in (r"<box>\s*([^<]+?)\s*</box>", r"\[([^\]]+)\]", r"\(([^\)]+)\)"):
        candidates.extend(re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL))
    if not candidates:
        candidates = [text]
    for candidate in candidates:
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", candidate)
        if len(numbers) < 4:
            continue
        raw = [float(numbers[index]) for index in range(4)]
        if raw[2] <= raw[0] or raw[3] <= raw[1]:
            return raw, None, "invalid_coordinate_order"
        pixels = [raw[0] * width, raw[1] * height, raw[2] * width, raw[3] * height]
        if pixels[2] <= 0 or pixels[3] <= 0 or pixels[0] >= width or pixels[1] >= height:
            return raw, None, "box_outside_image"
        clipped = [
            max(0.0, min(float(width), pixels[0])),
            max(0.0, min(float(height), pixels[1])),
            max(0.0, min(float(width), pixels[2])),
            max(0.0, min(float(height), pixels[3])),
        ]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return raw, None, "empty_box_after_clipping"
        return raw, clipped, None
    return None, None, "no_parseable_box"


def box_iou(left: list[float], right: list[float]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def semantic_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("query") or ""),
        str(row.get("image_path") or ""),
        tuple(float(value) for value in row.get("gt_box") or []),
    )


def align_rows(name: str, rows: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> list[str]:
    expected_ids = [str(row["sample_id"]) for row in manifest]
    if name == "Full":
        for index, (row, source) in enumerate(zip(rows, manifest, strict=True)):
            old_id = str((source.get("metadata") or {}).get("original_sample_id") or "")
            if str(row.get("sample_id") or "") != old_id:
                raise AnalysisError(f"Full original-ID alignment differs at row {index}")
            if semantic_identity(row) != semantic_identity(source):
                raise AnalysisError(f"Full semantic alignment differs at row {index}")
        return expected_ids
    actual = [str(row.get("sample_id") or "") for row in rows]
    if actual != expected_ids or len(set(actual)) != ROWS:
        raise AnalysisError(f"{name} does not exactly match final RefCOCO-val manifest order")
    for index, (row, source) in enumerate(zip(rows, manifest, strict=True)):
        if semantic_identity(row) != semantic_identity(source):
            raise AnalysisError(f"{name} semantic alignment differs at row {index}")
    return actual


def rescore(
    name: str,
    rows: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    *,
    progressive: bool,
) -> dict[str, Any]:
    align_rows(name, rows, manifest)
    accuracy = np.zeros(ROWS, dtype=np.float64)
    ious = np.zeros(ROWS, dtype=np.float64)
    parsed = np.zeros(ROWS, dtype=np.float64)
    valid = np.zeros(ROWS, dtype=np.float64)
    for index, (row, source) in enumerate(zip(rows, manifest, strict=True)):
        width, height = int(source["width"]), int(source["height"])
        if progressive:
            predicted, raw, reason = strict_parse(str(row.get("raw_model_output") or ""), width, height)
        else:
            raw, predicted, reason = historical_parse(str(row.get("raw_model_output") or ""), width, height)
        iou = box_iou(predicted, [float(value) for value in source["gt_box"]]) if predicted else 0.0
        success = iou >= 0.5
        parsed[index] = float(raw is not None)
        valid[index] = float(predicted is not None)
        ious[index] = iou
        accuracy[index] = float(success)
        close(row.get("box_iou", -1), iou, f"{name} row {index} IoU")
        if bool(row.get("success_iou_0_5")) != success:
            raise AnalysisError(f"{name} row {index} correctness differs after rescoring")
        if progressive:
            if row.get("parse_failure_reason") != reason:
                raise AnalysisError(f"{name} row {index} parse reason differs")
            if bool(row.get("valid")) != (predicted is not None):
                raise AnalysisError(f"{name} row {index} validity differs")
    return {
        "accuracy": accuracy,
        "iou": ious,
        "parsed": parsed,
        "valid": valid,
        "metrics": {
            "rows": ROWS,
            "unique_ids": ROWS,
            "parse_rate": float(parsed.mean()),
            "valid_output_rate": float(valid.mean()),
            "acc_at_0_5": float(accuracy.mean()),
            "mean_iou": float(ious.mean()),
            "median_iou": float(np.median(ious)),
        },
    }


def validate_comparator_accounting(name: str, config: dict[str, Any]) -> None:
    if name == "Full":
        if config.get("method/name") != "full" or int(config.get("tokens/actual_visual_tokens", -1)) != 576:
            raise AnalysisError("Full comparator configuration differs")
        return
    expected_boundary = 12 if name == "K12" else 20
    if int(config.get("ug/prune_after_layer", -1)) != expected_boundary:
        raise AnalysisError(f"{name} boundary differs")
    if int(config.get("tokens/actual_visual_tokens", -1)) != 48:
        raise AnalysisError(f"{name} budget differs")
    if config.get("position_mode") != "gap_preserving":
        raise AnalysisError(f"{name} positions differ")


def validate_progressive_accounting(name: str, rows: list[dict[str, Any]], cell: dict[str, Any]) -> None:
    expected_trace = [int(value) for value in cell["trace"]]
    expected_work = int(cell["token_layer_work"])
    expected_average = float(cell["average_visual_tokens"])
    expected_final = int(cell["final_visual_tokens"])
    if len(expected_trace) != 32 or sum(expected_trace) != expected_work:
        raise AnalysisError(f"{name} frozen trace accounting differs")
    close(expected_work / 32, expected_average, f"{name} average token accounting")
    for index, row in enumerate(rows):
        if [int(value) for value in row.get("realized_visual_tokens_by_layer") or []] != expected_trace:
            raise AnalysisError(f"{name} row {index} trace differs")
        close(row.get("candidate_average_visual_tokens", -1), expected_average, f"{name} row {index} average")
        if len(row.get("final_retained_indices") or []) != expected_final:
            raise AnalysisError(f"{name} row {index} final budget differs")


def provenance(run_dir: Path, prediction_name: str) -> dict[str, Any]:
    required = ["config.json", prediction_name, "metrics.json"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise AnalysisError(f"Missing provenance files in {run_dir}: {missing}")
    result = {
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "config_path": str((run_dir / "config.json").relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(run_dir / "config.json"),
        "predictions_path": str((run_dir / prediction_name).relative_to(REPO_ROOT)),
        "predictions_sha256": sha256_file(run_dir / prediction_name),
        "metrics_path": str((run_dir / "metrics.json").relative_to(REPO_ROOT)),
        "metrics_sha256": sha256_file(run_dir / "metrics.json"),
    }
    for filename, key in (("integrity_report.json", "integrity_sha256"), ("COMPLETED.json", "completion_sha256")):
        path = run_dir / filename
        if path.is_file():
            result[key] = sha256_file(path)
    return result


def paired_intervals(
    arrays: dict[str, dict[str, Any]], contrasts: list[list[str]]
) -> list[dict[str, Any]]:
    delta = {
        (left, right, "acc_at_0_5_pp"): (arrays[left]["accuracy"] - arrays[right]["accuracy"]) * 100.0
        for left, right in contrasts
    }
    delta.update(
        {
            (left, right, "mean_iou"): arrays[left]["iou"] - arrays[right]["iou"]
            for left, right in contrasts
        }
    )
    estimates = {key: np.empty(BOOTSTRAPS, dtype=np.float64) for key in delta}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for start in range(0, BOOTSTRAPS, 100):
        count = min(100, BOOTSTRAPS - start)
        indices = rng.integers(0, ROWS, size=(count, ROWS))
        for key, values in delta.items():
            estimates[key][start : start + count] = values[indices].mean(axis=1)
    output: list[dict[str, Any]] = []
    for contrast_index, (left, right) in enumerate(contrasts):
        left_acc, right_acc = arrays[left]["accuracy"], arrays[right]["accuracy"]
        discordant = {
            "left_correct_right_wrong": int(np.sum((left_acc == 1) & (right_acc == 0))),
            "left_wrong_right_correct": int(np.sum((left_acc == 0) & (right_acc == 1))),
        }
        for metric in ("acc_at_0_5_pp", "mean_iou"):
            key = (left, right, metric)
            lower, upper = np.quantile(estimates[key], [0.025, 0.975])
            output.append(
                {
                    "contrast_order": contrast_index,
                    "contrast": f"{left}-minus-{right}",
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "point_delta": float(delta[key].mean()),
                    "lower_95": float(lower),
                    "upper_95": float(upper),
                    "bootstrap_resamples": BOOTSTRAPS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "paired_rows": ROWS,
                    "left_correct_right_wrong": discordant["left_correct_right_wrong"],
                    "left_wrong_right_correct": discordant["left_wrong_right_correct"],
                }
            )
    return output


def error_movement(arrays: dict[str, dict[str, Any]], contrasts: list[list[str]]) -> list[dict[str, Any]]:
    output = []
    for left, right in contrasts:
        left_iou, right_iou = arrays[left]["iou"], arrays[right]["iou"]
        left_acc, right_acc = arrays[left]["accuracy"], arrays[right]["accuracy"]
        movement = left_iou - right_iou
        output.append(
            {
                "contrast": f"{left}-minus-{right}",
                "left": left,
                "right": right,
                "left_correct_right_wrong": int(np.sum((left_acc == 1) & (right_acc == 0))),
                "left_wrong_right_correct": int(np.sum((left_acc == 0) & (right_acc == 1))),
                "iou_improved": int(np.sum(movement > 1e-12)),
                "iou_unchanged": int(np.sum(np.abs(movement) <= 1e-12)),
                "iou_worsened": int(np.sum(movement < -1e-12)),
                "iou_gain_ge_0_25": int(np.sum(movement >= 0.25)),
                "iou_loss_ge_0_25": int(np.sum(movement <= -0.25)),
                "mean_iou_movement": float(movement.mean()),
            }
        )
    return output


def pareto_membership(methods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for metric in ("acc_at_0_5", "mean_iou"):
        for current in methods:
            dominated = any(
                other["token_layer_work"] <= current["token_layer_work"]
                and other[metric] >= current[metric]
                and (
                    other["token_layer_work"] < current["token_layer_work"]
                    or other[metric] > current[metric]
                )
                for other in methods
                if other["method"] != current["method"]
            )
            current[f"pareto_{metric}"] = not dominated
    for current in methods:
        dominated = any(
            other["token_layer_work"] <= current["token_layer_work"]
            and other["acc_at_0_5"] >= current["acc_at_0_5"]
            and other["mean_iou"] >= current["mean_iou"]
            and (
                other["token_layer_work"] < current["token_layer_work"]
                or other["acc_at_0_5"] > current["acc_at_0_5"]
                or other["mean_iou"] > current["mean_iou"]
            )
            for other in methods
            if other["method"] != current["method"]
        )
        current["pareto_joint_quality_work"] = not dominated
    return methods


def interval_direction(row: dict[str, Any]) -> str:
    """Classify a paired interval without treating inclusion of zero as equivalence."""
    if float(row["lower_95"]) > 0.0:
        return "reliable_positive"
    if float(row["upper_95"]) < 0.0:
        return "reliable_negative"
    return "inconclusive_includes_zero"


def campaign_classification(
    methods: list[dict[str, Any]], intervals: list[dict[str, Any]]
) -> dict[str, Any]:
    by_interval = {(row["contrast"], row["metric"]): row for row in intervals}

    def hypothesis(left: str, right: str, question: str) -> dict[str, Any]:
        contrast = f"{left}-minus-{right}"
        metrics = {}
        for metric in ("acc_at_0_5_pp", "mean_iou"):
            row = by_interval[(contrast, metric)]
            metrics[metric] = {
                "point_delta": float(row["point_delta"]),
                "lower_95": float(row["lower_95"]),
                "upper_95": float(row["upper_95"]),
                "direction": interval_direction(row),
            }
        return {
            "question": question,
            "contrast": contrast,
            "metrics": metrics,
            "claim_rule": "metric-specific paired interval direction; an interval containing zero is inconclusive",
        }

    return {
        "status": "FULL_VALIDATION_COMPLETE_POST_HOC_DEVELOPMENT_EVIDENCE",
        "selection_status": "NO_NEW_FINAL_METHOD_SELECTED_HUMAN_REVIEW_REQUIRED",
        "hypotheses": {
            "six_tail_tokens_help_over_zero": hypothesis(
                "V1_K12_48_K20_6",
                "V2_Z_K12_0",
                "Do six visual tail tokens improve quality over removing the tail?",
            ),
            "merge_helps_over_matched_hard_deletion": hypothesis(
                "V4_M448",
                "V3_C448",
                "Does information-preserving merging improve quality at matched token work?",
            ),
            "zero_tail_effect_after_m448": hypothesis(
                "V5_Z_M448_0",
                "V4_M448",
                "What quality change follows removal of M448's remaining visual tail?",
            ),
            "z_m448_0_effect_relative_to_k12": hypothesis(
                "V5_Z_M448_0",
                "K12",
                "What quality change accompanies reducing K12's 246 average tokens to 196?",
            ),
        },
        "joint_quality_work_frontier": [
            row["method"] for row in methods if bool(row["pareto_joint_quality_work"])
        ],
        "latency_claim": "NO_NEW_LATENCY_EVIDENCE; token-layer work and average tokens are analytical proxies",
        "held_out_claim": "UNCHANGED; this campaign does not replace the frozen testA/testB result",
    }


def immutable_context() -> dict[str, Any]:
    paths = {
        "router": REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815/analysis/GATE_RESULTS_AND_SELECTION.json",
        "oracle": REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814/RESEARCH_REPORT.md",
        "e256": REPO_ROOT / "experiments/hardware_aware_zero_tail_refcoco_val_20260816/early_compression/analysis/QUALITY_RESULTS.json",
        "saqr": REPO_ROOT / "experiments/trained_prellm_resampler_refcoco_20260817/RESEARCH_REPORT.md",
        "timing": REPO_ROOT / "experiments/hardware_aware_zero_tail_refcoco_val_20260816/matched_timing/analysis/NATURAL_RESULTS.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise AnalysisError("An immutable contextual evidence artifact is missing")
    gate = read_json(paths["router"])
    e256 = read_json(paths["e256"])
    timing = read_json(paths["timing"])
    gate_rates = {
        seed: float(value["metrics"]["aggressive_rate"])
        for seed, value in gate["physical_seeds"].items()
    }
    natural = timing["summaries"]
    return {
        "sources": {
            name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "router": {
            "m448_route_rates": gate_rates,
            "range": [min(gate_rates.values()), max(gate_rates.values())],
            "interpretation": "three learned gates routed almost every validation row to M448",
        },
        "oracle": {
            "strongest_at_or_below_64_average_tokens": "O8",
            "acc_at_0_5": 0.351,
            "mean_iou": 0.3881,
            "privileged_non_deployable": True,
        },
        "e256_pool": e256["metrics"]["E256_POOL"],
        "saqr": {
            "discovery": {"acc_at_0_5": 0.488, "mean_iou": 0.466047},
            "no_query": {"acc_at_0_5": 0.498, "mean_iou": 0.475474},
            "end_to_end_slower_fraction_vs_k12": 0.393,
            "classification": "C_QUALITY_PRESERVED_NO_REAL_ACCELERATION",
        },
        "existing_matched_timing": {
            method: {
                "end_to_end_latency_s": float(natural[method]["end_to_end_latency_s"]["mean"]),
                "prompt_kv_total_bytes": float(natural[method]["prompt_kv_total_bytes"]["mean"]),
                "final_kv_total_bytes": float(natural[method]["final_kv_total_bytes"]["mean"]),
            }
            for method in ("K12_ORIGINAL", "M448_OPT", "Z_M448_0")
        },
    }


def build_report(
    methods: list[dict[str, Any]], intervals: list[dict[str, Any]], context: dict[str, Any], inventory: dict[str, Any]
) -> str:
    by_method = {row["method"]: row for row in methods}
    by_contrast = {(row["contrast"], row["metric"]): row for row in intervals}

    def effect(left: str, right: str, metric: str) -> dict[str, Any]:
        return by_contrast[(f"{left}-minus-{right}", metric)]

    v1_v2_acc = effect("V1_K12_48_K20_6", "V2_Z_K12_0", "acc_at_0_5_pp")
    v1_v2_iou = effect("V1_K12_48_K20_6", "V2_Z_K12_0", "mean_iou")
    merge_acc = effect("V4_M448", "V3_C448", "acc_at_0_5_pp")
    merge_iou = effect("V4_M448", "V3_C448", "mean_iou")
    zero_merge_acc = effect("V5_Z_M448_0", "V4_M448", "acc_at_0_5_pp")
    zero_merge_iou = effect("V5_Z_M448_0", "V4_M448", "mean_iou")
    zero_k12_acc = effect("V5_Z_M448_0", "K12", "acc_at_0_5_pp")
    zero_k12_iou = effect("V5_Z_M448_0", "K12", "mean_iou")
    timing = context["existing_matched_timing"]

    def interval_reading(acc: dict[str, Any], iou: dict[str, Any]) -> str:
        readings = {
            "Acc@0.5": interval_direction(acc),
            "mIoU": interval_direction(iou),
        }
        labels = {
            "reliable_positive": "positive with its interval excluding zero",
            "reliable_negative": "negative with its interval excluding zero",
            "inconclusive_includes_zero": "inconclusive because its interval includes zero",
        }
        return "; ".join(f"{metric} is {labels[direction]}" for metric, direction in readings.items()) + "."

    lines = [
        "# Recent Methods Full RefCOCO-Validation Campaign",
        "",
        "Final status: analysis complete; dissertation integration and terminal audit pending.",
        "",
        "## Scope and protocol",
        "",
        "Exactly five preregistered deterministic methods were evaluated on all 8,811 rows of the frozen RefCOCO validation manifest. This is post-hoc development evidence, not a replacement for the dissertation's frozen held-out testA/testB result. No training, timing, transfer, external baseline, learned router, test split, or unregistered token budget was run.",
        "",
        "The K12 tail cells test whether the 48 visual states retained after layer 12 remain necessary after layer 20. The C448/M448 pair isolates whether merging discarded layer-4 content into 448 survivors preserves information better than hard top-k selection at identical token work. Z_M448_0 then removes the visual tail after layer 20 and reaches 196 average visual tokens.",
        "",
        "## Full-validation quality and analytical efficiency",
        "",
        "| Method | Schedule | Rows | Parse | Valid | Acc@0.5 | mIoU | Median IoU | Final tokens | Avg tokens | Token-layer work |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in methods:
        lines.append(
            f"| {row['display']} | {row['schedule']} | {row['rows']:,} | {row['parse_rate']:.4f} | "
            f"{row['valid_output_rate']:.4f} | {row['acc_at_0_5']:.4f} | {row['mean_iou']:.4f} | "
            f"{row['median_iou']:.4f} | {row['final_visual_tokens']} | {row['average_visual_tokens']:.2f} | "
            f"{row['token_layer_work']:,} |"
        )
    lines.extend(
        [
            "",
            "Average visual tokens are token-layer work divided by 32, not a latency measurement. All quality aggregates above were independently recomputed from raw generated strings and the frozen manifest.",
            "",
            "## Registered paired results",
            "",
            f"Six tail tokens minus a zero tail changes Acc@0.5 by {v1_v2_acc['point_delta']:+.2f} percentage points (95% paired bootstrap CI [{v1_v2_acc['lower_95']:+.2f}, {v1_v2_acc['upper_95']:+.2f}]) and mIoU by {v1_v2_iou['point_delta']:+.4f} (CI [{v1_v2_iou['lower_95']:+.4f}, {v1_v2_iou['upper_95']:+.4f}]).",
            interval_reading(v1_v2_acc, v1_v2_iou),
            "",
            f"At matched 6,848 token-layer work, M448 minus C448 changes Acc@0.5 by {merge_acc['point_delta']:+.2f} points (CI [{merge_acc['lower_95']:+.2f}, {merge_acc['upper_95']:+.2f}]) and mIoU by {merge_iou['point_delta']:+.4f} (CI [{merge_iou['lower_95']:+.4f}, {merge_iou['upper_95']:+.4f}]). This is the direct test of information-preserving merging.",
            interval_reading(merge_acc, merge_iou),
            "",
            f"Removing the layer-20 tail from M448 changes Acc@0.5 by {zero_merge_acc['point_delta']:+.2f} points (CI [{zero_merge_acc['lower_95']:+.2f}, {zero_merge_acc['upper_95']:+.2f}]) and mIoU by {zero_merge_iou['point_delta']:+.4f} (CI [{zero_merge_iou['lower_95']:+.4f}, {zero_merge_iou['upper_95']:+.4f}]). Z_M448_0 uses {by_method['V5_Z_M448_0']['average_visual_tokens']:.0f} average tokens and {by_method['V5_Z_M448_0']['token_layer_work']:,} token-layer work.",
            interval_reading(zero_merge_acc, zero_merge_iou),
            "",
            f"Relative to aligned K12, Z_M448_0 changes Acc@0.5 by {zero_k12_acc['point_delta']:+.2f} points (CI [{zero_k12_acc['lower_95']:+.2f}, {zero_k12_acc['upper_95']:+.2f}]) and mIoU by {zero_k12_iou['point_delta']:+.4f} (CI [{zero_k12_iou['lower_95']:+.4f}, {zero_k12_iou['upper_95']:+.4f}]) while reducing token-layer work by {100 * (1 - by_method['V5_Z_M448_0']['token_layer_work'] / by_method['K12']['token_layer_work']):.1f}%.",
            interval_reading(zero_k12_acc, zero_k12_iou),
            "",
            "The complete 17-contrast table, discordant correctness counts, and compact IoU movement are machine-readable under `analysis/`. Intervals quantify sampling uncertainty within this fixed validation manifest; they do not make the validation set held out or prove equivalence.",
            "",
            "## Efficiency interpretation",
            "",
            f"The existing matched natural-output timing study measured {timing['K12_ORIGINAL']['end_to_end_latency_s']:.6f} s for K12, {timing['M448_OPT']['end_to_end_latency_s']:.6f} s for M448_OPT, and {timing['Z_M448_0']['end_to_end_latency_s']:.6f} s for Z_M448_0. Thus lower token-layer work and KV storage did not produce lower end-to-end latency in that implementation. These are immutable historical measurements, not timing collected in this campaign; incidental latency fields in the quality runs are excluded from all speed claims.",
            "",
            "## Prior negative and preliminary evidence",
            "",
            f"The learned schedule gate selected M448 on {100*context['router']['range'][0]:.1f}%--{100*context['router']['range'][1]:.1f}% of validation rows across three seeds, so it learned an almost constant policy rather than useful per-example routing. Privileged early-selection oracle O8 reached only 0.3510 Acc@0.5/0.3881 mIoU at at most 64 average candidate tokens, showing that perfect early identity ranking was insufficient. E256_POOL reached {context['e256_pool']['acc_at_0_5']:.3f}/{context['e256_pool']['mean_iou']:.4f}, below K12 quality. The single-seed SAQR proof of concept reached 0.488/0.4660, but its no-query diagnostic was stronger at 0.498/0.4755 and matched end-to-end inference was 39.3% slower than K12. These exploratory findings were not rerun here and do not alter the training-free dissertation contribution.",
            "",
            "## Provenance, limitations, and next decision",
            "",
            f"All {inventory['accepted_count']} registered full-validation cells independently validated with exactly 8,811 ordered rows. Comparator identity was aligned to the final manifest, including the Full run's historical original IDs. Source/config/prediction hashes are recorded in `analysis/RUN_INVENTORY.json`.",
            "",
            "Among the new schedules, Z_M448_0 is the clearest candidate for human consideration in a later preregistered timing and held-out goal: it gives the strongest work reduction and a competitive full-validation point estimate. This is a development-set prioritisation, not a new frozen selection, latency claim, or held-out conclusion.",
            "",
            "The results use one checkpoint, one prompt, one validation split, deterministic seed 0, and no replicate model runs. Token-layer work is an analytical compute proxy; prompt/final KV bytes are storage proxies; neither is interchangeable with measured latency. A later timing or held-out goal is justified only for methods that combine a defensible full-validation quality result with a useful work reduction, and it must use a fresh preregistration.",
            "",
            "## Artifact index",
            "",
            "- `analysis/METHOD_METRICS.json` / `.csv`: independently rescored quality and token accounting.",
            "- `analysis/PAIRED_CONTRASTS.json` / `.csv`: registered 10,000-resample intervals and discordant counts.",
            "- `analysis/ERROR_MOVEMENT.json` / `.csv`: compact paired correctness/IoU movement.",
            "- `analysis/PARETO.json` / `.csv`: quality--work frontier membership.",
            "- `analysis/CAMPAIGN_CLASSIFICATION.json`: interval-direction hypothesis outcomes, frontier membership and the human-review selection lock.",
            "- `analysis/RUN_INVENTORY.json`: accepted evidence, hashes, and independent validation.",
            "- `analysis/IMMUTABLE_CONTEXT.json`: hashes and compact summaries of reused timing/router/oracle/E256/SAQR evidence.",
            "- `analysis/CLAIM_EVIDENCE_LEDGER.json`: claim-to-artifact provenance.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    protocol = read_json(PROTOCOL_PATH)
    registry = read_json(REGISTRY_PATH)
    amendment = read_json(SOURCE_AMENDMENT_PATH)
    if amendment.get("initial_protocol_commit") != protocol.get("source", {}).get("git_commit"):
        raise AnalysisError("Source amendment does not start from the frozen protocol epoch")
    for relative, expected in amendment.get("scientific_source_unchanged", {}).items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise AnalysisError(f"Scientific source differs from bounded amendment: {relative}")
    for relative, record in amendment.get("changed_files", {}).items():
        if sha256_file(REPO_ROOT / relative) != record.get("execution_sha256"):
            raise AnalysisError(f"Execution source differs from bounded amendment: {relative}")
    if registry.get("phase_status", {}).get("full_validation") != "pass":
        raise AnalysisError("Full validation is not terminally accepted")
    if protocol.get("manifest", {}).get("sha256") != sha256_file(MANIFEST_PATH):
        raise AnalysisError("Frozen manifest hash differs")
    manifest = read_jsonl(MANIFEST_PATH)
    manifest_ids = [str(row["sample_id"]) for row in manifest]
    if len(set(manifest_ids)) != ROWS:
        raise AnalysisError("Frozen manifest IDs are not unique")

    method_order = list(protocol["analysis"]["main_table_order"])
    expected_order = ["Full", "K20", "K12", *list(protocol["cells"])]
    if method_order != expected_order:
        raise AnalysisError("Frozen main table order differs")

    arrays: dict[str, dict[str, Any]] = {}
    inventory: dict[str, Any] = {
        "status": "complete",
        "created_at_utc": utc_now(),
        "manifest": {
            "path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(MANIFEST_PATH),
            "rows": ROWS,
            "unique_ids": ROWS,
            "order_sha256": hashlib.sha256("\n".join(manifest_ids).encode()).hexdigest(),
        },
        "methods": {},
        "accepted_count": 0,
        "rejected_full_attempts": [],
        "source_epoch": {
            "initial_protocol_commit": amendment["initial_protocol_commit"],
            "execution_commit": amendment["execution_commit"],
            "amendment_path": str(SOURCE_AMENDMENT_PATH.relative_to(REPO_ROOT)),
            "amendment_sha256": sha256_file(SOURCE_AMENDMENT_PATH),
            "scientific_source_unchanged": True,
        },
    }

    from scripts.validate_progressive_oracle_run import validate_run

    for name in method_order:
        if name in protocol["cells"]:
            cell = protocol["cells"][name]
            run_dir = CAMPAIGN_ROOT / cell["full_output"]
            validation = validate_run(run_dir, MANIFEST_PATH, expected_rows=ROWS, finalize=False)
            if validation.get("status") != "pass":
                raise AnalysisError(f"{name} independent validator failed")
            config = read_json(run_dir / "config.json")
            expected_commit = amendment.get("cell_execution_commits", {}).get(
                name, amendment["execution_commit"]
            )
            if config.get("source/git_commit") != expected_commit:
                raise AnalysisError(f"{name} did not execute at the amended source epoch")
            rows = read_jsonl(run_dir / "predictions.jsonl")
            validate_progressive_accounting(name, rows, cell)
            arrays[name] = rescore(name, rows, manifest, progressive=True)
            accounting = {
                "schedule": cell["schedule"],
                "trace": cell["trace"],
                "token_layer_work": int(cell["token_layer_work"]),
                "average_visual_tokens": float(cell["average_visual_tokens"]),
                "final_visual_tokens": int(cell["final_visual_tokens"]),
            }
            prediction_name = "predictions.jsonl"
            authority = "verified_current"
        else:
            relative = protocol["comparators"][name]
            run_dir = REPO_ROOT / relative
            config = read_json(run_dir / "config.json")
            validate_comparator_accounting(name, config)
            rows = read_jsonl(run_dir / "per_sample_results.jsonl")
            arrays[name] = rescore(name, rows, manifest, progressive=False)
            accounting = {"schedule": name, **COMPARATOR_ACCOUNTING[name]}
            prediction_name = "per_sample_results.jsonl"
            authority = "verified_historical_comparator"
        inventory["methods"][name] = {
            "authority": authority,
            "validation_status": "pass",
            "metrics": arrays[name]["metrics"],
            "accounting": accounting,
            "provenance": provenance(run_dir, prediction_name),
        }
        inventory["accepted_count"] += 1 if name in protocol["cells"] else 0

    run_dirs = sorted(path for path in (CAMPAIGN_ROOT / "runs").iterdir() if path.is_dir())
    registered_dirs = sorted((CAMPAIGN_ROOT / protocol["cells"][name]["full_output"]).resolve() for name in protocol["cells"])
    if [path.resolve() for path in run_dirs] != registered_dirs:
        raise AnalysisError("Full-run inventory contains an unregistered or missing directory")

    methods: list[dict[str, Any]] = []
    for name in method_order:
        metrics = arrays[name]["metrics"]
        accounting = inventory["methods"][name]["accounting"]
        methods.append(
            {
                "method": name,
                "display": DISPLAY[name],
                "schedule": accounting["schedule"],
                **metrics,
                "final_visual_tokens": accounting["final_visual_tokens"],
                "average_visual_tokens": accounting["average_visual_tokens"],
                "token_layer_work": accounting["token_layer_work"],
                "work_reduction_vs_full": 1.0 - accounting["token_layer_work"] / 18_432,
            }
        )
    methods = pareto_membership(methods)
    contrasts = [list(value) for value in protocol["analysis"]["contrasts"]]
    intervals = paired_intervals(arrays, contrasts)
    movement = error_movement(arrays, contrasts)
    context = immutable_context()
    classification = campaign_classification(methods, intervals)

    atomic_json(ANALYSIS_ROOT / "RUN_INVENTORY.json", inventory)
    atomic_json(ANALYSIS_ROOT / "METHOD_METRICS.json", methods)
    atomic_csv(ANALYSIS_ROOT / "METHOD_METRICS.csv", methods)
    atomic_json(ANALYSIS_ROOT / "PAIRED_CONTRASTS.json", intervals)
    atomic_csv(ANALYSIS_ROOT / "PAIRED_CONTRASTS.csv", intervals)
    atomic_json(ANALYSIS_ROOT / "ERROR_MOVEMENT.json", movement)
    atomic_csv(ANALYSIS_ROOT / "ERROR_MOVEMENT.csv", movement)
    atomic_json(ANALYSIS_ROOT / "PARETO.json", methods)
    atomic_csv(ANALYSIS_ROOT / "PARETO.csv", methods)
    atomic_json(ANALYSIS_ROOT / "IMMUTABLE_CONTEXT.json", context)
    atomic_json(ANALYSIS_ROOT / "CAMPAIGN_CLASSIFICATION.json", classification)

    report = build_report(methods, intervals, context, inventory)
    atomic_write(REPORT_PATH, report.encode("utf-8"))

    def evidence(*relative_paths: str) -> list[dict[str, str]]:
        result = []
        for relative in relative_paths:
            path = CAMPAIGN_ROOT / relative
            if not path.is_file():
                raise AnalysisError(f"Claim evidence is missing: {relative}")
            result.append({"path": relative, "sha256": sha256_file(path)})
        return result

    ledger = {
        "status": "analysis_complete",
        "created_at_utc": utc_now(),
        "source_epoch": inventory["source_epoch"],
        "claims": {
            "full_validation_quality": evidence("analysis/METHOD_METRICS.json", "analysis/RUN_INVENTORY.json"),
            "paired_uncertainty": evidence("analysis/PAIRED_CONTRASTS.json"),
            "six_vs_zero_tail": evidence("analysis/PAIRED_CONTRASTS.json", "analysis/ERROR_MOVEMENT.json"),
            "merge_vs_hard_selection": evidence("analysis/PAIRED_CONTRASTS.json"),
            "zero_tail_after_merge": evidence("analysis/PAIRED_CONTRASTS.json"),
            "quality_work_pareto": evidence("analysis/PARETO.json"),
            "campaign_classification": evidence("analysis/CAMPAIGN_CLASSIFICATION.json"),
            "latency_constraint_and_prior_negative_evidence": evidence("analysis/IMMUTABLE_CONTEXT.json"),
            "cumulative_interpretation": evidence("RESEARCH_REPORT.md"),
        },
    }
    atomic_json(ANALYSIS_ROOT / "CLAIM_EVIDENCE_LEDGER.json", ledger)

    outputs = sorted(
        path
        for path in ANALYSIS_ROOT.iterdir()
        if path.is_file() and path.name != "ANALYSIS_MANIFEST.json"
    )
    manifest_output = {
        "status": "complete",
        "created_at_utc": utc_now(),
        "bootstrap_resamples": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "files": {str(path.relative_to(CAMPAIGN_ROOT)): sha256_file(path) for path in outputs},
        "report": {"path": str(REPORT_PATH.relative_to(CAMPAIGN_ROOT)), "sha256": sha256_file(REPORT_PATH)},
    }
    atomic_json(ANALYSIS_ROOT / "ANALYSIS_MANIFEST.json", manifest_output)
    registry = read_json(REGISTRY_PATH)
    registry["phase_status"]["analysis"] = "pass"
    registry["status"] = "analysis_complete"
    registry["analysis"] = {
        "manifest_path": str((ANALYSIS_ROOT / "ANALYSIS_MANIFEST.json").relative_to(CAMPAIGN_ROOT)),
        "manifest_sha256": sha256_file(ANALYSIS_ROOT / "ANALYSIS_MANIFEST.json"),
        "report_sha256": sha256_file(REPORT_PATH),
    }
    atomic_json(REGISTRY_PATH, registry)
    print(json.dumps({"status": "complete", "methods": {row["method"]: row["acc_at_0_5"] for row in methods}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
