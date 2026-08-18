#!/usr/bin/env python
"""Independently validate and rescore final-campaign RefCOCO-val quality artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyse_recent_methods_full_val import (  # noqa: E402
    align_rows,
    historical_parse,
    strict_parse,
)
from vlm_pruning.artifacts import sha256_file  # noqa: E402
from vlm_pruning.final_statistics import paired_bootstrap_interval  # noqa: E402


CAMPAIGN = REPO_ROOT / "experiments/final_val_efficiency_baselines_20260818"
ANALYSIS = CAMPAIGN / "analysis"
MANIFEST = REPO_ROOT / "data/manifests/final_evaluation/refcoco_val_unique.jsonl"
MANIFEST_SHA256 = "f26d7b954189e86f55b50f9506bb670d3a0cf847d3152779af6c08dbf094e3d7"
EXPECTED_ROWS = 8_811
BOOTSTRAPS = 10_000
SEED = 20_260_818

METHODS = {
    "Full": {
        "path": REPO_ROOT / "experiments/runs_official_baselines/official_fullmatrix_refcoco_full_tokens576_seed0_bf16/per_sample_results.jsonl",
        "parser": "historical",
        "trace": [576] * 32,
        "stack": "canonical",
        "historical_key": "Full",
    },
    "K20": {
        "path": REPO_ROOT / "experiments/final_evaluation/runs/ug/final_selection_refcoco_val_k20_b48_query_max_more_anchors_bf16/per_sample_results.jsonl",
        "parser": "strict",
        "trace": [576] * 20 + [48] * 12,
        "stack": "canonical",
        "historical_key": "K20",
    },
    "K12": {
        "path": REPO_ROOT / "experiments/final_selector_ablation_and_repair_20260811/runs/repair_val_k12_b48_single_layer_more_context/per_sample_results.jsonl",
        "parser": "strict",
        "trace": [576] * 12 + [48] * 20,
        "stack": "canonical",
        "historical_key": "K12",
    },
    "M448_OPT": {
        "path": REPO_ROOT / "experiments/recent_methods_full_refcoco_val_20260817/runs/v4_m448_attempt1/predictions.jsonl",
        "parser": "strict",
        "trace": [576] * 4 + [448] * 8 + [48] * 20,
        "stack": "canonical",
        "historical_key": "M448",
    },
    "Z_M448_0": {
        "path": REPO_ROOT / "experiments/recent_methods_full_refcoco_val_20260817/runs/v5_z_m448_0_attempt1/predictions.jsonl",
        "parser": "strict",
        "trace": [576] * 4 + [448] * 8 + [48] * 8 + [0] * 12,
        "stack": "canonical",
        "historical_key": "Z_M448_0",
    },
    "FASTV_MATCHED_199": {
        "path": CAMPAIGN / "quality/fastv/e1_fastv_matched_199_attempt1/per_sample_results.jsonl",
        "parser": "strict",
        "trace": [576] * 3 + [160] * 29,
        "stack": "fastv_native_fork",
        "historical_key": None,
    },
    "FEATHER_LLAVA_MAIN": {
        "path": CAMPAIGN / "quality/feather/e2_feather_llava_main_attempt1/predictions.jsonl",
        "parser": "strict",
        "trace": None,
        "stack": "canonical",
        "historical_key": None,
    },
    "NUWA_STACK_FULL": {
        "path": CAMPAIGN / "quality/nuwa/nuwa_stack_full_attempt1/predictions.jsonl",
        "parser": "strict",
        "trace": [576] * 32,
        "stack": "nuwa_native",
        "historical_key": None,
    },
    "NUWA_NATIVE_192": {
        "path": CAMPAIGN / "quality/nuwa/nuwa_native_192_attempt1/predictions.jsonl",
        "parser": "strict",
        "trace": [336] * 16 + [48] * 16,
        "stack": "nuwa_native",
        "historical_key": None,
    },
}


class QualityAnalysisError(RuntimeError):
    """Raised when a quality artifact violates the frozen campaign contract."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise QualityAnalysisError(f"Missing quality artifact: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise QualityAnalysisError(f"Blank line at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise QualityAnalysisError(f"Non-object row at {path}:{line_number}")
            rows.append(value)
    if len(rows) != EXPECTED_ROWS:
        raise QualityAnalysisError(f"Expected {EXPECTED_ROWS} rows in {path}, found {len(rows)}")
    return rows


def atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise QualityAnalysisError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def box_iou(left: list[float], right: list[float]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def validate_trace(name: str, rows: list[dict[str, Any]], expected: list[int] | None) -> dict[str, Any]:
    if len(rows) != EXPECTED_ROWS:
        raise QualityAnalysisError(f"{name} trace validation requires {EXPECTED_ROWS} rows")
    if expected is not None:
        for index, row in enumerate(rows):
            recorded = row.get("realized_visual_tokens_by_layer") or row.get("per_layer_visual_tokens")
            if recorded is not None and [int(value) for value in recorded] != expected:
                raise QualityAnalysisError(f"{name} trace differs at row {index}")
        works = np.full(EXPECTED_ROWS, sum(expected), dtype=np.float64)
        finals = np.full(EXPECTED_ROWS, expected[-1], dtype=np.float64)
    else:
        works = np.empty(EXPECTED_ROWS, dtype=np.float64)
        finals = np.empty(EXPECTED_ROWS, dtype=np.float64)
        for index, row in enumerate(rows):
            trace = [int(value) for value in row.get("realized_visual_tokens_by_layer") or []]
            if (
                len(trace) != 32
                or trace[:8] != [576] * 8
                or len(set(trace[8:16])) != 1
                or len(set(trace[16:])) != 1
                or not 576 >= trace[8] >= trace[16] >= 0
            ):
                raise QualityAnalysisError(f"FEATHER adaptive trace differs at row {index}: {trace}")
            works[index] = sum(trace)
            finals[index] = trace[-1]
    return {
        "mean_token_layer_work": float(works.mean()),
        "mean_average_visual_tokens": float(works.mean() / 32),
        "mean_final_visual_tokens": float(finals.mean()),
        "min_final_visual_tokens": int(finals.min()),
        "max_final_visual_tokens": int(finals.max()),
    }


def rescore(
    name: str,
    rows: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    parser: str,
) -> dict[str, Any]:
    align_rows("Full" if name == "Full" else name, rows, manifest)
    correctness = np.zeros(EXPECTED_ROWS, dtype=np.float64)
    ious = np.zeros(EXPECTED_ROWS, dtype=np.float64)
    parsed = np.zeros(EXPECTED_ROWS, dtype=np.float64)
    valid = np.zeros(EXPECTED_ROWS, dtype=np.float64)
    for index, (row, source) in enumerate(zip(rows, manifest, strict=True)):
        width, height = int(source["width"]), int(source["height"])
        raw_output = str(row.get("raw_model_output") or "")
        if parser == "historical":
            raw_box, predicted, _ = historical_parse(raw_output, width, height)
        else:
            predicted, raw_box, _ = strict_parse(raw_output, width, height)
        iou = box_iou(predicted, [float(value) for value in source["gt_box"]]) if predicted else 0.0
        correct = iou >= 0.5
        parsed[index] = float(raw_box is not None)
        valid[index] = float(predicted is not None)
        ious[index] = iou
        correctness[index] = float(correct)
        if not np.isclose(float(row.get("box_iou", -1)), iou, atol=1e-10, rtol=0):
            raise QualityAnalysisError(f"{name} independently rescored IoU differs at row {index}")
        if bool(row.get("success_iou_0_5")) != correct:
            raise QualityAnalysisError(f"{name} independently rescored correctness differs at row {index}")
    return {
        "correctness": correctness,
        "ious": ious,
        "metrics": {
            "rows": EXPECTED_ROWS,
            "unique_ids": EXPECTED_ROWS,
            "parse_rate": float(parsed.mean()),
            "valid_output_rate": float(valid.mean()),
            "acc_at_0_5": float(correctness.mean()),
            "mean_iou": float(ious.mean()),
            "median_iou": float(np.median(ious)),
        },
    }


def paired_contrasts(arrays: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    requested = [
        ("K12", "Full"),
        ("M448_OPT", "Full"),
        ("Z_M448_0", "Full"),
        ("Z_M448_0", "K12"),
        ("Z_M448_0", "M448_OPT"),
        ("FEATHER_LLAVA_MAIN", "Full"),
        ("FEATHER_LLAVA_MAIN", "K12"),
        ("FEATHER_LLAVA_MAIN", "Z_M448_0"),
        ("NUWA_NATIVE_192", "NUWA_STACK_FULL"),
    ]
    output = []
    for contrast_index, (left, right) in enumerate(requested):
        if left not in arrays or right not in arrays:
            continue
        left_correct = arrays[left]["correctness"]
        right_correct = arrays[right]["correctness"]
        for metric, left_values, right_values, scale in (
            ("acc_at_0_5_pp", left_correct, right_correct, 100.0),
            ("mean_iou", arrays[left]["ious"], arrays[right]["ious"], 1.0),
        ):
            interval = paired_bootstrap_interval(
                left_values * scale,
                right_values * scale,
                resamples=BOOTSTRAPS,
                seed=SEED,
            )
            output.append(
                {
                    "contrast_order": contrast_index,
                    "contrast": f"{left}-minus-{right}",
                    "left": left,
                    "right": right,
                    "metric": metric,
                    **interval,
                    "left_correct_right_wrong": int(
                        np.sum((left_correct == 1) & (right_correct == 0))
                    ),
                    "left_wrong_right_correct": int(
                        np.sum((left_correct == 0) & (right_correct == 1))
                    ),
                }
            )
    return output


def main() -> int:
    if sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise QualityAnalysisError("Frozen full-validation manifest hash differs")
    manifest = read_jsonl(MANIFEST)
    protocol = json.loads((CAMPAIGN / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))
    arrays: dict[str, dict[str, Any]] = {}
    method_rows = []
    inventory: dict[str, Any] = {}
    for name, contract in METHODS.items():
        path = contract["path"]
        historical_key = contract["historical_key"]
        if historical_key is not None:
            frozen = protocol["historical_evidence"][historical_key]
            if sha256_file(path) != frozen["predictions_sha256"]:
                raise QualityAnalysisError(f"{name} historical prediction hash differs")
            completion_status = "immutable_historical_hash_pass"
        else:
            completion_path = path.parent / "COMPLETED.json"
            if not completion_path.is_file():
                raise QualityAnalysisError(f"{name} is missing COMPLETED.json")
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            completion_status = str(completion.get("status") or "").lower()
            if completion_status not in {"pass", "complete"}:
                raise QualityAnalysisError(f"{name} completion status is not accepted")
        rows = read_jsonl(path)
        rescored = rescore(name, rows, manifest, str(contract["parser"]))
        accounting = validate_trace(name, rows, contract["trace"])
        arrays[name] = rescored
        row = {"method": name, "stack": contract["stack"], **rescored["metrics"], **accounting}
        method_rows.append(row)
        inventory[name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(path),
            "rows": EXPECTED_ROWS,
            "stack": contract["stack"],
            "independent_rescore": "pass",
            "trace_validation": "pass",
            "completion_status": completion_status,
        }
    contrasts = paired_contrasts(arrays)
    exclusions = {
        "FASTV_MATCHED_199_vs_canonical": (
            "Not paired: the preregistered disabled-path gate found output divergence in the pinned "
            "official FastV fork, so the absolute FastV quality row is labelled by stack and is not "
            "treated as a canonical-stack paired effect."
        ),
        "NUWA_vs_canonical": (
            "Not paired: the native checkpoint/preprocessing stack failed exact parity; only "
            "NUWA_NATIVE_192 minus NUWA_STACK_FULL is calculated."
        ),
    }
    atomic_json(ANALYSIS / "QUALITY_METHOD_METRICS.json", method_rows)
    atomic_csv(ANALYSIS / "QUALITY_METHOD_METRICS.csv", method_rows)
    atomic_json(ANALYSIS / "QUALITY_PAIRED_CONTRASTS.json", contrasts)
    atomic_csv(ANALYSIS / "QUALITY_PAIRED_CONTRASTS.csv", contrasts)
    atomic_json(
        ANALYSIS / "QUALITY_RUN_INVENTORY.json",
        {
            "status": "complete",
            "manifest": {
                "path": str(MANIFEST.relative_to(REPO_ROOT)),
                "sha256": MANIFEST_SHA256,
                "rows": EXPECTED_ROWS,
            },
            "methods": inventory,
            "paired_comparison_exclusions": exclusions,
        },
    )
    print(json.dumps({"status": "complete", "methods": method_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
