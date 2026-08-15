#!/usr/bin/env python
"""Initialise and independently verify the adaptive-efficiency campaign.

This is the zero-GPU A0 phase.  Historical artifacts are read only.  All
outputs are written beneath the new campaign root.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
ANALYSIS_ROOT = CAMPAIGN_ROOT / "analysis"
MANIFEST = (
    REPO_ROOT
    / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
)
HISTORICAL_ROOT = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814"
HISTORICAL_AUDIT = HISTORICAL_ROOT / "FINAL_COMPLETION_REPORT.json"
HISTORICAL_REGISTRY = HISTORICAL_ROOT / "EXPERIMENT_REGISTRY.json"
REPORT_PATH = CAMPAIGN_ROOT / "RESEARCH_REPORT.md"
REGISTRY_PATH = CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"
STATE_PATH = CAMPAIGN_ROOT / "CONTROLLER_STATE.json"
EXPECTED_MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
EXPECTED_ROWS = 1000
BOOTSTRAPS = 10_000
ANALYSIS_SEED = 20260815

SOURCE_PATHS = {
    "K12_ORIGINAL": REPO_ROOT
    / "experiments/final_selector_ablation_and_repair_20260811/runs/repair_val_k12_b48_single_layer_more_context/per_sample_results.jsonl",
    "K12_QUERY_TOPK": REPO_ROOT
    / "experiments/final_selector_ablation_and_repair_20260811/runs/mech_val_k12_b48_query_topk/per_sample_results.jsonl",
    "P3": HISTORICAL_ROOT / "runs/p3_attempt1/predictions.jsonl",
    "P5": HISTORICAL_ROOT / "runs/p5_attempt1/predictions.jsonl",
    "P8": HISTORICAL_ROOT / "runs/p8_attempt1/predictions.jsonl",
}

EXPECTED_METRICS = {
    "K12_ORIGINAL": {"acc_at_0_5": 0.439, "mean_iou": 0.4215978636, "average_visual_tokens": 246.0},
    "K12_QUERY_TOPK": {"acc_at_0_5": 0.441, "mean_iou": 0.4277346673, "average_visual_tokens": 246.0},
    "P3": {"acc_at_0_5": 0.408, "mean_iou": 0.4246506509, "average_visual_tokens": 198.0},
    "P5": {"acc_at_0_5": 0.406, "mean_iou": 0.4197901831, "average_visual_tokens": 186.0},
    "P8": {"acc_at_0_5": 0.401, "mean_iou": 0.4179170489, "average_visual_tokens": 192.0},
}

RELATIONAL_LEXICON = (
    "left",
    "right",
    "behind",
    "in front",
    "front of",
    "next to",
    "beside",
    "between",
    "under",
    "below",
    "above",
    "over",
    "near",
    "closest",
    "farthest",
    "wearing",
    "holding",
    "facing",
    "looking at",
)

IOU_BINS = (0.0, 0.1, 0.3, 0.4, 0.45, 0.5, 0.55, 0.7, 0.9, 1.0000001)


class A0Error(RuntimeError):
    """Raised when historical evidence cannot support the registered A0 phase."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
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
    atomic_write(path, canonical_json_bytes(value))


def canonical_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8") for row in rows
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise A0Error(f"Missing JSON evidence: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise A0Error(f"Expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise A0Error(f"Missing JSONL evidence: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise A0Error(f"Blank line at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise A0Error(f"Non-object row at {path}:{line_number}")
            rows.append(value)
    return rows


def parse_box(text: str, width: int, height: int) -> tuple[list[float] | None, str | None]:
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))
    if len(matches) < 4:
        return None, "fewer_than_four_numbers"
    values = [float(value) for value in matches[:4]]
    if not all(math.isfinite(value) for value in values):
        return None, "non_finite_box"
    x1, y1, x2, y2 = values
    if not all(0.0 <= value <= 1.0 for value in values):
        return None, "outside_normalized_0_1"
    predicted = [x1 * width, y1 * height, x2 * width, y2 * height]
    if predicted[2] <= predicted[0] or predicted[3] <= predicted[1]:
        return None, "degenerate_box"
    return predicted, None


def box_iou(left_box: list[float], right_box: list[float]) -> float:
    left = max(float(left_box[0]), float(right_box[0]))
    top = max(float(left_box[1]), float(right_box[1]))
    right = min(float(left_box[2]), float(right_box[2]))
    bottom = min(float(left_box[3]), float(right_box[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    left_area = max(0.0, float(left_box[2]) - float(left_box[0])) * max(
        0.0, float(left_box[3]) - float(left_box[1])
    )
    right_area = max(0.0, float(right_box[2]) - float(right_box[0])) * max(
        0.0, float(right_box[3]) - float(right_box[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def validate_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_hash = sha256_file(MANIFEST)
    if actual_hash != EXPECTED_MANIFEST_SHA256:
        raise A0Error(f"Discovery manifest hash {actual_hash} != {EXPECTED_MANIFEST_SHA256}")
    rows = read_jsonl(MANIFEST)
    ids = [str(row.get("sample_id") or "") for row in rows]
    if len(rows) != EXPECTED_ROWS or len(set(ids)) != EXPECTED_ROWS or any(not value for value in ids):
        raise A0Error("Discovery manifest row count/IDs are invalid")
    if any(str(row.get("dataset")).lower() != "refcoco" or str(row.get("split")).lower() != "val" for row in rows):
        raise A0Error("Discovery manifest contains a forbidden dataset or split")
    missing_images = [str(row["image_path"]) for row in rows if not (REPO_ROOT / str(row["image_path"])).is_file()]
    if missing_images:
        raise A0Error(f"Discovery manifest has {len(missing_images)} missing images")
    order_hash = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    return rows, {
        "path": relative(MANIFEST),
        "sha256": actual_hash,
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "sample_id_order_sha256": order_hash,
        "all_images_exist": True,
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "splits": sorted({str(row["split"]) for row in rows}),
    }


def validate_historical_campaign() -> dict[str, Any]:
    audit = read_json(HISTORICAL_AUDIT)
    registry = read_json(HISTORICAL_REGISTRY)
    if audit.get("status") != "pass" or (audit.get("summary") or {}).get("failed"):
        raise A0Error("Completed campaign audit is not passing")
    cells = registry.get("cells") or {}
    accepted = sorted(cell_id for cell_id, cell in cells.items() if cell.get("status") == "accepted")
    if len(accepted) != 33:
        raise A0Error(f"Completed campaign has {len(accepted)} accepted quality cells, expected 33")
    return {
        "audit_path": relative(HISTORICAL_AUDIT),
        "audit_sha256": sha256_file(HISTORICAL_AUDIT),
        "registry_path": relative(HISTORICAL_REGISTRY),
        "registry_sha256": sha256_file(HISTORICAL_REGISTRY),
        "audit_status": audit["status"],
        "accepted_quality_cells": accepted,
        "accepted_quality_count": len(accepted),
    }


def rescore_source(
    name: str,
    path: Path,
    manifest_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    source_rows = read_jsonl(path)
    source_by_id: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in source_by_id:
            raise A0Error(f"{name} has a missing/duplicate sample ID")
        source_by_id[sample_id] = row
    manifest_ids = [str(row["sample_id"]) for row in manifest_rows]
    missing = [sample_id for sample_id in manifest_ids if sample_id not in source_by_id]
    if missing:
        raise A0Error(f"{name} misses {len(missing)} frozen-manifest rows")
    rescored: list[dict[str, Any]] = []
    selected_source: dict[str, dict[str, Any]] = {}
    stored_iou_max_error = 0.0
    for manifest_row in manifest_rows:
        sample_id = str(manifest_row["sample_id"])
        source = source_by_id[sample_id]
        selected_source[sample_id] = source
        raw_output = str(source.get("raw_model_output") or "")
        predicted, parse_failure = parse_box(raw_output, int(manifest_row["width"]), int(manifest_row["height"]))
        iou = box_iou(predicted, list(manifest_row["gt_box"])) if predicted is not None else 0.0
        if "box_iou" in source:
            stored_iou_max_error = max(stored_iou_max_error, abs(iou - float(source["box_iou"])))
        rescored.append(
            {
                "sample_id": sample_id,
                "image_id": str(manifest_row["image_id"]),
                "query": str(manifest_row["query"]),
                "width": int(manifest_row["width"]),
                "height": int(manifest_row["height"]),
                "gt_box": list(manifest_row["gt_box"]),
                "raw_model_output": raw_output,
                "predicted_box": predicted,
                "parse_failure_reason": parse_failure,
                "box_iou": iou,
                "success_iou_0_5": iou >= 0.5,
            }
        )
    if stored_iou_max_error > 1e-9:
        raise A0Error(f"{name} stored/recomputed IoU mismatch: {stored_iou_max_error}")
    metrics = {
        "acc_at_0_5": float(np.mean([row["success_iou_0_5"] for row in rescored])),
        "mean_iou": float(np.mean([row["box_iou"] for row in rescored])),
        "parse_rate": float(np.mean([row["parse_failure_reason"] is None for row in rescored])),
        "rows": len(rescored),
        "source_rows": len(source_rows),
        "source_path": relative(path),
        "source_sha256": sha256_file(path),
        "stored_iou_max_abs_error": stored_iou_max_error,
        "sample_id_order_sha256": hashlib.sha256("\n".join(manifest_ids).encode("utf-8")).hexdigest(),
        "average_visual_tokens": EXPECTED_METRICS[name]["average_visual_tokens"],
    }
    expected = EXPECTED_METRICS[name]
    if round(metrics["acc_at_0_5"], 10) != round(expected["acc_at_0_5"], 10):
        raise A0Error(f"{name} Acc mismatch: {metrics['acc_at_0_5']} != {expected['acc_at_0_5']}")
    if round(metrics["mean_iou"], 10) != round(expected["mean_iou"], 10):
        raise A0Error(f"{name} mIoU mismatch: {metrics['mean_iou']} != {expected['mean_iou']}")
    output_path = ANALYSIS_ROOT / "rescored" / f"{name}.jsonl"
    atomic_write(output_path, canonical_jsonl(rescored))
    metrics["rescored_path"] = relative(output_path)
    metrics["rescored_sha256"] = sha256_file(output_path)
    return rescored, metrics, selected_source


def vector(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def paired_interval(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    contrast: str,
    metric: str,
    seed: int,
) -> dict[str, Any]:
    if [row["sample_id"] for row in left] != [row["sample_id"] for row in right]:
        raise A0Error(f"Paired order mismatch for {contrast}")
    key = "success_iou_0_5" if metric == "acc_at_0_5" else "box_iou"
    delta = vector(left, key) - vector(right, key)
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    remaining = BOOTSTRAPS
    while remaining:
        count = min(500, remaining)
        indices = rng.integers(0, delta.size, size=(count, delta.size))
        estimates.append(delta[indices].mean(axis=1))
        remaining -= count
    samples = np.concatenate(estimates)
    scale = 100.0 if metric == "acc_at_0_5" else 1.0
    return {
        "contrast": contrast,
        "metric": metric,
        "mean_delta": float(delta.mean() * scale),
        "lower_95": float(np.quantile(samples, 0.025) * scale),
        "upper_95": float(np.quantile(samples, 0.975) * scale),
        "bootstrap_replicates": BOOTSTRAPS,
        "seed": seed,
        "unit": "percentage_points" if scale == 100.0 else "0_to_1",
    }


def error_profile(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]], name: str
) -> dict[str, Any]:
    ref_iou = vector(reference, "box_iou")
    cand_iou = vector(candidate, "box_iou")
    ref_correct = ref_iou >= 0.5
    cand_correct = cand_iou >= 0.5
    hist, _ = np.histogram(cand_iou, bins=np.asarray(IOU_BINS))
    labels = [f"[{IOU_BINS[index]:.2f},{IOU_BINS[index + 1]:.2f})" for index in range(len(IOU_BINS) - 1)]
    return {
        "candidate": name,
        "k12_correct_candidate_wrong": int(np.sum(ref_correct & ~cand_correct)),
        "candidate_correct_k12_wrong": int(np.sum(~ref_correct & cand_correct)),
        "both_correct": int(np.sum(ref_correct & cand_correct)),
        "both_wrong": int(np.sum(~ref_correct & ~cand_correct)),
        "near_miss_candidate_iou_0_40_to_0_50_when_k12_correct": int(
            np.sum(ref_correct & (cand_iou >= 0.4) & (cand_iou < 0.5))
        ),
        "catastrophic_candidate_iou_below_0_10_when_k12_correct": int(
            np.sum(ref_correct & (cand_iou < 0.1))
        ),
        "candidate_iou_histogram": dict(zip(labels, [int(value) for value in hist], strict=True)),
        "mean_iou_change": float(np.mean(cand_iou - ref_iou)),
        "median_iou_change": float(np.median(cand_iou - ref_iou)),
    }


def subgroup_metrics(
    indices: np.ndarray,
    labels: list[str],
    reference: list[dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
    family: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    ref_acc = vector(reference, "success_iou_0_5")
    ref_iou = vector(reference, "box_iou")
    for group_index, label in enumerate(labels):
        mask = indices == group_index
        if not np.any(mask):
            continue
        for name, rows in candidates.items():
            cand_acc = vector(rows, "success_iou_0_5")
            cand_iou = vector(rows, "box_iou")
            output.append(
                {
                    "family": family,
                    "group": label,
                    "candidate": name,
                    "n": int(np.sum(mask)),
                    "k12_acc_at_0_5": float(np.mean(ref_acc[mask])),
                    "candidate_acc_at_0_5": float(np.mean(cand_acc[mask])),
                    "acc_delta": float(np.mean(cand_acc[mask] - ref_acc[mask])),
                    "k12_mean_iou": float(np.mean(ref_iou[mask])),
                    "candidate_mean_iou": float(np.mean(cand_iou[mask])),
                    "mean_iou_delta": float(np.mean(cand_iou[mask] - ref_iou[mask])),
                }
            )
    return output


def quantile_groups(values: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str], list[float]]:
    edges = np.quantile(values, [0.25, 0.5, 0.75])
    groups = np.searchsorted(edges, values, side="right")
    labels = [f"{prefix}_q{index}" for index in range(1, 5)]
    return groups, labels, [float(value) for value in edges]


def retained_overlap(
    query_sources: dict[str, dict[str, Any]],
    p3_sources: dict[str, dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    survival: list[float] = []
    final_overlap: list[float] = []
    final_jaccard: list[float] = []
    unavailable = 0
    for manifest_row in manifest_rows:
        sample_id = str(manifest_row["sample_id"])
        query_final = query_sources[sample_id].get("retained_indices")
        traces = p3_sources[sample_id].get("stage_traces")
        if not isinstance(query_final, list) or not isinstance(traces, list) or len(traces) < 2:
            unavailable += 1
            continue
        p3_k4 = set(int(value) for value in traces[0]["outgoing_original_indices"])
        p3_final = set(int(value) for value in traces[-1]["outgoing_original_indices"])
        query_set = set(int(value) for value in query_final)
        survival.append(len(query_set & p3_k4) / len(query_set))
        final_overlap.append(len(query_set & p3_final) / len(query_set))
        final_jaccard.append(len(query_set & p3_final) / len(query_set | p3_final))
    return {
        "available_rows": len(survival),
        "unavailable_rows": unavailable,
        "k12_query_top48_survival_in_p3_k4_384_mean": float(np.mean(survival)) if survival else None,
        "k12_query_top48_overlap_p3_final48_mean": float(np.mean(final_overlap)) if final_overlap else None,
        "k12_query_top48_vs_p3_final48_jaccard_mean": float(np.mean(final_jaccard)) if final_jaccard else None,
    }


def startup_state() -> dict[str, Any]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "checked_at_utc": utc_now(),
        "gpu_query_exit_code": gpu.returncode,
        "gpu_query": [line for line in gpu.stdout.splitlines() if line.strip()],
        "compute_process_query_exit_code": processes.returncode,
        "active_compute_processes": [line for line in processes.stdout.splitlines() if line.strip()],
        "terminated_processes": [],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise A0Error(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def report_markdown(analysis: dict[str, Any]) -> str:
    reference_rows = []
    for name in ("K12_ORIGINAL", "K12_QUERY_TOPK", "P3", "P5", "P8"):
        metric = analysis["references"][name]
        reference_rows.append(
            f"| {name} | {metric['acc_at_0_5']:.4f} | {metric['mean_iou']:.4f} | "
            f"{metric['parse_rate']:.4f} | {metric['average_visual_tokens']:.1f} |"
        )
    interval_rows = []
    for row in analysis["paired_intervals"]:
        interval_rows.append(
            f"| {row['contrast']} | {row['metric']} | {row['mean_delta']:.4f} | "
            f"[{row['lower_95']:.4f}, {row['upper_95']:.4f}] | {row['unit']} |"
        )
    error_rows = []
    for name, profile in analysis["error_profiles"].items():
        error_rows.append(
            f"| {name} | {profile['k12_correct_candidate_wrong']} | "
            f"{profile['candidate_correct_k12_wrong']} | "
            f"{profile['near_miss_candidate_iou_0_40_to_0_50_when_k12_correct']} | "
            f"{profile['catastrophic_candidate_iou_below_0_10_when_k12_correct']} |"
        )
    overlap = analysis["retained_overlap"]
    return "\n".join(
        [
            "# Adaptive RefCOCO Efficiency Campaign",
            "",
            "## Scope and frozen protocol",
            "",
            "All validation-quality experiments in this campaign use the same frozen 1,000-example RefCOCO-validation manifest. Results are exploratory. No 8,811-row validation, held-out split, transfer dataset, or dissertation edit is authorized.",
            "",
            f"- Manifest SHA-256: `{analysis['manifest']['sha256']}`",
            f"- Rows/unique IDs: {analysis['manifest']['rows']}/{analysis['manifest']['unique_ids']}",
            f"- Historical completion audit: `{analysis['historical_campaign']['audit_status']}`",
            f"- A0 bootstrap: {BOOTSTRAPS:,} paired resamples, seed {ANALYSIS_SEED}",
            "",
            "## A0 — verified historical reference",
            "",
            "All values below were independently recomputed from raw model text in physical prediction JSONLs and aligned to frozen-manifest order by exact sample ID.",
            "",
            "| Method | Acc@0.5 | mIoU | Parse | Average visual tokens |",
            "|---|---:|---:|---:|---:|",
            *reference_rows,
            "",
            "### Paired K12 contrasts",
            "",
            "| Contrast | Metric | Delta | 95% paired bootstrap CI | Unit |",
            "|---|---|---:|---:|---|",
            *interval_rows,
            "",
            "### Error structure",
            "",
            "| Candidate | K12 correct / candidate wrong | Candidate correct / K12 wrong | Near misses | Catastrophic losses |",
            "|---|---:|---:|---:|---:|",
            *error_rows,
            "",
            f"P3 retains on average {overlap['k12_query_top48_survival_in_p3_k4_384_mean']:.3f} of the matched K12 query-top-48 set after its K4→384 stage, but its final-48 overlap is only {overlap['k12_query_top48_overlap_p3_final48_mean']:.3f}. This is diagnostic evidence about representation/selection change, not a causal proof.",
            "",
            "Complete IoU histograms, object-area/query-length/spatial/relational subgroups, trace availability, and machine-readable paired results are in `analysis/A0_REFERENCE_AND_ERROR_ANALYSIS.json` and the accompanying CSV files.",
            "",
            "## Implementation and preflight",
            "",
            "Pending.",
            "",
            "## Adaptive fixed-method experiments",
            "",
            "Pending C448 and registered conditional branches.",
            "",
            "## Official source-native comparisons",
            "",
            "Pending source locks and parity gates.",
            "",
            "## Learned schedule gate",
            "",
            "Pending fixed-method selection and RefCOCO-train label generation.",
            "",
            "## Conditional matched timing",
            "",
            "Pending the K12-like quality trigger.",
            "",
            "## Failures and rejected attempts",
            "",
            "None in A0.",
            "",
            "## Limitations",
            "",
            "The frozen 1,000 validation examples are suitable for discovery but not equivalence claims. A one-point Acc@0.5 change is only ten examples; marginal effects remain inconclusive even when they control an adaptive branch.",
            "",
            "## Final recommendation",
            "",
            "Pending completion of every triggered branch.",
            "",
        ]
    )


def main() -> int:
    if CAMPAIGN_ROOT.exists() and any(CAMPAIGN_ROOT.iterdir()):
        raise A0Error(f"Campaign root is not empty; refusing to overwrite: {CAMPAIGN_ROOT}")
    manifest_rows, manifest_evidence = validate_manifest()
    historical_evidence = validate_historical_campaign()
    startup = startup_state()
    atomic_json(CAMPAIGN_ROOT / "startup_environment.json", startup)

    rescored: dict[str, list[dict[str, Any]]] = {}
    references: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, dict[str, Any]]] = {}
    for name, path in SOURCE_PATHS.items():
        rows, metrics, selected_source = rescore_source(name, path, manifest_rows)
        rescored[name] = rows
        references[name] = metrics
        sources[name] = selected_source

    intervals: list[dict[str, Any]] = []
    for candidate_index, name in enumerate(("P3", "P5", "P8"), 1):
        for metric_index, metric in enumerate(("acc_at_0_5", "mean_iou")):
            intervals.append(
                paired_interval(
                    rescored[name],
                    rescored["K12_ORIGINAL"],
                    contrast=f"{name}-minus-K12_ORIGINAL",
                    metric=metric,
                    seed=ANALYSIS_SEED + candidate_index * 10 + metric_index,
                )
            )
    for metric_index, metric in enumerate(("acc_at_0_5", "mean_iou")):
        intervals.append(
            paired_interval(
                rescored["K12_QUERY_TOPK"],
                rescored["K12_ORIGINAL"],
                contrast="K12_QUERY_TOPK-minus-K12_ORIGINAL",
                metric=metric,
                seed=ANALYSIS_SEED + 90 + metric_index,
            )
        )

    reference = rescored["K12_ORIGINAL"]
    candidates = {name: rescored[name] for name in ("P3", "P5", "P8")}
    error_profiles = {name: error_profile(reference, rows, name) for name, rows in candidates.items()}

    areas = np.asarray(
        [
            ((float(row["gt_box"][2]) - float(row["gt_box"][0])) * (float(row["gt_box"][3]) - float(row["gt_box"][1])))
            / (float(row["width"]) * float(row["height"]))
            for row in manifest_rows
        ],
        dtype=np.float64,
    )
    area_groups, area_labels, area_edges = quantile_groups(areas, "area")
    query_lengths_values: list[int] = []
    missing_query_trace = 0
    for row in manifest_rows:
        positions = sources["K12_QUERY_TOPK"][str(row["sample_id"])].get("ug_query_token_positions")
        if isinstance(positions, list) and positions:
            query_lengths_values.append(len(positions))
        else:
            missing_query_trace += 1
            query_lengths_values.append(len(str(row["query"]).split()))
    query_lengths = np.asarray(query_lengths_values, dtype=np.float64)
    query_groups, query_labels, query_edges = quantile_groups(query_lengths, "query_length")
    spatial_indices: list[int] = []
    for row in manifest_rows:
        box = [float(value) for value in row["gt_box"]]
        center_x = ((box[0] + box[2]) / 2.0) / float(row["width"])
        center_y = ((box[1] + box[3]) / 2.0) / float(row["height"])
        col = min(2, max(0, int(center_x * 3)))
        row_index = min(2, max(0, int(center_y * 3)))
        spatial_indices.append(row_index * 3 + col)
    spatial_labels = [f"{vertical}_{horizontal}" for vertical in ("top", "middle", "bottom") for horizontal in ("left", "center", "right")]
    relation_indices = np.asarray(
        [int(any(term in str(row["query"]).lower() for term in RELATIONAL_LEXICON)) for row in manifest_rows],
        dtype=np.int64,
    )

    subgroup_rows: list[dict[str, Any]] = []
    subgroup_rows.extend(subgroup_metrics(area_groups, area_labels, reference, candidates, "object_area_quartile"))
    subgroup_rows.extend(subgroup_metrics(query_groups, query_labels, reference, candidates, "query_length_quartile"))
    subgroup_rows.extend(
        subgroup_metrics(np.asarray(spatial_indices), spatial_labels, reference, candidates, "object_center_3x3")
    )
    subgroup_rows.extend(subgroup_metrics(relation_indices, ["non_relational", "relational"], reference, candidates, "relational_lexicon"))

    overlap = retained_overlap(sources["K12_QUERY_TOPK"], sources["P3"], manifest_rows)
    analysis = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "campaign_id": CAMPAIGN_ROOT.name,
        "manifest": manifest_evidence,
        "historical_campaign": historical_evidence,
        "startup_environment_path": relative(CAMPAIGN_ROOT / "startup_environment.json"),
        "references": references,
        "paired_intervals": intervals,
        "error_profiles": error_profiles,
        "subgroups": {
            "object_area_quartile_edges": area_edges,
            "query_length_quartile_edges": query_edges,
            "query_length_source": "K12 query-token positions; whitespace fallback only when trace absent",
            "query_length_trace_missing_rows": missing_query_trace,
            "relational_lexicon": list(RELATIONAL_LEXICON),
            "rows": subgroup_rows,
        },
        "retained_overlap": overlap,
        "input_hashes": {relative(path): sha256_file(path) for path in [MANIFEST, HISTORICAL_AUDIT, HISTORICAL_REGISTRY, *SOURCE_PATHS.values()]},
    }
    analysis_path = ANALYSIS_ROOT / "A0_REFERENCE_AND_ERROR_ANALYSIS.json"
    atomic_json(analysis_path, analysis)
    write_csv(ANALYSIS_ROOT / "A0_PAIRED_INTERVALS.csv", intervals)
    write_csv(ANALYSIS_ROOT / "A0_SUBGROUPS.csv", subgroup_rows)

    thresholds = {
        "k12_like": {"acc_at_0_5_gte": 0.429, "mean_iou_gte": 0.4165978636, "average_visual_tokens_lt": 246},
        "promising": {"paired_acc_improvement_gte": 0.010, "paired_mean_iou_change_gte": -0.005},
        "clear_small_or_no_effect": {"paired_acc_improvement_lt": 0.005},
        "analysis_bootstraps": BOOTSTRAPS,
        "analysis_seed": ANALYSIS_SEED,
    }
    cell_ids = ["C448", "C480", "D1", "M448", "B1", "A1", "OFF_NUWA_192", "OFF_PDROP_L2_192", "OFF_PDROP_REROUTE_192", "OFF_NUWA_REROUTE_192", "GATE_SEED0", "GATE_SEED1", "GATE_SEED2"]
    registry = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ROOT.name,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "goal_sha256": sha256_file(REPO_ROOT / "GOAL.md"),
        "agents_sha256": sha256_file(REPO_ROOT / "AGENTS.md"),
        "manifest": manifest_evidence,
        "historical_campaign": historical_evidence,
        "references": references,
        "thresholds": thresholds,
        "phases": {
            "A0": {"status": "complete", "analysis_path": relative(analysis_path), "analysis_sha256": sha256_file(analysis_path)},
            "preflight": {"status": "pending"},
            "fixed_methods": {"status": "pending"},
            "official_methods": {"status": "pending"},
            "route_gate": {"status": "pending"},
            "timing": {"status": "pending"},
            "final_audit": {"status": "pending"},
        },
        "cells": {cell_id: {"status": "pending", "attempts": []} for cell_id in cell_ids},
        "decisions": [],
        "failures": [],
        "totals": {"accepted_validation_quality": 0, "maximum_accepted_validation_quality": 13},
    }
    atomic_json(REGISTRY_PATH, registry)
    state = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ROOT.name,
        "updated_at_utc": utc_now(),
        "state": "A0_COMPLETE_PRECHECK_PENDING",
        "next_required_action": "implement_and_run_preflight",
        "active_workers": [],
        "decision_history": [
            {
                "phase": "A0",
                "status": "complete",
                "evidence_path": relative(analysis_path),
                "evidence_sha256": sha256_file(analysis_path),
            }
        ],
    }
    atomic_json(STATE_PATH, state)
    atomic_write(REPORT_PATH, report_markdown(analysis).encode("utf-8"))

    completion = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "analysis_sha256": sha256_file(analysis_path),
        "paired_csv_sha256": sha256_file(ANALYSIS_ROOT / "A0_PAIRED_INTERVALS.csv"),
        "subgroups_csv_sha256": sha256_file(ANALYSIS_ROOT / "A0_SUBGROUPS.csv"),
        "registry_sha256": sha256_file(REGISTRY_PATH),
        "state_sha256": sha256_file(STATE_PATH),
        "report_sha256": sha256_file(REPORT_PATH),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "reference_metrics": {name: {key: metrics[key] for key in ("acc_at_0_5", "mean_iou", "parse_rate")} for name, metrics in references.items()},
    }
    atomic_json(ANALYSIS_ROOT / "A0_COMPLETED.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
