#!/usr/bin/env python
"""Compare two UG-Prune-Lite checkpoints on matched per-sample artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


GRID_SIZE = 24
BIN_COUNT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache", required=True, type=Path)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--left-selector-eval", required=True, type=Path)
    parser.add_argument("--left-generation", required=True, type=Path)
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--right-selector-eval", required=True, type=Path)
    parser.add_argument("--right-generation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--collapse-bin-threshold", type=int, default=4)
    parser.add_argument("--teacher-recall-threshold", type=float, default=0.35)
    parser.add_argument("--role-recall-threshold", type=float, default=0.35)
    parser.add_argument("--gt-coverage-threshold", type=float, default=0.15)
    parser.add_argument("--large-iou-delta", type=float, default=0.10)
    return parser.parse_args()


def resolve_file(path: Path, default_name: str) -> Path:
    return path / default_name if path.is_dir() else path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def index_by_sample(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            continue
        if sample_id in indexed:
            duplicates.append(sample_id)
        indexed[sample_id] = row
    if duplicates:
        raise ValueError(f"{label} contains duplicate sample_id values: {duplicates[:5]}")
    return indexed


def checksum_indices(indices: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for index in sorted(int(i) for i in indices):
        digest.update(str(index).encode("ascii"))
        digest.update(b",")
    return digest.hexdigest()[:16]


def clean_indices(indices: Any) -> list[int]:
    if indices is None:
        return []
    return sorted(int(i) for i in indices)


def mean(values: Iterable[float | int | None]) -> float | None:
    valid = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return sum(valid) / float(len(valid)) if valid else None


def std(values: list[float]) -> float | None:
    return statistics.pstdev(values) if values else None


def recall(selected: Iterable[int], reference: Iterable[int]) -> float | None:
    ref = set(int(i) for i in reference)
    if not ref:
        return None
    sel = set(int(i) for i in selected)
    return len(sel & ref) / float(len(ref))


def jaccard(left: Iterable[int], right: Iterable[int]) -> float | None:
    left_set = set(int(i) for i in left)
    right_set = set(int(i) for i in right)
    union = left_set | right_set
    if not union:
        return None
    return len(left_set & right_set) / float(len(union))


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    x_mean = sum(x_vals) / len(x_vals)
    y_mean = sum(y_vals) / len(y_vals)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in x_vals))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in y_vals))
    if x_den == 0.0 or y_den == 0.0:
        return None
    return numerator / (x_den * y_den)


def spatial_stats(indices: list[int], grid_size: int) -> dict[str, Any]:
    rows = [idx // grid_size for idx in indices]
    cols = [idx % grid_size for idx in indices]
    hist = [0 for _ in range(BIN_COUNT * BIN_COUNT)]
    for row, col in zip(rows, cols):
        bin_row = min(BIN_COUNT - 1, int(row * BIN_COUNT / grid_size))
        bin_col = min(BIN_COUNT - 1, int(col * BIN_COUNT / grid_size))
        hist[bin_row * BIN_COUNT + bin_col] += 1
    occupied = sum(1 for value in hist if value > 0)
    total = float(sum(hist))
    entropy = 0.0
    if total > 0:
        for value in hist:
            if value > 0:
                p = value / total
                entropy -= p * math.log(p)
    spread_values: list[float] = []
    coords = list(zip(rows, cols))
    for i, (row_a, col_a) in enumerate(coords):
        for row_b, col_b in coords[i + 1 :]:
            spread_values.append(math.hypot(float(row_a - row_b), float(col_a - col_b)))
    return {
        "selected_count": len(indices),
        "unique_count": len(set(indices)),
        "row_mean": mean(rows),
        "row_std": std([float(v) for v in rows]),
        "col_mean": mean(cols),
        "col_std": std([float(v) for v in cols]),
        "occupied_3x3_bins": occupied,
        "hist_3x3": hist,
        "spatial_entropy_3x3": entropy,
        "mean_pairwise_spread": mean(spread_values),
    }


def patch_coverage(indices: list[int], gt_box: Any, width: Any, height: Any, grid_size: int) -> float | None:
    if not isinstance(gt_box, list) or len(gt_box) != 4:
        return None
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        return None
    x1, y1, x2, y2 = [float(v) for v in gt_box]
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if box_area <= 0.0:
        return None
    patch_w = float(width) / float(grid_size)
    patch_h = float(height) / float(grid_size)
    covered = 0.0
    for idx in set(indices):
        row = int(idx) // grid_size
        col = int(idx) % grid_size
        px1 = col * patch_w
        py1 = row * patch_h
        px2 = px1 + patch_w
        py2 = py1 + patch_h
        ix1 = max(x1, px1)
        iy1 = max(y1, py1)
        ix2 = min(x2, px2)
        iy2 = min(y2, py2)
        covered += max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return covered / box_area


def generation_success(row: dict[str, Any]) -> bool:
    return bool(row.get("success_iou_0_5", False))


def generation_iou(row: dict[str, Any]) -> float | None:
    value = row.get("box_iou", row.get("score"))
    return float(value) if isinstance(value, (int, float)) else None


def failure_label(
    selector_row: dict[str, Any],
    generation_row: dict[str, Any],
    *,
    collapse_bin_threshold: int,
    teacher_recall_threshold: float,
    role_recall_threshold: float,
    gt_coverage_threshold: float,
) -> str:
    if generation_row.get("invalid") or generation_row.get("parse_failure_reason"):
        return "parse_or_format_issue"
    if generation_success(generation_row):
        return "success"
    if float(selector_row.get("selection/occupied_3x3_bins", 0.0)) < collapse_bin_threshold:
        return "spatial_collapse"
    if float(selector_row.get("selection/teacher_recall_at_58", 0.0)) < teacher_recall_threshold:
        return "teacher_overlap_low"
    if float(selector_row.get("selection/target_recall", 0.0)) < role_recall_threshold:
        return "target_overlap_low"
    if float(selector_row.get("selection/context_recall", 0.0)) < role_recall_threshold:
        return "context_overlap_low"
    if float(selector_row.get("selection/gt_box_patch_coverage_at_58", 0.0)) < gt_coverage_threshold:
        return "gt_coverage_low"
    return "box_wrong_despite_coverage"


def side_payload(
    selector_row: dict[str, Any],
    generation_row: dict[str, Any],
    teacher_row: dict[str, Any],
    *,
    grid_size: int,
    collapse_bin_threshold: int,
    teacher_recall_threshold: float,
    role_recall_threshold: float,
    gt_coverage_threshold: float,
) -> dict[str, Any]:
    selected = clean_indices(selector_row.get("selected_indices"))
    teacher_selected = clean_indices(teacher_row.get("selected_indices"))
    roles = teacher_row.get("role_labels") or {}
    stats = spatial_stats(selected, grid_size)
    computed_gt_coverage = patch_coverage(
        selected,
        teacher_row.get("gt_box"),
        teacher_row.get("width"),
        teacher_row.get("height"),
        grid_size,
    )
    return {
        "selected_checksum": selector_row.get("selected_checksum") or checksum_indices(selected),
        "generation_retained_checksum": generation_row.get("retained_indices_checksum"),
        "selected_indices": selected,
        "teacher_recall_at_58": recall(selected, teacher_selected),
        "target_recall": recall(selected, roles.get("target", [])),
        "context_recall": recall(selected, roles.get("context", [])),
        "anchor_recall": recall(selected, roles.get("anchor", [])),
        "gt_box_patch_coverage_at_58": selector_row.get("selection/gt_box_patch_coverage_at_58", computed_gt_coverage),
        "computed_gt_box_patch_coverage_at_58": computed_gt_coverage,
        "occupied_3x3_bins": selector_row.get("selection/occupied_3x3_bins", stats["occupied_3x3_bins"]),
        "mean_pairwise_spread": selector_row.get("selection/mean_pairwise_spread", stats["mean_pairwise_spread"]),
        "spatial": stats,
        "generated_text": generation_row.get("raw_model_output"),
        "predicted_box": generation_row.get("predicted_box"),
        "raw_parsed_box": generation_row.get("raw_parsed_box"),
        "iou": generation_iou(generation_row),
        "success_iou_0_5": generation_success(generation_row),
        "valid": generation_row.get("valid"),
        "invalid": generation_row.get("invalid"),
        "parse_failure_reason": generation_row.get("parse_failure_reason"),
        "latency_s": generation_row.get("latency_s", generation_row.get("total_inference_latency_s")),
        "selector_latency_s": generation_row.get("student_selector_latency_s", selector_row.get("selector_latency_s")),
        "failure_label": failure_label(
            selector_row,
            generation_row,
            collapse_bin_threshold=collapse_bin_threshold,
            teacher_recall_threshold=teacher_recall_threshold,
            role_recall_threshold=role_recall_threshold,
            gt_coverage_threshold=gt_coverage_threshold,
        ),
    }


def flip_type(left: dict[str, Any], right: dict[str, Any]) -> str:
    left_success = bool(left["success_iou_0_5"])
    right_success = bool(right["success_iou_0_5"])
    if left_success and right_success:
        return "both_success"
    if left_success and not right_success:
        return "left_success_right_fail"
    if not left_success and right_success:
        return "left_fail_right_success"
    return "both_fail"


def numeric_delta(right: Any, left: Any) -> float | None:
    if isinstance(right, (int, float)) and isinstance(left, (int, float)):
        return float(right) - float(left)
    return None


def side_aggregate(rows: list[dict[str, Any]], side_name: str) -> dict[str, Any]:
    side_rows = [row[side_name] for row in rows]
    return {
        "acc_at_0_5": mean(1.0 if row["success_iou_0_5"] else 0.0 for row in side_rows),
        "mean_iou": mean(row.get("iou") for row in side_rows),
        "valid_rate": mean(1.0 if row.get("valid") else 0.0 for row in side_rows),
        "mean_latency_s": mean(row.get("latency_s") for row in side_rows),
        "mean_selector_latency_s": mean(row.get("selector_latency_s") for row in side_rows),
        "mean_teacher_recall_at_58": mean(row.get("teacher_recall_at_58") for row in side_rows),
        "mean_target_recall": mean(row.get("target_recall") for row in side_rows),
        "mean_context_recall": mean(row.get("context_recall") for row in side_rows),
        "mean_anchor_recall": mean(row.get("anchor_recall") for row in side_rows),
        "mean_gt_box_patch_coverage_at_58": mean(row.get("gt_box_patch_coverage_at_58") for row in side_rows),
        "mean_occupied_3x3_bins": mean(row.get("occupied_3x3_bins") for row in side_rows),
        "mean_pairwise_spread": mean(row.get("mean_pairwise_spread") for row in side_rows),
        "failure_labels": dict(Counter(str(row.get("failure_label")) for row in side_rows)),
    }


def build_summary(rows: list[dict[str, Any]], left_name: str, right_name: str) -> dict[str, Any]:
    flip_counts = Counter(row["flip_type"] for row in rows)
    jaccards = [row["selection_jaccard"] for row in rows if row.get("selection_jaccard") is not None]
    left_teacher_recalls = [float(row[left_name]["teacher_recall_at_58"]) for row in rows]
    right_teacher_recalls = [float(row[right_name]["teacher_recall_at_58"]) for row in rows]
    left_coverages = [float(row[left_name]["gt_box_patch_coverage_at_58"]) for row in rows]
    right_coverages = [float(row[right_name]["gt_box_patch_coverage_at_58"]) for row in rows]
    left_ious = [float(row[left_name]["iou"]) for row in rows if row[left_name]["iou"] is not None]
    right_ious = [float(row[right_name]["iou"]) for row in rows if row[right_name]["iou"] is not None]
    left_hist = [row[left_name]["spatial"]["hist_3x3"] for row in rows]
    right_hist = [row[right_name]["spatial"]["hist_3x3"] for row in rows]
    mean_left_hist = [mean(hist[i] for hist in left_hist) for i in range(BIN_COUNT * BIN_COUNT)]
    mean_right_hist = [mean(hist[i] for hist in right_hist) for i in range(BIN_COUNT * BIN_COUNT)]

    left_agg = side_aggregate(rows, left_name)
    right_agg = side_aggregate(rows, right_name)
    return {
        "samples/compared": len(rows),
        "left/name": left_name,
        "right/name": right_name,
        "left": left_agg,
        "right": right_agg,
        "delta/right_minus_left": {
            "acc_at_0_5": numeric_delta(right_agg["acc_at_0_5"], left_agg["acc_at_0_5"]),
            "mean_iou": numeric_delta(right_agg["mean_iou"], left_agg["mean_iou"]),
            "mean_latency_s": numeric_delta(right_agg["mean_latency_s"], left_agg["mean_latency_s"]),
            "mean_teacher_recall_at_58": numeric_delta(right_agg["mean_teacher_recall_at_58"], left_agg["mean_teacher_recall_at_58"]),
            "mean_target_recall": numeric_delta(right_agg["mean_target_recall"], left_agg["mean_target_recall"]),
            "mean_context_recall": numeric_delta(right_agg["mean_context_recall"], left_agg["mean_context_recall"]),
            "mean_gt_box_patch_coverage_at_58": numeric_delta(
                right_agg["mean_gt_box_patch_coverage_at_58"],
                left_agg["mean_gt_box_patch_coverage_at_58"],
            ),
        },
        "flip_counts": dict(flip_counts),
        "selection_jaccard_mean": mean(jaccards),
        "selection_checksum_same_count": sum(1 for row in rows if row[left_name]["selected_checksum"] == row[right_name]["selected_checksum"]),
        "correlations": {
            f"{left_name}/teacher_recall_vs_iou": pearson(left_teacher_recalls, left_ious),
            f"{right_name}/teacher_recall_vs_iou": pearson(right_teacher_recalls, right_ious),
            f"{left_name}/gt_coverage_vs_iou": pearson(left_coverages, left_ious),
            f"{right_name}/gt_coverage_vs_iou": pearson(right_coverages, right_ious),
        },
        "spatial_distribution": {
            f"{left_name}/mean_hist_3x3": mean_left_hist,
            f"{right_name}/mean_hist_3x3": mean_right_hist,
            "right_minus_left_mean_hist_3x3": [
                (float(right or 0.0) - float(left or 0.0)) for left, right in zip(mean_left_hist, mean_right_hist)
            ],
            "mean_selection_jaccard": mean(jaccards),
            "mean_centroid_row_delta_right_minus_left": mean(
                numeric_delta(row[right_name]["spatial"]["row_mean"], row[left_name]["spatial"]["row_mean"]) for row in rows
            ),
            "mean_centroid_col_delta_right_minus_left": mean(
                numeric_delta(row[right_name]["spatial"]["col_mean"], row[left_name]["spatial"]["col_mean"]) for row in rows
            ),
        },
        "diagnostic_answers": {
            "right_picks_mostly_same_regions": (mean(jaccards) or 0.0) >= 0.65,
            "right_loses_target_recall": (right_agg["mean_target_recall"] or 0.0) < (left_agg["mean_target_recall"] or 0.0),
            "right_loses_context_recall": (right_agg["mean_context_recall"] or 0.0) < (left_agg["mean_context_recall"] or 0.0),
            "right_improves_selector_metrics_but_worsens_generation": (
                (right_agg["mean_teacher_recall_at_58"] or 0.0) > (left_agg["mean_teacher_recall_at_58"] or 0.0)
                and (right_agg["acc_at_0_5"] or 0.0) < (left_agg["acc_at_0_5"] or 0.0)
            ),
            "teacher_recall_predicts_iou_left": abs(pearson(left_teacher_recalls, left_ious) or 0.0) >= 0.25,
            "teacher_recall_predicts_iou_right": abs(pearson(right_teacher_recalls, right_ious) or 0.0) >= 0.25,
            "gt_coverage_predicts_iou_left": abs(pearson(left_coverages, left_ious) or 0.0) >= 0.25,
            "gt_coverage_predicts_iou_right": abs(pearson(right_coverages, right_ious) or 0.0) >= 0.25,
        },
    }


def write_failure_flips(path: Path, rows: list[dict[str, Any]], left_name: str, right_name: str, large_iou_delta: float) -> None:
    fieldnames = [
        "sample_id",
        "flip_type",
        "query",
        f"{left_name}_iou",
        f"{right_name}_iou",
        "right_minus_left_iou",
        f"{left_name}_success",
        f"{right_name}_success",
        f"{left_name}_failure_label",
        f"{right_name}_failure_label",
        f"{left_name}_teacher_recall",
        f"{right_name}_teacher_recall",
        f"{left_name}_gt_coverage",
        f"{right_name}_gt_coverage",
    ]
    interesting = []
    for row in rows:
        delta = numeric_delta(row[right_name].get("iou"), row[left_name].get("iou"))
        if row["flip_type"] != "both_success" or (delta is not None and abs(delta) >= large_iou_delta):
            interesting.append((abs(delta or 0.0), row))
    interesting.sort(key=lambda item: item[0], reverse=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for _delta_abs, row in interesting:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "flip_type": row["flip_type"],
                    "query": row.get("query"),
                    f"{left_name}_iou": row[left_name].get("iou"),
                    f"{right_name}_iou": row[right_name].get("iou"),
                    "right_minus_left_iou": numeric_delta(row[right_name].get("iou"), row[left_name].get("iou")),
                    f"{left_name}_success": row[left_name].get("success_iou_0_5"),
                    f"{right_name}_success": row[right_name].get("success_iou_0_5"),
                    f"{left_name}_failure_label": row[left_name].get("failure_label"),
                    f"{right_name}_failure_label": row[right_name].get("failure_label"),
                    f"{left_name}_teacher_recall": row[left_name].get("teacher_recall_at_58"),
                    f"{right_name}_teacher_recall": row[right_name].get("teacher_recall_at_58"),
                    f"{left_name}_gt_coverage": row[left_name].get("gt_box_patch_coverage_at_58"),
                    f"{right_name}_gt_coverage": row[right_name].get("gt_box_patch_coverage_at_58"),
                }
            )


def write_spatial_summary(path: Path, rows: list[dict[str, Any]], left_name: str, right_name: str) -> None:
    fieldnames = [
        "sample_id",
        "query",
        "selection_jaccard",
        f"{left_name}_row_mean",
        f"{right_name}_row_mean",
        "row_mean_delta",
        f"{left_name}_col_mean",
        f"{right_name}_col_mean",
        "col_mean_delta",
        f"{left_name}_occupied_bins",
        f"{right_name}_occupied_bins",
        f"{left_name}_hist_3x3",
        f"{right_name}_hist_3x3",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            left_spatial = row[left_name]["spatial"]
            right_spatial = row[right_name]["spatial"]
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "query": row.get("query"),
                    "selection_jaccard": row.get("selection_jaccard"),
                    f"{left_name}_row_mean": left_spatial.get("row_mean"),
                    f"{right_name}_row_mean": right_spatial.get("row_mean"),
                    "row_mean_delta": numeric_delta(right_spatial.get("row_mean"), left_spatial.get("row_mean")),
                    f"{left_name}_col_mean": left_spatial.get("col_mean"),
                    f"{right_name}_col_mean": right_spatial.get("col_mean"),
                    "col_mean_delta": numeric_delta(right_spatial.get("col_mean"), left_spatial.get("col_mean")),
                    f"{left_name}_occupied_bins": left_spatial.get("occupied_3x3_bins"),
                    f"{right_name}_occupied_bins": right_spatial.get("occupied_3x3_bins"),
                    f"{left_name}_hist_3x3": json.dumps(left_spatial.get("hist_3x3")),
                    f"{right_name}_hist_3x3": json.dumps(right_spatial.get("hist_3x3")),
                }
            )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    teacher_rows = read_jsonl(resolve_file(args.teacher_cache, "examples.jsonl"))
    left_selector_rows = read_jsonl(resolve_file(args.left_selector_eval, "per_sample_selector_results.jsonl"))
    right_selector_rows = read_jsonl(resolve_file(args.right_selector_eval, "per_sample_selector_results.jsonl"))
    left_generation_rows = read_jsonl(resolve_file(args.left_generation, "per_sample_results.jsonl"))
    right_generation_rows = read_jsonl(resolve_file(args.right_generation, "per_sample_results.jsonl"))

    teacher_by_id = index_by_sample(teacher_rows, "teacher cache")
    left_selector_by_id = index_by_sample(left_selector_rows, f"{args.left_name} selector")
    right_selector_by_id = index_by_sample(right_selector_rows, f"{args.right_name} selector")
    left_generation_by_id = index_by_sample(left_generation_rows, f"{args.left_name} generation")
    right_generation_by_id = index_by_sample(right_generation_rows, f"{args.right_name} generation")

    ordered_ids = [str(row["sample_id"]) for row in left_generation_rows if row.get("sample_id")]
    common_ids: list[str] = []
    missing_counts = Counter()
    for sample_id in ordered_ids:
        missing = []
        for label, index in [
            ("teacher", teacher_by_id),
            ("left_selector", left_selector_by_id),
            ("right_selector", right_selector_by_id),
            ("left_generation", left_generation_by_id),
            ("right_generation", right_generation_by_id),
        ]:
            if sample_id not in index:
                missing.append(label)
                missing_counts[label] += 1
        if not missing:
            common_ids.append(sample_id)
        if args.limit is not None and len(common_ids) >= args.limit:
            break

    rows: list[dict[str, Any]] = []
    for order_index, sample_id in enumerate(common_ids):
        teacher = teacher_by_id[sample_id]
        left_selector = left_selector_by_id[sample_id]
        right_selector = right_selector_by_id[sample_id]
        left_generation = left_generation_by_id[sample_id]
        right_generation = right_generation_by_id[sample_id]
        teacher_selected = clean_indices(teacher.get("selected_indices"))
        left_payload = side_payload(
            left_selector,
            left_generation,
            teacher,
            grid_size=args.grid_size,
            collapse_bin_threshold=args.collapse_bin_threshold,
            teacher_recall_threshold=args.teacher_recall_threshold,
            role_recall_threshold=args.role_recall_threshold,
            gt_coverage_threshold=args.gt_coverage_threshold,
        )
        right_payload = side_payload(
            right_selector,
            right_generation,
            teacher,
            grid_size=args.grid_size,
            collapse_bin_threshold=args.collapse_bin_threshold,
            teacher_recall_threshold=args.teacher_recall_threshold,
            role_recall_threshold=args.role_recall_threshold,
            gt_coverage_threshold=args.gt_coverage_threshold,
        )
        row = {
            "order_index": order_index,
            "sample_id": sample_id,
            "image_id": teacher.get("image_id", left_generation.get("image_id")),
            "image_path": teacher.get("image_path", left_generation.get("image_path")),
            "query": teacher.get("query", left_generation.get("query")),
            "gt_box": teacher.get("gt_box", left_generation.get("gt_box")),
            "width": teacher.get("width"),
            "height": teacher.get("height"),
            "teacher_selected_checksum": teacher.get("selected_checksum") or checksum_indices(teacher_selected),
            "teacher_selected_indices": teacher_selected,
            "teacher_role_counts": {key: len(value) for key, value in (teacher.get("role_labels") or {}).items()},
            args.left_name: left_payload,
            args.right_name: right_payload,
            "selection_jaccard": jaccard(left_payload["selected_indices"], right_payload["selected_indices"]),
        }
        row["flip_type"] = flip_type(left_payload, right_payload)
        row["right_minus_left"] = {
            "iou": numeric_delta(right_payload.get("iou"), left_payload.get("iou")),
            "teacher_recall_at_58": numeric_delta(right_payload.get("teacher_recall_at_58"), left_payload.get("teacher_recall_at_58")),
            "target_recall": numeric_delta(right_payload.get("target_recall"), left_payload.get("target_recall")),
            "context_recall": numeric_delta(right_payload.get("context_recall"), left_payload.get("context_recall")),
            "gt_box_patch_coverage_at_58": numeric_delta(
                right_payload.get("gt_box_patch_coverage_at_58"),
                left_payload.get("gt_box_patch_coverage_at_58"),
            ),
            "occupied_3x3_bins": numeric_delta(right_payload.get("occupied_3x3_bins"), left_payload.get("occupied_3x3_bins")),
        }
        rows.append(row)

    summary = build_summary(rows, args.left_name, args.right_name)
    summary["inputs"] = {
        "teacher_cache": str(args.teacher_cache),
        "left_selector_eval": str(args.left_selector_eval),
        "left_generation": str(args.left_generation),
        "right_selector_eval": str(args.right_selector_eval),
        "right_generation": str(args.right_generation),
        "limit": args.limit,
    }
    summary["join"] = {
        "left_generation_rows": len(left_generation_rows),
        "right_generation_rows": len(right_generation_rows),
        "teacher_rows": len(teacher_rows),
        "left_selector_rows": len(left_selector_rows),
        "right_selector_rows": len(right_selector_rows),
        "missing_counts_from_left_generation_order": dict(missing_counts),
    }
    summary["reserved_eval_note"] = (
        "This script only compares the supplied artifacts. For this goal, inputs must be allowed-dev artifacts, not reserved rows 0-499."
    )

    write_jsonl(args.output_dir / "comparison_per_sample.jsonl", rows)
    write_json(args.output_dir / "comparison_summary.json", summary)
    write_failure_flips(args.output_dir / "failure_flips.csv", rows, args.left_name, args.right_name, args.large_iou_delta)
    write_spatial_summary(args.output_dir / "spatial_distribution_summary.csv", rows, args.left_name, args.right_name)
    write_json(
        args.output_dir / "config.json",
        {
            "teacher_cache": str(args.teacher_cache),
            "left_name": args.left_name,
            "left_selector_eval": str(args.left_selector_eval),
            "left_generation": str(args.left_generation),
            "right_name": args.right_name,
            "right_selector_eval": str(args.right_selector_eval),
            "right_generation": str(args.right_generation),
            "limit": args.limit,
            "grid_size": args.grid_size,
            "collapse_bin_threshold": args.collapse_bin_threshold,
            "teacher_recall_threshold": args.teacher_recall_threshold,
            "role_recall_threshold": args.role_recall_threshold,
            "gt_coverage_threshold": args.gt_coverage_threshold,
        },
    )
    print(f"Compared {len(rows)} matched samples.")
    print(f"Saved comparison artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
