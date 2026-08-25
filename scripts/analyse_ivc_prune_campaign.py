#!/usr/bin/env python3
"""Independently rescore and analyse the frozen IVC-Prune replacement campaign."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_baseline import box_iou, parse_box
from scripts.run_ivc_prune_campaign import plans
from vlm_pruning.artifacts import sha256_file
from vlm_pruning.final_statistics import (
    hierarchical_paired_bootstrap_interval,
)
from vlm_pruning.ivc_campaign import (
    METHODS,
    IVCCampaignError,
    align_rows,
    paired_bootstrap,
    summarize_quality,
    validate_visual_trace,
)

CAMPAIGN = REPO_ROOT / "experiments/ivc_prune_refcoco_val_20260825"
QUALITY_MANIFEST = REPO_ROOT / "data/manifests/final_evaluation/refcoco_val_unique.jsonl"
ANALYSIS = CAMPAIGN / "analysis"
TIMING_FIELDS = (
    "end_to_end_latency_s",
    "time_to_first_token_s",
    "vision_encoder_latency_s",
    "projector_latency_s",
    "query_scoring_latency_s",
    "selector_latency_s",
    "llm_prefill_with_overhead_latency_s",
    "decode_latency_s",
    "prompt_kv_total_bytes",
    "final_kv_total_bytes",
    "peak_allocated_gpu_bytes",
    "peak_reserved_gpu_bytes",
    "visual_token_layer_work",
    "average_visual_tokens",
    "final_visual_tokens",
)


def atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_jsonl(path: Path, rows: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise IVCCampaignError(f"Missing artifact: {path}")
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != rows or any(not isinstance(value, dict) for value in values):
        raise IVCCampaignError(f"Expected {rows} JSON objects: {path}")
    return values


def quality_analysis() -> dict[str, Any]:
    manifest = read_jsonl(QUALITY_MANIFEST, 8811)
    evidence: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    inventory = []
    for method in METHODS:
        path = CAMPAIGN / f"quality/{method.lower()}/predictions.jsonl"
        rows = read_jsonl(path, 8811)
        align_rows(rows, manifest, label=method)
        rescored = []
        for index, (row, source) in enumerate(zip(rows, manifest, strict=True)):
            raw_box, predicted, failure = parse_box(
                str(row.get("raw_model_output") or ""),
                int(source["width"]),
                int(source["height"]),
                "normalized_0_1",
            )
            iou = box_iou(predicted, source["gt_box"]) if predicted is not None else 0.0
            if (
                abs(float(row["box_iou"]) - float(iou)) > 1e-12
                or bool(row["success_iou_0_5"]) != bool(predicted is not None and iou >= 0.5)
                or bool(row["valid"]) != bool(predicted is not None)
            ):
                raise IVCCampaignError(f"Stored score differs from independent rescore: {method} row {index}")
            validate_visual_trace(method, row["realized_visual_tokens_by_layer"])
            rescored.append(
                {
                    **row,
                    "raw_parsed_box": raw_box,
                    "predicted_box": predicted,
                    "parse_failure_reason": failure,
                    "box_iou": float(iou),
                    "valid": predicted is not None,
                    "success_iou_0_5": bool(predicted is not None and iou >= 0.5),
                }
            )
        evidence[method] = rescored
        summaries[method] = summarize_quality(rescored)
        summaries[method]["retained_visual_token_histogram"] = {
            str(key): count
            for key, count in sorted(
                Counter(int(row["final_visual_tokens"]) for row in rescored).items()
            )
        }
        inventory.append({"method": method, "path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)})
    full, pruned = evidence["IVC_STACK_FULL"], evidence["IVC_PRUNE"]
    uncertainty = {
        "ivc_minus_stack_full_acc_at_0_5": paired_bootstrap(
            [float(row["success_iou_0_5"]) for row in pruned],
            [float(row["success_iou_0_5"]) for row in full],
        ),
        "ivc_minus_stack_full_mean_iou": paired_bootstrap(
            [float(row["box_iou"]) for row in pruned],
            [float(row["box_iou"]) for row in full],
        ),
    }
    result = {
        "schema_version": 1,
        "status": "QUALITY_ANALYSIS_COMPLETE",
        "summaries": summaries,
        "paired_uncertainty": uncertainty,
        "evidence": inventory,
        "interpretation_boundary": (
            "IVC is paired only with IVC_STACK_FULL. Its absolute validation result may be shown "
            "descriptively beside canonical methods if the software/checkpoint boundary is explicit."
        ),
    }
    atomic_json(ANALYSIS / "quality_analysis.json", result)
    return result


def timing_analysis(mode: str) -> dict[str, Any]:
    jobs = plans(f"timing-{mode}")
    repetitions = 5 if mode == "natural" else 3
    by_method: dict[str, list[list[dict[str, Any]]]] = {method: [] for method in METHODS}
    inventory = []
    for repetition in range(1, repetitions + 1):
        rep_rows: dict[str, list[dict[str, Any]]] = {}
        for job in [value for value in jobs if value["repetition"] == repetition]:
            path = job["run_dir"] / "predictions.jsonl"
            rows = read_jsonl(path, 1000)
            if any(str(row.get("method")) != job["method"] for row in rows):
                raise IVCCampaignError(f"Timing method label differs: {path}")
            if [row["sample_id"] for row in rows] != [row["sample_id"] for row in read_jsonl(jobs[0]["manifest"], 1000)]:
                raise IVCCampaignError(f"Timing row order differs: {path}")
            for row in rows:
                validate_visual_trace(job["method"], row["realized_visual_tokens_by_layer"])
                for field in TIMING_FIELDS:
                    value = float(row[field])
                    if not np.isfinite(value) or value < 0:
                        raise IVCCampaignError(f"Invalid {field} in {path}")
            rep_rows[job["method"]] = rows
            inventory.append({
                "method": job["method"],
                "repetition": repetition,
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
            })
        if set(rep_rows) != set(METHODS):
            raise IVCCampaignError(f"Incomplete paired timing repetition: {repetition}")
        for method in METHODS:
            by_method[method].append(rep_rows[method])
    summaries: dict[str, Any] = {}
    for method, repeated in by_method.items():
        summaries[method] = {}
        for field in TIMING_FIELDS:
            repetition_means = [statistics.mean(float(row[field]) for row in rows) for rows in repeated]
            summaries[method][field] = {
                "mean": float(statistics.mean(repetition_means)),
                "sd_across_repetition_means": float(statistics.stdev(repetition_means)),
                "repetition_means": repetition_means,
            }
    contrasts = {}
    for field in TIMING_FIELDS:
        left = [[float(row[field]) for row in repetition] for repetition in by_method["IVC_PRUNE"]]
        right = [[float(row[field]) for row in repetition] for repetition in by_method["IVC_STACK_FULL"]]
        contrasts[f"ivc_minus_stack_full_{field}"] = hierarchical_paired_bootstrap_interval(
            left, right, resamples=10_000, seed=20_260_825
        )
    full_e2e = summaries["IVC_STACK_FULL"]["end_to_end_latency_s"]["mean"]
    ivc_e2e = summaries["IVC_PRUNE"]["end_to_end_latency_s"]["mean"]
    relative = 100.0 * (ivc_e2e - full_e2e) / full_e2e
    result = {
        "schema_version": 1,
        "status": f"TIMING_{mode.upper()}_ANALYSIS_COMPLETE",
        "mode": mode,
        "summaries": summaries,
        "paired_contrasts": contrasts,
        "ivc_relative_e2e_change_percent": relative,
        "evidence": inventory,
    }
    atomic_json(ANALYSIS / f"timing_{mode}_analysis.json", result)
    if mode == "natural":
        contrast = contrasts["ivc_minus_stack_full_end_to_end_latency_s"]
        rep_delta = [
            left - right
            for left, right in zip(
                summaries["IVC_PRUNE"]["end_to_end_latency_s"]["repetition_means"],
                summaries["IVC_STACK_FULL"]["end_to_end_latency_s"]["repetition_means"],
                strict=True,
            )
        ]
        signs = {int(np.sign(value)) for value in rep_delta if value != 0}
        residual_flags = {}
        for method in METHODS:
            values = summaries[method]
            components = (
                values["vision_encoder_latency_s"]["mean"]
                + values["projector_latency_s"]["mean"]
                + values["llm_prefill_with_overhead_latency_s"]["mean"]
                + values["decode_latency_s"]["mean"]
            )
            e2e = values["end_to_end_latency_s"]["mean"]
            residual_flags[method] = abs(e2e - components) / e2e > 0.30
        required = (
            float(contrast["lower_95"]) <= 0 <= float(contrast["upper_95"])
            or len(signs) > 1
            or any(residual_flags.values())
        )
        trigger = {
            "required": required,
            "rule": (
                "fixed32 is required if the natural paired 95% CI crosses zero, repetition-mean "
                "directions disagree, or measured components differ from end-to-end by over 30%"
            ),
            "paired_interval": contrast,
            "repetition_deltas_s": rep_delta,
            "component_residual_over_30_percent": residual_flags,
        }
        atomic_json(ANALYSIS / "FIXED32_TRIGGER.json", trigger)
        result["fixed32_trigger"] = trigger
        atomic_json(ANALYSIS / f"timing_{mode}_analysis.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("quality", "timing-natural", "timing-fixed32", "all"), required=True)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        output = {}
        if args.phase in {"quality", "all"}:
            output["quality"] = quality_analysis()
        if args.phase in {"timing-natural", "all"}:
            output["timing_natural"] = timing_analysis("natural")
        if args.phase == "timing-fixed32":
            output["timing_fixed32"] = timing_analysis("fixed32")
        print(json.dumps({key: value["status"] for key, value in output.items()}, indent=2))
        return 0
    except (IVCCampaignError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
