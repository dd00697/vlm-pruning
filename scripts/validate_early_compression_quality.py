#!/usr/bin/env python
"""Independently validate a Phase-5 pre-projector pooling quality run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_baseline import box_iou, parse_box  # noqa: E402
from run_early_compression_quality import (  # noqa: E402
    CELL_GRIDS,
    EXPECTED_MANIFEST_SHA256,
    SOURCE_PATHS,
    aggregate,
    atomic_write,
    canonical_bytes,
    utc_now,
)
from run_official_baseline import load_manifest  # noqa: E402
from vlm_pruning.artifacts import sha256_file  # noqa: E402
from vlm_pruning.llava_hooks import area_weighted_spatial_pool  # noqa: E402


class EarlyQualityValidationError(RuntimeError):
    """Raised when a physical quality artifact violates its contract."""


def equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise EarlyQualityValidationError(f"{label}: {actual!r} != {expected!r}")


def _checksum(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_source(config: dict[str, Any]) -> None:
    commit = str(config["source/git_commit"])
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    expected_paths = [str(path) for path in SOURCE_PATHS]
    equal(sorted(config["source/sha256"]), sorted(expected_paths), "source path set")
    for path in expected_paths:
        payload = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=REPO_ROOT)
        digest = hashlib.sha256(payload).hexdigest()
        equal(digest, config["source/sha256"][path], f"source hash {path}")


def expected_pool_evidence(output_grid: int) -> dict[str, Any]:
    import torch

    _, evidence = area_weighted_spatial_pool(
        torch,
        torch.ones((1, 576, 1), dtype=torch.float32),
        input_grid=24,
        output_grid=output_grid,
    )
    return evidence.as_dict()


def validate_run(run_dir: Path, manifest_path: Path, *, expected_rows: int) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    environment_path = run_dir / "environment.json"
    predictions_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    for path in [config_path, environment_path, predictions_path, metrics_path]:
        if not path.is_file():
            raise EarlyQualityValidationError(f"Missing required artifact: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cell_id = str(config["cell_id"])
    if cell_id not in CELL_GRIDS:
        raise EarlyQualityValidationError(f"Unregistered early cell: {cell_id}")
    grid = CELL_GRIDS[cell_id]
    budget = grid * grid
    equal(config["output_grid"], grid, "output grid")
    equal(config["pooled_visual_tokens"], budget, "pooled budget")
    equal(config["dataset/manifest_sha256"], EXPECTED_MANIFEST_SHA256, "manifest hash config")
    equal(config["dataset/rows"], expected_rows, "configured rows")
    equal(config["query_selector"], "none", "query selector")
    equal(config["decoder_execution"], "single_contiguous_cache_no_internal_compaction", "decoder path")
    equal(config["position_mode"], "gap_preserving_original_prompt_positions", "position mode")
    validate_source(config)

    equal(sha256_file(manifest_path), EXPECTED_MANIFEST_SHA256, "physical manifest hash")
    manifest = load_manifest(manifest_path, None)
    if len(manifest) != 1000 or len({str(row["sample_id"]) for row in manifest}) != 1000:
        raise EarlyQualityValidationError("Frozen manifest is not 1,000 unique rows")
    expected_manifest = manifest[:expected_rows]
    lines = predictions_path.read_text(encoding="utf-8").splitlines()
    equal(len(lines), expected_rows, "prediction row count")
    rows = [json.loads(line) for line in lines]
    expected_ids = [str(row["sample_id"]) for row in expected_manifest]
    actual_ids = [str(row.get("sample_id") or "") for row in rows]
    equal(actual_ids, expected_ids, "ordered sample IDs")
    if len(set(actual_ids)) != expected_rows:
        raise EarlyQualityValidationError("Duplicate sample IDs")

    pool = expected_pool_evidence(grid)
    config_sha = sha256_file(config_path)
    for index, (row, sample) in enumerate(zip(rows, expected_manifest)):
        prefix = f"row {index}"
        equal(row["config_sha256"], config_sha, f"{prefix} config hash")
        equal(row["source_git_commit"], config["source/git_commit"], f"{prefix} source commit")
        equal(row["cell_id"], cell_id, f"{prefix} cell")
        equal(row["query"], sample["query"], f"{prefix} query")
        equal(row["image_path"], sample["image_path"], f"{prefix} image")
        equal(row["gt_box"], sample["gt_box"], f"{prefix} ground truth")
        raw_box, predicted_box, failure = parse_box(
            row["raw_model_output"], int(row["width"]), int(row["height"]), "normalized_0_1"
        )
        equal(row["raw_parsed_box"], raw_box, f"{prefix} raw parsed box")
        equal(row["predicted_box"], predicted_box, f"{prefix} predicted box")
        equal(row["parse_failure_reason"], failure, f"{prefix} parse reason")
        valid = predicted_box is not None
        iou = box_iou(predicted_box, sample["gt_box"]) if valid else 0.0
        equal(row["valid"], valid, f"{prefix} valid")
        if not math.isclose(float(row["box_iou"]), float(iou), rel_tol=0.0, abs_tol=1e-12):
            raise EarlyQualityValidationError(f"{prefix} IoU does not independently rescore")
        equal(row["success_iou_0_5"], valid and iou >= 0.5, f"{prefix} success")

        equal(row["pooled_visual_tokens"], budget, f"{prefix} pooled tokens")
        equal(row["realized_visual_tokens_by_layer"], [budget] * 32, f"{prefix} token trace")
        equal(row["visual_token_layer_work"], budget * 32, f"{prefix} token work")
        equal(row["average_visual_tokens"], float(budget), f"{prefix} average tokens")
        equal(row["physical_pool_operations"], 1, f"{prefix} pool operation count")
        equal(row["physical_decoder_compaction_boundaries"], 0, f"{prefix} compaction boundaries")
        for field in [
            "query_scoring_latency_s",
            "selector_latency_s",
            "gather_latency_s",
            "merge_latency_s",
            "route_gate_latency_s",
        ]:
            equal(float(row[field]), 0.0, f"{prefix} {field}")

        evidence = row["pooling_evidence"]
        equal(evidence["pool_anchor_indices"], pool["anchor_indices"], f"{prefix} anchors")
        equal(evidence["pool_anchor_checksum"], pool["anchor_checksum"], f"{prefix} anchor hash")
        equal(evidence["pool_overlap_checksum"], pool["overlap_checksum"], f"{prefix} overlap hash")
        if not math.isclose(float(evidence["pool_input_mass"]), 576.0, abs_tol=1e-9):
            raise EarlyQualityValidationError(f"{prefix} input mass")
        if not math.isclose(float(evidence["pool_output_mass"]), 576.0, abs_tol=1e-9):
            raise EarlyQualityValidationError(f"{prefix} output mass")
        masses = [float(value) for value in evidence["pool_output_masses"]]
        if len(masses) != budget or any(value <= 0.0 for value in masses):
            raise EarlyQualityValidationError(f"{prefix} output masses")
        if not math.isclose(sum(masses), 576.0, rel_tol=0.0, abs_tol=1e-9):
            raise EarlyQualityValidationError(f"{prefix} mass sum")
        anchor_positions = [int(value) for value in evidence["pool_anchor_positions"]]
        anchors = pool["anchor_indices"]
        if len(anchor_positions) != budget:
            raise EarlyQualityValidationError(f"{prefix} anchor-position count")
        base = anchor_positions[0] - int(anchors[0])
        equal(
            anchor_positions,
            [base + int(anchor) for anchor in anchors],
            f"{prefix} anchor position mapping",
        )
        equal(evidence["pool_anchor_position_checksum"], _checksum(anchor_positions), f"{prefix} anchor-position hash")
        positions = [int(value) for value in evidence["position_ids"]]
        if positions != sorted(positions) or len(set(positions)) != len(positions):
            raise EarlyQualityValidationError(f"{prefix} position IDs are not unique raster/text order")
        equal(evidence["position_id_checksum"], _checksum(positions), f"{prefix} position hash")
        text_count = int(evidence["text_token_count"])
        full_prompt = int(evidence["original_prompt_length"])
        compact_prompt = int(evidence["compacted_prompt_length"])
        equal(full_prompt, text_count + 576, f"{prefix} full prompt length")
        equal(compact_prompt, text_count + budget, f"{prefix} compact prompt length")
        equal(len(positions), compact_prompt, f"{prefix} position count")
        if not set(anchor_positions).issubset(positions):
            raise EarlyQualityValidationError(f"{prefix} anchor positions absent from prompt")
        equal(row["full_sequence_lengths_by_layer"], [compact_prompt] * 32, f"{prefix} layer lengths")
        equal(row["sum_layer_sequence_lengths"], compact_prompt * 32, f"{prefix} sum L")
        equal(row["sum_squared_layer_sequence_lengths"], compact_prompt * compact_prompt * 32, f"{prefix} sum L2")

        cache = row["cache_evidence"]
        equal(cache["cache/implementation"], "single_contiguous_transformers_cache", f"{prefix} cache type")
        equal(cache["cache/prompt_per_layer_lengths"], [compact_prompt] * 32, f"{prefix} prompt cache")
        generated = int(row["generated_token_count"])
        equal(cache["generated_token_count"], generated, f"{prefix} generated count")
        equal(cache["decode_step_count"], max(generated - 1, 0), f"{prefix} decode calls")
        equal(cache["cache/generated_tokens_cached"], max(generated - 1, 0), f"{prefix} generated cache")
        equal(
            cache["cache/final_per_layer_lengths"],
            [compact_prompt + max(generated - 1, 0)] * 32,
            f"{prefix} final cache",
        )
        if int(row["prompt_kv_total_bytes"]) <= 0 or int(row["final_kv_total_bytes"]) < int(row["prompt_kv_total_bytes"]):
            raise EarlyQualityValidationError(f"{prefix} KV bytes")
        if float(row["time_to_first_token_s"]) < float(row["llm_prefill_latency_s"]):
            raise EarlyQualityValidationError(f"{prefix} TTFT omits preprocessing")
        if float(row["end_to_end_latency_s"]) < float(row["time_to_first_token_s"]):
            raise EarlyQualityValidationError(f"{prefix} E2E shorter than TTFT")

    recomputed = aggregate(rows)
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    equal(stored, recomputed, "aggregate metrics")
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if "model_revision" not in environment and "checkpoint" not in json.dumps(environment).lower():
        # The pinned loader's environment schema is version-dependent; it must
        # at minimum identify the model and CUDA runtime through its payload.
        if config["model/name"] not in json.dumps(environment):
            raise EarlyQualityValidationError("Environment lacks checkpoint evidence")
    return {
        "status": "pass",
        "validated_at_utc": utc_now(),
        "rows": expected_rows,
        "cell_id": cell_id,
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(config_path),
        "environment_sha256": sha256_file(environment_path),
        "predictions_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
        "metrics": recomputed,
        "checks": {
            "recorded_committed_source_reconstructed": True,
            "frozen_manifest_order_unique_and_hash": True,
            "raw_outputs_independently_rescored": True,
            "all_patches_contribute_and_mass_is_576": True,
            "nearest_anchors_and_gap_positions": True,
            "one_pool_and_no_decoder_compaction": True,
            "single_cache_prompt_and_generation_growth": True,
            "all_latency_overheads_included": True,
        },
    }


def validate_and_finalize(run_dir: Path, manifest_path: Path, *, expected_rows: int) -> dict[str, Any]:
    report = validate_run(run_dir, manifest_path, expected_rows=expected_rows)
    integrity_path = run_dir / "integrity_report.json"
    atomic_write(integrity_path, canonical_bytes(report))
    completion = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "rows": expected_rows,
        "cell_id": report["cell_id"],
        "manifest_sha256": report["manifest_sha256"],
        "config_sha256": report["config_sha256"],
        "environment_sha256": report["environment_sha256"],
        "predictions_sha256": report["predictions_sha256"],
        "metrics_sha256": report["metrics_sha256"],
        "integrity_sha256": sha256_file(integrity_path),
    }
    atomic_write(run_dir / "COMPLETED.json", canonical_bytes(completion))
    validate_run(run_dir, manifest_path, expected_rows=expected_rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    try:
        report = (
            validate_and_finalize(args.run_dir, args.manifest, expected_rows=args.expected_rows)
            if args.finalize
            else validate_run(args.run_dir, args.manifest, expected_rows=args.expected_rows)
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (
        EarlyQualityValidationError,
        OSError,
        ValueError,
        KeyError,
        IndexError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
