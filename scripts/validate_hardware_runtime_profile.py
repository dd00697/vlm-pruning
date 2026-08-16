#!/usr/bin/env python
"""Independently validate and finalize one bounded hardware profile run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for entry in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from run_hardware_runtime_profile import (  # noqa: E402
    CELL_IDS,
    K12_SOURCE,
    K12_SOURCE_SHA256,
    M448_SOURCE,
    M448_SOURCE_SHA256,
    MODES,
    compact_stage_traces,
    load_historical,
    stage_semantics_match,
    utc_now,
)
from run_official_baseline import load_manifest  # noqa: E402
from vlm_pruning.artifacts import sha256_file, source_provenance  # noqa: E402


SOURCE_PATHS = (
    Path("scripts/run_hardware_runtime_profile.py"),
    Path("scripts/validate_hardware_runtime_profile.py"),
    Path("scripts/run_progressive_oracle_refcoco.py"),
    Path("scripts/run_adaptive_matched_timing_trial.py"),
    Path("scripts/run_baseline.py"),
    Path("scripts/run_official_baseline.py"),
    Path("vlm_pruning/artifacts.py"),
    Path("vlm_pruning/features/query_tokens.py"),
    Path("vlm_pruning/llava_hooks/mid_layer_pruning.py"),
    Path("vlm_pruning/llava_hooks/progressive_pruning.py"),
    Path("vlm_pruning/llava_hooks/runtime_profile.py"),
)


class HardwareProfileValidationError(RuntimeError):
    """Raised when profile evidence is incomplete or semantically different."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HardwareProfileValidationError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            require(bool(line.strip()), f"Blank profile row {line_number}")
            value = json.loads(line)
            require(isinstance(value, dict), f"Non-object profile row {line_number}")
            rows.append(value)
    return rows


def close(actual: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    require(abs(float(actual) - float(expected)) <= tolerance, f"{label}: {actual} != {expected}")


def required_components(cell_id: str) -> tuple[set[str], set[str]]:
    common = {
        "cache_construction",
        "decoder_segment",
        "index_position_mask_gather",
        "prompt_kv_snapshot",
        "final_kv_snapshot",
        "first_generated_token",
        "later_decode_step",
    }
    if cell_id == "P_K12":
        return common | {"score_construction", "ranking_and_index_materialization"}, set()
    progressive = common | {
        "merge_assignment",
        "merge_weighted_reduction",
        "segment_cache_transition",
        "cache_proxy_dispatch",
    }
    if cell_id == "P_M448_REPLAY":
        return progressive | {"replay_index_lookup"}, {"score_construction", "ranking_and_index_materialization"}
    return progressive | {"score_construction", "ranking_and_index_materialization"}, {"replay_index_lookup"}


def validate_profile_components(row: dict[str, Any], cell_id: str, row_index: int) -> None:
    profile = row.get("runtime_profile") or {}
    require(profile.get("enabled") is True, f"row {row_index} profiler disabled")
    components = set(profile.get("components") or {})
    required, forbidden = required_components(cell_id)
    require(required.issubset(components), f"row {row_index} missing components: {sorted(required - components)}")
    require(not (forbidden & components), f"row {row_index} has forbidden components: {sorted(forbidden & components)}")
    records = profile.get("records") or []
    require(records, f"row {row_index} has no raw profiler records")
    require(all(float(record["cpu_wall_s"]) >= 0.0 for record in records), f"row {row_index} has negative CPU timing")
    require(all(record["cuda_event_s"] is None or float(record["cuda_event_s"]) >= 0.0 for record in records), f"row {row_index} has negative CUDA timing")
    segments = {
        (
            str(record["metadata"]["phase"]),
            int(record["metadata"]["start_layer"]),
            int(record["metadata"]["end_layer"]),
        )
        for record in records
        if record["component"] == "decoder_segment"
    }
    expected_ranges = {(0, 12), (12, 32)} if cell_id == "P_K12" else {(0, 4), (4, 12), (12, 32)}
    require({(start, end) for phase, start, end in segments if phase == "prefill"} == expected_ranges, f"row {row_index} prefill decoder ranges differ")
    require({(start, end) for phase, start, end in segments if phase == "decode"} == expected_ranges, f"row {row_index} decode decoder ranges differ")


def validate_semantics(
    row: dict[str, Any],
    historical: dict[str, Any],
    *,
    cell_id: str,
    row_index: int,
) -> None:
    require(row.get("raw_model_output") == historical.get("raw_model_output"), f"row {row_index} raw output drift")
    require(int(row.get("generated_token_count")) == int(historical.get("generated_token_count")), f"row {row_index} generated length drift")
    require(row.get("predicted_box") == historical.get("predicted_box"), f"row {row_index} box drift")
    close(float(row["box_iou"]), float(historical["box_iou"]), f"row {row_index} IoU")
    if cell_id == "P_K12":
        require(row.get("retained_indices") == historical.get("retained_indices"), f"row {row_index} K12 indices drift")
        cache = row.get("cache_trace") or {}
        for key in (
            "mid_layer/cache_generated_tokens",
            "mid_layer/early_cache_seq_len",
            "mid_layer/late_cache_seq_len",
            "mid_layer/early_attention_mask_len",
            "mid_layer/late_attention_mask_len",
            "cache/prompt_early_bank_bytes",
            "cache/prompt_late_bank_bytes",
            "cache/prompt_total_bytes",
            "cache/final_early_bank_bytes",
            "cache/final_late_bank_bytes",
            "cache/final_total_bytes",
        ):
            require(cache.get(key) == historical.get(key), f"row {row_index} K12 cache drift: {key}")
    else:
        require(row.get("retained_indices") == historical.get("final_retained_indices"), f"row {row_index} M448 indices drift")
        require(
            stage_semantics_match(
                list(row.get("stage_traces") or []),
                compact_stage_traces(historical.get("stage_traces") or []),
                replay=cell_id == "P_M448_REPLAY",
            ),
            f"row {row_index} M448 stage drift",
        )
        cache = row.get("cache_trace") or {}
        for key in (
            "progressive/cache_generated_tokens",
            "progressive/segment_prompt_lengths",
            "progressive/segment_cache_lengths",
            "progressive/per_layer_cache_lengths",
            "progressive/segment_attention_mask_lengths",
            "cache/prompt_segment_bytes",
            "cache/prompt_total_bytes",
            "cache/final_segment_bytes",
            "cache/final_total_bytes",
        ):
            require(cache.get(key) == historical.get(key), f"row {row_index} M448 cache drift: {key}")
    require(int(row["prompt_kv_total_bytes"]) < int(row["final_kv_total_bytes"]), f"row {row_index} prompt KV does not precede generated growth")
    require(row.get("semantic_match") is True, f"row {row_index} runner semantic gate failed")
    require(all((row.get("semantic_comparison") or {}).values()), f"row {row_index} semantic subcheck failed")


def recompute_metrics(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    require(metrics.get("rows") == len(rows), "Metrics row count differs")
    require(metrics.get("unique_samples") == len({row["sample_id"] for row in rows}), "Metrics unique sample count differs")
    require(metrics.get("all_semantic_matches") is True and metrics.get("semantic_mismatch_rows") == 0, "Metrics semantic gate differs")
    for field, recorded in (metrics.get("means") or {}).items():
        mean = sum(float(row[field]) for row in rows) / len(rows)
        close(mean, float(recorded), f"metrics mean {field}")
    components: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for name, summary in row["runtime_profile"]["components"].items():
            components.setdefault(name, []).append(summary)
    for name, recorded in (metrics.get("components") or {}).items():
        values = components[name]
        cpu_mean = sum(float(value["cpu_wall_total_s"]) for value in values) / len(values)
        close(cpu_mean, float(recorded["cpu_wall_mean_per_inference_s"]), f"component CPU {name}")
        cuda = [float(value["cuda_event_total_s"]) for value in values if value["cuda_event_total_s"] is not None]
        if cuda:
            close(sum(cuda) / len(cuda), float(recorded["cuda_event_mean_per_inference_s"]), f"component CUDA {name}")
        require(sum(int(value["count"]) for value in values) == int(recorded["event_count_total"]), f"component count {name}")
    return {"means": metrics["means"], "components": metrics["components"]}


def validate_run(run_dir: Path, manifest_path: Path) -> dict[str, Any]:
    config = read_json(run_dir / "config.json")
    metrics = read_json(run_dir / "metrics.json")
    rows = read_jsonl(run_dir / "profiles.jsonl")
    cell_id = str(config.get("cell_id"))
    mode = str(config.get("mode"))
    require(cell_id in CELL_IDS and mode in MODES, "Unknown profile cell or mode")
    contract = MODES[mode]
    expected_rows = int(contract["limit"]) * int(contract["repetitions"])
    require(len(rows) == expected_rows, f"Expected {expected_rows} profile rows, found {len(rows)}")
    manifest = load_manifest(manifest_path, None)
    require(len(manifest) == 1000, "Validation manifest must contain 1,000 rows")
    prefix = manifest[: int(contract["limit"])]
    prefix_ids = [str(row["sample_id"]) for row in prefix]
    require(
        hashlib.sha256("\n".join(prefix_ids).encode()).hexdigest() == config.get("manifest_prefix_order_sha256"),
        "Manifest prefix order hash mismatch",
    )
    expected_order = [
        (sample_id, repetition)
        for sample_id in prefix_ids
        for repetition in range(1, int(contract["repetitions"]) + 1)
    ]
    actual_order = [(str(row.get("sample_id")), int(row.get("repetition"))) for row in rows]
    require(actual_order == expected_order, "Profile row sample/repetition order differs")
    config_sha = sha256_file(run_dir / "config.json")
    require(all(row.get("config_sha256") == config_sha for row in rows), "Row config hash differs")
    source = source_provenance(REPO_ROOT, SOURCE_PATHS)
    require(source["git_commit"] == config.get("source_git_commit"), "Source commit differs")
    require(source["source_sha256"] == config.get("source_sha256"), "Source hash map differs")

    historical_path = K12_SOURCE if cell_id == "P_K12" else M448_SOURCE
    historical_hash = K12_SOURCE_SHA256 if cell_id == "P_K12" else M448_SOURCE_SHA256
    historical = load_historical(historical_path, historical_hash, set(prefix_ids))
    for row_index, row in enumerate(rows):
        require(row.get("cell_id") == cell_id, f"row {row_index} cell ID differs")
        require(int(row.get("manifest_index")) == prefix_ids.index(str(row["sample_id"])), f"row {row_index} manifest index differs")
        expected_trace = [576] * 12 + [48] * 20 if cell_id == "P_K12" else [576] * 4 + [448] * 8 + [48] * 20
        require(row.get("realized_visual_tokens_by_layer") == expected_trace, f"row {row_index} visual trace differs")
        close(float(row["average_visual_tokens"]), sum(expected_trace) / 32, f"row {row_index} average tokens")
        require(int(row["visual_token_layer_work"]) == sum(expected_trace), f"row {row_index} token work differs")
        validate_profile_components(row, cell_id, row_index)
        validate_semantics(row, historical[str(row["sample_id"])], cell_id=cell_id, row_index=row_index)
    for sample_id in prefix_ids:
        sample_rows = [row for row in rows if row["sample_id"] == sample_id]
        require(len({row["raw_model_output"] for row in sample_rows}) == 1, f"Repeated output drift: {sample_id}")
        require(len({row["retained_indices_checksum"] for row in sample_rows}) == 1, f"Repeated index drift: {sample_id}")
    recomputed = recompute_metrics(rows, metrics)
    return {
        "schema_version": 1,
        "status": "pass",
        "validated_at_utc": utc_now(),
        "cell_id": cell_id,
        "mode": mode,
        "rows": len(rows),
        "unique_samples": len(prefix_ids),
        "measured_repetitions_per_example": contract["repetitions"],
        "all_semantic_matches": True,
        "config_sha256": config_sha,
        "environment_sha256": sha256_file(run_dir / "environment.json"),
        "profiles_sha256": sha256_file(run_dir / "profiles.jsonl"),
        "metrics_sha256": sha256_file(run_dir / "metrics.json"),
        "source_git_commit": config["source_git_commit"],
        "recomputed": recomputed,
    }


def validate_and_finalize(run_dir: Path, manifest_path: Path) -> dict[str, Any]:
    integrity_path = run_dir / "integrity_report.json"
    completion_path = run_dir / "COMPLETED.json"
    require(not integrity_path.exists() and not completion_path.exists(), "Refusing to overwrite finalized profile")
    report = validate_run(run_dir, manifest_path)
    atomic_write(integrity_path, canonical_bytes(report))
    completion = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "cell_id": report["cell_id"],
        "mode": report["mode"],
        "rows": report["rows"],
        "unique_samples": report["unique_samples"],
        "config_sha256": report["config_sha256"],
        "environment_sha256": report["environment_sha256"],
        "profiles_sha256": report["profiles_sha256"],
        "metrics_sha256": report["metrics_sha256"],
        "integrity_sha256": sha256_file(integrity_path),
        "source_git_commit": report["source_git_commit"],
        "all_semantic_matches": True,
    }
    atomic_write(completion_path, canonical_bytes(completion))
    return {"integrity": report, "completion": completion, "metrics": read_json(run_dir / "metrics.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate_and_finalize(args.run_dir, args.manifest)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (HardwareProfileValidationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
