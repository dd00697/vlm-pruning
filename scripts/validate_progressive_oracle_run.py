#!/usr/bin/env python
"""Independently validate and finalize one progressive/oracle run artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.init_progressive_oracle_campaign import box_iou  # noqa: E402
from vlm_pruning.artifacts import sha256_file  # noqa: E402
from vlm_pruning.llava_hooks import (  # noqa: E402
    OracleIndexArtifact,
    PruningEvent,
    average_visual_tokens,
    deterministic_stage_seed,
    realized_visual_trace,
    validate_schedule,
)


EXPECTED_DISCOVERY_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
EXPECTED_DISCOVERY_ROWS = 1000
EXPECTED_FULL_VALIDATION_SHA256 = "f26d7b954189e86f55b50f9506bb670d3a0cf847d3152779af6c08dbf094e3d7"
EXPECTED_FULL_VALIDATION_ROWS = 8811
REGISTERED_PROMPT = (
    'Identify the target described by this referring expression: "{query}".\n'
    "Return exactly one bounding box in normalized 0-1 image coordinates as "
    "[x_min, y_min, x_max, y_max]. Do not include any other text."
)


class ProgressiveValidationError(RuntimeError):
    """Raised when a physical run artifact violates the registered contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_checksum(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def parse_box(text: str, width: int, height: int) -> tuple[list[float] | None, list[float] | None, str | None]:
    """Independent copy of the frozen runner's normalized-0-1 parser.

    The runner returns ``(raw, predicted, reason)``.  This validator keeps its
    historical ``(predicted, raw, reason)`` interface while reproducing the
    runner's candidate extraction, coordinate-order check and image clipping.
    """
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


def read_jsonl_strict(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ProgressiveValidationError(f"Missing physical JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ProgressiveValidationError(f"Blank line at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProgressiveValidationError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ProgressiveValidationError(f"Non-object JSON at {path}:{line_number}")
            rows.append(row)
    if len(rows) != expected_rows:
        raise ProgressiveValidationError(f"Expected {expected_rows} rows in {path}, found {len(rows)}")
    return rows


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ProgressiveValidationError(f"{label}: expected {expected!r}, found {actual!r}")


def _close(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise ProgressiveValidationError(f"{label}: expected {expected!r}, found {actual!r}")


def verify_source(config: dict[str, Any]) -> None:
    commit = str(config.get("source/git_commit") or "")
    if len(commit) != 40 or config.get("source/git_dirty") is not False:
        raise ProgressiveValidationError("Source revision is absent, abbreviated, or dirty")
    sources = config.get("source/sha256")
    if not isinstance(sources, dict) or not sources:
        raise ProgressiveValidationError("Source hashes are missing")
    for relative, expected in sources.items():
        try:
            payload = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=REPO_ROOT)
        except subprocess.CalledProcessError as exc:
            raise ProgressiveValidationError(f"Cannot retrieve source {relative} at {commit}") from exc
        _equal(hashlib.sha256(payload).hexdigest(), expected, f"source hash {relative}")


def config_events(config: dict[str, Any]) -> tuple[PruningEvent, ...]:
    raw = config.get("schedule")
    if not isinstance(raw, list):
        raise ProgressiveValidationError("Config schedule is not a list")
    return validate_schedule(raw)


def validate_config(config: dict[str, Any], manifest_path: Path, manifest_rows: list[dict[str, Any]]) -> None:
    checks = {
        "model/name": "llava-hf/llava-1.5-7b-hf",
        "model/precision": "bf16",
        "model/decoding": "deterministic_greedy",
        "model/max_new_tokens": 48,
        "dataset/name": "refcoco",
        "dataset/split": "val",
        "dataset/manifest_checksum": sha256_file(manifest_path),
        "dataset/num_samples": len(manifest_rows),
        "prompt/template": REGISTERED_PROMPT,
        "position_mode": "gap_preserving",
        "original_visual_tokens": 576,
        "grid": 24,
    }
    for key, expected in checks.items():
        _equal(config.get(key), expected, f"config {key}")
    implementation = config.get("runtime_implementation", "original")
    if implementation not in {"original", "fused_native", "feather_adapter"}:
        raise ProgressiveValidationError(
            f"Unsupported runtime implementation {implementation}"
        )
    if config.get("scope") == "discovery":
        _equal(len(manifest_rows), EXPECTED_DISCOVERY_ROWS, "discovery row count")
        _equal(sha256_file(manifest_path), EXPECTED_DISCOVERY_SHA256, "discovery manifest hash")
        if config.get("limit") is not None:
            raise ProgressiveValidationError("Discovery config unexpectedly has a limit")
        _equal(config.get("registered_scope", "auto"), "auto", "discovery registered scope")
    elif config.get("scope") == "full_validation":
        _equal(len(manifest_rows), EXPECTED_FULL_VALIDATION_ROWS, "full-validation row count")
        _equal(
            sha256_file(manifest_path),
            EXPECTED_FULL_VALIDATION_SHA256,
            "full-validation manifest hash",
        )
        if config.get("limit") is not None:
            raise ProgressiveValidationError("Full-validation config unexpectedly has a limit")
        _equal(config.get("registered_scope"), "full_validation", "full-validation registered scope")
        _equal(config.get("pre_llm_selector"), "none", "full-validation pre-LLM selector")
    elif config.get("scope") != "smoke":
        raise ProgressiveValidationError(f"Unsupported run scope {config.get('scope')}")
    if any(str(row.get("dataset") or "").lower() != "refcoco" or str(row.get("split") or "").lower() != "val" for row in manifest_rows):
        raise ProgressiveValidationError("Manifest contains a forbidden dataset or split")
    manifest_text = str(config.get("dataset/manifest") or "").lower()
    if "testa" in manifest_text or "testb" in manifest_text:
        raise ProgressiveValidationError("Forbidden RefCOCO test manifest in config")
    config_events(config)
    method = config.get("method/name", "progressive")
    if method == "feather_llava_main":
        feather_checks = {
            "runtime_implementation": "feather_adapter",
            "feather/official_repository": "https://github.com/markendo/FEATHER",
            "feather/official_commit": "c2a09b2765967601054c7b1fd513ac3e94ee4fc9",
            "feather/boundaries": [8, 16],
            "feather/fastv_ratio_upstream": 0.7,
            "feather/keep_ratio": 0.3,
            "feather/predefined_mask": "no_rope_s=3",
            "feather/adaptive_union": True,
            "schedule": [],
        }
        for key, expected in feather_checks.items():
            _equal(config.get(key), expected, f"config {key}")
    elif method != "progressive":
        raise ProgressiveValidationError(f"Unsupported method {method}")
    pre_selector = config.get("pre_llm_selector")
    pre_budget = config.get("pre_llm_budget")
    teacher = config.get("oracle_teacher_layer")
    if pre_selector == "none":
        if pre_budget is not None or teacher is not None:
            raise ProgressiveValidationError("Malformed no-pre-LLM config")
    elif pre_selector in {"oracle", "random", "uniform"}:
        if not isinstance(pre_budget, int) or not 1 <= pre_budget <= 576:
            raise ProgressiveValidationError("Malformed pre-LLM budget")
        if (pre_selector == "oracle") != (teacher in {12, 20}):
            raise ProgressiveValidationError("Malformed oracle teacher contract")
    else:
        raise ProgressiveValidationError(f"Unsupported pre-LLM selector {pre_selector}")
    verify_source(config)


def expected_visual_trace(config: dict[str, Any], row: dict[str, Any] | None = None) -> list[int]:
    if config.get("method/name") == "feather_llava_main":
        if row is None or not isinstance(row.get("stage_traces"), list) or len(row["stage_traces"]) != 2:
            raise ProgressiveValidationError("FEATHER row has no two-stage adaptive trace")
        first = int(row["stage_traces"][0].get("realized_budget", -1))
        second = int(row["stage_traces"][1].get("realized_budget", -1))
        return [576] * 8 + [first] * 8 + [second] * 16
    initial = int(config["pre_llm_budget"]) if config["pre_llm_selector"] != "none" else 576
    return realized_visual_trace(config_events(config), initial_visual_tokens=initial)


def expected_stage_specs(config: dict[str, Any], row: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if config.get("method/name") == "feather_llava_main":
        if row is None or not isinstance(row.get("stage_traces"), list) or len(row["stage_traces"]) != 2:
            raise ProgressiveValidationError("FEATHER row has no two adaptive stages")
        return [
            {
                "boundary": 8,
                "budget": int(row["stage_traces"][0].get("realized_budget", -1)),
                "selector": "feather_no_rope_topk_union_stride3",
            },
            {
                "boundary": 16,
                "budget": int(row["stage_traces"][1].get("realized_budget", -1)),
                "selector": "feather_no_rope_squared_keep_ratio",
            },
        ]
    specs: list[dict[str, Any]] = []
    if config["pre_llm_selector"] != "none":
        selector = config["pre_llm_selector"]
        if selector == "oracle":
            selector = f"oracle_k{int(config['oracle_teacher_layer'])}"
        specs.append({"boundary": 0, "budget": int(config["pre_llm_budget"]), "selector": selector})
    specs.extend(event.as_dict() for event in config_events(config))
    return specs


def validate_stage_traces(row: dict[str, Any], config: dict[str, Any], row_index: int) -> None:
    traces = row.get("stage_traces")
    if not isinstance(traces, list):
        raise ProgressiveValidationError(f"row {row_index} stage_traces missing")
    specs = expected_stage_specs(config, row)
    _equal(len(traces), len(specs), f"row {row_index} stage count")
    previous = list(range(576))
    original_visual_positions = row.get("progressive/original_visual_positions")
    original_text_positions = row.get("progressive/original_text_positions")
    if not isinstance(original_visual_positions, list) or len(original_visual_positions) != 576:
        raise ProgressiveValidationError(f"row {row_index} original visual positions malformed")
    if original_visual_positions != sorted(original_visual_positions) or len(set(original_visual_positions)) != 576:
        raise ProgressiveValidationError(f"row {row_index} original visual positions invalid")
    if not isinstance(original_text_positions, list) or original_text_positions != sorted(original_text_positions):
        raise ProgressiveValidationError(f"row {row_index} original text positions malformed")
    full_length = int(row.get("progressive/full_prompt_length", -1))
    _equal(sorted(original_visual_positions + original_text_positions), list(range(full_length)), f"row {row_index} original position partition")
    original_text_checksum = row.get("progressive/original_text_token_checksum")
    original_text_position_checksum = json_checksum(original_text_positions)
    _equal(row.get("progressive/original_text_position_checksum"), original_text_position_checksum, f"row {row_index} original text position checksum")
    previous_sequence_length = full_length
    previous_masses = {index: 1.0 for index in range(576)}
    for stage_index, (trace, spec) in enumerate(zip(traces, specs, strict=True)):
        label = f"row {row_index} stage {stage_index}"
        _equal(int(trace.get("stage_index", -1)), stage_index, f"{label} index")
        _equal(int(trace.get("boundary", -1)), spec["boundary"], f"{label} boundary")
        _equal(trace.get("selector"), spec["selector"], f"{label} selector")
        _equal(int(trace.get("requested_budget", -1)), spec["budget"], f"{label} requested budget")
        incoming = trace.get("incoming_original_indices")
        outgoing = trace.get("outgoing_original_indices")
        _equal(incoming, previous, f"{label} incoming subset")
        if not isinstance(outgoing, list) or outgoing != sorted(outgoing):
            raise ProgressiveValidationError(f"{label} outgoing indices malformed")
        if len(outgoing) != spec["budget"] or len(set(outgoing)) != len(outgoing):
            raise ProgressiveValidationError(f"{label} outgoing indices violate exact budget/uniqueness")
        if any(type(value) is not int or value not in set(previous) for value in outgoing):
            raise ProgressiveValidationError(f"{label} outgoing index is invalid or not a subset")
        _equal(int(trace.get("realized_budget", -1)), len(outgoing), f"{label} realized budget")
        _equal(trace.get("incoming_indices_checksum"), json_checksum(incoming), f"{label} incoming checksum")
        _equal(trace.get("outgoing_indices_checksum"), json_checksum(outgoing), f"{label} outgoing checksum")
        _equal(trace.get("text_token_checksum"), original_text_checksum, f"{label} text token preservation")
        _equal(trace.get("text_position_checksum"), original_text_position_checksum, f"{label} text position preservation")
        expected_positions = sorted(original_text_positions + [original_visual_positions[index] for index in outgoing])
        _equal(trace.get("outgoing_position_ids"), expected_positions, f"{label} gap-preserved positions")
        _equal(trace.get("position_id_checksum"), json_checksum(expected_positions), f"{label} position checksum")
        _equal(int(trace.get("incoming_sequence_length", -1)), previous_sequence_length, f"{label} incoming sequence")
        _equal(int(trace.get("outgoing_sequence_length", -1)), len(expected_positions), f"{label} outgoing sequence")
        if spec["selector"] in {
            "query_topk", "query_topk_merge", "k12_original",
            "feather_no_rope_topk_union_stride3", "feather_no_rope_squared_keep_ratio",
        }:
            _equal(trace.get("score_map_recomputed"), True, f"{label} score recomputation")
            if not isinstance(trace.get("score_map_checksum"), str) or len(trace["score_map_checksum"]) != 64:
                raise ProgressiveValidationError(f"{label} score checksum missing")
        elif spec["selector"] == "none":
            _equal(trace.get("score_map_recomputed"), False, f"{label} zero-tail scoring")
            _equal(trace.get("score_map_checksum"), None, f"{label} zero-tail score checksum")
        elif spec["selector"] == "random" and spec["boundary"] > 0:
            expected_seed = deterministic_stage_seed(
                sample_id=str(row["sample_id"]),
                cell_seed=int(config["seed"]),
                stage_index=stage_index,
                boundary=spec["boundary"],
                incoming_indices=incoming,
            )
            _equal(trace.get("derived_random_seed"), expected_seed, f"{label} random seed")
        if spec["selector"] == "query_topk_merge":
            _equal(bool(trace.get("merge_applied", False)), True, f"{label} merge applied")
            selected_slot = {original: slot for slot, original in enumerate(outgoing)}
            expected_masses = [float(previous_masses[original]) for original in outgoing]
            assignments: list[list[int]] = []
            for original in incoming:
                if original in selected_slot:
                    continue
                source_row, source_col = divmod(original, 24)
                retained = min(
                    outgoing,
                    key=lambda candidate: (
                        (source_row - candidate // 24) ** 2
                        + (source_col - candidate % 24) ** 2,
                        candidate,
                    ),
                )
                assignments.append([original, retained])
                expected_masses[selected_slot[retained]] += previous_masses[original]
            _equal(
                int(trace.get("merge_assignment_count", -1)),
                len(assignments),
                f"{label} merge assignment count",
            )
            _equal(
                trace.get("merge_assignment_checksum"),
                json_checksum(assignments),
                f"{label} merge assignment checksum",
            )
            _equal(trace.get("merge_output_masses"), expected_masses, f"{label} merge masses")
            _equal(
                trace.get("merge_mass_checksum"),
                json_checksum(expected_masses),
                f"{label} merge mass checksum",
            )
            _close(
                float(trace.get("merge_input_mass", -1)),
                sum(previous_masses.values()),
                f"{label} merge input mass",
            )
            _close(
                float(trace.get("merge_output_mass", -1)),
                sum(expected_masses),
                f"{label} merge output mass",
            )
            if any(mass <= 0 for mass in expected_masses):
                raise ProgressiveValidationError(f"{label} has a non-positive retained mass")
            if float(trace.get("merge_latency_s", -1)) < 0:
                raise ProgressiveValidationError(f"{label} merge latency is invalid")
            previous_masses = {
                original: expected_masses[slot] for slot, original in enumerate(outgoing)
            }
        else:
            _equal(bool(trace.get("merge_applied", False)), False, f"{label} unexpected merge")
            previous_masses = {original: previous_masses[original] for original in outgoing}
        previous = outgoing
        previous_sequence_length = len(expected_positions)
    _equal(row.get("final_retained_indices"), previous, f"row {row_index} final retained indices")
    _equal(row.get("final_retained_indices_checksum"), json_checksum(previous), f"row {row_index} final checksum")
    if config.get("method/name") == "feather_llava_main":
        metadata = row.get("feather/stage_metadata")
        if not isinstance(metadata, list) or len(metadata) != 2:
            raise ProgressiveValidationError(f"row {row_index} FEATHER metadata malformed")
        first, second = metadata
        _equal(int(first.get("stage_index", -1)), 0, f"row {row_index} FEATHER first stage")
        _equal(int(first.get("boundary", -1)), 8, f"row {row_index} FEATHER first boundary")
        _equal(int(first.get("incoming_tokens", -1)), 576, f"row {row_index} FEATHER first incoming")
        _equal(int(first.get("attention_topk_count", -1)), 172, f"row {row_index} FEATHER first top-k")
        _equal(int(first.get("uniform_stride_count", -1)), 64, f"row {row_index} FEATHER stride count")
        _equal(int(first.get("realized_tokens", -1)), len(traces[0]["outgoing_original_indices"]), f"row {row_index} FEATHER first realized")
        _equal(first.get("selected_local_indices"), traces[0]["outgoing_original_indices"], f"row {row_index} FEATHER first local order")
        first_count = len(traces[0]["outgoing_original_indices"])
        if not 172 <= first_count <= 236:
            raise ProgressiveValidationError(f"row {row_index} FEATHER union count outside exact bounds")
        _equal(int(second.get("stage_index", -1)), 1, f"row {row_index} FEATHER second stage")
        _equal(int(second.get("boundary", -1)), 16, f"row {row_index} FEATHER second boundary")
        _equal(int(second.get("incoming_tokens", -1)), first_count, f"row {row_index} FEATHER second incoming")
        _equal(int(second.get("attention_topk_count", -1)), int(first_count * 0.09), f"row {row_index} FEATHER squared top-k")
        _equal(int(second.get("uniform_stride_count", -1)), 0, f"row {row_index} FEATHER second uniform")
        second_local = second.get("selected_local_indices")
        if not isinstance(second_local, list):
            raise ProgressiveValidationError(f"row {row_index} FEATHER second local indices missing")
        _equal(
            [traces[0]["outgoing_original_indices"][int(local)] for local in second_local],
            traces[1]["outgoing_original_indices"],
            f"row {row_index} FEATHER second original mapping",
        )
        _equal(second.get("squared_ratio_at_second_stage"), True, f"row {row_index} FEATHER squared-ratio flag")
        for stage, trace_row in zip(metadata, traces, strict=True):
            _equal(stage.get("score_sha256"), trace_row.get("score_map_checksum"), f"row {row_index} FEATHER score hash")


def validate_cache_and_tokens(row: dict[str, Any], config: dict[str, Any], row_index: int) -> None:
    trace = expected_visual_trace(config, row)
    _equal(row.get("realized_visual_tokens_by_layer"), trace, f"row {row_index} visual trace")
    _equal(row.get("progressive/realized_visual_tokens_by_layer"), trace, f"row {row_index} diagnostic trace")
    expected_average = average_visual_tokens(trace)
    _close(float(row.get("candidate_average_visual_tokens", -1)), expected_average, f"row {row_index} candidate average")
    generated_cached = int(row.get("progressive/cache_generated_tokens", -1))
    _equal(generated_cached, int(row.get("generated_token_count", -1)) - 1, f"row {row_index} generated cache count")
    per_layer = row.get("progressive/per_layer_cache_lengths")
    if not isinstance(per_layer, dict) or set(per_layer) != {str(index) for index in range(32)}:
        raise ProgressiveValidationError(f"row {row_index} per-layer cache evidence is incomplete")
    text_count = len(row["progressive/original_text_positions"])
    for layer, visual_tokens in enumerate(trace):
        expected = text_count + int(visual_tokens) + generated_cached
        _equal(int(per_layer[str(layer)]), expected, f"row {row_index} layer {layer} cache")
    segment_cache = row.get("progressive/segment_cache_lengths")
    segment_masks = row.get("progressive/segment_attention_mask_lengths")
    if segment_cache != segment_masks:
        raise ProgressiveValidationError(f"row {row_index} segment cache/mask lengths differ")


def validate_oracle(
    row: dict[str, Any],
    config: dict[str, Any],
    artifact_by_id: dict[str, OracleIndexArtifact],
    row_index: int,
) -> None:
    sample_id = str(row["sample_id"])
    if config["pre_llm_selector"] == "oracle":
        artifact = artifact_by_id.get(sample_id)
        if artifact is None:
            raise ProgressiveValidationError(f"row {row_index} has no physical oracle index artifact")
        _equal(artifact.teacher_layer, config["oracle_teacher_layer"], f"row {row_index} oracle teacher")
        _equal(row.get("oracle_teacher_layer"), artifact.teacher_layer, f"row {row_index} reported teacher")
        _equal(row.get("oracle_ranking_checksum"), artifact.ranking_checksum, f"row {row_index} ranking checksum")
        _equal(
            int(row.get("oracle_teacher_visual_token_layer_work", -1)),
            artifact.teacher_visual_token_layer_work,
            f"row {row_index} teacher work",
        )
        _equal(row.get("oracle_privileged_non_deployable"), True, f"row {row_index} oracle label")
        first = row["stage_traces"][0]
        _equal(first["outgoing_original_indices"], artifact.selected_indices(int(config["pre_llm_budget"])), f"row {row_index} oracle selected indices")
    else:
        _equal(int(row.get("oracle_teacher_visual_token_layer_work", -1)), 0, f"row {row_index} non-oracle work")
        _equal(row.get("oracle_teacher_layer"), None, f"row {row_index} non-oracle teacher")
        _equal(row.get("oracle_ranking_checksum"), None, f"row {row_index} non-oracle ranking")
        _equal(row.get("oracle_privileged_non_deployable"), False, f"row {row_index} non-oracle label")


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "samples/evaluated": count,
        "quality/acc_at_0_5": sum(int(bool(row["success_iou_0_5"])) for row in rows) / count,
        "quality/mean_iou": sum(float(row["box_iou"]) for row in rows) / count,
        "quality/parsed_box_rate": sum(int(bool(row["valid"])) for row in rows) / count,
        "tokens/candidate_average_visual_tokens": sum(float(row["candidate_average_visual_tokens"]) for row in rows) / count,
        "tokens/oracle_teacher_visual_token_layer_work_mean": sum(
            int(row["oracle_teacher_visual_token_layer_work"]) for row in rows
        ) / count,
        "efficiency/peak_allocated_mb": max(
            (float(row["peak_allocated_mb"]) for row in rows if row.get("peak_allocated_mb") is not None),
            default=None,
        ),
        "efficiency/peak_reserved_mb": max(
            (float(row["peak_reserved_mb"]) for row in rows if row.get("peak_reserved_mb") is not None),
            default=None,
        ),
    }


def _load_oracle_artifacts(path: Path, expected_rows: int) -> dict[str, OracleIndexArtifact]:
    rows = read_jsonl_strict(path, expected_rows)
    artifacts: dict[str, OracleIndexArtifact] = {}
    for row in rows:
        artifact = OracleIndexArtifact.from_dict(row)
        if artifact.sample_id in artifacts:
            raise ProgressiveValidationError(f"Duplicate oracle index artifact {artifact.sample_id}")
        artifacts[artifact.sample_id] = artifact
    return artifacts


def validate_run(
    run_dir: Path,
    manifest_path: Path,
    *,
    expected_rows: int,
    finalize: bool = False,
) -> dict[str, Any]:
    del finalize
    config_path = run_dir / "config.json"
    predictions_path = run_dir / "predictions.jsonl"
    if not config_path.is_file():
        raise ProgressiveValidationError(f"Missing config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(manifest_path)
    if config.get("scope") == "smoke":
        if manifest_hash == EXPECTED_DISCOVERY_SHA256:
            manifest_physical_rows = EXPECTED_DISCOVERY_ROWS
        elif manifest_hash == EXPECTED_FULL_VALIDATION_SHA256:
            manifest_physical_rows = EXPECTED_FULL_VALIDATION_ROWS
        else:
            raise ProgressiveValidationError("Smoke manifest is not a registered RefCOCO-val manifest")
    else:
        manifest_physical_rows = expected_rows
    manifest_rows = read_jsonl_strict(manifest_path, manifest_physical_rows)
    if config.get("scope") == "smoke":
        _equal(config.get("limit"), expected_rows, "smoke limit")
        manifest_rows = manifest_rows[:expected_rows]
    validate_config(config, manifest_path, manifest_rows)
    predictions = read_jsonl_strict(predictions_path, expected_rows)
    manifest_ids = [str(row["sample_id"]) for row in manifest_rows]
    prediction_ids = [str(row.get("sample_id") or "") for row in predictions]
    _equal(prediction_ids, manifest_ids, "prediction manifest order")
    if len(set(prediction_ids)) != expected_rows:
        raise ProgressiveValidationError("Prediction IDs are not unique")
    config_sha = sha256_file(config_path)
    artifacts: dict[str, OracleIndexArtifact] = {}
    oracle_path = run_dir / "oracle_indices.jsonl"
    if config["pre_llm_selector"] == "oracle":
        artifacts = _load_oracle_artifacts(oracle_path, expected_rows)
        _equal(set(artifacts), set(manifest_ids), "oracle artifact IDs")
    elif oracle_path.exists():
        raise ProgressiveValidationError("Non-oracle run unexpectedly contains oracle artifacts")

    recomputed_rows: list[dict[str, Any]] = []
    for index, (manifest, row) in enumerate(zip(manifest_rows, predictions, strict=True)):
        for key in ("dataset", "split", "query", "image_id", "image_path", "gt_box"):
            _equal(row.get(key), manifest.get(key), f"row {index} {key}")
        _equal(row.get("config_sha256"), config_sha, f"row {index} config hash")
        _equal(row.get("source_git_commit"), config["source/git_commit"], f"row {index} source revision")
        validate_stage_traces(row, config, index)
        validate_cache_and_tokens(row, config, index)
        validate_oracle(row, config, artifacts, index)
        predicted_box, raw_box, parse_reason = parse_box(
            str(row.get("raw_model_output") or ""), int(manifest["width"]), int(manifest["height"])
        )
        iou = box_iou(predicted_box, list(manifest["gt_box"])) if predicted_box is not None else 0.0
        success = bool(iou >= 0.5)
        _equal(row.get("parse_failure_reason"), parse_reason, f"row {index} parse reason")
        _equal(bool(row.get("valid")), predicted_box is not None, f"row {index} validity")
        _equal(bool(row.get("invalid")), predicted_box is None, f"row {index} invalidity")
        _equal(bool(row.get("success_iou_0_5")), success, f"row {index} Acc@0.5")
        _close(float(row.get("box_iou", -1)), iou, f"row {index} IoU")
        if raw_box is not None:
            if not isinstance(row.get("raw_parsed_box"), list) or len(row["raw_parsed_box"]) != 4:
                raise ProgressiveValidationError(f"row {index} raw parsed box malformed")
            for actual, expected in zip(row["raw_parsed_box"], raw_box, strict=True):
                _close(float(actual), float(expected), f"row {index} raw box component")
        if predicted_box is not None:
            if not isinstance(row.get("predicted_box"), list) or len(row["predicted_box"]) != 4:
                raise ProgressiveValidationError(f"row {index} predicted box malformed")
            for actual, expected in zip(row["predicted_box"], predicted_box, strict=True):
                _close(float(actual), float(expected), f"row {index} predicted box component")
        recomputed_rows.append(row)
    metrics = aggregate_metrics(recomputed_rows)
    return {
        "status": "pass",
        "validated_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "rows": expected_rows,
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": config_sha,
        "predictions_sha256": sha256_file(predictions_path),
        "oracle_indices_sha256": sha256_file(oracle_path) if oracle_path.exists() else None,
        "source_git_commit": config["source/git_commit"],
        "metrics": metrics,
        "checks": {
            "manifest_order_unique": True,
            "config_source_protocol": True,
            "exact_stage_budgets_and_subsets": True,
            "text_and_gap_positions_preserved": True,
            "score_maps_recomputed_for_relevance_events_only": True,
            "zero_tail_has_no_scoring_or_ranking": True,
            "registered_merge_mass_and_assignment_contract": True,
            "per_layer_segment_caches": True,
            "raw_outputs_rescored": True,
            "oracle_indices_only": config["pre_llm_selector"] == "oracle",
            "oracle_teacher_work_separate": True,
        },
        "issues": [],
    }


def validate_and_finalize_run(run_dir: Path, manifest_path: Path, *, expected_rows: int) -> dict[str, Any]:
    report = validate_run(run_dir, manifest_path, expected_rows=expected_rows)
    metrics_path = run_dir / "metrics.json"
    integrity_path = run_dir / "integrity_report.json"
    completed_path = run_dir / "COMPLETED.json"
    atomic_write(metrics_path, canonical_json_bytes(report["metrics"]))
    atomic_write(integrity_path, canonical_json_bytes(report))
    completion = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "rows": expected_rows,
        "manifest_sha256": sha256_file(manifest_path),
        "source_git_commit": report["source_git_commit"],
        "config_sha256": sha256_file(run_dir / "config.json"),
        "predictions_sha256": sha256_file(run_dir / "predictions.jsonl"),
        "metrics_sha256": sha256_file(metrics_path),
        "integrity_sha256": sha256_file(integrity_path),
        "oracle_indices_sha256": report["oracle_indices_sha256"],
    }
    atomic_write(completed_path, canonical_json_bytes(completion))
    # Prove that the just-finalized physical artifacts remain valid; the
    # completion sentinel itself is metadata and is not an input to validation.
    validate_run(run_dir, manifest_path, expected_rows=expected_rows)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = (
            validate_and_finalize_run(args.run_dir, args.manifest, expected_rows=args.expected_rows)
            if args.finalize
            else validate_run(args.run_dir, args.manifest, expected_rows=args.expected_rows)
        )
    except (ProgressiveValidationError, OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
