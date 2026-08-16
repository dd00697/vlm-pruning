#!/usr/bin/env python
"""Run one bounded K12/M448 live-or-replay hardware profile cell."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for entry in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from run_adaptive_matched_timing_trial import MODEL  # noqa: E402
from run_baseline import apply_chat_template, box_iou, eos_token_ids, parse_box  # noqa: E402
from run_official_baseline import load_manifest, load_model, load_sample_images  # noqa: E402
from run_progressive_oracle_refcoco import (  # noqa: E402
    MAX_NEW_TOKENS,
    ORIGINAL_VISUAL_TOKENS,
    REGISTERED_PROMPT,
    json_checksum,
    prepare_full_inputs,
    query_resolution,
)
from vlm_pruning.artifacts import sha256_file, source_provenance  # noqa: E402
from vlm_pruning.llava_hooks import (  # noqa: E402
    ProfiledFusedGatherMergeForwarder,
    ProfiledFusedNativeForwarder,
    ProfiledMidLayerForwarder,
    ProfiledNativeCacheForwarder,
    ProfiledProgressiveForwarder,
    PruningEvent,
    RuntimeProfiler,
    load_stage_replay,
    ug_gap58_config_for_prune_layer,
)


CAMPAIGN_ROOT = REPO_ROOT / "experiments/hardware_aware_zero_tail_refcoco_val_20260816"
MANIFEST = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
M448_SOURCE = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815/runs/m448_attempt1/predictions.jsonl"
M448_SOURCE_SHA256 = "52e375657ba259a8785dcea3804770070e57f16f5ccc682b78f5edd167c9574b"
M448_TIMING_SOURCE = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815/matched_timing/natural/rep1_pos3_candidate_attempt1/timing.jsonl"
M448_TIMING_SOURCE_SHA256 = "ba171c2febf75af3246fad262686c74b2f7b07cfa5ebb5db4c113b7de6e1e34e"
K12_TIMING_SOURCE = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815/matched_timing/natural/rep1_pos2_k12_original_attempt1/timing.jsonl"
K12_TIMING_SOURCE_SHA256 = "df01a26cceeac1d52b833b85a6101ce71780415dfe997d57069f551c9ca78d6e"
M448_SCHEDULE = (
    PruningEvent(4, 448, "query_topk_merge"),
    PruningEvent(12, 48, "query_topk"),
)
CELL_IDS = ("P_K12", "P_M448", "P_M448_REPLAY")
REPAIR_CELL_IDS = ("R_FUSED_GATHER_MERGE", "R_NATIVE_CACHE", "R_FUSED_NATIVE")
ALL_CELL_IDS = CELL_IDS + REPAIR_CELL_IDS
MODES = {
    "smoke2": {"limit": 2, "warmup": 0, "repetitions": 1},
    "smoke20": {"limit": 20, "warmup": 2, "repetitions": 1},
    # Full profiling is split into isolated trials so method order can be
    # counterbalanced across the five repetitions without co-resident models.
    "profile100": {"limit": 100, "warmup": 20, "repetitions": 1},
}
PROFILE_REPETITIONS = (1, 2, 3, 4, 5)
PROFILE_ORDER_PLAN = {
    1: ("P_K12", "P_M448", "P_M448_REPLAY"),
    2: ("P_M448", "P_M448_REPLAY", "P_K12"),
    3: ("P_M448_REPLAY", "P_K12", "P_M448"),
    4: ("P_K12", "P_M448", "P_M448_REPLAY"),
    5: ("P_M448", "P_M448_REPLAY", "P_K12"),
}
REPAIR_ORDER_PLAN = {
    1: ("R_FUSED_GATHER_MERGE", "R_NATIVE_CACHE", "R_FUSED_NATIVE"),
    2: ("R_NATIVE_CACHE", "R_FUSED_NATIVE", "R_FUSED_GATHER_MERGE"),
    3: ("R_FUSED_NATIVE", "R_FUSED_GATHER_MERGE", "R_NATIVE_CACHE"),
    4: ("R_FUSED_GATHER_MERGE", "R_NATIVE_CACHE", "R_FUSED_NATIVE"),
    5: ("R_NATIVE_CACHE", "R_FUSED_NATIVE", "R_FUSED_GATHER_MERGE"),
}
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
    Path("vlm_pruning/llava_hooks/runtime_repairs.py"),
)


class HardwareProfileError(RuntimeError):
    """Raised when a bounded profile violates its frozen contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", choices=ALL_CELL_IDS, required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--attempt", type=int, default=1, choices=[1, 2])
    parser.add_argument("--repetition", type=int, choices=PROFILE_REPETITIONS)
    parser.add_argument("--order-position", type=int, choices=(1, 2, 3))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "profile100":
        if args.repetition is None or args.order_position is None:
            parser.error("profile100 requires --repetition and --order-position")
        order_plan = PROFILE_ORDER_PLAN if args.cell_id in CELL_IDS else REPAIR_ORDER_PLAN
        expected_position = order_plan[args.repetition].index(args.cell_id) + 1
        if args.order_position != expected_position:
            parser.error(
                f"counterbalanced plan requires position {expected_position} for "
                f"{args.cell_id} repetition {args.repetition}"
            )
    elif args.repetition is not None or args.order_position is not None:
        parser.error("--repetition/--order-position are reserved for profile100")
    return args


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def git_source() -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    if status:
        raise HardwareProfileError(f"Executable source must be committed and clean: {status}")
    return source_provenance(REPO_ROOT, SOURCE_PATHS)


def assert_storage_and_gpu_idle() -> dict[str, Any]:
    usage = shutil.disk_usage(REPO_ROOT)
    if usage.free < 100 * 1024**3:
        raise HardwareProfileError("Available storage is below the registered 100 GiB launch floor")
    compute = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    active = [line for line in compute.stdout.splitlines() if line.strip()]
    if compute.returncode or active:
        raise HardwareProfileError(f"Timing/profile launch requires an idle GPU: {active}")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    return {"available_bytes": usage.free, "available_gib": usage.free / 1024**3, "gpu": gpu, "active_compute_processes": active}


def output_dir(
    cell_id: str,
    mode: str,
    attempt: int,
    *,
    repetition: int | None = None,
    order_position: int | None = None,
) -> Path:
    if mode.startswith("smoke"):
        phase = "profiling" if cell_id in CELL_IDS else "runtime_repairs"
        return CAMPAIGN_ROOT / phase / "smokes" / f"{cell_id.lower()}_{mode}_attempt{attempt}"
    if repetition is None or order_position is None:
        raise HardwareProfileError("Full profile output requires repetition and order position")
    phase = "profiling" if cell_id in CELL_IDS else "runtime_repairs/profiling"
    return (
        CAMPAIGN_ROOT
        / phase
        / "trials"
        / f"rep{repetition}_pos{order_position}_{cell_id.lower()}_attempt{attempt}"
    )


def load_historical(path: Path, expected_hash: str, wanted: set[str]) -> dict[str, dict[str, Any]]:
    if sha256_file(path) != expected_hash:
        raise HardwareProfileError(f"Historical semantic source hash mismatch: {path}")
    output: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise HardwareProfileError(f"Blank historical row: {path}")
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or "")
            if sample_id in wanted:
                if sample_id in output:
                    raise HardwareProfileError(f"Duplicate historical sample: {sample_id}")
                output[sample_id] = row
    if set(output) != wanted:
        raise HardwareProfileError(f"Historical source misses {len(wanted - set(output))} profile samples")
    return output


def read_jsonl_prefix(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise HardwareProfileError(f"Blank partial row at line {line_number}: {path}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise HardwareProfileError(f"Non-object partial row at line {line_number}: {path}")
            rows.append(row)
    return rows


def validate_resume_prefix(
    rows: list[dict[str, Any]],
    *,
    expected_order: list[tuple[str, int]],
    config_sha256: str,
) -> None:
    if len(rows) > len(expected_order):
        raise HardwareProfileError("Partial profile contains more rows than the frozen contract")
    actual_order = [(str(row.get("sample_id")), int(row.get("repetition", -1))) for row in rows]
    if actual_order != expected_order[: len(rows)]:
        raise HardwareProfileError("Partial profile is not an exact ordered manifest prefix")
    if len(actual_order) != len(set(actual_order)):
        raise HardwareProfileError("Partial profile contains duplicate sample/repetition rows")
    for row_index, row in enumerate(rows):
        if row.get("config_sha256") != config_sha256:
            raise HardwareProfileError(f"Partial row {row_index} has a different config hash")
        if row.get("semantic_match") is not True:
            raise HardwareProfileError(f"Partial row {row_index} did not pass semantic parity")


def summarize_torch_profiler(profile: Any) -> dict[str, Any]:
    events = list(profile.events())
    kernel_events = [event for event in events if str(getattr(event, "device_type", "")).lower().endswith("cuda")]
    averages = list(profile.key_averages(group_by_input_shape=True))
    averages.sort(key=lambda event: float(getattr(event, "self_cuda_time_total", 0.0)), reverse=True)
    operators = []
    for event in averages[:100]:
        operators.append(
            {
                "name": str(getattr(event, "key", "")),
                "count": int(getattr(event, "count", 0)),
                "self_cpu_time_total_us": float(getattr(event, "self_cpu_time_total", 0.0)),
                "cpu_time_total_us": float(getattr(event, "cpu_time_total", 0.0)),
                "self_cuda_time_total_us": float(getattr(event, "self_cuda_time_total", 0.0)),
                "cuda_time_total_us": float(getattr(event, "cuda_time_total", 0.0)),
                "input_shapes": jsonable(getattr(event, "input_shapes", None)),
            }
        )
    return {
        "schema_version": 1,
        "capture_scope": "one_post_measurement_natural_inference",
        "event_count": len(events),
        "cuda_kernel_event_count": len(kernel_events),
        "operator_shape_group_count": len(averages),
        "operator_self_cpu_time_total_us": sum(
            float(getattr(event, "self_cpu_time_total", 0.0)) for event in averages
        ),
        "operator_self_cuda_time_total_us": sum(
            float(getattr(event, "self_cuda_time_total", 0.0)) for event in averages
        ),
        "operators_top100_by_self_cuda_time": operators,
    }


def compact_stage_traces(traces: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "stage_index",
        "boundary",
        "selector",
        "requested_budget",
        "realized_budget",
        "incoming_original_indices",
        "outgoing_original_indices",
        "incoming_indices_checksum",
        "outgoing_indices_checksum",
        "score_map_checksum",
        "score_map_recomputed",
        "incoming_sequence_length",
        "outgoing_sequence_length",
        "text_token_checksum",
        "text_position_checksum",
        "outgoing_position_ids",
        "position_id_checksum",
        "merge_applied",
        "merge_assignment_count",
        "merge_assignment_checksum",
        "merge_output_masses",
        "merge_mass_checksum",
        "merge_input_mass",
        "merge_output_mass",
    )
    return [{key: trace.get(key) for key in keys} for trace in traces]


def stage_semantics_match(actual: list[dict[str, Any]], historical: list[dict[str, Any]], *, replay: bool) -> bool:
    keys = (
        "boundary",
        "selector",
        "requested_budget",
        "realized_budget",
        "incoming_original_indices",
        "outgoing_original_indices",
        "incoming_indices_checksum",
        "outgoing_indices_checksum",
        "score_map_checksum",
        "incoming_sequence_length",
        "outgoing_sequence_length",
        "text_token_checksum",
        "text_position_checksum",
        "outgoing_position_ids",
        "position_id_checksum",
        "merge_applied",
        "merge_assignment_count",
        "merge_assignment_checksum",
        "merge_output_masses",
        "merge_mass_checksum",
        "merge_input_mass",
        "merge_output_mass",
    )
    if len(actual) != len(historical):
        return False
    for current, reference in zip(actual, historical, strict=True):
        if any(current.get(key) != reference.get(key) for key in keys):
            return False
        if replay:
            if current.get("score_map_recomputed") is not False:
                return False
        elif current.get("score_map_recomputed") != reference.get("score_map_recomputed"):
            return False
    return True


class ProfileTrial:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        replay: dict[str, Any] | None,
        historical: dict[str, dict[str, Any]],
        timing_historical: dict[str, dict[str, Any]],
    ) -> None:
        self.config = config
        self.historical = historical
        self.timing_historical = timing_historical
        self.torch, self.processor, self.model, self.device, self.model_environment = load_model(MODEL, "bf16")
        self.image_token_id = int(getattr(self.model.config, "image_token_index", 32000))
        profiler = RuntimeProfiler(self.torch, enabled=True)
        if config["cell_id"] == "P_K12":
            selector_config = ug_gap58_config_for_prune_layer(
                12,
                map_source="query_max",
                fusion_mode="single_layer",
                probe_layers=(12,),
                position_mode="gap_preserving",
                target_ratio=0.45,
                context_ratio=0.40,
                anchor_ratio=0.15,
            )
            self.forwarder: Any = ProfiledMidLayerForwarder(
                self.torch,
                self.model,
                image_token_id=self.image_token_id,
                grid=24,
                prune_after_layer=12,
                target_visual_tokens=48,
                selector_config=selector_config,
                diagnostics_mode="benchmark",
                synchronise_subtimings=False,
                runtime_profiler=profiler,
            )
        else:
            forwarder_class = {
                "P_M448": ProfiledProgressiveForwarder,
                "P_M448_REPLAY": ProfiledProgressiveForwarder,
                "R_FUSED_GATHER_MERGE": ProfiledFusedGatherMergeForwarder,
                "R_NATIVE_CACHE": ProfiledNativeCacheForwarder,
                "R_FUSED_NATIVE": ProfiledFusedNativeForwarder,
            }[config["cell_id"]]
            self.forwarder = forwarder_class(
                self.torch,
                self.model,
                image_token_id=self.image_token_id,
                schedule=M448_SCHEDULE,
                diagnostics_mode="benchmark",
                synchronise_subtimings=False,
                runtime_profiler=profiler,
                stage_replay_by_sample=replay,
            )

    def infer(self, sample: dict[str, Any], repetition: int) -> dict[str, Any]:
        torch = self.torch
        _, images = load_sample_images(sample)
        if len(images) != 1:
            raise HardwareProfileError("Hardware profiling requires exactly one image")
        prompt = REGISTERED_PROMPT.format(query=str(sample["query"]))
        text_prompt = apply_chat_template(self.processor, prompt, 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        wall_started = time.perf_counter()
        inputs, prep = prepare_full_inputs(torch, self.processor, self.model, self.device, images, prompt)
        full_prompt_length = int(inputs["inputs_embeds"].shape[1])
        text_token_count = full_prompt_length - ORIGINAL_VISUAL_TOKENS
        if text_token_count <= 0:
            raise HardwareProfileError("Expanded prompt does not contain a valid text-token complement")
        resolution = query_resolution(
            self.processor,
            text_prompt,
            str(sample["query"]),
            inputs["input_ids"],
            self.image_token_id,
        )
        self.forwarder.reset_runtime_profile()
        if self.config["cell_id"] == "P_K12":
            generation = self.forwarder.generate_with_layer_aware_cache(
                inputs_embeds=inputs["inputs_embeds"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                sample_id=str(sample["sample_id"]),
                original_tokens=ORIGINAL_VISUAL_TOKENS,
                seed=0,
                query_text=str(sample["query"]),
                query_token_status=resolution.resolver_status,
                query_token_positions=resolution.token_positions,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_ids=eos_token_ids(self.model, self.processor),
            )
            diagnostics = jsonable(generation.diagnostics)
            retained = [int(value) for value in generation.selection.selected_indices]
            stage_traces: list[dict[str, Any]] = []
            realized_trace = [576] * 12 + [48] * 20
        else:
            generation = self.forwarder.generate_with_segment_caches(
                inputs_embeds=inputs["inputs_embeds"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                sample_id=str(sample["sample_id"]),
                seed=0,
                query_original_positions=resolution.token_positions,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_ids=eos_token_ids(self.model, self.processor),
            )
            diagnostics = jsonable(generation.diagnostics)
            retained = [int(value) for value in generation.final_retained_indices]
            stage_traces = compact_stage_traces(generation.stage_traces)
            realized_trace = list(diagnostics["progressive/realized_visual_tokens_by_layer"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_latency = time.perf_counter() - wall_started
        raw_output = self.processor.decode(generation.generated_ids, skip_special_tokens=True).strip()
        raw_box, predicted_box, parse_failure = parse_box(
            raw_output,
            int(sample["width"]),
            int(sample["height"]),
            "normalized_0_1",
        )
        iou = box_iou(predicted_box, sample["gt_box"]) if predicted_box is not None else 0.0
        historical = self.historical[str(sample["sample_id"])]
        timing_historical = self.timing_historical[str(sample["sample_id"])]
        historical_raw_box, historical_predicted_box, _ = parse_box(
            str(historical["raw_model_output"]),
            int(sample["width"]),
            int(sample["height"]),
            "normalized_0_1",
        )
        historical_iou = (
            box_iou(historical_predicted_box, sample["gt_box"])
            if historical_predicted_box is not None
            else 0.0
        )
        del historical_raw_box
        if self.config["cell_id"] == "P_K12":
            cache_trace_match = (
                int(diagnostics["cache/prompt_total_bytes"]) == int(historical["prompt_kv_total_bytes"])
                and int(diagnostics["cache/final_total_bytes"]) == int(historical["final_kv_total_bytes"])
                and int(diagnostics["mid_layer/cache_generated_tokens"]) == int(historical["generated_token_count"]) - 1
            )
            stage_match = True
            retained_match = (
                len(retained) == 48
                and retained == sorted(retained)
                and len(set(retained)) == 48
                and min(retained) >= 0
                and max(retained) < ORIGINAL_VISUAL_TOKENS
            )
        else:
            cache_trace_match = (
                all(
                    diagnostics.get(key) == historical.get(key)
                    for key in (
                    "progressive/cache_generated_tokens",
                    "progressive/segment_prompt_lengths",
                    "progressive/segment_cache_lengths",
                    "progressive/per_layer_cache_lengths",
                    "progressive/segment_attention_mask_lengths",
                    )
                )
                and int(diagnostics["cache/prompt_total_bytes"]) == int(timing_historical["prompt_kv_total_bytes"])
                and int(diagnostics["cache/final_total_bytes"]) == int(timing_historical["final_kv_total_bytes"])
            )
            stage_match = stage_semantics_match(
                stage_traces,
                list(historical["stage_traces"]),
                replay=self.config["cell_id"] == "P_M448_REPLAY",
            )
            retained_match = retained == [int(value) for value in historical["final_retained_indices"]]
        semantic = {
            "raw_output_match": raw_output == str(historical["raw_model_output"]),
            "generated_token_count_match": int(generation.generated_ids.numel()) == int(historical["generated_token_count"]),
            "predicted_box_match": predicted_box == historical_predicted_box,
            "iou_match": abs(float(iou) - float(historical_iou)) <= 1e-12,
            "retained_indices_match": retained_match,
            "stage_semantics_match": stage_match,
            "cache_trace_match": cache_trace_match,
        }
        result = {
            "sample_id": str(sample["sample_id"]),
            "query": str(sample["query"]),
            "width": int(sample["width"]),
            "height": int(sample["height"]),
            "gt_box": list(sample["gt_box"]),
            "manifest_index": int(self.config["sample_index_by_id"][str(sample["sample_id"])]),
            "repetition": int(repetition),
            "cell_id": self.config["cell_id"],
            "raw_model_output": raw_output,
            "raw_parsed_box": raw_box,
            "predicted_box": predicted_box,
            "parse_failure_reason": parse_failure,
            "box_iou": float(iou),
            "success_iou_0_5": bool(predicted_box is not None and iou >= 0.5),
            "generated_token_count": int(generation.generated_ids.numel()),
            "retained_indices": retained,
            "retained_indices_checksum": json_checksum(retained),
            "stage_traces": stage_traces,
            "realized_visual_tokens_by_layer": realized_trace,
            "average_visual_tokens": sum(realized_trace) / 32,
            "visual_token_layer_work": sum(realized_trace),
            "prompt_text_token_count": text_token_count,
            "prompt_sequence_lengths_by_layer": [text_token_count + tokens for tokens in realized_trace],
            "sequence_layer_work": sum(text_token_count + tokens for tokens in realized_trace),
            "quadratic_sequence_layer_work": sum(
                (text_token_count + tokens) ** 2 for tokens in realized_trace
            ),
            "prompt_kv_total_bytes": int(diagnostics["cache/prompt_total_bytes"]),
            "final_kv_total_bytes": int(diagnostics["cache/final_total_bytes"]),
            "cache_trace": {
                key: value
                for key, value in diagnostics.items()
                if key.startswith("progressive/segment_")
                or key.startswith("progressive/per_layer_cache")
                or key == "progressive/cache_generated_tokens"
                or key.startswith("mid_layer/") and ("cache" in key or "attention_mask" in key)
                or key.startswith("cache/prompt_")
                or key.startswith("cache/final_")
            },
            "vision_encoder_latency_s": float(prep["vision_encoder_latency_s"]),
            "projector_latency_s": float(prep["projector_latency_s"]),
            "prefill_latency_s": float(diagnostics["llm_prefill_latency_s"]),
            "decode_latency_s": float(diagnostics["decode_latency_s"]),
            "decode_latency_per_generated_token_s": (
                float(diagnostics["decode_latency_s"])
                / max(int(generation.generated_ids.numel()) - 1, 1)
            ),
            "end_to_end_latency_s": float(wall_latency),
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
            "runtime_profile": self.forwarder.runtime_profile_snapshot(),
            "semantic_comparison": semantic,
            "semantic_match": all(semantic.values()),
            "historical_source_sha256": self.config["historical_source_sha256"],
            "historical_timing_source_sha256": self.config["historical_timing_source_sha256"],
        }
        for image in images:
            image.close()
        del inputs, prep, resolution, generation
        gc.collect()
        return result


def capture_torch_profiler(trial: ProfileTrial, sample: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    torch = trial.torch
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profile:
        row = trial.infer(sample, 0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if row.get("semantic_match") is not True:
        raise HardwareProfileError("PyTorch-profiler capture changed inference semantics")
    trace_path = run_dir / "torch_profiler_trace.json"
    profile.export_chrome_trace(str(trace_path))
    summary = summarize_torch_profiler(profile)
    summary.update(
        {
            "sample_id": row["sample_id"],
            "raw_model_output": row["raw_model_output"],
            "semantic_match": True,
            "synchronization_policy": row["runtime_profile"]["synchronization_policy"],
            "trace_sha256": sha256_file(trace_path),
        }
    )
    atomic_write(run_dir / "torch_profiler_summary.json", canonical_bytes(summary))
    return summary


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    components: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for name, summary in row["runtime_profile"]["components"].items():
            components.setdefault(name, []).append(summary)
    component_summary: dict[str, Any] = {}
    for name, values in sorted(components.items()):
        cpu = [float(value["cpu_wall_total_s"]) for value in values]
        cuda = [float(value["cuda_event_total_s"]) for value in values if value["cuda_event_total_s"] is not None]
        component_summary[name] = {
            "rows_present": len(values),
            "cpu_wall_mean_per_inference_s": sum(cpu) / len(cpu),
            "cuda_event_mean_per_inference_s": sum(cuda) / len(cuda) if cuda else None,
            "event_count_total": sum(int(value["count"]) for value in values),
        }
    scalar_fields = (
        "vision_encoder_latency_s",
        "projector_latency_s",
        "prefill_latency_s",
        "decode_latency_s",
        "end_to_end_latency_s",
        "prompt_kv_total_bytes",
        "final_kv_total_bytes",
        "average_visual_tokens",
        "visual_token_layer_work",
        "sequence_layer_work",
        "quadratic_sequence_layer_work",
        "generated_token_count",
        "decode_latency_per_generated_token_s",
    )
    return {
        "status": "measured_pending_independent_validation",
        "cell_id": rows[0]["cell_id"],
        "rows": len(rows),
        "unique_samples": len({row["sample_id"] for row in rows}),
        "repetitions": sorted({int(row["repetition"]) for row in rows}),
        "all_semantic_matches": all(bool(row["semantic_match"]) for row in rows),
        "semantic_mismatch_rows": sum(not bool(row["semantic_match"]) for row in rows),
        "means": {field: sum(float(row[field]) for row in rows) / len(rows) for field in scalar_fields},
        "components": component_summary,
        "peak_allocated_gpu_bytes": max(int(row["peak_allocated_gpu_bytes"]) for row in rows),
        "peak_reserved_gpu_bytes": max(int(row["peak_reserved_gpu_bytes"]) for row in rows),
    }


def run(args: argparse.Namespace) -> int:
    contract = MODES[args.mode]
    if sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise HardwareProfileError("Frozen validation manifest hash mismatch")
    manifest_rows = load_manifest(MANIFEST, None)
    if len(manifest_rows) != 1000 or len({str(row["sample_id"]) for row in manifest_rows}) != 1000:
        raise HardwareProfileError("Frozen validation manifest rows/IDs differ")
    rows = manifest_rows[: int(contract["limit"])]
    wanted = {str(row["sample_id"]) for row in rows}
    historical_path = K12_TIMING_SOURCE if args.cell_id == "P_K12" else M448_SOURCE
    historical_hash = K12_TIMING_SOURCE_SHA256 if args.cell_id == "P_K12" else M448_SOURCE_SHA256
    historical = load_historical(historical_path, historical_hash, wanted)
    timing_historical_path = K12_TIMING_SOURCE if args.cell_id == "P_K12" else M448_TIMING_SOURCE
    timing_historical_hash = K12_TIMING_SOURCE_SHA256 if args.cell_id == "P_K12" else M448_TIMING_SOURCE_SHA256
    timing_historical = load_historical(timing_historical_path, timing_historical_hash, wanted)
    replay = None
    if args.cell_id == "P_M448_REPLAY":
        replay = load_stage_replay(M448_SOURCE, expected_sha256=M448_SOURCE_SHA256, sample_ids=wanted)
    run_dir = output_dir(
        args.cell_id,
        args.mode,
        args.attempt,
        repetition=args.repetition,
        order_position=args.order_position,
    )
    if run_dir.exists() and not args.resume:
        raise HardwareProfileError(f"Profile run directory already exists: {run_dir}")
    if not run_dir.exists() and args.resume:
        raise HardwareProfileError(f"Cannot resume a missing profile run: {run_dir}")
    launch = assert_storage_and_gpu_idle()
    source = git_source()
    global_repetition = int(args.repetition or 1)
    semantic_command = {
        "program": "scripts/run_hardware_runtime_profile.py",
        "cell_id": args.cell_id,
        "mode": args.mode,
        "attempt": args.attempt,
        "global_repetition": global_repetition,
        "counterbalanced_order_position": args.order_position,
        **contract,
    }
    semantic_command_sha = hashlib.sha256(
        json.dumps(semantic_command, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if args.resume:
        for forbidden in ("profiles.jsonl", "metrics.json", "integrity_report.json", "COMPLETED.json"):
            if (run_dir / forbidden).exists():
                raise HardwareProfileError(f"Refusing to resume after finalized output exists: {forbidden}")
        with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        required_resume_values = {
            "cell_id": args.cell_id,
            "mode": args.mode,
            "attempt": args.attempt,
            "global_repetition": global_repetition,
            "counterbalanced_order_position": args.order_position,
            "manifest_sha256": MANIFEST_SHA256,
            "source_git_commit": source["git_commit"],
            "source_sha256": source["source_sha256"],
            "semantic_command_sha256": semantic_command_sha,
            "run_dir": relative(run_dir),
        }
        for key, expected in required_resume_values.items():
            if config.get(key) != expected:
                raise HardwareProfileError(f"Resume config differs for {key}")
        if not (run_dir / "environment.json").is_file():
            raise HardwareProfileError("Resume run lacks its immutable environment artifact")
    else:
        config = {
            "schema_version": 2,
            "created_at_utc": utc_now(),
            "scope": (
                "instrumentation_smoke"
                if args.mode.startswith("smoke") and args.cell_id in CELL_IDS
                else "runtime_repair_smoke"
                if args.mode.startswith("smoke")
                else "runtime_profile"
                if args.cell_id in CELL_IDS
                else "runtime_repair_profile"
            ),
            "cell_id": args.cell_id,
            "mode": args.mode,
            "attempt": args.attempt,
            "global_repetition": global_repetition,
            "counterbalanced_order_position": args.order_position,
            "limit": contract["limit"],
            "warmup_examples": contract["warmup"],
            "measured_repetitions_per_example": contract["repetitions"],
            "manifest_path": relative(MANIFEST),
            "manifest_sha256": MANIFEST_SHA256,
            "manifest_prefix_order_sha256": hashlib.sha256(
                "\n".join(str(row["sample_id"]) for row in rows).encode()
            ).hexdigest(),
            "sample_index_by_id": {
                str(row["sample_id"]): index for index, row in enumerate(rows)
            },
            "historical_source_path": relative(historical_path),
            "historical_source_sha256": historical_hash,
            "historical_timing_source_path": relative(timing_historical_path),
            "historical_timing_source_sha256": timing_historical_hash,
            "model": MODEL,
            "precision": "bf16",
            "decoding": "deterministic_greedy",
            "max_new_tokens": MAX_NEW_TOKENS,
            "schedule": (
                [event.as_dict() for event in M448_SCHEDULE]
                if args.cell_id != "P_K12"
                else [{"boundary": 12, "budget": 48, "selector": "original_k12_ug"}]
            ),
            "replay_score_and_rank_disabled_only": args.cell_id == "P_M448_REPLAY",
            "runtime_profiler": "synchronized_cpu_wall_and_cuda_event_v1",
            "torch_profiler_capture": args.mode == "profile100" and global_repetition == 1,
            "nsight_systems": {
                "available": shutil.which("nsys") is not None,
                "status": "skipped_not_installed" if shutil.which("nsys") is None else "available_noninteractive",
            },
            "source_git_commit": source["git_commit"],
            "source_sha256": source["source_sha256"],
            "semantic_command": semantic_command,
            "semantic_command_sha256": semantic_command_sha,
            "run_dir": relative(run_dir),
            "launch_storage_and_gpu": launch,
        }
        run_dir.mkdir(parents=True)
        atomic_write(run_dir / "config.json", canonical_bytes(config))
    config_sha = sha256_file(run_dir / "config.json")
    expected_order = [(str(sample["sample_id"]), global_repetition) for sample in rows]
    partial = run_dir / "profiles.partial.jsonl"
    measured = read_jsonl_prefix(partial) if args.resume else []
    validate_resume_prefix(measured, expected_order=expected_order, config_sha256=config_sha)
    model_load_started = time.perf_counter()
    trial = ProfileTrial(
        config,
        replay=replay,
        historical=historical,
        timing_historical=timing_historical,
    )
    model_load_cold_start_s = time.perf_counter() - model_load_started
    if not args.resume:
        environment = {
            "created_at_utc": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": trial.torch.__version__,
            "cuda": trial.torch.version.cuda,
            "model_environment": trial.model_environment,
            "model_load_cold_start_s": model_load_cold_start_s,
            "launch": launch,
        }
        atomic_write(run_dir / "environment.json", canonical_bytes(jsonable(environment)))
    for sample in manifest_rows[: int(contract["warmup"])]:
        trial.infer(sample, 0)
    if trial.torch.cuda.is_available():
        trial.torch.cuda.synchronize()
    with partial.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for row_index in range(len(measured), len(expected_order)):
            sample = rows[row_index]
            result = trial.infer(sample, global_repetition)
            result["config_sha256"] = config_sha
            result["source_git_commit"] = config["source_git_commit"]
            measured.append(result)
            handle.write(json.dumps(jsonable(result), sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(f"PROFILE {args.cell_id} {args.mode}: {row_index + 1}/{len(rows)}", flush=True)
    if config.get("torch_profiler_capture"):
        trace_path = run_dir / "torch_profiler_trace.json"
        summary_path = run_dir / "torch_profiler_summary.json"
        if trace_path.exists() or summary_path.exists():
            if not trace_path.is_file() or not summary_path.is_file():
                raise HardwareProfileError("Incomplete prior PyTorch-profiler capture")
            with summary_path.open("r", encoding="utf-8") as handle:
                prior_summary = json.load(handle)
            if prior_summary.get("trace_sha256") != sha256_file(trace_path):
                raise HardwareProfileError("Prior PyTorch-profiler trace hash differs")
        else:
            capture_torch_profiler(trial, rows[0], run_dir)
    os.replace(partial, run_dir / "profiles.jsonl")
    metrics = aggregate(measured)
    atomic_write(run_dir / "metrics.json", canonical_bytes(metrics))
    from validate_hardware_runtime_profile import validate_and_finalize

    validation = validate_and_finalize(run_dir, MANIFEST)
    print(json.dumps({"status": "complete", "cell_id": args.cell_id, "metrics": validation["metrics"]}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (HardwareProfileError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"UNEXPECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
