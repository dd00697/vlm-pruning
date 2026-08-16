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
    ProfiledMidLayerForwarder,
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
MODES = {
    "smoke2": {"limit": 2, "warmup": 0, "repetitions": 1},
    "smoke20": {"limit": 20, "warmup": 2, "repetitions": 1},
    "profile100": {"limit": 100, "warmup": 20, "repetitions": 5},
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
    parser.add_argument("--cell-id", choices=CELL_IDS, required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--attempt", type=int, default=1, choices=[1, 2])
    return parser.parse_args()


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


def output_dir(cell_id: str, mode: str, attempt: int) -> Path:
    if mode.startswith("smoke"):
        return CAMPAIGN_ROOT / "profiling/smokes" / f"{cell_id.lower()}_{mode}_attempt{attempt}"
    return CAMPAIGN_ROOT / "profiling" / f"{cell_id.lower()}_attempt{attempt}"


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
            self.forwarder = ProfiledProgressiveForwarder(
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
        "prefill_latency_s",
        "decode_latency_s",
        "end_to_end_latency_s",
        "prompt_kv_total_bytes",
        "final_kv_total_bytes",
        "average_visual_tokens",
        "visual_token_layer_work",
        "generated_token_count",
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
    run_dir = output_dir(args.cell_id, args.mode, args.attempt)
    if run_dir.exists():
        raise HardwareProfileError(f"Profile run directory already exists: {run_dir}")
    launch = assert_storage_and_gpu_idle()
    source = git_source()
    semantic_command = {
        "program": "scripts/run_hardware_runtime_profile.py",
        "cell_id": args.cell_id,
        "mode": args.mode,
        "attempt": args.attempt,
        **contract,
    }
    config = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "scope": "instrumentation_smoke" if args.mode.startswith("smoke") else "runtime_profile",
        "cell_id": args.cell_id,
        "mode": args.mode,
        "attempt": args.attempt,
        "limit": contract["limit"],
        "warmup_examples": contract["warmup"],
        "measured_repetitions_per_example": contract["repetitions"],
        "manifest_path": relative(MANIFEST),
        "manifest_sha256": MANIFEST_SHA256,
        "manifest_prefix_order_sha256": hashlib.sha256("\n".join(str(row["sample_id"]) for row in rows).encode()).hexdigest(),
        "sample_index_by_id": {str(row["sample_id"]): index for index, row in enumerate(rows)},
        "historical_source_path": relative(historical_path),
        "historical_source_sha256": historical_hash,
        "historical_timing_source_path": relative(timing_historical_path),
        "historical_timing_source_sha256": timing_historical_hash,
        "model": MODEL,
        "precision": "bf16",
        "decoding": "deterministic_greedy",
        "max_new_tokens": MAX_NEW_TOKENS,
        "schedule": [event.as_dict() for event in M448_SCHEDULE] if args.cell_id != "P_K12" else [{"boundary": 12, "budget": 48, "selector": "original_k12_ug"}],
        "replay_score_and_rank_disabled_only": args.cell_id == "P_M448_REPLAY",
        "runtime_profiler": "synchronized_cpu_wall_and_cuda_event_v1",
        "source_git_commit": source["git_commit"],
        "source_sha256": source["source_sha256"],
        "semantic_command": semantic_command,
        "semantic_command_sha256": hashlib.sha256(json.dumps(semantic_command, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "run_dir": relative(run_dir),
        "launch_storage_and_gpu": launch,
    }
    run_dir.mkdir(parents=True)
    atomic_write(run_dir / "config.json", canonical_bytes(config))
    config_sha = sha256_file(run_dir / "config.json")
    trial = ProfileTrial(
        config,
        replay=replay,
        historical=historical,
        timing_historical=timing_historical,
    )
    environment = {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": trial.torch.__version__,
        "cuda": trial.torch.version.cuda,
        "model_environment": trial.model_environment,
        "launch": launch,
    }
    atomic_write(run_dir / "environment.json", canonical_bytes(jsonable(environment)))
    for sample in manifest_rows[: int(contract["warmup"])]:
        trial.infer(sample, 0)
    if trial.torch.cuda.is_available():
        trial.torch.cuda.synchronize()
    measured: list[dict[str, Any]] = []
    partial = run_dir / "profiles.partial.jsonl"
    with partial.open("w", encoding="utf-8") as handle:
        for manifest_index, sample in enumerate(rows):
            for repetition in range(1, int(contract["repetitions"]) + 1):
                result = trial.infer(sample, repetition)
                result["config_sha256"] = config_sha
                result["source_git_commit"] = config["source_git_commit"]
                measured.append(result)
                handle.write(json.dumps(jsonable(result), sort_keys=True, ensure_ascii=True) + "\n")
                handle.flush()
            print(f"PROFILE {args.cell_id} {args.mode}: {manifest_index + 1}/{len(rows)}", flush=True)
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
