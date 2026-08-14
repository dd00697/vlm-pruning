#!/usr/bin/env python
"""Run one isolated, paired exploratory timing cell on RefCOCO validation."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_baseline import apply_chat_template, eos_token_ids  # noqa: E402
from run_official_baseline import load_manifest, load_model, load_sample_images, manifest_checksum  # noqa: E402
from run_progressive_oracle_refcoco import (  # noqa: E402
    GRID,
    MAX_NEW_TOKENS,
    ORIGINAL_VISUAL_TOKENS,
    REGISTERED_PROMPT,
    json_checksum,
    jsonable,
    make_prompt,
    parse_schedule,
    prepare_full_inputs,
    query_resolution,
    schedule_string,
)
from vlm_pruning.artifacts import sha256_file, source_provenance  # noqa: E402
from vlm_pruning.llava_hooks import ProgressivePruneForwarder, PruningEvent  # noqa: E402


CAMPAIGN_ROOT = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814"
DEFAULT_MANIFEST = CAMPAIGN_ROOT / "manifests/refcoco_val_discovery1000.jsonl"
DEFAULT_OUTPUT_ROOT = CAMPAIGN_ROOT / "timing"
DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
WARMUP_EXAMPLES = 20


class TimingRunError(RuntimeError):
    """Raised when the paired timing contract is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True, choices=["E1", "E2"])
    parser.add_argument("--candidate-cell-id", required=True)
    parser.add_argument("--candidate-schedule", type=parse_schedule, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--warmup-examples", type=int, default=WARMUP_EXAMPLES)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def source_contract() -> dict[str, Any]:
    paths = [
        Path("scripts/run_progressive_oracle_timing.py"),
        Path("scripts/run_progressive_oracle_timing_campaign.py"),
        Path("scripts/validate_progressive_oracle_timing.py"),
        Path("scripts/run_progressive_oracle_refcoco.py"),
        Path("scripts/run_baseline.py"),
        Path("vlm_pruning/artifacts.py"),
        Path("vlm_pruning/features/query_tokens.py"),
        Path("vlm_pruning/llava_hooks/mid_layer_pruning.py"),
        Path("vlm_pruning/llava_hooks/progressive_pruning.py"),
    ]
    provenance = source_provenance(REPO_ROOT, paths)
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *[str(path) for path in paths]],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    provenance["git_dirty"] = bool(status)
    provenance["code_paths_checked"] = [str(path) for path in paths]
    return provenance


def build_config(args: argparse.Namespace, rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    provenance = source_contract()
    if provenance["git_dirty"]:
        raise TimingRunError("Timing requires a clean committed timing/source revision")
    if len(rows) != 1000:
        raise TimingRunError(f"Timing requires exactly 1,000 manifest rows, found {len(rows)}")
    if not 0 <= int(args.warmup_examples) <= 1000:
        raise TimingRunError("Warm-up count must be in [0, 1000]")
    return {
        "schema_version": 1,
        "cell_id": args.cell_id,
        "scope": "exploratory_timing",
        "candidate_cell_id": args.candidate_cell_id,
        "candidate_schedule": [event.as_dict() for event in args.candidate_schedule],
        "candidate_schedule_string": schedule_string(args.candidate_schedule),
        "full_schedule": [],
        "matched_full_reference_internal": True,
        "pair_order": "alternating_full_first_by_zero_based_manifest_index",
        "warmup_examples_per_path": int(args.warmup_examples),
        "timing_implementation": "paired-progressive-cuda-sync-v1",
        "source/git_commit": provenance["git_commit"],
        "source/git_dirty": provenance["git_dirty"],
        "source/sha256": provenance["source_sha256"],
        "model/name": str(args.model_path),
        "model/precision": "bf16",
        "model/decoding": "deterministic_greedy",
        "model/max_new_tokens": MAX_NEW_TOKENS,
        "dataset/name": "refcoco",
        "dataset/split": "val",
        "dataset/manifest": str(args.manifest.resolve().relative_to(REPO_ROOT)),
        "dataset/manifest_checksum": manifest_checksum(args.manifest),
        "dataset/num_samples": len(rows),
        "prompt/template": REGISTERED_PROMPT,
        "position_mode": "gap_preserving",
        "original_visual_tokens": ORIGINAL_VISUAL_TOKENS,
        "grid": GRID,
        "created_at_utc": utc_now(),
        "run/output_dir": str(run_dir.relative_to(REPO_ROOT)),
    }


IMMUTABLE_CONFIG_KEYS = {
    "cell_id", "scope", "candidate_cell_id", "candidate_schedule", "full_schedule",
    "matched_full_reference_internal", "pair_order", "warmup_examples_per_path",
    "timing_implementation", "source/git_commit", "source/git_dirty", "source/sha256",
    "model/name", "model/precision", "model/decoding", "model/max_new_tokens",
    "dataset/name", "dataset/split", "dataset/manifest", "dataset/manifest_checksum",
    "dataset/num_samples", "prompt/template", "position_mode", "original_visual_tokens", "grid",
}


def compatible_config(existing: dict[str, Any], proposed: dict[str, Any]) -> bool:
    return all(existing.get(key) == proposed.get(key) for key in IMMUTABLE_CONFIG_KEYS)


def load_partial(path: Path, manifest_rows: list[dict[str, Any]], config_sha: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise TimingRunError(f"Blank timing partial row {line_number}")
            rows.append(json.loads(line))
    expected_ids = [str(row["sample_id"]) for row in manifest_rows[: len(rows)]]
    actual_ids = [str(row.get("sample_id") or "") for row in rows]
    if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise TimingRunError("Timing partial is not an exact unique manifest prefix")
    if any(row.get("config_sha256") != config_sha for row in rows):
        raise TimingRunError("Timing partial contains another source/config epoch")
    return rows


def segment_records(schedule: tuple[PruningEvent, ...], diagnostics: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    boundaries = [0] + [event.boundary for event in schedule] + [32]
    prompt_lengths = diagnostics["progressive/segment_prompt_lengths"]
    prompt_bytes = diagnostics["cache/prompt_segment_bytes"]
    final_bytes = diagnostics["cache/final_segment_bytes"]
    prompt: list[dict[str, Any]] = []
    final: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        base = {
            "segment_index": index,
            "start_layer": start,
            "end_layer": end,
            "prompt_length": int(prompt_lengths[index]),
        }
        prompt.append(base | {"kv_bytes": int(prompt_bytes[index])})
        final.append(base | {"kv_bytes": int(final_bytes[index])})
    return prompt, final


class PairedTrial:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.torch, self.processor, self.model, self.device, self.model_env = load_model(args.model_path, "bf16")
        image_token_id = int(getattr(self.model.config, "image_token_index", 32000))
        self.forwarders = {
            "full": ProgressivePruneForwarder(
                self.torch,
                self.model,
                image_token_id=image_token_id,
                schedule=(),
                grid=GRID,
                initial_visual_tokens=ORIGINAL_VISUAL_TOKENS,
                diagnostics_mode="benchmark",
            ),
            "candidate": ProgressivePruneForwarder(
                self.torch,
                self.model,
                image_token_id=image_token_id,
                schedule=args.candidate_schedule,
                grid=GRID,
                initial_visual_tokens=ORIGINAL_VISUAL_TOKENS,
                diagnostics_mode="benchmark",
            ),
        }
        self.image_token_id = image_token_id

    def infer(self, sample: dict[str, Any], variant: str, *, config_sha: str, source_commit: str) -> dict[str, Any]:
        torch = self.torch
        image_paths, images = load_sample_images(sample)
        if len(images) != 1:
            raise TimingRunError("Timing protocol requires exactly one RefCOCO image")
        prompt = make_prompt(sample)
        text_prompt = apply_chat_template(self.processor, prompt, 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        end_to_end_start = time.perf_counter()
        inputs, prep = prepare_full_inputs(torch, self.processor, self.model, self.device, images, prompt)
        resolution = query_resolution(
            self.processor,
            text_prompt,
            str(sample["query"]),
            inputs["input_ids"],
            self.image_token_id,
        )
        generation = self.forwarders[variant].generate_with_segment_caches(
            inputs_embeds=inputs["inputs_embeds"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            sample_id=str(sample["sample_id"]),
            seed=0,
            query_original_positions=resolution.token_positions,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_ids=eos_token_ids(self.model, self.processor),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_to_end = time.perf_counter() - end_to_end_start
        diagnostics = jsonable(generation.diagnostics)
        schedule = () if variant == "full" else self.args.candidate_schedule
        prompt_segments, final_segments = segment_records(schedule, diagnostics)
        generated = int(generation.generated_ids.numel())
        selector_stages = [
            {
                "stage_index": int(trace["stage_index"]),
                "boundary": int(trace["boundary"]),
                "selector": trace["selector"],
                "latency_s": float(trace["selection_latency_s"]),
            }
            for trace in generation.stage_traces
        ]
        selector_total = sum(stage["latency_s"] for stage in selector_stages)
        raw_output = self.processor.decode(generation.generated_ids, skip_special_tokens=True).strip()
        result = {
            "variant": variant,
            "raw_model_output": raw_output,
            "config_sha256": config_sha,
            "source_git_commit": source_commit,
            "generated_token_count": generated,
            "vision_encoder_latency_s": float(prep["vision_encoder_latency_s"]),
            "projector_latency_s": float(prep["projector_latency_s"]),
            "llm_prefill_latency_s": float(diagnostics["llm_prefill_latency_s"]),
            "selector_stage_latencies": selector_stages,
            "selector_total_latency_s": selector_total,
            "decode_latency_s": float(diagnostics["decode_latency_s"]),
            "decode_time_per_generated_token_s": float(diagnostics["decode_latency_s"]) / generated,
            "end_to_end_latency_s": end_to_end,
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
            "prompt_kv_total_bytes": int(diagnostics["cache/prompt_total_bytes"]),
            "final_kv_total_bytes": int(diagnostics["cache/final_total_bytes"]),
            "prompt_kv_by_segment": prompt_segments,
            "final_kv_by_segment": final_segments,
            "stage_traces": generation.stage_traces,
            "final_retained_indices": generation.final_retained_indices,
            "final_retained_indices_checksum": json_checksum(generation.final_retained_indices),
            "realized_visual_tokens_by_layer": diagnostics["progressive/realized_visual_tokens_by_layer"],
            "candidate_average_visual_tokens": float(diagnostics["candidate_average_visual_tokens"]),
            **diagnostics,
        }
        for image in images:
            image.close()
        del inputs, prep, resolution, generation
        gc.collect()
        return result


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(statistics.fmean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "median": float(statistics.median(values)),
        "p95": float(np.quantile(array, 0.95)),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "vision_encoder_latency_s", "projector_latency_s", "llm_prefill_latency_s",
        "selector_total_latency_s", "decode_latency_s", "decode_time_per_generated_token_s",
        "end_to_end_latency_s", "generated_token_count", "prompt_kv_total_bytes",
        "final_kv_total_bytes", "candidate_average_visual_tokens",
    ]
    result: dict[str, Any] = {"samples": len(rows), "variants": {}}
    for variant in ("full", "candidate"):
        variants = [row[variant] for row in rows]
        result["variants"][variant] = {
            field: summarize([float(row[field]) for row in variants]) for field in fields
        }
        result["variants"][variant]["peak_allocated_gpu_bytes"] = max(int(row["peak_allocated_gpu_bytes"]) for row in variants)
        result["variants"][variant]["peak_reserved_gpu_bytes"] = max(int(row["peak_reserved_gpu_bytes"]) for row in variants)
        stage_count = len(variants[0]["selector_stage_latencies"])
        result["variants"][variant]["selector_stages"] = [
            {
                "stage_index": stage,
                "boundary": variants[0]["selector_stage_latencies"][stage]["boundary"],
                "selector": variants[0]["selector_stage_latencies"][stage]["selector"],
                "latency_s": summarize([
                    float(row["selector_stage_latencies"][stage]["latency_s"]) for row in variants
                ]),
            }
            for stage in range(stage_count)
        ]
    for field in ("llm_prefill_latency_s", "decode_latency_s", "end_to_end_latency_s"):
        candidate = result["variants"]["candidate"][field]["mean"]
        full = result["variants"]["full"][field]["mean"]
        result[f"candidate_to_full/{field}_ratio"] = candidate / full
        result[f"candidate_to_full/{field}_reduction"] = 1.0 - candidate / full
    return result


def run(args: argparse.Namespace) -> int:
    rows = load_manifest(args.manifest, None)
    if any(str(row.get("dataset") or "").lower() != "refcoco" or str(row.get("split") or "").lower() != "val" for row in rows):
        raise TimingRunError("Timing accepts RefCOCO validation only")
    run_name = args.run_name or f"{args.cell_id.lower()}_attempt1"
    run_dir = (args.output_root / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    proposed = build_config(args, rows, run_dir)
    config_path = run_dir / "config.json"
    if config_path.exists():
        if not args.resume:
            raise TimingRunError(f"Timing directory already exists: {run_dir}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not compatible_config(config, proposed):
            raise TimingRunError("Existing timing config is incompatible with resume")
    else:
        config = proposed
        atomic_write(config_path, canonical_json_bytes(config))
    config_sha = sha256_file(config_path)
    partial = run_dir / "per_sample_timing.partial.jsonl"
    final = run_dir / "per_sample_timing.jsonl"
    existing = load_partial(partial, rows, config_sha) if args.resume else []
    if final.exists() and not (run_dir / "COMPLETED.json").exists():
        raise TimingRunError("Unvalidated final timing artifact already exists")

    load_start = time.perf_counter()
    trial = PairedTrial(args)
    environment = {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "model_load_latency_s": time.perf_counter() - load_start,
        **trial.model_env,
    }
    atomic_write(run_dir / "environment.json", canonical_json_bytes(jsonable(environment)))
    for index, sample in enumerate(rows[: int(args.warmup_examples)]):
        order = ("full", "candidate") if index % 2 == 0 else ("candidate", "full")
        for variant in order:
            trial.infer(sample, variant, config_sha=config_sha, source_commit=config["source/git_commit"])
    if trial.torch.cuda.is_available():
        trial.torch.cuda.synchronize()

    mode = "a" if args.resume and partial.exists() else "w"
    with partial.open(mode, encoding="utf-8") as handle:
        for index in range(len(existing), len(rows)):
            sample = rows[index]
            order = ("full", "candidate") if index % 2 == 0 else ("candidate", "full")
            measured: dict[str, Any] = {}
            for variant in order:
                measured[variant] = trial.infer(
                    sample,
                    variant,
                    config_sha=config_sha,
                    source_commit=config["source/git_commit"],
                )
            row = {
                "sample_id": str(sample["sample_id"]),
                "manifest_index": index,
                "pair_order": list(order),
                "config_sha256": config_sha,
                "source_git_commit": config["source/git_commit"],
                "full": measured["full"],
                "candidate": measured["candidate"],
            }
            handle.write(json.dumps(jsonable(row), ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            if (index + 1) % 25 == 0 or index + 1 == len(rows):
                print(f"{args.cell_id}: timed {index + 1}/{len(rows)} pairs", flush=True)
    os.replace(partial, final)
    from validate_progressive_oracle_timing import validate_and_finalize_timing

    report = validate_and_finalize_timing(run_dir, args.manifest)
    if report["status"] != "pass":
        raise TimingRunError("Independent timing validation failed")
    print(json.dumps({"status": "complete", "metrics": report["metrics"]}, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted; compatible timing pairs were preserved.", file=sys.stderr)
        return 130
    except (TimingRunError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"UNEXPECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
