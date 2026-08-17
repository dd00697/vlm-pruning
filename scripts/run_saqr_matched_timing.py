#!/usr/bin/env python
"""Run one preregistered natural-greedy K12 or SAQR196 timing repetition."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for entry in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from run_baseline import apply_chat_template, eos_token_ids  # noqa: E402
from run_hardware_matched_timing_trial import Trial as K12Trial  # noqa: E402
from run_official_baseline import load_manifest  # noqa: E402
from run_saqr_campaign import (  # noqa: E402
    CAMPAIGN_ROOT,
    CHECKPOINT_ROOT,
    DISCOVERY_MANIFEST,
    DISCOVERY_SHA256,
    MAX_NEW_TOKENS,
    REGISTERED_PROMPT,
    atomic_json,
    inference_modules,
    load_frozen_model,
    read_jsonl,
    set_determinism,
    sha256_file,
)
from vlm_pruning.llava_hooks.early_compression import ContiguousPoolForwarder  # noqa: E402
from vlm_pruning.llava_hooks.saqr import prepare_saqr_inputs  # noqa: E402


METHODS = ("K12_ORIGINAL", "SAQR196")
ORDER = {
    1: ("K12_ORIGINAL", "SAQR196"),
    2: ("SAQR196", "K12_ORIGINAL"),
    3: ("K12_ORIGINAL", "SAQR196"),
}
TIMING_ROOT = CAMPAIGN_ROOT / "matched_timing"


class TimingError(RuntimeError):
    pass


def check_no_other_cuda_process() -> None:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    if output:
        raise TimingError(f"Matched timing requires an idle GPU; found: {output}")


def source_clean() -> str:
    paths = [
        "scripts/run_saqr_matched_timing.py",
        "scripts/run_saqr_campaign.py",
        "vlm_pruning/llava_hooks/saqr.py",
        "vlm_pruning/llava_hooks/early_compression.py",
        "scripts/run_hardware_matched_timing_trial.py",
    ]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=REPO_ROOT, text=True
    ).strip()
    if dirty:
        raise TimingError("Matched timing requires committed executable source")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


class SAQRTimingTrial:
    def __init__(self, checkpoint: Path) -> None:
        import torch

        self.torch, self.processor, self.model, self.device, self.environment = load_frozen_model()
        set_determinism(self.torch)
        self.conditioner, self.resampler = inference_modules(
            self.torch, self.model, self.device, checkpoint
        )
        self.forwarder = ContiguousPoolForwarder(self.torch, self.model)
        self.eos_ids = eos_token_ids(self.model, self.processor)

    def infer(self, sample: dict[str, Any], *, repetition: int, order_position: int) -> dict[str, Any]:
        torch = self.torch
        image = Image.open(sample["image_path"]).convert("RGB")
        prompt = REGISTERED_PROMPT.format(query=str(sample["query"]))
        text_prompt = apply_chat_template(self.processor, prompt, 1)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            compacted, prep, assignment = prepare_saqr_inputs(
                self.processor,
                self.model,
                self.device,
                image,
                text_prompt,
                str(sample["query"]),
                self.conditioner,
                self.resampler,
                output_grid=14,
                force_zero_query=False,
                sample_id=str(sample["sample_id"]),
            )
            generated_ids, generation = self.forwarder.generate_natural(
                compacted, max_new_tokens=MAX_NEW_TOKENS, eos_ids=self.eos_ids
            )
        torch.cuda.synchronize()
        end_to_end = time.perf_counter() - started
        generated_count = int(generated_ids.numel())
        query = float(prep["query_adapter_latency_s"])
        resampler = float(prep["resampler_latency_s"])
        assembly = float(prep["input_assembly_latency_s"])
        prefill = float(generation["llm_prefill_latency_s"])
        decode = float(generation["decode_latency_s"])
        result = {
            "sample_id": str(sample["sample_id"]),
            "method": "SAQR196",
            "repetition": repetition,
            "order_position": order_position,
            "mode": "natural",
            "raw_model_output": self.processor.decode(generated_ids, skip_special_tokens=True).strip(),
            "generated_token_count": generated_count,
            "vision_encoder_latency_s": float(prep["vision_encoder_latency_s"]),
            "projector_latency_s": float(prep["projector_latency_s"]),
            "query_adapter_latency_s": query,
            "resampler_latency_s": resampler,
            "input_assembly_latency_s": assembly,
            "query_scoring_latency_s": 0.0,
            "selector_latency_s": 0.0,
            "gather_latency_s": 0.0,
            "merge_latency_s": resampler,
            "main_prefill_with_overhead_latency_s": query + resampler + assembly + prefill,
            "llm_prefill_latency_s": prefill,
            "time_to_first_token_s": (
                float(prep["vision_encoder_latency_s"])
                + float(prep["projector_latency_s"])
                + query
                + resampler
                + assembly
                + prefill
            ),
            "decode_latency_s": decode,
            "decode_step_count": max(generated_count - 1, 0),
            "decode_time_per_generated_token_s": decode / max(generated_count - 1, 1),
            "end_to_end_latency_s": end_to_end,
            "prompt_kv_total_bytes": int(generation["cache/prompt_total_bytes"]),
            "final_kv_total_bytes": int(generation["cache/final_total_bytes"]),
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
            "realized_visual_tokens_by_layer": [196] * 32,
            "visual_token_layer_work": 6272,
            "average_visual_tokens": 196.0,
            "component_timing_scope": "synchronised_full_saqr_regular_path",
            "cache_implementation": generation["cache/implementation"],
            "prompt_cache_lengths": generation["cache/prompt_per_layer_lengths"],
            "final_cache_lengths": generation["cache/final_per_layer_lengths"],
            "physical_routes_executed": 1,
            "selector_or_topk": False,
            "decoder_segmentation": False,
        }
        image.close()
        del compacted, assignment
        gc.collect()
        return result


def validate_rows(path: Path, rows: list[dict[str, Any]], method: str, repetition: int) -> dict[str, Any]:
    results = read_jsonl(path)
    if [row["sample_id"] for row in results] != [str(row["sample_id"]) for row in rows]:
        raise TimingError("Timing rows do not match exact manifest order")
    if any(row["method"] != method or int(row["repetition"]) != repetition for row in results):
        raise TimingError("Timing row method/repetition mismatch")
    if method == "SAQR196":
        if any(row["visual_token_layer_work"] != 6272 or row["realized_visual_tokens_by_layer"] != [196] * 32 for row in results):
            raise TimingError("SAQR timing token trace mismatch")
        if any(row["selector_or_topk"] or row["decoder_segmentation"] for row in results):
            raise TimingError("SAQR timing executed a forbidden irregular path")
    keys = (
        "end_to_end_latency_s",
        "time_to_first_token_s",
        "decode_latency_s",
        "prompt_kv_total_bytes",
        "final_kv_total_bytes",
    )
    for row in results:
        if any(float(row[key]) < 0.0 for key in keys):
            raise TimingError("Timing artifact contains a negative measurement")
    return {
        "status": "PASS",
        "rows": len(results),
        "method": method,
        "repetition": repetition,
        "predictions_sha256": sha256_file(path),
        "means": {
            key: statistics.fmean(float(row[key]) for row in results)
            for key in keys
        },
        "generated_token_mean": statistics.fmean(float(row["generated_token_count"]) for row in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--repetition", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--order-position", type=int, choices=[1, 2], required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--warmup-examples", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if ORDER[args.repetition][args.order_position - 1] != args.method:
        raise TimingError("Method/order position differs from the frozen protocol")
    if args.smoke:
        if not 1 <= args.limit <= 5 or not 0 <= args.warmup_examples <= 5:
            raise TimingError("Smoke timing limits must be within [1,5]")
    elif args.limit != 1000 or args.warmup_examples != 20:
        raise TimingError("Accepted timing requires exactly 1,000 rows and 20 warmups")
    if sha256_file(DISCOVERY_MANIFEST) != DISCOVERY_SHA256:
        raise TimingError("Frozen discovery manifest hash changed")
    check_no_other_cuda_process()
    commit = source_clean()
    checkpoint = CHECKPOINT_ROOT / "stage_B_selected.pt"
    if args.method == "SAQR196" and not checkpoint.exists():
        raise TimingError("Selected SAQR196 checkpoint is missing")
    rows = load_manifest(DISCOVERY_MANIFEST, None)[: args.limit]
    run_dir = TIMING_ROOT / ("smokes" if args.smoke else "natural") / f"rep{args.repetition}_pos{args.order_position}_{args.method.lower()}_attempt1"
    output = run_dir / "timing.jsonl"
    existing = read_jsonl(output) if output.exists() and args.resume else []
    if output.exists() and not args.resume:
        raise TimingError(f"Timing output exists; use --resume: {output}")
    if [item["sample_id"] for item in existing] != [str(row["sample_id"]) for row in rows[: len(existing)]]:
        raise TimingError("Timing resume is not an exact manifest prefix")
    config = {
        "status": "RUNNING",
        "method": args.method,
        "repetition": args.repetition,
        "order_position": args.order_position,
        "limit": args.limit,
        "warmup_examples": args.warmup_examples,
        "smoke": args.smoke,
        "source_commit": commit,
        "manifest_sha256": DISCOVERY_SHA256,
        "checkpoint_sha256": sha256_file(checkpoint) if args.method == "SAQR196" else None,
    }
    atomic_json(run_dir / "config.json", config)
    if args.method == "K12_ORIGINAL":
        trial: Any = K12Trial({
            "method": "K12_ORIGINAL",
            "mode": "natural",
            "repetition": args.repetition,
            "order_position": args.order_position,
        })
    else:
        trial = SAQRTimingTrial(checkpoint)
    for warmup_index in range(args.warmup_examples):
        if args.method == "K12_ORIGINAL":
            trial.infer(rows[warmup_index % len(rows)])
        else:
            trial.infer(rows[warmup_index % len(rows)], repetition=args.repetition, order_position=args.order_position)
        print(f"warmup {warmup_index + 1}/{args.warmup_examples}", flush=True)
    for index, sample in enumerate(rows):
        if index < len(existing):
            continue
        if args.method == "K12_ORIGINAL":
            result = trial.infer(sample)
        else:
            result = trial.infer(sample, repetition=args.repetition, order_position=args.order_position)
        run_dir.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            print(f"timing {args.method} rep={args.repetition} {index + 1}/{len(rows)}", flush=True)
    validation = validate_rows(output, rows, args.method, args.repetition)
    atomic_json(run_dir / "VALIDATION.json", validation)
    config["status"] = "COMPLETE"
    atomic_json(run_dir / "config.json", config)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TimingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
