#!/usr/bin/env python
"""Run paired source-native FastV timing cells on the frozen validation subset."""

from __future__ import annotations

import argparse
import hashlib
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


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for entry in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from run_baseline import apply_chat_template, eos_token_ids  # noqa: E402
from run_hardware_matched_timing_trial import FIXED_CONTINUATION_TOKEN_IDS  # noqa: E402
from run_official_baseline import (  # noqa: E402
    _move_processor_inputs,
    generate_official_fastv_answer,
    generate_official_fastv_disabled_answer,
    load_manifest,
    load_model,
    load_sample_images,
)
from run_progressive_oracle_refcoco import MAX_NEW_TOKENS, REGISTERED_PROMPT  # noqa: E402
from vlm_pruning.artifacts import sha256_file, source_provenance  # noqa: E402


CAMPAIGN_ROOT = REPO_ROOT / "experiments/final_val_efficiency_baselines_20260818"
MANIFEST = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
OUTPUT_ROOT = CAMPAIGN_ROOT / "timing/fastv_native"
MODEL = "llava-hf/llava-1.5-7b-hf"
FASTV_ROOT = REPO_ROOT / "third_party/FastV"
FASTV_COMMIT = "d1659729b5bf1be225e99ee15783deeea80f63b1"
METHODS = ("FASTV_STACK_FULL", "FASTV_MATCHED_199")
ORDER_PLAN = {
    1: METHODS, 2: tuple(reversed(METHODS)), 3: METHODS,
    4: tuple(reversed(METHODS)), 5: METHODS,
}
FIXED32_ORDER_PLAN = {1: METHODS, 2: tuple(reversed(METHODS)), 3: METHODS}
SOURCE_PATHS = (
    Path("scripts/run_final_val_fastv_timing.py"),
    Path("scripts/run_official_baseline.py"),
    Path("scripts/run_baseline.py"),
    Path("vlm_pruning/artifacts.py"),
)


class FastVTimingError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


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
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--mode", choices=("natural", "fixed32"), default="natural")
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--order-position", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--warmup-examples", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise FastVTimingError("Frozen timing manifest hash differs")
    plan = ORDER_PLAN if args.mode == "natural" else FIXED32_ORDER_PLAN
    if args.repetition not in plan:
        raise FastVTimingError("Repetition is outside the frozen plan")
    if args.order_position != plan[args.repetition].index(args.method) + 1:
        raise FastVTimingError("Order position differs from the frozen plan")
    if args.smoke:
        if not 1 <= args.limit <= 20 or not 0 <= args.warmup_examples <= 5:
            raise FastVTimingError("Invalid smoke limit/warmup")
    elif args.limit != 1000 or args.warmup_examples != 20:
        raise FastVTimingError("Accepted timing requires 1,000 rows and 20 warmups")


def checked_source() -> dict[str, Any]:
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=FASTV_ROOT, text=True).strip() != FASTV_COMMIT:
        raise FastVTimingError("FastV checkout commit differs")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=FASTV_ROOT, text=True).strip():
        raise FastVTimingError("FastV checkout is dirty")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *[str(path) for path in SOURCE_PATHS]],
        cwd=REPO_ROOT, text=True,
    ).strip()
    if status:
        raise FastVTimingError("FastV timing wrapper requires clean committed source")
    return source_provenance(REPO_ROOT, SOURCE_PATHS)


class Trial:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        active = config["method"] == "FASTV_MATCHED_199"
        self.torch, self.processor, self.model, self.device, self.environment = load_model(
            MODEL, "bf16", official_fastv=active, fastv_tokens=160,
        )

    def _fixed32(self, image: Any, prompt: str) -> tuple[str, int, dict[str, Any]]:
        torch = self.torch
        moved = _move_processor_inputs(torch, self.processor, self.model, self.device, image, prompt)
        active = self.config["method"] == "FASTV_MATCHED_199"
        self.model.config.fastv_config = ({
            "use_fastv": True, "fastv_k": 3, "fastv_r": 1.0 - 160 / 576,
            "image_token_start_index": int(
                torch.nonzero(
                    moved["input_ids"][0] == int(getattr(self.model.config, "image_token_index", 32000)),
                    as_tuple=False,
                )[0].item()
            ),
            "image_token_length": 576,
        } if active else None)
        lengths: list[int | None] = [None] * 32
        call = -1
        handles = []
        for index, layer in enumerate(self.model.language_model.model.layers):
            def hook(module: Any, args: tuple[Any, ...], index: int = index) -> None:
                nonlocal call
                del module
                if index == 0:
                    call += 1
                if call == 0:
                    lengths[index] = int(args[0].shape[1])
            handles.append(layer.register_forward_pre_hook(hook))
        state = dict(moved)
        torch.cuda.synchronize()
        prefill_started = time.perf_counter()
        with torch.inference_mode():
            self.model(**state, use_cache=False, return_dict=True)
        torch.cuda.synchronize()
        prefill = time.perf_counter() - prefill_started
        torch.cuda.synchronize()
        decode_started = time.perf_counter()
        with torch.inference_mode():
            for token_id in FIXED_CONTINUATION_TOKEN_IDS:
                token = torch.tensor([[token_id]], device=state["input_ids"].device, dtype=state["input_ids"].dtype)
                state["input_ids"] = torch.cat((state["input_ids"], token), dim=1)
                state["attention_mask"] = torch.cat((state["attention_mask"], torch.ones_like(token)), dim=1)
                self.model(**state, use_cache=False, return_dict=True)
        torch.cuda.synchronize()
        decode = time.perf_counter() - decode_started
        for handle in handles:
            handle.remove()
        expanded = int(lengths[0] or 0)
        text = expanded - 576
        trace = [int(value or 0) - text for value in lengths]
        expected = [576] * 3 + [160] * 29 if active else [576] * 32
        if trace != expected:
            raise FastVTimingError(f"Fixed32 official trace mismatch: {trace}")
        ids = torch.tensor(FIXED_CONTINUATION_TOKEN_IDS, device=state["input_ids"].device)
        return self.processor.decode(ids, skip_special_tokens=True).strip(), len(FIXED_CONTINUATION_TOKEN_IDS), {
            "llm_prefill_latency_s": prefill, "time_to_first_token_s": prefill,
            "decode_latency_s": decode, "total_inference_latency_s": prefill + decode,
            "per_layer_visual_tokens": trace, "pruning_selection_latency_s": None if active else 0.0,
        }

    def infer(self, sample: dict[str, Any]) -> dict[str, Any]:
        _, images = load_sample_images(sample)
        if len(images) != 1:
            raise FastVTimingError("FastV timing requires one image")
        prompt = REGISTERED_PROMPT.format(query=str(sample["query"]))
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()
        outer_started = time.perf_counter()
        if self.config["mode"] == "fixed32":
            raw, count, diagnostics = self._fixed32(images[0], prompt)
            selection = None
        elif self.config["method"] == "FASTV_MATCHED_199":
            raw, count, _, diagnostics, selection = generate_official_fastv_answer(
                self.torch, self.processor, self.model, self.device, images[0], prompt,
                MAX_NEW_TOKENS, 576, 160,
            )
        else:
            raw, count, _, diagnostics, selection = generate_official_fastv_disabled_answer(
                self.torch, self.processor, self.model, self.device, images[0], prompt,
                MAX_NEW_TOKENS, 576,
            )
        self.torch.cuda.synchronize()
        outer = time.perf_counter() - outer_started
        trace = list(diagnostics["per_layer_visual_tokens"])
        expected = [576] * 3 + [160] * 29 if self.config["method"] == "FASTV_MATCHED_199" else [576] * 32
        if trace != expected:
            raise FastVTimingError(f"Native FastV trace mismatch: {trace}")
        for image in images:
            image.close()
        return {
            "sample_id": str(sample["sample_id"]), "method": self.config["method"],
            "repetition": self.config["repetition"], "order_position": self.config["order_position"],
            "mode": self.config["mode"], "raw_model_output": raw, "generated_token_count": count,
            "vision_encoder_latency_s": None, "projector_latency_s": None,
            "query_scoring_latency_s": None, "selector_latency_s": diagnostics.get("pruning_selection_latency_s"),
            "gather_latency_s": None, "merge_latency_s": 0.0,
            "main_prefill_with_overhead_latency_s": float(diagnostics["llm_prefill_latency_s"]),
            "time_to_first_token_s": float(diagnostics["time_to_first_token_s"]),
            "decode_latency_s": float(diagnostics["decode_latency_s"]),
            "decode_time_per_generated_token_s": float(diagnostics["decode_latency_s"]) / max(count, 1),
            "source_native_inner_end_to_end_latency_s": float(diagnostics["total_inference_latency_s"]),
            "end_to_end_latency_s": outer, "throughput_examples_per_s": 1.0 / outer,
            "prompt_kv_total_bytes": 0, "final_kv_total_bytes": 0,
            "kv_cache_status": "source_native_use_cache_false",
            "peak_allocated_gpu_bytes": int(self.torch.cuda.max_memory_allocated()),
            "peak_reserved_gpu_bytes": int(self.torch.cuda.max_memory_reserved()),
            "realized_visual_tokens_by_layer": trace, "visual_token_layer_work": sum(trace),
            "average_visual_tokens": sum(trace) / 32,
            "retained_indices": list(selection.selected_indices) if selection is not None else None,
        }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "main_prefill_with_overhead_latency_s", "time_to_first_token_s", "decode_latency_s",
        "decode_time_per_generated_token_s", "source_native_inner_end_to_end_latency_s",
        "end_to_end_latency_s", "generated_token_count", "visual_token_layer_work",
        "average_visual_tokens", "peak_allocated_gpu_bytes", "peak_reserved_gpu_bytes",
    )
    result = {"rows": len(rows), "unique_ids": len({row["sample_id"] for row in rows}), "statistics": {}}
    for field in fields:
        values = [float(row[field]) for row in rows]
        result["statistics"][field] = {
            "mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0,
            "median": statistics.median(values), "p95": sorted(values)[int(0.95 * (len(values) - 1))],
        }
    return result


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    source = checked_source()
    manifest = load_manifest(MANIFEST, None)
    root = OUTPUT_ROOT / ("smokes" if args.smoke else args.mode)
    run_dir = root / f"rep{args.repetition}_pos{args.order_position}_{args.method.lower()}_attempt{args.attempt}"
    if run_dir.exists() and not args.resume:
        raise FastVTimingError(f"Run exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    semantic = vars(args).copy()
    config = {
        "schema_version": 1, "created_at_utc": utc_now(), **semantic,
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)), "manifest_sha256": MANIFEST_SHA256,
        "model": MODEL, "precision": "bf16", "max_new_tokens": 48,
        "fastv_commit": FASTV_COMMIT, "fastv_k": 3 if args.method == "FASTV_MATCHED_199" else None,
        "fastv_target": 160 if args.method == "FASTV_MATCHED_199" else 576,
        "source_git_commit": source["git_commit"], "source_sha256": source["source_sha256"],
        "semantic_sha256": hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest(),
    }
    config_path = run_dir / "config.json"
    if args.resume:
        prior = json.loads(config_path.read_text())
        config["created_at_utc"] = prior["created_at_utc"]
        if prior != config:
            raise FastVTimingError("Resume config differs")
    else:
        atomic_write(config_path, canonical_bytes(config))
    config_sha = sha256_file(config_path)
    trial = Trial(config)
    atomic_write(run_dir / "environment.json", canonical_bytes({
        "python": sys.version, "platform": platform.platform(), "torch": trial.torch.__version__,
        "cuda": trial.torch.version.cuda, "model_environment": trial.environment,
    }))
    for sample in manifest[:args.warmup_examples]:
        trial.infer(sample)
    partial = run_dir / "timing.partial.jsonl"
    rows: list[dict[str, Any]] = []
    if args.resume and partial.exists():
        rows = [json.loads(line) for line in partial.read_text().splitlines() if line.strip()]
        if [row["sample_id"] for row in rows] != [sample["sample_id"] for sample in manifest[:len(rows)]]:
            raise FastVTimingError("Resume is not a manifest prefix")
    with partial.open("a" if rows else "w", encoding="utf-8") as handle:
        for index in range(len(rows), args.limit):
            row = trial.infer(manifest[index])
            row.update({"manifest_index": index, "config_sha256": config_sha})
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            if (index + 1) % 25 == 0 or index + 1 == args.limit:
                print(f"FASTV_NATIVE_TIMING {args.mode} rep{args.repetition} {args.method}: {index + 1}/{args.limit}", flush=True)
    timing = run_dir / "timing.jsonl"
    os.replace(partial, timing)
    metrics = summarize(rows)
    atomic_write(run_dir / "metrics.json", canonical_bytes(metrics))
    integrity = {
        "status": "PASS", "rows": len(rows), "unique_ids": len({row["sample_id"] for row in rows}),
        "ordered": [row["sample_id"] for row in rows] == [sample["sample_id"] for sample in manifest[:args.limit]],
        "timing_sha256": sha256_file(timing),
    }
    if integrity["rows"] != args.limit or integrity["unique_ids"] != args.limit or not integrity["ordered"]:
        raise FastVTimingError("FastV timing integrity failed")
    atomic_write(run_dir / "integrity_report.json", canonical_bytes(integrity))
    atomic_write(run_dir / "COMPLETED.json", canonical_bytes({"status": "complete", **integrity}))
    print(json.dumps(metrics, indent=2))
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (FastVTimingError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
