#!/usr/bin/env python
"""Run one registered canonical-stack timing trial for the final val campaign."""

from __future__ import annotations

import argparse
import gc
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

from run_adaptive_matched_timing_trial import gpu_environment, jsonable  # noqa: E402
from run_baseline import apply_chat_template, eos_token_ids  # noqa: E402
from run_hardware_matched_timing_trial import (  # noqa: E402
    FIXED_CONTINUATION_TOKEN_IDS,
    Trial as InternalTrial,
)
from run_official_baseline import load_manifest, load_model, load_sample_images  # noqa: E402
from run_progressive_oracle_refcoco import (  # noqa: E402
    MAX_NEW_TOKENS,
    ORIGINAL_VISUAL_TOKENS,
    REGISTERED_PROMPT,
    prepare_full_inputs,
)
from vlm_pruning.artifacts import sha256_file, source_provenance  # noqa: E402
from vlm_pruning.efficiency import cache_tensor_bytes  # noqa: E402
from vlm_pruning.llava_hooks import FeatherLlavaForwarder  # noqa: E402


CAMPAIGN_ROOT = REPO_ROOT / "experiments/final_val_efficiency_baselines_20260818"
MANIFEST = (
    REPO_ROOT
    / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
)
MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
OUTPUT_ROOT = CAMPAIGN_ROOT / "timing/canonical"
MODEL = "llava-hf/llava-1.5-7b-hf"
METHODS = ("Full", "K12", "M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN")
INTERNAL_NAMES = {
    "K12": "K12_ORIGINAL",
    "M448_OPT": "M448_OPT",
    "Z_M448_0": "Z_M448_0",
}
ORDER_PLAN = {
    1: METHODS,
    2: ("K12", "M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN", "Full"),
    3: ("M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN", "Full", "K12"),
    4: ("Z_M448_0", "FEATHER_LLAVA_MAIN", "Full", "K12", "M448_OPT"),
    5: ("FEATHER_LLAVA_MAIN", "Full", "K12", "M448_OPT", "Z_M448_0"),
}
FIXED32_ORDER_PLAN = {key: ORDER_PLAN[key] for key in (1, 2, 3)}
SOURCE_PATHS = (
    Path("scripts/run_final_val_canonical_timing.py"),
    Path("scripts/run_hardware_matched_timing_trial.py"),
    Path("scripts/run_adaptive_matched_timing_trial.py"),
    Path("scripts/run_progressive_oracle_refcoco.py"),
    Path("scripts/run_baseline.py"),
    Path("scripts/run_official_baseline.py"),
    Path("vlm_pruning/artifacts.py"),
    Path("vlm_pruning/efficiency.py"),
    Path("vlm_pruning/llava_hooks/feather_pruning.py"),
    Path("vlm_pruning/llava_hooks/mid_layer_pruning.py"),
    Path("vlm_pruning/llava_hooks/progressive_pruning.py"),
    Path("vlm_pruning/llava_hooks/runtime_repairs.py"),
)


class CanonicalTimingError(RuntimeError):
    """Raised when a canonical timing cell violates its frozen contract."""


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
        raise CanonicalTimingError("Frozen 1,000-row timing manifest hash differs")
    plan = ORDER_PLAN if args.mode == "natural" else FIXED32_ORDER_PLAN
    if args.repetition not in plan:
        raise CanonicalTimingError("Repetition is outside the registered timing plan")
    expected = plan[args.repetition]
    if args.order_position != expected.index(args.method) + 1:
        raise CanonicalTimingError("Order position differs from the frozen counterbalance")
    if args.smoke:
        if not 1 <= args.limit <= 20 or not 0 <= args.warmup_examples <= 5:
            raise CanonicalTimingError("Smoke limit/warmup is outside [1,20]/[0,5]")
    elif args.limit != 1000 or args.warmup_examples != 20:
        raise CanonicalTimingError("Accepted timing requires 1,000 rows and 20 warmups")


def checked_source() -> dict[str, Any]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *[str(path) for path in SOURCE_PATHS]],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if status:
        raise CanonicalTimingError("Canonical timing requires clean committed executable source")
    result = source_provenance(REPO_ROOT, SOURCE_PATHS)
    result["git_dirty"] = False
    return result


def new_cache_bytes(cache: Any) -> int:
    return cache_tensor_bytes(cache)


class FullTrial:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.torch, self.processor, self.model, self.device, self.model_env = load_model(MODEL, "bf16")

    def _token(self, token_id: int, template: Any) -> Any:
        return self.torch.tensor([[token_id]], dtype=template.dtype, device=template.device)

    def infer(self, sample: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        _, images = load_sample_images(sample)
        if len(images) != 1:
            raise CanonicalTimingError("Canonical timing requires one image")
        prompt = REGISTERED_PROMPT.format(query=str(sample["query"]))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        end_started = time.perf_counter()
        inputs, prep = prepare_full_inputs(torch, self.processor, self.model, self.device, images, prompt)
        attention_mask = inputs["attention_mask"]
        torch.cuda.synchronize()
        prefill_started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(
                inputs_embeds=inputs["inputs_embeds"], attention_mask=attention_mask,
                use_cache=True, return_dict=True, logits_to_keep=1,
            )
            next_token = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize()
        prefill = time.perf_counter() - prefill_started
        prompt_kv = new_cache_bytes(output.past_key_values)
        generated = [next_token]
        cache = output.past_key_values
        eos = eos_token_ids(self.model, self.processor)
        finished = bool(eos and int(next_token.item()) in eos)
        torch.cuda.synchronize()
        decode_started = time.perf_counter()
        with torch.inference_mode():
            if self.config["mode"] == "fixed32":
                generated = []
                for token_id in FIXED_CONTINUATION_TOKEN_IDS:
                    token = self._token(token_id, inputs["input_ids"])
                    attention_mask = torch.cat([attention_mask, torch.ones_like(token)], dim=1)
                    output = self.model(
                        input_ids=token, attention_mask=attention_mask, past_key_values=cache,
                        use_cache=True, return_dict=True, logits_to_keep=1,
                    )
                    cache = output.past_key_values
                    generated.append(token)
            else:
                while len(generated) < MAX_NEW_TOKENS and not finished:
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
                    output = self.model(
                        input_ids=next_token, attention_mask=attention_mask, past_key_values=cache,
                        use_cache=True, return_dict=True, logits_to_keep=1,
                    )
                    cache = output.past_key_values
                    next_token = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
                    generated.append(next_token)
                    finished = bool(eos and int(next_token.item()) in eos)
        torch.cuda.synchronize()
        decode = time.perf_counter() - decode_started
        end_to_end = time.perf_counter() - end_started
        ids = torch.cat(generated, dim=1)[0]
        raw = self.processor.decode(ids, skip_special_tokens=True).strip()
        count = int(ids.numel())
        result = timing_row(
            sample, self.config, raw=raw, generated_count=count, prep=prep,
            scoring=0.0, selector=0.0, gather=0.0, merge=0.0, prefill=prefill,
            decode=decode, end_to_end=end_to_end, prompt_kv=prompt_kv,
            final_kv=new_cache_bytes(cache), trace=[576] * 32,
            peak_allocated=int(torch.cuda.max_memory_allocated()),
            peak_reserved=int(torch.cuda.max_memory_reserved()),
        )
        for image in images:
            image.close()
        del inputs, output, cache
        gc.collect()
        return result


class FeatherTrial:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.torch, self.processor, self.model, self.device, self.model_env = load_model(MODEL, "bf16")
        image_token = int(getattr(self.model.config, "image_token_index", 32000))
        self.forwarder = FeatherLlavaForwarder(
            self.torch, self.model, image_token_id=image_token,
            diagnostics_mode="benchmark", synchronise_subtimings=True,
        )

    def infer(self, sample: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        _, images = load_sample_images(sample)
        prompt = REGISTERED_PROMPT.format(query=str(sample["query"]))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        end_started = time.perf_counter()
        inputs, prep = prepare_full_inputs(torch, self.processor, self.model, self.device, images, prompt)
        if self.config["mode"] == "fixed32":
            torch.cuda.synchronize()
            prefill_started = time.perf_counter()
            with torch.inference_mode():
                prefill_result = self.forwarder.prefill_with_segment_caches(
                    inputs_embeds=inputs["inputs_embeds"], input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"], sample_id=str(sample["sample_id"]),
                    seed=0, query_original_positions=(),
                )
            torch.cuda.synchronize()
            prefill = time.perf_counter() - prefill_started
            state = prefill_result.state
            torch.cuda.synchronize()
            decode_started = time.perf_counter()
            generated = []
            with torch.inference_mode():
                for token_id in FIXED_CONTINUATION_TOKEN_IDS:
                    token = torch.tensor(
                        [[token_id]], dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device
                    )
                    self.forwarder.decode_one_with_segment_caches(state, token)
                    generated.append(token)
            torch.cuda.synchronize()
            decode = time.perf_counter() - decode_started
            ids = torch.cat(generated, dim=1)[0]
            diagnostics = {**prefill_result.diagnostics, **self.forwarder._diagnostics(state)}
            diagnostics["llm_prefill_latency_s"] = prefill
            diagnostics["decode_latency_s"] = decode
            final_kv = sum(cache_tensor_bytes(segment.cache) for segment in state.segments)
            raw = self.processor.decode(ids, skip_special_tokens=True).strip()
            count = len(FIXED_CONTINUATION_TOKEN_IDS)
        else:
            generation = self.forwarder.generate_with_segment_caches(
                inputs_embeds=inputs["inputs_embeds"], input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"], sample_id=str(sample["sample_id"]),
                seed=0, query_original_positions=(), max_new_tokens=MAX_NEW_TOKENS,
                eos_ids=eos_token_ids(self.model, self.processor),
            )
            diagnostics = generation.diagnostics
            raw = self.processor.decode(generation.generated_ids, skip_special_tokens=True).strip()
            count = int(generation.generated_ids.numel())
            final_kv = int(diagnostics["cache/final_total_bytes"])
        torch.cuda.synchronize()
        end_to_end = time.perf_counter() - end_started
        trace = list(diagnostics["progressive/realized_visual_tokens_by_layer"])
        result = timing_row(
            sample, self.config, raw=raw, generated_count=count, prep=prep,
            scoring=float(diagnostics["feather/score_latency_s"]),
            selector=float(diagnostics["feather/selector_latency_s"]),
            gather=float(diagnostics["feather/gather_latency_s"]), merge=0.0,
            prefill=float(diagnostics["llm_prefill_latency_s"]),
            decode=float(diagnostics["decode_latency_s"]), end_to_end=end_to_end,
            prompt_kv=int(diagnostics["cache/prompt_total_bytes"]), final_kv=final_kv,
            trace=trace, peak_allocated=int(torch.cuda.max_memory_allocated()),
            peak_reserved=int(torch.cuda.max_memory_reserved()),
        )
        result["stage_traces"] = list(diagnostics["progressive/stage_traces"])
        result["feather_stage_metadata"] = list(diagnostics["feather/stage_metadata"])
        for image in images:
            image.close()
        del inputs
        gc.collect()
        return result


def timing_row(
    sample: dict[str, Any], config: dict[str, Any], *, raw: str, generated_count: int,
    prep: dict[str, Any], scoring: float, selector: float, gather: float, merge: float,
    prefill: float, decode: float, end_to_end: float, prompt_kv: int, final_kv: int,
    trace: list[int], peak_allocated: int, peak_reserved: int,
) -> dict[str, Any]:
    if len(trace) != 32 or any(not 0 <= int(value) <= 576 for value in trace):
        raise CanonicalTimingError(f"Invalid physical visual trace: {trace}")
    return {
        "sample_id": str(sample["sample_id"]), "method": config["method"],
        "repetition": config["repetition"], "order_position": config["order_position"],
        "mode": config["mode"], "raw_model_output": raw,
        "generated_token_count": generated_count,
        "vision_encoder_latency_s": float(prep["vision_encoder_latency_s"]),
        "projector_latency_s": float(prep["projector_latency_s"]),
        "query_scoring_latency_s": scoring, "selector_latency_s": selector,
        "gather_latency_s": gather, "merge_latency_s": merge, "route_gate_latency_s": 0.0,
        "main_prefill_with_overhead_latency_s": prefill,
        "main_prefill_excluding_pruning_latency_s": max(0.0, prefill - scoring - selector - gather - merge),
        "time_to_first_token_s": float(prep["vision_encoder_latency_s"]) + float(prep["projector_latency_s"]) + prefill,
        "decode_latency_s": decode,
        "decode_step_count": generated_count if config["mode"] == "fixed32" else max(generated_count - 1, 1),
        "decode_time_per_generated_token_s": decode / max(generated_count, 1),
        "end_to_end_latency_s": end_to_end,
        "throughput_examples_per_s": 1.0 / end_to_end,
        "prompt_kv_total_bytes": prompt_kv, "final_kv_total_bytes": final_kv,
        "peak_allocated_gpu_bytes": peak_allocated, "peak_reserved_gpu_bytes": peak_reserved,
        "realized_visual_tokens_by_layer": [int(value) for value in trace],
        "visual_token_layer_work": int(sum(trace)), "average_visual_tokens": sum(trace) / 32,
        "physical_routes_executed": 1,
        "fixed_continuation_token_ids": list(FIXED_CONTINUATION_TOKEN_IDS) if config["mode"] == "fixed32" else None,
    }


def build_trial(config: dict[str, Any]) -> Any:
    method = config["method"]
    if method == "Full":
        return FullTrial(config)
    if method == "FEATHER_LLAVA_MAIN":
        return FeatherTrial(config)
    internal_config = dict(config)
    internal_config["method"] = INTERNAL_NAMES[method]
    trial = InternalTrial(internal_config)
    original_infer = trial.infer

    def infer(sample: dict[str, Any]) -> dict[str, Any]:
        row = original_infer(sample)
        row["method"] = method
        row["main_prefill_excluding_pruning_latency_s"] = max(
            0.0,
            float(row["main_prefill_with_overhead_latency_s"])
            - float(row["query_scoring_latency_s"])
            - float(row["selector_latency_s"])
            - float(row["gather_latency_s"])
            - float(row["merge_latency_s"]),
        )
        row["throughput_examples_per_s"] = 1.0 / float(row["end_to_end_latency_s"])
        return row

    trial.infer = infer
    return trial


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "vision_encoder_latency_s", "projector_latency_s", "query_scoring_latency_s",
        "selector_latency_s", "gather_latency_s", "merge_latency_s",
        "main_prefill_with_overhead_latency_s", "main_prefill_excluding_pruning_latency_s",
        "time_to_first_token_s", "decode_latency_s", "decode_time_per_generated_token_s",
        "end_to_end_latency_s", "generated_token_count", "prompt_kv_total_bytes",
        "final_kv_total_bytes", "peak_allocated_gpu_bytes", "peak_reserved_gpu_bytes",
        "average_visual_tokens", "visual_token_layer_work",
    )
    output: dict[str, Any] = {"rows": len(rows), "unique_ids": len({row["sample_id"] for row in rows})}
    output["statistics"] = {}
    for field in fields:
        values = [float(row[field]) for row in rows]
        ordered = sorted(values)
        output["statistics"][field] = {
            "mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values), "p95": ordered[int(0.95 * (len(ordered) - 1))],
        }
    output["all_physical_traces"] = all(len(row["realized_visual_tokens_by_layer"]) == 32 for row in rows)
    return output


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    source = checked_source()
    manifest = load_manifest(MANIFEST, None)
    if len(manifest) != 1000:
        raise CanonicalTimingError("Timing manifest must contain exactly 1,000 rows")
    root = OUTPUT_ROOT / ("smokes" if args.smoke else args.mode)
    run_dir = root / f"rep{args.repetition}_pos{args.order_position}_{args.method.lower()}_attempt{args.attempt}"
    if run_dir.exists() and not args.resume:
        raise CanonicalTimingError(f"Timing directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    semantic = {
        "method": args.method, "mode": args.mode, "repetition": args.repetition,
        "order_position": args.order_position, "attempt": args.attempt,
        "limit": args.limit, "warmup_examples": args.warmup_examples, "smoke": args.smoke,
    }
    config = {
        "schema_version": 1, "created_at_utc": utc_now(), **semantic,
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)), "manifest_sha256": MANIFEST_SHA256,
        "model": MODEL, "precision": "bf16", "decoding": "greedy", "max_new_tokens": 48,
        "source_git_commit": source["git_commit"], "source_sha256": source["source_sha256"],
        "semantic_sha256": hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest(),
    }
    config_path = run_dir / "config.json"
    if args.resume:
        prior = json.loads(config_path.read_text())
        config["created_at_utc"] = prior["created_at_utc"]
        if prior != config:
            raise CanonicalTimingError("Resume configuration differs")
    else:
        atomic_write(config_path, canonical_bytes(config))
    config_sha = sha256_file(config_path)
    trial = build_trial(config)
    environment = {
        "created_at_utc": utc_now(), "python": sys.version, "platform": platform.platform(),
        "torch": trial.torch.__version__, "cuda": trial.torch.version.cuda,
        "model_environment": trial.model_env, "gpu": gpu_environment(),
    }
    atomic_write(run_dir / "environment.json", canonical_bytes(jsonable(environment)))
    for sample in manifest[:args.warmup_examples]:
        trial.infer(sample)
    partial = run_dir / "timing.partial.jsonl"
    rows: list[dict[str, Any]] = []
    if args.resume and partial.exists():
        rows = [json.loads(line) for line in partial.read_text().splitlines() if line.strip()]
        if [row["sample_id"] for row in rows] != [sample["sample_id"] for sample in manifest[:len(rows)]]:
            raise CanonicalTimingError("Resume rows are not an exact manifest prefix")
        if any(row.get("config_sha256") != config_sha for row in rows):
            raise CanonicalTimingError("Resume config hash differs")
    with partial.open("a" if rows else "w", encoding="utf-8") as handle:
        for index in range(len(rows), args.limit):
            row = trial.infer(manifest[index])
            row.update({"manifest_index": index, "config_sha256": config_sha, "source_git_commit": source["git_commit"]})
            rows.append(row)
            handle.write(json.dumps(jsonable(row), sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush()
            if (index + 1) % 25 == 0 or index + 1 == args.limit:
                print(f"FINAL_CANONICAL_TIMING {args.mode} rep{args.repetition} {args.method}: {index + 1}/{args.limit}", flush=True)
    timing = run_dir / "timing.jsonl"
    os.replace(partial, timing)
    metrics = summarize(rows)
    atomic_write(run_dir / "metrics.json", canonical_bytes(metrics))
    integrity = {
        "status": "PASS", "rows": len(rows), "unique_ids": len({row["sample_id"] for row in rows}),
        "ordered": [row["sample_id"] for row in rows] == [sample["sample_id"] for sample in manifest[:args.limit]],
        "all_physical_traces": metrics["all_physical_traces"], "timing_sha256": sha256_file(timing),
    }
    if integrity["rows"] != args.limit or integrity["unique_ids"] != args.limit or not integrity["ordered"] or not integrity["all_physical_traces"]:
        raise CanonicalTimingError("Timing artifact failed final integrity")
    atomic_write(run_dir / "integrity_report.json", canonical_bytes(integrity))
    atomic_write(run_dir / "COMPLETED.json", canonical_bytes({"status": "complete", **integrity}))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (CanonicalTimingError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
