#!/usr/bin/env python
"""Run one isolated matched-timing method/repetition on frozen RefCOCO val."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_adaptive_route_gate import load_gate  # noqa: E402
from run_baseline import apply_chat_template, eos_token_ids, prepare_compacted_inputs  # noqa: E402
from run_official_baseline import load_manifest, load_model, load_sample_images  # noqa: E402
from run_progressive_oracle_refcoco import (  # noqa: E402
    MAX_NEW_TOKENS,
    ORIGINAL_VISUAL_TOKENS,
    REGISTERED_PROMPT,
    json_checksum,
    prepare_full_inputs,
    query_resolution,
)
from run_ug_prune_single_pass import generate_mid_layer_answer  # noqa: E402
from vlm_pruning.artifacts import sha256_file, source_provenance  # noqa: E402
from vlm_pruning.llava_hooks.mid_layer_pruning import (  # noqa: E402
    MidLayerUGPruneForwarder,
    ug_gap58_config_for_prune_layer,
)
from vlm_pruning.llava_hooks.progressive_pruning import (  # noqa: E402
    ProgressivePruneForwarder,
    PruningEvent,
    _tensor_checksum,
    deterministic_stage_seed,
)
from vlm_pruning.llava_hooks.route_gate import (  # noqa: E402
    AGGRESSIVE_ROUTE,
    AGGRESSIVE_SCHEDULE,
    SAFE_ROUTE,
    DynamicRouteForwarder,
    RouteDecision,
    route_schedule,
)


CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
DEFAULT_MANIFEST = REPO_ROOT / (
    "experiments/progressive_oracle_refcoco_val_20260814/"
    "manifests/refcoco_val_discovery1000.jsonl"
)
DEFAULT_OUTPUT_ROOT = CAMPAIGN_ROOT / "matched_timing"
SELECTION_PATH = CAMPAIGN_ROOT / "analysis/GATE_RESULTS_AND_SELECTION.json"
TRAINING_ROOT = CAMPAIGN_ROOT / "gate_training"
EXPECTED_MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
MODEL = "llava-hf/llava-1.5-7b-hf"
METHODS = ("full", "k12_original", "candidate")
FIXED_GENERATED_TOKENS = 32
SOURCE_PATHS = (
    Path("scripts/run_adaptive_matched_timing_trial.py"),
    Path("scripts/validate_adaptive_matched_timing_trial.py"),
    Path("scripts/run_adaptive_matched_timing_campaign.py"),
    Path("scripts/analyse_adaptive_matched_timing.py"),
    Path("scripts/run_adaptive_route_gate.py"),
    Path("scripts/run_progressive_oracle_refcoco.py"),
    Path("scripts/run_ug_prune_single_pass.py"),
    Path("scripts/run_baseline.py"),
    Path("scripts/run_official_baseline.py"),
    Path("vlm_pruning/artifacts.py"),
    Path("vlm_pruning/features/query_tokens.py"),
    Path("vlm_pruning/llava_hooks/mid_layer_pruning.py"),
    Path("vlm_pruning/llava_hooks/progressive_pruning.py"),
    Path("vlm_pruning/llava_hooks/route_gate.py"),
    Path("vlm_pruning/llava_hooks/route_gate_model.py"),
)


class MatchedTimingError(RuntimeError):
    """Raised when matched timing violates its frozen contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


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


def summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(statistics.fmean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "median": float(statistics.median(values)),
        "p95": float(np.quantile(array, 0.95)),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "vision_encoder_latency_s",
        "projector_latency_s",
        "query_scoring_latency_s",
        "selector_latency_s",
        "gather_latency_s",
        "merge_latency_s",
        "route_gate_latency_s",
        "main_prefill_with_overhead_latency_s",
        "decode_latency_s",
        "decode_time_per_generated_token_s",
        "end_to_end_latency_s",
        "generated_token_count",
        "prompt_kv_total_bytes",
        "final_kv_total_bytes",
        "visual_token_layer_work",
        "average_visual_tokens",
    )
    result = {"samples": len(rows), "method": rows[0]["method"]}
    result.update({field: summarize([float(row[field]) for row in rows]) for field in fields})
    result["peak_allocated_gpu_bytes"] = max(int(row["peak_allocated_gpu_bytes"]) for row in rows)
    result["peak_reserved_gpu_bytes"] = max(int(row["peak_reserved_gpu_bytes"]) for row in rows)
    return result


class TimingSelectionMixin:
    """Split query scoring, ranking, and tensor gather without changing choices."""

    timing_stage_components: list[dict[str, float]]
    timing_gather_latency_s: float

    def reset_timing_components(self) -> None:
        self.timing_stage_components = []
        self.timing_gather_latency_s = 0.0

    def _timing_sync(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def select_stage_indices(
        self,
        hidden_states: Any,
        *,
        input_ids: Any,
        original_positions: Any,
        incoming_original_indices: Sequence[int],
        event: PruningEvent,
        stage_index: int,
        sample_id: str,
        seed: int,
        query_original_positions: Sequence[int],
    ) -> tuple[list[int], str | None, bool, int | None]:
        incoming = [int(value) for value in incoming_original_indices]
        if len(incoming) < event.budget:
            raise MatchedTimingError("Timing selector budget exceeds incoming tokens")
        if event.selector in {"query_topk", "query_topk_merge", "none"}:
            self._timing_sync()
            score_started = time.perf_counter()
            scores, _ = self.score_current_visual_tokens(
                hidden_states,
                input_ids=input_ids,
                original_positions=original_positions,
                query_original_positions=query_original_positions,
            )
            score_checksum = _tensor_checksum(scores)
            self._timing_sync()
            score_latency = time.perf_counter() - score_started
            self._timing_sync()
            selector_started = time.perf_counter()
            if event.selector == "none":
                selected: list[int] = []
            else:
                ranked = self.torch.argsort(scores, descending=True, stable=True)
                selected = sorted(
                    incoming[int(value)]
                    for value in ranked[: event.budget].detach().cpu().tolist()
                )
            self._timing_sync()
            selector_latency = time.perf_counter() - selector_started
            derived_seed = None
            recomputed = True
        else:
            self._timing_sync()
            selector_started = time.perf_counter()
            derived_seed = deterministic_stage_seed(
                sample_id=sample_id,
                cell_seed=seed,
                stage_index=stage_index,
                boundary=event.boundary,
                incoming_indices=incoming,
            )
            selected = sorted(random.Random(derived_seed).sample(incoming, event.budget))
            self._timing_sync()
            selector_latency = time.perf_counter() - selector_started
            score_latency = 0.0
            score_checksum = None
            recomputed = False
        selected = self._validate_original_indices(selected, incoming=incoming, budget=event.budget)
        self.timing_stage_components.append(
            {
                "query_scoring_latency_s": score_latency,
                "selector_latency_s": selector_latency,
            }
        )
        return selected, score_checksum, recomputed, derived_seed

    def _gather_after_selection(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any, Any]:
        self._timing_sync()
        started = time.perf_counter()
        result = super()._gather_after_selection(*args, **kwargs)
        self._timing_sync()
        self.timing_gather_latency_s += time.perf_counter() - started
        return result


class TimingProgressiveForwarder(TimingSelectionMixin, ProgressivePruneForwarder):
    pass


class TimingDynamicForwarder(TimingSelectionMixin, DynamicRouteForwarder):
    pass


def checked_source() -> dict[str, Any]:
    source = source_provenance(REPO_ROOT, SOURCE_PATHS)
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *[str(path) for path in SOURCE_PATHS]],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if status:
        raise MatchedTimingError("Matched timing requires clean committed executable source")
    source["git_dirty"] = False
    source["code_paths_checked"] = [str(path) for path in SOURCE_PATHS]
    return source


def gpu_environment() -> dict[str, Any]:
    queries = {
        "gpu": ["nvidia-smi", "--query-gpu=name,uuid,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.limit", "--format=csv,noheader,nounits"],
        "compute_apps": ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
    }
    result = {}
    for name, command in queries.items():
        try:
            result[name] = subprocess.check_output(command, text=True).strip().splitlines()
        except (OSError, subprocess.SubprocessError) as exc:
            result[name] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--order-position", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--attempt", type=int, default=1, choices=[1, 2])
    parser.add_argument("--mode", choices=["natural", "fixed32"], required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--warmup-examples", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace, selection: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    if sha256_file(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise MatchedTimingError("Matched timing requires the frozen validation manifest")
    source = checked_source()
    selected = selection["project_selection"]["selected"]
    if selected not in {"M448", "GATE_SEED0"}:
        raise MatchedTimingError(f"Timing runner does not recognize selected candidate {selected}")
    if args.smoke:
        if not 1 <= args.limit <= 5 or not 0 <= args.warmup_examples <= 5:
            raise MatchedTimingError("Timing smoke limit/warm-up must be in [1, 5]")
    elif args.limit != 1000 or args.warmup_examples < 20:
        raise MatchedTimingError("Full matched timing requires 1,000 rows and at least 20 warm-ups")
    if args.mode == "natural" and not 1 <= args.repetition <= 5:
        raise MatchedTimingError("Natural timing repetition must be in [1, 5]")
    if args.mode == "fixed32" and not 1 <= args.repetition <= 3:
        raise MatchedTimingError("Fixed timing repetition must be in [1, 3]")
    semantic = {
        "program": "scripts/run_adaptive_matched_timing_trial.py",
        "method": args.method,
        "repetition": args.repetition,
        "order_position": args.order_position,
        "attempt": args.attempt,
        "mode": args.mode,
        "limit": args.limit,
        "warmup_examples": args.warmup_examples,
        "smoke": args.smoke,
        "selected_candidate": selected,
    }
    return {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "scope": "matched_timing_smoke" if args.smoke else "matched_timing",
        "method": args.method,
        "repetition": args.repetition,
        "order_position": args.order_position,
        "attempt": args.attempt,
        "mode": args.mode,
        "limit": args.limit,
        "warmup_examples": args.warmup_examples,
        "fixed_generated_tokens": FIXED_GENERATED_TOKENS if args.mode == "fixed32" else None,
        "selected_candidate": selected,
        "selection_path": relative(SELECTION_PATH),
        "selection_sha256": sha256_file(SELECTION_PATH),
        "model/name": MODEL,
        "model/precision": "bf16",
        "model/decoding": "deterministic_greedy",
        "model/max_new_tokens": FIXED_GENERATED_TOKENS if args.mode == "fixed32" else MAX_NEW_TOKENS,
        "dataset/manifest": relative(args.manifest),
        "dataset/manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "dataset/split": "val",
        "prompt/template": REGISTERED_PROMPT,
        "source/git_commit": source["git_commit"],
        "source/git_dirty": False,
        "source/sha256": source["source_sha256"],
        "command/semantic": semantic,
        "command/sha256": hashlib.sha256(
            json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "run/output_dir": relative(run_dir),
    }


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


class Trial:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.torch, self.processor, self.model, self.device, self.model_env = load_model(MODEL, "bf16")
        self.image_token_id = int(getattr(self.model.config, "image_token_index", 32000))
        self.forwarder: Any | None = None
        self.gate_model = self.gate_mean = self.gate_std = self.calibration = None
        method = config["method"]
        selected = config["selected_candidate"]
        if method == "full":
            self.forwarder = TimingProgressiveForwarder(
                self.torch, self.model, image_token_id=self.image_token_id, schedule=(), diagnostics_mode="benchmark", synchronise_subtimings=True
            )
        elif method == "candidate" and selected == "M448":
            self.forwarder = TimingProgressiveForwarder(
                self.torch, self.model, image_token_id=self.image_token_id, schedule=AGGRESSIVE_SCHEDULE, diagnostics_mode="benchmark", synchronise_subtimings=True
            )
        elif method == "candidate" and selected == "GATE_SEED0":
            self.forwarder = TimingDynamicForwarder(
                self.torch, self.model, image_token_id=self.image_token_id, diagnostics_mode="benchmark", synchronise_subtimings=True
            )
            self.gate_model, self.gate_mean, self.gate_std, self.calibration = load_gate(
                TRAINING_ROOT, 0, self.torch, self.device
            )
        elif method == "k12_original":
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
            self.forwarder = MidLayerUGPruneForwarder(
                self.torch,
                self.model,
                image_token_id=self.image_token_id,
                grid=24,
                prune_after_layer=12,
                target_visual_tokens=48,
                selector_config=selector_config,
                diagnostics_mode="benchmark",
                synchronise_subtimings=True,
            )

    def _generation_contract(self) -> tuple[int, set[int]]:
        if self.config["mode"] == "fixed32":
            return FIXED_GENERATED_TOKENS, set()
        return MAX_NEW_TOKENS, eos_token_ids(self.model, self.processor)

    def infer(self, sample: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        _, images = load_sample_images(sample)
        if len(images) != 1:
            raise MatchedTimingError("Matched timing requires exactly one image")
        prompt = REGISTERED_PROMPT.format(query=str(sample["query"]))
        text_prompt = apply_chat_template(self.processor, prompt, 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        end_started = time.perf_counter()
        inputs, prep = prepare_full_inputs(torch, self.processor, self.model, self.device, images, prompt)
        resolution = query_resolution(
            self.processor,
            text_prompt,
            str(sample["query"]),
            inputs["input_ids"],
            self.image_token_id,
        )
        max_tokens, eos_ids = self._generation_contract()
        method = self.config["method"]
        route_name = None
        route_gate_latency = 0.0
        if method == "k12_original":
            raw, generated, _reported, selection, diagnostics = generate_mid_layer_answer(
                torch,
                self.processor,
                self.model,
                full_inputs=inputs,
                prep_diag=prep,
                sample_id=str(sample["sample_id"]),
                query_text=str(sample["query"]),
                query_token_status=resolution.resolver_status,
                query_token_positions=resolution.token_positions,
                seed=0,
                max_new_tokens=max_tokens,
                original_tokens=ORIGINAL_VISUAL_TOKENS,
                forwarder=self.forwarder,
                decode_mode="layer_aware_cache",
                eos_ids_override=eos_ids,
            )
            query_scoring = float(diagnostics["mid_layer/map_extraction_latency_s"])
            selector = float(diagnostics["mid_layer/selection_latency_s"])
            prefill = float(diagnostics["llm_prefill_latency_s"])
            known = (
                float(diagnostics["mid_layer/early_layers_latency_s"])
                + query_scoring
                + selector
                + float(diagnostics["mid_layer/late_layers_latency_s"])
            )
            gather = max(0.0, prefill - known)
            merge = 0.0
            trace = [576] * 12 + [48] * 20
            prompt_kv = int(diagnostics["cache/prompt_total_bytes"])
            final_kv = int(diagnostics["cache/final_total_bytes"])
            raw_output = raw
            generated_count = int(generated)
            gather_scope = "measured_prefill_residual_including_gather_and_cache_bookkeeping"
        else:
            assert isinstance(self.forwarder, TimingSelectionMixin)
            self.forwarder.reset_timing_components()
            capture: dict[str, float] = {}

            def decide(vector: Any) -> RouteDecision:
                assert self.gate_model is not None and self.gate_mean is not None and self.gate_std is not None
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    outputs = self.gate_model(((vector - self.gate_mean) / self.gate_std).unsqueeze(0))[0]
                    probability = float(torch.sigmoid(outputs[0]).item())
                torch.cuda.synchronize()
                capture["model_latency_s"] = time.perf_counter() - started
                threshold = float(self.calibration["threshold"])
                selected_route = SAFE_ROUTE if probability >= threshold else AGGRESSIVE_ROUTE
                capture["probability_safe"] = probability
                return RouteDecision(selected_route, probability, threshold, route_schedule(selected_route))

            if isinstance(self.forwarder, TimingDynamicForwarder):
                generation = self.forwarder.generate_with_route_gate(
                    inputs_embeds=inputs["inputs_embeds"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    sample_id=str(sample["sample_id"]),
                    seed=0,
                    query_original_positions=resolution.token_positions,
                    width=int(sample["width"]),
                    height=int(sample["height"]),
                    decide=decide,
                    max_new_tokens=max_tokens,
                    eos_ids=eos_ids,
                )
            else:
                generation = self.forwarder.generate_with_segment_caches(
                    inputs_embeds=inputs["inputs_embeds"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    sample_id=str(sample["sample_id"]),
                    seed=0,
                    query_original_positions=resolution.token_positions,
                    max_new_tokens=max_tokens,
                    eos_ids=eos_ids,
                )
            diagnostics = generation.diagnostics
            query_scoring = sum(item["query_scoring_latency_s"] for item in self.forwarder.timing_stage_components)
            selector = sum(item["selector_latency_s"] for item in self.forwarder.timing_stage_components)
            gather = float(self.forwarder.timing_gather_latency_s)
            merge = max(
                0.0,
                float(diagnostics.get("progressive/merge_latency_s") or 0.0) - gather,
            )
            route_gate_latency = float(diagnostics.get("gate/latency_s") or 0.0)
            route_name = diagnostics.get("gate/route_name")
            trace = list(diagnostics["progressive/realized_visual_tokens_by_layer"])
            prompt_kv = int(diagnostics["cache/prompt_total_bytes"])
            final_kv = int(diagnostics["cache/final_total_bytes"])
            prefill = float(diagnostics["llm_prefill_latency_s"])
            raw_output = self.processor.decode(generation.generated_ids, skip_special_tokens=True).strip()
            generated_count = int(generation.generated_ids.numel())
            gather_scope = "synchronised_tensor_gather_only"
            selection = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_to_end = time.perf_counter() - end_started
        decode = float(diagnostics["decode_latency_s"])
        result = {
            "sample_id": str(sample["sample_id"]),
            "method": method,
            "selected_candidate": self.config["selected_candidate"],
            "repetition": self.config["repetition"],
            "order_position": self.config["order_position"],
            "mode": self.config["mode"],
            "raw_model_output": raw_output,
            "generated_token_count": generated_count,
            "vision_encoder_latency_s": float(prep["vision_encoder_latency_s"]),
            "projector_latency_s": float(prep["projector_latency_s"]),
            "query_scoring_latency_s": query_scoring,
            "selector_latency_s": selector,
            "gather_latency_s": gather,
            "gather_timing_scope": gather_scope,
            "merge_latency_s": merge,
            "route_gate_latency_s": route_gate_latency,
            "main_prefill_with_overhead_latency_s": prefill,
            "decode_latency_s": decode,
            "decode_time_per_generated_token_s": decode / generated_count,
            "end_to_end_latency_s": end_to_end,
            "prompt_kv_total_bytes": prompt_kv,
            "final_kv_total_bytes": final_kv,
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
            "realized_visual_tokens_by_layer": trace,
            "visual_token_layer_work": int(sum(trace)),
            "average_visual_tokens": sum(trace) / 32,
            "route_name": route_name,
            "physical_routes_executed": 1,
        }
        for image in images:
            image.close()
        del inputs, prep, resolution
        gc.collect()
        return result


def run(args: argparse.Namespace) -> int:
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    rows = load_manifest(args.manifest, None)
    if len(rows) != 1000:
        raise MatchedTimingError("Matched timing manifest does not have 1,000 rows")
    mode_root = "smoke" if args.smoke else args.mode
    run_dir = args.output_root / mode_root / (
        f"rep{args.repetition}_pos{args.order_position}_{args.method}_attempt{args.attempt}"
    )
    if run_dir.exists():
        raise MatchedTimingError(f"Timing trial directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    config = build_config(args, selection, run_dir.resolve())
    atomic_write(run_dir / "config.json", canonical_bytes(config))
    config_sha = sha256_file(run_dir / "config.json")
    trial = Trial(config)
    environment = {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "model_environment": trial.model_env,
        "gpu_before_warmup": gpu_environment(),
        "torch": trial.torch.__version__,
        "cuda": trial.torch.version.cuda,
    }
    for sample in rows[: args.warmup_examples]:
        trial.infer(sample)
    trial.torch.cuda.synchronize()
    environment["gpu_after_warmup"] = gpu_environment()
    atomic_write(run_dir / "environment.json", canonical_bytes(jsonable(environment)))
    measured = []
    with (run_dir / "timing.jsonl").open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(rows[: args.limit], 1):
            result = trial.infer(sample)
            result["manifest_index"] = index - 1
            result["config_sha256"] = config_sha
            result["source_git_commit"] = config["source/git_commit"]
            measured.append(result)
            handle.write(json.dumps(jsonable(result), sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush()
            if index % 25 == 0 or index == args.limit:
                print(
                    f"TIMING {args.mode} rep{args.repetition} pos{args.order_position} {args.method}: {index}/{args.limit}",
                    flush=True,
                )
    metrics = aggregate(measured)
    atomic_write(run_dir / "metrics.json", canonical_bytes(metrics))
    from scripts.validate_adaptive_matched_timing_trial import validate_and_finalize

    report = validate_and_finalize(run_dir, args.manifest, expected_rows=args.limit)
    print(json.dumps({"status": "complete", "metrics": report["metrics"]}, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except (MatchedTimingError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"UNEXPECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
