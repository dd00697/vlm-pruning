#!/usr/bin/env python
"""Run one isolated final matched-timing trial for the hardware campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for entry in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from run_adaptive_matched_timing_trial import (  # noqa: E402
    TimingSelectionMixin,
    aggregate,
    gpu_environment,
    jsonable,
)
from run_baseline import apply_chat_template, eos_token_ids  # noqa: E402
from run_official_baseline import load_manifest, load_model, load_sample_images  # noqa: E402
from run_progressive_oracle_refcoco import (  # noqa: E402
    MAX_NEW_TOKENS,
    ORIGINAL_VISUAL_TOKENS,
    REGISTERED_PROMPT,
    prepare_full_inputs,
    query_resolution,
)
from run_ug_prune_single_pass import generate_mid_layer_answer  # noqa: E402
from vlm_pruning.artifacts import sha256_file, source_provenance  # noqa: E402
from vlm_pruning.efficiency import cache_tensor_bytes  # noqa: E402
from vlm_pruning.llava_hooks import (  # noqa: E402
    FusedNativeProgressiveForwarder,
    ProgressivePruneForwarder,
    PruningEvent,
    ug_gap58_config_for_prune_layer,
)
from vlm_pruning.llava_hooks.mid_layer_pruning import MidLayerUGPruneForwarder  # noqa: E402
from vlm_pruning.llava_hooks.progressive_pruning import _tensor_checksum  # noqa: E402


CAMPAIGN_ROOT = REPO_ROOT / "experiments/hardware_aware_zero_tail_refcoco_val_20260816"
DEFAULT_MANIFEST = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
DEFAULT_OUTPUT_ROOT = CAMPAIGN_ROOT / "matched_timing"
M448_OPT_SELECTION = CAMPAIGN_ROOT / "runtime_repairs/aggregate/M448_OPT_SELECTION.json"
QUALITY_RESULTS = CAMPAIGN_ROOT / "zero_tail/analysis/QUALITY_RESULTS.json"
EXPECTED_MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
MODEL = "llava-hf/llava-1.5-7b-hf"
METHODS = ("K12_ORIGINAL", "M448_OPT", "Z_K12_0", "Z_M448_0")
FIXED32_METHODS = ("K12_ORIGINAL", "M448_OPT", "Z_M448_0")
FIXED_GENERATED_TOKENS = 32
# Method-independent non-special continuation for the secondary fixed-decode
# diagnostic. Every method consumes these exact token IDs after prefill.
FIXED_CONTINUATION_TOKEN_IDS = tuple(range(1000, 1032))
SCHEDULES = {
    "M448_OPT": (
        PruningEvent(4, 448, "query_topk_merge"),
        PruningEvent(12, 48, "query_topk"),
    ),
    "Z_K12_0": (
        PruningEvent(12, 48, "k12_original"),
        PruningEvent(20, 0, "none"),
    ),
    "Z_M448_0": (
        PruningEvent(4, 448, "query_topk_merge"),
        PruningEvent(12, 48, "query_topk"),
        PruningEvent(20, 0, "none"),
    ),
}
ORDER_PLAN = {
    1: METHODS,
    2: ("M448_OPT", "Z_K12_0", "Z_M448_0", "K12_ORIGINAL"),
    3: ("Z_K12_0", "Z_M448_0", "K12_ORIGINAL", "M448_OPT"),
    4: ("Z_M448_0", "K12_ORIGINAL", "M448_OPT", "Z_K12_0"),
    5: METHODS,
}
FIXED32_ORDER_PLAN = {
    1: FIXED32_METHODS,
    2: ("M448_OPT", "Z_M448_0", "K12_ORIGINAL"),
    3: ("Z_M448_0", "K12_ORIGINAL", "M448_OPT"),
}
SOURCE_PATHS = (
    Path("scripts/run_hardware_matched_timing_trial.py"),
    Path("scripts/validate_hardware_matched_timing_trial.py"),
    Path("scripts/run_adaptive_matched_timing_trial.py"),
    Path("scripts/run_progressive_oracle_refcoco.py"),
    Path("scripts/run_ug_prune_single_pass.py"),
    Path("scripts/run_baseline.py"),
    Path("scripts/run_official_baseline.py"),
    Path("vlm_pruning/artifacts.py"),
    Path("vlm_pruning/features/query_tokens.py"),
    Path("vlm_pruning/llava_hooks/mid_layer_pruning.py"),
    Path("vlm_pruning/llava_hooks/progressive_pruning.py"),
    Path("vlm_pruning/llava_hooks/runtime_repairs.py"),
)


class HardwareTimingError(RuntimeError):
    """Raised when a final timing trial violates its frozen contract."""


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


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def checked_source() -> dict[str, Any]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *[str(path) for path in SOURCE_PATHS]],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if status:
        raise HardwareTimingError("Final timing requires clean committed executable source")
    source = source_provenance(REPO_ROOT, SOURCE_PATHS)
    source["git_dirty"] = False
    return source


class HardwareTimingSelectionMixin(TimingSelectionMixin):
    """Add exact zero-tail/K12 events and synchronized merge timing."""

    timing_merge_latency_s: float

    def reset_timing_components(self) -> None:
        super().reset_timing_components()
        self.timing_merge_latency_s = 0.0

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
        if event.selector == "none":
            selected: list[int] = []
            self.timing_stage_components.append(
                {"query_scoring_latency_s": 0.0, "selector_latency_s": 0.0}
            )
            return self._validate_original_indices(
                selected,
                incoming=incoming_original_indices,
                budget=0,
            ), None, False, None
        if event.selector == "k12_original":
            config = ug_gap58_config_for_prune_layer(
                12,
                map_source="query_max",
                fusion_mode="single_layer",
                probe_layers=(12,),
                position_mode="gap_preserving",
                target_ratio=0.45,
                context_ratio=0.40,
                anchor_ratio=0.15,
            )
            query_positions = self._current_query_positions(
                original_positions,
                query_original_positions,
            )
            query_position_list = [
                int(value) for value in query_positions.detach().cpu().tolist()
            ]
            frozen_hidden_states = [None] * 12 + [hidden_states]
            self._timing_sync()
            score_started = time.perf_counter()
            scores, layer_keys, map_stats = self._extract_fused_map_torch(
                hidden_states=frozen_hidden_states,
                input_ids=input_ids,
                query_token_positions=query_position_list,
                selector_config=config,
                target_visual_tokens=event.budget,
            )
            score_checksum = _tensor_checksum(scores)
            self._timing_sync()
            score_latency = time.perf_counter() - score_started
            self._timing_sync()
            selector_started = time.perf_counter()
            selection, _ = self._selection_from_fused_torch(
                scores=scores,
                sample_id=sample_id,
                original_tokens=len(incoming_original_indices),
                seed=seed,
                map_key="+".join(layer_keys),
                map_stats=map_stats,
                query_text="",
                query_token_status="pre_resolved",
                query_token_positions=query_position_list,
                selector_config=config,
                target_visual_tokens=event.budget,
            )
            selected = [int(value) for value in selection.selected_indices]
            self._timing_sync()
            selector_latency = time.perf_counter() - selector_started
            self.timing_stage_components.append(
                {
                    "query_scoring_latency_s": score_latency,
                    "selector_latency_s": selector_latency,
                }
            )
            return (
                self._validate_original_indices(
                    selected,
                    incoming=incoming_original_indices,
                    budget=event.budget,
                ),
                score_checksum,
                True,
                None,
            )
        return super().select_stage_indices(
            hidden_states,
            input_ids=input_ids,
            original_positions=original_positions,
            incoming_original_indices=incoming_original_indices,
            event=event,
            stage_index=stage_index,
            sample_id=sample_id,
            seed=seed,
            query_original_positions=query_original_positions,
        )

    def _merge_after_selection(self, *args: Any, **kwargs: Any) -> Any:
        gather_before = self.timing_gather_latency_s
        self._timing_sync()
        started = time.perf_counter()
        result = super()._merge_after_selection(*args, **kwargs)
        self._timing_sync()
        elapsed = time.perf_counter() - started
        nested_gather = self.timing_gather_latency_s - gather_before
        self.timing_merge_latency_s += max(0.0, elapsed - nested_gather)
        return result


class TimedProgressiveForwarder(HardwareTimingSelectionMixin, ProgressivePruneForwarder):
    pass


class TimedFusedNativeForwarder(
    HardwareTimingSelectionMixin,
    FusedNativeProgressiveForwarder,
):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--mode", choices=["natural", "fixed32"], required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--order-position", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--warmup-examples", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if sha256_file(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise HardwareTimingError("Final timing requires the frozen 1,000-row manifest")
    plan = ORDER_PLAN if args.mode == "natural" else FIXED32_ORDER_PLAN
    if args.repetition not in plan:
        raise HardwareTimingError("Timing repetition is outside the registered plan")
    expected_order = plan[args.repetition]
    if args.method not in expected_order:
        raise HardwareTimingError("Method is not registered for this timing mode")
    if args.order_position != expected_order.index(args.method) + 1:
        raise HardwareTimingError("Method order position differs from the registered plan")
    if args.mode == "fixed32" and args.method not in FIXED32_METHODS:
        raise HardwareTimingError("Fixed32 is restricted to K12, M448_OPT and Z_M448_0")
    if args.smoke:
        if not 1 <= args.limit <= 5 or not 0 <= args.warmup_examples <= 5:
            raise HardwareTimingError("Timing smoke limit/warmup must be in [1, 5]")
    elif args.limit != 1000 or args.warmup_examples != 20:
        raise HardwareTimingError("Final timing requires 1,000 rows and exactly 20 warmups")


def build_config(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    validate_args(args)
    source = checked_source()
    selection = json.loads(M448_OPT_SELECTION.read_text(encoding="utf-8"))
    quality = json.loads(QUALITY_RESULTS.read_text(encoding="utf-8"))
    if selection.get("m448_opt") != "R_FUSED_NATIVE":
        raise HardwareTimingError("M448_OPT selection differs from R_FUSED_NATIVE")
    if not all(quality["quality_gates"][cell]["passed"] for cell in ("Z_K12_0", "Z_M448_0")):
        raise HardwareTimingError("Both registered zero-tail methods must pass quality before timing")
    semantic = {
        "program": "scripts/run_hardware_matched_timing_trial.py",
        "method": args.method,
        "mode": args.mode,
        "repetition": args.repetition,
        "order_position": args.order_position,
        "attempt": args.attempt,
        "limit": args.limit,
        "warmup_examples": args.warmup_examples,
        "smoke": args.smoke,
    }
    return {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "scope": "final_timing_smoke" if args.smoke else "final_matched_timing",
        **semantic,
        "fixed_generated_tokens": FIXED_GENERATED_TOKENS if args.mode == "fixed32" else None,
        "schedule": [event.as_dict() for event in SCHEDULES.get(args.method, ())],
        "runtime_implementation": (
            "fused_native" if args.method in {"M448_OPT", "Z_M448_0"} else "original"
        ),
        "m448_opt_selection": relative(M448_OPT_SELECTION),
        "m448_opt_selection_sha256": sha256_file(M448_OPT_SELECTION),
        "quality_results": relative(QUALITY_RESULTS),
        "quality_results_sha256": sha256_file(QUALITY_RESULTS),
        "model/name": MODEL,
        "model/precision": "bf16",
        "model/decoding": "deterministic_greedy",
        "model/max_new_tokens": FIXED_GENERATED_TOKENS if args.mode == "fixed32" else MAX_NEW_TOKENS,
        "fixed_continuation_token_ids": (
            list(FIXED_CONTINUATION_TOKEN_IDS) if args.mode == "fixed32" else None
        ),
        "dataset/manifest": relative(args.manifest),
        "dataset/manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "dataset/split": "val",
        "prompt/template": REGISTERED_PROMPT,
        "source/git_commit": source["git_commit"],
        "source/git_dirty": False,
        "source/sha256": source["source_sha256"],
        "command/semantic": semantic,
        "command/sha256": hashlib.sha256(
            json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "run/output_dir": relative(run_dir),
        "torch_profiler_capture": (
            args.mode == "natural" and args.repetition == 1 and not args.smoke
        ),
    }


class Trial:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.torch, self.processor, self.model, self.device, self.model_env = load_model(MODEL, "bf16")
        self.image_token_id = int(getattr(self.model.config, "image_token_index", 32000))
        method = config["method"]
        if method == "K12_ORIGINAL":
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
            self.forwarder: Any = MidLayerUGPruneForwarder(
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
        else:
            forwarder_class = (
                TimedFusedNativeForwarder
                if method in {"M448_OPT", "Z_M448_0"}
                else TimedProgressiveForwarder
            )
            self.forwarder = forwarder_class(
                self.torch,
                self.model,
                image_token_id=self.image_token_id,
                schedule=SCHEDULES[method],
                diagnostics_mode="benchmark",
                synchronise_subtimings=True,
            )

    def generation_contract(self) -> tuple[int, set[int]]:
        return MAX_NEW_TOKENS, eos_token_ids(self.model, self.processor)

    def _fixed_token(self, token_id: int, input_ids: Any) -> Any:
        return self.torch.tensor(
            [[int(token_id)]],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

    def fixed_k12_generation(
        self,
        *,
        inputs: dict[str, Any],
        sample: dict[str, Any],
        resolution: Any,
    ) -> tuple[Any, Any, dict[str, Any]]:
        torch = self.torch
        torch.cuda.synchronize()
        prefill_started = time.perf_counter()
        with torch.inference_mode():
            prefill = self.forwarder.prefill_with_layer_aware_cache(
                inputs_embeds=inputs["inputs_embeds"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                sample_id=str(sample["sample_id"]),
                original_tokens=ORIGINAL_VISUAL_TOKENS,
                seed=0,
                query_text=str(sample["query"]),
                query_token_status=resolution.resolver_status,
                query_token_positions=resolution.token_positions,
            )
        torch.cuda.synchronize()
        prefill_latency = time.perf_counter() - prefill_started
        state = prefill.state
        torch.cuda.synchronize()
        decode_started = time.perf_counter()
        with torch.inference_mode():
            for token_id in FIXED_CONTINUATION_TOKEN_IDS:
                self.forwarder.decode_one_with_layer_aware_cache(
                    state,
                    self._fixed_token(token_id, inputs["input_ids"]),
                )
        torch.cuda.synchronize()
        decode_latency = time.perf_counter() - decode_started
        final_early = cache_tensor_bytes(state.early_cache)
        final_late = cache_tensor_bytes(state.late_cache)
        diagnostics = {
            **prefill.diagnostics,
            **self.forwarder._base_cache_diagnostics(state),
            "llm_prefill_latency_s": prefill_latency,
            "prefill_latency_s": prefill_latency,
            "time_to_first_token_s": prefill_latency,
            "decode_latency_s": decode_latency,
            "cache/final_early_bank_bytes": final_early,
            "cache/final_late_bank_bytes": final_late,
            "cache/final_total_bytes": final_early + final_late,
        }
        generated = torch.tensor(
            FIXED_CONTINUATION_TOKEN_IDS,
            dtype=inputs["input_ids"].dtype,
            device=inputs["input_ids"].device,
        )
        return generated, state.selection, diagnostics

    def fixed_progressive_generation(
        self,
        *,
        inputs: dict[str, Any],
        sample: dict[str, Any],
        resolution: Any,
    ) -> tuple[Any, Any, dict[str, Any]]:
        torch = self.torch
        torch.cuda.synchronize()
        prefill_started = time.perf_counter()
        with torch.inference_mode():
            prefill = self.forwarder.prefill_with_segment_caches(
                inputs_embeds=inputs["inputs_embeds"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                sample_id=str(sample["sample_id"]),
                seed=0,
                query_original_positions=resolution.token_positions,
            )
        torch.cuda.synchronize()
        prefill_latency = time.perf_counter() - prefill_started
        state = prefill.state
        torch.cuda.synchronize()
        decode_started = time.perf_counter()
        with torch.inference_mode():
            for token_id in FIXED_CONTINUATION_TOKEN_IDS:
                self.forwarder.decode_one_with_segment_caches(
                    state,
                    self._fixed_token(token_id, inputs["input_ids"]),
                )
        torch.cuda.synchronize()
        decode_latency = time.perf_counter() - decode_started
        final_state = self.forwarder._diagnostics(state)
        final_segment_bytes = list(final_state["cache/prompt_segment_bytes"])
        diagnostics = {
            **prefill.diagnostics,
            **final_state,
            "cache/prompt_segment_bytes": prefill.diagnostics["cache/prompt_segment_bytes"],
            "cache/prompt_total_bytes": prefill.diagnostics["cache/prompt_total_bytes"],
            "cache/final_segment_bytes": final_segment_bytes,
            "cache/final_total_bytes": sum(final_segment_bytes),
            "llm_prefill_latency_s": prefill_latency,
            "prefill_latency_s": prefill_latency,
            "time_to_first_token_s": prefill_latency,
            "decode_latency_s": decode_latency,
        }
        generated = torch.tensor(
            FIXED_CONTINUATION_TOKEN_IDS,
            dtype=inputs["input_ids"].dtype,
            device=inputs["input_ids"].device,
        )
        return generated, state, diagnostics

    def infer(self, sample: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        _, images = load_sample_images(sample)
        if len(images) != 1:
            raise HardwareTimingError("Final timing requires exactly one image")
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
        max_tokens, eos_ids = self.generation_contract()
        method = self.config["method"]
        if method == "K12_ORIGINAL":
            if self.config["mode"] == "fixed32":
                generated_ids, selection, diagnostics = self.fixed_k12_generation(
                    inputs=inputs,
                    sample=sample,
                    resolution=resolution,
                )
                raw_output = self.processor.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()
                generated_count = len(FIXED_CONTINUATION_TOKEN_IDS)
            else:
                raw_output, generated_count, _, selection, diagnostics = generate_mid_layer_answer(
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
            retained_indices = list(selection.selected_indices)
            stage_traces: list[dict[str, Any]] = []
            component_scope = "k12_original_map_rank_and_prefill_residual"
        else:
            self.forwarder.reset_timing_components()
            if self.config["mode"] == "fixed32":
                generated_ids, state, diagnostics = self.fixed_progressive_generation(
                    inputs=inputs,
                    sample=sample,
                    resolution=resolution,
                )
                raw_output = self.processor.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()
                generated_count = len(FIXED_CONTINUATION_TOKEN_IDS)
                retained_indices = list(state.final_retained_indices)
                stage_traces = [trace.as_dict() for trace in state.stage_traces]
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
                raw_output = self.processor.decode(
                    generation.generated_ids, skip_special_tokens=True
                ).strip()
                generated_count = int(generation.generated_ids.numel())
                retained_indices = list(generation.final_retained_indices)
                stage_traces = list(generation.stage_traces)
            query_scoring = sum(
                item["query_scoring_latency_s"] for item in self.forwarder.timing_stage_components
            )
            selector = sum(
                item["selector_latency_s"] for item in self.forwarder.timing_stage_components
            )
            gather = float(self.forwarder.timing_gather_latency_s)
            merge = float(self.forwarder.timing_merge_latency_s)
            prefill = float(diagnostics["llm_prefill_latency_s"])
            trace = list(diagnostics["progressive/realized_visual_tokens_by_layer"])
            prompt_kv = int(diagnostics["cache/prompt_total_bytes"])
            final_kv = int(diagnostics["cache/final_total_bytes"])
            component_scope = "synchronised_score_rank_gather_merge"
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_to_end = time.perf_counter() - end_started
        decode = float(diagnostics["decode_latency_s"])
        result = {
            "sample_id": str(sample["sample_id"]),
            "method": method,
            "repetition": self.config["repetition"],
            "order_position": self.config["order_position"],
            "mode": self.config["mode"],
            "raw_model_output": raw_output,
            "generated_token_count": int(generated_count),
            "vision_encoder_latency_s": float(prep["vision_encoder_latency_s"]),
            "projector_latency_s": float(prep["projector_latency_s"]),
            "query_scoring_latency_s": query_scoring,
            "selector_latency_s": selector,
            "gather_latency_s": gather,
            "merge_latency_s": merge,
            "route_gate_latency_s": 0.0,
            "component_timing_scope": component_scope,
            "main_prefill_with_overhead_latency_s": prefill,
            "time_to_first_token_s": (
                float(prep["vision_encoder_latency_s"])
                + float(prep["projector_latency_s"])
                + prefill
            ),
            "decode_latency_s": decode,
            "decode_step_count": (
                len(FIXED_CONTINUATION_TOKEN_IDS)
                if self.config["mode"] == "fixed32"
                else max(int(generated_count) - 1, 1)
            ),
            "decode_time_per_generated_token_s": decode
            / (
                len(FIXED_CONTINUATION_TOKEN_IDS)
                if self.config["mode"] == "fixed32"
                else max(int(generated_count) - 1, 1)
            ),
            "end_to_end_latency_s": end_to_end,
            "prompt_kv_total_bytes": prompt_kv,
            "final_kv_total_bytes": final_kv,
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
            "realized_visual_tokens_by_layer": trace,
            "visual_token_layer_work": int(sum(trace)),
            "average_visual_tokens": sum(trace) / 32,
            "retained_indices": retained_indices,
            "stage_traces": stage_traces,
            "physical_routes_executed": 1,
            "fixed_continuation_token_ids": (
                list(FIXED_CONTINUATION_TOKEN_IDS)
                if self.config["mode"] == "fixed32"
                else None
            ),
        }
        for image in images:
            image.close()
        del inputs, prep, resolution
        gc.collect()
        return result


def capture_profiler(trial: Trial, sample: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    torch = trial.torch
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
        trial.infer(sample)
    torch.cuda.synchronize()
    trace_path = run_dir / "torch_profiler_trace.json"
    profile.export_chrome_trace(str(trace_path))
    events = list(profile.events())
    cuda_events = [
        event
        for event in events
        if str(getattr(event, "device_type", "")).lower().endswith("cuda")
    ]
    summary = {
        "event_count": len(events),
        "cuda_kernel_event_count": len(cuda_events),
        "self_cpu_time_total_us": float(sum(float(event.self_cpu_time_total) for event in events)),
        "self_cuda_time_total_us": float(sum(float(getattr(event, "self_cuda_time_total", 0.0)) for event in events)),
        "trace_sha256": sha256_file(trace_path),
    }
    atomic_write(run_dir / "torch_profiler_summary.json", canonical_bytes(summary))
    return summary


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    rows = load_manifest(args.manifest, None)
    if len(rows) != 1000:
        raise HardwareTimingError("Frozen timing manifest must contain 1,000 rows")
    mode_root = "smokes" if args.smoke else args.mode
    run_dir = args.output_root / mode_root / (
        f"rep{args.repetition}_pos{args.order_position}_{args.method.lower()}_attempt{args.attempt}"
    )
    if run_dir.exists():
        raise HardwareTimingError(f"Timing trial directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    config = build_config(args, run_dir.resolve())
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
    if config["torch_profiler_capture"]:
        environment["torch_profiler"] = capture_profiler(trial, rows[0], run_dir)
    atomic_write(run_dir / "environment.json", canonical_bytes(jsonable(environment)))
    measured: list[dict[str, Any]] = []
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
                    f"FINAL_TIMING {args.mode} rep{args.repetition} pos{args.order_position} "
                    f"{args.method}: {index}/{args.limit}",
                    flush=True,
                )
    metrics = aggregate(measured)
    metrics["time_to_first_token_s"] = {
        "mean": sum(row["time_to_first_token_s"] for row in measured) / len(measured)
    }
    atomic_write(run_dir / "metrics.json", canonical_bytes(metrics))
    from validate_hardware_matched_timing_trial import validate_and_finalize

    report = validate_and_finalize(run_dir, args.manifest, expected_rows=args.limit)
    print(json.dumps({"status": "complete", "metrics": report["metrics"]}, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except (HardwareTimingError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"UNEXPECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
