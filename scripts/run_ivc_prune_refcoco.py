#!/usr/bin/env python3
"""Run pinned official IVC-Prune or its same-stack Full control on RefCOCO."""

from __future__ import annotations

import argparse
import functools
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

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for entry in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from run_baseline import apply_chat_template, box_iou, parse_box
from run_official_baseline import load_manifest

from vlm_pruning.artifacts import sha256_file, source_provenance
from vlm_pruning.ivc_campaign import (
    FIXED_CONTINUATION_TOKEN_IDS,
    IMAGE_TOKEN_START_INDEX,
    IVC_PRUNE_LAYER_INDEX,
    IVC_TRANSFORMERS_VERSION,
    IVC_UPSTREAM_COMMIT,
    MAX_NEW_TOKENS,
    METHODS,
    MODEL_REVISION,
    ORIGINAL_VISUAL_TOKENS,
    REGISTERED_PROMPT,
    IVCCampaignError,
    expected_ivc_config,
    retained_visual_indices,
    timing_order,
    validate_visual_trace,
)

CAMPAIGN = REPO_ROOT / "experiments/ivc_prune_refcoco_val_20260825"
QUALITY_MANIFEST = REPO_ROOT / "data/manifests/final_evaluation/refcoco_val_unique.jsonl"
TIMING_MANIFEST = (
    REPO_ROOT
    / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
)
QUALITY_SHA256 = "f26d7b954189e86f55b50f9506bb670d3a0cf847d3152779af6c08dbf094e3d7"
TIMING_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
SOURCE_FILES = (
    Path("scripts/run_ivc_prune_refcoco.py"),
    Path("vlm_pruning/ivc_campaign.py"),
    Path("third_party_patches/ivc_prune_remove_debug_print.patch"),
)


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


def tensor_bytes(value: Any) -> int:
    seen: set[int] = set()

    def visit(item: Any) -> int:
        if item is None:
            return 0
        if hasattr(item, "data_ptr") and hasattr(item, "numel") and hasattr(item, "element_size"):
            pointer = int(item.data_ptr())
            if pointer in seen:
                return 0
            seen.add(pointer)
            return int(item.numel()) * int(item.element_size())
        if isinstance(item, dict):
            return sum(visit(entry) for entry in item.values())
        if isinstance(item, (tuple, list)):
            return sum(visit(entry) for entry in item)
        if hasattr(item, "layers"):
            return visit(item.layers)
        if hasattr(item, "key_cache") or hasattr(item, "value_cache"):
            return visit(getattr(item, "key_cache", None)) + visit(
                getattr(item, "value_cache", None)
            )
        return 0

    return visit(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--mode", choices=("smoke", "quality", "timing"), required=True)
    parser.add_argument("--generation-mode", choices=("natural", "fixed32"), default="natural")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup-examples", type=int, default=0)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--order-position", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, str, int]:
    manifest = args.manifest.resolve()
    digest = sha256_file(manifest)
    if args.mode == "quality":
        if manifest != QUALITY_MANIFEST.resolve() or digest != QUALITY_SHA256 or args.limit is not None:
            raise IVCCampaignError("Quality requires the exact complete 8,811-row validation manifest")
        if args.generation_mode != "natural" or args.warmup_examples != 0:
            raise IVCCampaignError("Quality permits natural generation with no warmup only")
        expected_rows = 8811
    elif args.mode == "timing":
        if manifest != TIMING_MANIFEST.resolve() or digest != TIMING_SHA256:
            raise IVCCampaignError("Timing requires the exact frozen 1,000-row subset")
        if args.limit not in (None, 1000) or args.warmup_examples != 20:
            raise IVCCampaignError("Accepted timing requires 1,000 rows and exactly 20 warmups")
        order = timing_order(args.generation_mode, args.repetition)
        if args.order_position != order.index(args.method) + 1:
            raise IVCCampaignError("Timing order differs from the frozen counterbalance")
        expected_rows = 1000
    else:
        if manifest != TIMING_MANIFEST.resolve() or digest != TIMING_SHA256:
            raise IVCCampaignError("Smoke requires the frozen timing subset")
        if args.limit is None or not 1 <= args.limit <= 20:
            raise IVCCampaignError("Smoke requires --limit in [1, 20]")
        if not 0 <= args.warmup_examples <= 5:
            raise IVCCampaignError("Smoke warmup count must be in [0, 5]")
        expected_rows = int(args.limit)
    return manifest, digest, expected_rows


def verify_sources(source_root: Path) -> dict[str, Any]:
    source = source_root.resolve()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != IVC_UPSTREAM_COMMIT:
        raise IVCCampaignError("IVC source checkout commit differs")
    changed = subprocess.check_output(
        ["git", "diff", "--name-only"], cwd=source, text=True
    ).strip().splitlines()
    expected = ["transformers/src/transformers/models/llava/modeling_llava.py"]
    if changed != expected:
        raise IVCCampaignError(f"IVC source changes differ from the instrumentation patch: {changed}")
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=source, text=True
    ).strip().splitlines()
    if untracked:
        raise IVCCampaignError(f"IVC source contains unregistered files: {untracked}")
    patch = REPO_ROOT / "third_party_patches/ivc_prune_remove_debug_print.patch"
    reverse = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=source,
        check=False,
        capture_output=True,
    )
    if reverse.returncode:
        raise IVCCampaignError("Registered IVC debug-print patch is not active")
    imported = Path(__import__("transformers").__file__).resolve()
    expected_import = (source / "transformers/src").resolve()
    if expected_import not in imported.parents:
        raise IVCCampaignError(
            f"Transformers import is not from the pinned IVC source: {imported}"
        )
    if __import__("transformers").__version__ != IVC_TRANSFORMERS_VERSION:
        raise IVCCampaignError("Imported IVC Transformers version differs")
    return {
        "source_root": str(source),
        "upstream_commit": IVC_UPSTREAM_COMMIT,
        "transformers_file": str(imported),
        "transformers_version": __import__("transformers").__version__,
        "changed_files": changed,
        "untracked_files": untracked,
    }


def verify_frozen_campaign(source_root: Path, model_snapshot: Path) -> dict[str, Any]:
    protocol_path = CAMPAIGN / "PROTOCOL.json"
    source_lock_path = CAMPAIGN / "SOURCE_LOCK.json"
    if not protocol_path.is_file() or not source_lock_path.is_file():
        raise IVCCampaignError("Source lock and frozen protocol must exist before execution")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "PROTOCOL_FROZEN_NO_ACCEPTED_GPU_RUNS_YET":
        raise IVCCampaignError("Campaign protocol has an unexpected status")
    if source_lock.get("status") != "SOURCE_LOCKED":
        raise IVCCampaignError("IVC source lock has an unexpected status")
    if protocol.get("source", {}).get("source_lock_sha256") != sha256_file(source_lock_path):
        raise IVCCampaignError("Protocol does not bind the active source lock")
    if source_lock.get("upstream_commit") != IVC_UPSTREAM_COMMIT:
        raise IVCCampaignError("Source lock commit differs")
    if Path(str(source_lock.get("checkout"))).resolve() != source_root.resolve():
        raise IVCCampaignError("Source lock checkout differs from the requested source")
    if Path(str(protocol.get("model", {}).get("snapshot"))).resolve() != model_snapshot.resolve():
        raise IVCCampaignError("Protocol model snapshot differs from the requested model")
    actual_diff = subprocess.check_output(["git", "diff", "--binary"], cwd=source_root.resolve())
    if hashlib.sha256(actual_diff).hexdigest() != source_lock.get("post_patch_diff_sha256"):
        raise IVCCampaignError("Active IVC source diff differs from the locked source")
    return {
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "source_lock_path": str(source_lock_path),
        "source_lock_sha256": sha256_file(source_lock_path),
    }


def checked_wrapper_source() -> dict[str, Any]:
    relative = [str(path).replace("\\", "/") for path in SOURCE_FILES]
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *relative], cwd=REPO_ROOT, text=True
    ).strip()
    if status:
        raise IVCCampaignError("IVC runner requires clean committed wrapper source")
    for path in relative:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", path],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
        )
        if tracked.returncode:
            raise IVCCampaignError(f"IVC executable source is not committed: {path}")
    return source_provenance(REPO_ROOT, SOURCE_FILES)


class ForcedContinuation:
    """A Transformers logits processor that deterministically emits the fixed32 sequence."""

    def __init__(self, token_ids: tuple[int, ...], prompt_length: int) -> None:
        self.token_ids = token_ids
        self.prompt_length = int(prompt_length)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        step = int(input_ids.shape[1]) - self.prompt_length
        if not 0 <= step < len(self.token_ids):
            raise IVCCampaignError(f"Unexpected fixed32 generation step: {step}")
        forced = scores.new_full(scores.shape, float("-inf"))
        forced[:, self.token_ids[step]] = 0.0
        return forced


class IVCTrial:
    """Official IVC LLaVA execution with measurement-only instrumentation."""

    def __init__(self, method: str, model_snapshot: Path) -> None:
        import torch
        import transformers
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        if not torch.cuda.is_available():
            raise IVCCampaignError("CUDA is required; this runner must not be used on the CPU host")
        self.torch = torch
        self.transformers = transformers
        self.method = method
        self.device = torch.device("cuda:0")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        model_path = str(model_snapshot.resolve())
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            ivcp_config=expected_ivc_config(method),
        ).eval().to(self.device)
        self._install_component_instrumentation()

    def sync(self) -> None:
        self.torch.cuda.synchronize(self.device)

    def _timed_module(self, module: Any, field: str) -> None:
        original = module.forward

        @functools.wraps(original)
        def timed(*args: Any, **kwargs: Any) -> Any:
            self.sync()
            started = time.perf_counter()
            output = original(*args, **kwargs)
            self.sync()
            setattr(self, field, getattr(self, field) + time.perf_counter() - started)
            return output

        module.forward = timed

    def _install_component_instrumentation(self) -> None:
        self._timed_module(self.model.vision_tower, "vision_encoder_latency_s")
        self._timed_module(self.model.multi_modal_projector, "projector_latency_s")
        language_model = self.model.language_model
        llama = self.model.language_model.model
        if self.method == "IVC_PRUNE":
            language_attribute = "ivcp_forward"
            original_scoring = llama.compute_attention_weights

            @functools.wraps(original_scoring)
            def timed_scoring(*args: Any, **kwargs: Any) -> Any:
                self.sync()
                started = time.perf_counter()
                output = original_scoring(*args, **kwargs)
                self.sync()
                self.query_scoring_latency_s += time.perf_counter() - started
                return output

            llama.compute_attention_weights = timed_scoring
        else:
            language_attribute = "forward"
        original_language = getattr(language_model, language_attribute)

        @functools.wraps(original_language)
        def timed_language(*args: Any, **kwargs: Any) -> Any:
            embeds = kwargs.get("inputs_embeds")
            ids = kwargs.get("input_ids")
            length = int(embeds.shape[1]) if embeds is not None else int(ids.shape[1])
            self.sync()
            started = time.perf_counter()
            output = original_language(*args, **kwargs)
            self.sync()
            elapsed = time.perf_counter() - started
            if length > 1:
                self.prefill_llm_latency_s += elapsed
            else:
                self.decode_llm_latency_s += elapsed
            self.llm_total_latency_s += elapsed
            return output

        setattr(language_model, language_attribute, timed_language)

    def reset(self) -> None:
        self.vision_encoder_latency_s = 0.0
        self.projector_latency_s = 0.0
        self.llm_total_latency_s = 0.0
        self.query_scoring_latency_s = 0.0
        self.selector_materialization_latency_s = 0.0
        self.trace_totals: dict[int, int] = {}
        self.prompt_kv_total_bytes = 0
        self.final_kv_total_bytes = 0
        self.prefill_llm_latency_s = 0.0
        self.decode_llm_latency_s = 0.0
        self._selector_started: float | None = None

    def _trace_hooks(self) -> list[Any]:
        hooks = []
        for index, layer in enumerate(self.model.language_model.model.layers):
            def pre_hook(module: Any, args: tuple[Any, ...], index: int = index) -> None:
                del module
                if args and int(args[0].shape[1]) > 1 and index not in self.trace_totals:
                    if index == IVC_PRUNE_LAYER_INDEX and self._selector_started is not None:
                        self.sync()
                        self.selector_materialization_latency_s += time.perf_counter() - self._selector_started
                        self._selector_started = None
                    self.trace_totals[index] = int(args[0].shape[1])

            hooks.append(layer.register_forward_pre_hook(pre_hook))
            if index == IVC_PRUNE_LAYER_INDEX - 1:
                def post_hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
                    del module, args, output
                    if self.method == "IVC_PRUNE" and self._selector_started is None:
                        self.sync()
                        self._selector_started = time.perf_counter()

                hooks.append(layer.register_forward_hook(post_hook))
        return hooks

    def _model_forward_hook(self) -> Any:
        def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            del module, args
            cache_bytes = tensor_bytes(getattr(output, "past_key_values", None))
            prefill = kwargs.get("pixel_values") is not None or (
                kwargs.get("input_ids") is not None and int(kwargs["input_ids"].shape[1]) > 1
            )
            if prefill:
                self.prompt_kv_total_bytes = cache_bytes
            else:
                self.final_kv_total_bytes = cache_bytes

        return self.model.register_forward_hook(hook, with_kwargs=True)

    def _prepare(self, sample: dict[str, Any]) -> tuple[dict[str, Any], Image.Image, int]:
        prompt = REGISTERED_PROMPT.format(query=str(sample["query"]))
        raw_path = Path(str(sample["image_path"]))
        image_path = raw_path if raw_path.is_absolute() else REPO_ROOT / raw_path
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        inputs = self.processor(
            images=image,
            text=apply_chat_template(self.processor, prompt),
            return_tensors="pt",
        )
        dtype = next(self.model.parameters()).dtype
        moved = {
            key: (
                value.to(device=self.device, dtype=dtype)
                if key == "pixel_values"
                else value.to(device=self.device)
            )
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }
        image_token = int(self.model.config.image_token_index)
        positions = self.torch.nonzero(moved["input_ids"][0] == image_token, as_tuple=False).flatten()
        if len(positions) != ORIGINAL_VISUAL_TOKENS:
            raise IVCCampaignError(
                f"Processor produced {len(positions)} image tokens, expected {ORIGINAL_VISUAL_TOKENS}"
            )
        image_start = int(positions[0].item())
        expected_positions = self.torch.arange(
            image_start, image_start + ORIGINAL_VISUAL_TOKENS, device=positions.device
        )
        if not self.torch.equal(positions, expected_positions):
            raise IVCCampaignError("Processor image-token block is not contiguous")
        if image_start != IMAGE_TOKEN_START_INDEX:
            raise IVCCampaignError(
                f"Registered IVC image start is {IMAGE_TOKEN_START_INDEX}, observed {image_start}; "
                "stop instead of silently changing the published configuration"
            )
        return moved, image, image_start

    def infer(self, sample: dict[str, Any], generation_mode: str) -> dict[str, Any]:
        from transformers import LogitsProcessorList

        self.reset()
        inputs, image, image_start = self._prepare(sample)
        prompt_length = int(inputs["input_ids"].shape[1])
        text_tokens = prompt_length - ORIGINAL_VISUAL_TOKENS
        hooks = self._trace_hooks()
        model_hook = self._model_forward_hook()
        generation: dict[str, Any] = {
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "return_dict_in_generate": True,
        }
        if generation_mode == "natural":
            generation["max_new_tokens"] = MAX_NEW_TOKENS
        elif generation_mode == "fixed32":
            generation.update(
                {
                    "max_new_tokens": len(FIXED_CONTINUATION_TOKEN_IDS),
                    "min_new_tokens": len(FIXED_CONTINUATION_TOKEN_IDS),
                    "eos_token_id": None,
                    "logits_processor": LogitsProcessorList(
                        [ForcedContinuation(FIXED_CONTINUATION_TOKEN_IDS, prompt_length)]
                    ),
                }
            )
        else:
            raise IVCCampaignError(f"Unregistered generation mode: {generation_mode}")
        self.sync()
        self.torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        self.sync()
        end_to_end = time.perf_counter() - started
        for hook in hooks:
            hook.remove()
        model_hook.remove()
        sequences = output.sequences
        generated_ids = sequences[:, prompt_length:]
        generated = int(generated_ids.shape[1])
        if generation_mode == "fixed32":
            observed = tuple(int(value) for value in generated_ids[0].tolist())
            if observed != FIXED_CONTINUATION_TOKEN_IDS:
                raise IVCCampaignError("Fixed32 continuation differs from the frozen token sequence")
        raw_output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        visual_trace = [self.trace_totals[index] - text_tokens for index in range(32)]
        retained_count = validate_visual_trace(self.method, visual_trace)
        if self.method == "IVC_PRUNE":
            keep = getattr(self.model.language_model.model, "keep_indexs", None)
            if keep is None:
                raise IVCCampaignError("Official IVC implementation did not expose keep_indexs")
            visual_indices = retained_visual_indices(keep.tolist(), image_start=image_start)
            if len(visual_indices) != retained_count:
                raise IVCCampaignError("IVC keep_indexs and physical layer trace disagree")
        else:
            visual_indices = list(range(ORIGINAL_VISUAL_TOKENS))
        raw_box, predicted_box, parse_failure = parse_box(
            raw_output, int(sample["width"]), int(sample["height"]), "normalized_0_1"
        )
        iou = box_iou(predicted_box, sample["gt_box"]) if predicted_box is not None else 0.0
        selector = self.query_scoring_latency_s + self.selector_materialization_latency_s
        llm_excluding_selector = max(0.0, self.llm_total_latency_s - selector)
        prefill_excluding_selector = max(0.0, self.prefill_llm_latency_s - selector)
        image.close()
        return {
            "sample_id": str(sample["sample_id"]),
            "image_path": str(sample["image_path"]),
            "query": str(sample["query"]),
            "width": int(sample["width"]),
            "height": int(sample["height"]),
            "gt_box": list(sample["gt_box"]),
            "method": self.method,
            "generation_mode": generation_mode,
            "raw_model_output": raw_output,
            "raw_parsed_box": raw_box,
            "predicted_box": predicted_box,
            "parse_failure_reason": parse_failure,
            "valid": predicted_box is not None,
            "box_iou": float(iou),
            "success_iou_0_5": bool(predicted_box is not None and iou >= 0.5),
            "generated_token_count": generated,
            "prompt_text_token_count": text_tokens,
            "image_token_start_index": image_start,
            "retained_visual_indices": visual_indices,
            "realized_visual_tokens_by_layer": visual_trace,
            "visual_token_layer_work": sum(visual_trace),
            "average_visual_tokens": sum(visual_trace) / 32,
            "final_visual_tokens": retained_count,
            "vision_encoder_latency_s": self.vision_encoder_latency_s,
            "projector_latency_s": self.projector_latency_s,
            "query_scoring_latency_s": self.query_scoring_latency_s,
            "selector_materialization_latency_s": self.selector_materialization_latency_s,
            "selector_latency_s": selector,
            "llm_total_latency_s": self.llm_total_latency_s,
            "llm_prefill_with_overhead_latency_s": self.prefill_llm_latency_s,
            "decode_latency_s": self.decode_llm_latency_s,
            "decode_time_per_generated_token_s": self.decode_llm_latency_s / max(generated - 1, 1),
            "llm_excluding_selector_latency_s": llm_excluding_selector,
            "llm_prefill_excluding_selector_latency_s": prefill_excluding_selector,
            "time_to_first_token_s": (
                self.vision_encoder_latency_s
                + self.projector_latency_s
                + self.prefill_llm_latency_s
            ),
            "end_to_end_latency_s": end_to_end,
            "throughput_examples_per_s": 1.0 / end_to_end,
            "prompt_kv_total_bytes": self.prompt_kv_total_bytes,
            "final_kv_total_bytes": self.final_kv_total_bytes or self.prompt_kv_total_bytes,
            "peak_allocated_gpu_bytes": int(self.torch.cuda.max_memory_allocated(self.device)),
            "peak_reserved_gpu_bytes": int(self.torch.cuda.max_memory_reserved(self.device)),
        }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scalar = (
        "box_iou",
        "generated_token_count",
        "vision_encoder_latency_s",
        "projector_latency_s",
        "query_scoring_latency_s",
        "selector_latency_s",
        "llm_total_latency_s",
        "llm_prefill_with_overhead_latency_s",
        "decode_latency_s",
        "decode_time_per_generated_token_s",
        "llm_excluding_selector_latency_s",
        "llm_prefill_excluding_selector_latency_s",
        "time_to_first_token_s",
        "end_to_end_latency_s",
        "prompt_kv_total_bytes",
        "final_kv_total_bytes",
        "peak_allocated_gpu_bytes",
        "peak_reserved_gpu_bytes",
        "visual_token_layer_work",
        "average_visual_tokens",
        "final_visual_tokens",
    )
    return {
        "rows": len(rows),
        "unique_ids": len({row["sample_id"] for row in rows}),
        "parse_rate": sum(bool(row["valid"]) for row in rows) / len(rows),
        "acc_at_0_5": sum(bool(row["success_iou_0_5"]) for row in rows) / len(rows),
        "mean_iou": sum(float(row["box_iou"]) for row in rows) / len(rows),
        "median_iou": statistics.median(float(row["box_iou"]) for row in rows),
        "means": {key: sum(float(row[key]) for row in rows) / len(rows) for key in scalar},
        "min_final_visual_tokens": min(int(row["final_visual_tokens"]) for row in rows),
        "max_final_visual_tokens": max(int(row["final_visual_tokens"]) for row in rows),
    }


def run(args: argparse.Namespace) -> int:
    manifest_path, manifest_sha, expected_rows = validate_args(args)
    source = verify_sources(args.source_root)
    model = args.model_snapshot.resolve()
    if MODEL_REVISION not in str(model) or not (model / "config.json").is_file():
        raise IVCCampaignError("Model path is not the frozen author-recommended HF revision")
    frozen_campaign = verify_frozen_campaign(args.source_root, model)
    rows = load_manifest(manifest_path, None)
    complete_manifest_rows = 8811 if args.mode == "quality" else 1000
    if len(rows) != complete_manifest_rows:
        raise IVCCampaignError("Manifest row count differs before applying the smoke limit")
    rows = rows[:expected_rows]
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and not args.resume:
        raise IVCCampaignError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    wrapper = checked_wrapper_source()
    config = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "method": args.method,
        "mode": args.mode,
        "generation_mode": args.generation_mode,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "expected_rows": expected_rows,
        "prompt": REGISTERED_PROMPT,
        "precision": "fp16_native",
        "batch_size": 1,
        "seed": 0,
        "decoding": "greedy",
        "max_new_tokens": MAX_NEW_TOKENS,
        "fixed_continuation_token_ids": (
            list(FIXED_CONTINUATION_TOKEN_IDS) if args.generation_mode == "fixed32" else None
        ),
        "ivcp_config": expected_ivc_config(args.method),
        "model_snapshot": str(model),
        "model_revision": MODEL_REVISION,
        "model_config_sha256": sha256_file(model / "config.json"),
        "source": source,
        "frozen_campaign": frozen_campaign,
        "wrapper_source": wrapper,
        "repetition": args.repetition,
        "order_position": args.order_position,
        "warmup_examples": args.warmup_examples,
    }
    config_path = run_dir / "config.json"
    if args.resume:
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        comparable = dict(config)
        comparable["created_at_utc"] = prior["created_at_utc"]
        if comparable != prior:
            raise IVCCampaignError("Resume configuration differs from the existing run")
    else:
        atomic_write(config_path, canonical_bytes(config))
    config_sha = sha256_file(config_path)
    partial = run_dir / "predictions.partial.jsonl"
    measured: list[dict[str, Any]] = []
    if args.resume and partial.is_file():
        measured = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines() if line.strip()]
        if [row["sample_id"] for row in measured] != [str(row["sample_id"]) for row in rows[: len(measured)]]:
            raise IVCCampaignError("Resume evidence is not an exact manifest prefix")
        if any(row.get("config_sha256") != config_sha for row in measured):
            raise IVCCampaignError("Resume evidence has a different configuration hash")
    trial = IVCTrial(args.method, model)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": trial.torch.__version__,
        "cuda": trial.torch.version.cuda,
        "transformers": trial.transformers.__version__,
        "transformers_file": str(Path(trial.transformers.__file__).resolve()),
        "gpu": trial.torch.cuda.get_device_name(0),
    }
    atomic_write(run_dir / "environment.json", canonical_bytes(environment))
    for sample in rows[: args.warmup_examples]:
        trial.infer(sample, args.generation_mode)
    with partial.open("a" if measured else "w", encoding="utf-8") as handle:
        for index in range(len(measured), len(rows)):
            result = trial.infer(rows[index], args.generation_mode)
            result.update(
                {
                    "manifest_index": index,
                    "config_sha256": config_sha,
                    "repetition": args.repetition,
                    "order_position": args.order_position,
                }
            )
            measured.append(result)
            handle.write(json.dumps(result, sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush()
            if (index + 1) % 25 == 0 or index + 1 == len(rows):
                print(f"IVC {args.method} {args.mode}: {index + 1}/{len(rows)}", flush=True)
    predictions = run_dir / "predictions.jsonl"
    os.replace(partial, predictions)
    metrics = {"status": "complete", "config_sha256": config_sha, **summarize(measured)}
    atomic_write(run_dir / "metrics.json", canonical_bytes(metrics))
    integrity = {
        "status": "PASS",
        "rows": len(measured),
        "ordered_unique_ids": [row["sample_id"] for row in measured]
        == [str(row["sample_id"]) for row in rows],
        "all_traces_valid": all(
            validate_visual_trace(args.method, row["realized_visual_tokens_by_layer"]) > 0
            for row in measured
        ),
        "predictions_sha256": sha256_file(predictions),
        "metrics_sha256": sha256_file(run_dir / "metrics.json"),
    }
    if len(measured) != expected_rows or not integrity["ordered_unique_ids"] or not integrity["all_traces_valid"]:
        raise IVCCampaignError("Completed IVC run failed integrity validation")
    atomic_write(run_dir / "integrity_report.json", canonical_bytes(integrity))
    atomic_write(run_dir / "COMPLETED.json", canonical_bytes({"status": "complete", **integrity}))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (
        IVCCampaignError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
