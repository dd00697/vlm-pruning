#!/usr/bin/env python
"""Run the pinned native Nüwa LLaVA stack on frozen RefCOCO validation data."""

from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
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

from run_baseline import box_iou, parse_box  # noqa: E402
from run_official_baseline import load_manifest  # noqa: E402
from vlm_pruning.artifacts import sha256_file  # noqa: E402


CAMPAIGN_ROOT = REPO_ROOT / "experiments/final_val_efficiency_baselines_20260818"
FULL_MANIFEST = REPO_ROOT / "data/manifests/final_evaluation/refcoco_val_unique.jsonl"
TIMING_MANIFEST = (
    REPO_ROOT
    / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
)
FULL_MANIFEST_SHA256 = "f26d7b954189e86f55b50f9506bb670d3a0cf847d3152779af6c08dbf094e3d7"
TIMING_MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
NUWA_ROOT = (
    REPO_ROOT
    / "experiments/adaptive_efficiency_refcoco_val_20260815/official_sources/checkouts/nuwa"
)
NUWA_COMMIT = "175b7d57a7890b9818dd26d82f2d91d9868b0d80"
LLAVA_ROOT = Path("/workspace/third_party_official/LLaVA")
MODEL_PATH = Path(
    "/workspace/cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b/"
    "snapshots/4481d270cc22fd5c4d1bb5df129622006ccd9234"
)
REGISTERED_PROMPT = (
    'Identify the target described by this referring expression: "{query}".\n'
    "Return exactly one bounding box in normalized 0-1 image coordinates as "
    "[x_min, y_min, x_max, y_max]. Do not include any other text."
)
METHODS = ("NUWA_STACK_FULL", "NUWA_NATIVE_192")
NUWA_TRACE = [336] * 16 + [48] * 16
FULL_TRACE = [576] * 32
FIXED_CONTINUATION_TOKEN_IDS = tuple(range(1000, 1032))
NATURAL_ORDER_PLAN = {
    1: METHODS,
    2: tuple(reversed(METHODS)),
    3: METHODS,
    4: tuple(reversed(METHODS)),
    5: METHODS,
}
FIXED32_ORDER_PLAN = {
    1: METHODS,
    2: tuple(reversed(METHODS)),
    3: METHODS,
}


class NativeNuwaError(RuntimeError):
    """Raised when a native Nüwa run violates the frozen contract."""


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


def git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def git_clean(path: Path) -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=path, text=True
    ).strip()


def checked_wrapper_source() -> dict[str, str]:
    relative = Path("scripts/run_native_nuwa_refcoco.py")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(relative)], cwd=REPO_ROOT, text=True
    ).strip()
    if status:
        raise NativeNuwaError("Native Nüwa wrapper requires clean committed executable source")
    return {
        "git_commit": git_commit(REPO_ROOT),
        "runner_sha256": sha256_file(REPO_ROOT / relative),
    }


def tree_hash(path: Path) -> str:
    """Hash Git's exact tracked blob inventory without rereading every worktree file."""
    inventory = subprocess.check_output(["git", "ls-files", "-s", "-z"], cwd=path)
    return hashlib.sha256(inventory).hexdigest()


def stage1_selector_latency(method: str, outer_latency_s: float, raw_latency_s: float) -> float:
    """Attribute Nüwa's vision-wrapper excess only when its stage-1 selector is active."""
    if method == "NUWA_STACK_FULL":
        return 0.0
    if method == "NUWA_NATIVE_192":
        return max(0.0, outer_latency_s - raw_latency_s)
    raise NativeNuwaError(f"Unregistered Nüwa method: {method}")


def expected_trace(method: str) -> list[int]:
    if method == "NUWA_STACK_FULL":
        return list(FULL_TRACE)
    if method == "NUWA_NATIVE_192":
        return list(NUWA_TRACE)
    raise NativeNuwaError(f"Unregistered Nüwa method: {method}")


def validate_visual_trace(method: str, trace: list[int]) -> None:
    if trace != expected_trace(method):
        raise NativeNuwaError(f"Physical Nüwa trace mismatch for {method}: {trace}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("quality", "smoke", "timing"), required=True)
    parser.add_argument("--generation-mode", choices=("natural", "fixed32"), default="natural")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--warmup-examples", type=int, default=0)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--order-position", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[str, int]:
    manifest = args.manifest.resolve()
    digest = sha256_file(manifest)
    if args.mode == "quality":
        if manifest != FULL_MANIFEST.resolve() or digest != FULL_MANIFEST_SHA256:
            raise NativeNuwaError("Quality requires the exact frozen 8,811-row RefCOCO-val manifest")
        if args.limit is not None:
            raise NativeNuwaError("Full quality does not permit a row limit")
        expected_rows = 8811
        if args.generation_mode != "natural":
            raise NativeNuwaError("Full quality permits natural generation only")
    else:
        if manifest != TIMING_MANIFEST.resolve() or digest != TIMING_MANIFEST_SHA256:
            raise NativeNuwaError("Smoke/timing requires the exact frozen 1,000-row RefCOCO-val manifest")
        if args.mode == "timing" and args.limit not in (None, 1000):
            raise NativeNuwaError("Native timing requires all 1,000 rows")
        if args.mode == "timing" and args.warmup_examples != 20:
            raise NativeNuwaError("Native timing requires exactly 20 warm-up requests")
        if args.mode == "smoke" and (args.limit is None or not 1 <= args.limit <= 20):
            raise NativeNuwaError("Smoke limit must be in [1, 20]")
        expected_rows = int(args.limit or 1000)
    if args.mode == "timing":
        plan = NATURAL_ORDER_PLAN if args.generation_mode == "natural" else FIXED32_ORDER_PLAN
        if args.repetition not in plan:
            raise NativeNuwaError("Repetition is outside the registered Nüwa timing plan")
        if args.order_position != plan[args.repetition].index(args.method) + 1:
            raise NativeNuwaError("Order position differs from the registered Nüwa counterbalance")
    if args.max_new_tokens != 48:
        raise NativeNuwaError("Natural Nüwa runs require the frozen 48-token output cap")
    return digest, expected_rows


def install_clip_shape_compatibility() -> bool:
    """Restore the exact legacy CLIP reshape helper removed by Transformers 4.54."""
    from transformers.models.clip.modeling_clip import CLIPAttention

    if hasattr(CLIPAttention, "_shape"):
        return False

    def _shape(self: Any, tensor: Any, sequence_length: int, batch_size: int) -> Any:
        return (
            tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    CLIPAttention._shape = _shape
    return True


def tensor_bytes(value: Any) -> int:
    if value is None:
        return 0
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
            return visit(getattr(item, "key_cache", None)) + visit(getattr(item, "value_cache", None))
        if hasattr(item, "keys") and hasattr(item, "values"):
            return visit(getattr(item, "keys")) + visit(getattr(item, "values"))
        return 0

    return visit(value)


class NativeTrial:
    """Thin instrumentation around the pinned official LLaVA/Nüwa APIs."""

    def __init__(self, method: str) -> None:
        import torch
        import transformers
        from llava.model.builder import load_pretrained_model

        self.torch = torch
        self.method = method
        self.clip_shape_shim = install_clip_shape_compatibility()
        tokenizer, model, image_processor, _ = load_pretrained_model(
            str(MODEL_PATH), None, "llava-v1.5-7b", device_map="cuda:0"
        )
        if method == "NUWA_NATIVE_192":
            from nuwa import nuwa

            source = Path(inspect.getsourcefile(nuwa) or "").resolve()
            if NUWA_ROOT.resolve() not in source.parents:
                raise NativeNuwaError(f"Nüwa import is outside the pinned checkout: {source}")
            model = nuwa(model, matain_token=192, distance=280)
        else:
            model.model.use_nuwa = False
        model.eval()
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.transformers = transformers
        self._install_forward_compatibility()
        self._install_component_instrumentation()

    def _sync(self) -> None:
        self.torch.cuda.synchronize()

    def _install_forward_compatibility(self) -> None:
        original = self.model.forward

        @functools.wraps(original)
        def compatible_forward(*args: Any, **kwargs: Any) -> Any:
            kwargs.pop("cache_position", None)
            kwargs.pop("logits_to_keep", None)
            kwargs.pop("num_logits_to_keep", None)
            sequence_length = 0
            embeds = kwargs.get("inputs_embeds")
            if embeds is not None:
                sequence_length = int(embeds.shape[1])
            elif args and getattr(args[0], "ndim", 0) >= 2:
                sequence_length = int(args[0].shape[1])
            self._sync()
            started = time.perf_counter()
            output = original(*args, **kwargs)
            self._sync()
            elapsed = time.perf_counter() - started
            if sequence_length > 1:
                self.llm_prefill_latency_s += elapsed
                self.prompt_kv_bytes = tensor_bytes(getattr(output, "past_key_values", None))
            else:
                self.decode_latency_s += elapsed
                self.final_kv_bytes = tensor_bytes(getattr(output, "past_key_values", None))
            return output

        self.model.forward = compatible_forward

    def _timed_module(self, module: Any, field: str) -> None:
        original = module.forward

        @functools.wraps(original)
        def timed(*args: Any, **kwargs: Any) -> Any:
            self._sync()
            started = time.perf_counter()
            result = original(*args, **kwargs)
            self._sync()
            setattr(self, field, getattr(self, field) + time.perf_counter() - started)
            return result

        module.forward = timed

    def _install_component_instrumentation(self) -> None:
        outer_vision = self.model.get_vision_tower()
        self._timed_module(outer_vision.vision_tower, "raw_vision_latency_s")
        self._timed_module(outer_vision, "outer_vision_latency_s")
        self._timed_module(self.model.model.mm_projector, "projector_latency_s")

    def reset(self) -> None:
        self.raw_vision_latency_s = 0.0
        self.outer_vision_latency_s = 0.0
        self.projector_latency_s = 0.0
        self.llm_prefill_latency_s = 0.0
        self.decode_latency_s = 0.0
        self.stage2_selector_latency_s = 0.0
        self.prompt_kv_bytes = 0
        self.final_kv_bytes = 0
        self.trace_totals: dict[int, int] = {}
        self._stage2_started: float | None = None

    def _trace_hooks(self) -> list[Any]:
        hooks = []
        for index, layer in enumerate(self.model.model.layers):
            def pre_hook(module: Any, args: tuple[Any, ...], index: int = index) -> None:
                del module
                if args and int(args[0].shape[1]) > 1:
                    if index == 16 and self._stage2_started is not None:
                        self._sync()
                        self.stage2_selector_latency_s += time.perf_counter() - self._stage2_started
                        self._stage2_started = None
                    self.trace_totals[index] = int(args[0].shape[1])

            hooks.append(layer.register_forward_pre_hook(pre_hook))
            if index == 15:
                def post_hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
                    del module, args, output
                    if self.method == "NUWA_NATIVE_192":
                        self._sync()
                        self._stage2_started = time.perf_counter()

                hooks.append(layer.register_forward_hook(post_hook))
        return hooks

    def infer(self, sample: dict[str, Any], *, generation_mode: str = "natural") -> dict[str, Any]:
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        self.reset()
        prompt = REGISTERED_PROMPT.format(query=str(sample["query"]))
        conversation = conv_templates["llava_v1"].copy()
        conversation.append_message(conversation.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conversation.append_message(conversation.roles[1], None)
        input_ids = tokenizer_image_token(
            conversation.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()
        image_path = REPO_ROOT / str(sample["image_path"])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            image_size = image.size
            image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0].unsqueeze(0)
        image_tensor = image_tensor.to(dtype=self.torch.float16, device="cuda")
        hooks = self._trace_hooks()
        self._sync()
        self.torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with self.torch.inference_mode():
            if generation_mode == "natural":
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=48,
                    use_cache=True,
                )
            elif generation_mode == "fixed32":
                (
                    _,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _,
                ) = self.model.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    None,
                    None,
                    None,
                    None,
                    image_tensor,
                    image_sizes=[image_size],
                )
                output = self.model(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                cache = output.past_key_values
                if attention_mask is None:
                    attention_mask = self.torch.ones(
                        inputs_embeds.shape[:2], dtype=self.torch.bool, device=inputs_embeds.device
                    )
                fixed = []
                for token_id in FIXED_CONTINUATION_TOKEN_IDS:
                    token = self.torch.tensor(
                        [[token_id]], dtype=input_ids.dtype, device=input_ids.device
                    )
                    attention_mask = self.torch.cat(
                        [attention_mask, self.torch.ones_like(token, dtype=attention_mask.dtype)], dim=1
                    )
                    output = self.model(
                        input_ids=token,
                        attention_mask=attention_mask,
                        past_key_values=cache,
                        use_cache=True,
                        return_dict=True,
                    )
                    cache = output.past_key_values
                    fixed.append(token)
                output_ids = self.torch.cat(fixed, dim=1)
            else:
                raise NativeNuwaError(f"Unregistered generation mode: {generation_mode}")
        self._sync()
        end_to_end = time.perf_counter() - started
        for hook in hooks:
            hook.remove()
        text_tokens = int(input_ids.shape[1]) - 1
        visual_trace = [self.trace_totals[index] - text_tokens for index in range(32)]
        validate_visual_trace(self.method, visual_trace)
        if self.method == "NUWA_NATIVE_192":
            keep = getattr(self.model, "keep_idxs", None)
            if keep is None or int(keep.numel()) != 336 or int(self.model.position_offset) != 576:
                raise NativeNuwaError("Nüwa did not expose its registered 336-token stage-1 state")
        raw_output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        raw_box, predicted_box, parse_failure = parse_box(
            raw_output, int(sample["width"]), int(sample["height"]), "normalized_0_1"
        )
        iou = box_iou(predicted_box, sample["gt_box"]) if predicted_box is not None else 0.0
        stage1_selector = stage1_selector_latency(
            self.method, self.outer_vision_latency_s, self.raw_vision_latency_s
        )
        generated = int(output_ids.numel())
        if generation_mode == "fixed32" and generated != len(FIXED_CONTINUATION_TOKEN_IDS):
            raise NativeNuwaError("Fixed32 Nüwa diagnostic did not execute exactly 32 tokens")
        return {
            "sample_id": str(sample["sample_id"]),
            "image_path": str(sample["image_path"]),
            "query": str(sample["query"]),
            "width": int(sample["width"]),
            "height": int(sample["height"]),
            "gt_box": list(sample["gt_box"]),
            "method": self.method,
            "generation_mode": generation_mode,
            "fixed_continuation_token_ids": (
                list(FIXED_CONTINUATION_TOKEN_IDS) if generation_mode == "fixed32" else None
            ),
            "raw_model_output": raw_output,
            "raw_parsed_box": raw_box,
            "predicted_box": predicted_box,
            "parse_failure_reason": parse_failure,
            "valid": predicted_box is not None,
            "box_iou": float(iou),
            "success_iou_0_5": bool(predicted_box is not None and iou >= 0.5),
            "generated_token_count": generated,
            "prompt_text_token_count": text_tokens,
            "realized_visual_tokens_by_layer": visual_trace,
            "visual_token_layer_work": sum(visual_trace),
            "average_visual_tokens": sum(visual_trace) / 32,
            "final_visual_tokens": visual_trace[-1],
            "vision_encoder_latency_s": self.raw_vision_latency_s,
            "projector_latency_s": self.projector_latency_s,
            "stage1_selector_latency_s": stage1_selector,
            "stage2_selector_latency_s": self.stage2_selector_latency_s,
            "selector_latency_s": stage1_selector + self.stage2_selector_latency_s,
            "llm_prefill_with_overhead_latency_s": self.llm_prefill_latency_s,
            "llm_prefill_excluding_selector_latency_s": max(
                0.0, self.llm_prefill_latency_s - self.stage2_selector_latency_s
            ),
            "time_to_first_token_s": (
                self.outer_vision_latency_s + self.projector_latency_s + self.llm_prefill_latency_s
            ),
            "decode_latency_s": self.decode_latency_s,
            "decode_time_per_generated_token_s": self.decode_latency_s / max(generated - 1, 1),
            "end_to_end_latency_s": end_to_end,
            "prompt_kv_total_bytes": self.prompt_kv_bytes,
            "final_kv_total_bytes": self.final_kv_bytes or self.prompt_kv_bytes,
            "peak_allocated_gpu_bytes": int(self.torch.cuda.max_memory_allocated()),
            "peak_reserved_gpu_bytes": int(self.torch.cuda.max_memory_reserved()),
            "clip_shape_compatibility_shim": self.clip_shape_shim,
            "llava_forward_compatibility_shim": "drop_generation_only_cache_position_and_logits_hint",
        }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scalar = (
        "box_iou", "generated_token_count", "vision_encoder_latency_s", "projector_latency_s",
        "selector_latency_s", "llm_prefill_with_overhead_latency_s", "time_to_first_token_s",
        "decode_latency_s", "end_to_end_latency_s", "prompt_kv_total_bytes",
        "final_kv_total_bytes", "peak_allocated_gpu_bytes", "peak_reserved_gpu_bytes",
    )
    return {
        "rows": len(rows),
        "unique_ids": len({row["sample_id"] for row in rows}),
        "valid_rate": sum(bool(row["valid"]) for row in rows) / len(rows),
        "acc_at_0_5": sum(bool(row["success_iou_0_5"]) for row in rows) / len(rows),
        "mean_iou": sum(float(row["box_iou"]) for row in rows) / len(rows),
        "median_iou": statistics.median(float(row["box_iou"]) for row in rows),
        "means": {key: sum(float(row[key]) for row in rows) / len(rows) for key in scalar},
        "visual_trace": rows[0]["realized_visual_tokens_by_layer"],
        "visual_token_layer_work": rows[0]["visual_token_layer_work"],
        "average_visual_tokens": rows[0]["average_visual_tokens"],
    }


def run(args: argparse.Namespace) -> int:
    manifest_sha, expected_rows = validate_args(args)
    wrapper_source = checked_wrapper_source()
    if git_commit(NUWA_ROOT) != NUWA_COMMIT or not git_clean(NUWA_ROOT):
        raise NativeNuwaError("Pinned Nüwa checkout is not at its clean registered revision")
    if not git_clean(LLAVA_ROOT):
        raise NativeNuwaError("Official LLaVA checkout is dirty")
    rows = load_manifest(args.manifest, None)
    if len(rows) != (8811 if args.mode == "quality" else 1000):
        raise NativeNuwaError("Manifest row count differs from the frozen contract")
    rows = rows[:expected_rows]
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and not args.resume:
        raise NativeNuwaError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    config = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "method": args.method,
        "mode": args.mode,
        "generation_mode": args.generation_mode,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "expected_rows": expected_rows,
        "max_new_tokens": 48,
        "fixed_continuation_token_ids": (
            list(FIXED_CONTINUATION_TOKEN_IDS) if args.generation_mode == "fixed32" else None
        ),
        "precision": "fp16_native",
        "batch_size": 1,
        "seed": 0,
        "decoding": "greedy",
        "prompt": REGISTERED_PROMPT,
        "NT": 192 if args.method == "NUWA_NATIVE_192" else None,
        "DIST": 280 if args.method == "NUWA_NATIVE_192" else None,
        "expected_visual_trace": expected_trace(args.method),
        "nuwa_root": str(NUWA_ROOT),
        "nuwa_commit": NUWA_COMMIT,
        "nuwa_tree_sha256": tree_hash(NUWA_ROOT),
        "llava_root": str(LLAVA_ROOT),
        "llava_commit": git_commit(LLAVA_ROOT),
        "llava_tree_sha256": tree_hash(LLAVA_ROOT),
        "checkpoint": str(MODEL_PATH),
        "checkpoint_config_sha256": sha256_file(MODEL_PATH / "config.json"),
        "wrapper_git_commit": wrapper_source["git_commit"],
        "wrapper_source_sha256": wrapper_source["runner_sha256"],
        "repetition": args.repetition,
        "order_position": args.order_position,
        "warmup_examples": args.warmup_examples,
    }
    config_path = run_dir / "config.json"
    if args.resume:
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        comparable = dict(config)
        comparable["created_at_utc"] = prior["created_at_utc"]
        if prior != comparable:
            raise NativeNuwaError("Resume configuration differs from the existing run")
    else:
        atomic_write(config_path, canonical_bytes(config))
    config_sha = sha256_file(config_path)
    partial = run_dir / "predictions.partial.jsonl"
    measured: list[dict[str, Any]] = []
    if args.resume and partial.exists():
        measured = [json.loads(line) for line in partial.read_text().splitlines() if line.strip()]
        if [row["sample_id"] for row in measured] != [row["sample_id"] for row in rows[:len(measured)]]:
            raise NativeNuwaError("Resume file is not an exact manifest prefix")
        if any(row.get("config_sha256") != config_sha for row in measured):
            raise NativeNuwaError("Resume row configuration hash differs")
    trial = NativeTrial(args.method)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": trial.torch.__version__,
        "cuda": trial.torch.version.cuda,
        "transformers": trial.transformers.__version__,
        "transformers_file": str(Path(trial.transformers.__file__).resolve()),
        "gpu": trial.torch.cuda.get_device_name(0),
        "clip_shape_compatibility_shim": trial.clip_shape_shim,
        "llava_forward_compatibility_shim": "drop_generation_only_cache_position_and_logits_hint",
    }
    atomic_write(run_dir / "environment.json", canonical_bytes(environment))
    for sample in rows[:args.warmup_examples]:
        trial.infer(sample, generation_mode=args.generation_mode)
    with partial.open("a" if measured else "w", encoding="utf-8") as handle:
        for index in range(len(measured), len(rows)):
            result = trial.infer(rows[index], generation_mode=args.generation_mode)
            result["manifest_index"] = index
            result["config_sha256"] = config_sha
            result["repetition"] = args.repetition
            result["order_position"] = args.order_position
            measured.append(result)
            handle.write(json.dumps(result, sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush()
            if (index + 1) % 25 == 0 or index + 1 == len(rows):
                print(f"NATIVE_NUWA {args.method} {args.mode}: {index + 1}/{len(rows)}", flush=True)
    predictions = run_dir / "predictions.jsonl"
    os.replace(partial, predictions)
    metrics = summarize(measured)
    metrics.update({"status": "complete", "config_sha256": config_sha})
    atomic_write(run_dir / "metrics.json", canonical_bytes(metrics))
    integrity = {
        "status": "PASS",
        "rows": len(measured),
        "ordered_unique_ids": [row["sample_id"] for row in measured] == [row["sample_id"] for row in rows],
        "all_traces_valid": all(
            row["realized_visual_tokens_by_layer"] == expected_trace(args.method) for row in measured
        ),
        "predictions_sha256": sha256_file(predictions),
        "metrics_sha256": sha256_file(run_dir / "metrics.json"),
    }
    if len(measured) != expected_rows or not all(value for key, value in integrity.items() if key.startswith("all_") or key == "ordered_unique_ids"):
        raise NativeNuwaError("Completed native Nüwa artifact failed integrity validation")
    atomic_write(run_dir / "integrity_report.json", canonical_bytes(integrity))
    atomic_write(run_dir / "COMPLETED.json", canonical_bytes({"status": "complete", **integrity}))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (NativeNuwaError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
