#!/usr/bin/env python
"""Run LLaVA-1.5 baseline grounding or VQA experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont


DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_GRID = 24
PROXY_LABEL = "image_masking"
TRUE_REMOVAL_LABEL = "true_token_removal"
TRUE_REMOVAL_LOCATION = "after_projector_before_llm_prefill"
PROXY_LOCATION = "image_masking_before_processor"


class BaselineError(RuntimeError):
    """Raised for recoverable experiment setup errors."""


class Tee:
    def __init__(self, stream: Any, log_handle: Any) -> None:
        self.stream = stream
        self.log_handle = log_handle

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.log_handle.write(text)
        self.log_handle.flush()
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.log_handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["grounding", "vqa"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    method_group = parser.add_mutually_exclusive_group(required=True)
    method_group.add_argument("--method", dest="method", choices=["full", "random", "uniform", "similarity_topk"])
    method_group.add_argument("--pruning-method", dest="method", choices=["full", "random", "uniform", "similarity_topk"])
    parser.add_argument("--keep-ratio", type=float, required=True)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--figures-root", type=Path, default=None)
    parser.add_argument("--tables-root", type=Path, default=Path("experiments/tables"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "vlm-pruning"))
    parser.add_argument("--resume", action="store_true", help="Append missing samples to an existing run directory.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--image-grid-size", type=int, default=DEFAULT_GRID)
    parser.add_argument("--max-figures", type=int, default=24)
    parser.add_argument("--quantization", default="none", choices=["none", "4bit", "8bit"])
    parser.add_argument(
        "--box-coordinate-mode",
        default="normalized_0_1",
        choices=["normalized_0_1", "normalized_0_1000", "absolute_pixels"],
    )
    parser.add_argument(
        "--pruning-implementation",
        default="masking_proxy",
        choices=["masking_proxy", TRUE_REMOVAL_LABEL],
        help="Use masking_proxy for the old image-masking baseline or true_token_removal for compacted LLM inputs.",
    )
    parser.add_argument(
        "--pruning-location",
        default=None,
        choices=[PROXY_LOCATION, TRUE_REMOVAL_LOCATION],
        help="Where pruning is applied. true_token_removal requires after_projector_before_llm_prefill.",
    )
    return parser.parse_args()


def run_name_from_args(args: argparse.Namespace, retained_tokens: int) -> str:
    if args.run_name:
        return args.run_name
    keep = int(round(args.keep_ratio * 100))
    model_name = "llava15" if "llava-1.5" in args.model.lower() else "model"
    return f"{args.dataset}_{model_name}_{args.task}_{args.method}_keep{keep:03d}_budget{retained_tokens}_seed{args.seed}"


def load_manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise BaselineError(f"Manifest does not exist: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    if not rows:
        raise BaselineError(f"Manifest is empty: {path}")
    return rows


def load_existing_results(path: Path, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], 0
    manifest_ids = {str(row["sample_id"]) for row in rows}
    by_id: dict[str, dict[str, Any]] = {}
    malformed_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue
            sample_id = str(result.get("sample_id"))
            if sample_id in manifest_ids and sample_id not in by_id:
                by_id[sample_id] = result
    ordered = [by_id[str(row["sample_id"])] for row in rows if str(row["sample_id"]) in by_id]
    return ordered, malformed_lines


def rewrite_per_sample_results(path: Path, results: list[dict[str, Any]]) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if path.exists():
        path.replace(backup)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=True) + "\n")


def stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def uniform_points(count: int, grid: int) -> list[int]:
    if count <= 0:
        return []
    if count >= grid * grid:
        return list(range(grid * grid))
    cols = max(1, int(round(math.sqrt(count))))
    rows = max(1, int(math.ceil(count / cols)))
    row_positions = [round(x) for x in linspace(0, grid - 1, rows)]
    col_positions = [round(x) for x in linspace(0, grid - 1, cols)]
    selected: list[int] = []
    seen = set()
    for r in row_positions:
        for c in col_positions:
            idx = int(r) * grid + int(c)
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
    if len(selected) > count:
        return selected[:count]
    all_indices = list(range(grid * grid))
    while len(selected) < count:
        best_idx = None
        best_distance = -1.0
        selected_coords = [(idx // grid, idx % grid) for idx in selected]
        for idx in all_indices:
            if idx in seen:
                continue
            r, c = idx // grid, idx % grid
            distance = min((r - sr) ** 2 + (c - sc) ** 2 for sr, sc in selected_coords) if selected_coords else 1e9
            if distance > best_distance:
                best_distance = distance
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        seen.add(best_idx)
    return selected[:count]


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [(start + stop) / 2]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]


def retained_indices(method: str, keep_count: int, original_count: int, seed: int, sample_id: str, grid: int) -> list[int]:
    if method == "full":
        return list(range(original_count))
    if method == "random":
        import random

        rng = random.Random(seed + stable_int(sample_id))
        values = list(range(original_count))
        rng.shuffle(values)
        return sorted(values[:keep_count])
    if method == "uniform":
        if keep_count >= original_count:
            return list(range(original_count))
        if keep_count >= original_count / 2:
            drop_count = original_count - keep_count
            drop = set(uniform_points(drop_count, grid))
            return [idx for idx in range(original_count) if idx not in drop][:keep_count]
        return sorted(uniform_points(keep_count, grid))
    raise BaselineError(f"Method '{method}' is not implemented.")


def indices_checksum(indices: list[int]) -> str:
    payload = json.dumps(indices, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cuda_sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mean_or_none(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def resolve_output_roots(args: argparse.Namespace) -> None:
    if args.pruning_implementation == TRUE_REMOVAL_LABEL:
        if args.pruning_location is None:
            args.pruning_location = TRUE_REMOVAL_LOCATION
        if args.pruning_location != TRUE_REMOVAL_LOCATION:
            raise BaselineError(
                f"{TRUE_REMOVAL_LABEL} requires --pruning-location {TRUE_REMOVAL_LOCATION}; "
                f"got {args.pruning_location}."
            )
        if args.method == "similarity_topk":
            raise BaselineError("true_token_removal currently supports full, random, and uniform only.")
        if args.output_root is None:
            args.output_root = Path("experiments/runs_true_removal")
        if args.figures_root is None:
            args.figures_root = Path("experiments/figures_true_removal")
    else:
        if args.pruning_location is None:
            args.pruning_location = "none" if args.method == "full" else PROXY_LOCATION
        if args.output_root is None:
            args.output_root = Path("experiments/runs")
        if args.figures_root is None:
            args.figures_root = Path("experiments/figures")


def dtype_nbytes(torch: Any, dtype: Any) -> int:
    if dtype in {torch.float64, torch.int64}:
        return 8
    if dtype in {torch.float32, torch.int32}:
        return 4
    if dtype in {torch.float16, torch.bfloat16, torch.int16}:
        return 2
    return 1


def llm_prefill_flops(sequence_length: float, model: Any) -> float | None:
    try:
        text_config = model.config.text_config
        hidden_size = int(text_config.hidden_size)
        intermediate_size = int(text_config.intermediate_size)
        num_layers = int(text_config.num_hidden_layers)
    except Exception:
        return None
    n = float(sequence_length)
    per_layer = (8.0 * n * hidden_size * hidden_size) + (4.0 * n * n * hidden_size) + (
        6.0 * n * hidden_size * intermediate_size
    )
    return per_layer * num_layers


def kv_cache_memory_mb(sequence_length: float, model: Any, dtype: Any, torch: Any) -> float | None:
    try:
        text_config = model.config.text_config
        hidden_size = int(text_config.hidden_size)
        num_layers = int(text_config.num_hidden_layers)
        num_attention_heads = int(text_config.num_attention_heads)
        num_key_value_heads = int(getattr(text_config, "num_key_value_heads", num_attention_heads))
        head_dim = hidden_size // num_attention_heads
        kv_dim = num_key_value_heads * head_dim
    except Exception:
        return None
    bytes_per_value = dtype_nbytes(torch, dtype)
    total_bytes = num_layers * 2 * float(sequence_length) * kv_dim * bytes_per_value
    return total_bytes / (1024**2)


def add_efficiency_estimates(metrics: dict[str, Any], model: Any, torch: Any) -> None:
    compacted = metrics.get("mean_compacted_input_sequence_length")
    full = metrics.get("mean_original_input_sequence_length")
    if compacted is None or full is None:
        return
    compacted_flops = llm_prefill_flops(float(compacted), model)
    full_flops = llm_prefill_flops(float(full), model)
    metrics["theoretical_llm_prefill_flops"] = compacted_flops
    metrics["theoretical_llm_prefill_flops_full_reference"] = full_flops
    if compacted_flops is not None and full_flops:
        ratio = compacted_flops / full_flops
        metrics["theoretical_llm_prefill_flops_ratio_vs_full"] = ratio
        metrics["theoretical_llm_prefill_flops_reduction_vs_full"] = 1.0 - ratio
    else:
        metrics["theoretical_llm_prefill_flops_ratio_vs_full"] = None
        metrics["theoretical_llm_prefill_flops_reduction_vs_full"] = None
    dtype = next(model.parameters()).dtype
    metrics["kv_cache_tokens_estimate"] = compacted
    metrics["kv_cache_memory_estimate_mb"] = kv_cache_memory_mb(float(compacted), model, dtype, torch)


def mask_image(image: Image.Image, keep_indices: list[int], grid: int) -> Image.Image:
    if len(keep_indices) >= grid * grid:
        return image
    keep = set(keep_indices)
    masked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(masked, "RGBA")
    width, height = masked.size
    cell_w = width / grid
    cell_h = height / grid
    for idx in range(grid * grid):
        if idx in keep:
            continue
        r, c = divmod(idx, grid)
        xy = [c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h]
        draw.rectangle(xy, fill=(0, 0, 0, 220))
    return masked


def parse_box(text: str, width: int, height: int, mode: str) -> tuple[list[float] | None, list[float] | None, str | None]:
    if not text or not text.strip():
        return None, None, "empty_response"
    candidates = []
    for pattern in [r"<box>\s*([^<]+?)\s*</box>", r"\[([^\]]+)\]", r"\(([^\)]+)\)"]:
        candidates.extend(re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL))
    if not candidates:
        candidates = [text]
    for candidate in candidates:
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", candidate)
        if len(numbers) < 4:
            continue
        raw = [float(numbers[i]) for i in range(4)]
        if raw[2] <= raw[0] or raw[3] <= raw[1]:
            return raw, None, "invalid_coordinate_order"
        if mode == "normalized_0_1":
            px = [raw[0] * width, raw[1] * height, raw[2] * width, raw[3] * height]
        elif mode == "normalized_0_1000":
            px = [raw[0] * width / 1000.0, raw[1] * height / 1000.0, raw[2] * width / 1000.0, raw[3] * height / 1000.0]
        else:
            px = raw[:]
        if px[2] <= 0 or px[3] <= 0 or px[0] >= width or px[1] >= height:
            return raw, None, "box_outside_image"
        clipped = [
            max(0.0, min(float(width), px[0])),
            max(0.0, min(float(height), px[1])),
            max(0.0, min(float(width), px[2])),
            max(0.0, min(float(height), px[3])),
        ]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return raw, None, "empty_box_after_clipping"
        return raw, clipped, None
    return None, None, "no_parseable_box"


def box_iou(a: list[float], b: list[float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def normalise_yes_no(text: str) -> tuple[str | None, bool]:
    cleaned = re.sub(r"[^A-Za-z]+", " ", text.lower()).strip()
    if not cleaned:
        return None, True
    first = cleaned.split()[0]
    if first in {"yes", "no"}:
        return first, False
    return None, True


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int, int]) -> None:
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox(xy, text, font=font)
    pad = 3
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=(255, 255, 255, 220))
    draw.text(xy, text, fill=fill, font=font)


def save_grounding_figure(
    sample: dict[str, Any],
    raw_image: Image.Image,
    pred_box: list[float] | None,
    retained: list[int],
    grid: int,
    method: str,
    keep_ratio: float,
    iou: float | None,
    success: bool,
    output_dir: Path,
) -> Path:
    canvas = raw_image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    if len(retained) < grid * grid:
        keep = set(retained)
        cell_w = width / grid
        cell_h = height / grid
        for idx in range(grid * grid):
            r, c = divmod(idx, grid)
            xy = [c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h]
            if idx in keep:
                draw.rectangle(xy, fill=(0, 180, 70, 45), outline=(0, 120, 50, 65))
            else:
                draw.rectangle(xy, fill=(0, 0, 0, 65))
    gt = sample["gt_box"]
    draw.rectangle(gt, outline=(0, 180, 0, 255), width=4)
    if pred_box is not None:
        draw.rectangle(pred_box, outline=(220, 30, 30, 255), width=4)
    title = f"{sample.get('query', '')[:96]} | IoU={iou if iou is not None else 0.0:.3f} | {'success' if success else 'fail'}"
    draw_label(draw, (8, 8), title, (0, 0, 0, 255))
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample["sample_id"]))
    iou_int = int(round((iou or 0.0) * 100))
    keep = int(round(keep_ratio * 100))
    status = "success" if success else "fail"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{safe_id}_{method}_keep{keep:03d}_iou{iou_int:03d}_{status}.png"
    canvas.save(path)
    return path


def require_model_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
    except Exception as exc:
        raise BaselineError(f"PyTorch is not installed or cannot be imported: {exc}") from exc
    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration
    except Exception as exc:
        raise BaselineError(f"Transformers LLaVA classes are unavailable: {exc}") from exc
    return torch, AutoProcessor, LlavaForConditionalGeneration


def load_model(model_id: str, offload_dir: Path, quantization: str) -> tuple[Any, Any, Any, Any]:
    torch, AutoProcessor, LlavaForConditionalGeneration = require_model_dependencies()
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if quantization != "none":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise BaselineError(f"Quantized loading requested but BitsAndBytesConfig is unavailable: {exc}") from exc
        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    if torch.cuda.is_available():
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        gpu_limit_mib = max(1024, int((free_bytes / (1024**2)) * 0.85))
        offload_dir.mkdir(parents=True, exist_ok=True)
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: f"{gpu_limit_mib}MiB", "cpu": "64GiB"}
        kwargs["offload_folder"] = str(offload_dir)
        kwargs["offload_state_dict"] = True
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
        model.eval()
    except Exception as exc:
        raise BaselineError(f"LLaVA-1.5-7B cannot be downloaded or loaded from '{model_id}': {type(exc).__name__}: {exc}") from exc
    device = next(model.parameters()).device
    return torch, processor, model, device


def make_prompt(task: str, sample: dict[str, Any], coordinate_mode: str) -> str:
    if task == "grounding":
        if coordinate_mode == "normalized_0_1":
            coord = "normalized 0-1 image coordinates"
        elif coordinate_mode == "normalized_0_1000":
            coord = "normalized 0-1000 image coordinates"
        else:
            coord = "absolute image pixel coordinates"
        return (
            f"Identify the target described by this referring expression: \"{sample['query']}\".\n"
            f"Return exactly one bounding box in {coord} as [x_min, y_min, x_max, y_max]. "
            "Do not include any other text."
        )
    return f"{sample['question']}\nAnswer with exactly one word: yes or no."


def apply_chat_template(processor: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(messages, add_generation_prompt=True)
        except Exception:
            pass
    return f"USER: <image>\n{prompt}\nASSISTANT:"


def selected_vision_features(model: Any, pixel_values: Any) -> tuple[Any, float, float]:
    import torch

    vision_feature_layer = getattr(model.config, "vision_feature_layer", -2)
    vision_feature_select_strategy = getattr(model.config, "vision_feature_select_strategy", "default")
    cuda_sync(torch)
    vision_start = time.perf_counter()
    with torch.inference_mode():
        image_outputs = model.model.vision_tower(pixel_values, output_hidden_states=True, return_dict=True)
    cuda_sync(torch)
    vision_latency = time.perf_counter() - vision_start

    if isinstance(vision_feature_layer, int):
        selected = image_outputs.hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            selected = selected[:, 1:]
    else:
        hidden_states = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
        if vision_feature_select_strategy == "default":
            hidden_states = [state[:, 1:] for state in hidden_states]
        selected = torch.cat(hidden_states, dim=-1)

    cuda_sync(torch)
    projector_start = time.perf_counter()
    with torch.inference_mode():
        image_features = model.model.multi_modal_projector(selected)
    cuda_sync(torch)
    projector_latency = time.perf_counter() - projector_start
    return image_features, vision_latency, projector_latency


def eos_token_ids(model: Any, processor: Any) -> set[int]:
    ids: set[int] = set()
    for candidate in [
        getattr(model.generation_config, "eos_token_id", None),
        getattr(model.config, "eos_token_id", None),
        getattr(getattr(processor, "tokenizer", None), "eos_token_id", None),
    ]:
        if candidate is None:
            continue
        if isinstance(candidate, int):
            ids.add(candidate)
        else:
            ids.update(int(item) for item in candidate)
    return ids


def prepare_compacted_inputs(
    torch: Any,
    processor: Any,
    model: Any,
    device: Any,
    image: Image.Image,
    prompt: str,
    retained: list[int],
    expected_original_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    text_prompt = apply_chat_template(processor, prompt)
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    moved = {}
    dtype = next(model.parameters()).dtype
    for key, value in inputs.items():
        if hasattr(value, "to"):
            if key == "pixel_values":
                value = value.to(device=device, dtype=dtype)
            else:
                value = value.to(device=device)
        moved[key] = value

    image_features, vision_latency, projector_latency = selected_vision_features(model, moved["pixel_values"])
    if image_features.ndim != 3 or image_features.shape[0] != 1:
        raise BaselineError(f"Expected one image feature tensor with shape [1, tokens, hidden], got {tuple(image_features.shape)}")

    input_ids = moved["input_ids"]
    attention_mask = moved.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    image_token_index = int(getattr(model.config, "image_token_index", 32000))
    image_positions = torch.nonzero(input_ids[0] == image_token_index, as_tuple=False).flatten()
    original_visual_tokens = int(image_positions.numel())
    feature_tokens = int(image_features.shape[1])
    if original_visual_tokens != feature_tokens:
        raise BaselineError(
            "Cannot prove true token removal because image placeholder count does not match projected features: "
            f"{original_visual_tokens} placeholders vs {feature_tokens} features."
        )
    if expected_original_tokens != original_visual_tokens:
        raise BaselineError(
            f"Configured visual-token grid expects {expected_original_tokens} tokens, "
            f"but processor/model produced {original_visual_tokens}."
        )
    if not retained:
        raise BaselineError("At least one visual token must be retained.")
    if max(retained) >= original_visual_tokens or min(retained) < 0:
        raise BaselineError(f"Retained index out of range for {original_visual_tokens} visual tokens.")

    selection_start = time.perf_counter()
    keep_tensor = torch.tensor(retained, dtype=torch.long, device=input_ids.device)
    text_embeds = model.get_input_embeddings()(input_ids).to(dtype=dtype)
    text_embeds = text_embeds.clone()
    text_embeds[0, image_positions, :] = image_features[0].to(device=text_embeds.device, dtype=text_embeds.dtype)
    keep_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
    keep_mask[image_positions] = False
    keep_mask[image_positions[keep_tensor]] = True
    selected_positions = torch.nonzero(keep_mask, as_tuple=False).flatten()
    compacted_inputs_embeds = text_embeds[:, selected_positions, :]
    compacted_attention_mask = attention_mask[:, selected_positions]
    compacted_input_ids = input_ids[:, selected_positions]
    cuda_sync(torch)
    pruning_selection_latency = time.perf_counter() - selection_start

    original_sequence_length = int(input_ids.shape[1])
    compacted_sequence_length = int(compacted_input_ids.shape[1])
    diagnostics = {
        "vision_encoder_latency_s": vision_latency,
        "projector_latency_s": projector_latency,
        "pruning_selection_latency_s": pruning_selection_latency,
        "original_visual_tokens": original_visual_tokens,
        "requested_keep_ratio": len(retained) / original_visual_tokens,
        "expected_retained_visual_tokens": len(retained),
        "actual_retained_visual_tokens": len(retained),
        "actual_llm_visual_tokens": len(retained),
        "original_input_sequence_length": original_sequence_length,
        "compacted_input_sequence_length": compacted_sequence_length,
        "sequence_length_reduction": original_sequence_length - compacted_sequence_length,
        "retained_indices_checksum": indices_checksum(retained),
        "pruning_implementation": TRUE_REMOVAL_LABEL,
        "pruning_location": TRUE_REMOVAL_LOCATION,
    }
    compacted = {
        "input_ids": compacted_input_ids,
        "inputs_embeds": compacted_inputs_embeds,
        "attention_mask": compacted_attention_mask,
    }
    return compacted, diagnostics


def greedy_decode_from_embeds(
    torch: Any,
    processor: Any,
    model: Any,
    compacted: dict[str, Any],
    max_new_tokens: int,
) -> tuple[str, int, float, float]:
    generated_ids: list[Any] = []
    eos_ids = eos_token_ids(model, processor)
    attention_mask = compacted["attention_mask"]
    cuda_sync(torch)
    prefill_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(
            inputs_embeds=compacted["inputs_embeds"],
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    cuda_sync(torch)
    prefill_latency = time.perf_counter() - prefill_start
    generated_ids.append(next_token)
    past_key_values = outputs.past_key_values
    finished = bool(eos_ids and int(next_token.item()) in eos_ids)

    cuda_sync(torch)
    decode_start = time.perf_counter()
    with torch.inference_mode():
        while len(generated_ids) < max_new_tokens and not finished:
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids.append(next_token)
            finished = bool(eos_ids and int(next_token.item()) in eos_ids)
    cuda_sync(torch)
    decode_latency = time.perf_counter() - decode_start

    generated = torch.cat(generated_ids, dim=1)[0] if generated_ids else torch.empty(0, dtype=torch.long)
    decoded = processor.decode(generated, skip_special_tokens=True).strip()
    return decoded, int(generated.numel()), prefill_latency, decode_latency


def generate_answer_true_removal(
    torch: Any,
    processor: Any,
    model: Any,
    device: Any,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    retained: list[int],
    expected_original_tokens: int,
) -> tuple[str, int, float, dict[str, Any]]:
    compacted, diagnostics = prepare_compacted_inputs(
        torch, processor, model, device, image, prompt, retained, expected_original_tokens
    )
    raw_output, generated_count, prefill_latency, decode_latency = greedy_decode_from_embeds(
        torch, processor, model, compacted, max_new_tokens
    )
    diagnostics["llm_prefill_latency_s"] = prefill_latency
    diagnostics["prefill_latency_s"] = prefill_latency
    diagnostics["time_to_first_token_s"] = prefill_latency
    diagnostics["decode_latency_s"] = decode_latency
    total_latency = (
        diagnostics["vision_encoder_latency_s"]
        + diagnostics["projector_latency_s"]
        + diagnostics["pruning_selection_latency_s"]
        + prefill_latency
        + decode_latency
    )
    diagnostics["total_inference_latency_s"] = total_latency
    return raw_output, generated_count, total_latency, diagnostics


def generate_answer(torch: Any, processor: Any, model: Any, device: Any, image: Image.Image, prompt: str, max_new_tokens: int) -> tuple[str, int, float]:
    text_prompt = apply_chat_template(processor, prompt)
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            if key == "pixel_values":
                value = value.to(device=device, dtype=next(model.parameters()).dtype)
            else:
                value = value.to(device=device)
        moved[key] = value
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**moved, max_new_tokens=max_new_tokens, do_sample=False)
    latency = time.perf_counter() - start
    input_len = moved["input_ids"].shape[1]
    generated = output_ids[0, input_len:]
    decoded = processor.decode(generated, skip_special_tokens=True).strip()
    return decoded, int(generated.numel()), latency


def build_config(args: argparse.Namespace, run_name: str, original_tokens: int, retained_tokens: int) -> dict[str, Any]:
    pruning_implementation = (
        TRUE_REMOVAL_LABEL
        if args.pruning_implementation == TRUE_REMOVAL_LABEL
        else ("none" if args.method == "full" else PROXY_LABEL)
    )
    return {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": args.task,
        "dataset": args.dataset,
        "manifest": str(args.manifest),
        "model": args.model,
        "method": args.method,
        "keep_ratio": args.keep_ratio,
        "token_budget": retained_tokens,
        "original_visual_tokens": original_tokens,
        "retained_visual_tokens": retained_tokens,
        "pruning_implementation": pruning_implementation,
        "pruning_location": args.pruning_location,
        "seed": args.seed,
        "limit": args.limit,
        "image_grid_size": args.image_grid_size,
        "box_coordinate_mode": args.box_coordinate_mode if args.task == "grounding" else None,
        "quantization": args.quantization,
        "resume": bool(args.resume),
        "wandb_enabled": bool(args.wandb),
        "wandb_project": args.wandb_project,
    }


def init_wandb(args: argparse.Namespace, config: dict[str, Any], run_dir: Path) -> Any | None:
    if not args.wandb:
        return None
    try:
        import wandb

        return wandb.init(project=args.wandb_project, name=config["run_name"], config=config, dir=str(run_dir), reinit=True)
    except Exception as exc:
        print(f"W&B logging failed unexpectedly; continuing with local logging. Error: {type(exc).__name__}: {exc}")
        print("To re-enable online logging, run:")
        print("wandb login")
        print('$env:WANDB_PROJECT="vlm-pruning"')
        return None


def update_summary_table(config: dict[str, Any], metrics: dict[str, Any], tables_root: Path) -> None:
    tables_root.mkdir(parents=True, exist_ok=True)
    if str(config.get("run_name") or "").startswith("smoke"):
        return
    is_true_removal = config.get("pruning_implementation") == TRUE_REMOVAL_LABEL
    if is_true_removal:
        table_name = "true_removal_grounding.csv" if config["task"] == "grounding" else "true_removal_vqa.csv"
    else:
        table_name = "baseline_grounding.csv" if config["task"] == "grounding" else "baseline_vqa.csv"
    path = tables_root / table_name
    row = {
        "run_name": config["run_name"],
        "task": config["task"],
        "dataset": config["dataset"],
        "model": config["model"],
        "method": config["method"],
        "keep_ratio": config["keep_ratio"],
        "token_budget": config["token_budget"],
        "seed": config["seed"],
        "pruning_implementation": config["pruning_implementation"],
        "pruning_location": config.get("pruning_location"),
    }
    row.update(metrics)
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
        existing = [item for item in existing if item.get("run_name") != config["run_name"]]
    existing.append({key: "" if value is None else value for key, value in row.items()})
    fieldnames = sorted({key for item in existing for key in item.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
    if is_true_removal:
        update_efficiency_table(config, metrics, tables_root)
        write_true_removal_summary(tables_root)


def update_efficiency_table(config: dict[str, Any], metrics: dict[str, Any], tables_root: Path) -> None:
    path = tables_root / "true_removal_efficiency.csv"
    efficiency_keys = [
        "run_name",
        "task",
        "dataset",
        "model",
        "method",
        "keep_ratio",
        "token_budget",
        "seed",
        "limit",
        "pruning_implementation",
        "pruning_location",
        "original_visual_tokens",
        "expected_retained_visual_tokens",
        "actual_retained_visual_tokens",
        "actual_llm_visual_tokens",
        "mean_original_input_sequence_length",
        "mean_compacted_input_sequence_length",
        "mean_sequence_length_reduction",
        "vision_encoder_latency_s",
        "projector_latency_s",
        "pruning_selection_latency_s",
        "llm_prefill_latency_s",
        "time_to_first_token_s",
        "decode_latency_s",
        "mean_latency_s",
        "total_inference_latency_s",
        "throughput_examples_per_s",
        "generated_token_count_mean",
        "generated_token_count_total",
        "gpu_peak_memory_allocated_mb",
        "gpu_peak_memory_reserved_mb",
        "kv_cache_tokens_estimate",
        "kv_cache_memory_estimate_mb",
        "theoretical_llm_prefill_flops",
        "theoretical_llm_prefill_flops_ratio_vs_full",
        "theoretical_llm_prefill_flops_reduction_vs_full",
    ]
    row = {
        "run_name": config["run_name"],
        "task": config["task"],
        "dataset": config["dataset"],
        "model": config["model"],
        "method": config["method"],
        "keep_ratio": config["keep_ratio"],
        "token_budget": config["token_budget"],
        "seed": config["seed"],
        "limit": config["limit"],
        "pruning_implementation": config["pruning_implementation"],
        "pruning_location": config.get("pruning_location"),
    }
    row.update({key: metrics.get(key) for key in efficiency_keys if key not in row})
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
        existing = [item for item in existing if item.get("run_name") != config["run_name"]]
    existing.append({key: "" if value is None else value for key, value in row.items()})
    fieldnames = [key for key in efficiency_keys if key in {field for item in existing for field in item.keys()}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["_No rows yet._"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_true_removal_summary(tables_root: Path) -> None:
    grounding = read_csv_rows(tables_root / "true_removal_grounding.csv")
    vqa = read_csv_rows(tables_root / "true_removal_vqa.csv")
    efficiency = read_csv_rows(tables_root / "true_removal_efficiency.csv")
    baseline_rows = read_csv_rows(tables_root / "baseline_grounding.csv") + read_csv_rows(tables_root / "baseline_vqa.csv")
    baseline_by_key = {
        (row.get("task"), row.get("dataset"), row.get("method"), row.get("keep_ratio"), row.get("seed")): row
        for row in baseline_rows
    }
    true_full_by_task = {
        row.get("task"): row
        for row in grounding + vqa
        if row.get("method") == "full" and str(row.get("keep_ratio")) in {"1.0", "1", "1.00"}
    }
    full_efficiency_by_task = {
        row.get("task"): row
        for row in efficiency
        if row.get("method") == "full" and str(row.get("keep_ratio")) in {"1.0", "1", "1.00"}
    }

    quality_rows: list[dict[str, Any]] = []
    for row in grounding + vqa:
        task = row.get("task")
        quality_value = as_float(row.get("mean_iou")) if task == "grounding" else as_float(row.get("vqa_accuracy"))
        accuracy_value = as_float(row.get("accuracy_at_0.5") or row.get("accuracy_iou_0_5")) if task == "grounding" else as_float(row.get("accuracy"))
        full_row = true_full_by_task.get(task)
        full_quality = None
        if full_row:
            full_quality = as_float(full_row.get("mean_iou")) if task == "grounding" else as_float(full_row.get("vqa_accuracy"))
        baseline = baseline_by_key.get((task, row.get("dataset"), row.get("method"), row.get("keep_ratio"), row.get("seed")))
        baseline_quality = None
        if baseline:
            baseline_quality = as_float(baseline.get("mean_iou")) if task == "grounding" else as_float(baseline.get("vqa_accuracy"))
        delta_full = quality_value - full_quality if quality_value is not None and full_quality is not None else None
        delta_proxy = quality_value - baseline_quality if quality_value is not None and baseline_quality is not None else None
        quality_rows.append(
            {
                "run_name": row.get("run_name"),
                "task": task,
                "method": row.get("method"),
                "keep_ratio": row.get("keep_ratio"),
                "tokens": row.get("actual_llm_visual_tokens") or row.get("token_budget"),
                "quality": quality_value if quality_value is not None else "",
                "accuracy": accuracy_value if accuracy_value is not None else "",
                "delta_vs_true_full": delta_full if delta_full is not None else "",
                "delta_vs_proxy": delta_proxy if delta_proxy is not None else "",
            }
        )

    efficiency_rows = []
    for row in efficiency:
        full_row = full_efficiency_by_task.get(row.get("task"))
        latency = as_float(row.get("mean_latency_s"))
        prefill = as_float(row.get("llm_prefill_latency_s"))
        memory = as_float(row.get("gpu_peak_memory_allocated_mb"))
        full_latency = as_float(full_row.get("mean_latency_s")) if full_row else None
        full_prefill = as_float(full_row.get("llm_prefill_latency_s")) if full_row else None
        full_memory = as_float(full_row.get("gpu_peak_memory_allocated_mb")) if full_row else None
        efficiency_rows.append(
            {
                "run_name": row.get("run_name"),
                "task": row.get("task"),
                "method": row.get("method"),
                "keep_ratio": row.get("keep_ratio"),
                "tokens": row.get("actual_llm_visual_tokens"),
                "seq_len": row.get("mean_compacted_input_sequence_length"),
                "latency_s": latency if latency is not None else "",
                "latency_delta_vs_full": (latency - full_latency) if latency is not None and full_latency is not None else "",
                "prefill_s": prefill if prefill is not None else "",
                "prefill_delta_vs_full": (prefill - full_prefill) if prefill is not None and full_prefill is not None else "",
                "memory_mb": memory if memory is not None else "",
                "memory_delta_vs_full": (memory - full_memory) if memory is not None and full_memory is not None else "",
                "flops_ratio": as_float(row.get("theoretical_llm_prefill_flops_ratio_vs_full")) or "",
            }
        )

    lines = [
        "# True Removal 500-Example Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "These runs gather the retained projected visual embeddings and feed a shorter sequence to the language model.",
        "",
        "## Quality",
        "",
        *markdown_table(
            quality_rows,
            ["run_name", "task", "method", "keep_ratio", "tokens", "quality", "accuracy", "delta_vs_true_full", "delta_vs_proxy"],
        ),
        "",
        "## Efficiency",
        "",
        *markdown_table(
            efficiency_rows,
            [
                "run_name",
                "task",
                "method",
                "keep_ratio",
                "tokens",
                "seq_len",
                "latency_s",
                "latency_delta_vs_full",
                "prefill_s",
                "prefill_delta_vs_full",
                "memory_mb",
                "memory_delta_vs_full",
                "flops_ratio",
            ],
        ),
        "",
        "## FLOPs Note",
        "",
        "LLM prefill FLOPs use a transparent transformer estimate with Q/K/V/O projections, attention score/value matmuls, and SwiGLU FFN terms. Because this goal prunes before LLM prefill, every LLM layer uses the compacted sequence length.",
        "",
    ]
    with (tables_root / "true_removal_500_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def aggregate_grounding(results: list[dict[str, Any]], total_wall_time: float, peak_memory_mb: float | None) -> dict[str, Any]:
    n = len(results)
    valid = [row for row in results if row["valid"]]
    parsed = [row for row in results if row["raw_parsed_box"] is not None]
    ious = [float(row["box_iou"]) for row in valid]
    latencies = [float(row["latency_s"]) for row in results if row.get("latency_s") is not None]
    generated = [int(row["generated_token_count"]) for row in results]
    metrics = {
        "num_samples": n,
        "valid_output_rate": len(valid) / n if n else 0.0,
        "valid_prediction_rate": len(valid) / n if n else 0.0,
        "invalid_prediction_count": n - len(valid),
        "invalid_output_rate": 1.0 - (len(valid) / n if n else 0.0),
        "parsed_box_rate": len(parsed) / n if n else 0.0,
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "accuracy_iou_0_5": sum(1 for row in valid if row["success_iou_0_5"]) / n if n else 0.0,
        "accuracy_at_0.5": sum(1 for row in valid if row["success_iou_0_5"]) / n if n else 0.0,
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else None,
        "total_inference_latency_s": sum(latencies) if latencies else None,
        "llm_prefill_latency_s": mean_or_none([row.get("llm_prefill_latency_s") for row in results]),
        "prefill_latency_s": mean_or_none([row.get("llm_prefill_latency_s") for row in results]),
        "time_to_first_token_s": mean_or_none([row.get("time_to_first_token_s") for row in results]),
        "decode_latency_s": mean_or_none([row.get("decode_latency_s") for row in results]),
        "vision_encoder_latency_s": mean_or_none([row.get("vision_encoder_latency_s") for row in results]),
        "projector_latency_s": mean_or_none([row.get("projector_latency_s") for row in results]),
        "pruning_selection_latency_s": mean_or_none([row.get("pruning_selection_latency_s") for row in results]),
        "generated_token_count_mean": sum(generated) / len(generated) if generated else 0.0,
        "generated_token_count_total": sum(generated),
        "throughput_examples_per_s": n / total_wall_time if total_wall_time > 0 else None,
        "gpu_peak_memory_allocated_mb": peak_memory_mb,
        "gpu_peak_memory_mb": peak_memory_mb,
        "original_visual_tokens": mean_or_none([row.get("original_visual_tokens") for row in results]),
        "requested_keep_ratio": mean_or_none([row.get("requested_keep_ratio") for row in results]),
        "expected_retained_visual_tokens": mean_or_none([row.get("expected_retained_visual_tokens") for row in results]),
        "actual_retained_visual_tokens": mean_or_none([row.get("actual_retained_visual_tokens") for row in results]),
        "actual_llm_visual_tokens": mean_or_none([row.get("actual_llm_visual_tokens") for row in results]),
        "mean_original_input_sequence_length": mean_or_none([row.get("original_input_sequence_length") for row in results]),
        "mean_compacted_input_sequence_length": mean_or_none([row.get("compacted_input_sequence_length") for row in results]),
        "mean_sequence_length_reduction": mean_or_none([row.get("sequence_length_reduction") for row in results]),
    }
    return metrics


def aggregate_vqa(results: list[dict[str, Any]], total_wall_time: float, peak_memory_mb: float | None) -> dict[str, Any]:
    n = len(results)
    valid = [row for row in results if not row["invalid"]]
    latencies = [float(row["latency_s"]) for row in results if row.get("latency_s") is not None]
    generated = [int(row["generated_token_count"]) for row in results]
    true_positive = sum(1 for row in results if row.get("normalised_answer") == "yes" and normalise_yes_no(str(row.get("target_answer")))[0] == "yes")
    false_positive = sum(1 for row in results if row.get("normalised_answer") == "yes" and normalise_yes_no(str(row.get("target_answer")))[0] == "no")
    false_negative = sum(1 for row in results if row.get("normalised_answer") == "no" and normalise_yes_no(str(row.get("target_answer")))[0] == "yes")
    predicted_yes = sum(1 for row in results if row.get("normalised_answer") == "yes")
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    metrics = {
        "num_samples": n,
        "vqa_accuracy": sum(1 for row in results if row["correct"]) / n if n else 0.0,
        "accuracy": sum(1 for row in results if row["correct"]) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        "yes_rate": predicted_yes / n if n else 0.0,
        "valid_output_rate": len(valid) / n if n else 0.0,
        "valid_prediction_rate": len(valid) / n if n else 0.0,
        "invalid_prediction_count": n - len(valid),
        "invalid_output_rate": 1.0 - (len(valid) / n if n else 0.0),
        "answer_extraction_failure_rate": sum(1 for row in results if row["extraction_failure"]) / n if n else 0.0,
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else None,
        "total_inference_latency_s": sum(latencies) if latencies else None,
        "llm_prefill_latency_s": mean_or_none([row.get("llm_prefill_latency_s") for row in results]),
        "prefill_latency_s": mean_or_none([row.get("llm_prefill_latency_s") for row in results]),
        "time_to_first_token_s": mean_or_none([row.get("time_to_first_token_s") for row in results]),
        "decode_latency_s": mean_or_none([row.get("decode_latency_s") for row in results]),
        "vision_encoder_latency_s": mean_or_none([row.get("vision_encoder_latency_s") for row in results]),
        "projector_latency_s": mean_or_none([row.get("projector_latency_s") for row in results]),
        "pruning_selection_latency_s": mean_or_none([row.get("pruning_selection_latency_s") for row in results]),
        "generated_token_count_mean": sum(generated) / len(generated) if generated else 0.0,
        "generated_token_count_total": sum(generated),
        "throughput_examples_per_s": n / total_wall_time if total_wall_time > 0 else None,
        "gpu_peak_memory_allocated_mb": peak_memory_mb,
        "gpu_peak_memory_mb": peak_memory_mb,
        "original_visual_tokens": mean_or_none([row.get("original_visual_tokens") for row in results]),
        "requested_keep_ratio": mean_or_none([row.get("requested_keep_ratio") for row in results]),
        "expected_retained_visual_tokens": mean_or_none([row.get("expected_retained_visual_tokens") for row in results]),
        "actual_retained_visual_tokens": mean_or_none([row.get("actual_retained_visual_tokens") for row in results]),
        "actual_llm_visual_tokens": mean_or_none([row.get("actual_llm_visual_tokens") for row in results]),
        "mean_original_input_sequence_length": mean_or_none([row.get("original_input_sequence_length") for row in results]),
        "mean_compacted_input_sequence_length": mean_or_none([row.get("compacted_input_sequence_length") for row in results]),
        "mean_sequence_length_reduction": mean_or_none([row.get("sequence_length_reduction") for row in results]),
    }
    return metrics


def run(args: argparse.Namespace) -> int:
    resolve_output_roots(args)
    original_tokens = args.image_grid_size * args.image_grid_size
    retained_tokens = args.token_budget if args.token_budget is not None else int(round(original_tokens * args.keep_ratio))
    retained_tokens = max(1, min(original_tokens, retained_tokens))
    if args.method == "full" and retained_tokens != original_tokens:
        raise BaselineError("Method 'full' must use keep ratio 1.0 / all visual tokens.")
    run_name = run_name_from_args(args, retained_tokens)
    run_dir = args.output_root / run_name
    figures_dir = args.figures_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "stdout.log"
    log_mode = "a" if args.resume and log_path.exists() else "w"
    with log_path.open(log_mode, encoding="utf-8") as log_handle:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(sys.stdout, log_handle), Tee(sys.stderr, log_handle)
        try:
            config = build_config(args, run_name, original_tokens, retained_tokens)
            with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=True)
            rows = load_manifest(args.manifest, args.limit)
            print(f"Loaded {len(rows)} samples from {args.manifest}")
            per_sample_path = run_dir / "per_sample_results.jsonl"
            existing_results: list[dict[str, Any]] = []
            malformed_existing_lines = 0
            if args.resume:
                existing_results, malformed_existing_lines = load_existing_results(per_sample_path, rows)
                if malformed_existing_lines:
                    print(f"Found {malformed_existing_lines} malformed existing JSONL line(s); rewriting valid rows before resume.")
                    rewrite_per_sample_results(per_sample_path, existing_results)
                print(f"Resume enabled: found {len(existing_results)}/{len(rows)} completed samples.")
            completed_ids = {str(result["sample_id"]) for result in existing_results}
            pending_rows = [row for row in rows if str(row["sample_id"]) not in completed_ids]
            if args.resume and not pending_rows:
                print("Resume found all requested samples complete; recomputing metrics without loading the model.")
                model_load_s = 0.0
                total_wall = sum(float(row.get("latency_s") or 0.0) for row in existing_results)
                metrics = (
                    aggregate_grounding(existing_results, total_wall, None)
                    if args.task == "grounding"
                    else aggregate_vqa(existing_results, total_wall, None)
                )
                metrics["model_load_latency_s"] = model_load_s
                metrics["original_visual_tokens"] = original_tokens
                metrics["retained_visual_tokens"] = retained_tokens
                metrics["keep_ratio"] = args.keep_ratio
                metrics["pruning_implementation"] = config["pruning_implementation"]
                metrics["pruning_location"] = config.get("pruning_location")
                metrics["resume_existing_samples"] = len(existing_results)
                metrics["resume_new_samples"] = 0
                with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                    json.dump(metrics, handle, indent=2, sort_keys=True)
                update_summary_table(config, metrics, args.tables_root)
                print(f"Saved run outputs to {run_dir}")
                print(f"Saved qualitative figures to {figures_dir}")
                return 0
            wandb_run = init_wandb(args, config, run_dir)
            model_load_start = time.perf_counter()
            offload_root = os.environ.get("VLM_PRUNING_OFFLOAD_ROOT")
            offload_dir = Path(offload_root) / run_name if offload_root else run_dir / "offload"
            torch, processor, model, device = load_model(args.model, offload_dir, args.quantization)
            model_load_s = time.perf_counter() - model_load_start
            print(f"Loaded model {args.model} on {device} in {model_load_s:.2f}s")
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            max_new_tokens = args.max_new_tokens or (48 if args.task == "grounding" else 8)
            results: list[dict[str, Any]] = list(existing_results)
            new_results: list[dict[str, Any]] = []
            figure_paths: list[Path] = []
            start_all = time.perf_counter()
            sample_mode = "a" if args.resume and per_sample_path.exists() else "w"
            with per_sample_path.open(sample_mode, encoding="utf-8") as sample_handle:
                for index, sample in enumerate(pending_rows):
                    sample_id = str(sample["sample_id"])
                    image_path = Path(sample["image_path"])
                    raw_image = Image.open(image_path).convert("RGB")
                    keep = retained_indices(args.method, retained_tokens, original_tokens, args.seed, sample_id, args.image_grid_size)
                    input_image = (
                        raw_image
                        if args.pruning_implementation == TRUE_REMOVAL_LABEL or args.method == "full"
                        else mask_image(raw_image, keep, args.image_grid_size)
                    )
                    prompt = make_prompt(args.task, sample, args.box_coordinate_mode)
                    sample_error = None
                    diagnostics: dict[str, Any] = {}
                    try:
                        if args.pruning_implementation == TRUE_REMOVAL_LABEL:
                            raw_output, generated_count, latency, diagnostics = generate_answer_true_removal(
                                torch,
                                processor,
                                model,
                                device,
                                input_image,
                                prompt,
                                max_new_tokens,
                                keep,
                                original_tokens,
                            )
                            if diagnostics["actual_llm_visual_tokens"] != retained_tokens:
                                raise BaselineError(
                                    f"actual_llm_visual_tokens={diagnostics['actual_llm_visual_tokens']} "
                                    f"but expected {retained_tokens}."
                                )
                            if args.method != "full" and diagnostics["compacted_input_sequence_length"] >= diagnostics["original_input_sequence_length"]:
                                raise BaselineError(
                                    "True removal did not reduce the compacted input sequence length for a pruned run."
                                )
                        else:
                            raw_output, generated_count, latency = generate_answer(
                                torch, processor, model, device, input_image, prompt, max_new_tokens
                            )
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower():
                            raise BaselineError(f"GPU memory is insufficient for {args.model}: {exc}") from exc
                        if args.pruning_implementation == TRUE_REMOVAL_LABEL:
                            raise BaselineError(f"True token removal generation failed for sample {sample_id}: {exc}") from exc
                        sample_error = f"{type(exc).__name__}: {exc}"
                        raw_output, generated_count, latency = "", 0, 0.0
                    if args.task == "grounding":
                        width = int(sample.get("width") or raw_image.size[0])
                        height = int(sample.get("height") or raw_image.size[1])
                        raw_box, clipped_box, parse_failure = parse_box(raw_output, width, height, args.box_coordinate_mode)
                        valid = clipped_box is not None and sample_error is None
                        if sample_error:
                            parse_failure = sample_error
                        iou = box_iou(clipped_box, sample["gt_box"]) if valid else 0.0
                        success = bool(valid and iou >= 0.5)
                        result = {
                            "sample_id": sample_id,
                            "image_path": str(image_path.as_posix()),
                            "image_id": sample.get("image_id"),
                            "query": sample.get("query"),
                            "method": args.method,
                            "keep_ratio": args.keep_ratio,
                            "original_token_count": original_tokens,
                            "retained_token_count": retained_tokens,
                            "pruning_implementation": config["pruning_implementation"],
                            "raw_model_output": raw_output,
                            "raw_parsed_box": raw_box,
                            "predicted_box": clipped_box,
                            "gt_box": sample["gt_box"],
                            "box_iou": iou,
                            "success_iou_0_5": success,
                            "valid": valid,
                            "invalid": not valid,
                            "parse_failure_reason": parse_failure,
                            "latency_s": latency,
                            "prefill_latency_s": None,
                            "decode_latency_s": None,
                            "generated_token_count": generated_count,
                            "retained_indices": keep,
                            "retained_indices_checksum": indices_checksum(keep),
                        }
                        result.update(diagnostics)
                        if index < args.max_figures:
                            fig_path = save_grounding_figure(
                                sample,
                                raw_image,
                                clipped_box,
                                keep,
                                args.image_grid_size,
                                args.method,
                                args.keep_ratio,
                                iou,
                                success,
                                figures_dir,
                            )
                            figure_paths.append(fig_path)
                    else:
                        norm_answer, extraction_failure = normalise_yes_no(raw_output)
                        target, target_failed = normalise_yes_no(str(sample["answer"]))
                        correct = bool(norm_answer is not None and target is not None and norm_answer == target)
                        invalid = bool(extraction_failure or sample_error)
                        result = {
                            "sample_id": sample_id,
                            "image_path": str(image_path.as_posix()),
                            "image_id": sample.get("image_id"),
                            "question": sample.get("question"),
                            "target_answer": sample.get("answer"),
                            "raw_model_answer": raw_output,
                            "normalised_answer": norm_answer,
                            "correct": correct,
                            "invalid": invalid,
                            "extraction_failure": bool(extraction_failure or target_failed),
                            "method": args.method,
                            "keep_ratio": args.keep_ratio,
                            "original_token_count": original_tokens,
                            "retained_token_count": retained_tokens,
                            "pruning_implementation": config["pruning_implementation"],
                            "latency_s": latency,
                            "prefill_latency_s": None,
                            "decode_latency_s": None,
                            "generated_token_count": generated_count,
                            "retained_indices": keep,
                            "retained_indices_checksum": indices_checksum(keep),
                        }
                        result.update(diagnostics)
                    results.append(result)
                    new_results.append(result)
                    sample_handle.write(json.dumps(result, ensure_ascii=True) + "\n")
                    completed_total = len(existing_results) + index + 1
                    if completed_total % 10 == 0 or completed_total == len(rows):
                        print(f"Processed {completed_total}/{len(rows)} samples")
            total_wall = time.perf_counter() - start_all
            if existing_results:
                total_wall += sum(float(row.get("latency_s") or 0.0) for row in existing_results)
            peak_memory_mb = None
            peak_reserved_memory_mb = None
            if torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
                peak_reserved_memory_mb = torch.cuda.max_memory_reserved() / (1024**2)
            metrics = (
                aggregate_grounding(results, total_wall, peak_memory_mb)
                if args.task == "grounding"
                else aggregate_vqa(results, total_wall, peak_memory_mb)
            )
            if args.pruning_implementation == TRUE_REMOVAL_LABEL:
                add_efficiency_estimates(metrics, model, torch)
                metrics["gpu_peak_memory_reserved_mb"] = peak_reserved_memory_mb
            metrics["model_load_latency_s"] = model_load_s
            metrics["configured_original_visual_tokens"] = original_tokens
            metrics["retained_visual_tokens"] = retained_tokens
            metrics["keep_ratio"] = args.keep_ratio
            metrics["pruning_implementation"] = config["pruning_implementation"]
            metrics["pruning_location"] = config.get("pruning_location")
            metrics["resume_existing_samples"] = len(existing_results)
            metrics["resume_new_samples"] = len(new_results)
            with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2, sort_keys=True)
            update_summary_table(config, metrics, args.tables_root)
            if wandb_run is not None:
                try:
                    import wandb

                    wandb.log(metrics)
                    for fig_path in figure_paths[: min(8, len(figure_paths))]:
                        wandb.log({"qualitative_grounding": wandb.Image(str(fig_path))})
                    wandb_run.finish()
                except Exception as exc:
                    print(f"W&B logging failed after run completion; local files are saved. Error: {type(exc).__name__}: {exc}")
            print(f"Saved run outputs to {run_dir}")
            print(f"Saved qualitative figures to {figures_dir}")
            return 0
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except BaselineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
