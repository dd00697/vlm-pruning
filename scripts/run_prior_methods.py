#!/usr/bin/env python
"""Run VisionZip and FastV-style prior-method experiments for LLaVA-1.5."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

import run_baseline as base


VISIONZIP_LABEL = "visionzip_hf_adapter"
FASTV_LABEL = "fastv_hf_adapter"
VISIONZIP_SOURCE = "JIA-Lab-research/VisionZip"
FASTV_SOURCE = "pkunlp-icler/FastV"
VISIONZIP_SHA = "8f86b55c6f000eb033e6912538af2dd7dcb30502"
FASTV_SHA = "d1659729b5bf1be225e99ee15783deeea80f63b1"
PRIOR_LOCATION = "method_specific_prior_adapter"
FASTV_ADAPTER_LOCATION = "selection_pass_layer_k_then_pre_llm_compaction"
VISIONZIP_ADAPTER_LOCATION = "vision_encoder_selection_then_pre_llm_compaction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["grounding", "vqa"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", default=base.DEFAULT_MODEL)
    parser.add_argument("--method", required=True, choices=["visionzip", "fastv"])
    parser.add_argument("--keep-ratio", type=float, required=True)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/runs_prior_methods"))
    parser.add_argument("--figures-root", type=Path, default=Path("experiments/figures_prior_methods"))
    parser.add_argument("--tables-root", type=Path, default=Path("experiments/tables"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "vlm-pruning"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--image-grid-size", type=int, default=base.DEFAULT_GRID)
    parser.add_argument("--max-figures", type=int, default=24)
    parser.add_argument("--quantization", default="none", choices=["none", "4bit", "8bit"])
    parser.add_argument(
        "--box-coordinate-mode",
        default="normalized_0_1",
        choices=["normalized_0_1", "normalized_0_1000", "absolute_pixels"],
    )
    parser.add_argument(
        "--visionzip-contextual-ratio",
        type=float,
        default=0.05,
        help="Fraction of original visual tokens used as contextual VisionZip tokens.",
    )
    parser.add_argument("--fastv-k", type=int, default=3, help="Run full visual tokens through this many LLM layers before selection.")
    return parser.parse_args()


def run_name_from_args(args: argparse.Namespace, retained_tokens: int) -> str:
    if args.run_name:
        return args.run_name
    keep = int(round(args.keep_ratio * 100))
    model_name = "llava15" if "llava-1.5" in args.model.lower() else "model"
    prefix = f"{args.dataset}500" if args.limit == 500 else args.dataset
    return f"{prefix}_{model_name}_{args.task}_{args.method}_keep{keep:03d}_prior_seed{args.seed}"


def source_sha(method: str) -> str:
    return VISIONZIP_SHA if method == "visionzip" else FASTV_SHA


def method_label(method: str) -> str:
    return VISIONZIP_LABEL if method == "visionzip" else FASTV_LABEL


def pruning_location(method: str) -> str:
    return VISIONZIP_ADAPTER_LOCATION if method == "visionzip" else FASTV_ADAPTER_LOCATION


def build_config(args: argparse.Namespace, run_name: str, original_tokens: int, retained_tokens: int) -> dict[str, Any]:
    contextual = visionzip_contextual_count(original_tokens, retained_tokens, args.visionzip_contextual_ratio)
    fastv_r = 1.0 - args.keep_ratio
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
        "pruning_implementation": method_label(args.method),
        "pruning_location": pruning_location(args.method),
        "seed": args.seed,
        "limit": args.limit,
        "image_grid_size": args.image_grid_size,
        "box_coordinate_mode": args.box_coordinate_mode if args.task == "grounding" else None,
        "quantization": args.quantization,
        "resume": bool(args.resume),
        "wandb_enabled": bool(args.wandb),
        "wandb_project": args.wandb_project,
        "third_party_source": VISIONZIP_SOURCE if args.method == "visionzip" else FASTV_SOURCE,
        "third_party_commit_sha": source_sha(args.method),
        "output_root": str(args.output_root),
        "figures_root": str(args.figures_root),
        "tables_root": str(args.tables_root),
        "visionzip_contextual_ratio": args.visionzip_contextual_ratio if args.method == "visionzip" else None,
        "visionzip_contextual_tokens": contextual if args.method == "visionzip" else None,
        "visionzip_dominant_tokens": (retained_tokens - contextual) if args.method == "visionzip" else None,
        "fastv_k": args.fastv_k if args.method == "fastv" else None,
        "fastv_r": fastv_r if args.method == "fastv" else None,
        "fastv_image_token_length": original_tokens if args.method == "fastv" else None,
    }


def init_wandb(args: argparse.Namespace, config: dict[str, Any], run_dir: Path) -> Any | None:
    if not args.wandb:
        return None
    try:
        import wandb

        return wandb.init(
            project=args.wandb_project,
            name=config["run_name"],
            config=config,
            dir=str(run_dir),
            reinit=True,
        )
    except Exception as exc:
        print(f"W&B logging unavailable; continuing with local logging ({type(exc).__name__}: {exc}).")
        return None


def move_inputs(torch: Any, processor: Any, model: Any, device: Any, image: Image.Image, prompt: str) -> dict[str, Any]:
    text_prompt = base.apply_chat_template(processor, prompt)
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    moved: dict[str, Any] = {}
    dtype = next(model.parameters()).dtype
    for key, value in inputs.items():
        if hasattr(value, "to"):
            if key == "pixel_values":
                value = value.to(device=device, dtype=dtype)
            else:
                value = value.to(device=device)
        moved[key] = value
    return moved


def image_positions_from_inputs(torch: Any, model: Any, input_ids: Any) -> Any:
    image_token_index = int(getattr(model.config, "image_token_index", 32000))
    return torch.nonzero(input_ids[0] == image_token_index, as_tuple=False).flatten()


def selected_vision_outputs(model: Any, pixel_values: Any, output_attentions: bool) -> tuple[Any, Any, Any, float, float]:
    import torch

    if output_attentions:
        set_eager_attention(model.model.vision_tower)
    vision_feature_layer = getattr(model.config, "vision_feature_layer", -2)
    vision_feature_select_strategy = getattr(model.config, "vision_feature_select_strategy", "default")
    base.cuda_sync(torch)
    vision_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.model.vision_tower(
            pixel_values,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=True,
        )
    base.cuda_sync(torch)
    vision_latency = time.perf_counter() - vision_start

    if isinstance(vision_feature_layer, int):
        selected = outputs.hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            selected = selected[:, 1:]
    else:
        hidden_states = [outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
        if vision_feature_select_strategy == "default":
            hidden_states = [state[:, 1:] for state in hidden_states]
        selected = torch.cat(hidden_states, dim=-1)

    base.cuda_sync(torch)
    projector_start = time.perf_counter()
    with torch.inference_mode():
        image_features = model.model.multi_modal_projector(selected)
    base.cuda_sync(torch)
    projector_latency = time.perf_counter() - projector_start
    return outputs, selected, image_features, vision_latency, projector_latency


def set_eager_attention(module: Any) -> None:
    candidates = [
        module,
        getattr(module, "config", None),
        getattr(module, "vision_model", None),
        getattr(getattr(module, "vision_model", None), "config", None),
    ]
    if hasattr(module, "modules"):
        for submodule in module.modules():
            candidates.append(submodule)
            candidates.append(getattr(submodule, "config", None))
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            setattr(candidate, "_attn_implementation", "eager")
        except Exception:
            pass


def visionzip_contextual_count(original_tokens: int, retained_tokens: int, contextual_ratio: float) -> int:
    if retained_tokens <= 1:
        return 0
    contextual = int(round(original_tokens * contextual_ratio))
    return max(1, min(contextual, retained_tokens - 1))


def visionzip_compress(
    torch: Any,
    model: Any,
    outputs: Any,
    retained_tokens: int,
    contextual_ratio: float,
) -> tuple[Any, list[int], dict[str, Any]]:
    selected_layer = getattr(model.config, "vision_feature_layer", -2)
    hidden_all = outputs.hidden_states[selected_layer] if isinstance(selected_layer, int) else outputs.hidden_states[-2]
    patch_hidden = hidden_all[:, 1:, :]
    original_tokens = int(patch_hidden.shape[1])
    contextual_count = visionzip_contextual_count(original_tokens, retained_tokens, contextual_ratio)
    dominant_count = retained_tokens - contextual_count

    attentions = getattr(outputs, "attentions", None)
    if attentions and attentions[selected_layer] is not None:
        cls_attention = attentions[selected_layer][:, :, 0, 1:]
        scores = cls_attention.sum(dim=1)
        scoring = "clip_cls_attention"
    else:
        scores = patch_hidden.norm(dim=-1)
        scoring = "hidden_norm_fallback"

    dominant_indices = scores.topk(dominant_count, dim=1).indices[0].sort().values
    keep_mask = torch.zeros(original_tokens, dtype=torch.bool, device=patch_hidden.device)
    keep_mask[dominant_indices] = True
    remaining_indices = torch.nonzero(~keep_mask, as_tuple=False).flatten()

    contextual_patch_indices = torch.empty(0, dtype=torch.long, device=patch_hidden.device)
    if contextual_count > 0 and int(remaining_indices.numel()) > 0:
        step = max(1, int(remaining_indices.numel()) // contextual_count)
        rel_targets = torch.arange(0, int(remaining_indices.numel()), step, device=patch_hidden.device)[:contextual_count]
        contextual_patch_indices = remaining_indices[rel_targets]
        remaining_hidden = patch_hidden[:, remaining_indices, :]
        target_hidden = patch_hidden[:, contextual_patch_indices, :]
        merge_rel_mask = torch.ones(int(remaining_indices.numel()), dtype=torch.bool, device=patch_hidden.device)
        merge_rel_mask[rel_targets] = False
        if bool(merge_rel_mask.any()):
            target_metric = target_hidden / target_hidden.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            merge_hidden = remaining_hidden[:, merge_rel_mask, :]
            merge_metric = merge_hidden / merge_hidden.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            similarity = torch.bmm(merge_metric, target_metric.transpose(1, 2))
            assignments = similarity.argmax(dim=2)
            one_hot = torch.zeros(
                merge_hidden.shape[0],
                merge_hidden.shape[1],
                contextual_count,
                dtype=merge_hidden.dtype,
                device=merge_hidden.device,
            )
            one_hot.scatter_(2, assignments.unsqueeze(-1), 1)
            counts = one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
            aggregated = torch.bmm(one_hot.transpose(1, 2), merge_hidden) / counts
            contextual_hidden = target_hidden + aggregated
        else:
            contextual_hidden = target_hidden
    else:
        contextual_hidden = patch_hidden[:, :0, :]

    dominant_hidden = patch_hidden[:, dominant_indices, :]
    compressed_hidden = torch.cat([dominant_hidden, contextual_hidden], dim=1)
    with torch.inference_mode():
        compressed_features = model.model.multi_modal_projector(compressed_hidden)
    retained_indices = torch.cat([dominant_indices, contextual_patch_indices]).sort().values.tolist()
    diagnostics = {
        "visionzip_dominant_tokens": int(dominant_count),
        "visionzip_contextual_tokens": int(contextual_count),
        "visionzip_scoring": scoring,
        "visionzip_contextual_note": "contextual tokens are aggregated from non-dominant patch features",
        "retained_indices_semantics": "dominant_patch_indices_plus_contextual_target_patch_indices",
    }
    return compressed_features, [int(idx) for idx in retained_indices], diagnostics


def build_compacted_inputs(
    torch: Any,
    model: Any,
    moved: dict[str, Any],
    image_features: Any,
    expected_original_tokens: int,
    retained_indices: list[int],
    diagnostics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_ids = moved["input_ids"]
    attention_mask = moved.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    image_positions = image_positions_from_inputs(torch, model, input_ids)
    original_visual_tokens = int(image_positions.numel())
    if original_visual_tokens != expected_original_tokens:
        raise base.BaselineError(
            f"Configured visual-token grid expects {expected_original_tokens} tokens, but processor/model produced {original_visual_tokens}."
        )
    if image_features.ndim != 3 or image_features.shape[0] != 1:
        raise base.BaselineError(f"Expected image features with shape [1, tokens, hidden], got {tuple(image_features.shape)}")
    retained_tokens = int(image_features.shape[1])
    image_start = int(image_positions[0].item())
    image_end = int(image_positions[-1].item()) + 1
    dtype = next(model.parameters()).dtype
    selection_start = time.perf_counter()
    text_embeds = model.get_input_embeddings()(input_ids).to(dtype=dtype)
    compacted_inputs_embeds = torch.cat(
        [
            text_embeds[:, :image_start, :],
            image_features.to(device=text_embeds.device, dtype=text_embeds.dtype),
            text_embeds[:, image_end:, :],
        ],
        dim=1,
    )
    compacted_attention_mask = torch.cat(
        [
            attention_mask[:, :image_start],
            torch.ones((attention_mask.shape[0], retained_tokens), dtype=attention_mask.dtype, device=attention_mask.device),
            attention_mask[:, image_end:],
        ],
        dim=1,
    )
    image_token_id = int(getattr(model.config, "image_token_index", 32000))
    compacted_input_ids = torch.cat(
        [
            input_ids[:, :image_start],
            torch.full((input_ids.shape[0], retained_tokens), image_token_id, dtype=input_ids.dtype, device=input_ids.device),
            input_ids[:, image_end:],
        ],
        dim=1,
    )
    base.cuda_sync(torch)
    pruning_selection_latency = time.perf_counter() - selection_start
    original_sequence_length = int(input_ids.shape[1])
    compacted_sequence_length = int(compacted_input_ids.shape[1])
    diagnostics.update(
        {
            "pruning_selection_latency_s": pruning_selection_latency + float(diagnostics.get("pruning_selection_latency_s", 0.0) or 0.0),
            "original_visual_tokens": original_visual_tokens,
            "requested_keep_ratio": retained_tokens / original_visual_tokens,
            "expected_retained_visual_tokens": retained_tokens,
            "actual_retained_visual_tokens": retained_tokens,
            "actual_llm_visual_tokens": retained_tokens,
            "original_input_sequence_length": original_sequence_length,
            "compacted_input_sequence_length": compacted_sequence_length,
            "sequence_length_reduction": original_sequence_length - compacted_sequence_length,
            "retained_indices_checksum": base.indices_checksum(retained_indices),
            "pruning_location": diagnostics.get("pruning_location", PRIOR_LOCATION),
        }
    )
    compacted = {
        "input_ids": compacted_input_ids,
        "inputs_embeds": compacted_inputs_embeds,
        "attention_mask": compacted_attention_mask,
    }
    return compacted, diagnostics


def fastv_set_eager_attention(model: Any) -> str:
    language_model = model.model.language_model
    previous = getattr(language_model.config, "_attn_implementation", None)
    try:
        language_model.config._attn_implementation = "eager"
    except Exception:
        pass
    return str(previous)


def run_llama_layer(layer: Any, hidden_states: Any, causal_mask: Any, position_ids: Any, position_embeddings: Any) -> tuple[Any, Any]:
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    attn_output, attn_weights = layer.self_attn(
        hidden_states=hidden_states,
        attention_mask=causal_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        use_cache=False,
    )
    hidden_states = residual + attn_output
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states, attn_weights


def fastv_select_indices(
    torch: Any,
    model: Any,
    moved: dict[str, Any],
    image_features: Any,
    retained_tokens: int,
    fastv_k: int,
) -> tuple[list[int], dict[str, Any]]:
    from transformers.models.llama.modeling_llama import create_causal_mask

    language_model = model.model.language_model
    input_ids = moved["input_ids"]
    attention_mask = moved.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    image_positions = image_positions_from_inputs(torch, model, input_ids)
    original_visual_tokens = int(image_positions.numel())
    image_start = int(image_positions[0].item())
    dtype = next(model.parameters()).dtype
    text_embeds = model.get_input_embeddings()(input_ids).to(dtype=dtype).clone()
    text_embeds[0, image_positions, :] = image_features[0].to(device=text_embeds.device, dtype=text_embeds.dtype)
    hidden_states = text_embeds
    position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
    last_attention = None
    fastv_k = max(1, min(int(fastv_k), len(language_model.layers) - 1))

    previous_attn_impl = fastv_set_eager_attention(model)
    base.cuda_sync(torch)
    start = time.perf_counter()
    with torch.inference_mode():
        for layer_idx in range(fastv_k):
            causal_mask = create_causal_mask(
                config=language_model.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )
            position_embeddings = language_model.rotary_emb(hidden_states, position_ids=position_ids)
            hidden_states, attn_weights = run_llama_layer(
                language_model.layers[layer_idx],
                hidden_states,
                causal_mask,
                position_ids,
                position_embeddings,
            )
            if layer_idx == fastv_k - 1:
                last_attention = attn_weights
    base.cuda_sync(torch)
    selection_latency = time.perf_counter() - start

    if last_attention is None:
        raise base.BaselineError("FastV selection did not produce attention weights.")
    mean_attention = last_attention.mean(dim=1)[0]
    last_token_attention = mean_attention[-1]
    image_attention = last_token_attention[image_positions]
    keep_tensor = image_attention.topk(retained_tokens).indices.sort().values
    retained_indices = [int(idx) for idx in keep_tensor.tolist()]
    diagnostics = {
        "fastv_k": fastv_k,
        "fastv_r": 1.0 - (retained_tokens / original_visual_tokens),
        "fastv_image_token_start_index": image_start,
        "fastv_image_token_length": original_visual_tokens,
        "fastv_attention_layer_index": fastv_k - 1,
        "fastv_prune_before_layer_index": fastv_k,
        "fastv_previous_attention_implementation": previous_attn_impl,
        "fastv_selection_prefill_latency_s": selection_latency,
        "fastv_adapter_note": "Pinned FastV source requires an older transformers/tokenizers stack; this adapter uses the same early-layer attention selection, then runs compacted HF LLaVA generation.",
        "retained_indices_semantics": "top_visual_indices_by_last_token_attention_after_fastv_k_layers",
        "pruning_selection_latency_s": selection_latency,
    }
    return retained_indices, diagnostics


def generate_prior_answer(
    torch: Any,
    processor: Any,
    model: Any,
    device: Any,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    method: str,
    retained_tokens: int,
    original_tokens: int,
    contextual_ratio: float,
    fastv_k: int,
) -> tuple[str, int, float, dict[str, Any], list[int]]:
    moved = move_inputs(torch, processor, model, device, image, prompt)
    output_attentions = method == "visionzip"
    outputs, _selected, full_image_features, vision_latency, projector_latency = selected_vision_outputs(
        model, moved["pixel_values"], output_attentions=output_attentions
    )
    diagnostics: dict[str, Any] = {
        "vision_encoder_latency_s": vision_latency,
        "projector_latency_s": projector_latency,
        "pruning_implementation": method_label(method),
        "pruning_location": pruning_location(method),
        "third_party_source": VISIONZIP_SOURCE if method == "visionzip" else FASTV_SOURCE,
        "third_party_commit_sha": source_sha(method),
    }
    if method == "visionzip":
        compressed_features, retained_indices, method_diagnostics = visionzip_compress(
            torch, model, outputs, retained_tokens, contextual_ratio
        )
        diagnostics.update(method_diagnostics)
        image_features_for_generation = compressed_features
    else:
        retained_indices, method_diagnostics = fastv_select_indices(
            torch, model, moved, full_image_features, retained_tokens, fastv_k
        )
        diagnostics.update(method_diagnostics)
        keep_tensor = torch.tensor(retained_indices, dtype=torch.long, device=full_image_features.device)
        image_features_for_generation = full_image_features[:, keep_tensor, :]

    compacted, diagnostics = build_compacted_inputs(
        torch,
        model,
        moved,
        image_features_for_generation,
        original_tokens,
        retained_indices,
        diagnostics,
    )
    raw_output, generated_count, prefill_latency, decode_latency = base.greedy_decode_from_embeds(
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
    return raw_output, generated_count, total_latency, diagnostics, retained_indices


def wrap_text(text: str, limit: int = 92) -> str:
    text = str(text or "")
    lines = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        lines.append(text[:cut])
        text = text[cut:].strip()
    if text:
        lines.append(text)
    return "\n".join(lines)


def save_vqa_figure(
    sample: dict[str, Any],
    raw_image: Image.Image,
    retained: list[int],
    grid: int,
    method: str,
    keep_ratio: float,
    prediction: str | None,
    correct: bool,
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
                draw.rectangle(xy, fill=(0, 0, 0, 55))
    status = "correct" if correct else "wrong"
    title = (
        f"Q: {wrap_text(sample.get('question', ''), 80)}\n"
        f"target={sample.get('answer')} pred={prediction} | {status}"
    )
    base.draw_label(draw, (8, 8), title, (0, 0, 0, 255))
    safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(sample["sample_id"]))
    keep = int(round(keep_ratio * 100))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{safe_id}_{method}_keep{keep:03d}_{status}.png"
    canvas.save(path)
    return path


def prior_efficiency_estimates(metrics: dict[str, Any], config: dict[str, Any], model: Any, torch: Any) -> None:
    compacted = metrics.get("mean_compacted_input_sequence_length")
    full = metrics.get("mean_original_input_sequence_length")
    if compacted is None or full is None:
        return
    compacted_flops = base.llm_prefill_flops(float(compacted), model)
    full_flops = base.llm_prefill_flops(float(full), model)
    metrics["theoretical_llm_prefill_flops"] = compacted_flops
    metrics["theoretical_llm_prefill_flops_full_reference"] = full_flops
    metrics["theoretical_llm_prefill_flops_ratio_vs_full"] = (
        compacted_flops / full_flops if compacted_flops is not None and full_flops else None
    )
    if metrics["theoretical_llm_prefill_flops_ratio_vs_full"] is not None:
        metrics["theoretical_llm_prefill_flops_reduction_vs_full"] = 1.0 - metrics["theoretical_llm_prefill_flops_ratio_vs_full"]
    else:
        metrics["theoretical_llm_prefill_flops_reduction_vs_full"] = None
    dtype = next(model.parameters()).dtype
    metrics["kv_cache_tokens_estimate"] = compacted
    metrics["kv_cache_memory_estimate_mb"] = base.kv_cache_memory_mb(float(compacted), model, dtype, torch)
    if config["method"] == "fastv" and full_flops:
        layers = int(model.config.text_config.num_hidden_layers)
        k = int(config["fastv_k"])
        compacted_per_layer = compacted_flops / layers if compacted_flops is not None else None
        full_per_layer = full_flops / layers
        original_style = (k * full_per_layer) + ((layers - k) * (compacted_per_layer or 0.0))
        adapter_style = (k * full_per_layer) + (compacted_flops or 0.0)
        metrics["theoretical_fastv_original_style_prefill_flops"] = original_style
        metrics["theoretical_fastv_original_style_flops_ratio_vs_full"] = original_style / full_flops
        metrics["theoretical_fastv_adapter_prefill_flops"] = adapter_style
        metrics["theoretical_fastv_adapter_flops_ratio_vs_full"] = adapter_style / full_flops


def first_non_none(values: list[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def min_or_none(values: list[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return min(clean) if clean else None


def max_or_none(values: list[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return max(clean) if clean else None


def enrich_prior_metrics(
    metrics: dict[str, Any],
    config: dict[str, Any],
    results: list[dict[str, Any]],
    retained_tokens: int,
    original_tokens: int,
) -> None:
    metrics["fastv_selection_prefill_latency_s"] = base.mean_or_none(
        [row.get("fastv_selection_prefill_latency_s") for row in results]
    )
    metrics["fastv_image_token_start_index"] = base.mean_or_none(
        [row.get("fastv_image_token_start_index") for row in results]
    )
    metrics["fastv_image_token_start_index_min"] = min_or_none(
        [row.get("fastv_image_token_start_index") for row in results]
    )
    metrics["fastv_image_token_start_index_max"] = max_or_none(
        [row.get("fastv_image_token_start_index") for row in results]
    )
    metrics["fastv_attention_layer_index"] = first_non_none(
        [row.get("fastv_attention_layer_index") for row in results]
    )
    metrics["fastv_prune_before_layer_index"] = first_non_none(
        [row.get("fastv_prune_before_layer_index") for row in results]
    )
    metrics["visionzip_dominant_tokens"] = base.mean_or_none(
        [row.get("visionzip_dominant_tokens") for row in results]
    )
    metrics["visionzip_contextual_tokens"] = base.mean_or_none(
        [row.get("visionzip_contextual_tokens") for row in results]
    )
    metrics["configured_original_visual_tokens"] = original_tokens
    metrics["retained_visual_tokens"] = retained_tokens
    metrics["keep_ratio"] = config["keep_ratio"]
    metrics["pruning_implementation"] = config["pruning_implementation"]
    metrics["pruning_location"] = config["pruning_location"]
    metrics["third_party_source"] = config["third_party_source"]
    metrics["third_party_commit_sha"] = config["third_party_commit_sha"]


def update_prior_tables(config: dict[str, Any], metrics: dict[str, Any], tables_root: Path) -> None:
    tables_root.mkdir(parents=True, exist_ok=True)
    table = tables_root / ("prior_methods_grounding.csv" if config["task"] == "grounding" else "prior_methods_vqa.csv")
    row = dict(config)
    row.update(metrics)
    write_csv_upsert(table, row, "run_name")
    update_prior_efficiency_table(config, metrics, tables_root)
    write_prior_summary(tables_root)


def write_csv_upsert(path: Path, row: dict[str, Any], key: str) -> None:
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
        existing = [item for item in existing if item.get(key) != str(row.get(key))]
    existing.append({name: "" if value is None else value for name, value in row.items()})
    fieldnames = sorted({field for item in existing for field in item.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)


def update_prior_efficiency_table(config: dict[str, Any], metrics: dict[str, Any], tables_root: Path) -> None:
    keys = [
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
        "third_party_source",
        "third_party_commit_sha",
        "visionzip_dominant_tokens",
        "visionzip_contextual_tokens",
        "fastv_k",
        "fastv_r",
        "fastv_image_token_length",
        "fastv_image_token_start_index",
        "fastv_image_token_start_index_min",
        "fastv_image_token_start_index_max",
        "fastv_attention_layer_index",
        "fastv_prune_before_layer_index",
        "fastv_selection_prefill_latency_s",
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
        "theoretical_llm_prefill_flops_ratio_vs_full",
        "theoretical_fastv_original_style_flops_ratio_vs_full",
        "theoretical_fastv_adapter_flops_ratio_vs_full",
    ]
    row = {key: config.get(key, metrics.get(key)) for key in keys}
    write_csv_upsert(tables_root / "prior_methods_efficiency.csv", row, "run_name")


def write_prior_summary(tables_root: Path) -> None:
    grounding = base.read_csv_rows(tables_root / "prior_methods_grounding.csv")
    vqa = base.read_csv_rows(tables_root / "prior_methods_vqa.csv")
    efficiency = base.read_csv_rows(tables_root / "prior_methods_efficiency.csv")
    true_grounding = base.read_csv_rows(tables_root / "true_removal_grounding.csv")
    true_vqa = base.read_csv_rows(tables_root / "true_removal_vqa.csv")
    true_full_by_task = {
        row.get("task"): row
        for row in true_grounding + true_vqa
        if row.get("method") == "full" and str(row.get("keep_ratio")) in {"1.0", "1", "1.00"}
    }

    quality_rows = []
    for row in grounding + vqa:
        task = row.get("task")
        quality = base.as_float(row.get("mean_iou")) if task == "grounding" else base.as_float(row.get("vqa_accuracy"))
        accuracy = base.as_float(row.get("accuracy_at_0.5") or row.get("accuracy_iou_0_5")) if task == "grounding" else base.as_float(row.get("accuracy"))
        full = true_full_by_task.get(task)
        full_quality = None
        if full:
            full_quality = base.as_float(full.get("mean_iou")) if task == "grounding" else base.as_float(full.get("vqa_accuracy"))
        quality_rows.append(
            {
                "run_name": row.get("run_name"),
                "task": task,
                "method": row.get("method"),
                "keep_ratio": row.get("keep_ratio"),
                "tokens": row.get("actual_llm_visual_tokens") or row.get("token_budget"),
                "quality": quality if quality is not None else "",
                "accuracy": accuracy if accuracy is not None else "",
                "delta_vs_true_full": (quality - full_quality) if quality is not None and full_quality is not None else "",
            }
        )

    efficiency_rows = []
    for row in efficiency:
        efficiency_rows.append(
            {
                "run_name": row.get("run_name"),
                "task": row.get("task"),
                "method": row.get("method"),
                "keep_ratio": row.get("keep_ratio"),
                "tokens": row.get("actual_llm_visual_tokens"),
                "seq_len": row.get("mean_compacted_input_sequence_length"),
                "latency_s": row.get("mean_latency_s"),
                "prefill_s": row.get("llm_prefill_latency_s"),
                "memory_mb": row.get("gpu_peak_memory_allocated_mb"),
                "flops_ratio": row.get("theoretical_llm_prefill_flops_ratio_vs_full"),
                "fastv_original_style_flops_ratio": row.get("theoretical_fastv_original_style_flops_ratio_vs_full"),
            }
        )

    lines = [
        "# Prior Methods 500-Example Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "These runs evaluate VisionZip and FastV adapters against the fixed 500-example baseline scope.",
        "",
        "## Quality",
        "",
        *base.markdown_table(
            quality_rows,
            ["run_name", "task", "method", "keep_ratio", "tokens", "quality", "accuracy", "delta_vs_true_full"],
        ),
        "",
        "## Efficiency",
        "",
        *base.markdown_table(
            efficiency_rows,
            [
                "run_name",
                "task",
                "method",
                "keep_ratio",
                "tokens",
                "seq_len",
                "latency_s",
                "prefill_s",
                "memory_mb",
                "flops_ratio",
                "fastv_original_style_flops_ratio",
            ],
        ),
        "",
        "## Method Notes",
        "",
        "- VisionZip uses the pinned VisionZip selection idea adapted to HF LLaVA: dominant CLIP-attention tokens plus contextual aggregated tokens.",
        "- FastV uses the pinned FastV early-layer last-token attention selector adapted to HF LLaVA, then runs compacted generation through the existing runner.",
        "- FastV has both adapter FLOPs and original-style FLOPs fields because the local adapter avoids replacing the installed transformers stack.",
        "",
    ]
    with (tables_root / "prior_methods_500_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run(args: argparse.Namespace) -> int:
    original_tokens = args.image_grid_size * args.image_grid_size
    retained_tokens = args.token_budget if args.token_budget is not None else int(round(original_tokens * args.keep_ratio))
    retained_tokens = max(1, min(original_tokens, retained_tokens))
    run_name = run_name_from_args(args, retained_tokens)
    run_dir = args.output_root / run_name
    figures_dir = args.figures_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "stdout.log"
    log_mode = "a" if args.resume and log_path.exists() else "w"
    with log_path.open(log_mode, encoding="utf-8") as log_handle:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = base.Tee(sys.stdout, log_handle), base.Tee(sys.stderr, log_handle)
        try:
            config = build_config(args, run_name, original_tokens, retained_tokens)
            with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=True)
            rows = base.load_manifest(args.manifest, args.limit)
            print(f"Loaded {len(rows)} samples from {args.manifest}")
            per_sample_path = run_dir / "per_sample_results.jsonl"
            existing_results: list[dict[str, Any]] = []
            malformed_existing_lines = 0
            if args.resume:
                existing_results, malformed_existing_lines = base.load_existing_results(per_sample_path, rows)
                if malformed_existing_lines:
                    print(f"Found {malformed_existing_lines} malformed existing JSONL line(s); rewriting valid rows before resume.")
                    base.rewrite_per_sample_results(per_sample_path, existing_results)
                print(f"Resume enabled: found {len(existing_results)}/{len(rows)} completed samples.")
            completed_ids = {str(result["sample_id"]) for result in existing_results}
            pending_rows = [row for row in rows if str(row["sample_id"]) not in completed_ids]
            if args.resume and not pending_rows:
                print("Resume found all requested samples complete; recomputing metrics without loading the model.")
                previous_metrics: dict[str, Any] = {}
                metrics_path = run_dir / "metrics.json"
                if metrics_path.exists():
                    with metrics_path.open("r", encoding="utf-8") as handle:
                        previous_metrics = json.load(handle)
                total_wall = sum(float(row.get("latency_s") or 0.0) for row in existing_results)
                metrics = (
                    base.aggregate_grounding(existing_results, total_wall, None)
                    if args.task == "grounding"
                    else base.aggregate_vqa(existing_results, total_wall, None)
                )
                enrich_prior_metrics(metrics, config, existing_results, retained_tokens, original_tokens)
                metrics["resume_existing_samples"] = len(existing_results)
                metrics["resume_new_samples"] = 0
                preserved = dict(previous_metrics)
                preserved.update({key: value for key, value in metrics.items() if value is not None})
                metrics = preserved
                with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                    json.dump(metrics, handle, indent=2, sort_keys=True)
                update_prior_tables(config, metrics, args.tables_root)
                return 0

            wandb_run = init_wandb(args, config, run_dir)
            model_load_start = time.perf_counter()
            offload_root = os.environ.get("VLM_PRUNING_OFFLOAD_ROOT")
            offload_dir = Path(offload_root) / run_name if offload_root else run_dir / "offload"
            torch, processor, model, device = base.load_model(args.model, offload_dir, args.quantization)
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
                    prompt = base.make_prompt(args.task, sample, args.box_coordinate_mode)
                    sample_error = None
                    diagnostics: dict[str, Any] = {}
                    retained_indices: list[int] = []
                    try:
                        raw_output, generated_count, latency, diagnostics, retained_indices = generate_prior_answer(
                            torch,
                            processor,
                            model,
                            device,
                            raw_image,
                            prompt,
                            max_new_tokens,
                            args.method,
                            retained_tokens,
                            original_tokens,
                            args.visionzip_contextual_ratio,
                            args.fastv_k,
                        )
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower():
                            raise base.BaselineError(f"GPU memory is insufficient for {args.model}: {exc}") from exc
                        raise base.BaselineError(f"{args.method} generation failed for sample {sample_id}: {type(exc).__name__}: {exc}") from exc

                    if args.task == "grounding":
                        width = int(sample.get("width") or raw_image.size[0])
                        height = int(sample.get("height") or raw_image.size[1])
                        raw_box, clipped_box, parse_failure = base.parse_box(raw_output, width, height, args.box_coordinate_mode)
                        valid = clipped_box is not None and sample_error is None
                        iou = base.box_iou(clipped_box, sample["gt_box"]) if valid else 0.0
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
                            "prefill_latency_s": diagnostics.get("llm_prefill_latency_s"),
                            "decode_latency_s": diagnostics.get("decode_latency_s"),
                            "generated_token_count": generated_count,
                            "retained_indices": retained_indices,
                            "retained_indices_checksum": base.indices_checksum(retained_indices),
                        }
                        result.update(diagnostics)
                        if index < args.max_figures:
                            fig_path = base.save_grounding_figure(
                                sample,
                                raw_image,
                                clipped_box,
                                retained_indices,
                                args.image_grid_size,
                                args.method,
                                args.keep_ratio,
                                iou,
                                success,
                                figures_dir,
                            )
                            figure_paths.append(fig_path)
                    else:
                        norm_answer, extraction_failure = base.normalise_yes_no(raw_output)
                        target, target_failed = base.normalise_yes_no(str(sample["answer"]))
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
                            "prefill_latency_s": diagnostics.get("llm_prefill_latency_s"),
                            "decode_latency_s": diagnostics.get("decode_latency_s"),
                            "generated_token_count": generated_count,
                            "retained_indices": retained_indices,
                            "retained_indices_checksum": base.indices_checksum(retained_indices),
                        }
                        result.update(diagnostics)
                        if index < args.max_figures or (not correct and len(figure_paths) < args.max_figures):
                            fig_path = save_vqa_figure(
                                sample,
                                raw_image,
                                retained_indices,
                                args.image_grid_size,
                                args.method,
                                args.keep_ratio,
                                norm_answer,
                                correct,
                                figures_dir,
                            )
                            figure_paths.append(fig_path)
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
                base.aggregate_grounding(results, total_wall, peak_memory_mb)
                if args.task == "grounding"
                else base.aggregate_vqa(results, total_wall, peak_memory_mb)
            )
            enrich_prior_metrics(metrics, config, results, retained_tokens, original_tokens)
            prior_efficiency_estimates(metrics, config, model, torch)
            metrics["gpu_peak_memory_reserved_mb"] = peak_reserved_memory_mb
            metrics["model_load_latency_s"] = model_load_s
            metrics["resume_existing_samples"] = len(existing_results)
            metrics["resume_new_samples"] = len(new_results)
            with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2, sort_keys=True)
            update_prior_tables(config, metrics, args.tables_root)
            if wandb_run is not None:
                try:
                    import wandb

                    wandb.log(metrics)
                    image_key = "qualitative_grounding" if args.task == "grounding" else "qualitative_vqa"
                    for fig_path in figure_paths[: min(8, len(figure_paths))]:
                        wandb.log({image_key: wandb.Image(str(fig_path))})
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
    except base.BaselineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
