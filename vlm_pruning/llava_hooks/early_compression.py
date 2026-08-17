"""Contiguous pre-projector spatial compression for the hardware-aware campaign.

The implementation is intentionally query independent.  A 24x24 CLIP patch
map is integrated over an output raster with exact rectangular overlap weights,
then projected once and passed through the language model without decoder
segmentation.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

from vlm_pruning.efficiency import cache_tensor_bytes


class EarlyCompressionError(RuntimeError):
    """Raised when the fixed early-compression contract is violated."""


def _checksum(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cuda_sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def nearest_cell_anchors(input_grid: int, output_grid: int) -> list[int]:
    """Return raster-ordered nearest input-cell anchors with lower-index ties.

    Cell centres are represented as exact rationals.  For an output-bin centre
    exactly halfway between two input centres, integer floor division selects
    the lower cell, which is the registered deterministic tie rule.
    """

    if input_grid <= 0 or output_grid <= 0 or output_grid > input_grid:
        raise EarlyCompressionError("Pooling grids must satisfy 0 < output <= input")
    axis: list[int] = []
    denominator = 2 * output_grid
    for output_index in range(output_grid):
        numerator = (2 * output_index + 1) * input_grid - output_grid
        anchor = numerator // denominator
        axis.append(max(0, min(input_grid - 1, int(anchor))))
    if len(set(axis)) != output_grid:
        raise EarlyCompressionError("Nearest-cell pooling anchors are not unique")
    anchors = [row * input_grid + column for row in axis for column in axis]
    if anchors != sorted(anchors) or len(set(anchors)) != output_grid * output_grid:
        raise EarlyCompressionError("Pooling anchors are not unique raster order")
    return anchors


def axis_area_overlap(torch: Any, input_grid: int, output_grid: int, *, device: Any) -> Any:
    """Exact 1-D overlap lengths between unit input cells and output bins."""

    if input_grid <= 0 or output_grid <= 0 or output_grid > input_grid:
        raise EarlyCompressionError("Pooling grids must satisfy 0 < output <= input")
    dtype = torch.float64
    input_left = torch.arange(input_grid, device=device, dtype=dtype)
    input_right = input_left + 1.0
    output_left = (
        torch.arange(output_grid, device=device, dtype=dtype) * float(input_grid) / float(output_grid)
    )
    output_right = (
        torch.arange(1, output_grid + 1, device=device, dtype=dtype)
        * float(input_grid)
        / float(output_grid)
    )
    overlap = torch.minimum(output_right[:, None], input_right[None, :]) - torch.maximum(
        output_left[:, None], input_left[None, :]
    )
    overlap = torch.clamp(overlap, min=0.0)
    column_mass = overlap.sum(dim=0)
    row_mass = overlap.sum(dim=1)
    if not torch.allclose(column_mass, torch.ones_like(column_mass), atol=1e-12, rtol=0.0):
        raise EarlyCompressionError("Area bins do not partition every input cell exactly")
    expected_width = torch.full_like(row_mass, float(input_grid) / float(output_grid))
    if not torch.allclose(row_mass, expected_width, atol=1e-12, rtol=0.0):
        raise EarlyCompressionError("Area-bin widths are inconsistent")
    return overlap


@dataclass(frozen=True)
class SpatialPoolEvidence:
    input_grid: int
    output_grid: int
    input_mass: float
    output_mass: float
    output_masses: list[float]
    anchor_indices: list[int]
    anchor_checksum: str
    overlap_checksum: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_grid": self.input_grid,
            "output_grid": self.output_grid,
            "input_mass": self.input_mass,
            "output_mass": self.output_mass,
            "output_masses": self.output_masses,
            "anchor_indices": self.anchor_indices,
            "anchor_checksum": self.anchor_checksum,
            "overlap_checksum": self.overlap_checksum,
        }


def area_weighted_spatial_pool(
    torch: Any,
    features: Any,
    *,
    input_grid: int = 24,
    output_grid: int,
) -> tuple[Any, SpatialPoolEvidence]:
    """Pool ``[batch, input_grid**2, hidden]`` using geometric area weights."""

    if features.ndim != 3:
        raise EarlyCompressionError("Vision features must have shape [batch, patches, hidden]")
    if int(features.shape[1]) != input_grid * input_grid:
        raise EarlyCompressionError(
            f"Expected {input_grid * input_grid} vision patches, got {features.shape[1]}"
        )
    overlap = axis_area_overlap(torch, input_grid, output_grid, device=features.device)
    normalised = (overlap / (float(input_grid) / float(output_grid))).to(
        dtype=torch.float32
    )
    feature_map = features.reshape(
        int(features.shape[0]), input_grid, input_grid, int(features.shape[-1])
    )
    pooled_fp32 = torch.einsum(
        "ri,bijd,cj->brcd",
        normalised,
        feature_map.to(dtype=torch.float32),
        normalised,
    )
    pooled = pooled_fp32.reshape(
        int(features.shape[0]), output_grid * output_grid, int(features.shape[-1])
    ).to(dtype=features.dtype)
    unit_mass = torch.ones(
        (input_grid, input_grid), device=features.device, dtype=torch.float64
    )
    output_mass_map = torch.einsum("ri,ij,cj->rc", overlap, unit_mass, overlap)
    output_masses = [float(value) for value in output_mass_map.detach().cpu().reshape(-1).tolist()]
    output_mass = float(sum(output_masses))
    input_mass = float(input_grid * input_grid)
    if not math.isclose(output_mass, input_mass, rel_tol=0.0, abs_tol=1e-9):
        raise EarlyCompressionError(
            f"Pooling mass is not conserved: input={input_mass}, output={output_mass}"
        )
    anchors = nearest_cell_anchors(input_grid, output_grid)
    rounded_overlap = [
        [round(float(value), 15) for value in row]
        for row in overlap.detach().cpu().tolist()
    ]
    evidence = SpatialPoolEvidence(
        input_grid=input_grid,
        output_grid=output_grid,
        input_mass=input_mass,
        output_mass=output_mass,
        output_masses=output_masses,
        anchor_indices=anchors,
        anchor_checksum=_checksum(anchors),
        overlap_checksum=_checksum(rounded_overlap),
    )
    return pooled, evidence


def _selected_vision_hidden(torch: Any, model: Any, pixel_values: Any) -> Any:
    vision_feature_layer = getattr(model.config, "vision_feature_layer", -2)
    strategy = getattr(model.config, "vision_feature_select_strategy", "default")
    image_outputs = model.model.vision_tower(
        pixel_values, output_hidden_states=True, return_dict=True
    )
    if isinstance(vision_feature_layer, int):
        selected = image_outputs.hidden_states[vision_feature_layer]
        if strategy == "default":
            selected = selected[:, 1:]
    else:
        hidden_states = [image_outputs.hidden_states[index] for index in vision_feature_layer]
        if strategy == "default":
            hidden_states = [state[:, 1:] for state in hidden_states]
        selected = torch.cat(hidden_states, dim=-1)
    return selected


def prepare_pooled_inputs(
    torch: Any,
    processor: Any,
    model: Any,
    device: Any,
    images: Sequence[Any],
    text_prompt: str,
    *,
    output_grid: int,
    input_grid: int = 24,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build gap-positioned inputs while pooling before the projector."""

    if len(images) != 1:
        raise EarlyCompressionError("Registered early compression requires exactly one image")
    inputs = processor(images=list(images), text=text_prompt, return_tensors="pt")
    moved: dict[str, Any] = {}
    dtype = next(model.parameters()).dtype
    for key, value in inputs.items():
        if hasattr(value, "to"):
            value = (
                value.to(device=device, dtype=dtype)
                if key == "pixel_values"
                else value.to(device=device)
            )
        moved[key] = value
    if "pixel_values" not in moved or "input_ids" not in moved:
        raise EarlyCompressionError("Processor did not produce pixel values and input IDs")

    _cuda_sync(torch)
    vision_started = time.perf_counter()
    with torch.inference_mode():
        selected = _selected_vision_hidden(torch, model, moved["pixel_values"])
    _cuda_sync(torch)
    vision_latency = time.perf_counter() - vision_started

    _cuda_sync(torch)
    pooling_started = time.perf_counter()
    pooled, pool_evidence = area_weighted_spatial_pool(
        torch, selected, input_grid=input_grid, output_grid=output_grid
    )
    _cuda_sync(torch)
    pooling_latency = time.perf_counter() - pooling_started

    _cuda_sync(torch)
    projector_started = time.perf_counter()
    with torch.inference_mode():
        projected = model.model.multi_modal_projector(pooled)
    _cuda_sync(torch)
    projector_latency = time.perf_counter() - projector_started

    input_ids = moved["input_ids"]
    attention_mask = moved.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    image_token_id = int(getattr(model.config, "image_token_index", 32000))
    image_positions = torch.nonzero(
        input_ids[0] == image_token_id, as_tuple=False
    ).flatten()
    expected_input_tokens = input_grid * input_grid
    if int(image_positions.numel()) != expected_input_tokens:
        raise EarlyCompressionError(
            f"Expected {expected_input_tokens} image placeholders, got {image_positions.numel()}"
        )
    expected_output_tokens = output_grid * output_grid
    if int(projected.shape[1]) != expected_output_tokens:
        raise EarlyCompressionError("Projected pooled-token count is inconsistent")
    anchor_tensor = torch.tensor(
        pool_evidence.anchor_indices, dtype=torch.long, device=input_ids.device
    )
    anchor_positions = image_positions[anchor_tensor]

    _cuda_sync(torch)
    assembly_started = time.perf_counter()
    text_embeds = model.get_input_embeddings()(input_ids).to(dtype=dtype).clone()
    text_embeds[0, anchor_positions, :] = projected[0].to(
        device=text_embeds.device, dtype=text_embeds.dtype
    )
    keep_mask = input_ids[0] != image_token_id
    keep_mask[anchor_positions] = True
    selected_positions = torch.nonzero(keep_mask, as_tuple=False).flatten()
    compacted = {
        "input_ids": input_ids[:, selected_positions],
        "inputs_embeds": text_embeds[:, selected_positions, :],
        "attention_mask": attention_mask[:, selected_positions],
        "position_ids": selected_positions.unsqueeze(0),
        "selected_positions": selected_positions,
        "full_prompt_length": int(input_ids.shape[1]),
    }
    _cuda_sync(torch)
    assembly_latency = time.perf_counter() - assembly_started

    compact_image_positions = torch.nonzero(
        compacted["input_ids"][0] == image_token_id, as_tuple=False
    ).flatten()
    if int(compact_image_positions.numel()) != expected_output_tokens:
        raise EarlyCompressionError("Compacted prompt has the wrong number of image tokens")
    original_text_positions = torch.nonzero(
        input_ids[0] != image_token_id, as_tuple=False
    ).flatten()
    compact_text_positions = compacted["position_ids"][0][
        compacted["input_ids"][0] != image_token_id
    ]
    if not torch.equal(original_text_positions, compact_text_positions):
        raise EarlyCompressionError("Pooling altered a text token or original text position")
    position_list = [
        int(value) for value in compacted["position_ids"][0].detach().cpu().tolist()
    ]
    anchor_position_list = [int(value) for value in anchor_positions.detach().cpu().tolist()]
    diagnostics = {
        "vision_encoder_latency_s": vision_latency,
        "pooling_latency_s": pooling_latency,
        "projector_latency_s": projector_latency,
        "input_assembly_latency_s": assembly_latency,
        "input_grid": input_grid,
        "output_grid": output_grid,
        "original_visual_tokens": expected_input_tokens,
        "pooled_visual_tokens": expected_output_tokens,
        "pool_input_mass": pool_evidence.input_mass,
        "pool_output_mass": pool_evidence.output_mass,
        "pool_output_masses": pool_evidence.output_masses,
        "pool_anchor_indices": pool_evidence.anchor_indices,
        "pool_anchor_checksum": pool_evidence.anchor_checksum,
        "pool_anchor_positions": anchor_position_list,
        "pool_anchor_position_checksum": _checksum(anchor_position_list),
        "pool_overlap_checksum": pool_evidence.overlap_checksum,
        "position_mode": "gap_preserving_original_prompt_positions",
        "position_ids": position_list,
        "position_id_checksum": _checksum(position_list),
        "original_prompt_length": int(input_ids.shape[1]),
        "compacted_prompt_length": int(compacted["input_ids"].shape[1]),
        "text_token_count": int(original_text_positions.numel()),
        "text_position_checksum": _checksum(
            [int(value) for value in original_text_positions.detach().cpu().tolist()]
        ),
    }
    return compacted, diagnostics


def cache_layer_lengths(cache: Any) -> list[int]:
    """Return physical sequence length for each layer in a Transformers cache."""

    if cache is None:
        return []
    if hasattr(cache, "layers"):
        result: list[int] = []
        for layer in cache.layers:
            keys = getattr(layer, "keys", None)
            if keys is None:
                keys = getattr(layer, "key_cache", None)
            if keys is None:
                raise EarlyCompressionError("Cache layer has no key tensor")
            result.append(int(keys.shape[-2]))
        return result
    if hasattr(cache, "key_cache"):
        return [int(keys.shape[-2]) for keys in cache.key_cache]
    if isinstance(cache, (list, tuple)):
        return [int(layer[0].shape[-2]) for layer in cache]
    raise EarlyCompressionError(f"Unsupported cache type: {type(cache)!r}")


@dataclass
class ContiguousCacheState:
    cache: Any
    attention_mask: Any
    full_prompt_length: int
    compacted_prompt_length: int
    prompt_cache_bytes: int
    prompt_layer_lengths: list[int]
    generated_tokens_cached: int = 0


class ContiguousPoolForwarder:
    """One-cache, no-boundary language-model execution for pooled prompts."""

    def __init__(self, torch: Any, model: Any) -> None:
        self.torch = torch
        self.model = model

    def prefill(self, pooled_inputs: dict[str, Any]) -> tuple[Any, ContiguousCacheState]:
        with self.torch.inference_mode():
            outputs = self.model(
                inputs_embeds=pooled_inputs["inputs_embeds"],
                attention_mask=pooled_inputs["attention_mask"],
                position_ids=pooled_inputs["position_ids"],
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        cache = outputs.past_key_values
        lengths = cache_layer_lengths(cache)
        prompt_length = int(pooled_inputs["input_ids"].shape[1])
        if not lengths or any(length != prompt_length for length in lengths):
            raise EarlyCompressionError(
                f"Contiguous prompt cache lengths {lengths} != {prompt_length}"
            )
        state = ContiguousCacheState(
            cache=cache,
            attention_mask=pooled_inputs["attention_mask"],
            full_prompt_length=int(pooled_inputs["full_prompt_length"]),
            compacted_prompt_length=prompt_length,
            prompt_cache_bytes=cache_tensor_bytes(cache),
            prompt_layer_lengths=lengths,
        )
        return outputs.logits, state

    def decode_one(self, state: ContiguousCacheState, token: Any) -> Any:
        state.attention_mask = self.torch.cat(
            [state.attention_mask, self.torch.ones_like(token)], dim=1
        )
        position_id = self.torch.full_like(
            token, state.full_prompt_length + state.generated_tokens_cached
        )
        with self.torch.inference_mode():
            outputs = self.model(
                input_ids=token,
                attention_mask=state.attention_mask,
                position_ids=position_id,
                past_key_values=state.cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        state.cache = outputs.past_key_values
        state.generated_tokens_cached += 1
        expected = state.compacted_prompt_length + state.generated_tokens_cached
        lengths = cache_layer_lengths(state.cache)
        if any(length != expected for length in lengths):
            raise EarlyCompressionError(
                f"Generated cache lengths {lengths} != expected {expected}"
            )
        return outputs.logits

    def diagnostics(self, state: ContiguousCacheState) -> dict[str, Any]:
        final_lengths = cache_layer_lengths(state.cache)
        return {
            "cache/implementation": "single_contiguous_transformers_cache",
            "cache/prompt_total_bytes": state.prompt_cache_bytes,
            "cache/final_total_bytes": cache_tensor_bytes(state.cache),
            "cache/prompt_per_layer_lengths": state.prompt_layer_lengths,
            "cache/final_per_layer_lengths": final_lengths,
            "cache/generated_tokens_cached": state.generated_tokens_cached,
            "cache/compacted_prompt_length": state.compacted_prompt_length,
            "cache/full_original_prompt_length": state.full_prompt_length,
        }

    def generate_natural(
        self,
        pooled_inputs: dict[str, Any],
        *,
        max_new_tokens: int,
        eos_ids: set[int],
    ) -> tuple[Any, dict[str, Any]]:
        self._sync()
        prefill_started = time.perf_counter()
        logits, state = self.prefill(pooled_inputs)
        next_token = self.torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        self._sync()
        prefill_latency = time.perf_counter() - prefill_started
        generated = [next_token]
        finished = bool(eos_ids and int(next_token.item()) in eos_ids)
        self._sync()
        decode_started = time.perf_counter()
        while len(generated) < max_new_tokens and not finished:
            logits = self.decode_one(state, next_token)
            next_token = self.torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)
            finished = bool(eos_ids and int(next_token.item()) in eos_ids)
        self._sync()
        decode_latency = time.perf_counter() - decode_started
        generated_ids = self.torch.cat(generated, dim=1)[0]
        diagnostics = {
            **self.diagnostics(state),
            "llm_prefill_latency_s": prefill_latency,
            "decode_latency_s": decode_latency,
            "decode_step_count": max(len(generated) - 1, 0),
            "generated_token_count": len(generated),
        }
        return generated_ids, diagnostics

    def generate_fixed(
        self,
        pooled_inputs: dict[str, Any],
        *,
        token_ids: Sequence[int],
    ) -> tuple[Any, dict[str, Any]]:
        self._sync()
        prefill_started = time.perf_counter()
        _, state = self.prefill(pooled_inputs)
        self._sync()
        prefill_latency = time.perf_counter() - prefill_started
        self._sync()
        decode_started = time.perf_counter()
        for token_id in token_ids:
            token = self.torch.tensor(
                [[int(token_id)]],
                dtype=pooled_inputs["input_ids"].dtype,
                device=pooled_inputs["input_ids"].device,
            )
            self.decode_one(state, token)
        self._sync()
        decode_latency = time.perf_counter() - decode_started
        generated_ids = self.torch.tensor(
            list(token_ids),
            dtype=pooled_inputs["input_ids"].dtype,
            device=pooled_inputs["input_ids"].device,
        )
        diagnostics = {
            **self.diagnostics(state),
            "llm_prefill_latency_s": prefill_latency,
            "decode_latency_s": decode_latency,
            "decode_step_count": len(token_ids),
            "generated_token_count": len(token_ids),
        }
        return generated_ids, diagnostics

    def _sync(self) -> None:
        _cuda_sync(self.torch)
