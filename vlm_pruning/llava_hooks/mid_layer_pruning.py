"""Mid-layer UG-Prune helpers for Hugging Face LLaVA models."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Sequence

from vlm_pruning.features import FusedMapResult, extract_maps_from_hidden_states, fuse_layer_maps
from vlm_pruning.efficiency import cache_tensor_bytes
from vlm_pruning.selectors import UGPruneConfig, UGPruneInputs, UGPruneSelector
from vlm_pruning.selectors.base import SelectionResult, SelectorError
from vlm_pruning.selectors.ug_prune import budget_split, stable_checksum


FROZEN_UG_GAP58_NAME = "UG-Prune-Gap-58"


@dataclass(frozen=True)
class MidLayerForwardResult:
    logits: Any
    selection: SelectionResult
    fused: FusedMapResult | None
    diagnostics: dict[str, Any]


@dataclass
class LayerAwareCacheState:
    early_cache: Any
    late_cache: Any
    selection: SelectionResult
    fused: FusedMapResult | None
    selected_positions: Any
    early_attention_mask: Any
    late_attention_mask: Any
    full_prompt_len: int
    pruned_prompt_len: int
    prune_after_layer: int = 12
    position_mode: str = "gap_preserving"
    generated_tokens_cached: int = 0

    def next_position_id(self) -> int:
        """Return the legacy shared position ID used by gap-preserving decoding."""
        return int(self.full_prompt_len + self.generated_tokens_cached)

    def next_early_position_id(self) -> int:
        return int(self.full_prompt_len + self.generated_tokens_cached)

    def next_late_position_id(self) -> int:
        base = self.full_prompt_len if self.position_mode == "gap_preserving" else self.pruned_prompt_len
        return int(base + self.generated_tokens_cached)

    def expected_early_cache_len(self) -> int:
        return int(self.full_prompt_len + self.generated_tokens_cached)

    def expected_late_cache_len(self) -> int:
        return int(self.pruned_prompt_len + self.generated_tokens_cached)


@dataclass(frozen=True)
class LayerAwarePrefillResult:
    logits: Any
    state: LayerAwareCacheState
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class LayerAwareGenerationResult:
    generated_ids: Any
    selection: SelectionResult
    fused: FusedMapResult | None
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SelectionPipelineResult:
    selection: SelectionResult
    fused: FusedMapResult | None
    selected_indices_tensor: Any | None
    map_latency: float
    selection_latency: float
    diagnostics: dict[str, Any]
    score_map: Any | None = None


class CacheLengthProxy:
    """Expose a non-zero default cache length for a late-layer cache bank."""

    def __init__(self, cache: Any, *, default_layer_idx: int) -> None:
        self.cache = cache
        self.default_layer_idx = int(default_layer_idx)

    def _map_layer_idx(self, layer_idx: int | None = 0) -> int:
        if layer_idx is None or int(layer_idx) == 0:
            return self.default_layer_idx
        return int(layer_idx)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.cache.get_seq_length(self._map_layer_idx(layer_idx))

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        return self.cache.get_mask_sizes(query_length, self._map_layer_idx(layer_idx))

    def update(self, key_states: Any, value_states: Any, layer_idx: int, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        return self.cache.update(key_states, value_states, int(layer_idx), *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.cache, name)


def frozen_ug_gap58_config() -> UGPruneConfig:
    """Return the frozen Phase 2 ablation winner."""
    return UGPruneConfig(
        map_family="unrolled",
        map_source="query_max",
        probe_layers=(4, 8, 12),
        fusion_mode="soft_average",
        role_mode="target_context_anchor",
        target_ratio=0.50,
        context_ratio=0.30,
        anchor_ratio=0.20,
        context_radius=1,
        anchor_pattern="fps",
        position_mode="gap_preserving",
    ).validate()


def ug_gap58_config_for_prune_layer(
    prune_after_layer: int,
    *,
    map_source: str | None = None,
    fusion_mode: str | None = None,
    probe_layers: Sequence[int] | None = None,
    position_mode: str | None = None,
    target_ratio: float | None = None,
    context_ratio: float | None = None,
    anchor_ratio: float | None = None,
    selector_strategy: str | None = None,
) -> UGPruneConfig:
    """Return the UG-Gap-58 config using only probe layers available before pruning."""
    layer = int(prune_after_layer)
    base = frozen_ug_gap58_config()
    requested_fusion = fusion_mode or base.fusion_mode
    if probe_layers is not None:
        probes = tuple(dict.fromkeys(int(item) for item in probe_layers))
        if not probes:
            raise SelectorError("probe_layers must contain at least one layer.")
        if any(item <= 0 or item > layer for item in probes):
            raise SelectorError(
                f"Probe layers must be in [1, {layer}], got {probes}."
            )
    elif requested_fusion == "single_layer":
        probes = (layer,)
    elif layer >= 12:
        probes = (4, 8, 12)
    elif layer in {10, 11}:
        probes = (4, 8, layer)
    else:
        probes = tuple(item for item in (4, 8, 12) if item <= layer)
    if not probes:
        raise SelectorError(f"prune_after_layer={layer} has no available UG probe layers.")
    if len(probes) == 1:
        requested_fusion = "single_layer"
    return UGPruneConfig(
        selector_strategy=selector_strategy or base.selector_strategy,
        map_family=base.map_family,
        map_source=map_source or base.map_source,
        probe_layers=probes,
        fusion_mode=requested_fusion,
        dynamic_layer_metric=base.dynamic_layer_metric,
        role_mode=base.role_mode,
        target_ratio=base.target_ratio if target_ratio is None else float(target_ratio),
        context_ratio=base.context_ratio if context_ratio is None else float(context_ratio),
        anchor_ratio=base.anchor_ratio if anchor_ratio is None else float(anchor_ratio),
        context_radius=base.context_radius,
        anchor_bins=base.anchor_bins,
        anchor_pattern=base.anchor_pattern,
        nms_radius=base.nms_radius,
        position_mode=position_mode or base.position_mode,
    ).validate()


def validate_retained_indices(retained: Sequence[int], *, original_tokens: int, budget: int) -> list[int]:
    indices = [int(index) for index in retained]
    if len(indices) != int(budget):
        raise SelectorError(f"Expected {budget} retained visual tokens, got {len(indices)}.")
    if len(set(indices)) != len(indices):
        raise SelectorError("Retained visual-token indices contain duplicates.")
    if not indices:
        raise SelectorError("At least one visual token must be retained.")
    if min(indices) < 0 or max(indices) >= int(original_tokens):
        raise SelectorError(f"Retained visual-token index out of range for {original_tokens} tokens.")
    return indices


def selected_positions_for_retained(
    torch: Any,
    input_ids: Any,
    image_token_id: int,
    retained: Sequence[int],
    *,
    sort_retained: bool = True,
) -> Any:
    """Return full-sequence positions retained after visual-token pruning."""
    ids = input_ids[0] if getattr(input_ids, "ndim", 1) == 2 else input_ids
    image_positions = torch.nonzero(ids == int(image_token_id), as_tuple=False).flatten()
    retained_list = [int(index) for index in retained]
    if sort_retained:
        retained_list = sorted(retained_list)
    validate_retained_indices(retained_list, original_tokens=int(image_positions.numel()), budget=len(retained_list))
    keep = torch.tensor(retained_list, dtype=torch.long, device=ids.device)
    keep_mask = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
    keep_mask[image_positions] = False
    keep_mask[image_positions[keep]] = True
    return torch.nonzero(keep_mask, as_tuple=False).flatten()


def selected_positions_for_retained_tensor(
    torch: Any,
    input_ids: Any,
    image_token_id: int,
    retained: Any,
    *,
    sort_retained: bool = True,
) -> Any:
    """Torch-only variant for the benchmark path."""
    ids = input_ids[0] if getattr(input_ids, "ndim", 1) == 2 else input_ids
    image_positions = torch.nonzero(ids == int(image_token_id), as_tuple=False).flatten()
    keep = retained.to(device=ids.device, dtype=torch.long).flatten()
    if sort_retained:
        keep = torch.sort(keep).values
    keep_mask = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
    keep_mask[image_positions] = False
    keep_mask[image_positions[keep]] = True
    return torch.nonzero(keep_mask, as_tuple=False).flatten()


class MidLayerUGPruneForwarder:
    """Run a LLaVA language-model prefix, prune visual tokens, and finish the forward pass."""

    def __init__(
        self,
        torch: Any,
        model: Any,
        *,
        image_token_id: int,
        grid: int = 24,
        prune_after_layer: int = 12,
        target_visual_tokens: int = 58,
        selector_config: UGPruneConfig | None = None,
        diagnostics_mode: str = "full",
        synchronise_subtimings: bool = False,
    ) -> None:
        self.torch = torch
        self.model = model
        self.image_token_id = int(image_token_id)
        self.grid = int(grid)
        self.prune_after_layer = int(prune_after_layer)
        self.target_visual_tokens = int(target_visual_tokens)
        self.selector_config = (selector_config or frozen_ug_gap58_config()).validate()
        self.selector = UGPruneSelector(self.selector_config)
        self.diagnostics_mode = str(diagnostics_mode)
        self.synchronise_subtimings = bool(synchronise_subtimings)
        if self.diagnostics_mode not in {"full", "benchmark"}:
            raise SelectorError(f"Unsupported diagnostics_mode: {self.diagnostics_mode}")
        self._grid_tensor_cache: dict[Any, dict[str, Any]] = {}
        if self.prune_after_layer <= 0:
            raise SelectorError("Phase 2B pilot expects prune_after_layer > 0.")
        if max(self.selector_config.probe_layers) > self.prune_after_layer:
            raise SelectorError(
                "Probe layers must be available before pruning: "
                f"probe_layers={self.selector_config.probe_layers}, prune_after_layer={self.prune_after_layer}."
            )

    @property
    def language_model(self) -> Any:
        return self.model.model.language_model

    @property
    def decoder_layers(self) -> Any:
        return self.language_model.layers

    @property
    def num_layers(self) -> int:
        return int(len(self.decoder_layers))

    def position_ids_for(self, length: int, device: Any) -> Any:
        return self.torch.arange(int(length), device=device, dtype=self.torch.long).unsqueeze(0)

    def _grid_tensors(self, device: Any, image_count: int = 1) -> dict[str, Any]:
        image_count = int(image_count)
        if image_count <= 0:
            raise SelectorError(f"image_count must be positive, got {image_count}.")
        key = f"{device}:images={image_count}"
        cached = self._grid_tensor_cache.get(key)
        if cached is not None:
            return cached
        torch = self.torch
        grid = self.grid
        per_image = grid * grid
        count = per_image * image_count
        indices = torch.arange(count, device=device, dtype=torch.long)
        image_ids = torch.div(indices, per_image, rounding_mode="floor")
        local_indices = indices.remainder(per_image)
        rows = torch.div(local_indices, grid, rounding_mode="floor")
        cols = local_indices.remainder(grid)
        same_image = image_ids[:, None] == image_ids[None, :]
        row_diff = (rows[:, None] - rows[None, :]).abs()
        col_diff = (cols[:, None] - cols[None, :]).abs()
        near_mask = same_image & (row_diff <= int(self.selector_config.nms_radius)) & (col_diff <= int(self.selector_config.nms_radius))
        neighbor_mask = same_image & ((row_diff <= int(self.selector_config.context_radius)) & (col_diff <= int(self.selector_config.context_radius))) & (
            indices[:, None] != indices[None, :]
        )
        row_bins, col_bins = self.selector_config.anchor_bins
        bin_rows = torch.clamp(torch.div(rows * row_bins, grid, rounding_mode="floor"), max=row_bins - 1)
        bin_cols = torch.clamp(torch.div(cols * col_bins, grid, rounding_mode="floor"), max=col_bins - 1)
        bins_per_image = row_bins * col_bins
        bin_ids = (image_ids * bins_per_image + bin_rows * col_bins + bin_cols).to(dtype=torch.long)
        global_cols = image_ids * grid + cols
        centers = torch.stack([rows.float() + 0.5, global_cols.float() + 0.5], dim=1)
        distance_sq = ((centers[:, None, :] - centers[None, :, :]) ** 2).sum(dim=-1)
        center_point = torch.tensor(
            [float(grid) / 2.0, float(grid * image_count) / 2.0],
            device=device,
            dtype=centers.dtype,
        )
        center_distance_sq = ((centers - center_point) ** 2).sum(dim=-1)
        cached = {
            "indices": indices,
            "rows": rows,
            "global_cols": global_cols,
            "near_mask": near_mask,
            "neighbor_mask": neighbor_mask,
            "bin_ids": bin_ids,
            "distance_sq": distance_sq,
            "center_distance_sq": center_distance_sq,
            "num_bins": int(bins_per_image * image_count),
            "image_count": image_count,
        }
        self._grid_tensor_cache[key] = cached
        return cached

    def _cuda_sync(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def _subtiming_sync(self) -> None:
        if self.synchronise_subtimings:
            self._cuda_sync()

    def _profile_scope(self, component: str, **metadata: Any) -> Any:
        """Return an inert scope unless an opt-in profiling subclass overrides it."""
        del component, metadata
        return nullcontext()

    def cache_seq_len(self, cache: Any, layer_idx: int) -> int:
        value = cache.get_seq_length(int(layer_idx))
        if hasattr(value, "item"):
            return int(value.item())
        return int(value)

    def _causal_mask(self, hidden_states: Any, attention_mask: Any, position_ids: Any, past_key_values: Any | None = None) -> Any:
        from transformers.models.llama.modeling_llama import create_causal_mask

        cache_position = self._cache_position(hidden_states, past_key_values)
        try:
            return create_causal_mask(
                config=self.language_model.config,
                input_embeds=hidden_states,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        except TypeError as exc:
            if "input_embeds" not in str(exc) and "cache_position" not in str(exc):
                raise
            return create_causal_mask(
                config=self.language_model.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

    def _cache_position(self, hidden_states: Any, past_key_values: Any | None) -> Any:
        past_length = 0
        if past_key_values is not None:
            try:
                past_length = self.cache_seq_len(past_key_values, 0)
            except Exception:
                past_length = self.cache_seq_len(past_key_values, self.prune_after_layer)
        return self.torch.arange(
            past_length,
            past_length + int(hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=self.torch.long,
        )

    def run_decoder_layers(
        self,
        hidden_states: Any,
        *,
        attention_mask: Any,
        position_ids: Any,
        start_layer: int,
        end_layer: int,
        collect_hidden_states: bool = False,
        past_key_values: Any | None = None,
        use_cache: bool = False,
    ) -> tuple[Any, list[Any] | None]:
        if start_layer < 0 or end_layer > self.num_layers or start_layer > end_layer:
            raise SelectorError(f"Invalid decoder layer range [{start_layer}, {end_layer}) for {self.num_layers} layers.")
        causal_mask = self._causal_mask(hidden_states, attention_mask, position_ids, past_key_values)
        cache_position = self._cache_position(hidden_states, past_key_values)
        position_embeddings = self.language_model.rotary_emb(hidden_states, position_ids=position_ids)
        captured: list[Any] | None = [hidden_states] if collect_hidden_states else None
        for layer_index in range(start_layer, end_layer):
            output = self.decoder_layers[layer_index](
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = output[0] if isinstance(output, (tuple, list)) else output
            if captured is not None:
                captured.append(hidden_states)
        return hidden_states, captured

    def run_decoder_layers_with_final_attention(
        self,
        hidden_states: Any,
        *,
        attention_mask: Any,
        position_ids: Any,
        start_layer: int,
        end_layer: int,
        past_key_values: Any,
    ) -> tuple[Any, Any]:
        """Run a cached prefix and retain attention from its final layer.

        FastV ranks image tokens with the final prompt token's attention after
        decoder layer ``K - 1`` and removes tokens before layer ``K``.  Current
        Transformers decoder layers discard the attention tensor, so the final
        layer is expanded here without changing its computation.
        """
        if start_layer < 0 or end_layer > self.num_layers or start_layer >= end_layer:
            raise SelectorError(f"Invalid attention-capture range [{start_layer}, {end_layer}).")
        causal_mask = self._causal_mask(hidden_states, attention_mask, position_ids, past_key_values)
        cache_position = self._cache_position(hidden_states, past_key_values)
        position_embeddings = self.language_model.rotary_emb(hidden_states, position_ids=position_ids)
        for layer_index in range(start_layer, end_layer - 1):
            hidden_states = self.decoder_layers[layer_index](
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=True,
                cache_position=cache_position,
            )

        layer = self.decoder_layers[end_layer - 1]
        residual = hidden_states
        normalised = layer.input_layernorm(hidden_states)
        attention_output, attention_weights = layer.self_attn(
            hidden_states=normalised,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )
        if attention_weights is None:
            raise SelectorError(
                "FastV requires attention weights; load the language model with eager attention."
            )
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + layer.mlp(hidden_states)
        return hidden_states, attention_weights

    def fastv_selection_from_attention(
        self,
        attention_weights: Any,
        *,
        input_ids: Any,
        original_tokens: int,
    ) -> SelectionResult:
        """Apply the official FastV last-token/mean-head ranking rule."""
        image_positions = self.torch.nonzero(
            input_ids[0] == int(self.image_token_id), as_tuple=False
        ).flatten()
        if int(image_positions.numel()) != int(original_tokens):
            raise SelectorError(
                f"FastV expected {original_tokens} image tokens, found {int(image_positions.numel())}."
            )
        scores = attention_weights.mean(dim=1)[0, -1].index_select(0, image_positions)
        selected = self.torch.topk(
            scores, self.target_visual_tokens, largest=True, sorted=False
        ).indices
        selected = self.torch.sort(selected).values
        indices = [int(item) for item in selected.detach().cpu().tolist()]
        validate_retained_indices(
            indices,
            original_tokens=original_tokens,
            budget=self.target_visual_tokens,
        )
        return SelectionResult(
            selected_indices=indices,
            method="fastv",
            budget=self.target_visual_tokens,
            original_tokens=original_tokens,
            grid=self.grid,
            role_labels={"attention_ranked": indices},
            metadata={
                "selection_backend": "official_fastv_attention_rule",
                "fastv_k": self.prune_after_layer,
                "fastv_attention_layer_index": self.prune_after_layer - 1,
                "fastv_prune_before_layer_index": self.prune_after_layer,
                "fastv_r": 1.0 - (self.target_visual_tokens / original_tokens),
                "fastv_static_decode_selection": True,
            },
        ).validate()

    def assert_cache_lengths(self, state: LayerAwareCacheState) -> None:
        early_len = self.cache_seq_len(state.early_cache, 0)
        late_len = self.cache_seq_len(state.late_cache, state.prune_after_layer)
        if early_len != state.expected_early_cache_len():
            raise SelectorError(
                f"Early cache length {early_len} != expected {state.expected_early_cache_len()}."
            )
        if late_len != state.expected_late_cache_len():
            raise SelectorError(f"Late cache length {late_len} != expected {state.expected_late_cache_len()}.")

    def _supports_fast_frozen_selection(self, original_tokens: int, selector_config: UGPruneConfig | None = None) -> bool:
        config = selector_config or self.selector_config
        return (
            self.diagnostics_mode == "benchmark"
            and int(original_tokens) >= self.grid * self.grid
            and int(original_tokens) % (self.grid * self.grid) == 0
            and self.grid == 24
            and config.map_family == "unrolled"
            and config.map_source in {"query_max", "query_pooled"}
            and config.fusion_mode in {"soft_average", "single_layer"}
            and config.selector_strategy in {"query_topk", "target_nms_only", "structured_tca"}
            and config.context_radius == 1
            and config.anchor_pattern == "fps"
            and config.position_mode in {"gap_preserving", "compact", "raster_compact"}
            and config.anchor_bins == (3, 3)
            and config.nms_radius == 1
        )

    def _normalise_states(self, states: Any) -> Any:
        torch = self.torch
        states = torch.nn.functional.layer_norm(states, (states.shape[-1],))
        return torch.nn.functional.normalize(states.float(), p=2, dim=-1)

    def _zscore(self, values: Any, eps: float = 1e-6) -> Any:
        torch = self.torch
        values = values.float()
        std = values.std(unbiased=False)
        normalised = (values - values.mean()) / torch.clamp(std, min=eps)
        return torch.where(std < eps, torch.zeros_like(normalised), normalised)

    def _score_map_stats_torch(
        self,
        scores: Any,
        *,
        target_visual_tokens: int | None = None,
    ) -> dict[str, float | int]:
        torch = self.torch
        values = scores.float().reshape(-1)
        probs = torch.softmax(values, dim=0)
        entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum()
        max_entropy = torch.log(torch.tensor(float(values.numel()), device=values.device, dtype=values.dtype))
        stats: dict[str, float | int] = {
            "map_mean": float(values.mean().item()),
            "map_std": float(values.std(unbiased=False).item()),
            "map_min": float(values.min().item()),
            "map_max": float(values.max().item()),
            "entropy": float(entropy.item()),
            "entropy_norm": float((entropy / max_entropy).item()) if int(values.numel()) > 1 else 0.0,
            "peakiness": float((values.max() - values.mean()).item()),
        }
        per_image = self.grid * self.grid
        if int(values.numel()) % per_image != 0:
            raise SelectorError(f"Score map has {int(values.numel())} tokens, not whole image grids.")
        grid_tensors = self._grid_tensors(values.device, int(values.numel()) // per_image)
        target_budget = int(
            self.target_visual_tokens if target_visual_tokens is None else target_visual_tokens
        )
        top_counts = sorted({16, 29, 48, 58, 72, 96, target_budget})
        for k in top_counts:
            count = min(int(k), int(values.numel()))
            top = torch.topk(probs, count, largest=True, sorted=False).indices
            stats[f"top_{k}_mass"] = float(probs.index_select(0, top).sum().item())
            bins = grid_tensors["bin_ids"].index_select(0, top)
            bin_counts = torch.bincount(bins, minlength=grid_tensors["num_bins"]).float()
            occupied = int((bin_counts > 0).sum().item())
            bin_probs = bin_counts / torch.clamp(bin_counts.sum(), min=1.0)
            bin_entropy = -(bin_probs * torch.log(torch.clamp(bin_probs, min=1e-12))).sum()
            stats[f"top_{k}_occupied_bins_3x3"] = occupied
            stats[f"top_{k}_spatial_entropy_3x3"] = float(
                (bin_entropy / torch.log(torch.tensor(float(grid_tensors["num_bins"]), device=values.device))).item()
            )
        return stats

    def _selected_spread_stats_torch(self, selected: Any, *, original_tokens: int) -> dict[str, float | int]:
        torch = self.torch
        indices = selected.to(dtype=torch.long).reshape(-1)
        if int(indices.numel()) == 0:
            return {
                "selected_occupied_bins_3x3": 0,
                "selected_row_span": 0,
                "selected_col_span": 0,
                "selected_mean_pairwise_spread": 0.0,
            }
        per_image = self.grid * self.grid
        if int(original_tokens) % per_image != 0:
            raise SelectorError(f"original_tokens={original_tokens} is not whole image grids.")
        grid_tensors = self._grid_tensors(indices.device, int(original_tokens) // per_image)
        bins = grid_tensors["bin_ids"].index_select(0, indices)
        rows = grid_tensors["rows"].index_select(0, indices).float()
        cols = grid_tensors["global_cols"].index_select(0, indices).float()
        coords = torch.stack([rows, cols], dim=1)
        if int(indices.numel()) > 1:
            width = self.grid * int(original_tokens // per_image)
            diagonal = max(((self.grid - 1) ** 2 + (width - 1) ** 2) ** 0.5, 1.0)
            spread = torch.pdist(coords, p=2).mean() / diagonal
        else:
            spread = torch.tensor(0.0, device=indices.device)
        return {
            "selected_occupied_bins_3x3": int(torch.unique(bins).numel()),
            "selected_row_span": int((rows.max() - rows.min() + 1).item()),
            "selected_col_span": int((cols.max() - cols.min() + 1).item()),
            "selected_mean_pairwise_spread": float(spread.item()),
        }

    def _extract_fused_map_torch(
        self,
        *,
        hidden_states: Sequence[Any],
        input_ids: Any,
        query_token_positions: Sequence[int],
        selector_config: UGPruneConfig | None = None,
        target_visual_tokens: int | None = None,
    ) -> tuple[Any, tuple[str, ...], dict[str, float]]:
        torch = self.torch
        config = selector_config or self.selector_config
        ids = input_ids[0] if getattr(input_ids, "ndim", 1) == 2 else input_ids
        image_positions = torch.nonzero(ids == int(self.image_token_id), as_tuple=False).flatten()
        per_image = self.grid * self.grid
        expected = int(image_positions.numel())
        if expected < per_image or expected % per_image != 0:
            raise SelectorError(
                f"Expected one or more complete {self.grid}x{self.grid} image grids, found {expected} tokens."
            )
        if not query_token_positions:
            raise SelectorError("query_token_positions is empty.")
        query_positions = torch.as_tensor(query_token_positions, device=ids.device, dtype=torch.long)
        maps: list[Any] = []
        keys: list[str] = []
        for layer in config.probe_layers:
            if layer <= 0 or layer >= len(hidden_states):
                raise SelectorError(f"Layer {layer} is outside hidden-state range [1, {len(hidden_states) - 1}].")
            states = hidden_states[layer]
            states = states[0] if getattr(states, "ndim", 0) == 3 else states
            image_states = states.index_select(0, image_positions)
            query_states = states.index_select(0, query_positions)
            image_norm = self._normalise_states(image_states)
            query_norm = self._normalise_states(query_states)
            if config.map_source == "query_max":
                scores = (image_norm @ query_norm.transpose(0, 1)).max(dim=1).values
            else:
                pooled_query = torch.nn.functional.normalize(query_norm.mean(dim=0, keepdim=True), p=2, dim=-1)
                scores = (image_norm @ pooled_query.transpose(0, 1)).squeeze(1)
            maps.append(self._zscore(scores).reshape(-1))
            keys.append(f"layer{layer}:{config.map_source}")
        if config.fusion_mode == "single_layer":
            if len(maps) != 1:
                raise SelectorError("single_layer fusion requires exactly one requested layer.")
            fused = self._zscore(maps[0].reshape(-1))
        elif config.fusion_mode == "soft_average":
            fused = self._zscore(torch.stack(maps, dim=0).mean(dim=0).reshape(-1))
        else:
            raise SelectorError(f"Fast benchmark path does not support fusion_mode={config.fusion_mode}.")
        flat = fused.reshape(-1)
        return flat, tuple(keys), self._score_map_stats_torch(
            flat,
            target_visual_tokens=target_visual_tokens,
        )

    def _select_top_ranked(self, ranked: Any, mask: Any, count: int) -> Any:
        if count <= 0:
            return ranked.new_empty((0,), dtype=self.torch.long)
        selected = ranked[mask.index_select(0, ranked)]
        return selected[: int(count)]

    def _select_target_torch(self, scores: Any, ranked: Any, count: int, grid_tensors: dict[str, Any]) -> tuple[Any, int]:
        torch = self.torch
        selected_mask = torch.zeros_like(scores, dtype=torch.bool)
        suppressed = torch.zeros_like(scores, dtype=torch.bool)
        selected: list[Any] = []
        for _ in range(int(count)):
            available_rank_positions = torch.nonzero(~suppressed.index_select(0, ranked), as_tuple=False).flatten()
            if int(available_rank_positions.numel()) == 0:
                break
            index = ranked[available_rank_positions[0]]
            selected.append(index)
            selected_mask[index] = True
            suppressed |= grid_tensors["near_mask"][index]
        nms_count = len(selected)
        if len(selected) < int(count):
            fill = self._select_top_ranked(ranked, ~selected_mask, int(count) - len(selected))
            if int(fill.numel()):
                selected.extend(fill.unbind(0))
        if not selected:
            return ranked.new_empty((0,), dtype=self.torch.long), 0
        return torch.stack(selected).to(dtype=torch.long), len(selected) - nms_count

    def _select_context_torch(self, ranked: Any, target: Any, count: int, grid_tensors: dict[str, Any]) -> Any:
        torch = self.torch
        if count <= 0:
            return ranked.new_empty((0,), dtype=torch.long)
        excluded = torch.zeros(ranked.shape[0], dtype=torch.bool, device=ranked.device)
        excluded[target] = True
        candidate_mask = grid_tensors["neighbor_mask"].index_select(0, target).any(dim=0) & ~excluded
        selected = self._select_top_ranked(ranked, candidate_mask, int(count))
        if int(selected.numel()) < int(count):
            excluded[selected] = True
            fill = self._select_top_ranked(ranked, ~excluded, int(count) - int(selected.numel()))
            selected = torch.cat([selected, fill], dim=0)
        return selected.to(dtype=torch.long)

    def _select_fps_anchors_torch(
        self,
        scores: Any,
        count: int,
        *,
        exclude: Any,
        grid_tensors: dict[str, Any],
    ) -> Any:
        torch = self.torch
        if count <= 0:
            return scores.new_empty((0,), dtype=torch.long)
        selected_mask = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
        selected_mask[exclude] = True
        anchors: list[Any] = []
        indices = grid_tensors["indices"]
        high_index = torch.full_like(indices, indices.numel())
        for _ in range(int(count)):
            candidates = ~selected_mask
            if int(candidates.sum().item()) == 0:
                break
            selected_indices = torch.nonzero(selected_mask, as_tuple=False).flatten()
            bin_counts = torch.bincount(
                grid_tensors["bin_ids"].index_select(0, selected_indices),
                minlength=grid_tensors["num_bins"],
            )
            occupancy = bin_counts.index_select(0, grid_tensors["bin_ids"])
            min_occupancy = occupancy[candidates].min()
            mask = candidates & (occupancy == min_occupancy)
            if int(selected_indices.numel()) == 0:
                min_distance = grid_tensors["center_distance_sq"]
            else:
                min_distance = grid_tensors["distance_sq"].index_select(1, selected_indices).min(dim=1).values
            max_distance = min_distance[mask].max()
            mask = mask & (min_distance == max_distance)
            max_score = scores[mask].max()
            mask = mask & (scores == max_score)
            best = torch.where(mask, indices, high_index).min()
            anchors.append(best)
            selected_mask[best] = True
        if not anchors:
            return scores.new_empty((0,), dtype=torch.long)
        return torch.stack(anchors).to(dtype=torch.long)

    def _selection_from_fused_torch(
        self,
        *,
        scores: Any,
        sample_id: str,
        original_tokens: int,
        seed: int,
        map_key: str,
        map_stats: dict[str, Any],
        query_text: str,
        query_token_status: str,
        query_token_positions: Sequence[int],
        selector_config: UGPruneConfig | None = None,
        target_visual_tokens: int | None = None,
    ) -> tuple[SelectionResult, Any]:
        del seed
        torch = self.torch
        config = selector_config or self.selector_config
        target_budget = int(target_visual_tokens or self.target_visual_tokens)
        target_count, context_count, anchor_count = budget_split(target_budget, config)
        ranked = torch.argsort(scores, descending=True, stable=True)
        per_image = self.grid * self.grid
        if int(original_tokens) % per_image != 0:
            raise SelectorError(f"original_tokens={original_tokens} is not whole image grids.")
        grid_tensors = self._grid_tensors(scores.device, int(original_tokens) // per_image)
        if config.selector_strategy == "query_topk":
            query_topk = ranked[:target_budget]
            target = ranked.new_empty((0,), dtype=torch.long)
            context = ranked.new_empty((0,), dtype=torch.long)
            anchor = ranked.new_empty((0,), dtype=torch.long)
            nms_fallback_count = 0
        else:
            query_topk = ranked.new_empty((0,), dtype=torch.long)
            target, nms_fallback_count = self._select_target_torch(scores, ranked, target_count, grid_tensors)
            context = self._select_context_torch(ranked, target, context_count, grid_tensors)
            anchor = self._select_fps_anchors_torch(
                scores,
                anchor_count,
                exclude=torch.cat([target, context], dim=0),
                grid_tensors=grid_tensors,
            )
        selected_mask = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
        selected_mask[target] = True
        selected_mask[context] = True
        selected_mask[anchor] = True
        selected_mask[query_topk] = True
        selected_count = int(selected_mask.sum().item())
        fill = ranked.new_empty((0,), dtype=torch.long)
        if selected_count < target_budget:
            fill = self._select_top_ranked(ranked, ~selected_mask, target_budget - selected_count)
            selected_mask[fill] = True
        if int(selected_mask.sum().item()) > target_budget:
            keep = self._select_top_ranked(ranked, selected_mask, target_budget)
            selected_mask = torch.zeros_like(selected_mask)
            selected_mask[keep] = True
            target = target[selected_mask.index_select(0, target)]
            context = context[selected_mask.index_select(0, context)]
            anchor = anchor[selected_mask.index_select(0, anchor)]
            fill = fill[selected_mask.index_select(0, fill)]
        selected_tensor = torch.nonzero(selected_mask, as_tuple=False).flatten().sort().values
        map_stats = dict(map_stats)
        map_stats.update(
            self._selected_spread_stats_torch(selected_tensor, original_tokens=original_tokens)
        )
        selected_indices = [int(index) for index in selected_tensor.detach().cpu().tolist()]
        roles = {
            "target": [int(index) for index in target.sort().values.detach().cpu().tolist()],
            "context": [int(index) for index in context.sort().values.detach().cpu().tolist()],
            "anchor": [int(index) for index in anchor.sort().values.detach().cpu().tolist()],
            "query_topk": [int(index) for index in query_topk.sort().values.detach().cpu().tolist()],
            "fill": [int(index) for index in fill.sort().values.detach().cpu().tolist()],
        }
        result = SelectionResult(
            selected_indices=selected_indices,
            method=f"{UGPruneSelector.name}_torch_fast",
            budget=target_budget,
            original_tokens=original_tokens,
            grid=self.grid,
            role_labels={role: values for role, values in roles.items() if values},
            metadata={
                "selector_family": "ug_prune_training_free_torch_fast",
                "selector_strategy": config.selector_strategy,
                "sample_id": sample_id,
                "config": config.as_dict(),
                "map_family": config.map_family,
                "budget_split": {
                    "query_topk": target_budget if config.selector_strategy == "query_topk" else 0,
                    "target": target_count,
                    "context": context_count,
                    "anchor": anchor_count,
                    "fill": len(roles["fill"]),
                },
                "realised_role_counts": {role: len(values) for role, values in roles.items()},
                "nms_fallback_count": nms_fallback_count,
                "nms_radius": config.nms_radius,
                "map_key": map_key,
                "map_source": config.map_source,
                "probe_layers": list(config.probe_layers),
                "fusion_mode": config.fusion_mode,
                "dynamic_layer_metric": config.dynamic_layer_metric,
                "role_mode": config.role_mode,
                "anchor_pattern": config.anchor_pattern,
                "position_mode": config.position_mode,
                "map_stats": map_stats,
                "query_text": query_text,
                "query_token_status": query_token_status,
                "query_token_positions": [int(position) for position in query_token_positions],
                "target_checksum": stable_checksum(roles["target"]),
                "context_checksum": stable_checksum(roles["context"]),
                "anchor_checksum": stable_checksum(roles["anchor"]),
            },
        ).validate()
        validate_retained_indices(result.selected_indices, original_tokens=original_tokens, budget=target_budget)
        return result, selected_tensor

    def select_from_hidden_states(
        self,
        *,
        hidden_states: Sequence[Any],
        input_ids: Any,
        sample_id: str,
        original_tokens: int,
        seed: int,
        query_text: str,
        query_token_status: str,
        query_token_positions: Sequence[int],
        selector_config: UGPruneConfig | None = None,
        target_visual_tokens: int | None = None,
    ) -> SelectionPipelineResult:
        config = selector_config or self.selector_config
        selector = self.selector if selector_config is None else UGPruneSelector(config)
        target_budget = int(target_visual_tokens or self.target_visual_tokens)
        self._subtiming_sync()
        map_start = time.perf_counter()
        if self._supports_fast_frozen_selection(original_tokens, config):
            with self._profile_scope("score_construction", method="k12_original"):
                fused_scores, layer_keys, map_stats = self._extract_fused_map_torch(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    query_token_positions=query_token_positions,
                    selector_config=config,
                    target_visual_tokens=target_budget,
                )
            self._subtiming_sync()
            map_latency = time.perf_counter() - map_start
            self._subtiming_sync()
            selection_start = time.perf_counter()
            with self._profile_scope("ranking_and_index_materialization", method="k12_original"):
                selection, selected_tensor = self._selection_from_fused_torch(
                    scores=fused_scores,
                    sample_id=sample_id,
                    original_tokens=original_tokens,
                    seed=seed,
                    map_key="+".join(layer_keys),
                    map_stats=map_stats,
                    query_text=query_text,
                    query_token_status=query_token_status,
                    query_token_positions=query_token_positions,
                    selector_config=config,
                    target_visual_tokens=target_budget,
                )
            self._subtiming_sync()
            selection_latency = time.perf_counter() - selection_start
            stat_diagnostics = {f"ug/map_{key}": value for key, value in selection.metadata.get("map_stats", {}).items()}
            return SelectionPipelineResult(
                selection=selection,
                fused=None,
                selected_indices_tensor=selected_tensor,
                map_latency=map_latency,
                selection_latency=selection_latency,
                diagnostics={
                    "ug/selection_backend": "torch_fast",
                    "ug/diagnostics_mode": self.diagnostics_mode,
                    "ug_fused_map_layer_keys": list(layer_keys),
                    **stat_diagnostics,
                },
                score_map=fused_scores.detach(),
            )

        layer_maps = extract_maps_from_hidden_states(
            hidden_states,
            input_ids,
            self.image_token_id,
            query_token_positions,
            config.probe_layers,
            grid=self.grid,
        )
        fused = fuse_layer_maps(
            layer_maps,
            source=config.map_source,
            layers=config.probe_layers,
            grid=self.grid,
            fusion_mode=config.fusion_mode,
            dynamic_metric=config.dynamic_layer_metric,
        )
        self._subtiming_sync()
        map_latency = time.perf_counter() - map_start

        self._subtiming_sync()
        selection_start = time.perf_counter()
        selection_inputs = UGPruneInputs(
            fused_map=fused.fused_map,
            map_stats=fused.stats,
            map_key="+".join(fused.layer_keys),
            query_text=query_text,
            query_token_status=query_token_status,
            query_token_positions=[int(position) for position in query_token_positions],
        )
        selection = selector.select_from_map(
            sample_id=sample_id,
            budget=target_budget,
            original_tokens=original_tokens,
            grid=self.grid,
            seed=seed,
            inputs=selection_inputs,
        )
        self._subtiming_sync()
        selection_latency = time.perf_counter() - selection_start
        validate_retained_indices(
            selection.selected_indices,
            original_tokens=original_tokens,
            budget=target_budget,
        )
        stat_diagnostics = {f"ug/map_{key}": value for key, value in selection.metadata.get("map_stats", {}).items()}
        return SelectionPipelineResult(
            selection=selection,
            fused=fused,
            selected_indices_tensor=None,
            map_latency=map_latency,
            selection_latency=selection_latency,
            diagnostics={
                "ug/selection_backend": "numpy_reference",
                "ug/diagnostics_mode": self.diagnostics_mode,
                **stat_diagnostics,
            },
            score_map=fused.fused_map,
        )

    def _new_cache(self) -> Any:
        from transformers.cache_utils import DynamicCache

        with self._profile_scope("cache_construction", implementation="dynamic_cache"):
            try:
                return DynamicCache(config=self.language_model.config)
            except TypeError as exc:
                # Transformers 4.55 accepts ``config`` at the public constructor
                # but forwards an unsupported ``max_cache_len`` argument to its
                # dynamic layer.  The lazily growing form is semantically
                # equivalent for these unbounded decoder caches.
                if "max_cache_len" not in str(exc):
                    raise
                cache = DynamicCache()
                cache.append_new_layers(self.num_layers - 1)
                return cache

    def _base_cache_diagnostics(self, state: LayerAwareCacheState) -> dict[str, Any]:
        return {
            "ug/decode_mode": "layer_aware_cache",
            "ug/cache_implementation": "layer_aware_two_bank_dynamic_cache",
            "mid_layer/early_cache_seq_len": self.cache_seq_len(state.early_cache, 0),
            "mid_layer/late_cache_seq_len": self.cache_seq_len(state.late_cache, state.prune_after_layer),
            "mid_layer/cache_generated_tokens": int(state.generated_tokens_cached),
            "mid_layer/early_attention_mask_len": int(state.early_attention_mask.shape[1]),
            "mid_layer/late_attention_mask_len": int(state.late_attention_mask.shape[1]),
        }

    def prefill_with_layer_aware_cache(
        self,
        *,
        inputs_embeds: Any,
        input_ids: Any,
        attention_mask: Any,
        sample_id: str,
        original_tokens: int,
        seed: int,
        query_text: str,
        query_token_status: str,
        query_token_positions: Sequence[int],
        selection: SelectionResult | None = None,
        selection_mode: str = "ug",
    ) -> LayerAwarePrefillResult:
        if int(inputs_embeds.shape[0]) != 1:
            raise SelectorError("Phase 2B pilot currently supports batch size 1.")
        full_prompt_len = int(inputs_embeds.shape[1])
        full_position_ids = self.position_ids_for(full_prompt_len, inputs_embeds.device)
        early_cache = self._new_cache()

        self._subtiming_sync()
        early_start = time.perf_counter()
        fastv_attention = None
        if selection is None and selection_mode == "fastv":
            early_hidden, fastv_attention = self.run_decoder_layers_with_final_attention(
                inputs_embeds,
                attention_mask=attention_mask,
                position_ids=full_position_ids,
                start_layer=0,
                end_layer=self.prune_after_layer,
                past_key_values=early_cache,
            )
            hidden_states = None
        else:
            early_hidden, hidden_states = self.run_decoder_layers(
                inputs_embeds,
                attention_mask=attention_mask,
                position_ids=full_position_ids,
                start_layer=0,
                end_layer=self.prune_after_layer,
                collect_hidden_states=selection is None,
                past_key_values=early_cache,
                use_cache=True,
            )
        self._subtiming_sync()
        early_latency = time.perf_counter() - early_start
        if selection is None and selection_mode == "ug" and hidden_states is None:
            raise SelectorError("Internal error: hidden states were not captured for UG selection.")

        if selection is not None:
            validate_retained_indices(
                selection.selected_indices,
                original_tokens=original_tokens,
                budget=self.target_visual_tokens,
            )
            selected = SelectionPipelineResult(
                selection=selection,
                fused=None,
                selected_indices_tensor=None,
                map_latency=0.0,
                selection_latency=0.0,
                diagnostics={
                    "ug/selection_backend": str(
                        selection.metadata.get("selection_backend")
                        or selection.metadata.get("selector_family")
                        or "precomputed_selection"
                    ),
                    "ug/diagnostics_mode": self.diagnostics_mode,
                },
            )
        elif selection_mode == "fastv":
            self._subtiming_sync()
            selection_start = time.perf_counter()
            selection = self.fastv_selection_from_attention(
                fastv_attention,
                input_ids=input_ids,
                original_tokens=original_tokens,
            )
            self._subtiming_sync()
            selected = SelectionPipelineResult(
                selection=selection,
                fused=None,
                selected_indices_tensor=None,
                map_latency=0.0,
                selection_latency=time.perf_counter() - selection_start,
                diagnostics={
                    "ug/selection_backend": "official_fastv_attention_rule",
                    "ug/diagnostics_mode": self.diagnostics_mode,
                    "baseline/family": "FastV",
                    "baseline/source_semantics": "mean-head last-prompt-token attention at K-1",
                },
            )
        else:
            selected = self.select_from_hidden_states(
                hidden_states=hidden_states,
                input_ids=input_ids,
                sample_id=sample_id,
                original_tokens=original_tokens,
                seed=seed,
                query_text=query_text,
                query_token_status=query_token_status,
                query_token_positions=query_token_positions,
            )
        selection = selected.selection
        fused = selected.fused
        with self._profile_scope("index_position_mask_gather", method="k12_original"):
            if selected.selected_indices_tensor is not None:
                selected_positions = selected_positions_for_retained_tensor(
                    self.torch,
                    input_ids,
                    self.image_token_id,
                    selected.selected_indices_tensor,
                    sort_retained=True,
                )
            else:
                selected_positions = selected_positions_for_retained(
                    self.torch,
                    input_ids,
                    self.image_token_id,
                    selection.selected_indices,
                    sort_retained=True,
                )
            pruned_hidden = early_hidden[:, selected_positions, :]
            pruned_attention_mask = attention_mask[:, selected_positions].clone()
            if self.selector_config.position_mode == "gap_preserving":
                pruned_position_ids = selected_positions.unsqueeze(0).to(
                    device=inputs_embeds.device,
                    dtype=self.torch.long,
                )
            else:
                pruned_position_ids = self.position_ids_for(
                    int(selected_positions.numel()),
                    inputs_embeds.device,
                )
        late_cache = self._new_cache()
        late_proxy = CacheLengthProxy(late_cache, default_layer_idx=self.prune_after_layer)

        self._subtiming_sync()
        late_start = time.perf_counter()
        late_hidden, _ = self.run_decoder_layers(
            pruned_hidden,
            attention_mask=pruned_attention_mask,
            position_ids=pruned_position_ids,
            start_layer=self.prune_after_layer,
            end_layer=self.num_layers,
            collect_hidden_states=False,
            past_key_values=late_proxy,
            use_cache=True,
        )
        late_hidden = self.language_model.norm(late_hidden)
        logits = self.model.lm_head(late_hidden[:, -1:, :])
        self._subtiming_sync()
        late_latency = time.perf_counter() - late_start

        state = LayerAwareCacheState(
            early_cache=early_cache,
            late_cache=late_cache,
            selection=selection,
            fused=fused,
            selected_positions=selected_positions,
            early_attention_mask=attention_mask.clone(),
            late_attention_mask=pruned_attention_mask,
            full_prompt_len=full_prompt_len,
            pruned_prompt_len=int(selected_positions.numel()),
            prune_after_layer=self.prune_after_layer,
            position_mode=self.selector_config.position_mode,
            generated_tokens_cached=0,
        )
        self.assert_cache_lengths(state)
        with self._profile_scope("prompt_kv_snapshot", method="k12_original"):
            prompt_early_cache_bytes = cache_tensor_bytes(early_cache)
            prompt_late_cache_bytes = cache_tensor_bytes(late_cache)
        diagnostics = {
            "ug/frozen_config": FROZEN_UG_GAP58_NAME,
            "ug/prune_after_layer": self.prune_after_layer,
            "ug/decode_mode": "layer_aware_cache",
            "ug/cache_implementation": "layer_aware_two_bank_dynamic_cache",
            "ug/manual_forward": True,
            "ug/probe_layers": list(self.selector_config.probe_layers),
            "ug/fusion_mode": self.selector_config.fusion_mode,
            "ug/map_source": self.selector_config.map_source,
            "ug/role_mode": self.selector_config.role_mode,
            "ug/context_radius": self.selector_config.context_radius,
            "ug/anchor_pattern": self.selector_config.anchor_pattern,
            "ug/position_mode": self.selector_config.position_mode,
            "mid_layer/early_layers_latency_s": early_latency,
            "mid_layer/map_extraction_latency_s": selected.map_latency,
            "mid_layer/selection_latency_s": selected.selection_latency,
            "mid_layer/late_layers_latency_s": late_latency,
            "mid_layer/selected_sequence_length": int(selected_positions.numel()),
            "mid_layer/selected_positions_first": int(selected_positions[0].item()),
            "mid_layer/selected_positions_last": int(selected_positions[-1].item()),
            "mid_layer/input_sequence_length": int(input_ids.shape[1]),
            "actual_llm_visual_tokens": int(selection.budget),
            "tokens/actual_visual_tokens": int(selection.budget),
            "cache/prompt_early_bank_bytes": prompt_early_cache_bytes,
            "cache/prompt_late_bank_bytes": prompt_late_cache_bytes,
            "cache/prompt_total_bytes": prompt_early_cache_bytes + prompt_late_cache_bytes,
            **selected.diagnostics,
            **self._base_cache_diagnostics(state),
        }
        if selection_mode == "fastv":
            diagnostics.update(
                {
                    "baseline/family": "FastV",
                    "baseline/selection_stage": self.prune_after_layer,
                    "baseline/official_selection_rule": True,
                    "baseline/hf_adaptation": "two-bank static-cache continuation",
                }
            )
        if fused is not None:
            diagnostics["ug_fused_map_layer_keys"] = list(fused.layer_keys)
            diagnostics["ug_selected_layer"] = fused.selected_layer
            diagnostics["ug_layer_scores"] = fused.layer_scores
        return LayerAwarePrefillResult(logits=logits, state=state, diagnostics=diagnostics)

    def decode_one_with_layer_aware_cache(self, state: LayerAwareCacheState, token: Any) -> Any:
        token_embed = self.model.get_input_embeddings()(token).to(
            device=state.early_attention_mask.device,
            dtype=next(self.model.parameters()).dtype,
        )
        early_position_id = self.torch.full_like(token, state.next_early_position_id())
        state.early_attention_mask = self.torch.cat(
            [
                state.early_attention_mask,
                self.torch.ones_like(token, device=state.early_attention_mask.device, dtype=state.early_attention_mask.dtype),
            ],
            dim=1,
        )
        early_hidden, _ = self.run_decoder_layers(
            token_embed,
            attention_mask=state.early_attention_mask,
            position_ids=early_position_id,
            start_layer=0,
            end_layer=state.prune_after_layer,
            collect_hidden_states=False,
            past_key_values=state.early_cache,
            use_cache=True,
        )

        state.late_attention_mask = self.torch.cat(
            [
                state.late_attention_mask,
                self.torch.ones_like(token, device=state.late_attention_mask.device, dtype=state.late_attention_mask.dtype),
            ],
            dim=1,
        )
        late_proxy = CacheLengthProxy(state.late_cache, default_layer_idx=state.prune_after_layer)
        late_position_id = self.torch.full_like(token, state.next_late_position_id())
        late_hidden, _ = self.run_decoder_layers(
            early_hidden,
            attention_mask=state.late_attention_mask,
            position_ids=late_position_id,
            start_layer=state.prune_after_layer,
            end_layer=self.num_layers,
            collect_hidden_states=False,
            past_key_values=late_proxy,
            use_cache=True,
        )
        late_hidden = self.language_model.norm(late_hidden)
        logits = self.model.lm_head(late_hidden[:, -1:, :])
        state.generated_tokens_cached += 1
        self.assert_cache_lengths(state)
        return logits

    def generate_with_layer_aware_cache(
        self,
        *,
        inputs_embeds: Any,
        input_ids: Any,
        attention_mask: Any,
        sample_id: str,
        original_tokens: int,
        seed: int,
        query_text: str,
        query_token_status: str,
        query_token_positions: Sequence[int],
        max_new_tokens: int,
        eos_ids: set[int],
        selection: SelectionResult | None = None,
        selection_mode: str = "ug",
    ) -> LayerAwareGenerationResult:
        generated_ids: list[Any] = []
        self._cuda_sync()
        prefill_start = time.perf_counter()
        with self._profile_scope("first_generated_token", method="k12_original"):
            with self.torch.inference_mode():
                prefill = self.prefill_with_layer_aware_cache(
                    inputs_embeds=inputs_embeds,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sample_id=sample_id,
                    original_tokens=original_tokens,
                    seed=seed,
                    query_text=query_text,
                    query_token_status=query_token_status,
                    query_token_positions=query_token_positions,
                    selection=selection,
                    selection_mode=selection_mode,
                )
                next_token = self.torch.argmax(prefill.logits[:, -1, :], dim=-1, keepdim=True)
        self._cuda_sync()
        prefill_latency = time.perf_counter() - prefill_start
        generated_ids.append(next_token)
        finished = bool(eos_ids and int(next_token.item()) in eos_ids)
        state = prefill.state

        self._cuda_sync()
        decode_start = time.perf_counter()
        with self.torch.inference_mode():
            while len(generated_ids) < max_new_tokens and not finished:
                with self._profile_scope("later_decode_step", method="k12_original", step=len(generated_ids)):
                    logits = self.decode_one_with_layer_aware_cache(state, next_token)
                    next_token = self.torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids.append(next_token)
                finished = bool(eos_ids and int(next_token.item()) in eos_ids)
        self._cuda_sync()
        decode_latency = time.perf_counter() - decode_start

        generated = (
            self.torch.cat(generated_ids, dim=1)[0]
            if generated_ids
            else self.torch.empty(0, dtype=self.torch.long, device=inputs_embeds.device)
        )
        diagnostics = {
            **prefill.diagnostics,
            **self._base_cache_diagnostics(state),
            "llm_prefill_latency_s": prefill_latency,
            "prefill_latency_s": prefill_latency,
            "time_to_first_token_s": prefill_latency,
            "decode_latency_s": decode_latency,
        }
        with self._profile_scope("final_kv_snapshot", method="k12_original"):
            final_early_cache_bytes = cache_tensor_bytes(state.early_cache)
            final_late_cache_bytes = cache_tensor_bytes(state.late_cache)
        diagnostics.update(
            {
                "cache/final_early_bank_bytes": final_early_cache_bytes,
                "cache/final_late_bank_bytes": final_late_cache_bytes,
                "cache/final_total_bytes": final_early_cache_bytes + final_late_cache_bytes,
            }
        )
        return LayerAwareGenerationResult(
            generated_ids=generated,
            selection=state.selection,
            fused=state.fused,
            diagnostics=diagnostics,
        )

    def forward_once(
        self,
        *,
        inputs_embeds: Any,
        input_ids: Any,
        attention_mask: Any,
        sample_id: str,
        original_tokens: int,
        seed: int,
        query_text: str,
        query_token_status: str,
        query_token_positions: Sequence[int],
        selection: SelectionResult | None = None,
    ) -> MidLayerForwardResult:
        if int(inputs_embeds.shape[0]) != 1:
            raise SelectorError("Phase 2B pilot currently supports batch size 1.")
        full_position_ids = self.position_ids_for(int(inputs_embeds.shape[1]), inputs_embeds.device)

        early_start = time.perf_counter()
        early_hidden, hidden_states = self.run_decoder_layers(
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=full_position_ids,
            start_layer=0,
            end_layer=self.prune_after_layer,
            collect_hidden_states=selection is None,
        )
        early_latency = time.perf_counter() - early_start

        fused: FusedMapResult | None = None
        map_latency = 0.0
        selection_latency = 0.0
        if selection is None:
            if hidden_states is None:
                raise SelectorError("Internal error: hidden states were not captured for UG selection.")
            selected = self.select_from_hidden_states(
                hidden_states=hidden_states,
                input_ids=input_ids,
                sample_id=sample_id,
                original_tokens=original_tokens,
                seed=seed,
                query_text=query_text,
                query_token_status=query_token_status,
                query_token_positions=query_token_positions,
            )
            selection = selected.selection
            fused = selected.fused
            selected_indices_tensor = selected.selected_indices_tensor
            map_latency = selected.map_latency
            selection_latency = selected.selection_latency
            selection_diagnostics = selected.diagnostics
        else:
            validate_retained_indices(
                selection.selected_indices,
                original_tokens=original_tokens,
                budget=self.target_visual_tokens,
            )
            selected_indices_tensor = None
            selection_diagnostics = {
                "ug/selection_backend": str(
                    selection.metadata.get("selection_backend")
                    or selection.metadata.get("selector_family")
                    or "precomputed_selection"
                ),
                "ug/diagnostics_mode": self.diagnostics_mode,
            }

        if selected_indices_tensor is not None:
            selected_positions = selected_positions_for_retained_tensor(
                self.torch,
                input_ids,
                self.image_token_id,
                selected_indices_tensor,
                sort_retained=True,
            )
        else:
            selected_positions = selected_positions_for_retained(
                self.torch,
                input_ids,
                self.image_token_id,
                selection.selected_indices,
                sort_retained=True,
            )
        pruned_hidden = early_hidden[:, selected_positions, :]
        pruned_attention_mask = attention_mask[:, selected_positions]
        if self.selector_config.position_mode == "gap_preserving":
            pruned_position_ids = selected_positions.unsqueeze(0).to(
                device=inputs_embeds.device,
                dtype=self.torch.long,
            )
        else:
            pruned_position_ids = self.position_ids_for(
                int(selected_positions.numel()),
                inputs_embeds.device,
            )

        late_start = time.perf_counter()
        late_hidden, _ = self.run_decoder_layers(
            pruned_hidden,
            attention_mask=pruned_attention_mask,
            position_ids=pruned_position_ids,
            start_layer=self.prune_after_layer,
            end_layer=self.num_layers,
            collect_hidden_states=False,
        )
        late_hidden = self.language_model.norm(late_hidden)
        logits = self.model.lm_head(late_hidden[:, -1:, :])
        late_latency = time.perf_counter() - late_start

        diagnostics = {
            "ug/frozen_config": FROZEN_UG_GAP58_NAME,
            "ug/prune_after_layer": self.prune_after_layer,
            "ug/decode_mode": "no_cache_recompute",
            "ug/cache_limitation": "layer_specific_kv_cache_not_used_in_v1",
            "ug/manual_forward": True,
            "ug/probe_layers": list(self.selector_config.probe_layers),
            "ug/fusion_mode": self.selector_config.fusion_mode,
            "ug/map_source": self.selector_config.map_source,
            "ug/role_mode": self.selector_config.role_mode,
            "ug/context_radius": self.selector_config.context_radius,
            "ug/anchor_pattern": self.selector_config.anchor_pattern,
            "ug/position_mode": self.selector_config.position_mode,
            "mid_layer/early_layers_latency_s": early_latency,
            "mid_layer/map_extraction_latency_s": map_latency,
            "mid_layer/selection_latency_s": selection_latency,
            "mid_layer/late_layers_latency_s": late_latency,
            "mid_layer/selected_sequence_length": int(selected_positions.numel()),
            "mid_layer/selected_positions_first": int(selected_positions[0].item()),
            "mid_layer/selected_positions_last": int(selected_positions[-1].item()),
            "mid_layer/input_sequence_length": int(input_ids.shape[1]),
            **selection_diagnostics,
        }
        if fused is not None:
            diagnostics["ug_fused_map_layer_keys"] = list(fused.layer_keys)
            diagnostics["ug_selected_layer"] = fused.selected_layer
            diagnostics["ug_layer_scores"] = fused.layer_scores
        return MidLayerForwardResult(logits=logits, selection=selection, fused=fused, diagnostics=diagnostics)
