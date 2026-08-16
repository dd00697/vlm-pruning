"""Semantics-preserving runtime repairs for progressive LLaVA pruning."""

from __future__ import annotations

from typing import Any, Sequence

from vlm_pruning.llava_hooks.mid_layer_pruning import CacheLengthProxy
from vlm_pruning.llava_hooks.progressive_pruning import (
    ProgressivePruneForwarder,
    SegmentCache,
    _json_checksum,
)
from vlm_pruning.selectors.base import SelectorError


class FusedGatherMergeProgressiveForwarder(ProgressivePruneForwarder):
    """Fuse M448's merge/gather path and reuse a compact raster distance table."""

    def _raster_squared_distances(self, device: Any) -> Any:
        cache = getattr(self, "_repair_distance_cache", None)
        if cache is None:
            cache = {}
            self._repair_distance_cache = cache
        key = str(device)
        if key not in cache:
            indices = self.torch.arange(
                int(self.grid) * int(self.grid),
                dtype=self.torch.long,
                device=device,
            )
            coords = self.torch.stack(
                (indices // int(self.grid), indices % int(self.grid)), dim=1
            ).to(dtype=self.torch.float32)
            # This 576x576 table replaces the per-example 576x448x2
            # broadcast while preserving the exact integer-valued FP32 metric.
            cache[key] = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(dim=2)
        return cache[key]

    def _merge_after_selection(
        self,
        hidden_states: Any,
        *,
        input_ids: Any,
        attention_mask: Any,
        original_positions: Any,
        incoming_original_indices: Sequence[int],
        selected_original_indices: Sequence[int],
        visual_masses: Any,
    ) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
        ids = input_ids[0] if input_ids.ndim == 2 else input_ids
        incoming = [int(value) for value in incoming_original_indices]
        selected = [int(value) for value in selected_original_indices]
        image_positions = self.torch.nonzero(ids == self.image_token_id, as_tuple=False).flatten()
        if int(image_positions.numel()) != len(incoming):
            raise SelectorError("Fused merge image-position count differs")
        if int(visual_masses.numel()) != len(incoming):
            raise SelectorError("Fused merge mass count differs")
        if not selected:
            raise SelectorError("Fused mass-preserving merge requires retained visual tokens")

        with self._profile_scope("fused_gather_merge", method="raster_table_index_add"):
            incoming_tensor = self.torch.tensor(
                incoming, dtype=self.torch.long, device=hidden_states.device
            )
            selected_tensor = self.torch.tensor(
                selected, dtype=self.torch.long, device=hidden_states.device
            )
            distances = self._raster_squared_distances(hidden_states.device)
            distances = distances.index_select(0, incoming_tensor).index_select(1, selected_tensor)
            assignment_slots = self.torch.argmin(distances, dim=1)
            local_by_original = {original: local for local, original in enumerate(incoming)}
            selected_slot = {original: slot for slot, original in enumerate(selected)}
            selected_local = self.torch.tensor(
                [local_by_original[original] for original in selected],
                dtype=self.torch.long,
                device=hidden_states.device,
            )
            assignment_slots.index_copy_(
                0,
                selected_local,
                self.torch.arange(len(selected), dtype=self.torch.long, device=hidden_states.device),
            )

            states = hidden_states[0] if hidden_states.ndim == 3 else hidden_states
            image_states = states.index_select(0, image_positions).to(dtype=self.torch.float32)
            masses = visual_masses.to(device=hidden_states.device, dtype=self.torch.float32)
            weighted_sums = self.torch.zeros(
                (len(selected), int(image_states.shape[1])),
                device=hidden_states.device,
                dtype=self.torch.float32,
            )
            output_masses = self.torch.zeros(
                len(selected), device=hidden_states.device, dtype=self.torch.float32
            )
            weighted_sums.index_add_(0, assignment_slots, image_states * masses[:, None])
            output_masses.index_add_(0, assignment_slots, masses)
            if bool(self.torch.any(output_masses <= 0).item()):
                raise SelectorError("Every retained fused token must have positive mass")
            merged_states = weighted_sums / output_masses[:, None]

            keep_mask = self.torch.ones(ids.shape[0], dtype=self.torch.bool, device=ids.device)
            keep_mask[image_positions] = False
            keep_mask[image_positions.index_select(0, selected_local)] = True
            keep_positions = self.torch.nonzero(keep_mask, as_tuple=False).flatten()
            gathered_hidden = hidden_states.index_select(1, keep_positions).clone()
            gathered_ids = input_ids.index_select(1, keep_positions)
            gathered_mask = attention_mask.index_select(1, keep_positions)
            gathered_positions = original_positions.index_select(0, keep_positions)
            gathered_image_positions = self.torch.nonzero(
                gathered_ids[0] == self.image_token_id, as_tuple=False
            ).flatten()
            if int(gathered_image_positions.numel()) != len(selected):
                raise SelectorError("Fused gather did not realize the selected budget")
            gathered_hidden[0, gathered_image_positions] = merged_states.to(
                dtype=gathered_hidden.dtype
            )

        dropped_assignments = [
            [original, selected[int(assignment_slots[local].item())]]
            for local, original in enumerate(incoming)
            if original not in selected_slot
        ]
        mass_values = [float(value) for value in output_masses.detach().cpu().tolist()]
        diagnostics = {
            "merge_assignment_count": len(dropped_assignments),
            "merge_assignment_checksum": _json_checksum(dropped_assignments),
            "merge_output_masses": mass_values,
            "merge_mass_checksum": _json_checksum(mass_values),
            "merge_input_mass": float(masses.sum().item()),
            "merge_output_mass": float(output_masses.sum().item()),
        }
        return (
            gathered_hidden,
            gathered_ids,
            gathered_mask,
            gathered_positions,
            output_masses,
            diagnostics,
        )


class NativeCacheProgressiveForwarder(ProgressivePruneForwarder):
    """Pass native DynamicCache directly to every decoder layer.

    A short-lived length view is used only once to construct the segment mask;
    the hot per-layer update path receives the native cache object directly.
    """

    def _cache_for_segment(self, segment: SegmentCache) -> Any:
        return segment

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
        if not isinstance(past_key_values, SegmentCache):
            return super().run_decoder_layers(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                start_layer=start_layer,
                end_layer=end_layer,
                collect_hidden_states=collect_hidden_states,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        if start_layer != past_key_values.start_layer or end_layer != past_key_values.end_layer:
            raise SelectorError("Native cache segment/layer range differs")
        native_cache = past_key_values.cache
        with self._profile_scope(
            "native_cache_direct",
            segment_index=past_key_values.segment_index,
            start_layer=start_layer,
        ):
            mask_view = CacheLengthProxy(native_cache, default_layer_idx=start_layer)
            causal_mask = self._causal_mask(
                hidden_states, attention_mask, position_ids, mask_view
            )
            cache_position = self._cache_position(hidden_states, mask_view)
        position_embeddings = self.language_model.rotary_emb(
            hidden_states, position_ids=position_ids
        )
        captured: list[Any] | None = [hidden_states] if collect_hidden_states else None
        for layer_index in range(start_layer, end_layer):
            output = self.decoder_layers[layer_index](
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_value=native_cache,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = output[0] if isinstance(output, (tuple, list)) else output
            if captured is not None:
                captured.append(hidden_states)
        return hidden_states, captured

    def assert_segment_cache_lengths(self, state: Any) -> None:
        for segment in state.segments:
            expected = segment.prompt_length + state.generated_tokens_cached
            if int(segment.attention_mask.shape[1]) != expected:
                raise SelectorError("Native segment attention-mask length differs")
            for layer in range(segment.start_layer, segment.end_layer):
                if self.cache_seq_len(segment.cache, layer) != expected:
                    raise SelectorError(f"Native layer {layer} cache length differs")


class FusedNativeProgressiveForwarder(
    FusedGatherMergeProgressiveForwarder,
    NativeCacheProgressiveForwarder,
):
    """Compose both exact repairs without changing M448's algorithm."""

