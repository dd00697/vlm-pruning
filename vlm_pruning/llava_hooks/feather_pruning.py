"""Source-faithful FEATHER token selection on the canonical HF LLaVA stack.

The upstream FEATHER implementation is bound to its Prismatic/SigLIP model
class.  This forwarder preserves its decoder-side Llama semantics exactly:
RoPE-free last-text-token attention is captured immediately before layers 7
and 15, pruning starts at layers 8 and 16, the first selection is the sorted
union of attention top-k and a stride-three 2-D grid, and the second selection
uses the squared keep ratio over the surviving tokens.
"""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

from vlm_pruning.efficiency import cache_tensor_bytes
from vlm_pruning.llava_hooks.progressive_pruning import (
    ProgressiveCacheState,
    ProgressivePrefillResult,
    ProgressivePruneForwarder,
    SegmentCache,
    StageTrace,
    _json_checksum,
    _tensor_checksum,
)
from vlm_pruning.selectors.base import SelectorError


FEATHER_BOUNDARIES = (8, 16)
FEATHER_UPSTREAM_REDUCTION_RATIO = 0.7
FEATHER_KEEP_RATIO = 1.0 - FEATHER_UPSTREAM_REDUCTION_RATIO
FEATHER_STRIDE = 3


def feather_select_original_indices(
    torch: Any,
    scores: Any,
    incoming_original_indices: Sequence[int],
    *,
    stage_index: int,
    keep_ratio: float = FEATHER_KEEP_RATIO,
    stride: int = FEATHER_STRIDE,
) -> tuple[list[int], dict[str, Any]]:
    """Apply the exact upstream FEATHER top-k/union/order operations."""
    incoming = [int(value) for value in incoming_original_indices]
    if stage_index not in {0, 1}:
        raise SelectorError(f"FEATHER supports exactly two stages, got {stage_index}")
    if int(scores.numel()) != len(incoming):
        raise SelectorError("FEATHER score count differs from the incoming visual-token count")
    if not 0.0 < float(keep_ratio) < 1.0 or int(stride) <= 0:
        raise SelectorError("Invalid frozen FEATHER ratio or stride")

    if stage_index == 0:
        topk_count = int(len(incoming) * float(keep_ratio))
        if topk_count <= 0:
            raise SelectorError("FEATHER first-stage top-k is empty")
        top_local = torch.topk(scores, topk_count).indices
        grid_size = int(math.sqrt(len(incoming)))
        if grid_size * grid_size != len(incoming):
            raise SelectorError("FEATHER first stage requires a square visual grid")
        grid = torch.arange(len(incoming), device=scores.device, dtype=torch.long).reshape(
            grid_size, grid_size
        )
        strided_local = grid[:: int(stride), :: int(stride)].flatten()
        # torch.unique(sorted=True) is the upstream union and ordering rule.
        selected_local = torch.unique(
            torch.cat((top_local, strided_local)), sorted=True
        )
    else:
        topk_count = int(len(incoming) * (float(keep_ratio) * float(keep_ratio)))
        if topk_count <= 0:
            raise SelectorError("FEATHER second-stage top-k is empty")
        top_local = torch.topk(scores, topk_count).indices
        strided_local = torch.empty(0, device=scores.device, dtype=torch.long)
        # Upstream sorts the local positions before mapping through the
        # first-stage original-position vector.
        selected_local = torch.sort(top_local).values

    local_values = [int(value) for value in selected_local.detach().cpu().tolist()]
    selected = [incoming[local] for local in local_values]
    if selected != sorted(selected) or len(selected) != len(set(selected)):
        raise SelectorError("FEATHER did not preserve unique raster ordering")
    return selected, {
        "stage_index": int(stage_index),
        "incoming_tokens": len(incoming),
        "attention_topk_count": int(topk_count),
        "uniform_stride_count": int(strided_local.numel()),
        "realized_tokens": len(selected),
        "keep_ratio": float(keep_ratio),
        "squared_ratio_at_second_stage": stage_index == 1,
        "stride": int(stride),
        "selected_local_indices": local_values,
    }


class FeatherLlavaForwarder(ProgressivePruneForwarder):
    """Execute FEATHER's fixed two-stage selection with physical cache banks."""

    def __init__(
        self,
        torch: Any,
        model: Any,
        *,
        image_token_id: int,
        grid: int = 24,
        initial_visual_tokens: int = 576,
        diagnostics_mode: str = "benchmark",
        synchronise_subtimings: bool = False,
    ) -> None:
        # The adaptive union has no fixed event budget, so the inherited
        # schedule is intentionally empty and this class owns both boundaries.
        super().__init__(
            torch,
            model,
            image_token_id=image_token_id,
            schedule=(),
            grid=grid,
            initial_visual_tokens=initial_visual_tokens,
            diagnostics_mode=diagnostics_mode,
            synchronise_subtimings=synchronise_subtimings,
        )
        if self.num_layers != 32 or self.grid != 24 or self.initial_visual_tokens != 576:
            raise SelectorError("Registered FEATHER LLaVA adapter requires 32 layers and a 24x24 grid")
        self.reset_feather_timings()

    def reset_feather_timings(self) -> None:
        self.feather_score_latency_s = 0.0
        self.feather_selector_latency_s = 0.0
        self.feather_gather_latency_s = 0.0
        self.feather_stage_metadata: list[dict[str, Any]] = []

    def _rope_free_attention_scores(
        self,
        hidden_states: Any,
        *,
        input_ids: Any,
        layer_index: int,
    ) -> Any:
        """Reproduce upstream ``calculate_attention_weight`` without RoPE."""
        torch = self.torch
        layer = self.decoder_layers[int(layer_index)]
        attention = layer.self_attn
        normalised = layer.input_layernorm(hidden_states)
        query_states = attention.q_proj(normalised)
        key_states = attention.k_proj(normalised)
        batch, sequence, _ = normalised.shape
        num_heads = int(getattr(attention, "num_heads", self.language_model.config.num_attention_heads))
        num_kv_heads = int(
            getattr(attention, "num_key_value_heads", self.language_model.config.num_key_value_heads)
        )
        head_dim = int(getattr(attention, "head_dim", query_states.shape[-1] // num_heads))
        query_states = query_states.view(batch, sequence, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch, sequence, num_kv_heads, head_dim).transpose(1, 2)
        groups = num_heads // num_kv_heads
        if groups * num_kv_heads != num_heads:
            raise SelectorError("FEATHER attention head grouping is invalid")
        if groups > 1:
            key_states = key_states.repeat_interleave(groups, dim=1)
        ids = input_ids[0] if input_ids.ndim == 2 else input_ids
        image_positions = torch.nonzero(ids == self.image_token_id, as_tuple=False).flatten()
        if int(image_positions.numel()) == 0:
            raise SelectorError("FEATHER cannot score an empty visual sequence")
        text_query_states = query_states[:, :, -1:, :]
        image_key_states = key_states.index_select(2, image_positions)
        weights = torch.matmul(
            text_query_states, image_key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)
        weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        return weights.mean(dim=1)[0, 0]

    def _run_segment_and_capture_score(
        self,
        hidden_states: Any,
        *,
        input_ids: Any,
        attention_mask: Any,
        position_ids: Any,
        start_layer: int,
        end_layer: int,
        cache: Any,
    ) -> tuple[Any, Any, float]:
        """Run one segment while scoring before its final decoder layer."""
        if end_layer <= start_layer:
            raise SelectorError("FEATHER scoring segment must contain a decoder layer")
        causal_mask = self._causal_mask(hidden_states, attention_mask, position_ids, cache)
        cache_position = self._cache_position(hidden_states, cache)
        position_embeddings = self.language_model.rotary_emb(
            hidden_states, position_ids=position_ids
        )
        score = None
        score_latency = 0.0
        for layer_index in range(start_layer, end_layer):
            if layer_index == end_layer - 1:
                self._subtiming_sync()
                started = time.perf_counter()
                score = self._rope_free_attention_scores(
                    hidden_states, input_ids=input_ids, layer_index=layer_index
                )
                self._subtiming_sync()
                score_latency = time.perf_counter() - started
            output = self.decoder_layers[layer_index](
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_value=cache,
                use_cache=True,
                cache_position=cache_position,
            )
            hidden_states = output[0] if isinstance(output, (tuple, list)) else output
        if score is None:
            raise SelectorError("FEATHER failed to capture its RoPE-free score")
        return hidden_states, score, score_latency

    def prefill_with_segment_caches(
        self,
        *,
        inputs_embeds: Any,
        input_ids: Any,
        attention_mask: Any,
        sample_id: str,
        seed: int,
        query_original_positions: Sequence[int],
        initial_retained_indices: Sequence[int] | None = None,
        initial_selector: str | None = None,
        initial_selection_checksum: str | None = None,
        initial_derived_random_seed: int | None = None,
    ) -> ProgressivePrefillResult:
        del sample_id, query_original_positions
        if seed != 0:
            raise SelectorError("Registered FEATHER inference uses seed 0")
        if any(
            value is not None
            for value in (
                initial_retained_indices,
                initial_selector,
                initial_selection_checksum,
                initial_derived_random_seed,
            )
        ):
            raise SelectorError("FEATHER does not accept a pre-LLM selection")
        if tuple(inputs_embeds.shape[:1]) != (1,) or tuple(input_ids.shape[:1]) != (1,):
            raise SelectorError("FEATHER LLaVA adapter supports batch size 1 only")

        self.reset_feather_timings()
        ids = input_ids[0]
        image_positions = self.torch.nonzero(
            ids == self.image_token_id, as_tuple=False
        ).flatten()
        if int(image_positions.numel()) != self.initial_visual_tokens:
            raise SelectorError(
                f"Expected {self.initial_visual_tokens} visual tokens, found {image_positions.numel()}"
            )
        full_prompt_length = int(input_ids.shape[1])
        original_positions = self.torch.arange(
            full_prompt_length, device=input_ids.device, dtype=self.torch.long
        )
        original_text_checksum, original_text_position_checksum = self._text_evidence(
            input_ids, original_positions
        )
        original_visual_positions = [int(value) for value in image_positions.detach().cpu().tolist()]
        original_text_positions = [
            int(value)
            for value in original_positions[ids != self.image_token_id].detach().cpu().tolist()
        ]
        hidden = inputs_embeds
        current_ids = input_ids
        current_mask = attention_mask
        current_positions = original_positions
        surviving = list(range(self.initial_visual_tokens))
        visual_masses = self.torch.ones(
            self.initial_visual_tokens, device=inputs_embeds.device, dtype=self.torch.float32
        )
        segments: list[SegmentCache] = []
        traces: list[StageTrace] = []
        start_layer = 0

        for stage_index, boundary in enumerate(FEATHER_BOUNDARIES):
            segment = self._new_segment(
                segment_index=len(segments),
                start_layer=start_layer,
                end_layer=boundary,
                attention_mask=current_mask,
                prompt_length=int(current_ids.shape[1]),
                visual_tokens=len(surviving),
            )
            hidden, scores, score_latency = self._run_segment_and_capture_score(
                hidden,
                input_ids=current_ids,
                attention_mask=current_mask,
                position_ids=current_positions.unsqueeze(0),
                start_layer=start_layer,
                end_layer=boundary,
                cache=self._cache_for_segment(segment),
            )
            segments.append(segment)
            self.feather_score_latency_s += score_latency
            self._subtiming_sync()
            selector_started = time.perf_counter()
            selected, metadata = feather_select_original_indices(
                self.torch, scores, surviving, stage_index=stage_index
            )
            self._subtiming_sync()
            selector_latency = time.perf_counter() - selector_started
            self.feather_selector_latency_s += selector_latency
            metadata["boundary"] = boundary
            metadata["score_sha256"] = _tensor_checksum(scores)
            self.feather_stage_metadata.append(metadata)
            incoming = list(surviving)
            before_length = int(current_ids.shape[1])
            self._subtiming_sync()
            gather_started = time.perf_counter()
            hidden, current_ids, current_mask, current_positions = self._gather_after_selection(
                hidden,
                input_ids=current_ids,
                attention_mask=current_mask,
                original_positions=current_positions,
                incoming_original_indices=incoming,
                selected_original_indices=selected,
            )
            visual_masses = self._selected_visual_masses(
                visual_masses,
                incoming_original_indices=incoming,
                selected_original_indices=selected,
            )
            self._subtiming_sync()
            gather_latency = time.perf_counter() - gather_started
            self.feather_gather_latency_s += gather_latency
            text_checksum, text_position_checksum = self._text_evidence(
                current_ids, current_positions
            )
            if (
                text_checksum != original_text_checksum
                or text_position_checksum != original_text_position_checksum
            ):
                raise SelectorError(f"FEATHER K{boundary} altered text tokens or positions")
            traces.append(
                StageTrace(
                    stage_index=stage_index,
                    boundary=boundary,
                    selector=(
                        "feather_no_rope_topk_union_stride3"
                        if stage_index == 0
                        else "feather_no_rope_squared_keep_ratio"
                    ),
                    role="runtime_pruning",
                    requested_budget=len(selected),
                    realized_budget=len(selected),
                    incoming_original_indices=incoming,
                    outgoing_original_indices=selected,
                    incoming_indices_checksum=_json_checksum(incoming),
                    outgoing_indices_checksum=_json_checksum(selected),
                    score_map_checksum=_tensor_checksum(scores),
                    score_map_recomputed=True,
                    derived_random_seed=None,
                    incoming_sequence_length=before_length,
                    outgoing_sequence_length=int(current_ids.shape[1]),
                    text_token_checksum=text_checksum,
                    text_position_checksum=text_position_checksum,
                    outgoing_position_ids=[
                        int(value) for value in current_positions.detach().cpu().tolist()
                    ],
                    position_id_checksum=_json_checksum(
                        [int(value) for value in current_positions.detach().cpu().tolist()]
                    ),
                    selection_latency_s=selector_latency,
                )
            )
            surviving = selected
            start_layer = boundary

        final_segment = self._new_segment(
            segment_index=len(segments),
            start_layer=start_layer,
            end_layer=self.num_layers,
            attention_mask=current_mask,
            prompt_length=int(current_ids.shape[1]),
            visual_tokens=len(surviving),
        )
        hidden, _ = self.run_decoder_layers(
            hidden,
            attention_mask=current_mask,
            position_ids=current_positions.unsqueeze(0),
            start_layer=start_layer,
            end_layer=self.num_layers,
            past_key_values=self._cache_for_segment(final_segment),
            use_cache=True,
        )
        segments.append(final_segment)
        hidden = self.language_model.norm(hidden)
        logits = self.model.lm_head(hidden[:, -1:, :])
        realized_trace: list[int] = []
        for segment in segments:
            realized_trace.extend(
                [segment.visual_tokens] * (segment.end_layer - segment.start_layer)
            )
        state = ProgressiveCacheState(
            segments=segments,
            stage_traces=traces,
            final_retained_indices=list(surviving),
            full_prompt_length=full_prompt_length,
            final_prompt_length=int(current_ids.shape[1]),
            original_text_token_checksum=original_text_checksum,
            original_text_position_checksum=original_text_position_checksum,
            original_visual_positions=original_visual_positions,
            original_text_positions=original_text_positions,
            realized_visual_tokens_by_layer=realized_trace,
        )
        self.assert_segment_cache_lengths(state)
        diagnostics = self._diagnostics(state)
        return ProgressivePrefillResult(logits=logits, state=state, diagnostics=diagnostics)

    def _diagnostics(self, state: ProgressiveCacheState) -> dict[str, Any]:
        result = super()._diagnostics(state)
        result.update(
            {
                "feather/source_semantics": "official_no_rope_s=3_ratio=0.7_boundaries=8,16",
                "feather/score_latency_s": self.feather_score_latency_s,
                "feather/selector_latency_s": self.feather_selector_latency_s,
                "feather/gather_latency_s": self.feather_gather_latency_s,
                "feather/stage_metadata": list(self.feather_stage_metadata),
                "cache/prompt_total_bytes": sum(
                    cache_tensor_bytes(segment.cache) for segment in state.segments
                ),
            }
        )
        return result
