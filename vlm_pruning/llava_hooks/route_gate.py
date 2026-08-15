"""Train-time features and one-pass dynamic routing for the registered gate."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from vlm_pruning.efficiency import cache_tensor_bytes
from vlm_pruning.llava_hooks.progressive_pruning import (
    ProgressiveCacheState,
    ProgressiveGenerationResult,
    ProgressivePrefillResult,
    ProgressivePruneForwarder,
    PruningEvent,
    StageTrace,
    validate_schedule,
)
from vlm_pruning.selectors.base import SelectorError


GATE_BOUNDARY = 2
HIDDEN_SIZE = 4096
SAFE_ROUTE = "K12_QUERY_TOPK"
AGGRESSIVE_ROUTE = "M448"
SAFE_SCHEDULE = (PruningEvent(12, 48, "query_topk"),)
AGGRESSIVE_SCHEDULE = (
    PruningEvent(4, 448, "query_topk_merge"),
    PruningEvent(12, 48, "query_topk"),
)
SCALAR_FEATURE_NAMES = (
    "score_mean",
    "score_std",
    "score_min",
    "score_max",
    "softmax_entropy_normalized",
    "softmax_top1_mass",
    "softmax_top8_mass",
    "softmax_top48_mass",
    "score_top1_top2_margin",
    "score_top48_top49_margin",
    "top48_row_std_normalized",
    "top48_col_std_normalized",
    "top48_bbox_area_fraction",
    "top48_quadrant_occupancy_fraction",
    "query_token_count",
    "log_image_aspect_ratio",
    "log_image_pixels",
)
FEATURE_DIMENSION = HIDDEN_SIZE * 2 + len(SCALAR_FEATURE_NAMES)
FEATURE_SCHEMA = {
    "schema_version": 1,
    "gate_boundary": GATE_BOUNDARY,
    "blocks": [
        {
            "name": "pooled_query_hidden_l2_normalized",
            "dimension": HIDDEN_SIZE,
            "source": "mean K2 query-token hidden state then L2 normalization",
        },
        {
            "name": "pooled_visual_hidden_l2_normalized",
            "dimension": HIDDEN_SIZE,
            "source": "mean K2 visual-token hidden state then L2 normalization",
        },
        {"name": "registered_scalars", "dimension": len(SCALAR_FEATURE_NAMES), "fields": list(SCALAR_FEATURE_NAMES)},
    ],
    "score": "maximum raw cosine similarity over resolved query tokens for each of 576 patches",
    "spatial_grid": [24, 24],
    "topk_summary": 48,
    "dimension": FEATURE_DIMENSION,
}


def canonical_json_checksum(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


FEATURE_SCHEMA_SHA256 = canonical_json_checksum(FEATURE_SCHEMA)


@dataclass(frozen=True)
class GateFeatureResult:
    vector: Any
    scalar_values: dict[str, float]
    vector_sha256: str


@dataclass(frozen=True)
class RouteDecision:
    route_name: str
    probability_safe: float
    threshold: float
    schedule: tuple[PruningEvent, ...]

    @property
    def aggressive(self) -> bool:
        return self.route_name == AGGRESSIVE_ROUTE


def feature_tensor_sha256(vector: Any) -> str:
    payload = vector.detach().to(device="cpu", dtype=vector.float().dtype).contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def route_schedule(name: str) -> tuple[PruningEvent, ...]:
    if name == SAFE_ROUTE:
        return SAFE_SCHEDULE
    if name == AGGRESSIVE_ROUTE:
        return AGGRESSIVE_SCHEDULE
    raise SelectorError(f"Unregistered route-gate route {name!r}")


class DynamicRouteForwarder(ProgressivePruneForwarder):
    """Execute a dense K2 prefix once, choose one route, and execute only it."""

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

    def extract_gate_features(
        self,
        hidden_states: Any,
        *,
        input_ids: Any,
        original_positions: Any,
        query_original_positions: Sequence[int],
        width: int,
        height: int,
    ) -> GateFeatureResult:
        torch = self.torch
        ids = input_ids[0] if input_ids.ndim == 2 else input_ids
        image_positions = torch.nonzero(ids == self.image_token_id, as_tuple=False).flatten()
        if int(image_positions.numel()) != self.initial_visual_tokens:
            raise SelectorError("Route-gate features require all 576 dense K2 visual tokens")
        query_positions = self._current_query_positions(original_positions, query_original_positions)
        states = hidden_states[0] if hidden_states.ndim == 3 else hidden_states
        image_states = states.index_select(0, image_positions).float()
        query_states = states.index_select(0, query_positions).float()
        image_norm = torch.nn.functional.normalize(image_states, p=2, dim=-1)
        query_norm = torch.nn.functional.normalize(query_states, p=2, dim=-1)
        pooled_query = torch.nn.functional.normalize(query_states.mean(dim=0), p=2, dim=0)
        pooled_visual = torch.nn.functional.normalize(image_states.mean(dim=0), p=2, dim=0)
        scores = (image_norm @ query_norm.transpose(0, 1)).max(dim=1).values
        ranked = torch.argsort(scores, descending=True, stable=True)
        sorted_scores = scores.index_select(0, ranked)
        probabilities = torch.softmax(scores, dim=0)
        sorted_probabilities = probabilities.index_select(0, ranked)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum() / math.log(
            self.initial_visual_tokens
        )
        top48 = ranked[:48]
        rows = (top48 // int(self.grid)).float()
        cols = (top48 % int(self.grid)).float()
        quadrants = (rows >= self.grid / 2).long() * 2 + (cols >= self.grid / 2).long()
        quadrant_occupancy = torch.unique(quadrants).numel() / 4.0
        scalar_tensors = (
            scores.mean(),
            scores.std(unbiased=False),
            scores.min(),
            scores.max(),
            entropy,
            sorted_probabilities[:1].sum(),
            sorted_probabilities[:8].sum(),
            sorted_probabilities[:48].sum(),
            sorted_scores[0] - sorted_scores[1],
            sorted_scores[47] - sorted_scores[48],
            rows.std(unbiased=False) / (self.grid - 1),
            cols.std(unbiased=False) / (self.grid - 1),
            ((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)) / (self.grid**2),
            scores.new_tensor(float(quadrant_occupancy)),
            scores.new_tensor(float(query_positions.numel())),
            scores.new_tensor(math.log(max(float(width), 1.0) / max(float(height), 1.0))),
            scores.new_tensor(math.log(max(float(width * height), 1.0))),
        )
        scalars = torch.stack(scalar_tensors).to(dtype=torch.float32)
        if int(pooled_query.numel()) != HIDDEN_SIZE or int(pooled_visual.numel()) != HIDDEN_SIZE:
            raise SelectorError("Route-gate hidden size differs from the frozen 4,096-dimensional schema")
        vector = torch.cat((pooled_query, pooled_visual, scalars), dim=0).to(dtype=torch.float32)
        if int(vector.numel()) != FEATURE_DIMENSION or not bool(torch.isfinite(vector).all().item()):
            raise SelectorError("Route-gate feature vector is malformed or non-finite")
        scalar_values = {
            name: float(value)
            for name, value in zip(SCALAR_FEATURE_NAMES, scalars.detach().cpu().tolist(), strict=True)
        }
        return GateFeatureResult(
            vector=vector,
            scalar_values=scalar_values,
            vector_sha256=feature_tensor_sha256(vector),
        )

    def features_after_k2(
        self,
        *,
        inputs_embeds: Any,
        input_ids: Any,
        attention_mask: Any,
        query_original_positions: Sequence[int],
        width: int,
        height: int,
    ) -> GateFeatureResult:
        original_positions = self.torch.arange(
            int(input_ids.shape[1]), device=input_ids.device, dtype=self.torch.long
        )
        hidden, _ = self.run_decoder_layers(
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=original_positions.unsqueeze(0),
            start_layer=0,
            end_layer=GATE_BOUNDARY,
            collect_hidden_states=False,
            use_cache=False,
        )
        return self.extract_gate_features(
            hidden,
            input_ids=input_ids,
            original_positions=original_positions,
            query_original_positions=query_original_positions,
            width=width,
            height=height,
        )

    @staticmethod
    def _validate_decision(decision: RouteDecision) -> tuple[PruningEvent, ...]:
        expected = route_schedule(decision.route_name)
        actual = validate_schedule(decision.schedule)
        if actual != expected:
            raise SelectorError(f"Route {decision.route_name} did not select its frozen schedule")
        if not math.isfinite(float(decision.probability_safe)) or not 0 <= float(decision.probability_safe) <= 1:
            raise SelectorError("Route-gate probability is outside [0, 1]")
        if not math.isfinite(float(decision.threshold)):
            raise SelectorError("Route-gate threshold is not finite")
        if actual[0].boundary <= GATE_BOUNDARY:
            raise SelectorError("A route prunes at or before the registered K2 gate")
        return actual

    def prefill_with_route_gate(
        self,
        *,
        inputs_embeds: Any,
        input_ids: Any,
        attention_mask: Any,
        sample_id: str,
        seed: int,
        query_original_positions: Sequence[int],
        width: int,
        height: int,
        decide: Callable[[Any], RouteDecision],
    ) -> ProgressivePrefillResult:
        if int(inputs_embeds.shape[0]) != 1 or int(input_ids.shape[0]) != 1:
            raise SelectorError("Dynamic route gating supports batch size 1 only")
        ids = input_ids[0]
        original_image_positions = self.torch.nonzero(
            ids == self.image_token_id, as_tuple=False
        ).flatten()
        if int(original_image_positions.numel()) != self.initial_visual_tokens:
            raise SelectorError("Dynamic route gating requires the 576-token dense prompt")
        full_prompt_length = int(input_ids.shape[1])
        original_positions = self.torch.arange(
            full_prompt_length, device=input_ids.device, dtype=self.torch.long
        )
        original_text_checksum, original_text_position_checksum = self._text_evidence(
            input_ids, original_positions
        )
        original_visual_position_list = [
            int(value) for value in original_image_positions.detach().cpu().tolist()
        ]
        original_text_position_list = [
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
        traces: list[StageTrace] = []
        segments = []

        shared_segment = self._new_segment(
            segment_index=0,
            start_layer=0,
            end_layer=GATE_BOUNDARY,
            attention_mask=current_mask,
            prompt_length=int(current_ids.shape[1]),
            visual_tokens=len(surviving),
        )
        hidden, _ = self.run_decoder_layers(
            hidden,
            attention_mask=current_mask,
            position_ids=current_positions.unsqueeze(0),
            start_layer=0,
            end_layer=GATE_BOUNDARY,
            collect_hidden_states=False,
            past_key_values=self._cache_for_segment(shared_segment),
            use_cache=True,
        )
        segments.append(shared_segment)
        self._subtiming_sync()
        gate_started = time.perf_counter()
        features = self.extract_gate_features(
            hidden,
            input_ids=current_ids,
            original_positions=current_positions,
            query_original_positions=query_original_positions,
            width=width,
            height=height,
        )
        decision = decide(features.vector)
        schedule = self._validate_decision(decision)
        self._subtiming_sync()
        gate_latency = time.perf_counter() - gate_started

        start_layer = GATE_BOUNDARY
        for event_index, event in enumerate(schedule):
            segment = self._new_segment(
                segment_index=len(segments),
                start_layer=start_layer,
                end_layer=event.boundary,
                attention_mask=current_mask,
                prompt_length=int(current_ids.shape[1]),
                visual_tokens=len(surviving),
            )
            hidden, _ = self.run_decoder_layers(
                hidden,
                attention_mask=current_mask,
                position_ids=current_positions.unsqueeze(0),
                start_layer=start_layer,
                end_layer=event.boundary,
                collect_hidden_states=False,
                past_key_values=self._cache_for_segment(segment),
                use_cache=True,
            )
            segments.append(segment)
            selection_started = time.perf_counter()
            selected, score_checksum, recomputed, derived_seed = self.select_stage_indices(
                hidden,
                input_ids=current_ids,
                original_positions=current_positions,
                incoming_original_indices=surviving,
                event=event,
                stage_index=event_index,
                sample_id=sample_id,
                seed=seed,
                query_original_positions=query_original_positions,
            )
            selection_latency = time.perf_counter() - selection_started
            before_len = int(current_ids.shape[1])
            incoming = list(surviving)
            merge_diagnostics: dict[str, Any] = {}
            merge_latency = 0.0
            if event.selector == "query_topk_merge":
                merge_started = time.perf_counter()
                (
                    hidden,
                    current_ids,
                    current_mask,
                    current_positions,
                    visual_masses,
                    merge_diagnostics,
                ) = self._merge_after_selection(
                    hidden,
                    input_ids=current_ids,
                    attention_mask=current_mask,
                    original_positions=current_positions,
                    incoming_original_indices=incoming,
                    selected_original_indices=selected,
                    visual_masses=visual_masses,
                )
                merge_latency = time.perf_counter() - merge_started
            else:
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
            text_checksum, text_position_checksum = self._text_evidence(current_ids, current_positions)
            if text_checksum != original_text_checksum or text_position_checksum != original_text_position_checksum:
                raise SelectorError(f"Dynamic route K{event.boundary} altered text evidence")
            outgoing_positions = [int(value) for value in current_positions.detach().cpu().tolist()]
            traces.append(
                StageTrace(
                    stage_index=event_index,
                    boundary=event.boundary,
                    selector=event.selector,
                    role="runtime_pruning",
                    requested_budget=event.budget,
                    realized_budget=len(selected),
                    incoming_original_indices=incoming,
                    outgoing_original_indices=selected,
                    incoming_indices_checksum=canonical_json_checksum(incoming),
                    outgoing_indices_checksum=canonical_json_checksum(selected),
                    score_map_checksum=score_checksum,
                    score_map_recomputed=recomputed,
                    derived_random_seed=derived_seed,
                    incoming_sequence_length=before_len,
                    outgoing_sequence_length=int(current_ids.shape[1]),
                    text_token_checksum=text_checksum,
                    text_position_checksum=text_position_checksum,
                    outgoing_position_ids=outgoing_positions,
                    position_id_checksum=canonical_json_checksum(outgoing_positions),
                    selection_latency_s=selection_latency,
                    merge_applied=event.selector == "query_topk_merge",
                    merge_assignment_count=int(merge_diagnostics.get("merge_assignment_count", 0)),
                    merge_assignment_checksum=merge_diagnostics.get("merge_assignment_checksum"),
                    merge_output_masses=merge_diagnostics.get("merge_output_masses"),
                    merge_mass_checksum=merge_diagnostics.get("merge_mass_checksum"),
                    merge_input_mass=merge_diagnostics.get("merge_input_mass"),
                    merge_output_mass=merge_diagnostics.get("merge_output_mass"),
                    merge_latency_s=merge_latency,
                )
            )
            surviving = selected
            start_layer = event.boundary

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
            collect_hidden_states=False,
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
            original_visual_positions=original_visual_position_list,
            original_text_positions=original_text_position_list,
            realized_visual_tokens_by_layer=realized_trace,
        )
        self.assert_segment_cache_lengths(state)
        diagnostics = {
            **self._diagnostics(state),
            "gate/shared_dense_prefix_boundary": GATE_BOUNDARY,
            "gate/shared_dense_prefix_executions": 1,
            "gate/feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "gate/feature_dimension": FEATURE_DIMENSION,
            "gate/feature_vector_sha256": features.vector_sha256,
            "gate/feature_scalars": features.scalar_values,
            "gate/latency_s": gate_latency,
            "gate/route_name": decision.route_name,
            "gate/aggressive": decision.aggressive,
            "gate/probability_safe": float(decision.probability_safe),
            "gate/threshold": float(decision.threshold),
        }
        return ProgressivePrefillResult(logits=logits, state=state, diagnostics=diagnostics)

    def generate_with_route_gate(
        self,
        *,
        inputs_embeds: Any,
        input_ids: Any,
        attention_mask: Any,
        sample_id: str,
        seed: int,
        query_original_positions: Sequence[int],
        width: int,
        height: int,
        decide: Callable[[Any], RouteDecision],
        max_new_tokens: int,
        eos_ids: set[int],
    ) -> ProgressiveGenerationResult:
        self._cuda_sync()
        prefill_started = time.perf_counter()
        with self.torch.inference_mode():
            prefill = self.prefill_with_route_gate(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                sample_id=sample_id,
                seed=seed,
                query_original_positions=query_original_positions,
                width=width,
                height=height,
                decide=decide,
            )
            next_token = self.torch.argmax(prefill.logits[:, -1, :], dim=-1, keepdim=True)
        self._cuda_sync()
        prefill_latency = time.perf_counter() - prefill_started
        generated = [next_token]
        finished = bool(eos_ids and int(next_token.item()) in eos_ids)
        state = prefill.state
        self._cuda_sync()
        decode_started = time.perf_counter()
        with self.torch.inference_mode():
            while len(generated) < int(max_new_tokens) and not finished:
                logits = self.decode_one_with_segment_caches(state, next_token)
                next_token = self.torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated.append(next_token)
                finished = bool(eos_ids and int(next_token.item()) in eos_ids)
        self._cuda_sync()
        decode_latency = time.perf_counter() - decode_started
        generated_ids = self.torch.cat(generated, dim=1)[0]
        final_state_diagnostics = self._diagnostics(state)
        # Preserve the true pre-decode storage snapshot.  Recomputing the
        # shared diagnostic block here is still required for final cache
        # lengths and generated-token counters, but its prompt-byte fields
        # inspect the now-expanded live cache.
        final_state_diagnostics["cache/prompt_segment_bytes"] = prefill.diagnostics[
            "cache/prompt_segment_bytes"
        ]
        final_state_diagnostics["cache/prompt_total_bytes"] = prefill.diagnostics[
            "cache/prompt_total_bytes"
        ]
        diagnostics = {
            **prefill.diagnostics,
            **final_state_diagnostics,
            "llm_prefill_latency_s": prefill_latency,
            "prefill_latency_s": prefill_latency,
            "time_to_first_token_s": prefill_latency,
            "decode_latency_s": decode_latency,
            "cache/final_segment_bytes": [cache_tensor_bytes(segment.cache) for segment in state.segments],
        }
        diagnostics["cache/final_total_bytes"] = sum(diagnostics["cache/final_segment_bytes"])
        return ProgressiveGenerationResult(
            generated_ids=generated_ids,
            final_retained_indices=list(state.final_retained_indices),
            stage_traces=[trace.as_dict() for trace in state.stage_traces],
            diagnostics=diagnostics,
        )
