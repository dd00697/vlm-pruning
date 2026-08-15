"""Generalized physical visual-token pruning for LLaVA decoder schedules.

The existing two-bank single-stage implementation remains unchanged.  This
module generalizes the same boundary convention to an arbitrary ordered set of
physical pruning events and one cache bank per decoder-layer segment.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from vlm_pruning.efficiency import cache_tensor_bytes
from vlm_pruning.llava_hooks.mid_layer_pruning import CacheLengthProxy, MidLayerUGPruneForwarder
from vlm_pruning.selectors import UGPruneConfig
from vlm_pruning.selectors.base import SelectorError


def _json_checksum(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tensor_checksum(tensor: Any) -> str:
    values = tensor.detach().to(device="cpu", dtype=tensor.float().dtype).contiguous().numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


@dataclass(frozen=True)
class PruningEvent:
    boundary: int
    budget: int
    selector: str = "query_topk"

    def as_dict(self) -> dict[str, Any]:
        return {"boundary": self.boundary, "budget": self.budget, "selector": self.selector}


def validate_schedule(
    events: Iterable[PruningEvent | dict[str, Any] | Sequence[Any]],
    *,
    num_layers: int = 32,
    initial_visual_tokens: int = 576,
) -> tuple[PruningEvent, ...]:
    parsed: list[PruningEvent] = []
    for raw in events:
        if isinstance(raw, PruningEvent):
            event = raw
        elif isinstance(raw, dict):
            event = PruningEvent(
                boundary=int(raw["boundary"]),
                budget=int(raw["budget"]),
                selector=str(raw.get("selector") or "query_topk"),
            )
        else:
            values = list(raw)
            if len(values) not in {2, 3}:
                raise SelectorError(f"Schedule tuple must have 2 or 3 values, got {values!r}")
            event = PruningEvent(
                boundary=int(values[0]),
                budget=int(values[1]),
                selector=str(values[2]) if len(values) == 3 else "query_topk",
            )
        parsed.append(event)
    previous_boundary = 0
    previous_budget = int(initial_visual_tokens)
    allowed = {"query_topk", "query_topk_merge", "random", "none"}
    merge_seen = False
    for event in parsed:
        if event.boundary <= previous_boundary or event.boundary >= int(num_layers):
            raise SelectorError(
                f"Boundaries must be strictly increasing in [1, {num_layers - 1}], got {event.boundary}."
            )
        if event.budget < 0 or event.budget > previous_budget:
            raise SelectorError(
                f"Budget {event.budget} is not in [0, {previous_budget}] at K{event.boundary}."
            )
        if event.selector not in allowed:
            raise SelectorError(f"Unsupported progressive selector: {event.selector}")
        if event.selector == "none" and event.budget != 0:
            raise SelectorError("selector=none is reserved for the exact B=0 event")
        if event.budget == 0 and event.selector != "none":
            raise SelectorError("B=0 must use selector=none")
        if event.selector == "query_topk_merge":
            if merge_seen or event.boundary != 4 or event.budget != 448 or previous_budget != 576:
                raise SelectorError(
                    "query_topk_merge is restricted to the registered first event K4->448 from 576 tokens"
                )
            merge_seen = True
        previous_boundary = event.boundary
        previous_budget = event.budget
    return tuple(parsed)


def realized_visual_trace(
    schedule: Sequence[PruningEvent],
    *,
    num_layers: int = 32,
    initial_visual_tokens: int = 576,
) -> list[int]:
    events = validate_schedule(schedule, num_layers=num_layers, initial_visual_tokens=initial_visual_tokens)
    trace: list[int] = []
    start = 0
    budget = int(initial_visual_tokens)
    for event in events:
        trace.extend([budget] * (event.boundary - start))
        start = event.boundary
        budget = event.budget
    trace.extend([budget] * (num_layers - start))
    if len(trace) != num_layers:
        raise AssertionError(f"Visual trace has {len(trace)} layers, expected {num_layers}")
    return trace


def average_visual_tokens(trace: Sequence[int]) -> float:
    if not trace:
        raise SelectorError("Visual-token trace cannot be empty")
    return sum(int(value) for value in trace) / len(trace)


def deterministic_stage_seed(
    *, sample_id: str, cell_seed: int, stage_index: int, boundary: int, incoming_indices: Sequence[int]
) -> int:
    payload = {
        "algorithm": "sha256-json-v1",
        "boundary": int(boundary),
        "cell_seed": int(cell_seed),
        "incoming_indices_checksum": _json_checksum([int(value) for value in incoming_indices]),
        "sample_id": str(sample_id),
        "stage_index": int(stage_index),
    }
    return int(hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16], 16)


@dataclass(frozen=True)
class StageTrace:
    stage_index: int
    boundary: int
    selector: str
    role: str
    requested_budget: int
    realized_budget: int
    incoming_original_indices: list[int]
    outgoing_original_indices: list[int]
    incoming_indices_checksum: str
    outgoing_indices_checksum: str
    score_map_checksum: str | None
    score_map_recomputed: bool
    derived_random_seed: int | None
    incoming_sequence_length: int
    outgoing_sequence_length: int
    text_token_checksum: str
    text_position_checksum: str
    outgoing_position_ids: list[int]
    position_id_checksum: str
    selection_latency_s: float
    merge_applied: bool = False
    merge_assignment_count: int = 0
    merge_assignment_checksum: str | None = None
    merge_output_masses: list[float] | None = None
    merge_mass_checksum: str | None = None
    merge_input_mass: float | None = None
    merge_output_mass: float | None = None
    merge_latency_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class SegmentCache:
    segment_index: int
    start_layer: int
    end_layer: int
    cache: Any
    attention_mask: Any
    prompt_length: int
    visual_tokens: int


@dataclass
class ProgressiveCacheState:
    segments: list[SegmentCache]
    stage_traces: list[StageTrace]
    final_retained_indices: list[int]
    full_prompt_length: int
    final_prompt_length: int
    original_text_token_checksum: str
    original_text_position_checksum: str
    original_visual_positions: list[int]
    original_text_positions: list[int]
    realized_visual_tokens_by_layer: list[int]
    generated_tokens_cached: int = 0

    @property
    def candidate_average_visual_tokens(self) -> float:
        return average_visual_tokens(self.realized_visual_tokens_by_layer)


@dataclass(frozen=True)
class ProgressivePrefillResult:
    logits: Any
    state: ProgressiveCacheState
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ProgressiveGenerationResult:
    generated_ids: Any
    final_retained_indices: list[int]
    stage_traces: list[dict[str, Any]]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class OracleIndexArtifact:
    sample_id: str
    teacher_layer: int
    original_visual_tokens: int
    ranked_original_indices: tuple[int, ...]
    ranking_checksum: str
    query_token_status: str
    teacher_visual_token_layer_work: int
    format_version: int = 1

    def validate(self) -> "OracleIndexArtifact":
        if self.format_version != 1:
            raise SelectorError(f"Unsupported oracle index format {self.format_version}")
        if self.teacher_layer not in {12, 20}:
            raise SelectorError(f"Oracle teacher layer must be 12 or 20, got {self.teacher_layer}")
        if self.original_visual_tokens <= 0:
            raise SelectorError("Oracle original token count must be positive")
        ranking = list(self.ranked_original_indices)
        if any(type(value) is not int for value in ranking):
            raise SelectorError("Oracle ranking must contain integer indices only")
        if len(ranking) != self.original_visual_tokens or sorted(ranking) != list(range(self.original_visual_tokens)):
            raise SelectorError("Oracle ranking must be a complete permutation of original patch indices")
        if self.ranking_checksum != _json_checksum(ranking):
            raise SelectorError("Oracle ranking checksum mismatch")
        expected_work = self.original_visual_tokens * self.teacher_layer
        if self.teacher_visual_token_layer_work != expected_work:
            raise SelectorError(
                f"Oracle teacher work {self.teacher_visual_token_layer_work} != expected {expected_work}"
            )
        return self

    def selected_indices(self, budget: int) -> list[int]:
        self.validate()
        if budget <= 0 or budget > self.original_visual_tokens:
            raise SelectorError(f"Oracle budget must be in [1, {self.original_visual_tokens}], got {budget}")
        return sorted(self.ranked_original_indices[: int(budget)])

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "format_version": self.format_version,
            "sample_id": self.sample_id,
            "teacher_layer": self.teacher_layer,
            "original_visual_tokens": self.original_visual_tokens,
            "ranked_original_indices": list(self.ranked_original_indices),
            "ranking_checksum": self.ranking_checksum,
            "query_token_status": self.query_token_status,
            "oracle_teacher_visual_token_layer_work": self.teacher_visual_token_layer_work,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OracleIndexArtifact":
        allowed = {
            "format_version", "sample_id", "teacher_layer", "original_visual_tokens",
            "ranked_original_indices", "ranking_checksum", "query_token_status",
            "oracle_teacher_visual_token_layer_work",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise SelectorError(f"Oracle index artifact contains unregistered fields: {sorted(unknown)}")
        artifact = cls(
            format_version=int(payload["format_version"]),
            sample_id=str(payload["sample_id"]),
            teacher_layer=int(payload["teacher_layer"]),
            original_visual_tokens=int(payload["original_visual_tokens"]),
            ranked_original_indices=tuple(payload["ranked_original_indices"]),
            ranking_checksum=str(payload["ranking_checksum"]),
            query_token_status=str(payload["query_token_status"]),
            teacher_visual_token_layer_work=int(payload["oracle_teacher_visual_token_layer_work"]),
        )
        return artifact.validate()


class ProgressivePruneForwarder(MidLayerUGPruneForwarder):
    """Run ordered pruning events with one dynamic-cache bank per segment."""

    def __init__(
        self,
        torch: Any,
        model: Any,
        *,
        image_token_id: int,
        schedule: Iterable[PruningEvent | dict[str, Any] | Sequence[Any]],
        grid: int = 24,
        initial_visual_tokens: int = 576,
        diagnostics_mode: str = "benchmark",
        synchronise_subtimings: bool = False,
    ) -> None:
        # Base helpers provide the exact historical layer execution, state
        # normalization and cache construction.  The placeholder boundary is
        # not used for progressive execution.
        super().__init__(
            torch,
            model,
            image_token_id=image_token_id,
            grid=grid,
            prune_after_layer=1,
            target_visual_tokens=max(1, int(initial_visual_tokens)),
            selector_config=UGPruneConfig(
                selector_strategy="query_topk",
                probe_layers=(1,),
                fusion_mode="single_layer",
                position_mode="gap_preserving",
            ),
            diagnostics_mode=diagnostics_mode,
            synchronise_subtimings=synchronise_subtimings,
        )
        self.initial_visual_tokens = int(initial_visual_tokens)
        self.schedule = validate_schedule(
            schedule,
            num_layers=self.num_layers,
            initial_visual_tokens=self.initial_visual_tokens,
        )

    @staticmethod
    def _validate_original_indices(indices: Sequence[int], *, incoming: Sequence[int], budget: int) -> list[int]:
        selected = [int(value) for value in indices]
        if len(selected) != int(budget):
            raise SelectorError(f"Selected {len(selected)} visual tokens, expected {budget}")
        if selected != sorted(selected):
            raise SelectorError("Selected original indices must remain in raster order")
        if len(set(selected)) != len(selected):
            raise SelectorError("Selected original indices contain duplicates")
        incoming_set = set(int(value) for value in incoming)
        if any(value not in incoming_set for value in selected):
            raise SelectorError("A pruning stage selected an index removed by an earlier stage")
        return selected

    def _text_evidence(self, input_ids: Any, original_positions: Any) -> tuple[str, str]:
        ids = input_ids[0] if input_ids.ndim == 2 else input_ids
        text_mask = ids != self.image_token_id
        text_ids = [int(value) for value in ids[text_mask].detach().cpu().tolist()]
        text_positions = [int(value) for value in original_positions[text_mask].detach().cpu().tolist()]
        return _json_checksum(text_ids), _json_checksum(text_positions)

    def _current_query_positions(self, original_positions: Any, query_original_positions: Sequence[int]) -> Any:
        mapping = {
            int(original): current
            for current, original in enumerate(original_positions.detach().cpu().tolist())
        }
        missing = [int(value) for value in query_original_positions if int(value) not in mapping]
        if missing:
            raise SelectorError(f"Query text positions were removed: {missing}")
        return self.torch.tensor(
            [mapping[int(value)] for value in query_original_positions],
            dtype=self.torch.long,
            device=original_positions.device,
        )

    def score_current_visual_tokens(
        self,
        hidden_states: Any,
        *,
        input_ids: Any,
        original_positions: Any,
        query_original_positions: Sequence[int],
    ) -> tuple[Any, Any]:
        ids = input_ids[0] if input_ids.ndim == 2 else input_ids
        image_positions = self.torch.nonzero(ids == self.image_token_id, as_tuple=False).flatten()
        if int(image_positions.numel()) == 0:
            return self.torch.empty(0, dtype=self.torch.float, device=hidden_states.device), image_positions
        query_positions = self._current_query_positions(original_positions, query_original_positions)
        states = hidden_states[0] if hidden_states.ndim == 3 else hidden_states
        image_states = states.index_select(0, image_positions)
        query_states = states.index_select(0, query_positions)
        image_norm = self._normalise_states(image_states)
        query_norm = self._normalise_states(query_states)
        scores = (image_norm @ query_norm.transpose(0, 1)).max(dim=1).values
        return self._zscore(scores).reshape(-1), image_positions

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
        incoming = [int(value) for value in incoming_original_indices]
        if len(incoming) < event.budget:
            raise SelectorError(f"K{event.boundary} budget exceeds incoming tokens")
        score_checksum: str | None = None
        recomputed = False
        derived_seed: int | None = None
        if event.selector in {"query_topk", "query_topk_merge", "none"}:
            scores, _ = self.score_current_visual_tokens(
                hidden_states,
                input_ids=input_ids,
                original_positions=original_positions,
                query_original_positions=query_original_positions,
            )
            score_checksum = _tensor_checksum(scores)
            recomputed = True
            if event.selector == "none":
                selected: list[int] = []
            else:
                ranked_local = self.torch.argsort(scores, descending=True, stable=True)
                selected = sorted(incoming[int(value)] for value in ranked_local[: event.budget].detach().cpu().tolist())
        else:
            derived_seed = deterministic_stage_seed(
                sample_id=sample_id,
                cell_seed=seed,
                stage_index=stage_index,
                boundary=event.boundary,
                incoming_indices=incoming,
            )
            selected = sorted(random.Random(derived_seed).sample(incoming, event.budget))
        return self._validate_original_indices(selected, incoming=incoming, budget=event.budget), score_checksum, recomputed, derived_seed

    def _gather_after_selection(
        self,
        hidden_states: Any,
        *,
        input_ids: Any,
        attention_mask: Any,
        original_positions: Any,
        incoming_original_indices: Sequence[int],
        selected_original_indices: Sequence[int],
    ) -> tuple[Any, Any, Any, Any]:
        ids = input_ids[0] if input_ids.ndim == 2 else input_ids
        image_positions = self.torch.nonzero(ids == self.image_token_id, as_tuple=False).flatten()
        if int(image_positions.numel()) != len(incoming_original_indices):
            raise SelectorError("Current image-position count does not match surviving original indices")
        selected_set = set(int(value) for value in selected_original_indices)
        keep_mask = self.torch.ones(ids.shape[0], dtype=self.torch.bool, device=ids.device)
        for local, original in enumerate(incoming_original_indices):
            if int(original) not in selected_set:
                keep_mask[image_positions[local]] = False
        keep_positions = self.torch.nonzero(keep_mask, as_tuple=False).flatten()
        return (
            hidden_states.index_select(1, keep_positions),
            input_ids.index_select(1, keep_positions),
            attention_mask.index_select(1, keep_positions),
            original_positions.index_select(0, keep_positions),
        )

    def _selected_visual_masses(
        self,
        visual_masses: Any,
        *,
        incoming_original_indices: Sequence[int],
        selected_original_indices: Sequence[int],
    ) -> Any:
        local_by_original = {
            int(original): local for local, original in enumerate(incoming_original_indices)
        }
        local_indices = self.torch.tensor(
            [local_by_original[int(original)] for original in selected_original_indices],
            dtype=self.torch.long,
            device=visual_masses.device,
        )
        return visual_masses.index_select(0, local_indices)

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
        """Merge every dropped visual state into its nearest retained raster patch."""
        ids = input_ids[0] if input_ids.ndim == 2 else input_ids
        image_positions = self.torch.nonzero(ids == self.image_token_id, as_tuple=False).flatten()
        incoming = [int(value) for value in incoming_original_indices]
        selected = [int(value) for value in selected_original_indices]
        if int(image_positions.numel()) != len(incoming):
            raise SelectorError("Merge image-position count does not match surviving original indices")
        if int(visual_masses.numel()) != len(incoming):
            raise SelectorError("Merge mass count does not match surviving original indices")
        if not selected:
            raise SelectorError("Mass-preserving merge requires at least one retained visual token")

        selected_slot = {original: slot for slot, original in enumerate(selected)}
        selected_tensor = self.torch.tensor(selected, dtype=self.torch.long, device=hidden_states.device)
        incoming_tensor = self.torch.tensor(incoming, dtype=self.torch.long, device=hidden_states.device)
        selected_coords = self.torch.stack(
            (selected_tensor // int(self.grid), selected_tensor % int(self.grid)), dim=1
        ).to(dtype=self.torch.float32)
        incoming_coords = self.torch.stack(
            (incoming_tensor // int(self.grid), incoming_tensor % int(self.grid)), dim=1
        ).to(dtype=self.torch.float32)
        distances = ((incoming_coords[:, None, :] - selected_coords[None, :, :]) ** 2).sum(dim=2)
        assignment_slots = self.torch.argmin(distances, dim=1)
        for local, original in enumerate(incoming):
            if original in selected_slot:
                assignment_slots[local] = int(selected_slot[original])

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
            raise SelectorError("Every retained merged visual token must have positive mass")
        merged_states = weighted_sums / output_masses[:, None]

        gathered_hidden, gathered_ids, gathered_mask, gathered_positions = self._gather_after_selection(
            hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            original_positions=original_positions,
            incoming_original_indices=incoming,
            selected_original_indices=selected,
        )
        gathered_image_positions = self.torch.nonzero(
            gathered_ids[0] == self.image_token_id, as_tuple=False
        ).flatten()
        if int(gathered_image_positions.numel()) != len(selected):
            raise SelectorError("Merged gather did not realize the selected budget")
        gathered_hidden = gathered_hidden.clone()
        gathered_hidden[0, gathered_image_positions] = merged_states.to(dtype=gathered_hidden.dtype)

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

    def _new_segment(
        self,
        *,
        segment_index: int,
        start_layer: int,
        end_layer: int,
        attention_mask: Any,
        prompt_length: int,
        visual_tokens: int,
    ) -> SegmentCache:
        return SegmentCache(
            segment_index=segment_index,
            start_layer=start_layer,
            end_layer=end_layer,
            cache=self._new_cache(),
            attention_mask=attention_mask.clone(),
            prompt_length=int(prompt_length),
            visual_tokens=int(visual_tokens),
        )

    def _cache_for_segment(self, segment: SegmentCache) -> Any:
        if segment.start_layer == 0:
            return segment.cache
        return CacheLengthProxy(segment.cache, default_layer_idx=segment.start_layer)

    def assert_segment_cache_lengths(self, state: ProgressiveCacheState) -> None:
        for segment in state.segments:
            expected = segment.prompt_length + state.generated_tokens_cached
            if int(segment.attention_mask.shape[1]) != expected:
                raise SelectorError(
                    f"Segment {segment.segment_index} mask length {segment.attention_mask.shape[1]} != {expected}"
                )
            for layer in range(segment.start_layer, segment.end_layer):
                actual = self.cache_seq_len(segment.cache, layer)
                if actual != expected:
                    raise SelectorError(
                        f"Layer {layer} cache length {actual} != segment prompt {segment.prompt_length} "
                        f"+ generated {state.generated_tokens_cached}"
                    )

    def oracle_teacher_indices(
        self,
        *,
        inputs_embeds: Any,
        input_ids: Any,
        attention_mask: Any,
        sample_id: str,
        teacher_layer: int,
        query_token_status: str,
        query_original_positions: Sequence[int],
    ) -> OracleIndexArtifact:
        if teacher_layer not in {12, 20}:
            raise SelectorError("Registered oracle teachers are K12 and K20 only")
        ids = input_ids[0] if input_ids.ndim == 2 else input_ids
        image_positions = self.torch.nonzero(ids == self.image_token_id, as_tuple=False).flatten()
        original_tokens = int(image_positions.numel())
        if original_tokens != self.initial_visual_tokens:
            raise SelectorError(
                f"Dense oracle teacher requires {self.initial_visual_tokens} patches, got {original_tokens}"
            )
        original_positions = self.torch.arange(ids.shape[0], device=ids.device, dtype=self.torch.long)
        position_ids = original_positions.unsqueeze(0)
        hidden, _ = self.run_decoder_layers(
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            start_layer=0,
            end_layer=teacher_layer,
            collect_hidden_states=False,
            use_cache=False,
        )
        scores, _ = self.score_current_visual_tokens(
            hidden,
            input_ids=input_ids,
            original_positions=original_positions,
            query_original_positions=query_original_positions,
        )
        ranking = tuple(int(value) for value in self.torch.argsort(scores, descending=True, stable=True).detach().cpu().tolist())
        artifact = OracleIndexArtifact(
            sample_id=str(sample_id),
            teacher_layer=int(teacher_layer),
            original_visual_tokens=original_tokens,
            ranked_original_indices=ranking,
            ranking_checksum=_json_checksum(list(ranking)),
            query_token_status=str(query_token_status),
            teacher_visual_token_layer_work=original_tokens * int(teacher_layer),
        ).validate()
        # Only the immutable integer artifact escapes this method.
        del hidden, scores
        return artifact

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
        if int(inputs_embeds.shape[0]) != 1 or int(input_ids.shape[0]) != 1:
            raise SelectorError("Progressive pruning supports batch size 1 only")
        ids = input_ids[0]
        original_image_positions = self.torch.nonzero(ids == self.image_token_id, as_tuple=False).flatten()
        if int(original_image_positions.numel()) != self.initial_visual_tokens:
            raise SelectorError(
                f"Expected {self.initial_visual_tokens} original image tokens, found {original_image_positions.numel()}"
            )
        full_prompt_length = int(input_ids.shape[1])
        original_positions = self.torch.arange(full_prompt_length, device=input_ids.device, dtype=self.torch.long)
        original_text_checksum, original_text_position_checksum = self._text_evidence(input_ids, original_positions)
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

        if initial_retained_indices is not None:
            selected = self._validate_original_indices(
                initial_retained_indices,
                incoming=surviving,
                budget=len(initial_retained_indices),
            )
            before_len = int(current_ids.shape[1])
            hidden, current_ids, current_mask, current_positions = self._gather_after_selection(
                hidden,
                input_ids=current_ids,
                attention_mask=current_mask,
                original_positions=current_positions,
                incoming_original_indices=surviving,
                selected_original_indices=selected,
            )
            visual_masses = self._selected_visual_masses(
                visual_masses,
                incoming_original_indices=surviving,
                selected_original_indices=selected,
            )
            text_checksum, position_checksum = self._text_evidence(current_ids, current_positions)
            if text_checksum != original_text_checksum or position_checksum != original_text_position_checksum:
                raise SelectorError("Pre-LLM selection altered text tokens or original text positions")
            traces.append(
                StageTrace(
                    stage_index=0,
                    boundary=0,
                    selector=str(initial_selector or "precomputed"),
                    role="pre_llm_selection",
                    requested_budget=len(selected),
                    realized_budget=len(selected),
                    incoming_original_indices=surviving,
                    outgoing_original_indices=selected,
                    incoming_indices_checksum=_json_checksum(surviving),
                    outgoing_indices_checksum=_json_checksum(selected),
                    score_map_checksum=initial_selection_checksum,
                    score_map_recomputed=False,
                    derived_random_seed=initial_derived_random_seed,
                    incoming_sequence_length=before_len,
                    outgoing_sequence_length=int(current_ids.shape[1]),
                    text_token_checksum=text_checksum,
                    text_position_checksum=position_checksum,
                    outgoing_position_ids=[int(v) for v in current_positions.detach().cpu().tolist()],
                    position_id_checksum=_json_checksum([int(v) for v in current_positions.detach().cpu().tolist()]),
                    selection_latency_s=0.0,
                )
            )
            surviving = selected

        segments: list[SegmentCache] = []
        start_layer = 0
        runtime_stage_offset = len(traces)
        for event_index, event in enumerate(self.schedule):
            segment = self._new_segment(
                segment_index=len(segments),
                start_layer=start_layer,
                end_layer=event.boundary,
                attention_mask=current_mask,
                prompt_length=int(current_ids.shape[1]),
                visual_tokens=len(surviving),
            )
            position_ids = current_positions.unsqueeze(0)
            hidden, _ = self.run_decoder_layers(
                hidden,
                attention_mask=current_mask,
                position_ids=position_ids,
                start_layer=start_layer,
                end_layer=event.boundary,
                collect_hidden_states=False,
                past_key_values=self._cache_for_segment(segment),
                use_cache=True,
            )
            segments.append(segment)
            selection_start = time.perf_counter()
            selected, score_checksum, recomputed, derived_seed = self.select_stage_indices(
                hidden,
                input_ids=current_ids,
                original_positions=current_positions,
                incoming_original_indices=surviving,
                event=event,
                stage_index=runtime_stage_offset + event_index,
                sample_id=sample_id,
                seed=seed,
                query_original_positions=query_original_positions,
            )
            selection_latency = time.perf_counter() - selection_start
            before_len = int(current_ids.shape[1])
            incoming = list(surviving)
            merge_diagnostics: dict[str, Any] = {}
            merge_latency = 0.0
            if event.selector == "query_topk_merge":
                merge_start = time.perf_counter()
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
                merge_latency = time.perf_counter() - merge_start
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
                raise SelectorError(f"K{event.boundary} altered text tokens or text positions")
            traces.append(
                StageTrace(
                    stage_index=runtime_stage_offset + event_index,
                    boundary=event.boundary,
                    selector=event.selector,
                    role="runtime_pruning",
                    requested_budget=event.budget,
                    realized_budget=len(selected),
                    incoming_original_indices=incoming,
                    outgoing_original_indices=selected,
                    incoming_indices_checksum=_json_checksum(incoming),
                    outgoing_indices_checksum=_json_checksum(selected),
                    score_map_checksum=score_checksum,
                    score_map_recomputed=recomputed,
                    derived_random_seed=derived_seed,
                    incoming_sequence_length=before_len,
                    outgoing_sequence_length=int(current_ids.shape[1]),
                    text_token_checksum=text_checksum,
                    text_position_checksum=text_position_checksum,
                    outgoing_position_ids=[int(v) for v in current_positions.detach().cpu().tolist()],
                    position_id_checksum=_json_checksum([int(v) for v in current_positions.detach().cpu().tolist()]),
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
            realized_trace.extend([segment.visual_tokens] * (segment.end_layer - segment.start_layer))
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
        diagnostics = self._diagnostics(state)
        return ProgressivePrefillResult(logits=logits, state=state, diagnostics=diagnostics)

    def _diagnostics(self, state: ProgressiveCacheState) -> dict[str, Any]:
        prompt_bytes = [cache_tensor_bytes(segment.cache) for segment in state.segments]
        return {
            "progressive/cache_implementation": "segment_aware_dynamic_cache",
            "progressive/position_mode": "gap_preserving",
            "progressive/stage_traces": [trace.as_dict() for trace in state.stage_traces],
            "progressive/query_selection_latency_s": sum(
                float(trace.selection_latency_s) for trace in state.stage_traces
            ),
            "progressive/merge_latency_s": sum(
                float(trace.merge_latency_s) for trace in state.stage_traces
            ),
            "progressive/final_retained_indices": list(state.final_retained_indices),
            "progressive/final_retained_indices_checksum": _json_checksum(state.final_retained_indices),
            "progressive/realized_visual_tokens_by_layer": list(state.realized_visual_tokens_by_layer),
            "candidate_average_visual_tokens": state.candidate_average_visual_tokens,
            "progressive/full_prompt_length": state.full_prompt_length,
            "progressive/final_prompt_length": state.final_prompt_length,
            "progressive/cache_generated_tokens": state.generated_tokens_cached,
            "progressive/segment_prompt_lengths": [segment.prompt_length for segment in state.segments],
            "progressive/segment_cache_lengths": [
                self.cache_seq_len(segment.cache, segment.start_layer) for segment in state.segments
            ],
            "progressive/per_layer_cache_lengths": {
                str(layer): self.cache_seq_len(segment.cache, layer)
                for segment in state.segments
                for layer in range(segment.start_layer, segment.end_layer)
            },
            "progressive/segment_attention_mask_lengths": [int(segment.attention_mask.shape[1]) for segment in state.segments],
            "progressive/original_text_token_checksum": state.original_text_token_checksum,
            "progressive/original_text_position_checksum": state.original_text_position_checksum,
            "progressive/original_visual_positions": list(state.original_visual_positions),
            "progressive/original_text_positions": list(state.original_text_positions),
            "cache/prompt_segment_bytes": prompt_bytes,
            "cache/prompt_total_bytes": sum(prompt_bytes),
        }

    def decode_one_with_segment_caches(self, state: ProgressiveCacheState, token: Any) -> Any:
        hidden = self.model.get_input_embeddings()(token).to(
            device=state.segments[0].attention_mask.device,
            dtype=next(self.model.parameters()).dtype,
        )
        position_id = self.torch.full_like(token, state.full_prompt_length + state.generated_tokens_cached)
        for segment in state.segments:
            segment.attention_mask = self.torch.cat(
                [
                    segment.attention_mask,
                    self.torch.ones_like(
                        token,
                        device=segment.attention_mask.device,
                        dtype=segment.attention_mask.dtype,
                    ),
                ],
                dim=1,
            )
            hidden, _ = self.run_decoder_layers(
                hidden,
                attention_mask=segment.attention_mask,
                position_ids=position_id,
                start_layer=segment.start_layer,
                end_layer=segment.end_layer,
                collect_hidden_states=False,
                past_key_values=self._cache_for_segment(segment),
                use_cache=True,
            )
        hidden = self.language_model.norm(hidden)
        logits = self.model.lm_head(hidden[:, -1:, :])
        state.generated_tokens_cached += 1
        self.assert_segment_cache_lengths(state)
        return logits

    def generate_with_segment_caches(
        self,
        *,
        inputs_embeds: Any,
        input_ids: Any,
        attention_mask: Any,
        sample_id: str,
        seed: int,
        query_original_positions: Sequence[int],
        max_new_tokens: int,
        eos_ids: set[int],
        initial_retained_indices: Sequence[int] | None = None,
        initial_selector: str | None = None,
        initial_selection_checksum: str | None = None,
        initial_derived_random_seed: int | None = None,
    ) -> ProgressiveGenerationResult:
        self._cuda_sync()
        prefill_start = time.perf_counter()
        with self.torch.inference_mode():
            prefill = self.prefill_with_segment_caches(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                sample_id=sample_id,
                seed=seed,
                query_original_positions=query_original_positions,
                initial_retained_indices=initial_retained_indices,
                initial_selector=initial_selector,
                initial_selection_checksum=initial_selection_checksum,
                initial_derived_random_seed=initial_derived_random_seed,
            )
            next_token = self.torch.argmax(prefill.logits[:, -1, :], dim=-1, keepdim=True)
        self._cuda_sync()
        prefill_latency = time.perf_counter() - prefill_start
        generated = [next_token]
        finished = bool(eos_ids and int(next_token.item()) in eos_ids)
        state = prefill.state
        self._cuda_sync()
        decode_start = time.perf_counter()
        with self.torch.inference_mode():
            while len(generated) < int(max_new_tokens) and not finished:
                logits = self.decode_one_with_segment_caches(state, next_token)
                next_token = self.torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated.append(next_token)
                finished = bool(eos_ids and int(next_token.item()) in eos_ids)
        self._cuda_sync()
        decode_latency = time.perf_counter() - decode_start
        generated_ids = self.torch.cat(generated, dim=1)[0]
        final_state_diagnostics = self._diagnostics(state)
        # ``_diagnostics`` inspects the live cache.  At this point that cache
        # includes every decoded token, so its prompt-byte fields are final
        # storage rather than the prefill snapshot.  Refresh mutable cache
        # lengths/counters while preserving the bytes captured by prefill.
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


def uniform_original_indices(budget: int, *, grid: int = 24, original_tokens: int = 576) -> list[int]:
    """Use the repository's registered uniform control over original patches."""
    from scripts.run_baseline import retained_indices

    return retained_indices("uniform", int(budget), int(original_tokens), 0, "uniform-control", int(grid))


def random_original_indices(
    budget: int, *, sample_id: str, seed: int, original_tokens: int = 576
) -> tuple[list[int], int]:
    incoming = list(range(int(original_tokens)))
    derived = deterministic_stage_seed(
        sample_id=sample_id,
        cell_seed=seed,
        stage_index=0,
        boundary=0,
        incoming_indices=incoming,
    )
    return sorted(random.Random(derived).sample(incoming, int(budget))), derived
