"""Opt-in synchronized runtime profiling and exact progressive-stage replay."""

from __future__ import annotations

import contextlib
import hashlib
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from vlm_pruning.llava_hooks.mid_layer_pruning import MidLayerUGPruneForwarder
from vlm_pruning.llava_hooks.progressive_pruning import (
    PruningEvent,
    ProgressivePrefillResult,
    ProgressivePruneForwarder,
    StageTrace,
    _json_checksum,
)
from vlm_pruning.llava_hooks.runtime_repairs import (
    FusedGatherMergeProgressiveForwarder,
    FusedNativeProgressiveForwarder,
    NativeCacheProgressiveForwarder,
)
from vlm_pruning.selectors.base import SelectorError


@dataclass(frozen=True)
class ProfileRecord:
    component: str
    cpu_wall_s: float
    cuda_event_s: float | None
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "cpu_wall_s": self.cpu_wall_s,
            "cuda_event_s": self.cuda_event_s,
            "metadata": self.metadata,
        }


class RuntimeProfiler:
    """Record synchronized CPU-wall and CUDA-event intervals.

    Synchronization is intentional: this profiler is diagnostic and is never
    used for the final deployable latency claim.
    """

    def __init__(self, torch: Any, *, enabled: bool = True) -> None:
        self.torch = torch
        self.enabled = bool(enabled)
        self.records: list[ProfileRecord] = []

    def reset(self) -> None:
        self.records = []

    @contextlib.contextmanager
    def measure(self, component: str, **metadata: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        cuda = bool(self.torch.cuda.is_available())
        start_event = end_event = None
        if cuda:
            start_event = self.torch.cuda.Event(enable_timing=True)
            end_event = self.torch.cuda.Event(enable_timing=True)
            start_event.record()
        cpu_started = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                assert start_event is not None and end_event is not None
                end_event.record()
                end_event.synchronize()
                cuda_event_s: float | None = float(start_event.elapsed_time(end_event)) / 1000.0
            else:
                cuda_event_s = None
            cpu_wall_s = time.perf_counter() - cpu_started
            self.records.append(
                ProfileRecord(
                    component=str(component),
                    cpu_wall_s=float(cpu_wall_s),
                    cuda_event_s=cuda_event_s,
                    metadata={str(key): value for key, value in metadata.items()},
                )
            )

    def snapshot(self) -> dict[str, Any]:
        groups: dict[str, list[ProfileRecord]] = {}
        for record in self.records:
            groups.setdefault(record.component, []).append(record)
        components: dict[str, Any] = {}
        for component, records in sorted(groups.items()):
            cpu = [record.cpu_wall_s for record in records]
            cuda = [record.cuda_event_s for record in records if record.cuda_event_s is not None]
            components[component] = {
                "count": len(records),
                "cpu_wall_total_s": float(sum(cpu)),
                "cpu_wall_mean_s": float(statistics.fmean(cpu)),
                "cuda_event_total_s": float(sum(cuda)) if cuda else None,
                "cuda_event_mean_s": float(statistics.fmean(cuda)) if cuda else None,
            }
        return {
            "enabled": self.enabled,
            "synchronization_policy": "each measured interval synchronizes its terminal CUDA event",
            "records": [record.as_dict() for record in self.records],
            "components": components,
        }


@dataclass(frozen=True)
class StageReplay:
    boundary: int
    selector: str
    requested_budget: int
    incoming_original_indices: tuple[int, ...]
    outgoing_original_indices: tuple[int, ...]
    incoming_indices_checksum: str
    outgoing_indices_checksum: str
    score_map_checksum: str | None
    merge_assignment_checksum: str | None
    merge_mass_checksum: str | None
    merge_output_masses: tuple[float, ...] | None

    @classmethod
    def from_trace(cls, trace: Mapping[str, Any]) -> "StageReplay":
        incoming = tuple(int(value) for value in trace["incoming_original_indices"])
        outgoing = tuple(int(value) for value in trace["outgoing_original_indices"])
        artifact = cls(
            boundary=int(trace["boundary"]),
            selector=str(trace["selector"]),
            requested_budget=int(trace["requested_budget"]),
            incoming_original_indices=incoming,
            outgoing_original_indices=outgoing,
            incoming_indices_checksum=str(trace["incoming_indices_checksum"]),
            outgoing_indices_checksum=str(trace["outgoing_indices_checksum"]),
            score_map_checksum=trace.get("score_map_checksum"),
            merge_assignment_checksum=trace.get("merge_assignment_checksum"),
            merge_mass_checksum=trace.get("merge_mass_checksum"),
            merge_output_masses=(
                tuple(float(value) for value in trace["merge_output_masses"])
                if trace.get("merge_output_masses") is not None
                else None
            ),
        )
        artifact.validate()
        return artifact

    def validate(self) -> "StageReplay":
        incoming = list(self.incoming_original_indices)
        outgoing = list(self.outgoing_original_indices)
        if self.incoming_indices_checksum != _json_checksum(incoming):
            raise SelectorError("Replay incoming-index checksum mismatch")
        if self.outgoing_indices_checksum != _json_checksum(outgoing):
            raise SelectorError("Replay outgoing-index checksum mismatch")
        if outgoing != sorted(outgoing) or len(outgoing) != len(set(outgoing)):
            raise SelectorError("Replay outgoing indices are not unique raster order")
        if len(outgoing) != self.requested_budget or not set(outgoing).issubset(incoming):
            raise SelectorError("Replay budget or subset relation is invalid")
        return self

    def validate_request(self, event: PruningEvent, incoming: Sequence[int]) -> list[int]:
        self.validate()
        if self.boundary != event.boundary or self.selector != event.selector:
            raise SelectorError("Replay stage does not match the requested pruning event")
        if self.requested_budget != event.budget:
            raise SelectorError("Replay stage budget does not match the requested pruning event")
        if list(self.incoming_original_indices) != [int(value) for value in incoming]:
            raise SelectorError("Replay incoming indices differ from live progressive state")
        return list(self.outgoing_original_indices)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stage_replay(
    path: Path,
    *,
    expected_sha256: str,
    sample_ids: Iterable[str] | None = None,
) -> dict[str, tuple[StageReplay, ...]]:
    if sha256_file(path) != expected_sha256:
        raise SelectorError("Replay source prediction SHA-256 mismatch")
    wanted = set(str(value) for value in sample_ids) if sample_ids is not None else None
    output: dict[str, tuple[StageReplay, ...]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise SelectorError(f"Blank replay source row at line {line_number}")
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or "")
            if wanted is not None and sample_id not in wanted:
                continue
            if not sample_id or sample_id in output:
                raise SelectorError("Replay source has a missing or duplicate sample ID")
            traces = row.get("stage_traces")
            if not isinstance(traces, list) or not traces:
                raise SelectorError(f"Replay source lacks stage traces for {sample_id}")
            output[sample_id] = tuple(StageReplay.from_trace(trace) for trace in traces)
    if wanted is not None and set(output) != wanted:
        raise SelectorError(f"Replay source misses {len(wanted - set(output))} requested samples")
    return output


class HardwareProfileMixin:
    """Attach the profiler to shared decoder-layer calls without changing math."""

    runtime_profiler: RuntimeProfiler

    def __init__(self, *args: Any, runtime_profiler: RuntimeProfiler | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.runtime_profiler = runtime_profiler or RuntimeProfiler(self.torch, enabled=False)

    def _profile_scope(self, component: str, **metadata: Any) -> Any:
        return self.runtime_profiler.measure(component, **metadata)

    def reset_runtime_profile(self) -> None:
        self.runtime_profiler.reset()

    def runtime_profile_snapshot(self) -> dict[str, Any]:
        return self.runtime_profiler.snapshot()

    def run_decoder_layers(self, hidden_states: Any, **kwargs: Any) -> Any:
        start_layer = int(kwargs["start_layer"])
        end_layer = int(kwargs["end_layer"])
        sequence_length = int(hidden_states.shape[1])
        phase = "prefill" if sequence_length > 1 else "decode"
        with self._profile_scope(
            "decoder_segment",
            phase=phase,
            start_layer=start_layer,
            end_layer=end_layer,
            sequence_length=sequence_length,
        ):
            return super().run_decoder_layers(hidden_states, **kwargs)


class ProfiledMidLayerForwarder(HardwareProfileMixin, MidLayerUGPruneForwarder):
    """Exact original K12 forwarder with opt-in component profiling."""


class ProfiledProgressiveForwarder(HardwareProfileMixin, ProgressivePruneForwarder):
    """Progressive forwarder supporting exact accepted-stage replay."""

    def __init__(
        self,
        *args: Any,
        stage_replay_by_sample: Mapping[str, Sequence[StageReplay]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.stage_replay_by_sample = {
            str(sample_id): tuple(stages)
            for sample_id, stages in (stage_replay_by_sample or {}).items()
        }

    def _replay_stage(self, sample_id: str, boundary: int) -> StageReplay:
        stages = self.stage_replay_by_sample.get(str(sample_id))
        if stages is None:
            raise SelectorError(f"No replay stages registered for {sample_id}")
        matches = [stage for stage in stages if stage.boundary == int(boundary)]
        if len(matches) != 1:
            raise SelectorError(f"Replay requires exactly one K{boundary} stage for {sample_id}")
        return matches[0]

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
        if not self.stage_replay_by_sample:
            return super().select_stage_indices(
                hidden_states,
                input_ids=input_ids,
                original_positions=original_positions,
                incoming_original_indices=incoming_original_indices,
                event=event,
                stage_index=stage_index,
                sample_id=sample_id,
                seed=seed,
                query_original_positions=query_original_positions,
            )
        with self._profile_scope("replay_index_lookup", boundary=event.boundary):
            stage = self._replay_stage(sample_id, event.boundary)
            selected = stage.validate_request(event, incoming_original_indices)
            selected = self._validate_original_indices(
                selected,
                incoming=incoming_original_indices,
                budget=event.budget,
            )
        return selected, stage.score_map_checksum, False, None

    def prefill_with_segment_caches(self, **kwargs: Any) -> ProgressivePrefillResult:
        result = super().prefill_with_segment_caches(**kwargs)
        if not self.stage_replay_by_sample:
            return result
        sample_id = str(kwargs["sample_id"])
        expected = {stage.boundary: stage for stage in self.stage_replay_by_sample[sample_id]}
        for actual in result.state.stage_traces:
            if actual.role != "runtime_pruning":
                continue
            stage = expected.get(actual.boundary)
            if stage is None:
                raise SelectorError(f"Unexpected live replay stage K{actual.boundary}")
            self._validate_replayed_trace(actual, stage)
        if {trace.boundary for trace in result.state.stage_traces if trace.role == "runtime_pruning"} != set(expected):
            raise SelectorError("Replay did not execute every registered stage")
        return result

    @staticmethod
    def _validate_replayed_trace(actual: StageTrace, expected: StageReplay) -> None:
        pairs = {
            "incoming_indices_checksum": (actual.incoming_indices_checksum, expected.incoming_indices_checksum),
            "outgoing_indices_checksum": (actual.outgoing_indices_checksum, expected.outgoing_indices_checksum),
            "merge_assignment_checksum": (actual.merge_assignment_checksum, expected.merge_assignment_checksum),
            "merge_mass_checksum": (actual.merge_mass_checksum, expected.merge_mass_checksum),
        }
        for label, (left, right) in pairs.items():
            if left != right:
                raise SelectorError(f"Replay/live {label} mismatch")
        actual_masses = tuple(actual.merge_output_masses) if actual.merge_output_masses is not None else None
        if actual_masses != expected.merge_output_masses:
            raise SelectorError("Replay/live merge output masses mismatch")
        if actual.score_map_recomputed:
            raise SelectorError("Replay unexpectedly recomputed a score map")


class ProfiledFusedGatherMergeForwarder(
    HardwareProfileMixin,
    FusedGatherMergeProgressiveForwarder,
):
    """Profile the exact fused gather/merge repair."""


class ProfiledNativeCacheForwarder(HardwareProfileMixin, NativeCacheProgressiveForwarder):
    """Profile the exact direct native-cache repair."""


class ProfiledFusedNativeForwarder(HardwareProfileMixin, FusedNativeProgressiveForwarder):
    """Profile the composition of both exact repairs."""
