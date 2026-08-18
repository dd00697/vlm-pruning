from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from vlm_pruning.llava_hooks import (
    FusedGatherMergeProgressiveForwarder,
    NativeCacheProgressiveForwarder,
    OracleIndexArtifact,
    ProgressivePruneForwarder,
    PruningEvent,
    ProfiledProgressiveForwarder,
    RuntimeProfiler,
    SegmentCache,
    StageReplay,
    average_visual_tokens,
    deterministic_stage_seed,
    realized_visual_trace,
    validate_schedule,
)
from vlm_pruning.selectors.base import SelectorError
from scripts.run_progressive_oracle_refcoco import ProgressiveRunError, validate_cli


class ProgressiveScheduleTests(unittest.TestCase):
    def test_registered_full_validation_schedule_accounting(self) -> None:
        expected = {
            ((12, 48, "k12_original"), (20, 6, "query_topk")): (7368, 230.25, 6),
            ((12, 48, "k12_original"), (20, 0, "none")): (7296, 228.0, 0),
            ((4, 448, "query_topk"), (12, 48, "query_topk")): (6848, 214.0, 48),
            ((4, 448, "query_topk_merge"), (12, 48, "query_topk")): (6848, 214.0, 48),
            (
                (4, 448, "query_topk_merge"),
                (12, 48, "query_topk"),
                (20, 0, "none"),
            ): (6272, 196.0, 0),
        }
        for raw, (work, average, final_budget) in expected.items():
            with self.subTest(schedule=raw):
                trace = realized_visual_trace(validate_schedule(raw))
                self.assertEqual(sum(trace), work)
                self.assertEqual(average_visual_tokens(trace), average)
                self.assertEqual(trace[-1], final_budget)

    def test_fused_native_runner_is_restricted_to_m448_prefix(self) -> None:
        base = {
            "pre_llm_selector": "none",
            "pre_llm_budget": None,
            "oracle_teacher_layer": None,
            "oracle_index_input": None,
            "runtime_implementation": "fused_native",
            "limit": 2,
            "registered_scope": "auto",
        }
        validate_cli(
            SimpleNamespace(
                **base,
                schedule=validate_schedule(
                    [(4, 448, "query_topk_merge"), (12, 48), (20, 0, "none")]
                ),
            )
        )
        with self.assertRaises(ProgressiveRunError):
            validate_cli(
                SimpleNamespace(
                    **base,
                    schedule=validate_schedule([(12, 48, "k12_original"), (20, 0, "none")]),
                )
            )

    def test_registered_progressive_averages(self) -> None:
        expected = {
            ((8, 256), (12, 48)): 206,
            ((8, 192), (12, 48)): 198,
            ((4, 384), (12, 48)): 198,
            ((4, 256), (12, 48)): 166,
            ((2, 384), (12, 48)): 186,
            ((2, 288), (12, 48)): 156,
        }
        for raw, average in expected.items():
            events = validate_schedule(raw)
            self.assertEqual(average_visual_tokens(realized_visual_trace(events)), average)

    def test_zero_budget_is_supported_only_by_none(self) -> None:
        events = validate_schedule([(20, 0, "none")])
        self.assertEqual(realized_visual_trace(events), [576] * 20 + [0] * 12)
        with self.assertRaisesRegex(SelectorError, "B=0"):
            validate_schedule([(20, 0, "query_topk")])
        with self.assertRaisesRegex(SelectorError, "reserved"):
            validate_schedule([(20, 4, "none")])

    def test_original_k12_selector_is_restricted_to_frozen_stage(self) -> None:
        events = validate_schedule([(12, 48, "k12_original"), (20, 0, "none")])
        self.assertEqual(events[0], PruningEvent(12, 48, "k12_original"))
        for schedule in (
            [(8, 48, "k12_original")],
            [(12, 96, "k12_original")],
            [(4, 448), (12, 48, "k12_original")],
        ):
            with self.subTest(schedule=schedule):
                with self.assertRaises(SelectorError):
                    validate_schedule(schedule)

    def test_schedule_fails_closed_on_order_and_budget_growth(self) -> None:
        for schedule in (
            [(12, 48), (8, 32)],
            [(8, 48), (12, 96)],
            [(0, 48)],
            [(32, 48)],
        ):
            with self.subTest(schedule=schedule):
                with self.assertRaises(SelectorError):
                    validate_schedule(schedule)

    def test_merge_selector_is_restricted_to_registered_m448_event(self) -> None:
        events = validate_schedule(
            [(4, 448, "query_topk_merge"), (12, 48, "query_topk")]
        )
        self.assertEqual(average_visual_tokens(realized_visual_trace(events)), 214)
        for schedule in (
            [(4, 384, "query_topk_merge")],
            [(8, 448, "query_topk_merge")],
            [(2, 512), (4, 448, "query_topk_merge")],
            [(4, 448, "query_topk_merge"), (12, 48, "query_topk_merge")],
        ):
            with self.subTest(schedule=schedule):
                with self.assertRaisesRegex(SelectorError, "restricted"):
                    validate_schedule(schedule)

    def test_random_stage_seed_is_deterministic_and_stage_specific(self) -> None:
        args = dict(sample_id="sample", cell_seed=2, incoming_indices=list(range(10)))
        first = deterministic_stage_seed(stage_index=0, boundary=8, **args)
        self.assertEqual(first, deterministic_stage_seed(stage_index=0, boundary=8, **args))
        self.assertNotEqual(first, deterministic_stage_seed(stage_index=1, boundary=12, **args))


class OracleArtifactTests(unittest.TestCase):
    def make_artifact(self) -> OracleIndexArtifact:
        ranking = tuple(range(575, -1, -1))
        import hashlib
        checksum = hashlib.sha256(
            json.dumps(list(ranking), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return OracleIndexArtifact(
            sample_id="sample",
            teacher_layer=12,
            original_visual_tokens=576,
            ranked_original_indices=ranking,
            ranking_checksum=checksum,
            query_token_status="matched",
            teacher_visual_token_layer_work=576 * 12,
        )

    def test_artifact_is_index_only_round_trippable_and_selects_raster_order(self) -> None:
        artifact = self.make_artifact().validate()
        payload = artifact.as_dict()
        self.assertTrue(all(not hasattr(value, "shape") for value in payload.values()))
        restored = OracleIndexArtifact.from_dict(payload)
        self.assertEqual(restored, artifact)
        self.assertEqual(restored.selected_indices(3), [573, 574, 575])

    def test_corrupt_duplicate_out_of_range_and_value_fields_fail_closed(self) -> None:
        payload = self.make_artifact().as_dict()
        payload["future_hidden_state"] = [1.0]
        with self.assertRaisesRegex(SelectorError, "unregistered"):
            OracleIndexArtifact.from_dict(payload)
        payload = self.make_artifact().as_dict()
        payload["ranked_original_indices"][0] = payload["ranked_original_indices"][1]
        with self.assertRaisesRegex(SelectorError, "permutation"):
            OracleIndexArtifact.from_dict(payload)


class ProgressiveSelectionTests(unittest.TestCase):
    def make_forwarder_without_model(self):
        import torch

        forwarder = object.__new__(ProgressivePruneForwarder)
        forwarder.torch = torch
        forwarder.image_token_id = 32000
        forwarder.grid = 2
        forwarder.initial_visual_tokens = 4
        return forwarder

    def test_query_topk_recomputes_current_map_and_preserves_raster_order(self) -> None:
        import torch

        forwarder = self.make_forwarder_without_model()
        input_ids = torch.tensor([[1, 32000, 32000, 32000, 32000, 7, 8]])
        original_positions = torch.arange(input_ids.shape[1])
        hidden = torch.zeros((1, input_ids.shape[1], 4), dtype=torch.float32)
        hidden[0, 1:5] = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0]]
        )
        hidden[0, 5] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        selected, checksum, recomputed, derived = forwarder.select_stage_indices(
            hidden,
            input_ids=input_ids,
            original_positions=original_positions,
            incoming_original_indices=[0, 1, 2, 3],
            event=PruningEvent(2, 2, "query_topk"),
            stage_index=0,
            sample_id="sample",
            seed=0,
            query_original_positions=[5],
        )
        self.assertEqual(selected, [0, 1])
        self.assertIsNotNone(checksum)
        self.assertTrue(recomputed)
        self.assertIsNone(derived)

        merge_selected, merge_checksum, merge_recomputed, merge_derived = (
            forwarder.select_stage_indices(
                hidden,
                input_ids=input_ids,
                original_positions=original_positions,
                incoming_original_indices=[0, 1, 2, 3],
                event=PruningEvent(2, 2, "query_topk_merge"),
                stage_index=0,
                sample_id="sample",
                seed=0,
                query_original_positions=[5],
            )
        )
        self.assertEqual(merge_selected, selected)
        self.assertEqual(merge_checksum, checksum)
        self.assertEqual(merge_recomputed, recomputed)
        self.assertEqual(merge_derived, derived)

    def test_zero_tail_drops_without_scoring_or_ranking(self) -> None:
        import torch

        forwarder = self.make_forwarder_without_model()

        def fail_if_scored(*_args, **_kwargs):
            raise AssertionError("zero-tail must not construct a relevance score map")

        forwarder.score_current_visual_tokens = fail_if_scored
        input_ids = torch.tensor([[1, 32000, 32000, 32000, 32000, 7]])
        selected, checksum, recomputed, derived = forwarder.select_stage_indices(
            torch.zeros((1, input_ids.shape[1], 4), dtype=torch.float32),
            input_ids=input_ids,
            original_positions=torch.arange(input_ids.shape[1]),
            incoming_original_indices=[0, 1, 2, 3],
            event=PruningEvent(20, 0, "none"),
            stage_index=1,
            sample_id="sample",
            seed=0,
            query_original_positions=[5],
        )
        self.assertEqual(selected, [])
        self.assertIsNone(checksum)
        self.assertFalse(recomputed)
        self.assertIsNone(derived)

    def test_frozen_original_k12_selector_returns_structured_exact_budget(self) -> None:
        import torch
        from vlm_pruning.selectors import UGPruneConfig

        torch.manual_seed(0)
        forwarder = self.make_forwarder_without_model()
        forwarder.grid = 24
        forwarder.initial_visual_tokens = 576
        forwarder.target_visual_tokens = 576
        forwarder.selector_config = UGPruneConfig(
            selector_strategy="query_topk",
            probe_layers=(1,),
            fusion_mode="single_layer",
            position_mode="gap_preserving",
        )
        forwarder._grid_tensor_cache = {}
        input_ids = torch.tensor([[32000] * 576 + [7, 8]])
        hidden = torch.randn((1, input_ids.shape[1], 16), dtype=torch.float32)
        selected, checksum, recomputed, derived = forwarder.select_stage_indices(
            hidden,
            input_ids=input_ids,
            original_positions=torch.arange(input_ids.shape[1]),
            incoming_original_indices=list(range(576)),
            event=PruningEvent(12, 48, "k12_original"),
            stage_index=0,
            sample_id="sample",
            seed=0,
            query_original_positions=[576, 577],
        )
        self.assertEqual(len(selected), 48)
        self.assertEqual(selected, sorted(selected))
        self.assertEqual(len(set(selected)), 48)
        self.assertIsNotNone(checksum)
        self.assertTrue(recomputed)
        self.assertIsNone(derived)

    def test_gather_retains_all_text_and_supports_empty_visual_set(self) -> None:
        import torch

        forwarder = self.make_forwarder_without_model()
        ids = torch.tensor([[1, 32000, 32000, 2, 32000, 32000, 3]])
        hidden = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)
        mask = torch.ones_like(ids)
        positions = torch.arange(7)
        gathered, new_ids, new_mask, new_positions = forwarder._gather_after_selection(
            hidden,
            input_ids=ids,
            attention_mask=mask,
            original_positions=positions,
            incoming_original_indices=[0, 1, 2, 3],
            selected_original_indices=[],
        )
        self.assertEqual(new_ids.tolist(), [[1, 2, 3]])
        self.assertEqual(new_positions.tolist(), [0, 3, 6])
        self.assertEqual(gathered.shape[1], 3)
        self.assertEqual(new_mask.tolist(), [[1, 1, 1]])

    def test_registered_merge_is_mass_preserving_deterministic_and_text_safe(self) -> None:
        import torch

        forwarder = self.make_forwarder_without_model()
        ids = torch.tensor([[1, 32000, 32000, 2, 32000, 32000, 3]])
        hidden = torch.tensor(
            [[[10.0], [0.0], [1.0], [20.0], [2.0], [3.0], [30.0]]],
            dtype=torch.float32,
        )
        mask = torch.ones_like(ids)
        positions = torch.arange(7)
        result = forwarder._merge_after_selection(
            hidden,
            input_ids=ids,
            attention_mask=mask,
            original_positions=positions,
            incoming_original_indices=[0, 1, 2, 3],
            selected_original_indices=[0, 3],
            visual_masses=torch.ones(4, dtype=torch.float32),
        )
        merged, new_ids, new_mask, new_positions, masses, diagnostics = result
        self.assertEqual(new_ids.tolist(), [[1, 32000, 2, 32000, 3]])
        self.assertEqual(new_positions.tolist(), [0, 1, 3, 5, 6])
        self.assertEqual(new_mask.tolist(), [[1, 1, 1, 1, 1]])
        # Both equidistant dropped patches choose retained original index 0.
        self.assertEqual(merged[0, [1, 3], 0].tolist(), [1.0, 3.0])
        self.assertEqual(masses.tolist(), [3.0, 1.0])
        self.assertEqual(diagnostics["merge_assignment_count"], 2)
        self.assertEqual(diagnostics["merge_input_mass"], 4.0)
        self.assertEqual(diagnostics["merge_output_mass"], 4.0)
        repeat = forwarder._merge_after_selection(
            hidden,
            input_ids=ids,
            attention_mask=mask,
            original_positions=positions,
            incoming_original_indices=[0, 1, 2, 3],
            selected_original_indices=[0, 3],
            visual_masses=torch.ones(4, dtype=torch.float32),
        )
        self.assertEqual(
            diagnostics["merge_assignment_checksum"], repeat[-1]["merge_assignment_checksum"]
        )
        self.assertTrue(torch.equal(merged, repeat[0]))


class HardwareRuntimeProfileTests(unittest.TestCase):
    @staticmethod
    def checksum(values) -> str:
        import hashlib

        return hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def replay_stage(self) -> StageReplay:
        incoming = [0, 1, 2, 3]
        outgoing = [0, 3]
        return StageReplay.from_trace(
            {
                "boundary": 4,
                "selector": "query_topk_merge",
                "requested_budget": 2,
                "incoming_original_indices": incoming,
                "outgoing_original_indices": outgoing,
                "incoming_indices_checksum": self.checksum(incoming),
                "outgoing_indices_checksum": self.checksum(outgoing),
                "score_map_checksum": "scores",
                "merge_assignment_checksum": "assignments",
                "merge_mass_checksum": "masses",
                "merge_output_masses": [3.0, 1.0],
            }
        )

    def test_profiler_disabled_is_inert_and_enabled_reports_cpu_wall(self) -> None:
        import torch

        disabled = RuntimeProfiler(torch, enabled=False)
        with disabled.measure("work"):
            value = torch.arange(4).sum().item()
        self.assertEqual(value, 6)
        self.assertEqual(disabled.snapshot()["records"], [])

        enabled = RuntimeProfiler(torch, enabled=True)
        with enabled.measure("work", stage=1):
            torch.arange(4).sum().item()
        snapshot = enabled.snapshot()
        self.assertEqual(snapshot["components"]["work"]["count"], 1)
        self.assertGreaterEqual(snapshot["components"]["work"]["cpu_wall_total_s"], 0.0)
        if torch.cuda.is_available():
            self.assertGreaterEqual(snapshot["components"]["work"]["cuda_event_total_s"], 0.0)
        else:
            self.assertIsNone(snapshot["components"]["work"]["cuda_event_total_s"])

    def test_profiler_disabled_gather_has_exact_base_parity(self) -> None:
        import torch

        base = object.__new__(ProgressivePruneForwarder)
        profiled = object.__new__(ProfiledProgressiveForwarder)
        for forwarder in (base, profiled):
            forwarder.torch = torch
            forwarder.image_token_id = 32000
        profiled.runtime_profiler = RuntimeProfiler(torch, enabled=False)
        ids = torch.tensor([[1, 32000, 32000, 2]])
        hidden = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
        mask = torch.ones_like(ids)
        positions = torch.arange(4)
        kwargs = {
            "input_ids": ids,
            "attention_mask": mask,
            "original_positions": positions,
            "incoming_original_indices": [0, 1],
            "selected_original_indices": [1],
        }
        expected = base._gather_after_selection(hidden, **kwargs)
        actual = profiled._gather_after_selection(hidden, **kwargs)
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(expected, actual)))
        self.assertEqual(profiled.runtime_profile_snapshot()["records"], [])

    def test_fused_gather_merge_has_exact_base_tensor_and_trace_parity(self) -> None:
        import torch

        base = object.__new__(ProgressivePruneForwarder)
        fused = object.__new__(FusedGatherMergeProgressiveForwarder)
        for forwarder in (base, fused):
            forwarder.torch = torch
            forwarder.image_token_id = 32000
            forwarder.grid = 2
        ids = torch.tensor([[1, 32000, 32000, 2, 32000, 32000, 3]])
        hidden = torch.tensor(
            [[[10.0], [0.0], [1.0], [20.0], [2.0], [3.0], [30.0]]],
            dtype=torch.float32,
        )
        kwargs = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "original_positions": torch.arange(7),
            "incoming_original_indices": [0, 1, 2, 3],
            "selected_original_indices": [0, 3],
            "visual_masses": torch.ones(4, dtype=torch.float32),
        }
        expected = base._merge_after_selection(hidden, **kwargs)
        actual = fused._merge_after_selection(hidden, **kwargs)
        for left, right in zip(expected[:-1], actual[:-1]):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(expected[-1], actual[-1])

    def test_native_cache_passes_underlying_cache_directly_to_layers(self) -> None:
        import torch

        received = []

        class Layer:
            def __call__(self, hidden_states, **kwargs):
                received.append(kwargs["past_key_value"])
                return (hidden_states,)

        native_cache = object()
        language_model = SimpleNamespace(
            layers=[Layer(), Layer(), Layer()],
            rotary_emb=lambda hidden_states, position_ids: None,
        )
        forwarder = object.__new__(NativeCacheProgressiveForwarder)
        forwarder.torch = torch
        forwarder.model = SimpleNamespace(model=SimpleNamespace(language_model=language_model))
        forwarder._causal_mask = lambda *_args: None
        forwarder._cache_position = lambda hidden_states, _cache: torch.arange(hidden_states.shape[1])
        segment = SegmentCache(
            segment_index=0,
            start_layer=1,
            end_layer=3,
            cache=native_cache,
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            prompt_length=2,
            visual_tokens=1,
        )
        forwarder.run_decoder_layers(
            torch.ones((1, 1, 4)),
            attention_mask=segment.attention_mask,
            position_ids=torch.zeros((1, 1), dtype=torch.long),
            start_layer=1,
            end_layer=3,
            past_key_values=segment,
            use_cache=True,
        )
        self.assertEqual(received, [native_cache, native_cache])

    def test_replay_bypasses_score_and_rank_but_validates_live_incoming_state(self) -> None:
        import torch

        forwarder = object.__new__(ProfiledProgressiveForwarder)
        forwarder.torch = torch
        forwarder.runtime_profiler = RuntimeProfiler(torch, enabled=False)
        forwarder.stage_replay_by_sample = {"sample": (self.replay_stage(),)}
        selected, checksum, recomputed, derived = forwarder.select_stage_indices(
            None,
            input_ids=None,
            original_positions=None,
            incoming_original_indices=[0, 1, 2, 3],
            event=PruningEvent(4, 2, "query_topk_merge"),
            stage_index=0,
            sample_id="sample",
            seed=0,
            query_original_positions=[],
        )
        self.assertEqual(selected, [0, 3])
        self.assertEqual(checksum, "scores")
        self.assertFalse(recomputed)
        self.assertIsNone(derived)
        with self.assertRaisesRegex(SelectorError, "incoming indices differ"):
            forwarder.select_stage_indices(
                None,
                input_ids=None,
                original_positions=None,
                incoming_original_indices=[0, 1, 2, 4],
                event=PruningEvent(4, 2, "query_topk_merge"),
                stage_index=0,
                sample_id="sample",
                seed=0,
                query_original_positions=[],
            )

    def test_corrupt_replay_checksum_fails_closed(self) -> None:
        trace = self.replay_stage().__dict__.copy()
        trace["incoming_indices_checksum"] = "corrupt"
        with self.assertRaisesRegex(SelectorError, "incoming-index checksum"):
            StageReplay(**trace).validate()


class ProgressiveRunnerTests(unittest.TestCase):
    def test_generation_preserves_prefill_cache_storage_snapshot(self) -> None:
        import torch

        forwarder = object.__new__(ProgressivePruneForwarder)
        forwarder.torch = torch
        forwarder._cuda_sync = lambda: None
        state = SimpleNamespace(
            generated_tokens_cached=0,
            segments=[],
            final_retained_indices=[],
            stage_traces=[],
        )
        forwarder.prefill_with_segment_caches = lambda **_kwargs: SimpleNamespace(
            logits=torch.tensor([[[0.0, 1.0]]]),
            state=state,
            diagnostics={
                "progressive/cache_generated_tokens": 0,
                "cache/prompt_segment_bytes": [10],
                "cache/prompt_total_bytes": 10,
            },
        )

        def decode(current_state, _token):
            current_state.generated_tokens_cached += 1
            return torch.tensor([[[0.0, 1.0]]])

        forwarder.decode_one_with_segment_caches = decode
        forwarder._diagnostics = lambda current_state: {
            "progressive/cache_generated_tokens": current_state.generated_tokens_cached,
            "cache/prompt_segment_bytes": [20],
            "cache/prompt_total_bytes": 20,
        }
        result = forwarder.generate_with_segment_caches(
            inputs_embeds=None,
            input_ids=None,
            attention_mask=None,
            sample_id="sample",
            seed=0,
            query_original_positions=(),
            max_new_tokens=2,
            eos_ids=set(),
        )
        self.assertEqual(result.diagnostics["progressive/cache_generated_tokens"], 1)
        self.assertEqual(result.diagnostics["cache/prompt_segment_bytes"], [10])
        self.assertEqual(result.diagnostics["cache/prompt_total_bytes"], 10)

    def test_relative_output_root_is_normalized_before_provenance_paths(self) -> None:
        from scripts.run_progressive_oracle_refcoco import REPO_ROOT

        relative = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814/preflight/smoke_runs/full"
        resolved = relative.resolve()
        self.assertEqual(
            str(resolved.relative_to(REPO_ROOT)),
            "experiments/progressive_oracle_refcoco_val_20260814/preflight/smoke_runs/full",
        )

    def test_constructor_uses_valid_inert_base_probe_configuration(self) -> None:
        import torch

        fake_model = SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(layers=[object()] * 32)
            )
        )
        forwarder = ProgressivePruneForwarder(
            torch,
            fake_model,
            image_token_id=32000,
            schedule=[],
        )
        self.assertEqual(forwarder.selector_config.probe_layers, (1,))
        self.assertEqual(forwarder.selector_config.position_mode, "gap_preserving")

    def test_oracle_replay_cli_is_restricted_to_oracle_and_existing_file(self) -> None:
        import tempfile
        from pathlib import Path
        from scripts.run_progressive_oracle_refcoco import ProgressiveRunError, validate_cli

        with tempfile.TemporaryDirectory() as directory:
            indices = Path(directory) / "indices.jsonl"
            indices.write_text("{}\n", encoding="utf-8")
            invalid = SimpleNamespace(
                pre_llm_selector="random",
                pre_llm_budget=64,
                oracle_teacher_layer=None,
                oracle_index_input=indices,
                schedule=(),
                limit=2,
            )
            with self.assertRaisesRegex(ProgressiveRunError, "valid only for oracle"):
                validate_cli(invalid)


if __name__ == "__main__":
    unittest.main()
