from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from vlm_pruning.llava_hooks import (
    OracleIndexArtifact,
    ProgressivePruneForwarder,
    PruningEvent,
    average_visual_tokens,
    deterministic_stage_seed,
    realized_visual_trace,
    validate_schedule,
)
from vlm_pruning.selectors.base import SelectorError


class ProgressiveScheduleTests(unittest.TestCase):
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
