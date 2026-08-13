from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from vlm_pruning.llava_hooks import (
    CacheLengthProxy,
    FROZEN_UG_GAP58_NAME,
    LayerAwareCacheState,
    MidLayerUGPruneForwarder,
    frozen_ug_gap58_config,
    selected_positions_for_retained,
    selected_positions_for_retained_tensor,
    ug_gap58_config_for_prune_layer,
    validate_retained_indices,
)
from vlm_pruning.features import extract_maps_from_hidden_states, fuse_layer_maps
from vlm_pruning.selectors import UGPruneInputs, UGPruneSelector
from vlm_pruning.selectors.base import SelectorError


class FakeCache:
    is_sliding = [False] * 32

    def __init__(self, lengths: dict[int, int]) -> None:
        self.lengths = dict(lengths)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.lengths.get(int(layer_idx), 0)

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        return self.lengths.get(int(layer_idx), 0) + int(query_length), 0

    def update(self, key_states, value_states, layer_idx: int, *args, **kwargs):
        return key_states, value_states


class Phase2BFrozenConfigTests(unittest.TestCase):
    def test_cli_exposes_registered_prompt_profile_to_shared_prompt_builder(self) -> None:
        from scripts.run_ug_prune_single_pass import parse_args

        argv = [
            "run_ug_prune_single_pass.py",
            "--dataset", "mme",
            "--task", "vqa",
            "--manifest", "mme.jsonl",
            "--prompt-profile", "registered",
        ]
        with patch("sys.argv", argv):
            args = parse_args()
        self.assertEqual(args.prompt_profile, "registered")

    def test_frozen_config_matches_phase2_winner(self) -> None:
        config = frozen_ug_gap58_config()

        self.assertEqual(FROZEN_UG_GAP58_NAME, "UG-Prune-Gap-58")
        self.assertEqual(config.map_family, "unrolled")
        self.assertEqual(config.map_source, "query_max")
        self.assertEqual(config.probe_layers, (4, 8, 12))
        self.assertEqual(config.fusion_mode, "soft_average")
        self.assertEqual(config.role_mode, "target_context_anchor")
        self.assertEqual(config.context_radius, 1)
        self.assertEqual(config.anchor_pattern, "fps")
        self.assertEqual(config.position_mode, "gap_preserving")

    def test_forwarder_rejects_missing_probe_layers(self) -> None:
        import torch

        with self.assertRaises(SelectorError):
            MidLayerUGPruneForwarder(
                torch,
                object(),
                image_token_id=32000,
                prune_after_layer=8,
                target_visual_tokens=58,
                selector_config=frozen_ug_gap58_config(),
            )

    def test_k_specific_config_uses_available_probe_layers(self) -> None:
        self.assertEqual(ug_gap58_config_for_prune_layer(12).probe_layers, (4, 8, 12))
        self.assertEqual(ug_gap58_config_for_prune_layer(11).probe_layers, (4, 8, 11))
        self.assertEqual(ug_gap58_config_for_prune_layer(10).probe_layers, (4, 8, 10))
        self.assertEqual(ug_gap58_config_for_prune_layer(8).probe_layers, (4, 8))
        self.assertEqual(ug_gap58_config_for_prune_layer(4).probe_layers, (4,))
        self.assertEqual(ug_gap58_config_for_prune_layer(4).fusion_mode, "single_layer")
        with self.assertRaises(SelectorError):
            ug_gap58_config_for_prune_layer(3)

    def test_config_overrides_support_layer_specific_maps_and_role_ratios(self) -> None:
        config = ug_gap58_config_for_prune_layer(
            11,
            map_source="query_pooled",
            fusion_mode="single_layer",
            target_ratio=0.45,
            context_ratio=0.40,
            anchor_ratio=0.15,
        )

        self.assertEqual(config.probe_layers, (11,))
        self.assertEqual(config.map_source, "query_pooled")
        self.assertEqual(config.fusion_mode, "single_layer")
        self.assertAlmostEqual(config.target_ratio, 0.45)
        self.assertAlmostEqual(config.context_ratio, 0.40)
        self.assertAlmostEqual(config.anchor_ratio, 0.15)

    def test_config_overrides_support_explicit_probe_layers_and_compact_positions(self) -> None:
        config = ug_gap58_config_for_prune_layer(
            12,
            fusion_mode="soft_average",
            probe_layers=(8, 12),
            position_mode="compact",
        )

        self.assertEqual(config.probe_layers, (8, 12))
        self.assertEqual(config.position_mode, "compact")
        with self.assertRaises(SelectorError):
            ug_gap58_config_for_prune_layer(12, probe_layers=(8, 16))


class Phase2BPositionTests(unittest.TestCase):
    def test_gap_preserving_positions_keep_text_and_selected_visual_slots(self) -> None:
        import torch

        image_token_id = 32000
        input_ids = torch.tensor([[11, image_token_id, image_token_id, image_token_id, 22, image_token_id, image_token_id, 33]])

        positions = selected_positions_for_retained(torch, input_ids, image_token_id, [3, 0])

        self.assertEqual(positions.tolist(), [0, 1, 4, 5, 7])

    def test_gap_preserving_tensor_positions_keep_text_and_selected_visual_slots(self) -> None:
        import torch

        image_token_id = 32000
        input_ids = torch.tensor([[11, image_token_id, image_token_id, image_token_id, 22, image_token_id, image_token_id, 33]])

        positions = selected_positions_for_retained_tensor(torch, input_ids, image_token_id, torch.tensor([3, 0]))

        self.assertEqual(positions.tolist(), [0, 1, 4, 5, 7])

    def test_validate_retained_indices_requires_exact_unique_budget(self) -> None:
        self.assertEqual(validate_retained_indices([2, 0, 1], original_tokens=4, budget=3), [2, 0, 1])
        with self.assertRaises(SelectorError):
            validate_retained_indices([0, 1], original_tokens=4, budget=3)
        with self.assertRaises(SelectorError):
            validate_retained_indices([0, 1, 1], original_tokens=4, budget=3)
        with self.assertRaises(SelectorError):
            validate_retained_indices([0, 1, 4], original_tokens=4, budget=3)


class Phase2BCacheStateTests(unittest.TestCase):
    def test_cache_length_proxy_maps_default_queries_to_late_start_layer(self) -> None:
        cache = FakeCache({0: 0, 12: 128})
        proxy = CacheLengthProxy(cache, default_layer_idx=12)

        self.assertEqual(proxy.get_seq_length(), 128)
        self.assertEqual(proxy.get_seq_length(0), 128)
        self.assertEqual(proxy.get_seq_length(12), 128)
        self.assertEqual(proxy.get_mask_sizes(1, 0), (129, 0))

    def test_compact_cache_state_uses_layer_specific_next_positions(self) -> None:
        state = LayerAwareCacheState(
            early_cache=FakeCache({0: 646}),
            late_cache=FakeCache({12: 118}),
            selection=SimpleNamespace(),
            fused=None,
            selected_positions=None,
            early_attention_mask=SimpleNamespace(shape=(1, 646)),
            late_attention_mask=SimpleNamespace(shape=(1, 118)),
            full_prompt_len=646,
            pruned_prompt_len=118,
            prune_after_layer=12,
            position_mode="compact",
            generated_tokens_cached=3,
        )

        self.assertEqual(state.next_early_position_id(), 649)
        self.assertEqual(state.next_late_position_id(), 121)

    def test_layer_aware_cache_state_position_and_expected_lengths(self) -> None:
        import torch

        state = LayerAwareCacheState(
            early_cache=FakeCache({0: 646}),
            late_cache=FakeCache({12: 128}),
            selection=None,
            fused=None,
            selected_positions=torch.arange(128),
            early_attention_mask=torch.ones((1, 646), dtype=torch.long),
            late_attention_mask=torch.ones((1, 128), dtype=torch.long),
            full_prompt_len=646,
            pruned_prompt_len=128,
            prune_after_layer=10,
            generated_tokens_cached=0,
        )

        self.assertEqual(state.prune_after_layer, 10)
        self.assertEqual(state.next_position_id(), 646)
        self.assertEqual(state.expected_early_cache_len(), 646)
        self.assertEqual(state.expected_late_cache_len(), 128)
        state.generated_tokens_cached = 3
        self.assertEqual(state.next_position_id(), 649)
        self.assertEqual(state.expected_early_cache_len(), 649)
        self.assertEqual(state.expected_late_cache_len(), 131)


class Phase2BSelectionTests(unittest.TestCase):
    def test_benchmark_map_matches_reference_for_final_factorial_sources(self) -> None:
        import torch

        generator = torch.Generator().manual_seed(20260729)
        input_ids = torch.tensor([[7] + [32000] * 576 + [11, 12, 13]])
        hidden_states = [
            torch.randn(1, input_ids.shape[1], 16, generator=generator)
            for _ in range(21)
        ]
        query_positions = [577, 578]
        for source in ("query_max", "query_pooled"):
            with self.subTest(source=source):
                config = ug_gap58_config_for_prune_layer(
                    20,
                    map_source=source,
                    fusion_mode="single_layer",
                    probe_layers=(20,),
                    position_mode="gap_preserving",
                )
                forwarder = MidLayerUGPruneForwarder(
                    torch,
                    object(),
                    image_token_id=32000,
                    grid=24,
                    prune_after_layer=20,
                    target_visual_tokens=48,
                    selector_config=config,
                    diagnostics_mode="benchmark",
                )
                fast, _, _ = forwarder._extract_fused_map_torch(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    query_token_positions=query_positions,
                )
                reference_maps = extract_maps_from_hidden_states(
                    hidden_states,
                    input_ids,
                    32000,
                    query_positions,
                    (20,),
                    grid=24,
                )
                reference = fuse_layer_maps(
                    reference_maps,
                    source=source,
                    layers=(20,),
                    grid=24,
                    fusion_mode="single_layer",
                ).fused_map
                self.assertTrue(
                    np.allclose(
                        fast.detach().cpu().numpy().reshape(24, 24),
                        reference,
                        rtol=1e-5,
                        atol=1e-5,
                    )
                )

    def test_forwarder_selector_matches_direct_frozen_selector(self) -> None:
        import torch

        scores = np.zeros((24, 24), dtype=np.float32)
        scores[5, 5] = 10.0
        scores[10, 10] = 7.5
        inputs = UGPruneInputs(
            fused_map=scores,
            map_stats={"map_max": 10.0},
            map_key="layer4:query_max+layer8:query_max+layer12:query_max",
            query_text="red cup",
            query_token_status="matched_actual_input_subsequence",
            query_token_positions=[600, 601],
        )
        config = frozen_ug_gap58_config()
        direct = UGPruneSelector(config).select_from_map(
            sample_id="sample",
            budget=58,
            original_tokens=576,
            grid=24,
            seed=0,
            inputs=inputs,
        )
        forwarder = MidLayerUGPruneForwarder(
            torch,
            object(),
            image_token_id=32000,
            prune_after_layer=12,
            target_visual_tokens=58,
            selector_config=config,
        )
        via_forwarder = forwarder.selector.select_from_map(
            sample_id="sample",
            budget=58,
            original_tokens=576,
            grid=24,
            seed=0,
            inputs=inputs,
        )

        self.assertEqual(via_forwarder.selected_indices, direct.selected_indices)
        self.assertEqual(len(via_forwarder.selected_indices), 58)
        self.assertEqual(len(set(via_forwarder.selected_indices)), 58)
        self.assertIn("target", via_forwarder.role_labels)
        self.assertIn("context", via_forwarder.role_labels)
        self.assertIn("anchor", via_forwarder.role_labels)

    def test_fast_torch_selector_matches_reference_selector(self) -> None:
        import torch

        rng = np.random.default_rng(7)
        scores = rng.normal(size=(24, 24)).astype(np.float32)
        scores[3, 4] = 8.0
        scores[12, 12] = 7.5
        config = frozen_ug_gap58_config()
        inputs = UGPruneInputs(
            fused_map=scores,
            map_stats={},
            map_key="layer4:query_max+layer8:query_max+layer12:query_max",
            query_text="red cup",
            query_token_status="matched_actual_input_subsequence",
            query_token_positions=[600, 601],
        )
        direct = UGPruneSelector(config).select_from_map(
            sample_id="sample",
            budget=58,
            original_tokens=576,
            grid=24,
            seed=0,
            inputs=inputs,
        )
        forwarder = MidLayerUGPruneForwarder(
            torch,
            object(),
            image_token_id=32000,
            prune_after_layer=12,
            target_visual_tokens=58,
            selector_config=config,
            diagnostics_mode="benchmark",
        )

        fast, fast_tensor = forwarder._selection_from_fused_torch(
            scores=torch.tensor(scores.reshape(-1)),
            sample_id="sample",
            original_tokens=576,
            seed=0,
            map_key=inputs.map_key,
            map_stats=forwarder._score_map_stats_torch(torch.tensor(scores.reshape(-1))),
            query_text=inputs.query_text,
            query_token_status=inputs.query_token_status,
            query_token_positions=inputs.query_token_positions,
        )

        self.assertEqual(fast.selected_indices, direct.selected_indices)
        self.assertEqual(fast.role_labels, direct.role_labels)
        self.assertEqual(fast_tensor.numel(), 58)
        self.assertEqual(len(set(fast.selected_indices)), 58)
        stats = fast.metadata["map_stats"]
        self.assertIn("entropy_norm", stats)
        self.assertIn("top_58_mass", stats)
        self.assertIn("selected_occupied_bins_3x3", stats)
        self.assertIn("selected_mean_pairwise_spread", stats)

    def test_fast_torch_selector_matches_reference_for_variable_budgets(self) -> None:
        import torch

        rng = np.random.default_rng(19)
        scores = rng.normal(size=(24, 24)).astype(np.float32)
        scores[2, 2] = 8.0
        scores[18, 7] = 7.5
        config = frozen_ug_gap58_config()
        inputs = UGPruneInputs(
            fused_map=scores,
            map_stats={},
            map_key="layer4:query_max+layer8:query_max+layer12:query_max",
            query_text="red cup",
            query_token_status="matched_actual_input_subsequence",
            query_token_positions=[600, 601],
        )
        for budget in (48, 72, 96):
            with self.subTest(budget=budget):
                direct = UGPruneSelector(config).select_from_map(
                    sample_id="sample",
                    budget=budget,
                    original_tokens=576,
                    grid=24,
                    seed=0,
                    inputs=inputs,
                )
                forwarder = MidLayerUGPruneForwarder(
                    torch,
                    object(),
                    image_token_id=32000,
                    prune_after_layer=12,
                    target_visual_tokens=budget,
                    selector_config=config,
                    diagnostics_mode="benchmark",
                )
                flat = torch.tensor(scores.reshape(-1))
                fast, fast_tensor = forwarder._selection_from_fused_torch(
                    scores=flat,
                    sample_id="sample",
                    original_tokens=576,
                    seed=0,
                    map_key=inputs.map_key,
                    map_stats=forwarder._score_map_stats_torch(flat),
                    query_text=inputs.query_text,
                    query_token_status=inputs.query_token_status,
                    query_token_positions=inputs.query_token_positions,
                )

                self.assertEqual(fast.selected_indices, direct.selected_indices)
                self.assertEqual(fast.role_labels, direct.role_labels)
                self.assertEqual(fast_tensor.numel(), budget)
                self.assertEqual(len(set(fast.selected_indices)), budget)
                self.assertIn(f"top_{budget}_mass", fast.metadata["map_stats"])

    def test_benchmark_result_keeps_required_metrics_and_omits_heavy_fields(self) -> None:
        from scripts.run_ug_prune_single_pass import build_result

        scores = np.zeros((24, 24), dtype=np.float32)
        scores[5, 5] = 10.0
        selection = UGPruneSelector(frozen_ug_gap58_config()).select_from_map(
            sample_id="sample",
            budget=58,
            original_tokens=576,
            grid=24,
            seed=0,
            inputs=UGPruneInputs(
                fused_map=scores,
                map_stats={},
                map_key="layer4:query_max+layer8:query_max+layer12:query_max",
                query_text="red cup",
                query_token_status="matched_actual_input_subsequence",
                query_token_positions=[600, 601],
            ),
        )
        args = SimpleNamespace(dataset="refcoco", split="val", seed=0, diagnostics_mode="benchmark")
        sample = {"sample_id": "sample", "image_id": "image"}
        diagnostics = {
            "actual_llm_visual_tokens": 58,
            "mid_layer/early_cache_seq_len": 650,
            "mid_layer/late_cache_seq_len": 132,
            "ug/selection_backend": "torch_fast",
        }
        config = {"tokens/target_visual_tokens": 58, "tokens/keep_ratio": 58 / 576}

        row = build_result(args, sample, Path("image.jpg"), "[0,0,1,1]", 4, 1.0, selection, diagnostics, config)

        self.assertTrue(row["retained_indices_checksum"])
        self.assertEqual(row["actual_llm_visual_tokens"], 58)
        self.assertEqual(row["ug/selection_backend"], "torch_fast")
        self.assertEqual(row["retained_indices"], selection.selected_indices)
        self.assertNotIn("selector_metadata", row)
        self.assertNotIn("role_labels", row)
        self.assertNotIn("ug_fused_map_stats", row)


if __name__ == "__main__":
    unittest.main()
