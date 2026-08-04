from __future__ import annotations

import unittest

import numpy as np

from vlm_pruning.features.map_fusion import fuse_layer_maps, parse_bins, parse_int_tuple, parse_probe_layers
from vlm_pruning.features.unrolled_maps import LayerMapResult
from vlm_pruning.geometry.grid import bin_id, rc_to_patch
from vlm_pruning.selectors.ug_prune import (
    UGPruneConfig,
    UGPruneInputs,
    UGPruneSelector,
    budget_split,
    select_target,
)


def phase2_inputs(score_map: np.ndarray) -> UGPruneInputs:
    return UGPruneInputs(
        fused_map=score_map,
        map_stats={"map_max": float(score_map.max())},
        map_key="layer4:query_max+layer8:query_max+layer12:query_max",
        query_text="the red cup",
        query_token_status="matched_actual_input_subsequence",
        query_token_positions=[100, 101, 102],
    )


class Phase2SplitTests(unittest.TestCase):
    def test_budget_split_defaults_match_expected_minimums(self) -> None:
        config = UGPruneConfig()
        self.assertEqual(budget_split(16, config), (8, 4, 4))
        self.assertEqual(budget_split(32, config), (16, 8, 8))
        self.assertEqual(budget_split(48, config), (24, 12, 12))
        self.assertEqual(budget_split(58, config), (29, 17, 12))

    def test_budget_split_is_exact_for_common_budgets(self) -> None:
        config = UGPruneConfig()
        for budget in [1, 16, 32, 48, 58, 144, 288]:
            self.assertEqual(sum(budget_split(budget, config)), budget)


class Phase2SelectorTests(unittest.TestCase):
    def test_selector_returns_exact_budget_with_roles(self) -> None:
        scores = np.zeros((24, 24), dtype=np.float32)
        scores[4, 5] = 10.0
        scores[10, 10] = 7.0
        selector = UGPruneSelector(UGPruneConfig(nms_radius=1))

        result = selector.select_from_map(
            sample_id="sample",
            budget=32,
            original_tokens=576,
            grid=24,
            seed=0,
            inputs=phase2_inputs(scores),
        )

        self.assertEqual(len(result.selected_indices), 32)
        self.assertEqual(len(set(result.selected_indices)), 32)
        self.assertIn("target", result.role_labels)
        self.assertIn("context", result.role_labels)
        self.assertIn("anchor", result.role_labels)
        role_union = set().union(*(set(values) for values in result.role_labels.values()))
        self.assertTrue(role_union.issubset(set(result.selected_indices)))
        self.assertEqual(result.metadata["map_source"], "query_max")

    def test_role_modes_change_role_labels(self) -> None:
        scores = np.arange(576, dtype=np.float32).reshape(24, 24)
        target_only = UGPruneSelector(UGPruneConfig(role_mode="target_only")).select_from_map(
            sample_id="sample",
            budget=16,
            original_tokens=576,
            grid=24,
            seed=0,
            inputs=phase2_inputs(scores),
        )
        target_context = UGPruneSelector(UGPruneConfig(role_mode="target_context")).select_from_map(
            sample_id="sample",
            budget=16,
            original_tokens=576,
            grid=24,
            seed=0,
            inputs=phase2_inputs(scores),
        )

        self.assertIn("target", target_only.role_labels)
        self.assertNotIn("context", target_only.role_labels)
        self.assertNotIn("anchor", target_only.role_labels)
        self.assertIn("context", target_context.role_labels)
        self.assertNotIn("anchor", target_context.role_labels)

    def test_target_nms_suppresses_nearby_peak_first(self) -> None:
        scores = np.zeros(576, dtype=np.float32)
        scores[rc_to_patch(5, 5)] = 10.0
        scores[rc_to_patch(5, 6)] = 9.0
        scores[rc_to_patch(12, 12)] = 8.0

        selected = select_target(scores, 2, nms_radius=1, grid=24)

        self.assertEqual(selected, [rc_to_patch(5, 5), rc_to_patch(12, 12)])

    def test_anchors_cover_multiple_bins(self) -> None:
        scores = np.zeros((24, 24), dtype=np.float32)
        result = UGPruneSelector().select_from_map(
            sample_id="sample",
            budget=48,
            original_tokens=576,
            grid=24,
            seed=0,
            inputs=phase2_inputs(scores),
        )

        anchor_bins = {bin_id(index, bins=(3, 3), grid=24) for index in result.role_labels["anchor"]}
        self.assertGreaterEqual(len(anchor_bins), 6)

    def test_anchor_patterns_are_exact_budget(self) -> None:
        scores = np.arange(576, dtype=np.float32).reshape(24, 24)
        for pattern in ["none", "corners", "grid", "fps"]:
            with self.subTest(pattern=pattern):
                result = UGPruneSelector(UGPruneConfig(anchor_pattern=pattern)).select_from_map(
                    sample_id="sample",
                    budget=58,
                    original_tokens=576,
                    grid=24,
                    seed=0,
                    inputs=phase2_inputs(scores),
                )
                self.assertEqual(len(result.selected_indices), 58)
                if pattern == "none":
                    self.assertNotIn("anchor", result.role_labels)
                else:
                    self.assertIn("anchor", result.role_labels)


class Phase2MapFusionTests(unittest.TestCase):
    def test_parse_helpers(self) -> None:
        self.assertEqual(parse_int_tuple("4,8,12"), (4, 8, 12))
        self.assertEqual(parse_probe_layers("4,final", final_layer=32), (4, 32))
        self.assertEqual(parse_bins("3,3"), (3, 3))
        with self.assertRaises(ValueError):
            parse_bins("3")

    def test_fuse_layer_maps_averages_requested_z_maps(self) -> None:
        first = np.zeros((24, 24), dtype=np.float32)
        second = np.ones((24, 24), dtype=np.float32)
        third = np.zeros((24, 24), dtype=np.float32)
        first[1, 1] = 10.0
        second[2, 2] = 10.0
        third[3, 3] = 10.0
        maps = [
            LayerMapResult(4, list(range(576)), [580], {"query_max": first}, {"query_max": first}),
            LayerMapResult(8, list(range(576)), [580], {"query_max": second}, {"query_max": second}),
            LayerMapResult(12, list(range(576)), [580], {"query_max": third}, {"query_max": third}),
        ]

        fused = fuse_layer_maps(maps, source="query_max", layers=(4, 8, 12), grid=24)

        self.assertEqual(fused.fused_map.shape, (24, 24))
        self.assertEqual(fused.layer_keys, ("layer4:query_max", "layer8:query_max", "layer12:query_max"))
        self.assertIn("entropy_norm", fused.stats)

    def test_fusion_modes(self) -> None:
        low = np.zeros((24, 24), dtype=np.float32)
        high = np.zeros((24, 24), dtype=np.float32)
        low[1, 1] = 2.0
        high[10, 10] = 20.0
        maps = [
            LayerMapResult(4, list(range(576)), [580], {"query_max": low}, {"query_max": low}),
            LayerMapResult(8, list(range(576)), [580], {"query_max": high}, {"query_max": high}),
        ]

        single = fuse_layer_maps(maps, source="query_max", layers=(4,), grid=24, fusion_mode="single_layer")
        hard = fuse_layer_maps(maps, source="query_max", layers=(4, 8), grid=24, fusion_mode="hard_best")

        self.assertEqual(single.selected_layer, 4)
        self.assertEqual(hard.selected_layer, 8)


if __name__ == "__main__":
    unittest.main()
