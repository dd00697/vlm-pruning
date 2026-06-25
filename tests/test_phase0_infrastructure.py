from __future__ import annotations

import unittest

from vlm_pruning.analysis.coverage import selection_diagnostics
from vlm_pruning.geometry.grid import bin_id, col_band, neighbours, patch_to_rc, rc_to_patch, ring, row_band
from vlm_pruning.selectors import SelectionResult, SelectorError, selector_for_method


class GridTests(unittest.TestCase):
    def test_patch_round_trip(self) -> None:
        for index in [0, 23, 24, 287, 575]:
            row, col = patch_to_rc(index, 24)
            self.assertEqual(rc_to_patch(row, col, 24), index)

    def test_neighbours_center_edge_corner(self) -> None:
        self.assertEqual(len(neighbours(rc_to_patch(12, 12), 1, 24)), 8)
        self.assertEqual(len(neighbours(0, 1, 24)), 3)
        self.assertEqual(len(neighbours(23, 1, 24)), 3)

    def test_ring_excludes_seeds(self) -> None:
        seeds = {rc_to_patch(12, 12), rc_to_patch(12, 13)}
        out = ring(seeds, radius=1, grid=24)
        self.assertTrue(out)
        self.assertFalse(seeds & out)

    def test_bins_and_bands(self) -> None:
        self.assertEqual(bin_id(0, (3, 3), 24), (0, 0))
        self.assertEqual(row_band(0), "top")
        self.assertEqual(col_band(23), "right")
        self.assertEqual(row_band(575), "bottom")


class SelectorTests(unittest.TestCase):
    def test_full_selector(self) -> None:
        result = selector_for_method("full").select(
            sample_id="sample", budget=576, original_tokens=576, grid=24, seed=0
        )
        self.assertEqual(result.selected_indices[0], 0)
        self.assertEqual(result.selected_indices[-1], 575)
        self.assertEqual(result.role_labels, {})

    def test_random_selector_is_deterministic(self) -> None:
        selector = selector_for_method("random")
        first = selector.select(sample_id="sample", budget=32, original_tokens=576, grid=24, seed=7)
        second = selector.select(sample_id="sample", budget=32, original_tokens=576, grid=24, seed=7)
        self.assertEqual(first.selected_indices, second.selected_indices)

    def test_uniform_selector_budget(self) -> None:
        result = selector_for_method("uniform").select(
            sample_id="sample", budget=48, original_tokens=576, grid=24, seed=0
        )
        self.assertEqual(len(result.selected_indices), 48)
        self.assertEqual(len(set(result.selected_indices)), 48)

    def test_validation_rejects_duplicates(self) -> None:
        with self.assertRaises(SelectorError):
            SelectionResult([1, 1], "bad", 2, 576, 24).validate()


class CoverageTests(unittest.TestCase):
    def test_full_selection_covers_box(self) -> None:
        diagnostics = selection_diagnostics(list(range(576)), [25, 25, 75, 75], (100, 100), 24)
        self.assertAlmostEqual(float(diagnostics["target_coverage"]), 1.0)
        self.assertEqual(int(diagnostics["selected_occupied_bins_3x3"]), 9)

    def test_empty_selection_has_zero_coverage(self) -> None:
        diagnostics = selection_diagnostics([], [25, 25, 75, 75], (100, 100), 24)
        self.assertEqual(float(diagnostics["target_coverage"]), 0.0)
        self.assertEqual(float(diagnostics["selected_spatial_entropy_3x3"]), 0.0)


if __name__ == "__main__":
    unittest.main()
