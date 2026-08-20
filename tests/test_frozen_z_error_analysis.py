from __future__ import annotations

import unittest

import numpy as np

import tempfile
from pathlib import Path

from scripts.analyse_frozen_z_error_analysis import (
    ALLOWED_ANALYSIS_FILES,
    iou_summaries,
    paired_bootstrap,
    require_allowed_output_inventory,
    require_exact_pairing,
    require_expected_transitions,
    transition_counts,
    verify_ledger_file,
)
from scripts.frozen_z_error_analysis_common import ErrorAnalysisFailure


class FrozenZErrorAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.full = np.asarray([0.8, 0.6, 0.4, 0.2, 0.55, 0.45], dtype=np.float64)
        self.z = np.asarray([0.7, 0.4, 0.6, 0.1, 0.55, 0.51], dtype=np.float64)

    def test_transition_counts_cover_exact_pair_states(self) -> None:
        self.assertEqual(transition_counts(self.full, self.z), {
            "both_correct": 2, "full_only": 1, "z_only": 2, "both_incorrect": 1,
        })

    def test_transition_threshold_is_inclusive(self) -> None:
        full = np.asarray([0.5, 0.499999], dtype=np.float64)
        z = np.asarray([0.499999, 0.5], dtype=np.float64)
        self.assertEqual(transition_counts(full, z), {
            "both_correct": 0, "full_only": 1, "z_only": 1, "both_incorrect": 0,
        })

    def test_transition_rejects_unpaired_arrays(self) -> None:
        with self.assertRaises(ErrorAnalysisFailure):
            transition_counts(self.full, self.z[:-1])

    def test_bootstrap_point_is_full_minus_z(self) -> None:
        curve, states = paired_bootstrap(self.full, self.z, [0.5], seed=7, resamples=200, batch=20)
        expected = float(np.mean((self.full >= 0.5).astype(int) - (self.z >= 0.5).astype(int)))
        self.assertAlmostEqual(curve[0.5]["point"], expected)
        self.assertAlmostEqual(sum(value["rate"] for value in states.values()), 1.0)

    def test_bootstrap_is_deterministic_for_frozen_seed(self) -> None:
        left = paired_bootstrap(self.full, self.z, [0.3, 0.5, 0.7], seed=20260820, resamples=250, batch=25)
        right = paired_bootstrap(self.full, self.z, [0.3, 0.5, 0.7], seed=20260820, resamples=250, batch=25)
        self.assertEqual(left, right)

    def test_bootstrap_child_seed_changes_interval_samples(self) -> None:
        full = np.linspace(0.2, 0.8, 53)
        z = full + np.sin(np.arange(53)) * 0.13
        left = paired_bootstrap(full, z, [0.4, 0.5, 0.6], seed=20260820, resamples=333, batch=37)
        right = paired_bootstrap(full, z, [0.4, 0.5, 0.6], seed=20260821, resamples=333, batch=37)
        self.assertNotEqual(left, right)

    def test_bootstrap_interval_contains_point_for_fixture(self) -> None:
        curve, _ = paired_bootstrap(self.full, self.z, [0.5], seed=3, resamples=1000, batch=50)
        row = curve[0.5]
        self.assertLessEqual(row["lower_95"], row["point"])
        self.assertGreaterEqual(row["upper_95"], row["point"])

    def test_iou_summary_uses_z_minus_full_direction(self) -> None:
        result = iou_summaries(self.full, self.z)
        self.assertEqual(result["direction"], "Z_M448_0_minus_Full")
        self.assertAlmostEqual(result["mean"], float(np.mean(self.z - self.full)))
        self.assertAlmostEqual(result["median"], float(np.median(self.z - self.full)))

    def test_iou_summary_tail_boundaries_are_frozen(self) -> None:
        full = np.asarray([0.5, 0.5, 0.5, 0.5])
        z = np.asarray([0.55, 0.40, 0.25, 0.61])
        result = iou_summaries(full, z)
        self.assertEqual(result["abs_delta_le_0_05"], 0.25)
        self.assertEqual(result["delta_lt_minus_0_10"], 0.25)
        self.assertEqual(result["delta_lt_minus_0_25"], 0.0)
        self.assertEqual(result["delta_gt_0_10"], 0.25)

    def test_threshold_grid_has_exact_frozen_values(self) -> None:
        grid = [round(0.30 + 0.05 * index, 2) for index in range(9)]
        self.assertEqual(grid, [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])

    def test_pairing_rejects_reordered_or_duplicate_ids(self) -> None:
        expected = ["a", "b", "c"]
        require_exact_pairing(expected, expected.copy(), expected, "fixture")
        with self.assertRaises(ErrorAnalysisFailure):
            require_exact_pairing(expected, ["a", "c", "b"], expected, "fixture")
        with self.assertRaises(ErrorAnalysisFailure):
            require_exact_pairing(["a", "a", "c"], expected, expected, "fixture")

    def test_pairing_rejects_missing_or_wrong_id(self) -> None:
        expected = ["a", "b", "c"]
        with self.assertRaises(ErrorAnalysisFailure):
            require_exact_pairing(["a", "b"], expected, expected, "fixture")
        with self.assertRaises(ErrorAnalysisFailure):
            require_exact_pairing(expected, ["a", "b", "x"], expected, "fixture")

    def test_hash_binding_rejects_tamper_and_size_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_bytes(b"one\n")
            expected = {"bytes": 4, "sha256": "2c8b08da5ce60398e1f19af0e5dccc744df274b826abe585eaba68c525434806"}
            verify_ledger_file(path, expected, "fixture")
            path.write_bytes(b"two\n")
            with self.assertRaises(ErrorAnalysisFailure):
                verify_ledger_file(path, expected, "fixture")

    def test_frozen_transition_mismatch_is_rejected(self) -> None:
        actual = {"both_correct": 2, "full_only": 1, "z_only": 2, "both_incorrect": 1}
        require_expected_transitions(actual, actual.copy(), "fixture")
        wrong = {**actual, "full_only": 2}
        with self.assertRaises(ErrorAnalysisFailure):
            require_expected_transitions(wrong, actual, "fixture")

    def test_output_inventory_rejects_extra_or_forbidden_analysis(self) -> None:
        require_allowed_output_inventory(ALLOWED_ANALYSIS_FILES)
        with self.assertRaises(ErrorAnalysisFailure):
            require_allowed_output_inventory(ALLOWED_ANALYSIS_FILES | {"semantic_subgroup.csv"})
        with self.assertRaises(ErrorAnalysisFailure):
            require_allowed_output_inventory(ALLOWED_ANALYSIS_FILES - {"ERROR_ANALYSIS_SUMMARIES.csv"})


if __name__ == "__main__":
    unittest.main()
