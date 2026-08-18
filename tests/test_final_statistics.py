from __future__ import annotations

import unittest

from vlm_pruning.final_statistics import (
    hierarchical_paired_bootstrap_interval,
    paired_bootstrap_interval,
    pareto_frontier,
    select_internal_candidate,
)


class FinalStatisticsTests(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic_and_directional(self) -> None:
        first = paired_bootstrap_interval([1, 2, 3], [0, 1, 2], resamples=200, seed=7)
        second = paired_bootstrap_interval([1, 2, 3], [0, 1, 2], resamples=200, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["point_delta"], 1.0)
        self.assertGreater(first["lower_95"], 0.0)

    def test_hierarchical_bootstrap_retains_both_pairing_levels(self) -> None:
        result = hierarchical_paired_bootstrap_interval(
            [[2, 3], [4, 5]], [[1, 2], [3, 4]], resamples=200, seed=9
        )
        self.assertEqual(result["point_delta"], 1.0)
        self.assertEqual(result["paired_repetitions"], 2)
        self.assertEqual(result["paired_rows_per_repetition"], 2)

    def test_pareto_uses_higher_quality_and_lower_cost(self) -> None:
        records = [
            {"method": "a", "quality": 0.8, "cost": 2},
            {"method": "b", "quality": 0.7, "cost": 3},
            {"method": "c", "quality": 0.6, "cost": 1},
        ]
        self.assertEqual(
            pareto_frontier(records, quality_field="quality", cost_field="cost"), ["a", "c"]
        )

    def test_candidate_rule_selects_lowest_work_that_passes_both_gates(self) -> None:
        quality = {
            "Full": {"parse_rate": 1, "acc_at_0_5": 0.45, "mean_iou": 0.435},
            "K12": {"parse_rate": 1, "acc_at_0_5": 0.422, "mean_iou": 0.423},
            "M448_OPT": {"parse_rate": 1, "acc_at_0_5": 0.421, "mean_iou": 0.429},
            "Z_M448_0": {"parse_rate": 1, "acc_at_0_5": 0.423, "mean_iou": 0.429},
        }
        timing = {
            "Full": {"end_to_end_latency_s": 1.0},
            "K12": {"end_to_end_latency_s": 0.8, "delta_vs_full_upper_95_s": -0.1},
            "M448_OPT": {"end_to_end_latency_s": 0.9, "delta_vs_full_upper_95_s": -0.01},
            "Z_M448_0": {"end_to_end_latency_s": 1.1, "delta_vs_full_upper_95_s": 0.2},
        }
        work = {"K12": 7872, "M448_OPT": 6848, "Z_M448_0": 6272}
        decision = select_internal_candidate(quality, timing, work)
        self.assertEqual(decision["selected_method"], "M448_OPT")
        self.assertFalse(decision["method_decisions"]["Z_M448_0"]["acceleration_gate_pass"])


if __name__ == "__main__":
    unittest.main()
