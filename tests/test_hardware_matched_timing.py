from __future__ import annotations

import unittest
from unittest import mock

import pytest

pytest.importorskip("torch")

from scripts import analyse_hardware_matched_timing as analysis
from scripts.run_hardware_matched_timing_trial import (
    FIXED32_ORDER_PLAN,
    FIXED_CONTINUATION_TOKEN_IDS,
    METHODS,
    ORDER_PLAN,
    HardwareTimingSelectionMixin,
)
from scripts.validate_hardware_matched_timing_trial import (
    HardwareTimingValidationError,
    validate_stage_traces,
)
from vlm_pruning.llava_hooks import PruningEvent


class HardwareMatchedTimingTests(unittest.TestCase):
    def test_registered_orders_are_balanced_and_fixed_inputs_are_exact(self) -> None:
        positions = {method: [] for method in METHODS}
        for order in ORDER_PLAN.values():
            self.assertEqual(set(order), set(METHODS))
            for position, method in enumerate(order, 1):
                positions[method].append(position)
        for observed in positions.values():
            self.assertLessEqual(max(observed.count(position) for position in range(1, 5)), 2)
            self.assertTrue(set(range(1, 5)).issubset(observed))
        for order in FIXED32_ORDER_PLAN.values():
            self.assertEqual(len(order), len(set(order)))
        self.assertEqual(len(FIXED_CONTINUATION_TOKEN_IDS), 32)
        self.assertEqual(FIXED_CONTINUATION_TOKEN_IDS, tuple(range(1000, 1032)))

    def test_zero_tail_selector_performs_no_scoring(self) -> None:
        forwarder = object.__new__(HardwareTimingSelectionMixin)
        forwarder.timing_stage_components = []
        forwarder._validate_original_indices = mock.Mock(return_value=[])
        result = forwarder.select_stage_indices(
            None,
            input_ids=None,
            original_positions=None,
            incoming_original_indices=list(range(48)),
            event=PruningEvent(20, 0, "none"),
            stage_index=1,
            sample_id="sample",
            seed=0,
            query_original_positions=[],
        )
        self.assertEqual(result, ([], None, False, None))
        self.assertEqual(
            forwarder.timing_stage_components,
            [{"query_scoring_latency_s": 0.0, "selector_latency_s": 0.0}],
        )

    def test_validator_rejects_scored_zero_tail(self) -> None:
        first = {
            "stage_index": 0,
            "boundary": 12,
            "selector": "k12_original",
            "requested_budget": 48,
            "realized_budget": 48,
            "incoming_original_indices": list(range(576)),
            "outgoing_original_indices": list(range(48)),
        }
        zero = {
            "stage_index": 1,
            "boundary": 20,
            "selector": "none",
            "requested_budget": 0,
            "realized_budget": 0,
            "incoming_original_indices": list(range(48)),
            "outgoing_original_indices": [],
            "score_map_checksum": "not-allowed",
            "score_map_recomputed": True,
        }
        with self.assertRaises(HardwareTimingValidationError):
            validate_stage_traces({"stage_traces": [first, zero]}, "Z_K12_0", 0)

    def test_hierarchical_interval_is_paired_and_deterministic(self) -> None:
        left = [
            [{"sample_id": "a", "x": 2.0}, {"sample_id": "b", "x": 4.0}],
            [{"sample_id": "a", "x": 3.0}, {"sample_id": "b", "x": 5.0}],
        ]
        right = [
            [{"sample_id": "a", "x": 1.0}, {"sample_id": "b", "x": 3.0}],
            [{"sample_id": "a", "x": 2.0}, {"sample_id": "b", "x": 4.0}],
        ]
        with mock.patch.object(analysis, "BOOTSTRAP_RESAMPLES", 200):
            first = analysis.hierarchical_paired_interval(left, right, "x", seed=7)
            second = analysis.hierarchical_paired_interval(left, right, "x", seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["point_delta"], 1.0)
        self.assertEqual(first["lower_95"], 1.0)
        self.assertEqual(first["upper_95"], 1.0)


if __name__ == "__main__":
    unittest.main()
