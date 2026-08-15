from __future__ import annotations

import unittest

import numpy as np
import torch

from vlm_pruning.llava_hooks.route_gate import FEATURE_DIMENSION, FEATURE_SCHEMA_SHA256
from vlm_pruning.llava_hooks.route_gate_model import (
    TRAINING_SPEC,
    TRAINING_SPEC_SHA256,
    build_gate_model,
    calibrate_threshold,
    canonical_checksum,
    normalization_sha256,
)


class AdaptiveRouteGateTrainingTests(unittest.TestCase):
    def test_frozen_architecture_has_registered_parameter_count(self) -> None:
        model = build_gate_model(torch)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 525570)
        self.assertLessEqual(525570, 1_000_000)
        outputs = model(torch.zeros((2, FEATURE_DIMENSION), dtype=torch.float32))
        self.assertEqual(tuple(outputs.shape), (2, 2))
        self.assertEqual(TRAINING_SPEC["feature_schema_sha256"], FEATURE_SCHEMA_SHA256)
        self.assertEqual(TRAINING_SPEC_SHA256, canonical_checksum(TRAINING_SPEC))

    def test_normalization_hash_depends_on_schema_and_exact_float32_values(self) -> None:
        mean = np.zeros(FEATURE_DIMENSION, dtype=np.float32)
        standard_deviation = np.ones(FEATURE_DIMENSION, dtype=np.float32)
        first = normalization_sha256(mean, standard_deviation)
        mean[17] = np.float32(0.25)
        second = normalization_sha256(mean, standard_deviation)
        self.assertNotEqual(first, second)
        self.assertEqual(second, normalization_sha256(mean.astype(np.float64), standard_deviation))

    def test_threshold_maximizes_aggressive_rate_under_both_constraints(self) -> None:
        probabilities = np.linspace(0.001, 0.999, 600).tolist()
        safe_iou = [0.6] * 600
        aggressive_iou = [0.6] * 596 + [0.0] * 4
        valid = [True] * 600
        result = calibrate_threshold(probabilities, safe_iou, aggressive_iou, valid, valid)
        self.assertEqual(result["status"], "calibrated")
        self.assertEqual(result["chosen_metrics"]["aggressive_rows"], 599)
        self.assertAlmostEqual(result["chosen_metrics"]["acc_loss_vs_always_safe"], 0.005)
        self.assertLessEqual(result["chosen_metrics"]["mean_iou_loss_vs_always_safe"], 0.005)

    def test_tied_unsafe_probabilities_trigger_always_safe_fallback(self) -> None:
        probabilities = [0.5] * 600
        safe_iou = [1.0] * 600
        aggressive_iou = [0.0] * 600
        valid = [True] * 600
        result = calibrate_threshold(probabilities, safe_iou, aggressive_iou, valid, valid)
        self.assertEqual(result["status"], "always_safe_fallback")
        self.assertEqual(result["threshold"], 0.0)
        self.assertEqual(result["chosen_metrics"]["aggressive_rows"], 0)


if __name__ == "__main__":
    unittest.main()
