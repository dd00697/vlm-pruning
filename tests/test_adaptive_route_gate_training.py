from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.analyse_adaptive_route_gate import pareto_frontier, select_project_candidate
from scripts.analyse_adaptive_matched_timing import fixed_trigger, timing_gate
from scripts.build_adaptive_gate_controls import aggregate as aggregate_control
from scripts.run_adaptive_matched_timing_campaign import FIXED_ORDERS, NATURAL_ORDERS
from scripts.validate_adaptive_route_gate import aggregate, route_config
from vlm_pruning.llava_hooks.progressive_pruning import PruningEvent
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

    def test_dynamic_validator_maps_only_the_two_frozen_routes(self) -> None:
        safe = route_config("K12_QUERY_TOPK", 2)
        aggressive = route_config("M448", 1)
        self.assertEqual(safe["schedule"], [PruningEvent(12, 48, "query_topk").as_dict()])
        self.assertEqual(
            aggressive["schedule"],
            [
                PruningEvent(4, 448, "query_topk_merge").as_dict(),
                PruningEvent(12, 48, "query_topk").as_dict(),
            ],
        )

    def test_dynamic_aggregate_counts_routes_and_sequence_work(self) -> None:
        common = {
            "success_iou_0_5": True,
            "box_iou": 0.75,
            "valid": True,
            "gate/latency_s": 0.02,
            "gate_model_latency_s": 0.003,
            "peak_allocated_mb": 12.0,
            "peak_reserved_mb": 14.0,
        }
        rows = [
            {
                **common,
                "gate_route_choice": "K12_QUERY_TOPK",
                "candidate_average_visual_tokens": 246.0,
                "realized_visual_tokens_by_layer": [576] * 12 + [48] * 20,
            },
            {
                **common,
                "gate_route_choice": "M448",
                "candidate_average_visual_tokens": 214.0,
                "realized_visual_tokens_by_layer": [576] * 4 + [448] * 8 + [48] * 20,
            },
        ]
        metrics = aggregate(rows)
        self.assertEqual(metrics["gate/safe_rows"], 1)
        self.assertEqual(metrics["gate/aggressive_rows"], 1)
        self.assertEqual(metrics["tokens/candidate_average_visual_tokens"], 230.0)
        self.assertEqual(metrics["tokens/visual_token_layer_work_mean"], (7872 + 6848) / 2)

    def test_project_ranking_applies_within_point005_compute_tiebreak(self) -> None:
        registry = {
            "cells": {
                "C448": {"status": "accepted", "metrics": {"acc_at_0_5": 0.415, "mean_iou": 0.426, "average_visual_tokens": 214}},
                "C480": {"status": "accepted", "metrics": {"acc_at_0_5": 0.424, "mean_iou": 0.427, "average_visual_tokens": 222}},
                "D1": {"status": "accepted", "metrics": {"acc_at_0_5": 0.410, "mean_iou": 0.427, "average_visual_tokens": 222}},
                "M448": {"status": "accepted", "metrics": {"acc_at_0_5": 0.430, "mean_iou": 0.4285, "average_visual_tokens": 214}},
            }
        }
        gate = {
            "metrics": {
                "acc_at_0_5": 0.434,
                "mean_iou": 0.430,
                "average_visual_tokens": 220.0,
                "visual_token_layer_work": 7040.0,
            },
            "k12_like": True,
        }
        self.assertEqual(select_project_candidate(registry, gate)["selected"], "M448")
        gate["metrics"]["acc_at_0_5"] = 0.436
        self.assertEqual(select_project_candidate(registry, gate)["selected"], "GATE_SEED0")

    def test_pareto_frontier_uses_both_quality_and_compute_axes(self) -> None:
        points = [
            {
                "candidate": "dominated",
                "acc_at_0_5": 0.40,
                "mean_iou": 0.40,
                "average_visual_tokens": 220.0,
                "visual_token_layer_work": 7000.0,
            },
            {
                "candidate": "quality",
                "acc_at_0_5": 0.43,
                "mean_iou": 0.43,
                "average_visual_tokens": 220.0,
                "visual_token_layer_work": 7000.0,
            },
            {
                "candidate": "compute",
                "acc_at_0_5": 0.39,
                "mean_iou": 0.39,
                "average_visual_tokens": 190.0,
                "visual_token_layer_work": 6100.0,
            },
        ]
        self.assertEqual(pareto_frontier(points), ["compute", "quality"])

    def test_privileged_control_suppresses_efficiency_claim(self) -> None:
        rows = [
            {
                "route_choice": "M448",
                "success_iou_0_5": True,
                "box_iou": 0.8,
                "valid": True,
                "candidate_average_visual_tokens": 214.0,
            }
        ]
        metrics = aggregate_control(rows, privileged=True)
        self.assertIsNone(metrics["tokens/analytical_average_visual_tokens"])
        self.assertFalse(metrics["efficiency_claim"])

    def test_matched_timing_orders_and_triggers_are_exact(self) -> None:
        self.assertEqual(
            NATURAL_ORDERS,
            (
                ("full", "k12_original", "candidate"),
                ("k12_original", "candidate", "full"),
                ("candidate", "full", "k12_original"),
                ("full", "candidate", "k12_original"),
                ("k12_original", "full", "candidate"),
            ),
        )
        self.assertEqual(FIXED_ORDERS, NATURAL_ORDERS[:3])
        natural = {
            "methods": {
                "full": {
                    "generated_token_count": {"mean": 20.0},
                    "end_to_end_latency_s": {"mean": 1.0},
                },
                "k12_original": {
                    "generated_token_count": {"mean": 20.1},
                    "main_prefill_with_overhead_latency_s": {"mean": 1.0},
                    "end_to_end_latency_s": {"mean": 1.0},
                    "prompt_kv_total_bytes": {"mean": 100.0},
                },
                "candidate": {
                    "generated_token_count": {"mean": 20.4},
                    "main_prefill_with_overhead_latency_s": {"mean": 0.94},
                    "end_to_end_latency_s": {"mean": 0.97},
                    "prompt_kv_total_bytes": {"mean": 90.0},
                },
            },
            "hierarchical_paired_contrasts": {
                "candidate-minus-k12_original": {
                    "main_prefill_with_overhead_latency_s": {"upper_95": -0.01},
                    "end_to_end_latency_s": {"upper_95": -0.01},
                }
            },
        }
        self.assertTrue(fixed_trigger(natural)["triggered"])
        self.assertTrue(timing_gate(natural)["passed"])


if __name__ == "__main__":
    unittest.main()
