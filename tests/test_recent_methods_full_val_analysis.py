from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path

from scripts.analyse_recent_methods_full_val import (
    box_iou,
    campaign_classification,
    historical_parse,
    interval_direction,
    pareto_membership,
    strict_parse,
    immutable_context,
)


class RecentMethodsFullValAnalysisTests(unittest.TestCase):
    def test_strict_parser_accepts_registered_normalized_box(self) -> None:
        predicted, raw, failure = strict_parse("[0.1, 0.2, 0.8, 0.9]", 640, 480)
        self.assertEqual(raw, [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(predicted, [64.0, 96.0, 512.0, 432.0])
        self.assertIsNone(failure)

    def test_strict_parser_reproduces_clipping_and_coordinate_order(self) -> None:
        self.assertEqual(
            strict_parse("[-0.1, 0.2, 0.8, 0.9]", 640, 480),
            ([0.0, 96.0, 512.0, 432.0], [-0.1, 0.2, 0.8, 0.9], None),
        )
        self.assertEqual(
            strict_parse("[0.8, 0.2, 0.1, 0.9]", 640, 480)[2],
            "invalid_coordinate_order",
        )

    def test_historical_parser_reproduces_clipping_contract(self) -> None:
        raw, predicted, failure = historical_parse("box=[-0.1, 0.2, 1.1, 0.9]", 100, 50)
        self.assertEqual(raw, [-0.1, 0.2, 1.1, 0.9])
        self.assertEqual(predicted, [0.0, 10.0, 100.0, 45.0])
        self.assertIsNone(failure)

    def test_box_iou(self) -> None:
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 5, 15, 15]), 25 / 175)

    def test_pareto_membership_uses_lower_work_and_higher_quality(self) -> None:
        methods = [
            {"method": "a", "token_layer_work": 100, "acc_at_0_5": 0.4, "mean_iou": 0.4},
            {"method": "b", "token_layer_work": 80, "acc_at_0_5": 0.4, "mean_iou": 0.39},
            {"method": "c", "token_layer_work": 120, "acc_at_0_5": 0.39, "mean_iou": 0.39},
        ]
        result = {row["method"]: row for row in pareto_membership(methods)}
        self.assertFalse(result["a"]["pareto_acc_at_0_5"])
        self.assertTrue(result["b"]["pareto_acc_at_0_5"])
        self.assertFalse(result["c"]["pareto_acc_at_0_5"])
        self.assertTrue(result["a"]["pareto_joint_quality_work"])
        self.assertTrue(result["b"]["pareto_joint_quality_work"])
        self.assertFalse(result["c"]["pareto_joint_quality_work"])

    def test_interval_direction_does_not_call_zero_inclusion_equivalence(self) -> None:
        self.assertEqual(interval_direction({"lower_95": 0.1, "upper_95": 0.4}), "reliable_positive")
        self.assertEqual(interval_direction({"lower_95": -0.4, "upper_95": -0.1}), "reliable_negative")
        self.assertEqual(
            interval_direction({"lower_95": -0.1, "upper_95": 0.2}),
            "inconclusive_includes_zero",
        )

    def test_campaign_classification_forbids_automatic_final_selection(self) -> None:
        methods = [
            {"method": "V5_Z_M448_0", "pareto_joint_quality_work": True},
            {"method": "V4_M448", "pareto_joint_quality_work": False},
        ]
        pairs = (
            ("V1_K12_48_K20_6", "V2_Z_K12_0"),
            ("V4_M448", "V3_C448"),
            ("V5_Z_M448_0", "V4_M448"),
            ("V5_Z_M448_0", "K12"),
        )
        intervals = []
        for left, right in pairs:
            for metric in ("acc_at_0_5_pp", "mean_iou"):
                intervals.append(
                    {
                        "contrast": f"{left}-minus-{right}",
                        "metric": metric,
                        "point_delta": 0.1,
                        "lower_95": -0.1,
                        "upper_95": 0.2,
                    }
                )
        result = campaign_classification(methods, intervals)
        self.assertEqual(result["selection_status"], "NO_NEW_FINAL_METHOD_SELECTED_HUMAN_REVIEW_REQUIRED")
        self.assertEqual(result["joint_quality_work_frontier"], ["V5_Z_M448_0"])

    def test_immutable_context_reads_frozen_router_and_timing_evidence(self) -> None:
        context = immutable_context()
        self.assertEqual(context["router"]["range"], [0.976, 0.993])
        self.assertAlmostEqual(
            context["existing_matched_timing"]["M448_OPT"]["end_to_end_latency_s"],
            0.8222054659146816,
        )

    def test_generated_tex_contains_required_tables_and_labels(self) -> None:
        path = Path("/workspace/Dissertation_Draft/analysis/build_recent_methods_full_val.py")
        spec = importlib.util.spec_from_file_location("recent_assets_for_test", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        methods = []
        for index, name in enumerate(module.TABLE_ORDER):
            methods.append(
                {
                    "method": name,
                    "display": name,
                    "acc_at_0_5": 0.45 - index / 1000,
                    "mean_iou": 0.43 - index / 1000,
                    "final_visual_tokens": 48,
                    "average_visual_tokens": 200 + index,
                    "token_layer_work": 6400 + index * 100,
                    "work_reduction_vs_full": 0.5,
                }
            )
        intervals = []
        for left, right, _ in module.IMPORTANT_CONTRASTS:
            for metric in ("acc_at_0_5_pp", "mean_iou"):
                intervals.append(
                    {
                        "contrast": f"{left}-minus-{right}",
                        "metric": metric,
                        "point_delta": 0.1,
                        "lower_95": -0.2,
                        "upper_95": 0.3,
                    }
                )
        text = module.generated_tex(methods, intervals)
        self.assertIn(r"\label{tab:recent-methods-full-val}", text)
        self.assertIn(r"\label{tab:recent-methods-full-val-paired}", text)
        self.assertIn("Six tail minus zero tail", text)


if __name__ == "__main__":
    unittest.main()
