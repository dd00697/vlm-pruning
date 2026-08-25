from __future__ import annotations

import unittest

from scripts.run_ivc_prune_campaign import plans
from vlm_pruning.ivc_campaign import (
    FIXED_CONTINUATION_TOKEN_IDS,
    IVC_PRUNE_LAYER_INDEX,
    METHODS,
    IVCCampaignError,
    expected_ivc_config,
    paired_bootstrap,
    retained_visual_indices,
    summarize_quality,
    timing_order,
    validate_visual_trace,
    visual_trace,
)


class IVCCampaignContractTests(unittest.TestCase):
    def test_author_configuration_and_boundary_are_frozen(self) -> None:
        self.assertIsNone(expected_ivc_config("IVC_STACK_FULL"))
        self.assertEqual(
            expected_ivc_config("IVC_PRUNE"),
            {"ivcp_k": 16, "image_token_start_index": 5, "image_token_length": 576},
        )
        self.assertEqual(IVC_PRUNE_LAYER_INDEX, 16)
        self.assertEqual(FIXED_CONTINUATION_TOKEN_IDS, tuple(range(1000, 1032)))
        with self.assertRaises(IVCCampaignError):
            expected_ivc_config("unregistered")

    def test_adaptive_trace_validation_fails_closed(self) -> None:
        full = visual_trace("IVC_STACK_FULL")
        pruned = visual_trace("IVC_PRUNE", 161)
        self.assertEqual(full, [576] * 32)
        self.assertEqual(pruned, [576] * 16 + [161] * 16)
        self.assertEqual(validate_visual_trace("IVC_PRUNE", pruned), 161)
        with self.assertRaises(IVCCampaignError):
            validate_visual_trace("IVC_PRUNE", [576] * 15 + [161] * 17)
        with self.assertRaises(IVCCampaignError):
            validate_visual_trace("IVC_STACK_FULL", [575] * 32)

    def test_keep_indices_are_reduced_to_visual_patch_indices(self) -> None:
        full_sequence = [0, 1, 2, 3, 4, 5, 8, 580, 581, 900]
        self.assertEqual(retained_visual_indices(full_sequence), [0, 3, 575])
        with self.assertRaises(IVCCampaignError):
            retained_visual_indices([0, 5, 5, 6])
        with self.assertRaises(IVCCampaignError):
            retained_visual_indices([0, 1, 2, 3, 4, 581])

    def test_timing_plans_are_paired_and_counterbalanced(self) -> None:
        self.assertEqual(timing_order("natural", 1), METHODS)
        self.assertEqual(timing_order("natural", 2), tuple(reversed(METHODS)))
        natural = plans("timing-natural")
        fixed = plans("timing-fixed32")
        self.assertEqual(len(natural), 10)
        self.assertEqual(len(fixed), 6)
        self.assertEqual([row["method"] for row in natural[:4]], [
            "IVC_STACK_FULL", "IVC_PRUNE", "IVC_PRUNE", "IVC_STACK_FULL"
        ])

    def test_quality_summary_and_bootstrap_are_deterministic(self) -> None:
        rows = [
            {
                "box_iou": 0.75,
                "valid": True,
                "success_iou_0_5": True,
                "visual_token_layer_work": 11792,
                "final_visual_tokens": 161,
            },
            {
                "box_iou": 0.25,
                "valid": True,
                "success_iou_0_5": False,
                "visual_token_layer_work": 11600,
                "final_visual_tokens": 149,
            },
        ]
        summary = summarize_quality(rows)
        self.assertEqual(summary["acc_at_0_5"], 0.5)
        self.assertEqual(summary["mean_iou"], 0.5)
        self.assertEqual(summary["mean_final_visual_tokens"], 155.0)
        first = paired_bootstrap([1, 0, 1], [0, 0, 1], resamples=200, seed=7)
        second = paired_bootstrap([1, 0, 1], [0, 0, 1], resamples=200, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["point_delta"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
