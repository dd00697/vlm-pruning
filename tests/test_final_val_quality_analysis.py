from __future__ import annotations

import unittest

from scripts.analyse_final_val_quality import (
    EXPECTED_ROWS,
    QualityAnalysisError,
    box_iou,
    validate_trace,
)


class FinalValQualityAnalysisTests(unittest.TestCase):
    def test_box_iou_handles_identical_and_disjoint_boxes(self) -> None:
        self.assertEqual(box_iou([0, 0, 2, 2], [0, 0, 2, 2]), 1.0)
        self.assertEqual(box_iou([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)

    def test_fixed_trace_accounting_is_exact(self) -> None:
        trace = [576] * 3 + [160] * 29
        result = validate_trace("FASTV_MATCHED_199", [{}] * EXPECTED_ROWS, trace)
        self.assertEqual(result["mean_token_layer_work"], 6368)
        self.assertEqual(result["mean_average_visual_tokens"], 199)

    def test_feather_trace_must_have_exact_boundaries_and_monotonic_reduction(self) -> None:
        good = [576] * 8 + [217] * 8 + [19] * 16
        rows = [{"realized_visual_tokens_by_layer": good}] * EXPECTED_ROWS
        result = validate_trace("FEATHER_LLAVA_MAIN", rows, None)
        self.assertEqual(result["mean_token_layer_work"], sum(good))
        bad = list(good)
        bad[7] = 575
        with self.assertRaises(QualityAnalysisError):
            validate_trace(
                "FEATHER_LLAVA_MAIN",
                [{"realized_visual_tokens_by_layer": bad}] * EXPECTED_ROWS,
                None,
            )


if __name__ == "__main__":
    unittest.main()
