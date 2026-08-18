from __future__ import annotations

import unittest

from scripts.build_final_val_accounting import (
    clip_vit_l_14_336_flops,
    feather_score_flops,
    m448_merge_flops,
    projector_flops,
    query_similarity_flops,
)


class FinalValAccountingTests(unittest.TestCase):
    def test_vision_and_projector_formulas_are_positive_and_token_linear(self) -> None:
        self.assertGreater(clip_vit_l_14_336_flops(), 0)
        self.assertEqual(projector_flops(576), 2 * projector_flops(288))

    def test_query_similarity_formula_matches_hand_calculation(self) -> None:
        # normalization=4*(3+2)*5=100, similarities=2*3*2*5=60, zscore=12.
        self.assertEqual(query_similarity_flops(3, 2, hidden=5), 172)

    def test_merge_separates_raster_distance_and_weighted_reduction(self) -> None:
        distance, merge = m448_merge_flops(incoming=3, retained=2, hidden=5)
        self.assertEqual(distance, 30)
        self.assertEqual(merge, 40)

    def test_feather_score_counts_extra_projection_work(self) -> None:
        self.assertGreater(feather_score_flops(10, 4), 4 * 10 * 4096 * 4096)


if __name__ == "__main__":
    unittest.main()
