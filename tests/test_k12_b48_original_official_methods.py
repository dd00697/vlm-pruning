from __future__ import annotations

import unittest

from scripts.run_original_llava_official_method import expected_visual_trace, final_budget


class OriginalOfficialMethodTests(unittest.TestCase):
    def test_visionzip_is_exact_pre_llm_b48(self) -> None:
        self.assertEqual(expected_visual_trace("official_visionzip"), [48] * 32)
        self.assertEqual(final_budget("official_visionzip"), 48)

    def test_transprune_exact_schedule(self) -> None:
        self.assertEqual(
            expected_visual_trace("official_transprune"),
            [576] * 7 + [504] * 2 + [360] * 3 + [48] * 20,
        )
        self.assertEqual(final_budget("official_transprune"), 48)

    def test_official_full_parity_trace(self) -> None:
        self.assertEqual(expected_visual_trace("official_llava_full"), [576] * 32)
        self.assertEqual(final_budget("official_llava_full"), 576)


if __name__ == "__main__":
    unittest.main()
