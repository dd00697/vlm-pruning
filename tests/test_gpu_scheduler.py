from __future__ import annotations

import unittest

from vlm_pruning.gpu_scheduler import admissible_widths


class GPUSchedulerTests(unittest.TestCase):
    def test_three_is_rejected_when_headroom_would_be_consumed(self) -> None:
        widths, headroom = admissible_widths(46068, 14500)
        self.assertEqual(widths, [2, 1])
        self.assertEqual(headroom, 4606.8)

    def test_three_is_allowed_only_when_measured_sum_fits(self) -> None:
        widths, _ = admissible_widths(46068, 12000)
        self.assertEqual(widths, [3, 2, 1])

    def test_missing_measurement_fails_closed_to_serial(self) -> None:
        self.assertEqual(admissible_widths(46068, None)[0], [1])


if __name__ == "__main__":
    unittest.main()
