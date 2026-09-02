from __future__ import annotations

import unittest

import pytest

pytest.importorskip("torch")

from scripts.run_final_val_canonical_timing import METHODS, ORDER_PLAN, timing_row


class FinalCanonicalTimingTests(unittest.TestCase):
    def test_order_is_counterbalanced(self) -> None:
        self.assertEqual(set(ORDER_PLAN), {1, 2, 3, 4, 5})
        for order in ORDER_PLAN.values():
            self.assertEqual(set(order), set(METHODS))
            self.assertEqual(len(order), 5)
        for position in range(5):
            self.assertEqual({ORDER_PLAN[rep][position] for rep in ORDER_PLAN}, set(METHODS))

    def test_timing_row_rejects_non_physical_trace(self) -> None:
        with self.assertRaises(Exception):
            timing_row(
                {"sample_id": "x"}, {"method": "Full", "repetition": 1, "order_position": 1, "mode": "natural"},
                raw="", generated_count=1, prep={"vision_encoder_latency_s": 0, "projector_latency_s": 0},
                scoring=0, selector=0, gather=0, merge=0, prefill=1, decode=1, end_to_end=2,
                prompt_kv=1, final_kv=1, trace=[576] * 31, peak_allocated=1, peak_reserved=1,
            )


if __name__ == "__main__":
    unittest.main()
