from __future__ import annotations

import unittest

import pytest

pytest.importorskip("torch")

from scripts.run_final_val_fastv_timing import METHODS, ORDER_PLAN


class FinalFastVTimingTests(unittest.TestCase):
    def test_alternating_order(self) -> None:
        self.assertEqual(ORDER_PLAN[1], METHODS)
        self.assertEqual(ORDER_PLAN[2], tuple(reversed(METHODS)))
        self.assertEqual(ORDER_PLAN[4], tuple(reversed(METHODS)))
        for order in ORDER_PLAN.values():
            self.assertEqual(set(order), set(METHODS))


if __name__ == "__main__":
    unittest.main()
