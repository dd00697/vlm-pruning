from __future__ import annotations

import unittest

from scripts.run_native_nuwa_refcoco import (
    FIXED32_ORDER_PLAN,
    FIXED_CONTINUATION_TOKEN_IDS,
    FULL_TRACE,
    NATURAL_ORDER_PLAN,
    NUWA_TRACE,
    NativeNuwaError,
    expected_trace,
    stage1_selector_latency,
    validate_visual_trace,
)


class NativeNuwaContractTests(unittest.TestCase):
    def test_registered_traces(self) -> None:
        self.assertEqual(expected_trace("NUWA_STACK_FULL"), [576] * 32)
        self.assertEqual(expected_trace("NUWA_NATIVE_192"), [336] * 16 + [48] * 16)
        self.assertEqual(sum(NUWA_TRACE), 6144)
        self.assertEqual(sum(FULL_TRACE), 18432)

    def test_registered_timing_orders_and_fixed_continuation(self) -> None:
        self.assertEqual(FIXED_CONTINUATION_TOKEN_IDS, tuple(range(1000, 1032)))
        self.assertEqual(len(NATURAL_ORDER_PLAN), 5)
        self.assertEqual(len(FIXED32_ORDER_PLAN), 3)
        self.assertEqual(
            NATURAL_ORDER_PLAN[2], ("NUWA_NATIVE_192", "NUWA_STACK_FULL")
        )
        self.assertEqual(
            FIXED32_ORDER_PLAN[3], ("NUWA_STACK_FULL", "NUWA_NATIVE_192")
        )

    def test_trace_validator_fails_closed(self) -> None:
        with self.assertRaises(NativeNuwaError):
            validate_visual_trace("NUWA_NATIVE_192", [336] * 15 + [48] * 17)
        with self.assertRaises(NativeNuwaError):
            expected_trace("unregistered")

    def test_stage1_selector_is_only_attributed_to_native_nuwa(self) -> None:
        self.assertEqual(stage1_selector_latency("NUWA_STACK_FULL", 0.3, 0.2), 0.0)
        self.assertAlmostEqual(stage1_selector_latency("NUWA_NATIVE_192", 0.3, 0.2), 0.1)
        self.assertEqual(stage1_selector_latency("NUWA_NATIVE_192", 0.1, 0.2), 0.0)
        with self.assertRaises(NativeNuwaError):
            stage1_selector_latency("unregistered", 0.3, 0.2)


if __name__ == "__main__":
    unittest.main()
