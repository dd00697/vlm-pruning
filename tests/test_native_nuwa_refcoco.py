from __future__ import annotations

import unittest

from scripts.run_native_nuwa_refcoco import (
    FULL_TRACE,
    NUWA_TRACE,
    NativeNuwaError,
    expected_trace,
    validate_visual_trace,
)


class NativeNuwaContractTests(unittest.TestCase):
    def test_registered_traces(self) -> None:
        self.assertEqual(expected_trace("NUWA_STACK_FULL"), [576] * 32)
        self.assertEqual(expected_trace("NUWA_NATIVE_192"), [336] * 16 + [48] * 16)
        self.assertEqual(sum(NUWA_TRACE), 6144)
        self.assertEqual(sum(FULL_TRACE), 18432)

    def test_trace_validator_fails_closed(self) -> None:
        with self.assertRaises(NativeNuwaError):
            validate_visual_trace("NUWA_NATIVE_192", [336] * 15 + [48] * 17)
        with self.assertRaises(NativeNuwaError):
            expected_trace("unregistered")


if __name__ == "__main__":
    unittest.main()
