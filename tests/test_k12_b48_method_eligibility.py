from __future__ import annotations

import unittest

from vlm_pruning.final_campaign import METHODS


class MethodEligibilityTests(unittest.TestCase):
    def test_preclassification_has_exactly_five_unique_official_methods(self) -> None:
        self.assertEqual(len(METHODS), 5)
        self.assertEqual(len({method["key"] for method in METHODS}), 5)

    def test_fixed_exclusions_do_not_request_non_b48_runs(self) -> None:
        by_key = {method["key"]: method for method in METHODS}
        self.assertEqual(by_key["nuwa"]["status"], "excluded_no_exact_b48_support")
        self.assertEqual(by_key["feather"]["status"], "excluded_checkpoint_or_parity_mismatch")


if __name__ == "__main__":
    unittest.main()
