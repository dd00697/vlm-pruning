import unittest

from scripts.init_final_val_efficiency_campaign import (
    FASTV_TIMING_ORDER,
    INTERNAL,
    INTERNAL_TIMING_ORDER,
)


class FinalEfficiencyProtocolTest(unittest.TestCase):
    def test_known_internal_accounting(self):
        for method, record in INTERNAL.items():
            with self.subTest(method=method):
                self.assertEqual(len(record["trace"]), 32)
                self.assertEqual(sum(record["trace"]), record["work"])
                self.assertEqual(record["work"] / 32, record["average"])
                self.assertEqual(record["trace"][-1], record["final"])

    def test_counterbalanced_orders_are_complete(self):
        expected = {"Full", "K12", "M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN"}
        self.assertEqual(set(INTERNAL_TIMING_ORDER), {"1", "2", "3", "4", "5"})
        for order in INTERNAL_TIMING_ORDER.values():
            self.assertEqual(set(order), expected)
            self.assertEqual(len(order), len(expected))
        for order in FASTV_TIMING_ORDER.values():
            self.assertEqual(set(order), {"FASTV_STACK_FULL", "FASTV_MATCHED_199"})


if __name__ == "__main__":
    unittest.main()
