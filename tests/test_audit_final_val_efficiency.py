from __future__ import annotations

import unittest

from scripts.audit_final_val_efficiency import expected_fixed_counts


class FinalValEfficiencyAuditTests(unittest.TestCase):
    def test_fixed_counts_include_only_triggered_stack(self) -> None:
        trigger = {
            "stacks": {
                "canonical": {"required": False},
                "fastv": {"required": True},
                "nuwa": {"required": False},
            }
        }
        self.assertEqual(
            expected_fixed_counts(trigger),
            {"FASTV_STACK_FULL": 3, "FASTV_MATCHED_199": 3},
        )

    def test_fixed_counts_cover_all_canonical_methods(self) -> None:
        trigger = {"stacks": {"canonical": {"required": True}}}
        self.assertEqual(
            set(expected_fixed_counts(trigger)),
            {"Full", "K12", "M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN"},
        )


if __name__ == "__main__":
    unittest.main()
