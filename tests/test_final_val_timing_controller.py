from __future__ import annotations

import unittest

from scripts.run_final_val_timing_campaign import plans


class FinalValTimingControllerTests(unittest.TestCase):
    def test_canonical_natural_plan_is_exact_and_counterbalanced(self) -> None:
        jobs = plans("canonical", "natural")
        self.assertEqual(len(jobs), 25)
        self.assertEqual([job[2] for job in jobs[:5]], [
            "Full", "K12", "M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN"
        ])
        self.assertEqual([job[2] for job in jobs[-5:]], [
            "FEATHER_LLAVA_MAIN", "Full", "K12", "M448_OPT", "Z_M448_0"
        ])

    def test_native_plans_use_exact_full_pruned_alternation(self) -> None:
        fastv = plans("fastv", "natural")
        nuwa = plans("nuwa", "natural")
        self.assertEqual(len(fastv), 10)
        self.assertEqual(len(nuwa), 10)
        self.assertEqual([job[2] for job in fastv[:4]], [
            "FASTV_STACK_FULL", "FASTV_MATCHED_199",
            "FASTV_MATCHED_199", "FASTV_STACK_FULL",
        ])
        self.assertEqual([job[2] for job in nuwa[:4]], [
            "NUWA_STACK_FULL", "NUWA_NATIVE_192",
            "NUWA_NATIVE_192", "NUWA_STACK_FULL",
        ])

    def test_fixed32_plans_are_three_repetitions_only(self) -> None:
        self.assertEqual(len(plans("canonical", "fixed32")), 15)
        self.assertEqual(len(plans("fastv", "fixed32")), 6)
        self.assertEqual(len(plans("nuwa", "fixed32")), 6)


if __name__ == "__main__":
    unittest.main()
