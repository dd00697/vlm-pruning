from __future__ import annotations

import unittest

from scripts.run_adaptive_official_reroute import method_contract


class AdaptiveOfficialRerouteTests(unittest.TestCase):
    def test_pinned_avg192_contracts_have_exact_native_configs_and_realized_work(self) -> None:
        pdrop = method_contract("pdrop")
        reroute = method_contract("pdrop_reroute")
        self.assertEqual(pdrop["action"], "physical_delete")
        self.assertEqual(reroute["action"], "compact_route")
        self.assertEqual(pdrop["drop_layers"], [2, 7, 15, 23])
        self.assertEqual(reroute["drop_layers"], [2, 7, 15, 23])
        self.assertEqual(pdrop["budgets"], [343, 204, 122, 72])
        self.assertEqual(reroute["budgets"], [259, 259, 129, 64])
        self.assertAlmostEqual(sum(pdrop["realized_visual_tokens_by_layer"]) / 32, 191.34375)
        self.assertAlmostEqual(sum(reroute["realized_visual_tokens_by_layer"]) / 32, 191.46875)

    def test_full_contract_has_no_algorithmic_routing(self) -> None:
        full = method_contract("none")
        self.assertEqual(full["action"], "none")
        self.assertEqual(full["drop_layers"], [])
        self.assertEqual(full["realized_visual_tokens_by_layer"], [576] * 32)


if __name__ == "__main__":
    unittest.main()
