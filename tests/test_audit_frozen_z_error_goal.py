from __future__ import annotations

import unittest

from scripts.audit_frozen_z_error_analysis_goal import PROTECTED
from scripts.frozen_z_error_analysis_common import TERMINAL_STATUS


class FrozenZErrorTerminalAuditTests(unittest.TestCase):
    def test_terminal_status_is_exact(self) -> None:
        self.assertEqual(TERMINAL_STATUS, "GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED")

    def test_protected_inventory_is_narrow(self) -> None:
        self.assertEqual(set(PROTECTED), {"PRO_REASONING_RESEARCH_HANDOFF.md", "legacy_goal2.md"})
        self.assertTrue(all(len(value) == 64 for value in PROTECTED.values()))


if __name__ == "__main__":
    unittest.main()
