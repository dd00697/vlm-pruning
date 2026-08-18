from __future__ import annotations

import unittest

from scripts.audit_final_val_parity import maximum_box_delta


class FinalValidationParityAuditTests(unittest.TestCase):
    def test_maximum_box_delta(self) -> None:
        self.assertEqual(maximum_box_delta([0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 2.0, 5.0]), 2.0)
        self.assertIsNone(maximum_box_delta(None, [0.0, 1.0, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
