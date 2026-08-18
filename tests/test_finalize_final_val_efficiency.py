from __future__ import annotations

import unittest

from scripts.finalize_final_val_efficiency import markdown_table


class FinalValFinalizerTests(unittest.TestCase):
    def test_markdown_table_has_stable_header_separator_and_rows(self) -> None:
        table = markdown_table(["A", "B"], [["1", "2"], ["3", "4"]])
        self.assertEqual(table[0], "| A | B |")
        self.assertEqual(table[1], "|---|---|")
        self.assertEqual(table[-1], "| 3 | 4 |")


if __name__ == "__main__":
    unittest.main()
