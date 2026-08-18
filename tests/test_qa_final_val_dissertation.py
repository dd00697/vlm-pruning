from __future__ import annotations

import unittest

from scripts.qa_final_val_dissertation import parse_pages


class FinalValDissertationQATests(unittest.TestCase):
    def test_parse_pages_preserves_explicit_page_inventory(self) -> None:
        self.assertEqual(parse_pages("3,7,11"), [3, 7, 11])


if __name__ == "__main__":
    unittest.main()
