from __future__ import annotations

import unittest

from scripts.build_final_val_dissertation_assets import table, tex


class FinalValDissertationAssetTests(unittest.TestCase):
    def test_tex_escapes_method_identifiers_and_percent(self) -> None:
        self.assertEqual(tex("Z_M448_0 95%"), "Z\\_M448\\_0 95\\%")

    def test_table_has_booktabs_and_stable_label(self) -> None:
        output = table(
            caption="Caption", label="tab:test", columns="lr", header="A & B", rows=["x & 1"]
        )
        self.assertIn("\\toprule", output)
        self.assertIn("\\label{tab:test}", output)
        self.assertIn("x & 1\\\\", output)


if __name__ == "__main__":
    unittest.main()
