from __future__ import annotations

import unittest

from scripts.run_k12_b48_original_official_campaign import METHODS, cells, command


class OriginalOfficialCampaignTests(unittest.TestCase):
    def test_matrix_contains_twelve_datasets_by_two_methods(self) -> None:
        matrix = cells()
        self.assertEqual(len(matrix), 24)
        self.assertEqual({cell.method for cell in matrix}, set(METHODS))
        self.assertEqual(len({cell.dataset for cell in matrix}), 12)

    def test_commands_lock_official_method_and_original_checkpoint(self) -> None:
        cell = next(cell for cell in cells() if cell.dataset == "gqa" and cell.method == "official_visionzip")
        cmd = command(cell, "full", None)
        self.assertEqual(cmd[cmd.index("--method") + 1], "official_visionzip")
        self.assertIn("liuhaotian--llava-v1.5-7b", cmd[cmd.index("--model-path") + 1])
        self.assertIn("--skip-existing", cmd)


if __name__ == "__main__":
    unittest.main()
