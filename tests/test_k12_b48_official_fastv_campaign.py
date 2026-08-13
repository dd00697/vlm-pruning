from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_k12_b48_official_fastv_campaign",
    ROOT / "scripts/run_k12_b48_official_fastv_campaign.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class OfficialFastVCampaignTests(unittest.TestCase):
    def test_matrix_contains_every_locked_dataset(self) -> None:
        cells = MODULE.cells()
        self.assertEqual(len(cells), 12)
        self.assertEqual(len({cell.key for cell in cells}), 12)
        mmmu = next(cell for cell in cells if cell.key == "mmmu_val")
        self.assertEqual(mmmu.expected_rows, 857)
        self.assertIn("single-image", mmmu.subset_note)

    def test_command_executes_official_fastv_at_exact_b48(self) -> None:
        cell = next(cell for cell in MODULE.cells() if cell.key == "gqa")
        command = MODULE.command(cell, "full")
        joined = " ".join(command)
        self.assertEqual(command[0], str(MODULE.OFFICIAL_PYTHON))
        self.assertIn("--method official_fastv", joined)
        self.assertIn("--target-visual-tokens 48", joined)
        self.assertIn("--precision bf16", joined)
        self.assertNotIn("--keep-ratio", joined)

    def test_expected_visual_trace_is_exact(self) -> None:
        self.assertEqual([576, 576, 576] + [48] * 29, [576] * 3 + [48] * 29)


if __name__ == "__main__":
    unittest.main()
