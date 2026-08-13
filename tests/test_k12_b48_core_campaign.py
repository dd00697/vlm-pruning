from __future__ import annotations

import unittest

from scripts.run_k12_b48_core_campaign import (
    MODEL_SNAPSHOT,
    broad_cells,
    command,
    execution_env,
    random_variance_cells,
    refcoco_repair_cells,
)


class CoreCampaignTests(unittest.TestCase):
    def test_broad_matrix_is_twelve_by_four_and_only_b48_plus_full(self) -> None:
        cells = broad_cells()
        self.assertEqual(len(cells), 48)
        self.assertEqual(len({cell.dataset for cell in cells}), 12)
        for cell in cells:
            self.assertIn(cell.method, {"full", "random", "uniform", "structured_tca"})
            self.assertEqual(cell.budget, 576 if cell.method == "full" else 48)

    def test_primary_command_locks_every_scientific_argument(self) -> None:
        cell = next(cell for cell in broad_cells() if cell.dataset == "gqa" and cell.method == "structured_tca")
        joined = " ".join(command(cell, "full"))
        for fragment in (
            "--prune-after-layer 12", "--target-visual-tokens 48", "--ug-map-source query_max",
            "--ug-fusion-mode single_layer", "--ug-probe-layers 12", "--ug-role-preset more_anchors",
            "--position-mode gap_preserving", "--decode-mode layer_aware_cache", "--precision bf16",
        ):
            self.assertIn(fragment, joined)
        self.assertNotIn("--target-visual-tokens 72", joined)
        self.assertNotIn("--target-visual-tokens 128", joined)

    def test_random_variance_is_separate_three_seed_evidence(self) -> None:
        variance = random_variance_cells()
        self.assertEqual(len(variance), 26)
        self.assertEqual({cell.seed for cell in variance}, {1, 2})
        self.assertTrue(
            all(cell.method == "random" and cell.stage == "random_variance" for cell in variance)
        )
        seed_zero = {cell.seed for cell in broad_cells() if cell.method == "random"}
        self.assertEqual(seed_zero, {0})

    def test_refcoco_completion_contains_only_missing_or_wrong_config_cells(self) -> None:
        cells = refcoco_repair_cells()
        self.assertEqual({cell.method for cell in cells}, {"full", "random", "uniform", "structured_compact"})
        self.assertTrue(all(cell.split == "val" for cell in cells))

    def test_execution_uses_pinned_local_snapshot_offline(self) -> None:
        environment = execution_env()
        self.assertEqual(environment["VLM_LOCAL_MODEL_PATH"], str(MODEL_SNAPSHOT))
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")


if __name__ == "__main__":
    unittest.main()
