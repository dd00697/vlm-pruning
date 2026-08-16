from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.init_hardware_aware_zero_tail_campaign import (
    CELLS,
    order_sha256,
    tree_sha256,
    validate_ordered_ids,
    Phase0Error,
)


class HardwareAwarePhase0Tests(unittest.TestCase):
    def test_all_registered_cells_exist(self) -> None:
        self.assertEqual(
            set(CELLS),
            {
                "P_K12", "P_M448", "P_M448_REPLAY",
                "R_FUSED_GATHER_MERGE", "R_NATIVE_CACHE", "R_SCORE_REUSE",
                "R_COMPILED_TORCH", "R_COMPILED_CUDA_GRAPH",
                "Z_K12_0", "Z_M448_0", "Z_M448_16",
                "E256_POOL", "E196_POOL", "E256_LORA",
            },
        )

    def test_zero_tail_token_accounting_is_frozen(self) -> None:
        self.assertEqual(CELLS["Z_K12_0"]["average_visual_tokens"], 228.0)
        self.assertEqual(CELLS["Z_M448_0"]["average_visual_tokens"], 196.0)
        self.assertEqual(CELLS["Z_M448_16"]["average_visual_tokens"], 202.0)

    def test_order_validator_rejects_reordering(self) -> None:
        rows = [{"sample_id": "a"}, {"sample_id": "b"}]
        evidence = validate_ordered_ids(rows, ["a", "b"], Path("rows.jsonl"))
        self.assertEqual(evidence["sample_id_order_sha256"], order_sha256(["a", "b"]))
        with self.assertRaisesRegex(Phase0Error, "order differs"):
            validate_ordered_ids(rows, ["b", "a"], Path("rows.jsonl"))

    def test_tree_hash_is_content_and_path_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "one.txt").write_text("one", encoding="utf-8")
            first = tree_sha256(root)
            self.assertEqual(first[1:], (1, 3))
            (root / "two.txt").write_text("two", encoding="utf-8")
            second = tree_sha256(root)
            self.assertNotEqual(first[0], second[0])
            self.assertEqual(second[1:], (2, 6))


if __name__ == "__main__":
    unittest.main()
