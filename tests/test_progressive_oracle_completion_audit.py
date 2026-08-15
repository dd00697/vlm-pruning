from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.audit_progressive_oracle_campaign import (
    CAMPAIGN1_IDS,
    CAMPAIGN2_IDS,
    CAMPAIGN3_IDS,
    EXPECTED_CELL_IDS,
    exact_metrics,
    valid_completion,
)
from vlm_pruning.artifacts import sha256_file


class CompletionAuditUnitTests(unittest.TestCase):
    def test_registered_cell_partition_is_exact(self) -> None:
        self.assertEqual(len(CAMPAIGN1_IDS), 14)
        self.assertEqual(len(CAMPAIGN2_IDS), 9)
        self.assertEqual(len(CAMPAIGN3_IDS), 10)
        self.assertEqual(len(EXPECTED_CELL_IDS), 33)

    def test_exact_metrics_does_not_accept_partial_or_rounded_values(self) -> None:
        self.assertTrue(exact_metrics({"a": 1.0}, {"a": 1.0}))
        self.assertFalse(exact_metrics({"a": 1.0}, {"a": 1.0, "b": 2.0}))
        self.assertFalse(exact_metrics({"a": 1.0001}, {"a": 1.0}))

    def test_completion_hash_check_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("config.json", "metrics.json", "integrity_report.json", "predictions.jsonl"):
                (root / name).write_text(name, encoding="utf-8")
            completion = {
                "status": "complete",
                "rows": 1000,
                "manifest_sha256": "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd",
                "config_sha256": sha256_file(root / "config.json"),
                "metrics_sha256": sha256_file(root / "metrics.json"),
                "integrity_sha256": sha256_file(root / "integrity_report.json"),
                "predictions_sha256": sha256_file(root / "predictions.jsonl"),
                "oracle_indices_sha256": None,
            }
            self.assertTrue(valid_completion(root, completion, timing=False))
            (root / "predictions.jsonl").write_text("changed", encoding="utf-8")
            self.assertFalse(valid_completion(root, completion, timing=False))


if __name__ == "__main__":
    unittest.main()
