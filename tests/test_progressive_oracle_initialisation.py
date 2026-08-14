from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.init_progressive_oracle_campaign import (
    CampaignInitialisationError,
    box_iou,
    indices_checksum,
    parse_box,
    read_jsonl_strict,
    registered_cells,
)


class ProgressiveOracleInitialisationTests(unittest.TestCase):
    def test_registered_cells_are_exactly_the_33_preregistered_quality_cells(self) -> None:
        cells = registered_cells()
        self.assertEqual(len(cells), 33)
        self.assertEqual(sum(row["campaign"] == 1 for row in cells.values()), 14)
        self.assertEqual(sum(row["campaign"] == 2 for row in cells.values()), 9)
        self.assertEqual(sum(row["campaign"] == 3 for row in cells.values()), 10)
        self.assertEqual(cells["P3"]["schedule"][0], {"boundary": 4, "budget": 384, "selector": "query_topk"})
        self.assertEqual(cells["T5"]["schedule"], [{"boundary": 20, "budget": 0, "selector": "none"}])

    def test_independent_parser_and_iou(self) -> None:
        predicted, raw, reason = parse_box("[0.1, 0.2, 0.9, 0.8]", 640, 400)
        self.assertEqual(raw, [0.1, 0.2, 0.9, 0.8])
        self.assertEqual(predicted, [64.0, 80.0, 576.0, 320.0])
        self.assertIsNone(reason)
        self.assertEqual(box_iou(predicted, predicted), 1.0)
        self.assertEqual(parse_box("none", 1, 1)[2], "fewer_than_four_numbers")
        self.assertEqual(parse_box("[2,0,3,1]", 1, 1)[2], "outside_normalized_0_1")

    def test_index_checksum_matches_historical_compact_json_contract(self) -> None:
        self.assertEqual(indices_checksum([1, 2, 3]), "a615eeaee21de517")

    def test_strict_jsonl_rejects_blank_and_wrong_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps({"a": 1}) + "\n\n", encoding="utf-8")
            with self.assertRaisesRegex(CampaignInitialisationError, "Blank"):
                read_jsonl_strict(path)
            path.write_text(json.dumps({"a": 1}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(CampaignInitialisationError, "Expected 2"):
                read_jsonl_strict(path, expected_rows=2)


if __name__ == "__main__":
    unittest.main()
