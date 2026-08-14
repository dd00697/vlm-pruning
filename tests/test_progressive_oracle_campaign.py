from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_progressive_oracle_campaign import (
    CampaignPreparationError,
    SAMPLING_NAMESPACE,
    prepare_campaign,
    select_discovery_rows,
    selection_digest,
)


class DiscoveryManifestTests(unittest.TestCase):
    def make_source(self, root: Path, *, count: int = 8) -> Path:
        source = root / "source.jsonl"
        rows = []
        for index in range(count):
            image = root / f"image-{index}.jpg"
            image.write_bytes(f"image-{index}".encode("utf-8"))
            rows.append(
                {
                    "dataset": "refcoco",
                    "split": "val",
                    "sample_id": f"sample-{index}",
                    "image_path": str(image),
                    "query": f"query {index}",
                }
            )
        source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return source

    def test_selection_uses_registered_nul_salted_digest_and_source_order(self) -> None:
        rows = [{"sample_id": f"sample-{index}"} for index in range(8)]
        selected, ranked = select_discovery_rows(rows, selected_rows=3)
        expected = sorted(
            rows,
            key=lambda row: (
                hashlib.sha256(
                    SAMPLING_NAMESPACE.encode("utf-8") + b"\0" + row["sample_id"].encode("utf-8")
                ).hexdigest(),
                row["sample_id"].encode("utf-8"),
            ),
        )[:3]
        expected_ids = {row["sample_id"] for row in expected}
        self.assertEqual([row["sample_id"] for row in selected], [row["sample_id"] for row in rows if row["sample_id"] in expected_ids])
        self.assertEqual([item["sample_id"] for item in ranked], [row["sample_id"] for row in expected])
        self.assertEqual(
            selection_digest("sample-0"),
            hashlib.sha256(SAMPLING_NAMESPACE.encode() + b"\0sample-0").hexdigest(),
        )

    def test_prepare_writes_and_then_verifies_immutable_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            output = root / "campaign/manifests/discovery.jsonl"
            lock_path = root / "campaign/manifests/lock.json"
            with patch(
                "scripts.prepare_progressive_oracle_campaign.git_output",
                side_effect=lambda _root, *args: "abc123" if args[:2] == ("rev-parse", "HEAD") else "",
            ):
                first = prepare_campaign(
                    repo_root=root,
                    source_manifest=source,
                    output_manifest=output,
                    lock_path=lock_path,
                    expected_source_rows=8,
                    selected_rows=3,
                )
                second = prepare_campaign(
                    repo_root=root,
                    source_manifest=source,
                    output_manifest=output,
                    lock_path=lock_path,
                    expected_source_rows=8,
                    selected_rows=3,
                )
            self.assertEqual(first, second)
            self.assertEqual(first["selected_rows"], 3)
            self.assertEqual(first["status"], "frozen")
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 3)

            output.write_text(output.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with patch("scripts.prepare_progressive_oracle_campaign.git_output", return_value="abc123"):
                with self.assertRaisesRegex(CampaignPreparationError, "differs"):
                    prepare_campaign(
                        repo_root=root,
                        source_manifest=source,
                        output_manifest=output,
                        lock_path=lock_path,
                        expected_source_rows=8,
                        selected_rows=3,
                    )

    def test_duplicate_id_and_missing_image_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root, count=2)
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            rows[1]["sample_id"] = rows[0]["sample_id"]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(CampaignPreparationError, "Duplicate"):
                prepare_campaign(
                    repo_root=root,
                    source_manifest=source,
                    output_manifest=root / "out.jsonl",
                    lock_path=root / "lock.json",
                    expected_source_rows=2,
                    selected_rows=1,
                )

            rows[1]["sample_id"] = "sample-1"
            rows[1]["image_path"] = str(root / "missing.jpg")
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(CampaignPreparationError, "Missing image"):
                prepare_campaign(
                    repo_root=root,
                    source_manifest=source,
                    output_manifest=root / "out.jsonl",
                    lock_path=root / "lock.json",
                    expected_source_rows=2,
                    selected_rows=1,
                )


if __name__ == "__main__":
    unittest.main()
