from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.frozen_z_error_analysis_common import CAMPAIGN, DISSERTATION, sha256


class FrozenZErrorDissertationTests(unittest.TestCase):
    def test_analysis_report_is_bounded_and_cautious(self) -> None:
        report = (CAMPAIGN / "RESEARCH_REPORT.md").read_text(encoding="utf-8")
        self.assertGreaterEqual(len(report.split()), 400)
        self.assertLessEqual(len(report.split()), 600)
        self.assertIn("post-hoc", report.lower())
        self.assertIn("does not establish equivalence", report)
        self.assertIn("no semantic categories", report)

    def test_generated_block_has_exactly_one_table_and_figure(self) -> None:
        main = (DISSERTATION / "chapters/generated_frozen_z_error_analysis/main.tex").read_text()
        self.assertEqual(main.count(r"\begin{table}"), 1)
        self.assertEqual(main.count(r"\begin{figure}"), 1)
        self.assertIn("748 Full-only and 526 Z-M448-0-only", main)
        self.assertIn("cannot attribute", main)

    def test_asset_manifest_binds_generated_files(self) -> None:
        root = DISSERTATION / "analysis/derived/frozen_z_error_analysis_report_audit"
        manifest = json.loads((root / "ASSET_MANIFEST.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["logical_tables"], 1)
        self.assertEqual(manifest["logical_figures"], 1)
        for row in manifest["generated"].values():
            path = Path(row["path"])
            self.assertTrue(path.is_file())
            self.assertEqual(sha256(path), row["sha256"])

    def test_claim_audit_has_no_unresolved_result(self) -> None:
        audit = json.loads((CAMPAIGN / "analysis/RESULT_CLAIM_AUDIT.json").read_text())
        self.assertEqual(audit["status"], "pass")
        self.assertGreaterEqual(audit["numeric_result_claims"], 100)
        self.assertEqual(audit["unsupported_claims"], [])
        self.assertEqual(audit["unresolved_discrepancies"], [])

    def test_old_error_assets_are_recoverably_retired(self) -> None:
        completion = json.loads((CAMPAIGN / "RETIREMENT_COMPLETION.json").read_text())
        self.assertEqual(completion["status"], "complete")
        self.assertTrue(completion["active_namespace_sources_absent"])
        self.assertEqual(len(completion["actions"]), 14)
        for name, action in completion["actions"].items():
            self.assertFalse((DISSERTATION / name).exists())
            if action["action"] == "moved":
                self.assertTrue(Path(action["archive_path"]).exists())

    def test_reader_facing_status_and_heldout_units(self) -> None:
        macros = (DISSERTATION / "chapters/generated_final_val_efficiency/macros.tex").read_text()
        self.assertIn("FinalValDecisionStatusRaw", macros)
        self.assertIn("candidate selected from validation", macros)
        heldout = (DISSERTATION / "chapters/generated_frozen_z_heldout/main.tex").read_text()
        self.assertIn("Acc@0.5 (\\%)", heldout)
        self.assertIn("50.96", heldout)
        self.assertNotIn("Z\\_M448\\_0", heldout)

    def test_confirmed_e256_correction_is_active(self) -> None:
        discussion = (DISSERTATION / "chapters/discussion.tex").read_text()
        conclusions = (DISSERTATION / "chapters/conclusions.tex").read_text()
        for text in (discussion, conclusions):
            self.assertIn(r"40.20\%", text)
            self.assertNotIn(r"40.69\%", text)


if __name__ == "__main__":
    unittest.main()
