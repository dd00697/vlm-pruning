from __future__ import annotations

import unittest

from scripts.frozen_z_error_analysis_common import ErrorAnalysisFailure
from scripts.qa_frozen_z_error_dissertation import marker_issues, scan_log, validate_visual_report


class FrozenZErrorQATests(unittest.TestCase):
    def test_log_scanner_rejects_fatal_undefined_and_overfull(self) -> None:
        self.assertEqual(scan_log("clean log"), [])
        self.assertIn("fatal", scan_log("\n! Undefined control sequence"))
        self.assertIn("undefined_reference", scan_log("LaTeX Warning: Reference `x' undefined"))
        self.assertIn("overfull_box", scan_log(r"Overfull \hbox (1.0pt too wide)"))

    def test_marker_scanner_requires_new_analysis_and_rejects_stale_value(self) -> None:
        good = " ".join((
            "Paired localisation threshold-crossing analysis", "748 Full-only", "526 Z-M448-0-only",
            "622 Full-only", "505 Z-M448-0-only", "40.20%", "candidate selected from validation",
            "mixed split result",
        ))
        self.assertEqual(marker_issues(good), [])
        self.assertIn("forbidden:40.69%", marker_issues(good + " 40.69%"))

    def test_visual_report_requires_exact_pages_notes_and_hashes(self) -> None:
        hashes = {"page_001.png": {"bytes": 1, "sha256": "a"}}
        report = {"status": "pass", "pages": {"1": {"status": "pass", "notes": "A substantive page inspection note.", "render_sha256": "a"}}}
        validate_visual_report(report, [1], hashes)
        report["pages"]["1"]["render_sha256"] = "b"
        with self.assertRaises(ErrorAnalysisFailure):
            validate_visual_report(report, [1], hashes)


if __name__ == "__main__":
    unittest.main()
