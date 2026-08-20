from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import audit_frozen_z_heldout as audit


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row(sample_id: str, correct: bool, iou: float, trace=None):
    trace = trace if trace is not None else ([576] * 4 + [448] * 8 + [48] * 8 + [0] * 12)
    return {
        "sample_id": sample_id,
        "valid": True,
        "success_iou_0_5": correct,
        "box_iou": iou,
        "candidate_average_visual_tokens": sum(trace) / 32,
        "realized_visual_tokens_by_layer": trace,
    }


class FrozenHeldoutTerminalAuditTests(unittest.TestCase):
    def test_split_gates_are_inclusive_at_every_frozen_boundary(self) -> None:
        thresholds = {
            "minimum_valid_parsed_box_rate": 0.995,
            "maximum_accuracy_loss_absolute": 0.03,
            "maximum_miou_loss_absolute": 0.015,
        }
        full = {"acc_at_0_5": 0.50, "mean_iou": 0.40}
        # Use the nearest representable values on the passing side of the
        # mathematically inclusive decimal boundaries, matching the frozen analyser.
        z = {"parsed_box_rate": 0.995, "acc_at_0_5": 0.47000000000000003, "mean_iou": 0.38500000000000006}
        result = audit.split_gates(z, full, thresholds)
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["gates"].values()))

    def test_split_gates_fail_each_threshold_independently(self) -> None:
        thresholds = {
            "minimum_valid_parsed_box_rate": 0.995,
            "maximum_accuracy_loss_absolute": 0.03,
            "maximum_miou_loss_absolute": 0.015,
        }
        full = {"acc_at_0_5": 0.50, "mean_iou": 0.40}
        cases = (
            {"parsed_box_rate": 0.9949, "acc_at_0_5": 0.50, "mean_iou": 0.40},
            {"parsed_box_rate": 1.0, "acc_at_0_5": 0.4699, "mean_iou": 0.40},
            {"parsed_box_rate": 1.0, "acc_at_0_5": 0.50, "mean_iou": 0.3849},
        )
        for z in cases:
            with self.subTest(z=z):
                self.assertFalse(audit.split_gates(z, full, thresholds)["passed"])

    def test_all_three_status_branches_are_exact(self) -> None:
        self.assertEqual(
            audit.classify_heldout({"testA": {"passed": True}, "testB": {"passed": True}}),
            "HELDOUT_CONFIRMED_BOTH_SPLITS",
        )
        self.assertEqual(
            audit.classify_heldout({"testA": {"passed": True}, "testB": {"passed": False}}),
            "HELDOUT_MIXED_SPLIT_RESULT",
        )
        self.assertEqual(
            audit.classify_heldout({"testA": {"passed": False}, "testB": {"passed": False}}),
            "HELDOUT_NOT_CONFIRMED",
        )
        with self.assertRaises(audit.TerminalAuditError):
            audit.classify_heldout({"testA": {"passed": True}})

    def test_bootstrap_is_paired_directional_and_seeded(self) -> None:
        full = [row("a", True, 0.8), row("b", True, 0.6), row("c", False, 0.2)]
        z = [row("a", True, 0.7), row("b", False, 0.4), row("c", False, 0.1)]
        left = audit.paired_bootstrap(z, full, 20260820)
        right = audit.paired_bootstrap(z, full, 20260820)
        self.assertEqual(left, right)
        self.assertAlmostEqual(left["acc_at_0_5"]["point_delta"], -1 / 3)
        self.assertAlmostEqual(left["mean_iou"]["point_delta"], -0.4 / 3)
        self.assertEqual(left["acc_at_0_5"]["resamples"], 10000)
        self.assertEqual(left["acc_at_0_5"]["seed"], 20260820)
        self.assertEqual(left["acc_at_0_5"]["direction"], "Z_M448_0_minus_Full")
        with self.assertRaisesRegex(audit.TerminalAuditError, "order"):
            audit.paired_bootstrap(z[::-1], full, 20260820)

    def test_nested_float_comparison_rejects_missing_and_tampered_values(self) -> None:
        audit.compare_nested_floats({"a": {"b": 1.0}}, {"a": {"b": 1.0}}, "x")
        with self.assertRaises(audit.TerminalAuditError):
            audit.compare_nested_floats({"a": {}}, {"a": {"b": 1.0}}, "x")
        with self.assertRaises(audit.TerminalAuditError):
            audit.compare_nested_floats({"a": {"b": 1.1}}, {"a": {"b": 1.0}}, "x")

    def test_artifact_ledger_binds_every_required_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            for name in (
                "config.json", "environment.json", "predictions.jsonl",
                "metrics.json", "integrity_report.json", "COMPLETED.json",
            ):
                (run / name).write_text(name, encoding="utf-8")
            names = (
                "config.json", "environment.json", "predictions.jsonl",
                "metrics.json", "integrity_report.json", "COMPLETED.json",
            )
            record = {"path": "run", "files": {
                name: {
                    "bytes": (run / name).stat().st_size,
                    "sha256": hashlib.sha256((run / name).read_bytes()).hexdigest(),
                }
                for name in names
            }}
            with patch.object(audit, "REPO", root):
                audit.verify_ledger_artifacts("Full_testA", run, record, set(names))
                (run / "predictions.jsonl").write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(audit.TerminalAuditError, "bytes differ"):
                    audit.verify_ledger_artifacts("Full_testA", run, record, set(names))
                with self.assertRaisesRegex(audit.TerminalAuditError, "internally incomplete"):
                    audit.verify_ledger_artifacts("Full_testA", run, record, {"missing.json"})
                (run / "predictions.jsonl").write_text("predictions.jsonl", encoding="utf-8")
                record["files"]["unregistered.json"] = {"bytes": 0, "sha256": "0" * 64}
                with self.assertRaisesRegex(audit.TerminalAuditError, "coverage differs"):
                    audit.verify_ledger_artifacts("Full_testA", run, record, set(names))

    def test_quality_artifact_ledger_uses_native_pass_schema_and_exact_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            write_json(campaign / "QUALITY_RUN_LEDGER.json", {"status": "complete"})
            write_json(campaign / "OUTCOME_EMBARGO_RELEASE.json", {"status": "outcome_embargo_released"})
            cells = {cell: {"path": f"quality/{cell}", "files": {}} for cell in audit.EXPECTED_CELLS}
            write_json(campaign / "QUALITY_ARTIFACT_LEDGER.json", {
                "schema_version": 1,
                "status": "pass",
                "exact_cells": list(audit.EXPECTED_CELLS),
                "cells": cells,
                "quality_run_ledger_sha256": hashlib.sha256(
                    (campaign / "QUALITY_RUN_LEDGER.json").read_bytes()
                ).hexdigest(),
                "outcome_embargo_release_sha256": hashlib.sha256(
                    (campaign / "OUTCOME_EMBARGO_RELEASE.json").read_bytes()
                ).hexdigest(),
            })
            with patch.object(audit, "CAMPAIGN", campaign):
                self.assertEqual(audit.artifact_ledger_cells(), cells)
                value = json.loads((campaign / "QUALITY_ARTIFACT_LEDGER.json").read_text())
                value["status"] = "complete"
                write_json(campaign / "QUALITY_ARTIFACT_LEDGER.json", value)
                with self.assertRaisesRegex(audit.TerminalAuditError, "incomplete"):
                    audit.artifact_ledger_cells()

    def test_quality_seriality_rejects_overlap_extra_attempt_and_live_pid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "campaign"
            campaign.mkdir()
            records = []
            names = list(audit.EXPECTED_QUALITY_ORDER)
            times = (
                ("2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"),
                ("2026-01-01T00:01:00Z", "2026-01-01T00:02:00Z"),
                ("2026-01-01T00:02:00Z", "2026-01-01T00:03:00Z"),
                ("2026-01-01T00:03:00Z", "2026-01-01T00:04:00Z"),
            )
            for cell, (started, finished) in zip(names, times):
                method, split = ("Z_M448_0", cell.rsplit("_", 1)[1]) if cell.startswith("Z") else ("Full", cell.rsplit("_", 1)[1])
                records.append({
                    "method": method, "split": split, "limit": None, "exit_code": 0,
                    "pids_before": [], "pids_after": [], "started_at_utc": started,
                    "finished_at_utc": finished,
                    "run_dir": str(Path("campaign") / audit.EXPECTED_QUALITY_DIRS[cell]),
                })
            ledger_path = campaign / "QUALITY_RUN_LEDGER.json"
            write_json(ledger_path, {"status": "raw_sealed_both_splits", "serial_execution": True, "records": records})
            write_json(campaign / "OUTCOME_EMBARGO_RELEASE.json", {
                "quality_ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest()
            })
            (campaign / "quality").mkdir()
            with patch.object(audit, "REPO", root), patch.object(audit, "CAMPAIGN", campaign):
                _, evidence = audit.audit_attempts_seriality()
                self.assertTrue(evidence["serial_execution"])
                records[2]["started_at_utc"] = "2026-01-01T00:01:30Z"
                write_json(ledger_path, {"status": "raw_sealed_both_splits", "serial_execution": True, "records": records})
                with self.assertRaisesRegex(audit.TerminalAuditError, "overlap"):
                    audit.audit_attempts_seriality()
                records[2]["started_at_utc"] = "2026-01-01T00:02:00Z"
                records[0]["pids_before"] = ["unexpected"]
                write_json(ledger_path, {"status": "raw_sealed_both_splits", "serial_execution": True, "records": records})
                with self.assertRaisesRegex(audit.TerminalAuditError, "GPU isolation"):
                    audit.audit_attempts_seriality()

    def _decision_fixture(self, root: Path, tamper: bool = False) -> None:
        campaign = root / "campaign"
        campaign.mkdir()
        events = [
            "ledger_initialized_from_immutable_pre_z_evidence",
            "historical_full_reuse_decision",
            "physical_smokes_passed",
            "preregistration_report_snapshot_preserved",
            "quality_cells_complete_and_bound",
            "outcome_embargo_released_after_both_raw_seals",
            "preregistered_heldout_rule_applied",
            "cumulative_report_finalized",
            "dissertation_and_pdf_qa_complete",
        ]
        rows = []
        previous = None
        for index, event in enumerate(events):
            evidence = campaign / f"evidence_{index}.json"
            write_json(evidence, {"event": event})
            payload = {
                "event": event,
                "recorded_at_utc": f"2026-01-01T00:{index:02d}:00Z",
                "evidence_path": evidence.name,
                "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                "note": f"Synthetic evidence for {event}",
                "previous_entry_sha256": previous,
            }
            if event == "preregistered_heldout_rule_applied":
                payload["heldout_status"] = "HELDOUT_CONFIRMED_BOTH_SPLITS"
            payload["entry_sha256"] = audit.json_checksum(payload)
            previous = payload["entry_sha256"]
            rows.append(payload)
        ledger = campaign / "DECISION_LEDGER.jsonl"
        serialize = lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        ledger.write_text("".join(serialize(item) for item in rows), encoding="utf-8")
        prefix = b"".join(serialize(item).encode() for item in rows[:4])
        write_json(campaign / "DECISION_LEDGER_PRE_CANDIDATE_LOCK.json", {
            "status": "locked", "entries": 4, "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
            "last_entry_sha256": rows[3]["entry_sha256"], "locked_at_utc": "2026-01-01T00:03:30Z",
        })
        write_json(campaign / audit.EXPECTED_QUALITY_DIRS["Z_M448_0_testA"] / "config.json", {
            "created_at_utc": "2026-01-01T01:00:00Z"
        })
        report = (
            "Preregistered Full testA testB. The outcome embargo passed. Paired 95% intervals and "
            "heterogeneity are reported. HELDOUT_CONFIRMED_BOTH_SPLITS. This is not population-level "
            "equivalence and creates no timing claim."
        )
        (campaign / "RESEARCH_REPORT.md").write_text(report, encoding="utf-8")
        if tamper:
            rows[4]["event"] = "changed"
            ledger.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in rows), encoding="utf-8")

    def test_decision_ledger_chain_and_pre_candidate_prefix_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._decision_fixture(root)
            campaign = root / "campaign"
            rows = audit.read_jsonl(campaign / "DECISION_LEDGER.jsonl")
            prefix_hash = audit.sha256_bytes(b"".join(
                (json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
                for item in rows[:4]
            ))
            with patch.object(audit, "CAMPAIGN", campaign), \
                    patch.object(audit, "REPORT", campaign / "RESEARCH_REPORT.md"), \
                    patch.object(audit, "PRE_CANDIDATE_LEDGER_PREFIX_SHA256", prefix_hash), \
                    patch.object(audit, "PRE_CANDIDATE_LEDGER_HEAD_SHA256", rows[3]["entry_sha256"]):
                evidence = audit.audit_decision_ledger_and_report("HELDOUT_CONFIRMED_BOTH_SPLITS")
                self.assertEqual(evidence["entries"], 9)
                rows[4]["previous_entry_sha256"] = "bad"
                (campaign / "DECISION_LEDGER.jsonl").write_text(
                    "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in rows),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(audit.TerminalAuditError, "chain"):
                    audit.audit_decision_ledger_and_report("HELDOUT_CONFIRMED_BOTH_SPLITS")

    def _terminal_fixture(self, root: Path):
        campaign = root / "campaign"
        derived = root / "derived"
        campaign.mkdir()
        derived.mkdir()
        registry = campaign / "EXPERIMENT_REGISTRY.json"
        controller = campaign / "CONTROLLER_STATE.json"
        report = campaign / "RESEARCH_REPORT.md"
        write_json(registry, {"phase_status": {"terminal_audit": "pending"}, "heldout_status": "HELDOUT_NOT_CONFIRMED"})
        write_json(controller, {"active_workers": [], "candidate_outcome_embargo": False})
        report.write_text("cumulative\n", encoding="utf-8")
        for path in (
            campaign / "analysis/HELDOUT_DECISION.json",
            campaign / "analysis/HELDOUT_RESULTS.json",
            campaign / "analysis/PAIRED_CONTRASTS.json",
            campaign / "analysis/SPLIT_HETEROGENEITY.json",
            campaign / "QUALITY_ARTIFACT_LEDGER.json",
            campaign / "PDF_QA.json",
            campaign / "PDF_PUBLICATION_RECORD.json",
            campaign / "DISSERTATION_CHANGE_RECORD.json",
        ):
            write_json(path, {"x": 1})
        write_json(derived / "ASSET_MANIFEST.json", {"x": 1})
        write_json(derived / "CLAIM_EVIDENCE_LEDGER.json", {"x": 1})
        (campaign / "DECISION_LEDGER.jsonl").write_text("{}\n", encoding="utf-8")
        source = root / "source.pdf"
        user = root / "user.pdf"
        source.write_bytes(b"pdf")
        user.write_bytes(b"pdf")
        return campaign, derived, registry, controller, report, source, user

    def test_terminal_finalization_sets_exact_status_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign, derived, registry, controller, report, source, user = self._terminal_fixture(root)
            final = campaign / "FINAL_COMPLETION_REPORT.json"
            test_log = campaign / "analysis/FINAL_TEST_SUITE.log"
            test_json = campaign / "analysis/FINAL_TEST_SUITE.json"
            patches = (
                patch.object(audit, "REPO", root),
                patch.object(audit, "CAMPAIGN", campaign), patch.object(audit, "DERIVED", derived),
                patch.object(audit, "REGISTRY", registry), patch.object(audit, "CONTROLLER", controller),
                patch.object(audit, "REPORT", report), patch.object(audit, "FINAL", final),
                patch.object(audit, "FINAL_TEST_LOG", test_log), patch.object(audit, "FINAL_TEST_JSON", test_json),
                patch.object(audit, "SOURCE_PDF", source), patch.object(audit, "USER_PDF", user),
            )
            for item in patches:
                item.start()
            try:
                output = "Ran 1 test in 0.001s\n\nOK\n"
                requirements = {"source_tests_git_processes": {
                    "status": "pass", "tests": 1,
                    "test_output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                    "_test_output": output,
                }}
                result = audit.finalize_terminal_state("HELDOUT_NOT_CONFIRMED", requirements)
                self.assertEqual(result["status"], audit.TERMINAL_STATE)
                self.assertEqual(result["heldout_status"], "HELDOUT_NOT_CONFIRMED")
                self.assertEqual(audit.read_json(registry)["status"], audit.TERMINAL_STATE)
                self.assertEqual(audit.read_json(controller)["active_workers"], [])
                self.assertTrue(final.is_file())
                self.assertEqual(report.read_text(encoding="utf-8"), "cumulative\n")
                before = {path: path.read_bytes() for path in (registry, controller, report, final)}
                with self.assertRaisesRegex(audit.TerminalAuditError, "refusing"):
                    audit.finalize_terminal_state("HELDOUT_NOT_CONFIRMED", {})
                self.assertEqual(before, {path: path.read_bytes() for path in before})
            finally:
                for item in reversed(patches):
                    item.stop()

    def test_terminal_finalization_failure_before_write_leaves_state_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign, derived, registry, controller, report, source, user = self._terminal_fixture(root)
            (campaign / "analysis/HELDOUT_RESULTS.json").unlink()
            final = campaign / "FINAL_COMPLETION_REPORT.json"
            test_log = campaign / "analysis/FINAL_TEST_SUITE.log"
            test_json = campaign / "analysis/FINAL_TEST_SUITE.json"
            before = {path: path.read_bytes() for path in (registry, controller, report)}
            patches = (
                patch.object(audit, "REPO", root),
                patch.object(audit, "CAMPAIGN", campaign), patch.object(audit, "DERIVED", derived),
                patch.object(audit, "REGISTRY", registry), patch.object(audit, "CONTROLLER", controller),
                patch.object(audit, "REPORT", report), patch.object(audit, "FINAL", final),
                patch.object(audit, "FINAL_TEST_LOG", test_log), patch.object(audit, "FINAL_TEST_JSON", test_json),
                patch.object(audit, "SOURCE_PDF", source), patch.object(audit, "USER_PDF", user),
            )
            for item in patches:
                item.start()
            try:
                with self.assertRaises(audit.TerminalAuditError):
                    audit.finalize_terminal_state("HELDOUT_NOT_CONFIRMED", {})
                self.assertEqual(before, {path: path.read_bytes() for path in before})
                self.assertFalse(final.exists())
            finally:
                for item in reversed(patches):
                    item.stop()


if __name__ == "__main__":
    unittest.main()
