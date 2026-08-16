from __future__ import annotations

import unittest

import numpy as np

from scripts.finalize_adaptive_efficiency_goal import (
    CAMPAIGN_ROOT,
    FinalAuditError,
    is_campaign_report_markdown,
    validate_registry,
)
from scripts.init_adaptive_efficiency_campaign import (
    box_iou,
    error_profile,
    parse_box,
    quantile_groups,
)
from vlm_pruning.llava_hooks import average_visual_tokens, realized_visual_trace, validate_schedule


class AdaptiveScheduleAccountingTests(unittest.TestCase):
    def test_registered_custom_schedule_averages(self) -> None:
        schedules = {
            "C448": ([(4, 448), (12, 48)], 214.0),
            "C480": ([(4, 480), (12, 48)], 222.0),
            "D1": ([(8, 384), (12, 48)], 222.0),
            "B1": ([(4, 384), (12, 96), (20, 16)], 198.0),
            "A1": ([(4, 384), (12, 96), (20, 48)], 210.0),
        }
        for cell_id, (raw, expected) in schedules.items():
            with self.subTest(cell_id=cell_id):
                events = validate_schedule(raw)
                self.assertEqual(average_visual_tokens(realized_visual_trace(events)), expected)


class AdaptiveA0AnalysisTests(unittest.TestCase):
    def test_registered_box_parser_and_iou(self) -> None:
        predicted, reason = parse_box("[0.1, 0.2, 0.5, 0.8]", 100, 200)
        self.assertIsNone(reason)
        self.assertEqual(predicted, [10.0, 40.0, 50.0, 160.0])
        self.assertEqual(box_iou(predicted, predicted), 1.0)
        self.assertEqual(parse_box("nothing", 100, 100), (None, "fewer_than_four_numbers"))
        self.assertEqual(parse_box("[-0.1, 0, 1, 1]", 100, 100), (None, "outside_normalized_0_1"))

    def test_quantile_groups_are_deterministic_and_complete(self) -> None:
        groups, labels, edges = quantile_groups(np.arange(8, dtype=np.float64), "x")
        self.assertEqual(labels, ["x_q1", "x_q2", "x_q3", "x_q4"])
        self.assertEqual(len(edges), 3)
        self.assertEqual(len(groups), 8)
        self.assertEqual(set(groups.tolist()), {0, 1, 2, 3})

    def test_error_profile_distinguishes_near_and_catastrophic_losses(self) -> None:
        reference = [
            {"sample_id": "a", "box_iou": 0.8, "success_iou_0_5": True},
            {"sample_id": "b", "box_iou": 0.7, "success_iou_0_5": True},
            {"sample_id": "c", "box_iou": 0.2, "success_iou_0_5": False},
        ]
        candidate = [
            {"sample_id": "a", "box_iou": 0.45, "success_iou_0_5": False},
            {"sample_id": "b", "box_iou": 0.05, "success_iou_0_5": False},
            {"sample_id": "c", "box_iou": 0.6, "success_iou_0_5": True},
        ]
        profile = error_profile(reference, candidate, "candidate")
        self.assertEqual(profile["k12_correct_candidate_wrong"], 2)
        self.assertEqual(profile["candidate_correct_k12_wrong"], 1)
        self.assertEqual(profile["near_miss_candidate_iou_0_40_to_0_50_when_k12_correct"], 1)
        self.assertEqual(profile["catastrophic_candidate_iou_below_0_10_when_k12_correct"], 1)


class AdaptiveFinalAuditTests(unittest.TestCase):
    def registry(self) -> dict:
        return {
            "phases": {
                name: {"status": "complete"}
                for name in ("A0", "preflight", "fixed_methods", "official_methods", "route_gate", "timing")
            },
            "cells": {
                **{
                    name: {"status": "accepted"}
                    for name in (
                        "C448",
                        "C480",
                        "D1",
                        "M448",
                        "OFF_PDROP_L2_192",
                        "OFF_PDROP_REROUTE_192",
                        "GATE_SEED0",
                        "GATE_SEED1",
                        "GATE_SEED2",
                        "GATE_RANDOM_RATE_CONTROL",
                        "GATE_PRIVILEGED_ORACLE_CONTROL",
                    )
                },
                **{
                    name: {"status": "skipped"}
                    for name in ("B1", "A1", "OFF_NUWA_192", "OFF_NUWA_REROUTE_192")
                },
            },
        }

    def test_final_auditor_uses_canonical_uppercase_a0_phase_key(self) -> None:
        registry = self.registry()
        self.assertEqual(validate_registry(registry)["required_phases_complete"][0], "A0")
        registry["phases"]["a0"] = registry["phases"].pop("A0")
        with self.assertRaisesRegex(FinalAuditError, "A0"):
            validate_registry(registry)

    def test_single_report_audit_excludes_only_pinned_official_source_docs(self) -> None:
        self.assertFalse(
            is_campaign_report_markdown(CAMPAIGN_ROOT / "official_sources/checkouts/method/README.md")
        )
        self.assertTrue(is_campaign_report_markdown(CAMPAIGN_ROOT / "RESEARCH_REPORT.md"))
        self.assertTrue(is_campaign_report_markdown(CAMPAIGN_ROOT / "analysis/EXTRA_REPORT.md"))


if __name__ == "__main__":
    unittest.main()
