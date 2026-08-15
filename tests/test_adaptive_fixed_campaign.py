from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from scripts.run_adaptive_fixed_campaign import (
    FixedCampaignError,
    active_runner_processes,
    k12_like,
    promising,
    rank_candidates,
    recover_stale_workers,
    route_pool,
)


class AdaptiveFixedDecisionTests(unittest.TestCase):
    def test_k12_like_uses_all_three_strict_registered_conditions(self) -> None:
        passing = {
            "acc_at_0_5": 0.429,
            "mean_iou": 0.4165978636,
            "average_visual_tokens": 245.999,
        }
        self.assertTrue(k12_like(passing))
        for key, value in (
            ("acc_at_0_5", 0.428999),
            ("mean_iou", 0.4165978635),
            ("average_visual_tokens", 246.0),
        ):
            failing = dict(passing)
            failing[key] = value
            self.assertFalse(k12_like(failing))

    def test_promising_requires_accuracy_and_miou_guard(self) -> None:
        reference = {"acc_at_0_5": 0.408, "mean_iou": 0.424}
        self.assertTrue(promising({"acc_at_0_5": 0.418, "mean_iou": 0.419}, reference))
        self.assertFalse(promising({"acc_at_0_5": 0.4179, "mean_iou": 0.430}, reference))
        self.assertFalse(promising({"acc_at_0_5": 0.430, "mean_iou": 0.4189}, reference))

    def test_near_accuracy_ranking_prefers_lower_token_work_then_miou(self) -> None:
        candidates = {
            "HIGH": {"acc_at_0_5": 0.430, "mean_iou": 0.42, "average_visual_tokens": 222, "boundaries": 2},
            "LEAN": {"acc_at_0_5": 0.426, "mean_iou": 0.41, "average_visual_tokens": 198, "boundaries": 2},
            "FAR": {"acc_at_0_5": 0.424, "mean_iou": 0.45, "average_visual_tokens": 100, "boundaries": 1},
        }
        selected, trace = rank_candidates(candidates)
        self.assertEqual(selected, "LEAN")
        self.assertFalse(next(item for item in trace if item["candidate"] == "FAR")["within_0_005_of_highest_acc"])

    def test_route_pool_uses_registered_p5_fallback_without_new_eligible_route(self) -> None:
        registry = {
            "cells": {
                name: {"status": "skipped"}
                for name in ("C448", "C480", "D1", "M448", "B1", "A1")
            }
        }
        selected, trace = route_pool(registry)
        self.assertEqual(selected, "P5")
        self.assertEqual(trace[0]["reason"], "registered_no_new_eligible_fallback")

    def test_process_scan_matches_only_exact_progressive_run_name(self) -> None:
        with TemporaryDirectory() as temporary:
            proc_root = Path(temporary)
            for pid, command in (
                ("101", ["python", "scripts/run_progressive_oracle_refcoco.py", "--run-name", "c480_attempt1"]),
                ("102", ["python", "scripts/run_progressive_oracle_refcoco.py", "--run-name", "d1_attempt1"]),
                ("103", ["python", "unrelated.py", "--run-name", "c480_attempt1"]),
            ):
                process = proc_root / pid
                process.mkdir()
                (process / "cmdline").write_bytes(b"\0".join(item.encode() for item in command) + b"\0")
            matches = active_runner_processes("c480_attempt1", proc_root=proc_root)
        self.assertEqual([item["pid"] for item in matches], [101])

    def test_recovery_accepts_completed_attempt_one_without_relaunch(self) -> None:
        metrics = {
            "rows": 1000,
            "acc_at_0_5": 0.424,
            "mean_iou": 0.4268,
            "parse_rate": 1.0,
            "average_visual_tokens": 222.0,
            "peak_allocated_mb": 1.0,
            "peak_reserved_mb": 2.0,
        }
        evidence = {
            "run_dir": "unused",
            "config_sha256": "a",
            "predictions_sha256": "b",
            "metrics_sha256": "c",
            "integrity_sha256": "d",
            "completion_sha256": "e",
        }
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "COMPLETED.json").write_text("{}", encoding="utf-8")
            state = {"active_workers": ["C480"], "decision_history": []}
            registry = {
                "cells": {
                    "C480": {
                        "status": "running",
                        "attempts": [
                            {
                                "attempt": 1,
                                "status": "running",
                                "run_dir": str(run_dir),
                                "command": "python runner --run-name c480_attempt1",
                            }
                        ],
                    }
                },
                "decisions": [],
                "totals": {"accepted_validation_quality": 1},
            }
            with (
                mock.patch("scripts.run_adaptive_fixed_campaign.active_runner_processes", return_value=[]),
                mock.patch(
                    "scripts.run_adaptive_fixed_campaign.validated_completed_attempt",
                    return_value=(metrics, evidence),
                ),
                mock.patch("scripts.run_adaptive_fixed_campaign.read_json", return_value={"completed_at_utc": "2026-08-15T00:00:00Z"}),
                mock.patch("scripts.run_adaptive_fixed_campaign.save_state") as save,
            ):
                recovered = recover_stale_workers(state, registry)
        self.assertEqual([item["cell_id"] for item in recovered], ["C480"])
        self.assertEqual(state["active_workers"], [])
        self.assertEqual(registry["cells"]["C480"]["status"], "accepted")
        self.assertEqual(registry["cells"]["C480"]["attempts"][0]["attempt"], 1)
        self.assertEqual(registry["totals"]["accepted_validation_quality"], 2)
        self.assertEqual(registry["decisions"][0]["decision"], "recover_completed_stale_worker_C480")
        save.assert_called_once()

    def test_recovery_refuses_to_clear_a_live_worker(self) -> None:
        state = {"active_workers": ["C480"], "decision_history": []}
        registry = {
            "cells": {
                "C480": {
                    "status": "running",
                    "attempts": [{"attempt": 1, "run_dir": "runs/c480_attempt1"}],
                }
            }
        }
        with mock.patch(
            "scripts.run_adaptive_fixed_campaign.active_runner_processes",
            return_value=[{"pid": 123, "command": ["python"]}],
        ):
            with self.assertRaises(FixedCampaignError):
                recover_stale_workers(state, registry)
        self.assertEqual(state["active_workers"], ["C480"])


if __name__ == "__main__":
    unittest.main()
