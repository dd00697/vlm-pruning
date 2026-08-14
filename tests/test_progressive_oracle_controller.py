from __future__ import annotations

import unittest

import numpy as np

from scripts.analyse_progressive_oracle_campaign import paired_bootstrap, values
from scripts.run_progressive_oracle_campaign import (
    build_command,
    near_best_select,
    schedule_arg,
    verify_preflight,
)


class ProgressiveControllerTests(unittest.TestCase):
    def test_preflight_registers_two_quality_jobs_but_not_three(self) -> None:
        report = verify_preflight()
        self.assertEqual(report["safe_quality_parallel_width"], 2)
        self.assertGreater(
            report["capacity"]["three_job_reserved_with_10_percent_headroom_mib"],
            report["capacity"]["memory_free_mib_before_smoke"],
        )

    def test_near_best_rule_prefers_efficiency_then_miou_then_id(self) -> None:
        metrics = {
            "P1": {"acc_at_0_5": 0.440, "mean_iou": 0.42, "average_visual_tokens": 206},
            "P2": {"acc_at_0_5": 0.436, "mean_iou": 0.43, "average_visual_tokens": 198},
            "P3": {"acc_at_0_5": 0.436, "mean_iou": 0.44, "average_visual_tokens": 198},
            "P4": {"acc_at_0_5": 0.430, "mean_iou": 0.45, "average_visual_tokens": 166},
        }
        selected, trace = near_best_select(metrics, count=2)
        self.assertEqual(selected, ["P3", "P2"])
        self.assertEqual(trace[0]["near_best_ids"], ["P1", "P2", "P3"])

    def test_dynamic_oracle_command_preserves_budget_and_tail_schedule(self) -> None:
        cell = {
            "attempts": [],
            "schedule": [{"boundary": 20, "budget": 8, "selector": "query_topk"}],
            "seed": 0,
            "pre_llm_budget": 97,
            "oracle_teacher_layer": 12,
        }
        command, run_dir, attempt, resume = build_command("O7", cell)
        self.assertEqual(attempt, 1)
        self.assertFalse(resume)
        self.assertIn("20:8:query_topk", command)
        self.assertEqual(command[command.index("--pre-llm-budget") + 1], "97")
        self.assertEqual(command[command.index("--oracle-teacher-layer") + 1], "12")
        self.assertEqual(run_dir.name, "o7_attempt1")
        self.assertEqual(schedule_arg(cell["schedule"]), "20:8:query_topk")

    def test_interrupted_running_cell_resumes_same_attempt_and_directory(self) -> None:
        cell = {
            "status": "running",
            "attempts": [{"attempt": 1, "run_dir": "experiments/progressive_oracle_refcoco_val_20260814/runs/t1_attempt1"}],
            "schedule": [{"boundary": 20, "budget": 32, "selector": "query_topk"}],
            "seed": 0,
        }
        command, run_dir, attempt, resume = build_command("T1", cell)
        self.assertTrue(resume)
        self.assertEqual(attempt, 1)
        self.assertEqual(run_dir.name, "t1_attempt1")
        self.assertIn("--resume", command)


class ProgressiveAnalysisTests(unittest.TestCase):
    def test_accuracy_values_preserve_three_seed_per_example_means(self) -> None:
        rows = [{"success_iou_0_5": value, "box_iou": 0.0} for value in (0.0, 1 / 3, 2 / 3, 1.0)]
        self.assertTrue(np.allclose(values(rows, "acc_at_0_5"), [0.0, 1 / 3, 2 / 3, 1.0]))

    def test_bootstrap_is_deterministic_and_uses_percentage_points_for_accuracy(self) -> None:
        left = np.ones(1000)
        right = np.zeros(1000)
        first = paired_bootstrap(left, right, seed=7, scale=100.0)
        second = paired_bootstrap(left, right, seed=7, scale=100.0)
        self.assertEqual(first, second)
        self.assertEqual(first["mean_delta"], 100.0)
        self.assertEqual(first["lower_95"], 100.0)
        self.assertEqual(first["upper_95"], 100.0)


if __name__ == "__main__":
    unittest.main()
