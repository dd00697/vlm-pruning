from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.analyse_frozen_z_heldout import metrics, paired_bootstrap
from scripts.frozen_z_heldout_common import MANIFESTS, THRESHOLDS, Z_SCHEDULE, Z_TRACE
from scripts.run_frozen_z_heldout_campaign import command
from scripts.run_progressive_oracle_refcoco import ProgressiveRunError, parse_schedule, validate_cli


def args(**updates):
    values = {
        "cell_id": "Z_M448_0_testA",
        "method": "progressive",
        "pre_llm_selector": "none",
        "pre_llm_budget": None,
        "oracle_teacher_layer": None,
        "oracle_index_input": None,
        "runtime_implementation": "original",
        "schedule": parse_schedule(Z_SCHEDULE),
        "limit": None,
        "registered_scope": "heldout_testA",
        "seed": 0,
        "model_path": "llava-hf/llava-1.5-7b-hf",
        "outcome_embargo": True,
    }
    values.update(updates)
    return SimpleNamespace(**values)


class FrozenZHeldoutTests(unittest.TestCase):
    def test_frozen_contract(self) -> None:
        self.assertEqual(len(Z_TRACE), 32)
        self.assertEqual(sum(Z_TRACE), 6272)
        self.assertEqual(sum(Z_TRACE) / 32, 196)
        self.assertEqual(MANIFESTS["testA"]["rows"], 5657)
        self.assertEqual(MANIFESTS["testB"]["rows"], 5095)
        self.assertEqual(THRESHOLDS["maximum_accuracy_loss_absolute"], 0.03)
        self.assertEqual(THRESHOLDS["maximum_miou_loss_absolute"], 0.015)

    def test_complete_candidate_requires_exact_schedule_and_embargo(self) -> None:
        validate_cli(args())
        with self.assertRaisesRegex(ProgressiveRunError, "exact frozen"):
            validate_cli(args(schedule=parse_schedule("4:448:query_topk_merge,12:48:query_topk")))
        with self.assertRaisesRegex(ProgressiveRunError, "outcome embargo"):
            validate_cli(args(outcome_embargo=False))
        with self.assertRaisesRegex(ProgressiveRunError, "cell ID"):
            validate_cli(args(cell_id="Z_M448_0_testB"))

    def test_full_is_empty_schedule_and_never_embargoed(self) -> None:
        validate_cli(args(cell_id="Full_testA", schedule=(), outcome_embargo=False))
        with self.assertRaisesRegex(ProgressiveRunError, "empty schedule"):
            validate_cli(args(cell_id="Full_testA", outcome_embargo=False))
        with self.assertRaisesRegex(ProgressiveRunError, "reserved"):
            validate_cli(args(cell_id="Full_testA", schedule=(), outcome_embargo=True))

    def test_controller_command_is_frozen(self) -> None:
        from pathlib import Path

        result = command("Z_M448_0", "testB", Path("/tmp/run"), limit=None)
        self.assertEqual(result[result.index("--registered-scope") + 1], "heldout_testB")
        self.assertEqual(result[result.index("--schedule") + 1], Z_SCHEDULE)
        self.assertIn("--outcome-embargo", result)
        self.assertNotIn("--limit", result)
        full = command("Full", "testA", Path("/tmp/full"), limit=None)
        self.assertNotIn("--schedule", full)
        self.assertNotIn("--outcome-embargo", full)

    def test_paired_statistics_preserve_direction(self) -> None:
        def row(index, correct, iou):
            return {
                "sample_id": str(index), "success_iou_0_5": correct, "box_iou": iou,
                "valid": True, "candidate_average_visual_tokens": 196.0,
                "realized_visual_tokens_by_layer": Z_TRACE,
            }

        full = [row(0, True, 0.8), row(1, True, 0.6), row(2, False, 0.2)]
        z = [row(0, True, 0.7), row(1, False, 0.4), row(2, False, 0.1)]
        result = paired_bootstrap(z, full, 0)
        self.assertAlmostEqual(result["acc_at_0_5"]["point_delta"], -1 / 3)
        self.assertAlmostEqual(result["mean_iou"]["point_delta"], -0.4 / 3)
        self.assertEqual(metrics(z)["token_layer_work"], 6272)


if __name__ == "__main__":
    unittest.main()
