from __future__ import annotations

import unittest

from scripts.run_adaptive_fixed_campaign import k12_like, promising, rank_candidates, route_pool


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


if __name__ == "__main__":
    unittest.main()
