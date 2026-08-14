from __future__ import annotations

import unittest

from scripts.run_progressive_oracle_timing import aggregate, segment_records
from scripts.run_progressive_oracle_timing_campaign import evaluate_gates
from vlm_pruning.llava_hooks import PruningEvent


def timing_variant(value: float, *, selector: bool) -> dict:
    return {
        "vision_encoder_latency_s": value,
        "projector_latency_s": value,
        "llm_prefill_latency_s": value,
        "selector_total_latency_s": value if selector else 0.0,
        "decode_latency_s": value,
        "decode_time_per_generated_token_s": value,
        "end_to_end_latency_s": value,
        "generated_token_count": 1,
        "prompt_kv_total_bytes": value,
        "final_kv_total_bytes": value,
        "candidate_average_visual_tokens": 360.0 if selector else 576.0,
        "peak_allocated_gpu_bytes": int(value * 100),
        "peak_reserved_gpu_bytes": int(value * 200),
        "selector_stage_latencies": (
            [{"stage_index": 0, "boundary": 20, "selector": "none", "latency_s": value}]
            if selector else []
        ),
    }


class TimingAggregationTests(unittest.TestCase):
    def test_segment_records_preserve_layer_ranges_and_storage(self) -> None:
        schedule = (PruningEvent(20, 0, "none"),)
        diagnostics = {
            "progressive/segment_prompt_lengths": [600, 24],
            "cache/prompt_segment_bytes": [100, 20],
            "cache/final_segment_bytes": [110, 30],
        }
        prompt, final = segment_records(schedule, diagnostics)
        self.assertEqual([(row["start_layer"], row["end_layer"]) for row in prompt], [(0, 20), (20, 32)])
        self.assertEqual([row["kv_bytes"] for row in prompt], [100, 20])
        self.assertEqual([row["kv_bytes"] for row in final], [110, 30])

    def test_paired_aggregate_reports_ratios_and_stage_timing(self) -> None:
        rows = [
            {"full": timing_variant(2.0, selector=False), "candidate": timing_variant(1.0, selector=True)},
            {"full": timing_variant(4.0, selector=False), "candidate": timing_variant(2.0, selector=True)},
        ]
        metrics = aggregate(rows)
        self.assertEqual(metrics["samples"], 2)
        self.assertEqual(metrics["candidate_to_full/end_to_end_latency_s_ratio"], 0.5)
        self.assertEqual(metrics["candidate_to_full/end_to_end_latency_s_reduction"], 0.5)
        stage = metrics["variants"]["candidate"]["selector_stages"][0]
        self.assertEqual(stage["boundary"], 20)
        self.assertEqual(stage["latency_s"]["mean"], 1.5)


class TimingGateTests(unittest.TestCase):
    def make_registry(self) -> dict:
        cells = {
            "T5": {
                "status": "accepted",
                "requested_schedule": [{"boundary": 20, "budget": 0, "selector": "none"}],
                "metrics": {"quality/acc_at_0_5": 0.452, "quality/mean_iou": 0.4405},
                "candidate_average_visual_tokens": 360.0,
            }
        }
        progressive = {
            "P1": (0.391, 0.4105, 206),
            "P2": (0.377, 0.4038, 198),
            "P3": (0.408, 0.4247, 198),
            "P4": (0.382, 0.4081, 166),
            "P5": (0.406, 0.4198, 186),
            "P6": (0.368, 0.4045, 156),
            "P7": (0.390, 0.4122, 180),
            "P8": (0.401, 0.4179, 192),
            "P9": (0.400, 0.4160, 193.5),
        }
        for cell_id, (acc, miou, average) in progressive.items():
            cells[cell_id] = {
                "status": "accepted",
                "metrics": {
                    "quality/acc_at_0_5": acc,
                    "quality/mean_iou": miou,
                    "quality/parsed_box_rate": 1.0,
                },
                "candidate_average_visual_tokens": average,
                "validation": {"all_checks": True},
            }
        return {
            "cells": cells,
            "decisions": {"B_tail": {"value": 0}},
            "reference": {"discovery_recomputed": {"acc_at_0_5": 0.439, "mean_iou": 0.4328321325}},
        }

    def test_registered_gates_run_e1_and_skip_e2(self) -> None:
        gates = evaluate_gates(self.make_registry())
        self.assertTrue(gates["E1"]["passed"])
        self.assertEqual(gates["E1"]["candidate_cell_id"], "T5")
        self.assertFalse(gates["E2"]["passed"])
        self.assertEqual(gates["E2"]["candidate_cell_id"], "P5")
        self.assertGreater(gates["E2"]["acc_at_0_5_loss"], 0.02)
        self.assertGreater(gates["E2"]["mean_iou_loss"], 0.01)


if __name__ == "__main__":
    unittest.main()
