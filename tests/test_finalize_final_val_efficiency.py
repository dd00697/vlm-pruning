from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import finalize_final_val_efficiency as finalizer
from scripts.finalize_final_val_efficiency import markdown_table


class FinalValFinalizerTests(unittest.TestCase):
    def test_markdown_table_has_stable_header_separator_and_rows(self) -> None:
        table = markdown_table(["A", "B"], [["1", "2"], ["3", "4"]])
        self.assertEqual(table[0], "| A | B |")
        self.assertEqual(table[1], "|---|---|")
        self.assertEqual(table[-1], "| 3 | 4 |")

    def test_report_prints_quality_timing_and_decision_evidence(self) -> None:
        methods = (
            "Full", "K20", "K12", "M448_OPT", "Z_M448_0", "FASTV_MATCHED_199",
            "FEATHER_LLAVA_MAIN", "NUWA_STACK_FULL", "NUWA_NATIVE_192",
        )
        records = [
            {
                "method": method,
                "stack": "canonical",
                "acc_at_0_5": 0.4,
                "mean_iou": 0.4,
                "average_visual_tokens": 196.0,
                "visual_token_layer_work": 6272.0,
                "natural_end_to_end_s": 1.0,
                "prefill_including_overhead_s": 0.5,
                "time_to_first_token_s": 0.6,
                "peak_allocated_gpu_bytes": float(2**30),
                "prompt_kv_bytes": float(2**20),
            }
            for method in methods
        ]
        metric = {"mean": 1.0, "available_rows": 1}
        timing_fields = {
            name: dict(metric)
            for name in (
                "end_to_end_latency_s", "prefill_including_pruning_latency_s",
                "time_to_first_token_s", "selector_latency_s", "gather_latency_s",
                "merge_latency_s", "decode_latency_s", "generated_token_count",
                "throughput_examples_per_s",
            )
        }
        timing = {
            ("canonical", method): {"method": method, **timing_fields}
            for method in ("Full", "K12", "M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN")
        }
        timing.update(
            {
                (stack, method): {"method": method, **timing_fields}
                for stack, stack_methods in {
                    "fastv": ("FASTV_STACK_FULL", "FASTV_MATCHED_199"),
                    "nuwa": ("NUWA_STACK_FULL", "NUWA_NATIVE_192"),
                }.items()
                for method in stack_methods
            }
        )
        accounting = {
            method: {
                "mean_llm_prefill_flops": 1e12,
                "mean_modeled_total_multimodal_prefill_flops": 1e12,
                "mean_score_flops": 1e9,
                "mean_merge_flops": 1e9,
            }
            for method in (
                "Full", "K12", "M448_OPT", "Z_M448_0", "FASTV_MATCHED_199",
                "FEATHER_LLAVA_MAIN", "NUWA_NATIVE_192",
            )
        }
        decision = {
            "status": "FINAL_CANDIDATE_SELECTED",
            "selected_method": "Z_M448_0",
            "method_decisions": {
                method: {
                    "quality_gate_pass": True,
                    "acceleration_gate_pass": True,
                    "visual_token_layer_work": work,
                }
                for method, work in (("K12", 7872), ("M448_OPT", 6848), ("Z_M448_0", 6272))
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            analysis = Path(directory)
            quality_effects = [
                {
                    "contrast": "K12-minus-Full", "metric": "acc_at_0_5_pp",
                    "point_delta": -1.0, "lower_95": -2.0, "upper_95": 0.0,
                }
            ]
            timing_effects = [
                {
                    "stack": "canonical", "contrast": "K12-minus-Full",
                    "metric": "end_to_end_latency_s", "point_delta": -0.1,
                    "lower_95": -0.2, "upper_95": -0.01,
                }
            ]
            (analysis / "QUALITY_PAIRED_CONTRASTS.json").write_text(json.dumps(quality_effects))
            timing_path = analysis / "TIMING_NATURAL_CONTRASTS.json"
            timing_path.write_text(json.dumps(timing_effects))
            (analysis / "FIXED32_TRIGGER.json").write_text(
                json.dumps({"stacks": {stack: {"required": False} for stack in ("canonical", "fastv", "nuwa")}})
            )
            with patch.object(finalizer, "ANALYSIS", analysis), patch.object(
                finalizer, "TIMING_CONTRASTS_PATH", timing_path
            ):
                report = finalizer.build_report(
                    records,
                    {"quality": {}, "timing": timing, "accounting": accounting},
                    decision,
                    {},
                    [finalizer.CAMPAIGN / "analysis/figures/example.pdf"],
                )
        self.assertIn("### Paired full-validation effects", report)
        self.assertIn("### Primary repeated-timing effects", report)
        self.assertIn("| K12/B48 | pass | pass | 7872 | -- |", report)
        self.assertIn("Gather/merge s", report)


if __name__ == "__main__":
    unittest.main()
