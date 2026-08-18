from __future__ import annotations

import unittest

from scripts.analyse_final_val_timing import (
    TimingAnalysisError,
    analysis_stacks,
    contrasts,
    normalize,
    validate_trace,
)


class FinalValTimingAnalysisTests(unittest.TestCase):
    def test_normalize_maps_native_nuwa_fields_and_derives_throughput(self) -> None:
        row = {
            "llm_prefill_with_overhead_latency_s": 0.2,
            "llm_prefill_excluding_selector_latency_s": 0.1,
            "end_to_end_latency_s": 0.5,
        }
        result = normalize(row)
        self.assertEqual(result["prefill_including_pruning_latency_s"], 0.2)
        self.assertEqual(result["prefill_excluding_pruning_latency_s"], 0.1)
        self.assertEqual(result["throughput_examples_per_s"], 2.0)

    def test_trace_validation_accepts_only_registered_shapes(self) -> None:
        validate_trace("FASTV_MATCHED_199", [576] * 3 + [160] * 29)
        validate_trace("FEATHER_LLAVA_MAIN", [576] * 8 + [217] * 8 + [19] * 16)
        with self.assertRaises(TimingAnalysisError):
            validate_trace("NUWA_NATIVE_192", [336] * 15 + [48] * 17)

    def test_contrasts_skip_unavailable_stack_and_metric(self) -> None:
        sample = {
            "end_to_end_latency_s": 1.0,
            "prefill_including_pruning_latency_s": None,
            "time_to_first_token_s": None,
            "decode_latency_s": None,
            "generated_token_count": None,
        }
        stacks = {
            "fastv": {
                "FASTV_MATCHED_199": [[dict(sample)]],
                "FASTV_STACK_FULL": [[dict(sample)]],
            }
        }
        rows = contrasts(stacks)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["stack"], "fastv")
        self.assertEqual(rows[0]["metric"], "end_to_end_latency_s")

    def test_natural_analysis_always_requires_all_stacks(self) -> None:
        self.assertEqual(analysis_stacks("natural"), ("canonical", "fastv", "nuwa"))


if __name__ == "__main__":
    unittest.main()
