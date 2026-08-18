from __future__ import annotations

import unittest

from vlm_pruning.final_efficiency import account_trace, known_trace


class FinalEfficiencyAccountingTests(unittest.TestCase):
    def test_known_work_and_average(self) -> None:
        expected = {
            "Full": (18432, 576),
            "K12": (7872, 246),
            "M448_OPT": (6848, 214),
            "Z_M448_0": (6272, 196),
            "FASTV_MATCHED_199": (6368, 199),
            "NUWA_NATIVE_192": (6144, 192),
        }
        for method, (work, average) in expected.items():
            result = account_trace(known_trace(method), text_tokens=0)
            self.assertEqual(result.visual_token_layer_work, work)
            self.assertEqual(result.average_visual_tokens, average)

    def test_toy_flop_and_kv_formula(self) -> None:
        result = account_trace(
            [2, 1], text_tokens=1, hidden_size=2, intermediate_size=3,
            kv_hidden_size=2, dtype_bytes=2, pruning_metric_flops=7,
        )
        # n=[3,2]: attention=sum(8*n*4 + 4*n^2*2)=264; MLP=sum(6*n*2*3)=180.
        self.assertEqual(result.attention_flops, 264)
        self.assertEqual(result.linear_and_mlp_flops, 180)
        self.assertEqual(result.llm_prefill_flops, 444)
        self.assertEqual(result.modeled_total_prefill_flops, 451)
        self.assertEqual(result.analytical_prompt_kv_bytes, 40)

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            account_trace([], text_tokens=1)
        with self.assertRaises(ValueError):
            known_trace("unknown")


if __name__ == "__main__":
    unittest.main()
