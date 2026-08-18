from __future__ import annotations

import unittest

from scripts.build_final_val_dissertation_assets import DISPLAY, table, tex, timing_tables


class FinalValDissertationAssetTests(unittest.TestCase):
    def test_tex_escapes_method_identifiers_and_percent(self) -> None:
        self.assertEqual(tex("Z_M448_0 95%"), "Z\\_M448\\_0 95\\%")

    def test_table_has_booktabs_and_stable_label(self) -> None:
        output = table(
            caption="Caption", label="tab:test", columns="lr", header="A & B", rows=["x & 1"]
        )
        self.assertIn("\\toprule", output)
        self.assertIn("\\label{tab:test}", output)
        self.assertIn("x & 1\\\\", output)

    def test_nuwa_display_uses_latex_umlaut(self) -> None:
        self.assertEqual(DISPLAY["NUWA_STACK_FULL"], 'N\\"uwa-stack Full')

    def test_timing_tables_emit_fixed32_only_when_supplied(self) -> None:
        def metric(value: float) -> dict[str, float | int]:
            return {"mean": value, "std": 0.0, "available_rows": 1}

        fields = {
            name: metric(1.0)
            for name in (
                "end_to_end_latency_s", "prefill_including_pruning_latency_s",
                "time_to_first_token_s", "selector_latency_s", "gather_latency_s",
                "merge_latency_s", "decode_latency_s", "generated_token_count",
                "throughput_examples_per_s", "average_visual_tokens",
            )
        }
        timing = {
            ("canonical", method): {"method": method, **fields}
            for method in ("Full", "K12", "M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN")
        }
        timing.update({
            (stack, method): {"method": method, **fields}
            for stack, methods in {
                "fastv": ("FASTV_STACK_FULL", "FASTV_MATCHED_199"),
                "nuwa": ("NUWA_STACK_FULL", "NUWA_NATIVE_192"),
            }.items()
            for method in methods
        })
        accounting = {
            method: {
                "mean_analytical_prompt_kv_bytes": 1.0,
                "mean_analytical_final_kv_bytes": 1.0,
                "mean_peak_allocated_gpu_bytes": 1.0,
                "mean_peak_reserved_gpu_bytes": 1.0,
            }
            for _, method in timing
            if method in {"Full", "K12", "M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN"}
        }
        natural_only = timing_tables(timing, accounting)
        self.assertNotIn("fixed-32 diagnostic", natural_only)
        with_fixed = timing_tables(
            timing,
            accounting,
            {("fastv", "FASTV_MATCHED_199"): timing[("fastv", "FASTV_MATCHED_199")]},
        )
        self.assertIn("fixed-32 diagnostic", with_fixed)


if __name__ == "__main__":
    unittest.main()
