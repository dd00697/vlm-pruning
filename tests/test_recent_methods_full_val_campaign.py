from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.init_recent_methods_full_val_campaign import CELLS, MANIFEST_SHA256, ROWS
from scripts.run_progressive_oracle_refcoco import ProgressiveRunError, validate_cli
from scripts.run_recent_methods_full_val_campaign import (
    compare_semantics,
    runner_command,
    semantic_signature,
)
from vlm_pruning.llava_hooks import average_visual_tokens, realized_visual_trace, validate_schedule


class RecentMethodsProtocolTests(unittest.TestCase):
    def test_exact_five_cells_and_accounting(self) -> None:
        self.assertEqual(
            list(CELLS),
            [
                "V1_K12_48_K20_6",
                "V2_Z_K12_0",
                "V3_C448",
                "V4_M448",
                "V5_Z_M448_0",
            ],
        )
        self.assertEqual(ROWS, 8811)
        self.assertEqual(
            MANIFEST_SHA256,
            "f26d7b954189e86f55b50f9506bb670d3a0cf847d3152779af6c08dbf094e3d7",
        )
        for cell_id, cell in CELLS.items():
            with self.subTest(cell=cell_id):
                events = validate_schedule(cell["events"])
                trace = realized_visual_trace(events)
                self.assertEqual(trace, cell["trace"])
                self.assertEqual(sum(trace), cell["token_layer_work"])
                self.assertEqual(average_visual_tokens(trace), cell["average_visual_tokens"])
                self.assertEqual(trace[-1], cell["final_visual_tokens"])

    def test_full_command_has_frozen_scope_and_no_limit(self) -> None:
        from pathlib import Path

        command = runner_command(
            "V1_K12_48_K20_6",
            manifest=Path("manifest.jsonl"),
            output_root=Path("output"),
            run_name="run",
            limit=None,
            full_validation=True,
        )
        self.assertEqual(command[command.index("--registered-scope") + 1], "full_validation")
        self.assertNotIn("--limit", command)
        self.assertEqual(
            command[command.index("--schedule") + 1],
            "12:48:k12_original,20:6:query_topk",
        )

    def test_full_scope_rejects_a_limit(self) -> None:
        args = SimpleNamespace(
            pre_llm_selector="none",
            pre_llm_budget=None,
            oracle_teacher_layer=None,
            oracle_index_input=None,
            runtime_implementation="original",
            schedule=validate_schedule([(12, 48, "k12_original"), (20, 6, "query_topk")]),
            limit=20,
            registered_scope="full_validation",
        )
        with self.assertRaisesRegex(ProgressiveRunError, "cannot use --limit"):
            validate_cli(args)

    def test_semantic_signature_ignores_only_runtime_measurements(self) -> None:
        base = {
            "sample_id": "id",
            "raw_model_output": "[0,0,1,1]",
            "predicted_box": [0, 0, 1, 1],
            "box_iou": 1.0,
            "success_iou_0_5": True,
            "generated_token_count": 2,
            "final_retained_indices": [],
            "realized_visual_tokens_by_layer": [0] * 32,
            "stage_traces": [
                {
                    "boundary": 20,
                    "requested_budget": 0,
                    "selection_latency_s": 0.1,
                    "merge_latency_s": 0.2,
                }
            ],
        }
        changed = {
            **base,
            "stage_traces": [
                {
                    **base["stage_traces"][0],
                    "selection_latency_s": 99.0,
                    "merge_latency_s": 88.0,
                }
            ],
        }
        self.assertEqual(semantic_signature(base), semantic_signature(changed))
        changed["raw_model_output"] = "different"
        self.assertNotEqual(semantic_signature(base), semantic_signature(changed))

    def test_historical_parity_accepts_an_aligned_reference_prefix(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        row = {
            "sample_id": "id",
            "raw_model_output": "[0,0,1,1]",
            "predicted_box": [0, 0, 1, 1],
            "box_iou": 1.0,
            "success_iou_0_5": True,
            "generated_token_count": 2,
            "final_retained_indices": [],
            "realized_visual_tokens_by_layer": [0] * 32,
            "stage_traces": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.jsonl"
            right = root / "right.jsonl"
            left.write_text(json.dumps(row) + "\n", encoding="utf-8")
            right.write_text(json.dumps(row) + "\n" + json.dumps({**row, "sample_id": "later"}) + "\n", encoding="utf-8")
            compare_semantics(left, right, expected_rows=1, right_prefix=True)
            with self.assertRaisesRegex(Exception, "wrong row count"):
                compare_semantics(left, right, expected_rows=1, right_prefix=False)


if __name__ == "__main__":
    unittest.main()
