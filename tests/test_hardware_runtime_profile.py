from __future__ import annotations

import unittest

from scripts.run_hardware_runtime_profile import (
    MODES,
    PROFILE_ORDER_PLAN,
    HardwareProfileError,
    aggregate,
    stage_semantics_match,
    validate_resume_prefix,
)
from scripts.validate_hardware_runtime_profile import (
    HardwareProfileValidationError,
    required_components,
    validate_profile_components,
)


class HardwareProfileProtocolTests(unittest.TestCase):
    def test_frozen_modes_encode_two_twenty_and_profile_contracts(self) -> None:
        self.assertEqual(MODES["smoke2"], {"limit": 2, "warmup": 0, "repetitions": 1})
        self.assertEqual(MODES["smoke20"], {"limit": 20, "warmup": 2, "repetitions": 1})
        self.assertEqual(MODES["profile100"], {"limit": 100, "warmup": 20, "repetitions": 1})
        self.assertEqual(set(PROFILE_ORDER_PLAN), {1, 2, 3, 4, 5})
        for order in PROFILE_ORDER_PLAN.values():
            self.assertEqual(set(order), {"P_K12", "P_M448", "P_M448_REPLAY"})
        for cell_id in ("P_K12", "P_M448", "P_M448_REPLAY"):
            counts = [sum(order[position] == cell_id for order in PROFILE_ORDER_PLAN.values()) for position in range(3)]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_resume_requires_an_exact_unique_compatible_prefix(self) -> None:
        expected = [("a", 2), ("b", 2), ("c", 2)]
        prefix = [
            {"sample_id": "a", "repetition": 2, "config_sha256": "config", "semantic_match": True},
            {"sample_id": "b", "repetition": 2, "config_sha256": "config", "semantic_match": True},
        ]
        validate_resume_prefix(prefix, expected_order=expected, config_sha256="config")
        with self.assertRaisesRegex(HardwareProfileError, "ordered manifest prefix"):
            validate_resume_prefix(list(reversed(prefix)), expected_order=expected, config_sha256="config")
        duplicate = [prefix[0], prefix[0]]
        with self.assertRaises(HardwareProfileError):
            validate_resume_prefix(duplicate, expected_order=expected, config_sha256="config")
        wrong_hash = [{**prefix[0], "config_sha256": "other"}]
        with self.assertRaisesRegex(HardwareProfileError, "different config hash"):
            validate_resume_prefix(wrong_hash, expected_order=expected, config_sha256="config")

    def profile_row(self, cell_id: str) -> dict:
        required, _ = required_components(cell_id)
        ranges = ((0, 12), (12, 32)) if cell_id == "P_K12" else ((0, 4), (4, 12), (12, 32))
        records = []
        for phase in ("prefill", "decode"):
            for start, end in ranges:
                records.append(
                    {
                        "component": "decoder_segment",
                        "cpu_wall_s": 0.1,
                        "cuda_event_s": 0.09,
                        "metadata": {
                            "phase": phase,
                            "start_layer": start,
                            "end_layer": end,
                            "sequence_length": 10 if phase == "prefill" else 1,
                        },
                    }
                )
        records.extend(
            {
                "component": component,
                "cpu_wall_s": 0.01,
                "cuda_event_s": 0.009,
                "metadata": {},
            }
            for component in required - {"decoder_segment"}
        )
        return {
            "prompt_sequence_lengths_by_layer": [10] * 32,
            "runtime_profile": {
                "enabled": True,
                "records": records,
                "components": {
                    component: {
                        "count": 1,
                        "cpu_wall_total_s": 0.01,
                        "cuda_event_total_s": 0.009,
                    }
                    for component in required
                },
            }
        }

    def test_component_gate_distinguishes_live_and_replay(self) -> None:
        for cell_id in ("P_K12", "P_M448", "P_M448_REPLAY"):
            with self.subTest(cell_id=cell_id):
                validate_profile_components(self.profile_row(cell_id), cell_id, 0)
        replay = self.profile_row("P_M448_REPLAY")
        replay["runtime_profile"]["components"]["score_construction"] = {
            "count": 1,
            "cpu_wall_total_s": 0.1,
            "cuda_event_total_s": 0.1,
        }
        with self.assertRaisesRegex(HardwareProfileValidationError, "forbidden"):
            validate_profile_components(replay, "P_M448_REPLAY", 0)

    def test_replay_stage_parity_allows_only_recomputed_flag_change(self) -> None:
        historical = [{
            "boundary": 4,
            "selector": "query_topk_merge",
            "requested_budget": 2,
            "realized_budget": 2,
            "incoming_original_indices": [0, 1, 2],
            "outgoing_original_indices": [0, 2],
            "incoming_indices_checksum": "in",
            "outgoing_indices_checksum": "out",
            "score_map_checksum": "score",
            "score_map_recomputed": True,
            "incoming_sequence_length": 5,
            "outgoing_sequence_length": 4,
            "text_token_checksum": "text",
            "text_position_checksum": "position",
            "outgoing_position_ids": [0, 1, 3, 4],
            "position_id_checksum": "pid",
            "merge_applied": True,
            "merge_assignment_count": 1,
            "merge_assignment_checksum": "assignment",
            "merge_output_masses": [2.0, 1.0],
            "merge_mass_checksum": "mass",
            "merge_input_mass": 3.0,
            "merge_output_mass": 3.0,
        }]
        replay = [{**historical[0], "score_map_recomputed": False}]
        self.assertTrue(stage_semantics_match(replay, historical, replay=True))
        replay[0]["merge_mass_checksum"] = "changed"
        self.assertFalse(stage_semantics_match(replay, historical, replay=True))

    def test_aggregate_requires_no_semantic_drift(self) -> None:
        rows = []
        for repetition in (1, 2):
            rows.append({
                "cell_id": "P_K12",
                "sample_id": "sample",
                "repetition": repetition,
                "semantic_match": True,
                "vision_encoder_latency_s": 0.1,
                "projector_latency_s": 0.01,
                "prefill_latency_s": 1.0,
                "decode_latency_s": 2.0,
                "end_to_end_latency_s": 3.0,
                "prompt_kv_total_bytes": 10,
                "final_kv_total_bytes": 12,
                "average_visual_tokens": 246,
                "visual_token_layer_work": 7872,
                "sequence_layer_work": 9000,
                "quadratic_sequence_layer_work": 3_000_000,
                "generated_token_count": 20,
                "decode_latency_per_generated_token_s": 2.0 / 19.0,
                "peak_allocated_gpu_bytes": 100,
                "peak_reserved_gpu_bytes": 120,
                "runtime_profile": {
                    "components": {
                        "score_construction": {
                            "count": 1,
                            "cpu_wall_total_s": 0.2,
                            "cuda_event_total_s": 0.1,
                        }
                    }
                },
            })
        result = aggregate(rows)
        self.assertTrue(result["all_semantic_matches"])
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["components"]["score_construction"]["event_count_total"], 2)


if __name__ == "__main__":
    unittest.main()
