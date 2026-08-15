from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from vlm_pruning.llava_hooks.progressive_pruning import PruningEvent
from vlm_pruning.llava_hooks.route_gate import (
    AGGRESSIVE_ROUTE,
    AGGRESSIVE_SCHEDULE,
    FEATURE_DIMENSION,
    FEATURE_SCHEMA_SHA256,
    SAFE_ROUTE,
    SAFE_SCHEDULE,
    DynamicRouteForwarder,
    RouteDecision,
)


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AdaptiveRouteGateManifestTests(unittest.TestCase):
    def test_hash_sampling_is_deterministic_grouped_and_leak_free(self) -> None:
        module = load_script("prepare_adaptive_gate_manifest")
        refs = []
        for index in range(3105):
            refs.append(
                {
                    "split": "train",
                    "ann_id": 100000 + index,
                    "ref_id": 200000 + index,
                    "image_id": 300000 + index,
                }
            )
        # A second reference on an existing image must not create a second record.
        refs.append(
            {
                "split": "train",
                "ann_id": 999001,
                "ref_id": 999002,
                "image_id": 300000,
            }
        )
        refs.append(
            {
                "split": "val",
                "ann_id": 999003,
                "ref_id": 999004,
                "image_id": 999005,
            }
        )
        excluded = {"100001_0"}
        first, first_evidence = module.choose_records(refs, excluded_original_ids=excluded)
        second, second_evidence = module.choose_records(refs, excluded_original_ids=excluded)
        self.assertEqual(first, second)
        self.assertEqual(first_evidence, second_evidence)
        self.assertEqual(len(first), 3000)
        self.assertEqual(len({item["ref"]["image_id"] for item in first}), 3000)
        self.assertTrue(all(f"{item['ref']['ann_id']}_0" not in excluded for item in first))
        counts = {name: 0 for name in ("fit", "early_stop", "calibration")}
        for item in first:
            counts[item["training_partition"]] += 1
        self.assertEqual(counts, {"fit": 2160, "early_stop": 240, "calibration": 600})
        self.assertEqual(first_evidence["validation_original_ids_excluded"], 1)

    def test_validation_original_ids_requires_provenance(self) -> None:
        module = load_script("prepare_adaptive_gate_manifest")
        rows = [
            {"metadata": {"original_sample_id": "1_0"}},
            {"metadata": {"original_sample_id": "2_0"}},
        ]
        self.assertEqual(module.validation_original_ids(rows), {"1_0", "2_0"})


class AdaptiveRouteGateFeatureTests(unittest.TestCase):
    def test_feature_schema_is_fixed_finite_and_deterministic(self) -> None:
        forwarder = object.__new__(DynamicRouteForwarder)
        forwarder.torch = torch
        forwarder.image_token_id = 32000
        forwarder.initial_visual_tokens = 576
        forwarder.grid = 24
        generator = torch.Generator().manual_seed(20260815)
        hidden = torch.randn((1, 580, 4096), generator=generator)
        input_ids = torch.tensor([[1, 2] + [32000] * 576 + [3, 4]], dtype=torch.long)
        positions = torch.arange(580, dtype=torch.long)
        first = forwarder.extract_gate_features(
            hidden,
            input_ids=input_ids,
            original_positions=positions,
            query_original_positions=[578, 579],
            width=640,
            height=427,
        )
        second = forwarder.extract_gate_features(
            hidden,
            input_ids=input_ids,
            original_positions=positions,
            query_original_positions=[578, 579],
            width=640,
            height=427,
        )
        self.assertEqual(int(first.vector.numel()), FEATURE_DIMENSION)
        self.assertTrue(torch.isfinite(first.vector).all().item())
        self.assertTrue(FEATURE_SCHEMA_SHA256)
        self.assertEqual(first.vector_sha256, second.vector_sha256)
        self.assertTrue(torch.equal(first.vector, second.vector))

    def test_route_decision_accepts_only_the_two_frozen_schedules(self) -> None:
        safe = RouteDecision(SAFE_ROUTE, 0.75, 0.5, SAFE_SCHEDULE)
        aggressive = RouteDecision(AGGRESSIVE_ROUTE, 0.25, 0.5, AGGRESSIVE_SCHEDULE)
        self.assertEqual(DynamicRouteForwarder._validate_decision(safe), SAFE_SCHEDULE)
        self.assertEqual(DynamicRouteForwarder._validate_decision(aggressive), AGGRESSIVE_SCHEDULE)
        malformed = RouteDecision(SAFE_ROUTE, 0.75, 0.5, (PruningEvent(4, 448, "query_topk"),))
        with self.assertRaises(Exception):
            DynamicRouteForwarder._validate_decision(malformed)

    def test_dynamic_generation_refreshes_state_diagnostics_after_decode(self) -> None:
        forwarder = object.__new__(DynamicRouteForwarder)
        forwarder.torch = torch
        forwarder._cuda_sync = lambda: None
        state = SimpleNamespace(
            generated_tokens_cached=0,
            segments=[],
            final_retained_indices=[],
            stage_traces=[],
        )
        forwarder.prefill_with_route_gate = lambda **_kwargs: SimpleNamespace(
            logits=torch.tensor([[[0.0, 1.0]]]),
            state=state,
            diagnostics={
                "progressive/cache_generated_tokens": 0,
                "cache/prompt_segment_bytes": [10],
                "cache/prompt_total_bytes": 10,
                "gate/route_name": AGGRESSIVE_ROUTE,
            },
        )

        def decode(current_state, _token):
            current_state.generated_tokens_cached += 1
            return torch.tensor([[[0.0, 1.0]]])

        forwarder.decode_one_with_segment_caches = decode
        forwarder._diagnostics = lambda current_state: {
            "progressive/cache_generated_tokens": current_state.generated_tokens_cached,
            "cache/prompt_segment_bytes": [20],
            "cache/prompt_total_bytes": 20,
        }
        result = forwarder.generate_with_route_gate(
            inputs_embeds=None,
            input_ids=None,
            attention_mask=None,
            sample_id="sample",
            seed=0,
            query_original_positions=(),
            width=1,
            height=1,
            decide=lambda _vector: RouteDecision(
                AGGRESSIVE_ROUTE, 0.0, 0.5, AGGRESSIVE_SCHEDULE
            ),
            max_new_tokens=2,
            eos_ids=set(),
        )
        self.assertEqual(result.diagnostics["progressive/cache_generated_tokens"], 1)
        self.assertEqual(result.diagnostics["gate/route_name"], AGGRESSIVE_ROUTE)
        self.assertEqual(result.diagnostics["cache/prompt_segment_bytes"], [10])
        self.assertEqual(result.diagnostics["cache/prompt_total_bytes"], 10)


if __name__ == "__main__":
    unittest.main()
