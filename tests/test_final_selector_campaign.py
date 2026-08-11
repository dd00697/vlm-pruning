from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.run_ug_prune_single_pass import run_name
from vlm_pruning.artifacts import (
    ArtifactIntegrityError,
    finalize_partial_jsonl,
    load_resume_rows,
    validate_completed_run,
    validate_metric_agreement,
    validate_rows,
)
from vlm_pruning.llava_hooks import MidLayerUGPruneForwarder, ug_gap58_config_for_prune_layer
from vlm_pruning.selectors import UGPruneConfig, UGPruneInputs, UGPruneSelector
from vlm_pruning.selectors.ug_prune import budget_split


def inputs(scores: np.ndarray) -> UGPruneInputs:
    return UGPruneInputs(scores, {}, "layer20:query_max", "object", "matched", [580])


class SelectorStrategyTests(unittest.TestCase):
    def test_query_topk_has_stable_low_index_ties_and_raster_sorted_gather(self) -> None:
        scores = np.zeros((24, 24), dtype=np.float32)
        scores.reshape(-1)[100] = 5.0
        config = UGPruneConfig(selector_strategy="query_topk")
        result = UGPruneSelector(config).select_from_map(
            sample_id="x", budget=4, original_tokens=576, grid=24, seed=0, inputs=inputs(scores)
        )
        self.assertEqual(result.selected_indices, [0, 1, 2, 100])
        self.assertEqual(result.role_labels["query_topk"], [0, 1, 2, 100])
        self.assertEqual(result.metadata["nms_fallback_count"], 0)

    def test_target_nms_only_records_deterministic_fallback(self) -> None:
        scores = np.arange(9, dtype=np.float32).reshape(3, 3)
        config = UGPruneConfig(selector_strategy="target_nms_only", role_mode="target_only", nms_radius=1)
        result = UGPruneSelector(config).select_from_map(
            sample_id="x", budget=9, original_tokens=9, grid=3, seed=0, inputs=inputs(scores)
        )
        self.assertEqual(result.selected_indices, list(range(9)))
        self.assertGreater(result.metadata["nms_fallback_count"], 0)
        self.assertEqual(result.metadata["budget_split"]["target"], 9)

    def test_structured_more_anchors_is_21_12_15(self) -> None:
        config = UGPruneConfig(selector_strategy="structured_tca", target_ratio=.45, context_ratio=.25, anchor_ratio=.30)
        self.assertEqual(budget_split(48, config), (21, 12, 15))

    def test_reference_and_torch_parity_for_all_strategies_and_ties(self) -> None:
        import torch

        rng = np.random.default_rng(811)
        for tied in (False, True):
            scores = np.zeros((24, 24), dtype=np.float32) if tied else rng.normal(size=(24, 24)).astype(np.float32)
            for strategy in ("query_topk", "target_nms_only", "structured_tca"):
                with self.subTest(tied=tied, strategy=strategy):
                    config = ug_gap58_config_for_prune_layer(
                        20, fusion_mode="single_layer", probe_layers=(20,), selector_strategy=strategy,
                        target_ratio=.45, context_ratio=.25, anchor_ratio=.30,
                    )
                    reference = UGPruneSelector(config).select_from_map(
                        sample_id="x", budget=48, original_tokens=576, grid=24, seed=0, inputs=inputs(scores)
                    )
                    forwarder = MidLayerUGPruneForwarder(
                        torch, object(), image_token_id=32000, grid=24, prune_after_layer=20,
                        target_visual_tokens=48, selector_config=config, diagnostics_mode="benchmark",
                    )
                    tensor = torch.tensor(scores.reshape(-1))
                    accelerated, retained = forwarder._selection_from_fused_torch(
                        scores=tensor, sample_id="x", original_tokens=576, seed=0,
                        map_key="layer20:query_max", map_stats={}, query_text="object",
                        query_token_status="matched", query_token_positions=[580],
                    )
                    self.assertEqual(accelerated.selected_indices, reference.selected_indices)
                    self.assertEqual(accelerated.role_labels, reference.role_labels)
                    self.assertEqual(retained.numel(), 48)

    def test_run_names_separate_strategies(self) -> None:
        base = dict(run_name=None, limit=None, decode_mode="layer_aware_cache", diagnostics_mode="benchmark",
                    selection_method="ug", ug_map_source="query_max", ug_fusion_mode="single_layer",
                    ug_probe_layers="20", position_mode="gap_preserving", ug_role_preset="more_anchors",
                    dataset="refcoco", prune_after_layer=20, seed=0, precision="bf16")
        names = {run_name(SimpleNamespace(**base, selector_strategy=strategy), 48) for strategy in ("query_topk", "target_nms_only", "structured_tca")}
        self.assertEqual(len(names), 3)

    def test_k12_and_k20_configs_preserve_gap_positions_and_cache_boundary(self) -> None:
        for layer in (12, 20):
            with self.subTest(layer=layer):
                config = ug_gap58_config_for_prune_layer(
                    layer, fusion_mode="single_layer", probe_layers=(layer,),
                    position_mode="gap_preserving", selector_strategy="structured_tca",
                )
                self.assertEqual(config.probe_layers, (layer,))
                self.assertEqual(config.position_mode, "gap_preserving")
                self.assertEqual(max(config.probe_layers), layer)


class ArtifactContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = [{"sample_id": "a"}, {"sample_id": "b"}, {"sample_id": "c"}]

    @staticmethod
    def row(sample_id: str) -> dict:
        return {"sample_id": sample_id, "target_visual_tokens": 2, "actual_llm_visual_tokens": 2,
                "retained_indices": [0, 1], "selector_strategy": "query_topk", "box_iou": 0.6,
                "success_iou_0_5": True, "raw_parsed_box": [0, 0, 1, 1]}

    def test_resume_reconstructs_manifest_order_and_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.jsonl"
            path.write_text(json.dumps(self.row("c")) + "\n" + json.dumps(self.row("a")) + "\n", encoding="utf-8")
            self.assertEqual([row["sample_id"] for row in load_resume_rows(path, self.manifest)], ["a", "c"])
            path.write_text(json.dumps(self.row("a")) + "\n" + json.dumps(self.row("a")) + "\n", encoding="utf-8")
            with self.assertRaises(ArtifactIntegrityError):
                load_resume_rows(path, self.manifest)

    def test_malformed_missing_extra_and_misordered_are_rejected(self) -> None:
        report = validate_rows([self.row("b"), self.row("a"), self.row("x")], self.manifest, budget=2, original_tokens=4, strategy="query_topk")
        self.assertEqual(report["status"], "fail")
        self.assertTrue(report["missing_ids"])
        self.assertTrue(report["unexpected_ids"])
        self.assertFalse(report["manifest_order_equal"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"sample_id": "a"}\n{"sample_id":', encoding="utf-8")
            with self.assertRaises(ArtifactIntegrityError):
                load_resume_rows(path, self.manifest)

    def test_atomic_finalization_requires_complete_and_publishes_canonical_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "partial.jsonl"
            final = Path(directory) / "final.jsonl"
            partial.write_text("\n".join(json.dumps(self.row(sid)) for sid in ("c", "a", "b")) + "\n", encoding="utf-8")
            rows, report = finalize_partial_jsonl(partial, final, self.manifest, budget=2, original_tokens=4, strategy="query_topk")
            self.assertEqual(report["status"], "pass")
            self.assertEqual([row["sample_id"] for row in rows], ["a", "b", "c"])

    def test_aggregate_mismatch_and_metrics_only_completion_are_rejected(self) -> None:
        rows = [self.row(sid) for sid in ("a", "b", "c")]
        metric_report = validate_metric_agreement(rows, {"samples/evaluated": 3, "quality/acc_at_0_5": 0.0}, task="grounding")
        self.assertEqual(metric_report["status"], "fail")
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "metrics.json").write_text(json.dumps({"samples/evaluated": 3}), encoding="utf-8")
            report = validate_completed_run(run, self.manifest, budget=2, original_tokens=4, task="grounding", strategy="query_topk")
            self.assertEqual(report["status"], "fail")

    def test_strong_campaign_does_not_accept_metrics_only(self) -> None:
        import scripts.run_strong_dissertation_ablations as strong

        with tempfile.TemporaryDirectory() as directory:
            previous = strong.RUNS_ROOT
            try:
                strong.RUNS_ROOT = Path(directory)
                run = strong.Run("metrics_only", "quality", 3, ("true",))
                run.run_dir.mkdir()
                (run.run_dir / "metrics.json").write_text(json.dumps({"samples/evaluated": 3}), encoding="utf-8")
                self.assertFalse(strong.is_complete(run))
            finally:
                strong.RUNS_ROOT = previous


if __name__ == "__main__":
    unittest.main()
