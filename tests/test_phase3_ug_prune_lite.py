from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class Phase3UGPruneLiteTests(unittest.TestCase):
    def test_model_forward_shapes(self) -> None:
        import torch

        from vlm_pruning.training.ug_prune_lite_model import UGPruneLiteModel, UGPruneLiteModelConfig

        model = UGPruneLiteModel.build(torch, UGPruneLiteModelConfig(visual_dim=32, query_dim=32, hidden_dim=64, coord_dim=16))
        outputs = model(torch.randn(2, 576, 32), torch.randn(2, 32))

        self.assertEqual(tuple(outputs["target_logits"].shape), (2, 576))
        self.assertEqual(tuple(outputs["context_logits"].shape), (2, 576))
        self.assertEqual(tuple(outputs["keep_logits"].shape), (2, 576))

    def test_selection_rule_returns_exact_budget_and_roles(self) -> None:
        import torch

        from vlm_pruning.selectors.ug_prune_lite import UGPruneLiteSelector

        selector = UGPruneLiteSelector(torch)
        target = torch.zeros(576)
        context = torch.zeros(576)
        keep = torch.zeros(576)
        target[10:50] = torch.arange(40).float()
        context[0:120] = torch.linspace(0, 1, 120)
        keep[:] = torch.linspace(0, 1, 576)

        result = selector.select(target, context, keep, sample_id="sample")

        self.assertEqual(len(result.selected_indices), 58)
        self.assertEqual(len(set(result.selected_indices)), 58)
        self.assertGreaterEqual(min(result.selected_indices), 0)
        self.assertLess(max(result.selected_indices), 576)
        self.assertEqual(len(result.role_labels["target"]), 29)
        self.assertEqual(len(result.role_labels["context"]), 17)
        self.assertEqual(len(result.role_labels["anchor"]), 12)

    def test_box_patch_labels_handles_normal_and_tiny_boxes(self) -> None:
        import torch

        from vlm_pruning.selectors.ug_prune_lite import box_patch_labels

        normal = box_patch_labels(torch, [100, 100, 200, 200], 640, 480)
        tiny = box_patch_labels(torch, [0, 0, 1, 1], 640, 480)

        self.assertEqual(tuple(normal.shape), (576,))
        self.assertGreater(float(normal.sum().item()), 0.0)
        self.assertGreater(float(tiny.sum().item()), 0.0)

    def _write_tiny_cache(self, root: Path, torch: object, count: int = 2, feature_dim: int = 8, score_maps: bool = False) -> None:
        (root / "features").mkdir(parents=True)
        (root / "metadata.json").write_text(
            json.dumps({"method": "ug_prune_lite_teacher_cache", "teacher_score_maps/enabled": score_maps}),
            encoding="utf-8",
        )
        rows = []
        for idx in range(count):
            payload = {
                "visual_features": torch.randn(576, feature_dim),  # type: ignore[attr-defined]
                "query_embedding": torch.randn(feature_dim),  # type: ignore[attr-defined]
            }
            if score_maps:
                payload["teacher_score_map"] = torch.linspace(-1.0, 1.0, 576)  # type: ignore[attr-defined]
            torch.save(  # type: ignore[attr-defined]
                payload,
                root / "features" / f"s{idx}.pt",
            )
            rows.append(
                {
                    "sample_id": f"s{idx}",
                    "gt_box": [10, 10, 30, 30],
                    "width": 100,
                    "height": 100,
                    "selected_indices": list(range(58)),
                    "role_labels": {
                        "target": list(range(29)),
                        "context": list(range(29, 46)),
                        "anchor": list(range(46, 58)),
                    },
                    "feature_path": f"features/s{idx}.pt",
                }
            )
        (root / "examples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_cache_dataset_loads_expected_fields(self) -> None:
        import torch

        from vlm_pruning.training.ug_prune_lite_dataset import UGPruneLiteCacheDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_tiny_cache(root, torch, count=1, feature_dim=16)

            dataset = UGPruneLiteCacheDataset(torch, root)
            item = dataset[0]

            self.assertEqual(tuple(item["visual_features"].shape), (576, 16))
            self.assertEqual(tuple(item["query_embedding"].shape), (16,))
            self.assertEqual(tuple(item["target_labels"].shape), (576,))
            self.assertEqual(float(item["keep_labels"].sum().item()), 58.0)
            self.assertEqual(tuple(item["teacher_score_map"].shape), (576,))
            self.assertEqual(float(item["has_teacher_score_map"].item()), 0.0)

    def test_cache_dataset_loads_teacher_score_maps_when_present(self) -> None:
        import torch

        from vlm_pruning.training.ug_prune_lite_dataset import UGPruneLiteCacheDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_tiny_cache(root, torch, count=1, feature_dim=16, score_maps=True)

            dataset = UGPruneLiteCacheDataset(torch, root)
            item = dataset[0]

            self.assertEqual(tuple(item["teacher_score_map"].shape), (576,))
            self.assertEqual(float(item["has_teacher_score_map"].item()), 1.0)
            self.assertAlmostEqual(float(item["teacher_score_map"][0].item()), -1.0, places=5)

    def test_cache_dataset_applies_checkpoint_projection_to_full_features(self) -> None:
        import torch

        from vlm_pruning.training.ug_prune_lite_dataset import UGPruneLiteCacheDataset
        from vlm_pruning.training.ug_prune_lite_features import random_projection_spec

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_tiny_cache(root, torch, count=1, feature_dim=16)

            dataset = UGPruneLiteCacheDataset(
                torch,
                root,
                feature_projection=random_projection_spec(input_dim=16, output_dim=4, seed=3),
            )
            item = dataset[0]

            self.assertEqual(tuple(item["visual_features"].shape), (576, 4))
            self.assertEqual(tuple(item["query_embedding"].shape), (4,))

    def test_checkpoint_save_load_roundtrip_preserves_outputs(self) -> None:
        import torch

        from vlm_pruning.training.ug_prune_lite_model import UGPruneLiteModel, UGPruneLiteModelConfig

        torch.manual_seed(0)
        config = UGPruneLiteModelConfig(visual_dim=8, query_dim=8, hidden_dim=16, coord_dim=8, dropout=0.0)
        model = UGPruneLiteModel.build(torch, config)
        visual = torch.randn(1, 576, 8)
        query = torch.randn(1, 8)
        before = model(visual, query)["keep_logits"].detach()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save({"model_config": config.as_dict(), "model_state_dict": model.state_dict()}, path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            loaded = UGPruneLiteModel.build(torch, UGPruneLiteModelConfig(**payload["model_config"]))
            loaded.load_state_dict(payload["model_state_dict"])
            after = loaded(visual, query)["keep_logits"].detach()

        self.assertTrue(torch.allclose(before, after))

    def test_runner_loads_projection_metadata_from_checkpoint(self) -> None:
        import torch

        from scripts.run_ug_prune_lite import load_student_bundle
        from vlm_pruning.training.ug_prune_lite_features import random_projection_spec
        from vlm_pruning.training.ug_prune_lite_model import UGPruneLiteModel, UGPruneLiteModelConfig

        config = UGPruneLiteModelConfig(visual_dim=4, query_dim=4, hidden_dim=16, coord_dim=8, dropout=0.0)
        model = UGPruneLiteModel.build(torch, config)
        spec = random_projection_spec(input_dim=16, output_dim=4, seed=5)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.pt"
            torch.save(
                {
                    "model_config": config.as_dict(),
                    "model_state_dict": model.state_dict(),
                    "train_config": {"feature_projection": spec},
                },
                checkpoint,
            )
            _loaded, loaded_spec = load_student_bundle(torch, checkpoint, torch.device("cpu"))

        self.assertEqual(loaded_spec, spec)

    def test_runner_selection_sanity_metrics(self) -> None:
        from scripts.run_ug_prune_lite import add_selection_sanity_metrics

        metrics: dict[str, float] = {}
        rows = [
            {
                "actual_retained_visual_tokens": 58,
                "retained_unique_visual_tokens": 58,
                "retained_duplicate_count": 0,
                "retained_out_of_range_count": 0,
                "retained_exact_budget": True,
            },
            {
                "actual_retained_visual_tokens": 57,
                "retained_unique_visual_tokens": 56,
                "retained_duplicate_count": 1,
                "retained_out_of_range_count": 1,
                "retained_exact_budget": False,
            },
        ]

        add_selection_sanity_metrics(metrics, rows, expected_budget=58)

        self.assertEqual(metrics["selection/exact_budget_rate"], 0.5)
        self.assertEqual(metrics["selection/duplicate_count_mean"], 0.5)
        self.assertEqual(metrics["selection/out_of_range_count_mean"], 0.5)

    def test_selector_metrics_include_recall_coverage_and_validity(self) -> None:
        import torch

        from vlm_pruning.training.ug_prune_lite_losses import selection_metrics_from_indices

        selected = list(range(58))
        teacher = list(range(29)) + list(range(100, 129))
        roles = {"target": list(range(29)), "context": list(range(100, 117)), "anchor": list(range(200, 212))}
        box_labels = torch.zeros(576)
        box_labels[:10] = 1.0

        metrics = selection_metrics_from_indices(selected, teacher, roles, box_labels)

        self.assertIn("selection/teacher_recall_at_58", metrics)
        self.assertIn("selection/gt_box_patch_coverage_at_58", metrics)
        self.assertIn("selection/occupied_3x3_bins", metrics)
        self.assertEqual(metrics["selection/unique_token_count"], 58.0)
        self.assertEqual(metrics["selection/duplicate_count"], 0.0)
        self.assertEqual(metrics["selection/out_of_range_count"], 0.0)

    def test_loss_metrics_use_requested_names(self) -> None:
        import torch

        from vlm_pruning.training.ug_prune_lite_losses import UGPruneLiteLossWeights, ug_prune_lite_loss

        outputs = {
            "target_logits": torch.randn(2, 576),
            "context_logits": torch.randn(2, 576),
            "keep_logits": torch.randn(2, 576),
        }
        batch = {
            "target_labels": torch.zeros(2, 576),
            "context_labels": torch.zeros(2, 576),
            "keep_labels": torch.zeros(2, 576),
            "box_labels": torch.zeros(2, 576),
        }
        _loss, metrics = ug_prune_lite_loss(torch, outputs, batch)

        self.assertIn("loss/target_bce", metrics)
        self.assertIn("loss/context_bce", metrics)
        self.assertIn("loss/keep_bce", metrics)
        self.assertIn("loss/score_map_distill", metrics)
        self.assertEqual(metrics["loss/score_map_distill_active"], 0.0)
        self.assertEqual(UGPruneLiteLossWeights().target, 1.5)
        self.assertEqual(UGPruneLiteLossWeights().gt_box, 2.0)

    def test_score_map_distillation_loss_activates_when_maps_present(self) -> None:
        import torch

        from vlm_pruning.training.ug_prune_lite_losses import UGPruneLiteLossWeights, ug_prune_lite_loss

        outputs = {
            "target_logits": torch.randn(2, 576),
            "context_logits": torch.randn(2, 576),
            "keep_logits": torch.randn(2, 576),
        }
        batch = {
            "target_labels": torch.zeros(2, 576),
            "context_labels": torch.zeros(2, 576),
            "keep_labels": torch.zeros(2, 576),
            "box_labels": torch.zeros(2, 576),
            "teacher_score_map": torch.randn(2, 576),
            "has_teacher_score_map": torch.tensor([1.0, 0.0]),
        }
        _loss, metrics = ug_prune_lite_loss(
            torch,
            outputs,
            batch,
            UGPruneLiteLossWeights(score_map_distill=0.5, score_map_temperature=2.0, score_map_loss="kl"),
        )

        self.assertGreater(metrics["loss/score_map_distill"], 0.0)
        self.assertEqual(metrics["loss/score_map_distill_active"], 0.5)

    def test_selector_does_not_require_hidden_states(self) -> None:
        import torch

        from vlm_pruning.selectors.ug_prune_lite import UGPruneLiteSelector

        selector = UGPruneLiteSelector(torch)
        result = selector.select(torch.randn(576), torch.randn(576), torch.randn(576))

        self.assertEqual(len(result.selected_indices), 58)

    def test_checkpoint_comparison_joins_matched_samples(self) -> None:
        def write_jsonl(path: Path, rows: list[dict]) -> None:
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher = root / "teacher"
            left_selector = root / "left_selector"
            right_selector = root / "right_selector"
            left_generation = root / "left_generation"
            right_generation = root / "right_generation"
            output = root / "analysis"
            for path in [teacher, left_selector, right_selector, left_generation, right_generation]:
                path.mkdir()

            teacher_rows = [
                {
                    "sample_id": "s0",
                    "image_id": "img0",
                    "query": "left cup",
                    "gt_box": [0, 0, 10, 10],
                    "width": 100,
                    "height": 100,
                    "selected_indices": list(range(58)),
                    "role_labels": {"target": list(range(29)), "context": list(range(29, 46)), "anchor": list(range(46, 58))},
                },
                {
                    "sample_id": "s1",
                    "image_id": "img1",
                    "query": "right cup",
                    "gt_box": [0, 0, 10, 10],
                    "width": 100,
                    "height": 100,
                    "selected_indices": list(range(58)),
                    "role_labels": {"target": list(range(29)), "context": list(range(29, 46)), "anchor": list(range(46, 58))},
                },
            ]
            selector_row = {
                "selected_indices": list(range(58)),
                "selection/gt_box_patch_coverage_at_58": 0.5,
                "selection/occupied_3x3_bins": 9.0,
                "selection/mean_pairwise_spread": 1.0,
            }
            write_jsonl(teacher / "examples.jsonl", teacher_rows)
            write_jsonl(
                left_selector / "per_sample_selector_results.jsonl",
                [{"sample_id": "s0", **selector_row}, {"sample_id": "s1", **selector_row}],
            )
            write_jsonl(
                right_selector / "per_sample_selector_results.jsonl",
                [{"sample_id": "s0", **selector_row}, {"sample_id": "s1", **selector_row}],
            )
            generation_rows = [
                {
                    "sample_id": "s0",
                    "raw_model_output": "[0,0,1,1]",
                    "predicted_box": [0, 0, 10, 10],
                    "box_iou": 0.75,
                    "success_iou_0_5": True,
                    "valid": True,
                    "invalid": False,
                    "latency_s": 0.1,
                },
                {
                    "sample_id": "s1",
                    "raw_model_output": "[0,0,1,1]",
                    "predicted_box": [0, 0, 1, 1],
                    "box_iou": 0.1,
                    "success_iou_0_5": False,
                    "valid": True,
                    "invalid": False,
                    "latency_s": 0.1,
                },
            ]
            write_jsonl(left_generation / "per_sample_results.jsonl", generation_rows)
            write_jsonl(right_generation / "per_sample_results.jsonl", generation_rows)

            subprocess.run(
                [
                    sys.executable,
                    "scripts/compare_ug_prune_lite_checkpoints.py",
                    "--teacher-cache",
                    str(teacher),
                    "--left-selector-eval",
                    str(left_selector),
                    "--left-generation",
                    str(left_generation),
                    "--right-selector-eval",
                    str(right_selector),
                    "--right-generation",
                    str(right_generation),
                    "--output-dir",
                    str(output),
                    "--limit",
                    "1",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )

            summary = json.loads((output / "comparison_summary.json").read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in (output / "comparison_per_sample.jsonl").read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary["samples/compared"], 1)
            self.assertEqual(rows[0]["sample_id"], "s0")
            self.assertTrue((output / "failure_flips.csv").exists())
            self.assertTrue((output / "spatial_distribution_summary.csv").exists())

    def test_manifest_builder_use_all_allowed_holds_out_dev(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "refcoco_val.jsonl"
            rows = [
                {
                    "sample_id": f"s{i}",
                    "dataset": "refcoco",
                    "split": "val",
                    "image_id": f"image{i}",
                    "image_path": f"image{i}.jpg",
                    "query": "object",
                    "gt_box": [0, 0, 10, 10],
                    "width": 20,
                    "height": 20,
                }
                for i in range(12)
            ]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            out = root / "out"

            subprocess.run(
                [
                    sys.executable,
                    "scripts/build_ug_prune_lite_manifests.py",
                    "--source-dev-manifest",
                    str(manifest),
                    "--reserved-eval-manifest",
                    str(manifest),
                    "--reserved-eval-limit",
                    "2",
                    "--dev-limit",
                    "3",
                    "--use-all-allowed",
                    "--output-dir",
                    str(out),
                    "--allow-nonfinal-val-fallback",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))

            self.assertEqual(metadata["train_count"], 7)
            self.assertEqual(metadata["dev_count"], 3)
            self.assertEqual(metadata["overlap/train_reserved_eval"], 0)
            self.assertEqual(metadata["overlap/dev_reserved_eval"], 0)

    def test_teacher_cache_plan_reports_disk_estimate(self) -> None:
        from scripts.build_ug_prune_lite_teacher_cache import cache_plan

        with tempfile.TemporaryDirectory() as tmp:
            plan = cache_plan(
                Path(tmp) / "cache",
                examples=10,
                cache_size_mb_per_example=4.56,
                seconds_per_example=0.37,
                min_free_disk_gb=0.001,
            )

            self.assertEqual(plan["examples"], 10)
            self.assertGreater(plan["estimated_cache_size_bytes"], 0)
            self.assertIn("proceed", plan)

    def test_train_script_writes_best_and_last_checkpoints(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_cache = root / "train_cache"
            val_cache = root / "val_cache"
            out = root / "out"
            self._write_tiny_cache(train_cache, torch, count=2, feature_dim=8)
            self._write_tiny_cache(val_cache, torch, count=2, feature_dim=8)

            subprocess.run(
                [
                    sys.executable,
                    "scripts/train_ug_prune_lite.py",
                    "--teacher-cache",
                    str(train_cache),
                    "--val-cache",
                    str(val_cache),
                    "--output-dir",
                    str(out),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "1",
                    "--device",
                    "cpu",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                stdout=subprocess.DEVNULL,
            )

            self.assertTrue((out / "checkpoint_best.pt").exists())
            self.assertTrue((out / "checkpoint_best_selector.pt").exists())
            self.assertTrue((out / "checkpoint_last.pt").exists())
            self.assertTrue((out / "validation_log.jsonl").exists())

    def test_generation_checkpoint_selector_refuses_reserved_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "reserved.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "reserved0",
                        "metadata": {"source_row": 0},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/select_ug_prune_lite_checkpoint_by_generation.py",
                    "--train-dir",
                    str(root / "train"),
                    "--manifest",
                    str(manifest),
                    "--reserved-manifest",
                    str(manifest),
                    "--reserved-limit",
                    "1",
                    "--output-dir",
                    str(root / "out"),
                    "--limit",
                    "1",
                    "--dry-run",
                    "--wandb-mode",
                    "disabled",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("reserved final rows", result.stderr)

    def test_generation_checkpoint_selector_dry_run_discovers_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train"
            train.mkdir()
            for name in ["checkpoint_best.pt", "checkpoint_last.pt", "checkpoint_epoch_001.pt", "checkpoint_epoch_002.pt"]:
                (train / name).write_bytes(b"")
            (train / "metrics.json").write_text(json.dumps({"best/epoch": 2}), encoding="utf-8")
            validation_rows = [
                {
                    "epoch": 1,
                    "val/selection/teacher_recall_at_58": 0.2,
                    "val/selection/gt_box_patch_coverage_at_58": 0.1,
                    "val/loss/total": 2.0,
                },
                {
                    "epoch": 2,
                    "val/selection/teacher_recall_at_58": 0.4,
                    "val/selection/gt_box_patch_coverage_at_58": 0.2,
                    "val/loss/total": 1.5,
                },
            ]
            (train / "validation_log.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in validation_rows),
                encoding="utf-8",
            )
            manifest = root / "allowed.jsonl"
            manifest.write_text(
                json.dumps({"sample_id": "allowed0", "metadata": {"source_row": 500}}) + "\n",
                encoding="utf-8",
            )
            reserved = root / "reserved.jsonl"
            reserved.write_text(
                json.dumps({"sample_id": "reserved0", "metadata": {"source_row": 0}}) + "\n",
                encoding="utf-8",
            )
            out = root / "selection"

            subprocess.run(
                [
                    sys.executable,
                    "scripts/select_ug_prune_lite_checkpoint_by_generation.py",
                    "--train-dir",
                    str(train),
                    "--manifest",
                    str(manifest),
                    "--reserved-manifest",
                    str(reserved),
                    "--reserved-limit",
                    "1",
                    "--output-dir",
                    str(out),
                    "--limit",
                    "1",
                    "--top-k-selector-checkpoints",
                    "2",
                    "--dry-run",
                    "--wandb-mode",
                    "disabled",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                stdout=subprocess.DEVNULL,
            )

            payload = json.loads((out / "generation_model_selection.json").read_text(encoding="utf-8"))

            self.assertEqual(payload["reserved_guard"]["reserved_violation_count"], 0)
            self.assertGreaterEqual(len(payload["candidates"]), 2)
            self.assertTrue((out / "candidate_generation_results.csv").exists())
            self.assertEqual((out / "best_generation_checkpoint.txt").read_text(encoding="utf-8").strip(), payload["best"]["checkpoint"])


if __name__ == "__main__":
    unittest.main()
