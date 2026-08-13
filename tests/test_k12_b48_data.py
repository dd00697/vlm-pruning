from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_k12_b48_data import (
    _images_as_bytes,
    answer_strings,
    image_record,
    mmbench_choices,
    parse_choices,
)
from scripts.run_baseline import retained_indices
from vlm_pruning.artifacts import validate_rows
from vlm_pruning.llava_hooks.mid_layer_pruning import MidLayerUGPruneForwarder, ug_gap58_config_for_prune_layer


class K12B48DataTests(unittest.TestCase):
    def test_image_id_is_never_cast_as_an_image_column(self) -> None:
        class FakeDataset:
            column_names = ["image", "image_1", "image_id", "image_width"]

            def __init__(self) -> None:
                self.cast = []

            def cast_column(self, name, value):
                self.cast.append((name, value))
                return self

        class FakeImage:
            def __init__(self, *, decode):
                self.decode = decode

        dataset = FakeDataset()
        _images_as_bytes(dataset, FakeImage)
        self.assertEqual([name for name, _ in dataset.cast], ["image", "image_1"])

    def test_parse_choices(self) -> None:
        self.assertEqual(parse_choices("['one', 'two']"), ["one", "two"])
        self.assertEqual(parse_choices([1, 2]), ["1", "2"])
        self.assertEqual(parse_choices(None), [])

    def test_mmbench_nan_choices_are_removed(self) -> None:
        row = {"A": "yes", "B": "no", "C": "nan", "D": None}
        self.assertEqual(mmbench_choices(row), ["yes", "no"])

    def test_answer_strings_support_vqa_dicts(self) -> None:
        self.assertEqual(answer_strings([{"answer": "cat"}, {"answer": "cat"}]), ["cat", "cat"])

    def test_image_storage_preserves_source_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            image = Image.new("RGB", (5, 7), color=(10, 20, 30))
            image.save(source, format="PNG")
            payload = source.read_bytes()
            record = image_record({"bytes": payload, "path": "source.png"}, Path(directory) / "images")
            stored = Path(record["image_path"])
            if not stored.is_absolute():
                from scripts.prepare_k12_b48_data import ROOT

                stored = ROOT / stored
            self.assertEqual(stored.read_bytes(), payload)
            self.assertEqual((record["width"], record["height"]), (5, 7))

    def test_multi_image_uniform_control_has_global_exact_budget(self) -> None:
        selected = retained_indices("uniform", 48, 1152, 0, "sample", 24)
        self.assertEqual(len(selected), 48)
        self.assertEqual(len(set(selected)), 48)
        self.assertEqual(sum(index < 576 for index in selected), 24)
        self.assertEqual(sum(index >= 576 for index in selected), 24)

    def test_multi_image_grid_never_wraps_context_between_images(self) -> None:
        import torch

        forwarder = object.__new__(MidLayerUGPruneForwarder)
        forwarder.torch = torch
        forwarder.grid = 24
        forwarder.selector_config = ug_gap58_config_for_prune_layer(12)
        forwarder._grid_tensor_cache = {}
        tensors = forwarder._grid_tensors(torch.device("cpu"), image_count=2)
        self.assertEqual(tensors["indices"].numel(), 1152)
        self.assertEqual(tensors["num_bins"], 18)
        self.assertFalse(bool(tensors["neighbor_mask"][575, 576]))

    def test_artifact_budget_validation_uses_per_sample_original_count(self) -> None:
        manifest = [{"sample_id": "full"}, {"sample_id": "qtac"}]
        rows = [
            {
                "sample_id": "full",
                "method": "full",
                "original_visual_tokens": 1152,
                "target_visual_tokens": 1152,
                "actual_llm_visual_tokens": 1152,
                "retained_indices": list(range(1152)),
            },
            {
                "sample_id": "qtac",
                "method": "ug_prune_gap58_mid_layer_single_pass",
                "original_visual_tokens": 1152,
                "target_visual_tokens": 48,
                "actual_llm_visual_tokens": 48,
                "retained_indices": list(range(1104, 1152)),
            },
        ]
        report = validate_rows(rows, manifest, budget=48, original_tokens=576)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(
            validate_rows(rows[:1], manifest[:1], budget=576, original_tokens=576)["status"],
            "pass",
        )
        self.assertEqual(
            validate_rows(rows[1:], manifest[1:], budget=48, original_tokens=576)["status"],
            "pass",
        )


if __name__ == "__main__":
    unittest.main()
