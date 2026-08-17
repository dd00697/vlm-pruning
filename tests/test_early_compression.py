from __future__ import annotations

import types
import unittest

import torch

from vlm_pruning.llava_hooks.early_compression import (
    ContiguousPoolForwarder,
    area_weighted_spatial_pool,
    nearest_cell_anchors,
    prepare_pooled_inputs,
)

from scripts.analyse_early_compression_quality import paired_interval


class _Processor:
    def __call__(self, *, images, text, return_tensors):
        self.last_text = text
        return {
            "input_ids": torch.tensor([[7, 99, 99, 99, 99, 8]], dtype=torch.long),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
        }


class _VisionTower(torch.nn.Module):
    def forward(self, pixel_values, *, output_hidden_states, return_dict):
        patches = torch.tensor(
            [[[100.0], [1.0], [2.0], [3.0], [4.0]]], dtype=pixel_values.dtype
        )
        return types.SimpleNamespace(hidden_states=[patches, patches])


class _PrepareModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.config = types.SimpleNamespace(
            vision_feature_layer=-2,
            vision_feature_select_strategy="default",
            image_token_index=99,
        )
        self.model = types.SimpleNamespace(
            vision_tower=_VisionTower(), multi_modal_projector=torch.nn.Identity()
        )
        self.embedding = torch.nn.Embedding(128, 1)

    def get_input_embeddings(self):
        return self.embedding


class _Cache:
    def __init__(self, length: int, layers: int = 2):
        self.key_cache = [torch.zeros((1, 1, length, 1)) for _ in range(layers)]
        self.value_cache = [torch.zeros((1, 1, length, 1)) for _ in range(layers)]


class _GenerationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.position_history: list[list[int]] = []

    def forward(
        self,
        *,
        inputs_embeds=None,
        input_ids=None,
        attention_mask,
        position_ids,
        past_key_values=None,
        use_cache,
        return_dict,
        logits_to_keep,
    ):
        self.position_history.append(position_ids.detach().cpu().reshape(-1).tolist())
        if past_key_values is None:
            length = int(inputs_embeds.shape[1])
        else:
            length = int(past_key_values.key_cache[0].shape[-2]) + 1
        logits = torch.zeros((1, 1, 16), dtype=torch.float32)
        logits[..., 3] = 1.0
        return types.SimpleNamespace(logits=logits, past_key_values=_Cache(length))


class EarlyCompressionTests(unittest.TestCase):
    def test_registered_anchor_axes_are_deterministic_and_raster_ordered(self):
        anchors16 = nearest_cell_anchors(24, 16)
        axis16 = [index % 24 for index in anchors16[:16]]
        self.assertEqual(axis16, [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22])
        anchors14 = nearest_cell_anchors(24, 14)
        axis14 = [index % 24 for index in anchors14[:14]]
        self.assertEqual(axis14, [0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22])
        self.assertEqual(anchors16, sorted(anchors16))
        self.assertEqual(len(set(anchors14)), 196)

    def test_area_pool_conserves_mass_and_every_patch_contributes(self):
        features = torch.arange(576, dtype=torch.float32).reshape(1, 576, 1)
        features.requires_grad_(True)
        pooled, evidence = area_weighted_spatial_pool(
            torch, features, input_grid=24, output_grid=16
        )
        self.assertEqual(tuple(pooled.shape), (1, 256, 1))
        self.assertAlmostEqual(evidence.input_mass, 576.0, places=9)
        self.assertAlmostEqual(evidence.output_mass, 576.0, places=9)
        self.assertEqual(len(evidence.output_masses), 256)
        pooled.sum().backward()
        self.assertTrue(torch.all(features.grad > 0))

    def test_area_pool_preserves_constant_features_for_both_registered_grids(self):
        features = torch.full((1, 576, 3), 7.25, dtype=torch.float32)
        for grid in (16, 14):
            pooled, evidence = area_weighted_spatial_pool(
                torch, features, input_grid=24, output_grid=grid
            )
            self.assertTrue(torch.allclose(pooled, torch.full_like(pooled, 7.25)))
            self.assertAlmostEqual(sum(evidence.output_masses), 576.0, places=9)

    def test_prepare_pooled_inputs_projects_only_pool_and_preserves_position_gaps(self):
        model = _PrepareModel()
        compacted, diagnostics = prepare_pooled_inputs(
            torch,
            _Processor(),
            model,
            torch.device("cpu"),
            [object()],
            "registered prompt",
            output_grid=1,
            input_grid=2,
        )
        self.assertEqual(compacted["input_ids"].tolist(), [[7, 99, 8]])
        self.assertEqual(compacted["position_ids"].tolist(), [[0, 1, 5]])
        self.assertEqual(compacted["full_prompt_length"], 6)
        self.assertEqual(diagnostics["pool_anchor_indices"], [0])
        self.assertEqual(diagnostics["pool_anchor_positions"], [1])
        self.assertEqual(diagnostics["text_token_count"], 2)
        self.assertAlmostEqual(diagnostics["pool_output_mass"], 4.0)

    def test_contiguous_cache_grows_and_decode_positions_follow_full_prompt(self):
        model = _GenerationModel()
        forwarder = ContiguousPoolForwarder(torch, model)
        pooled_inputs = {
            "input_ids": torch.tensor([[7, 99, 8]], dtype=torch.long),
            "inputs_embeds": torch.ones((1, 3, 4)),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 5]], dtype=torch.long),
            "full_prompt_length": 6,
        }
        generated, diagnostics = forwarder.generate_natural(
            pooled_inputs, max_new_tokens=3, eos_ids=set()
        )
        self.assertEqual(generated.tolist(), [3, 3, 3])
        self.assertEqual(diagnostics["cache/prompt_per_layer_lengths"], [3, 3])
        self.assertEqual(diagnostics["cache/final_per_layer_lengths"], [5, 5])
        self.assertEqual(diagnostics["cache/generated_tokens_cached"], 2)
        self.assertEqual(model.position_history, [[0, 1, 5], [6], [7]])

    def test_fixed_decode_executes_every_registered_token(self):
        model = _GenerationModel()
        forwarder = ContiguousPoolForwarder(torch, model)
        pooled_inputs = {
            "input_ids": torch.tensor([[7, 99, 8]], dtype=torch.long),
            "inputs_embeds": torch.ones((1, 3, 4)),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 5]], dtype=torch.long),
            "full_prompt_length": 6,
        }
        generated, diagnostics = forwarder.generate_fixed(
            pooled_inputs, token_ids=[10, 11, 12]
        )
        self.assertEqual(generated.tolist(), [10, 11, 12])
        self.assertEqual(diagnostics["decode_step_count"], 3)
        self.assertEqual(diagnostics["cache/final_per_layer_lengths"], [6, 6])
        self.assertEqual(model.position_history, [[0, 1, 5], [6], [7], [8]])

    def test_quality_bootstrap_is_paired_and_deterministic(self):
        candidate = torch.tensor([1.0, 0.0, 1.0, 0.0]).numpy()
        reference = torch.tensor([1.0, 1.0, 0.0, 0.0]).numpy()
        first = paired_interval(candidate, reference, seed=19)
        second = paired_interval(candidate, reference, seed=19)
        self.assertEqual(first, second)
        self.assertEqual(first["point_delta"], 0.0)
        self.assertEqual(first["paired_rows"], 4)


if __name__ == "__main__":
    unittest.main()
