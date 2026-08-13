from __future__ import annotations

import unittest

from scripts.audit_llava_checkpoint_equivalence import semantic_canonical_name, tensor_digest


class LLaVACheckpointEquivalenceTests(unittest.TestCase):
    def test_language_model_packaging_prefix_is_removed(self) -> None:
        self.assertEqual(
            semantic_canonical_name("language_model.model.layers.12.self_attn.q_proj.weight"),
            "model.layers.12.self_attn.q_proj.weight",
        )

    def test_projector_layers_map_to_original_sequential_names(self) -> None:
        self.assertEqual(
            semantic_canonical_name("multi_modal_projector.linear_2.bias"),
            "model.mm_projector.2.bias",
        )

    def test_vision_tower_uses_separate_clip_namespace(self) -> None:
        self.assertEqual(
            semantic_canonical_name("vision_tower.vision_model.post_layernorm.weight"),
            "clip::vision_model.post_layernorm.weight",
        )

    def test_runtime_dtype_normalisation_matches_fp16_serialisation(self) -> None:
        import torch

        source = torch.tensor([1.125, -2.25], dtype=torch.float32)
        expected = source.to(torch.float16)
        self.assertEqual(
            tensor_digest(source, target_dtype="torch.float16"),
            tensor_digest(expected),
        )


if __name__ == "__main__":
    unittest.main()
