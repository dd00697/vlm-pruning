import math
import unittest
from pathlib import Path

import torch

from vlm_pruning.llava_hooks.feather_pruning import (
    FEATHER_BOUNDARIES,
    FEATHER_KEEP_RATIO,
    FEATHER_STRIDE,
    FeatherLlavaForwarder,
    feather_select_original_indices,
)
from scripts.validate_progressive_oracle_run import (
    expected_stage_specs,
    expected_visual_trace,
)


def upstream_fixture(scores, incoming, stage):
    """Independent transcription of the pinned upstream lines 142--164."""
    if stage == 0:
        ranked = scores.topk(int(len(incoming) * FEATHER_KEEP_RATIO)).indices
        size = int(math.sqrt(len(incoming)))
        grid = torch.arange(len(incoming)).reshape(size, size)
        uniform = grid[::FEATHER_STRIDE, ::FEATHER_STRIDE].flatten()
        local = torch.unique(torch.cat((ranked, uniform)), sorted=True)
    else:
        ranked = scores.topk(
            int(len(incoming) * FEATHER_KEEP_RATIO * FEATHER_KEEP_RATIO)
        ).indices
        local = ranked.sort().values
    return [incoming[int(index)] for index in local.tolist()]


class FeatherSelectionTests(unittest.TestCase):
    def test_frozen_upstream_configuration(self):
        self.assertEqual(FEATHER_BOUNDARIES, (8, 16))
        self.assertAlmostEqual(FEATHER_KEEP_RATIO, 0.3)
        self.assertEqual(FEATHER_STRIDE, 3)

    def test_both_stages_match_upstream_fixture_exactly(self):
        generator = torch.Generator().manual_seed(20260818)
        first_scores = torch.rand(576, generator=generator)
        incoming = list(range(576))
        first, first_meta = feather_select_original_indices(
            torch, first_scores, incoming, stage_index=0
        )
        self.assertEqual(first, upstream_fixture(first_scores, incoming, 0))
        second_scores = torch.rand(len(first), generator=generator)
        second, second_meta = feather_select_original_indices(
            torch, second_scores, first, stage_index=1
        )
        self.assertEqual(second, upstream_fixture(second_scores, first, 1))
        self.assertEqual(first_meta["attention_topk_count"], 172)
        self.assertEqual(first_meta["uniform_stride_count"], 64)
        self.assertEqual(second_meta["attention_topk_count"], int(len(first) * 0.09))
        self.assertEqual(first, sorted(first))
        self.assertEqual(second, sorted(second))

    def test_adapter_source_matches_pinned_official_operations(self):
        source = Path(
            "/workspace/third_party_official/FEATHER/prismatic-vlms/prismatic/models/"
            "backbones/llm/llama2_models.py"
        ).read_text(encoding="utf-8")
        for fragment in (
            "self.fastV_ratio * self.fastV_ratio",
            "torch.unique(combined_stride_fast_indices, sorted=True",
            "fastv_predefined_mask.replace('no_rope_s=', '')",
            "text_query_states = query_states[:,:,-1:,:]",
        ):
            self.assertIn(fragment, source)

    def test_rope_free_score_matches_manual_identity_projection(self):
        class Identity(torch.nn.Module):
            def forward(self, value):
                return value

        class Attention:
            q_proj = Identity()
            k_proj = Identity()
            num_heads = 2
            num_key_value_heads = 2
            head_dim = 2

        class Layer:
            input_layernorm = Identity()
            self_attn = Attention()

        forwarder = object.__new__(FeatherLlavaForwarder)
        forwarder.torch = torch
        forwarder.image_token_id = 99
        forwarder.model = type(
            "Model",
            (),
            {
                "model": type(
                    "Inner",
                    (),
                    {
                        "language_model": type(
                            "LM",
                            (),
                            {
                                "layers": [Layer()],
                                "config": type(
                                    "Cfg",
                                    (),
                                    {"num_attention_heads": 2, "num_key_value_heads": 2},
                                )(),
                            },
                        )()
                    },
                )()
            },
        )()
        hidden = torch.tensor(
            [[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]]
        )
        scores = forwarder._rope_free_attention_scores(
            hidden, input_ids=torch.tensor([[99, 99, 7]]), layer_index=0
        )
        q = hidden.view(1, 3, 2, 2).transpose(1, 2)[:, :, -1:, :]
        k = hidden.view(1, 3, 2, 2).transpose(1, 2)[:, :, :2, :]
        expected = torch.softmax(q @ k.transpose(2, 3) / math.sqrt(2), dim=-1).mean(1)[0, 0]
        torch.testing.assert_close(scores, expected)

    def test_adaptive_validator_derives_each_physical_trace(self):
        row = {
            "stage_traces": [
                {"realized_budget": 217},
                {"realized_budget": 19},
            ]
        }
        config = {"method/name": "feather_llava_main"}
        self.assertEqual(
            expected_visual_trace(config, row),
            [576] * 8 + [217] * 8 + [19] * 16,
        )
        specs = expected_stage_specs(config, row)
        self.assertEqual(
            [(item["boundary"], item["budget"], item["selector"]) for item in specs],
            [
                (8, 217, "feather_no_rope_topk_union_stride3"),
                (16, 19, "feather_no_rope_squared_keep_ratio"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
