from __future__ import annotations

import unittest

import numpy as np

from vlm_pruning.features.map_stats import (
    local_maxima_count,
    map_statistics,
    pearson_correlation,
    top_k_indices,
    zscore_normalise,
)
from vlm_pruning.features.query_tokens import resolve_query_token_positions

try:
    import torch
except Exception:
    torch = None


class FakeTokenizer:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"USER:": 1, "<image>": 32000, "ASSISTANT:": 2}

    def _id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab) + 10
        return self.vocab[token]

    def __call__(self, text: str, return_offsets_mapping: bool = False, add_special_tokens: bool = True):
        del add_special_tokens
        tokens: list[str] = []
        offsets: list[tuple[int, int]] = []
        start = None
        for idx, char in enumerate(text):
            if char.isspace() or char in {'"', ".", "\n"}:
                if start is not None:
                    tokens.append(text[start:idx])
                    offsets.append((start, idx))
                    start = None
            elif start is None:
                start = idx
        if start is not None:
            tokens.append(text[start:])
            offsets.append((start, len(text)))
        payload = {"input_ids": [self._id(token) for token in tokens]}
        if return_offsets_mapping:
            payload["offset_mapping"] = offsets
        return payload

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        inverse = {value: key for key, value in self.vocab.items()}
        return [inverse[token_id] for token_id in token_ids]


class SlowFakeTokenizer(FakeTokenizer):
    def __call__(self, text: str, return_offsets_mapping: bool = False, add_special_tokens: bool = True):
        if return_offsets_mapping:
            raise NotImplementedError("slow tokenizer has no offsets")
        return super().__call__(text, return_offsets_mapping=False, add_special_tokens=add_special_tokens)


class QueryTokenResolverTests(unittest.TestCase):
    def test_resolves_query_positions_in_actual_input_ids(self) -> None:
        tokenizer = FakeTokenizer()
        prompt = 'USER: <image>\nFind "red cup".\nASSISTANT:'
        encoded = tokenizer(prompt, return_offsets_mapping=True)
        image_token = tokenizer.vocab["<image>"]
        actual_ids = []
        for token_id in encoded["input_ids"]:
            if token_id == image_token:
                actual_ids.extend([image_token] * 4)
            else:
                actual_ids.append(token_id)

        result = resolve_query_token_positions(tokenizer, prompt, "red cup", input_ids=actual_ids, image_token_id=image_token)

        self.assertEqual(result.resolver_status, "matched_actual_input_subsequence")
        self.assertEqual(result.token_texts, ["red", "cup"])
        self.assertEqual([actual_ids[pos] for pos in result.token_positions], result.token_ids)

    def test_missing_query_is_logged(self) -> None:
        tokenizer = FakeTokenizer()
        result = resolve_query_token_positions(tokenizer, "USER: <image>\nFind a mug.", "red cup")
        self.assertEqual(result.resolver_status, "query_not_found")
        self.assertEqual(result.token_positions, [])

    def test_slow_tokenizer_aligns_exact_query_subsequence(self) -> None:
        tokenizer = SlowFakeTokenizer()
        prompt = 'USER: <image> Find "red cup". ASSISTANT:'
        actual_ids = tokenizer(prompt)["input_ids"]
        result = resolve_query_token_positions(
            tokenizer,
            prompt,
            "red cup",
            input_ids=actual_ids,
            image_token_id=32000,
        )
        self.assertEqual(result.resolver_status, "matched_slow_tokenizer_subsequence")
        self.assertEqual([actual_ids[pos] for pos in result.token_positions], result.token_ids)


class MapStatsTests(unittest.TestCase):
    def test_zscore_constant_map_is_zero(self) -> None:
        out = zscore_normalise(np.ones((24, 24), dtype=np.float32))
        self.assertTrue(np.allclose(out, 0.0))

    def test_top_k_indices_descend(self) -> None:
        values = np.arange(24 * 24, dtype=np.float32).reshape(24, 24)
        self.assertEqual(top_k_indices(values, 3), [575, 574, 573])

    def test_map_stats_include_core_metrics(self) -> None:
        values = np.zeros((24, 24), dtype=np.float32)
        values[5, 6] = 10.0
        stats = map_statistics(values, top_ks=(16,), grid=24)
        self.assertGreater(float(stats["peakiness"]), 0.0)
        self.assertEqual(int(stats["local_maxima_count"]), 1)
        self.assertIn("top_16_mass", stats)

    def test_local_maxima_ignores_flat_map(self) -> None:
        self.assertEqual(local_maxima_count(np.ones((24, 24), dtype=np.float32)), 0)

    def test_pearson_handles_constant_arrays(self) -> None:
        self.assertIsNone(pearson_correlation(np.ones(4), np.arange(4)))
        self.assertAlmostEqual(float(pearson_correlation(np.arange(4), np.arange(4))), 1.0)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class UnrolledMapTests(unittest.TestCase):
    def test_hidden_state_maps_are_24_by_24(self) -> None:
        from vlm_pruning.features.unrolled_maps import extract_maps_from_hidden_states

        image_token_id = 32000
        text_tokens = torch.tensor([[1, 2, 3, 4, 5]])
        image_tokens = torch.full((1, 576), image_token_id)
        input_ids = torch.cat([text_tokens[:, :2], image_tokens, text_tokens[:, 2:]], dim=1)
        hidden = torch.randn(1, input_ids.shape[1], 8)
        hidden_states = [hidden.clone() for _ in range(5)]
        query_positions = [579, 580]

        results = extract_maps_from_hidden_states(hidden_states, input_ids, image_token_id, query_positions, [4])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].maps["query_max"].shape, (24, 24))
        self.assertEqual(results[0].maps_z["query_pooled"].shape, (24, 24))


if __name__ == "__main__":
    unittest.main()
