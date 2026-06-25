"""Compute LLaVA intermediate query-image similarity maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .map_stats import zscore_normalise


@dataclass(frozen=True)
class LayerMapResult:
    layer: int
    image_positions: list[int]
    query_positions: list[int]
    maps: dict[str, np.ndarray]
    maps_z: dict[str, np.ndarray]


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"PyTorch is required for hidden-state map extraction: {exc}") from exc
    return torch


def visual_token_positions(input_ids: Any, image_token_id: int) -> list[int]:
    torch = _require_torch()
    ids = input_ids[0] if getattr(input_ids, "ndim", 1) == 2 else input_ids
    if not torch.is_tensor(ids):
        ids = torch.tensor(ids)
    return [int(idx) for idx in torch.nonzero(ids == int(image_token_id), as_tuple=False).flatten().tolist()]


def _normalise_states(states: Any) -> Any:
    torch = _require_torch()
    states = torch.nn.functional.layer_norm(states, (states.shape[-1],))
    return torch.nn.functional.normalize(states.float(), p=2, dim=-1)


def compute_query_image_scores(image_states: Any, query_states: Any) -> dict[str, Any]:
    torch = _require_torch()
    if int(query_states.shape[0]) <= 0:
        raise ValueError("At least one query token state is required.")
    image_norm = _normalise_states(image_states)
    query_norm = _normalise_states(query_states)
    sims = image_norm @ query_norm.transpose(0, 1)
    pooled_query = torch.nn.functional.normalize(query_norm.mean(dim=0, keepdim=True), p=2, dim=-1)
    return {
        "query_max": sims.max(dim=1).values,
        "query_pooled": (image_norm @ pooled_query.transpose(0, 1)).squeeze(1),
    }


def _to_grid(scores: Any, grid: int) -> np.ndarray:
    arr = scores.detach().float().cpu().numpy().astype(np.float32)
    if arr.size != grid * grid:
        raise ValueError(f"Expected {grid * grid} scores, got {arr.size}.")
    return arr.reshape(grid, grid)


def extract_maps_from_hidden_states(
    hidden_states: Sequence[Any],
    input_ids: Any,
    image_token_id: int,
    query_positions: Sequence[int],
    layers: Sequence[int],
    *,
    grid: int = 24,
) -> list[LayerMapResult]:
    image_positions = visual_token_positions(input_ids, image_token_id)
    expected = grid * grid
    if len(image_positions) != expected:
        raise ValueError(f"Expected {expected} image-token positions, found {len(image_positions)}.")
    if not query_positions:
        raise ValueError("query_positions is empty.")

    results: list[LayerMapResult] = []
    for layer in layers:
        if layer <= 0 or layer >= len(hidden_states):
            raise ValueError(f"Layer {layer} is outside hidden-state range [1, {len(hidden_states) - 1}].")
        states = hidden_states[layer]
        states = states[0] if getattr(states, "ndim", 0) == 3 else states
        image_states = states[image_positions, :]
        query_states = states[list(query_positions), :]
        score_tensors = compute_query_image_scores(image_states, query_states)
        maps = {name: _to_grid(scores, grid) for name, scores in score_tensors.items()}
        maps_z = {name: zscore_normalise(values, grid) for name, values in maps.items()}
        results.append(
            LayerMapResult(
                layer=int(layer),
                image_positions=image_positions,
                query_positions=[int(pos) for pos in query_positions],
                maps=maps,
                maps_z=maps_z,
            )
        )
    return results

