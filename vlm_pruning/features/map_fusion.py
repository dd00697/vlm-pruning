"""Map fusion helpers for UG-Prune selectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .map_stats import as_grid_map, map_statistics, zscore_normalise
from .unrolled_maps import LayerMapResult


@dataclass(frozen=True)
class FusedMapResult:
    fused_map: np.ndarray
    source: str
    layers: tuple[int, ...]
    layer_keys: tuple[str, ...]
    stats: dict[str, float | int | None]
    fusion_mode: str = "soft_average"
    selected_layer: int | None = None
    layer_scores: dict[str, float] | None = None


def parse_int_tuple(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        if not items:
            raise ValueError("Expected at least one integer.")
        return tuple(int(item) for item in items)
    out = tuple(int(item) for item in value)
    if not out:
        raise ValueError("Expected at least one integer.")
    return out


def parse_probe_layers(value: str | Iterable[int], *, final_layer: int | None = None) -> tuple[int, ...]:
    if isinstance(value, str):
        items = [item.strip().lower() for item in value.split(",") if item.strip()]
        if not items:
            raise ValueError("Expected at least one probe layer.")
        out: list[int] = []
        for item in items:
            if item == "final":
                if final_layer is None:
                    raise ValueError("'final' probe layer requires a known final layer index.")
                out.append(int(final_layer))
            else:
                out.append(int(item))
        return tuple(out)
    return parse_int_tuple(value)


def parse_bins(value: str | Iterable[int]) -> tuple[int, int]:
    parsed = parse_int_tuple(value)
    if len(parsed) != 2:
        raise ValueError(f"Expected exactly two bin dimensions, got {parsed}.")
    if parsed[0] <= 0 or parsed[1] <= 0:
        raise ValueError(f"Bin dimensions must be positive, got {parsed}.")
    return parsed


def dynamic_layer_score(stats: dict[str, float | int | None], metric: str = "peakiness_minus_entropy") -> float:
    if metric != "peakiness_minus_entropy":
        raise ValueError(f"Unsupported dynamic layer metric: {metric}")
    peakiness = float(stats.get("peakiness") or 0.0)
    entropy_norm = float(stats.get("entropy_norm") or 0.0)
    return peakiness - entropy_norm


def fuse_layer_maps(
    layer_maps: Sequence[LayerMapResult],
    *,
    source: str = "query_max",
    layers: Sequence[int] | None = None,
    grid: int = 24,
    fusion_mode: str = "soft_average",
    dynamic_metric: str = "peakiness_minus_entropy",
) -> FusedMapResult:
    wanted = tuple(int(layer) for layer in layers) if layers is not None else tuple(result.layer for result in layer_maps)
    by_layer = {int(result.layer): result for result in layer_maps}
    arrays: list[np.ndarray] = []
    keys: list[str] = []
    layer_scores: dict[str, float] = {}
    for layer in wanted:
        if layer not in by_layer:
            raise ValueError(f"Requested layer {layer} is missing from extracted maps.")
        result = by_layer[layer]
        if source not in result.maps_z:
            raise ValueError(f"Map source '{source}' is missing for layer {layer}.")
        arr = as_grid_map(result.maps_z[source], grid)
        arrays.append(arr)
        keys.append(f"layer{layer}:{source}")
        layer_scores[keys[-1]] = dynamic_layer_score(map_statistics(arr, top_ks=(16, 32, 48), grid=grid), dynamic_metric)
    if fusion_mode == "soft_average":
        selected_layer = None
        fused = zscore_normalise(np.mean(np.stack(arrays, axis=0), axis=0), grid)
    elif fusion_mode == "single_layer":
        if len(arrays) != 1:
            raise ValueError("single_layer fusion requires exactly one requested layer.")
        selected_layer = wanted[0]
        fused = zscore_normalise(arrays[0], grid)
    elif fusion_mode == "hard_best":
        best_idx = max(range(len(arrays)), key=lambda idx: (layer_scores[keys[idx]], -wanted[idx]))
        selected_layer = wanted[best_idx]
        fused = zscore_normalise(arrays[best_idx], grid)
        keys = [keys[best_idx]]
    else:
        raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
    return FusedMapResult(
        fused_map=fused,
        source=source,
        layers=wanted,
        layer_keys=tuple(keys),
        stats=map_statistics(fused, top_ks=(16, 32, 48), grid=grid),
        fusion_mode=fusion_mode,
        selected_layer=selected_layer,
        layer_scores=layer_scores,
    )
