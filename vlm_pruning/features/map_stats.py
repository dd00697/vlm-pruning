"""Map normalisation and statistics for query-image diagnostics."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from vlm_pruning.analysis.coverage import occupied_bins, spatial_entropy


def as_grid_map(values: np.ndarray | Iterable[float], grid: int = 24) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape == (grid, grid):
        return arr
    if arr.size != grid * grid:
        raise ValueError(f"Expected {grid * grid} map values, got shape {arr.shape}.")
    return arr.reshape(grid, grid)


def zscore_normalise(values: np.ndarray | Iterable[float], grid: int = 24, eps: float = 1e-6) -> np.ndarray:
    arr = as_grid_map(values, grid)
    std = float(arr.std())
    if std < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - float(arr.mean())) / std).astype(np.float32)


def minmax_normalise(values: np.ndarray | Iterable[float], grid: int = 24, eps: float = 1e-6) -> np.ndarray:
    arr = as_grid_map(values, grid)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def softmax_flat(values: np.ndarray | Iterable[float], temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    flat = np.asarray(values, dtype=np.float64).reshape(-1) / temperature
    flat = flat - float(np.max(flat))
    exp = np.exp(flat)
    total = float(exp.sum())
    if total <= 0.0:
        return np.full_like(exp, 1.0 / exp.size)
    return exp / total


def top_k_indices(values: np.ndarray | Iterable[float], k: int) -> list[int]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if k <= 0:
        return []
    k = min(k, flat.size)
    return [int(idx) for idx in np.argsort(-flat, kind="mergesort")[:k]]


def top_k_mass(values: np.ndarray | Iterable[float], k: int, temperature: float = 1.0) -> float:
    probs = softmax_flat(values, temperature=temperature)
    if k <= 0:
        return 0.0
    top = np.argsort(-probs, kind="mergesort")[: min(k, probs.size)]
    return float(probs[top].sum())


def entropy(values: np.ndarray | Iterable[float], temperature: float = 1.0) -> tuple[float, float]:
    probs = softmax_flat(values, temperature=temperature)
    ent = float(-(probs * np.log(probs + 1e-12)).sum())
    max_ent = math.log(probs.size) if probs.size > 1 else 1.0
    return ent, ent / max_ent if max_ent > 0 else 0.0


def local_maxima_count(values: np.ndarray | Iterable[float], grid: int = 24) -> int:
    arr = as_grid_map(values, grid)
    threshold = float(arr.mean())
    count = 0
    for row in range(grid):
        for col in range(grid):
            value = float(arr[row, col])
            if value <= threshold:
                continue
            r0 = max(0, row - 1)
            r1 = min(grid, row + 2)
            c0 = max(0, col - 1)
            c1 = min(grid, col + 2)
            neighbourhood = arr[r0:r1, c0:c1]
            if value >= float(neighbourhood.max()) and int((neighbourhood == value).sum()) == 1:
                count += 1
    return count


def pearson_correlation(left: np.ndarray | Iterable[float], right: np.ndarray | Iterable[float]) -> float | None:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size == 0:
        raise ValueError(f"Cannot correlate arrays with shapes {a.shape} and {b.shape}.")
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return None
    return float(np.dot(a, b) / denom)


def map_statistics(
    values: np.ndarray | Iterable[float],
    *,
    top_ks: tuple[int, ...] = (16, 32, 48),
    grid: int = 24,
    bins: tuple[int, int] = (3, 3),
) -> dict[str, float | int | None]:
    arr = as_grid_map(values, grid)
    flat = arr.reshape(-1)
    ent, ent_norm = entropy(flat)
    stats: dict[str, float | int | None] = {
        "map_mean": float(flat.mean()),
        "map_std": float(flat.std()),
        "map_min": float(flat.min()),
        "map_max": float(flat.max()),
        "entropy": ent,
        "entropy_norm": ent_norm,
        "peakiness": float(flat.max() - flat.mean()),
        "local_maxima_count": local_maxima_count(arr, grid),
    }
    for k in top_ks:
        selected = top_k_indices(flat, k)
        stats[f"top_{k}_mass"] = top_k_mass(flat, k)
        stats[f"top_{k}_occupied_bins_3x3"] = occupied_bins(selected, bins=bins, grid=grid)
        stats[f"top_{k}_spatial_entropy_3x3"] = spatial_entropy(selected, bins=bins, grid=grid)
    return stats

