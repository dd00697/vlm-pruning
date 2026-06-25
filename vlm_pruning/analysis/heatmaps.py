"""Heatmap overlay rendering for Phase 1 map diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from vlm_pruning.features.map_stats import minmax_normalise, top_k_indices
from vlm_pruning.geometry.grid import patch_bounds


def _colourise(values: np.ndarray) -> Image.Image:
    norm = minmax_normalise(values)
    try:
        import matplotlib.cm as cm

        rgba = cm.get_cmap("magma")(norm)
        data = (rgba[:, :, :3] * 255).astype(np.uint8)
    except Exception:
        red = (norm * 255).astype(np.uint8)
        green = np.zeros_like(red)
        blue = ((1.0 - norm) * 180).astype(np.uint8)
        data = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(data, mode="RGB")


def save_heatmap_overlay(
    image: Image.Image,
    heatmap: np.ndarray | Iterable[float],
    output_path: Path,
    *,
    gt_box: list[float] | tuple[float, float, float, float] | None = None,
    top_k: int | None = 48,
    grid: int = 24,
    alpha: float = 0.45,
) -> Path:
    base = image.convert("RGB")
    arr = np.asarray(heatmap, dtype=np.float32).reshape(grid, grid)
    heat = _colourise(arr).resize(base.size, resample=Image.Resampling.BILINEAR)
    overlay = Image.blend(base, heat, alpha=max(0.0, min(1.0, alpha)))
    draw = ImageDraw.Draw(overlay)
    width, height = base.size
    if top_k is not None and top_k > 0:
        for index in top_k_indices(arr, top_k):
            draw.rectangle(patch_bounds(index, width, height, grid), outline=(255, 235, 59), width=1)
    if gt_box is not None:
        draw.rectangle(tuple(float(x) for x in gt_box), outline=(0, 255, 128), width=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)
    return output_path

