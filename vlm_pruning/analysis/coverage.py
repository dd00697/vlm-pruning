"""Coverage and spatial diagnostics for retained patch indices."""

from __future__ import annotations

import math

from vlm_pruning.geometry.grid import bin_id, patch_bounds, ring


def _area(box: list[float] | tuple[float, float, float, float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _intersection(a: tuple[float, float, float, float], b: list[float] | tuple[float, float, float, float]) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def box_coverage(
    selected_indices: list[int] | set[int],
    box: list[float],
    image_size: tuple[int, int],
    grid: int = 24,
) -> float:
    box_area = _area(box)
    if box_area <= 0.0:
        return 0.0
    width, height = image_size
    covered = sum(_intersection(patch_bounds(index, width, height, grid), box) for index in set(selected_indices))
    return min(1.0, covered / box_area)


def spatial_entropy(selected_indices: list[int] | set[int], bins: tuple[int, int] = (3, 3), grid: int = 24) -> float:
    selected = list(selected_indices)
    if not selected:
        return 0.0
    counts: dict[tuple[int, int], int] = {}
    for index in selected:
        key = bin_id(index, bins=bins, grid=grid)
        counts[key] = counts.get(key, 0) + 1
    total = float(len(selected))
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    max_entropy = math.log(bins[0] * bins[1])
    return entropy / max_entropy if max_entropy > 0 else 0.0


def occupied_bins(selected_indices: list[int] | set[int], bins: tuple[int, int] = (3, 3), grid: int = 24) -> int:
    return len({bin_id(index, bins=bins, grid=grid) for index in selected_indices})


def selection_diagnostics(
    selected_indices: list[int],
    gt_box: list[float],
    image_size: tuple[int, int],
    grid: int = 24,
) -> dict[str, float | int]:
    selected = set(selected_indices)
    dilated = selected | ring(selected, radius=1, grid=grid)
    bins = (3, 3)
    return {
        "target_coverage": box_coverage(selected, gt_box, image_size, grid),
        "context_coverage_1hop": box_coverage(dilated, gt_box, image_size, grid),
        "selected_spatial_entropy_3x3": spatial_entropy(selected, bins=bins, grid=grid),
        "selected_occupied_bins_3x3": occupied_bins(selected, bins=bins, grid=grid),
    }

