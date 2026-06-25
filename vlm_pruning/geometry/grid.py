"""Patch-grid geometry utilities."""

from __future__ import annotations


def _check_grid(grid: int) -> None:
    if grid <= 0:
        raise ValueError(f"grid must be positive, got {grid}.")


def patch_to_rc(index: int, grid: int = 24) -> tuple[int, int]:
    _check_grid(grid)
    if index < 0 or index >= grid * grid:
        raise ValueError(f"patch index {index} is outside [0, {grid * grid - 1}].")
    return divmod(index, grid)


def rc_to_patch(row: int, col: int, grid: int = 24) -> int:
    _check_grid(grid)
    if row < 0 or row >= grid or col < 0 or col >= grid:
        raise ValueError(f"row/col {(row, col)} is outside a {grid} x {grid} grid.")
    return row * grid + col


def neighbours(index: int, radius: int = 1, grid: int = 24) -> list[int]:
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}.")
    row, col = patch_to_rc(index, grid)
    out: list[int] = []
    for rr in range(max(0, row - radius), min(grid - 1, row + radius) + 1):
        for cc in range(max(0, col - radius), min(grid - 1, col + radius) + 1):
            candidate = rc_to_patch(rr, cc, grid)
            if candidate != index:
                out.append(candidate)
    return sorted(out)


def ring(indices: list[int] | set[int], radius: int = 1, grid: int = 24) -> set[int]:
    seeds = set(indices)
    out: set[int] = set()
    for index in seeds:
        out.update(neighbours(index, radius=radius, grid=grid))
    return out - seeds


def bin_id(index: int, bins: tuple[int, int] = (3, 3), grid: int = 24) -> tuple[int, int]:
    row_bins, col_bins = bins
    if row_bins <= 0 or col_bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}.")
    row, col = patch_to_rc(index, grid)
    return min(row_bins - 1, row * row_bins // grid), min(col_bins - 1, col * col_bins // grid)


def row_band(index: int, grid: int = 24) -> str:
    row, _ = patch_to_rc(index, grid)
    third = row * 3 // grid
    return ["top", "middle", "bottom"][min(2, third)]


def col_band(index: int, grid: int = 24) -> str:
    _, col = patch_to_rc(index, grid)
    third = col * 3 // grid
    return ["left", "center", "right"][min(2, third)]


def patch_bounds(index: int, width: int, height: int, grid: int = 24) -> tuple[float, float, float, float]:
    row, col = patch_to_rc(index, grid)
    cell_w = width / grid
    cell_h = height / grid
    return col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h

