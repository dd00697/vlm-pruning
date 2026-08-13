"""Fail-closed GPU concurrency calculations for the final campaign."""

from __future__ import annotations


def admissible_widths(total_memory_mb: float, measured_peak_mb: float | None,
                      *, max_width: int = 3) -> tuple[list[int], float]:
    """Return descending concurrency widths that preserve locked headroom.

    If a representative serial peak is unavailable, only serial execution is
    allowed. The full GPU, rather than currently-free memory, defines the
    fixed headroom; a separate pre-launch snapshot still catches co-tenants.
    """
    if total_memory_mb <= 0:
        raise ValueError("total_memory_mb must be positive")
    if max_width <= 0:
        raise ValueError("max_width must be positive")
    headroom = max(total_memory_mb * 0.10, 4096.0)
    if measured_peak_mb is None or measured_peak_mb <= 0:
        return [1], headroom
    widths = [
        width for width in range(max_width, 0, -1)
        if width * measured_peak_mb + headroom <= total_memory_mb
    ]
    return widths or [1], headroom
