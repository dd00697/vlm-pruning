"""Training-free UG-Prune target/context/anchor selector."""

from __future__ import annotations

import math
import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from vlm_pruning.features.map_stats import as_grid_map, map_statistics
from vlm_pruning.geometry.grid import bin_id, patch_to_rc, rc_to_patch, ring

from .base import SelectionResult, SelectorError


@dataclass(frozen=True)
class UGPruneConfig:
    selector_strategy: str = "structured_tca"
    map_family: str = "unrolled"
    map_source: str = "query_max"
    probe_layers: tuple[int, ...] = (4, 8, 12)
    fusion_mode: str = "soft_average"
    dynamic_layer_metric: str = "peakiness_minus_entropy"
    role_mode: str = "target_context_anchor"
    target_ratio: float = 0.50
    context_ratio: float = 0.30
    anchor_ratio: float = 0.20
    context_radius: int = 1
    anchor_bins: tuple[int, int] = (3, 3)
    anchor_pattern: str = "fps"
    nms_radius: int = 1
    position_mode: str = "compact"

    def validate(self) -> "UGPruneConfig":
        if self.selector_strategy not in {"query_topk", "target_nms_only", "structured_tca"}:
            raise SelectorError(f"Unsupported selector_strategy: {self.selector_strategy}")
        if not self.probe_layers:
            raise SelectorError("UG-Prune requires at least one probe layer.")
        if any(layer <= 0 for layer in self.probe_layers):
            raise SelectorError(f"Probe layers must be positive, got {self.probe_layers}.")
        if self.map_family not in {"unrolled", "clip"}:
            raise SelectorError(f"Unsupported map_family: {self.map_family}")
        if self.map_source not in {"query_max", "query_pooled"}:
            raise SelectorError(f"Unsupported map_source: {self.map_source}")
        if self.fusion_mode not in {"soft_average", "hard_best", "single_layer"}:
            raise SelectorError(f"Unsupported fusion_mode: {self.fusion_mode}")
        if self.dynamic_layer_metric != "peakiness_minus_entropy":
            raise SelectorError(f"Unsupported dynamic_layer_metric: {self.dynamic_layer_metric}")
        if self.role_mode not in {"target_only", "target_context", "target_context_anchor"}:
            raise SelectorError(f"Unsupported role_mode: {self.role_mode}")
        ratios = [self.target_ratio, self.context_ratio, self.anchor_ratio]
        if any(value < 0.0 for value in ratios):
            raise SelectorError(f"UG-Prune ratios must be non-negative, got {ratios}.")
        if sum(ratios) <= 0.0:
            raise SelectorError("At least one UG-Prune ratio must be positive.")
        if self.context_radius < 0:
            raise SelectorError(f"context_radius must be non-negative, got {self.context_radius}.")
        if self.nms_radius < 0:
            raise SelectorError(f"nms_radius must be non-negative, got {self.nms_radius}.")
        if self.anchor_bins[0] <= 0 or self.anchor_bins[1] <= 0:
            raise SelectorError(f"anchor_bins must be positive, got {self.anchor_bins}.")
        if self.anchor_pattern not in {"none", "corners", "grid", "fps"}:
            raise SelectorError(f"Unsupported anchor_pattern: {self.anchor_pattern}")
        if self.position_mode not in {"compact", "raster_compact", "gap_preserving"}:
            raise SelectorError(f"Unsupported position_mode: {self.position_mode}")
        return self

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["probe_layers"] = list(self.probe_layers)
        payload["anchor_bins"] = list(self.anchor_bins)
        return payload


@dataclass(frozen=True)
class UGPruneInputs:
    fused_map: np.ndarray
    map_stats: dict[str, Any]
    map_key: str
    query_text: str
    query_token_status: str
    query_token_positions: list[int]


def stable_checksum(indices: Iterable[int]) -> str:
    payload = ",".join(str(int(index)) for index in indices)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _rank_indices(scores: np.ndarray, candidates: Iterable[int] | None = None) -> list[int]:
    if candidates is None:
        values = list(range(scores.size))
    else:
        values = sorted({int(index) for index in candidates})
    return sorted(values, key=lambda idx: (-float(scores[idx]), idx))


def _is_near_any(index: int, selected: list[int], radius: int, grid: int) -> bool:
    if radius <= 0:
        return False
    row, col = patch_to_rc(index, grid)
    for existing in selected:
        erow, ecol = patch_to_rc(existing, grid)
        if max(abs(row - erow), abs(col - ecol)) <= radius:
            return True
    return False


def _min_anchor_count(budget: int) -> int:
    if budget >= 48:
        return 12
    if budget >= 32:
        return 8
    if budget >= 16:
        return 4
    return 0


def budget_split(budget: int, config: UGPruneConfig) -> tuple[int, int, int]:
    config.validate()
    if budget <= 0:
        raise SelectorError(f"budget must be positive, got {budget}.")
    if config.selector_strategy == "query_topk":
        return 0, 0, 0
    if config.selector_strategy == "target_nms_only" or config.role_mode == "target_only":
        return budget, 0, 0
    if config.role_mode == "target_context" or config.anchor_pattern == "none":
        total_tc = config.target_ratio + config.context_ratio
        if total_tc <= 0.0:
            return budget, 0, 0
        target = max(1, int(math.floor(budget * config.target_ratio / total_tc)))
        context = budget - target
        if budget >= 16 and config.context_ratio > 0:
            context = max(1, context)
        while target + context > budget:
            if context > (1 if budget >= 16 and config.context_ratio > 0 else 0):
                context -= 1
            else:
                target -= 1
        while target + context < budget:
            target += 1
        return target, context, 0
    total = config.target_ratio + config.context_ratio + config.anchor_ratio
    target = int(math.floor(budget * config.target_ratio / total))
    context = int(math.floor(budget * config.context_ratio / total))
    anchor = budget - target - context
    target = max(1, target)
    if budget >= 16:
        context = max(1, context)
        anchor = max(_min_anchor_count(budget), anchor)
    while target + context + anchor > budget:
        if context > (1 if budget >= 16 else 0):
            context -= 1
        elif anchor > _min_anchor_count(budget):
            anchor -= 1
        elif context > 0:
            context -= 1
        elif anchor > 0:
            anchor -= 1
        else:
            target -= 1
    while target + context + anchor < budget:
        anchor += 1
    return target, context, anchor


def select_target(scores: np.ndarray, count: int, *, nms_radius: int, grid: int) -> list[int]:
    selected, _ = select_target_with_fallback(scores, count, nms_radius=nms_radius, grid=grid)
    return selected


def select_target_with_fallback(
    scores: np.ndarray, count: int, *, nms_radius: int, grid: int
) -> tuple[list[int], int]:
    selected: list[int] = []
    for index in _rank_indices(scores):
        if not _is_near_any(index, selected, nms_radius, grid):
            selected.append(index)
            if len(selected) == count:
                return selected, 0
    nms_count = len(selected)
    for index in _rank_indices(scores):
        if index not in selected:
            selected.append(index)
            if len(selected) == count:
                break
    return selected, len(selected) - nms_count


def select_context(scores: np.ndarray, target: list[int], count: int, *, radius: int, grid: int) -> list[int]:
    if count <= 0:
        return []
    excluded = set(target)
    candidates = ring(target, radius=radius, grid=grid) if target and radius > 0 else set()
    selected = [idx for idx in _rank_indices(scores, candidates) if idx not in excluded][:count]
    excluded.update(selected)
    if len(selected) < count:
        for index in _rank_indices(scores):
            if index not in excluded:
                selected.append(index)
                excluded.add(index)
                if len(selected) == count:
                    break
    return selected


def _patch_center(index: int, grid: int) -> tuple[float, float]:
    row, col = patch_to_rc(index, grid)
    return row + 0.5, col + 0.5


def _min_distance_sq(index: int, selected: set[int], grid: int) -> float:
    row, col = _patch_center(index, grid)
    if not selected:
        center = grid / 2.0
        return (row - center) ** 2 + (col - center) ** 2
    return min((row - r) ** 2 + (col - c) ** 2 for r, c in (_patch_center(item, grid) for item in selected))


def _append_unique(out: list[int], selected: set[int], index: int) -> bool:
    if index in selected:
        return False
    out.append(index)
    selected.add(index)
    return True


def _corner_candidates(grid: int) -> list[int]:
    return [
        rc_to_patch(0, 0, grid),
        rc_to_patch(0, grid - 1, grid),
        rc_to_patch(grid - 1, 0, grid),
        rc_to_patch(grid - 1, grid - 1, grid),
    ]


def _grid_candidates(bins: tuple[int, int], grid: int) -> list[int]:
    row_bins, col_bins = bins
    candidates: list[int] = []
    for brow in range(row_bins):
        row = min(grid - 1, int(round((brow + 0.5) * grid / row_bins - 0.5)))
        for bcol in range(col_bins):
            col = min(grid - 1, int(round((bcol + 0.5) * grid / col_bins - 0.5)))
            candidates.append(rc_to_patch(row, col, grid))
    return candidates


def _fps_anchors(scores: np.ndarray, count: int, *, selected: set[int], bins: tuple[int, int], grid: int) -> list[int]:
    anchors: list[int] = []
    while len(anchors) < count:
        candidates = [index for index in range(scores.size) if index not in selected]
        if not candidates:
            break
        occupancy: dict[tuple[int, int], int] = {}
        for index in selected:
            key = bin_id(index, bins=bins, grid=grid)
            occupancy[key] = occupancy.get(key, 0) + 1
        best = max(
            candidates,
            key=lambda idx: (
                -occupancy.get(bin_id(idx, bins=bins, grid=grid), 0),
                _min_distance_sq(idx, selected, grid),
                float(scores[idx]),
                -idx,
            ),
        )
        anchors.append(best)
        selected.add(best)
    return anchors


def select_anchors(
    scores: np.ndarray,
    count: int,
    *,
    exclude: set[int],
    bins: tuple[int, int],
    grid: int,
    pattern: str = "fps",
) -> list[int]:
    if count <= 0 or pattern == "none":
        return []
    selected: set[int] = set(exclude)
    anchors: list[int] = []
    if pattern == "corners":
        for index in _corner_candidates(grid):
            _append_unique(anchors, selected, index)
            if len(anchors) == count:
                return anchors
    elif pattern == "grid":
        for index in _grid_candidates(bins, grid):
            _append_unique(anchors, selected, index)
            if len(anchors) == count:
                return anchors
    elif pattern != "fps":
        raise SelectorError(f"Unsupported anchor pattern: {pattern}")
    anchors.extend(_fps_anchors(scores, count - len(anchors), selected=selected, bins=bins, grid=grid))
    return anchors


def selected_spread_stats(indices: Iterable[int], *, bins: tuple[int, int], grid: int) -> dict[str, float | int]:
    selected = sorted({int(index) for index in indices})
    if not selected:
        return {
            "selected_occupied_bins_3x3": 0,
            "selected_row_span": 0,
            "selected_col_span": 0,
            "selected_mean_pairwise_spread": 0.0,
        }
    rows_cols = [patch_to_rc(index, grid) for index in selected]
    rows = np.asarray([row for row, _ in rows_cols], dtype=np.float32)
    cols = np.asarray([col for _, col in rows_cols], dtype=np.float32)
    coords = np.stack([rows, cols], axis=1)
    if len(selected) > 1:
        diffs = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt((diffs**2).sum(axis=-1))
        tri = distances[np.triu_indices(len(selected), k=1)]
        spread = float(tri.mean() / ((2.0**0.5) * max(grid - 1, 1))) if tri.size else 0.0
    else:
        spread = 0.0
    return {
        "selected_occupied_bins_3x3": len({bin_id(index, bins=bins, grid=grid) for index in selected}),
        "selected_row_span": int(rows.max() - rows.min() + 1),
        "selected_col_span": int(cols.max() - cols.min() + 1),
        "selected_mean_pairwise_spread": spread,
    }


class UGPruneSelector:
    name = "ug_prune_tf"

    def __init__(self, config: UGPruneConfig | None = None) -> None:
        self.config = (config or UGPruneConfig()).validate()

    def select_from_map(
        self,
        *,
        sample_id: str,
        budget: int,
        original_tokens: int,
        grid: int,
        seed: int,
        inputs: UGPruneInputs,
    ) -> SelectionResult:
        del seed
        if original_tokens != grid * grid:
            raise SelectorError(f"UG-Prune expects original_tokens == grid * grid, got {original_tokens} and {grid}.")
        scores_2d = as_grid_map(inputs.fused_map, grid)
        scores = scores_2d.reshape(-1)
        target_count, context_count, anchor_count = budget_split(budget, self.config)

        nms_fallback_count = 0
        if self.config.selector_strategy == "query_topk":
            target, context, anchor = [], [], []
            query_topk = _rank_indices(scores)[:budget]
        else:
            target, nms_fallback_count = select_target_with_fallback(
                scores, target_count, nms_radius=self.config.nms_radius, grid=grid
            )
            context = select_context(scores, target, context_count, radius=self.config.context_radius, grid=grid)
            anchor = select_anchors(
                scores,
                anchor_count,
                exclude=set(target) | set(context),
                bins=self.config.anchor_bins,
                grid=grid,
                pattern=self.config.anchor_pattern,
            )
            query_topk = []

        roles = {
            "target": sorted(target),
            "context": sorted(context),
            "anchor": sorted(anchor),
            "query_topk": sorted(query_topk),
            "fill": [],
        }
        selected = set(target) | set(context) | set(anchor) | set(query_topk)
        if len(selected) < budget:
            for index in _rank_indices(scores):
                if index not in selected:
                    roles["fill"].append(index)
                    selected.add(index)
                    if len(selected) == budget:
                        break
        if len(selected) > budget:
            keep = set(_rank_indices(scores, selected)[:budget])
            for role, values in roles.items():
                roles[role] = sorted(index for index in values if index in keep)
            selected = keep

        map_stats = dict(inputs.map_stats or map_statistics(scores_2d, grid=grid))
        map_stats.update(selected_spread_stats(selected, bins=self.config.anchor_bins, grid=grid))
        result = SelectionResult(
            selected_indices=sorted(selected),
            method=self.name,
            budget=budget,
            original_tokens=original_tokens,
            grid=grid,
            role_labels={role: sorted(values) for role, values in roles.items() if values},
            metadata={
                "selector_family": "ug_prune_training_free",
                "selector_strategy": self.config.selector_strategy,
                "sample_id": sample_id,
                "config": self.config.as_dict(),
                "map_family": self.config.map_family,
                "budget_split": {
                    "query_topk": budget if self.config.selector_strategy == "query_topk" else 0,
                    "target": target_count,
                    "context": context_count,
                    "anchor": anchor_count,
                    "fill": len(roles["fill"]),
                },
                "realised_role_counts": {role: len(values) for role, values in roles.items()},
                "nms_fallback_count": nms_fallback_count,
                "nms_radius": self.config.nms_radius,
                "map_key": inputs.map_key,
                "map_source": self.config.map_source,
                "probe_layers": list(self.config.probe_layers),
                "fusion_mode": self.config.fusion_mode,
                "dynamic_layer_metric": self.config.dynamic_layer_metric,
                "role_mode": self.config.role_mode,
                "anchor_pattern": self.config.anchor_pattern,
                "position_mode": self.config.position_mode,
                "map_stats": map_stats,
                "query_text": inputs.query_text,
                "query_token_status": inputs.query_token_status,
                "query_token_positions": inputs.query_token_positions,
                "target_checksum": stable_checksum(roles["target"]),
                "context_checksum": stable_checksum(roles["context"]),
                "anchor_checksum": stable_checksum(roles["anchor"]),
            },
        )
        return result.validate()
