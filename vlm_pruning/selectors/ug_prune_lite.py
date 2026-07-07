"""Cheap pre-LLM student selector distilled from UG-Prune-Gap-58."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vlm_pruning.selectors.base import SelectionResult, SelectorError
from vlm_pruning.selectors.ug_prune import stable_checksum


@dataclass(frozen=True)
class UGPruneLiteConfig:
    visual_tokens: int = 576
    budget: int = 58
    grid: int = 24
    target_count: int = 29
    context_count: int = 17
    anchor_count: int = 12
    context_radius: int = 1
    anchor_bins: tuple[int, int] = (3, 3)

    def validate(self) -> "UGPruneLiteConfig":
        if self.visual_tokens != self.grid * self.grid:
            raise SelectorError("UG-Prune-Lite currently expects visual_tokens == grid * grid.")
        if self.target_count + self.context_count + self.anchor_count != self.budget:
            raise SelectorError("UG-Prune-Lite role counts must sum to budget.")
        if self.context_radius != 1:
            raise SelectorError("UG-Prune-Lite v1 expects context_radius=1.")
        if self.anchor_bins != (3, 3):
            raise SelectorError("UG-Prune-Lite v1 expects anchor_bins=(3, 3).")
        return self


def patch_coordinates(torch: Any, *, grid: int = 24, device: Any | None = None) -> Any:
    indices = torch.arange(grid * grid, device=device, dtype=torch.float32)
    rows = torch.div(indices, grid, rounding_mode="floor")
    cols = indices.remainder(grid)
    denom = float(max(1, grid - 1))
    return torch.stack([rows / denom, cols / denom], dim=-1)


def coordinate_features(torch: Any, *, grid: int = 24, device: Any | None = None, fourier: bool = True) -> Any:
    coords = patch_coordinates(torch, grid=grid, device=device)
    if not fourier:
        return coords
    pi = torch.tensor(3.141592653589793, device=coords.device, dtype=coords.dtype)
    feats = [coords]
    for freq in (1.0, 2.0, 4.0):
        scaled = coords * (2.0 * pi * freq)
        feats.extend([torch.sin(scaled), torch.cos(scaled)])
    return torch.cat(feats, dim=-1)


def box_patch_labels(
    torch: Any,
    gt_box: list[float] | tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    grid: int = 24,
    device: Any | None = None,
) -> Any:
    """Return positive labels for patches whose centers fall inside the GT box.

    If a tiny edge box contains no patch center, fall back to any patch with
    non-zero box overlap.
    """
    x0, y0, x1, y1 = [float(value) for value in gt_box]
    coords = patch_coordinates(torch, grid=grid, device=device)
    rows = coords[:, 0] * float(max(1, grid - 1))
    cols = coords[:, 1] * float(max(1, grid - 1))
    cell_w = float(width) / float(grid)
    cell_h = float(height) / float(grid)
    centers_x = (cols + 0.5) * cell_w
    centers_y = (rows + 0.5) * cell_h
    labels = (centers_x >= x0) & (centers_x <= x1) & (centers_y >= y0) & (centers_y <= y1)
    if bool(labels.any().item()):
        return labels.float()
    patch_x0 = cols * cell_w
    patch_y0 = rows * cell_h
    patch_x1 = patch_x0 + cell_w
    patch_y1 = patch_y0 + cell_h
    overlap_w = torch.clamp(torch.minimum(patch_x1, torch.tensor(x1, device=patch_x1.device)) - torch.maximum(patch_x0, torch.tensor(x0, device=patch_x0.device)), min=0.0)
    overlap_h = torch.clamp(torch.minimum(patch_y1, torch.tensor(y1, device=patch_y1.device)) - torch.maximum(patch_y0, torch.tensor(y0, device=patch_y0.device)), min=0.0)
    return ((overlap_w * overlap_h) > 0).float()


class UGPruneLiteSelector:
    name = "ug_prune_lite"

    def __init__(self, torch: Any, config: UGPruneLiteConfig | None = None) -> None:
        self.torch = torch
        self.config = (config or UGPruneLiteConfig()).validate()
        self._cache: dict[str, Any] = {}

    def _grid_cache(self, device: Any) -> dict[str, Any]:
        key = str(device)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        torch = self.torch
        grid = self.config.grid
        count = grid * grid
        indices = torch.arange(count, device=device, dtype=torch.long)
        rows = torch.div(indices, grid, rounding_mode="floor")
        cols = indices.remainder(grid)
        row_diff = (rows[:, None] - rows[None, :]).abs()
        col_diff = (cols[:, None] - cols[None, :]).abs()
        neighbor_mask = ((row_diff <= self.config.context_radius) & (col_diff <= self.config.context_radius)) & (
            indices[:, None] != indices[None, :]
        )
        row_bins, col_bins = self.config.anchor_bins
        bin_rows = torch.clamp(torch.div(rows * row_bins, grid, rounding_mode="floor"), max=row_bins - 1)
        bin_cols = torch.clamp(torch.div(cols * col_bins, grid, rounding_mode="floor"), max=col_bins - 1)
        bin_ids = (bin_rows * col_bins + bin_cols).long()
        centers = torch.stack([rows.float() + 0.5, cols.float() + 0.5], dim=1)
        distance_sq = ((centers[:, None, :] - centers[None, :, :]) ** 2).sum(dim=-1)
        center = float(grid) / 2.0
        center_distance_sq = ((centers - center) ** 2).sum(dim=-1)
        cached = {
            "indices": indices,
            "neighbor_mask": neighbor_mask,
            "bin_ids": bin_ids,
            "num_bins": row_bins * col_bins,
            "distance_sq": distance_sq,
            "center_distance_sq": center_distance_sq,
        }
        self._cache[key] = cached
        return cached

    def _topk_excluding(self, scores: Any, count: int, exclude: Any) -> Any:
        torch = self.torch
        if count <= 0:
            return torch.empty(0, dtype=torch.long, device=scores.device)
        masked = scores.float().clone()
        masked[exclude] = -torch.inf
        return torch.topk(masked, k=min(int(count), int(scores.numel())), largest=True, sorted=False).indices

    def _context(self, context_scores: Any, target: Any, selected_mask: Any, grid_cache: dict[str, Any]) -> Any:
        torch = self.torch
        candidate_mask = grid_cache["neighbor_mask"].index_select(0, target).any(dim=0) & ~selected_mask
        scores = context_scores.float().clone()
        scores[~candidate_mask] = -torch.inf
        valid = int(candidate_mask.sum().item())
        take = min(self.config.context_count, valid)
        context = torch.topk(scores, k=take, largest=True, sorted=False).indices if take > 0 else target.new_empty((0,))
        if int(context.numel()) < self.config.context_count:
            tmp_mask = selected_mask.clone()
            tmp_mask[context] = True
            fill = self._topk_excluding(context_scores, self.config.context_count - int(context.numel()), tmp_mask)
            context = torch.cat([context, fill], dim=0)
        return context.long()

    def _anchors(self, keep_scores: Any, selected_mask: Any, grid_cache: dict[str, Any]) -> Any:
        torch = self.torch
        anchors: list[Any] = []
        selected = selected_mask.clone()
        indices = grid_cache["indices"]
        high_index = torch.full_like(indices, indices.numel())
        for _ in range(self.config.anchor_count):
            candidates = ~selected
            if int(candidates.sum().item()) == 0:
                break
            selected_indices = torch.nonzero(selected, as_tuple=False).flatten()
            bin_counts = torch.bincount(
                grid_cache["bin_ids"].index_select(0, selected_indices),
                minlength=grid_cache["num_bins"],
            )
            occupancy = bin_counts.index_select(0, grid_cache["bin_ids"])
            min_occupancy = occupancy[candidates].min()
            mask = candidates & (occupancy == min_occupancy)
            if int(selected_indices.numel()) == 0:
                min_distance = grid_cache["center_distance_sq"]
            else:
                min_distance = grid_cache["distance_sq"].index_select(1, selected_indices).min(dim=1).values
            max_distance = min_distance[mask].max()
            mask = mask & (min_distance == max_distance)
            max_score = keep_scores[mask].max()
            mask = mask & (keep_scores == max_score)
            best = torch.where(mask, indices, high_index).min()
            anchors.append(best)
            selected[best] = True
        if not anchors:
            return torch.empty(0, dtype=torch.long, device=keep_scores.device)
        return torch.stack(anchors).long()

    def select(self, target_logits: Any, context_logits: Any, keep_logits: Any, *, sample_id: str = "") -> SelectionResult:
        torch = self.torch
        if target_logits.ndim != 1 or int(target_logits.numel()) != self.config.visual_tokens:
            raise SelectorError(f"Expected target_logits shape [576], got {tuple(target_logits.shape)}")
        target = torch.topk(target_logits.float(), k=self.config.target_count, largest=True, sorted=False).indices.long()
        selected_mask = torch.zeros(self.config.visual_tokens, dtype=torch.bool, device=target_logits.device)
        selected_mask[target] = True
        grid_cache = self._grid_cache(target_logits.device)
        context = self._context(context_logits, target, selected_mask, grid_cache)
        selected_mask[context] = True
        anchor = self._anchors(keep_logits.float(), selected_mask, grid_cache)
        selected_mask[anchor] = True
        if int(selected_mask.sum().item()) < self.config.budget:
            fill = self._topk_excluding(keep_logits, self.config.budget - int(selected_mask.sum().item()), selected_mask)
            selected_mask[fill] = True
        selected = torch.nonzero(selected_mask, as_tuple=False).flatten().sort().values[: self.config.budget]
        selected_indices = [int(index) for index in selected.detach().cpu().tolist()]
        roles = {
            "target": [int(index) for index in target.sort().values.detach().cpu().tolist()],
            "context": [int(index) for index in context.sort().values.detach().cpu().tolist()],
            "anchor": [int(index) for index in anchor.sort().values.detach().cpu().tolist()],
        }
        return SelectionResult(
            selected_indices=selected_indices,
            method=self.name,
            budget=self.config.budget,
            original_tokens=self.config.visual_tokens,
            grid=self.config.grid,
            role_labels={role: values for role, values in roles.items() if values},
            metadata={
                "selector_family": "ug_prune_lite_distilled",
                "sample_id": sample_id,
                "budget_split": {
                    "target": self.config.target_count,
                    "context": self.config.context_count,
                    "anchor": self.config.anchor_count,
                    "fill": 0,
                },
                "target_checksum": stable_checksum(roles["target"]),
                "context_checksum": stable_checksum(roles["context"]),
                "anchor_checksum": stable_checksum(roles["anchor"]),
            },
        ).validate()

