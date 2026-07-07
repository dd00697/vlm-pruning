"""Losses for UG-Prune-Lite distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vlm_pruning.selectors.ug_prune_lite import UGPruneLiteSelector


@dataclass(frozen=True)
class UGPruneLiteLossWeights:
    target: float = 1.5
    context: float = 0.7
    keep: float = 1.0
    gt_box: float = 2.0
    diversity: float = 0.1
    score_map_distill: float = 0.0
    score_map_temperature: float = 2.0
    score_map_loss: str = "kl"


def dice_loss(torch: Any, logits: Any, labels: Any, eps: float = 1e-6) -> Any:
    probs = torch.sigmoid(logits.float())
    labels = labels.float()
    intersection = (probs * labels).sum(dim=1)
    denom = probs.sum(dim=1) + labels.sum(dim=1)
    return (1.0 - ((2.0 * intersection + eps) / (denom + eps))).mean()


def spatial_diversity_loss(torch: Any, keep_logits: Any, *, grid: int = 24, bins: tuple[int, int] = (3, 3)) -> Any:
    batch, tokens = keep_logits.shape
    row_bins, col_bins = bins
    indices = torch.arange(tokens, device=keep_logits.device)
    rows = torch.div(indices, grid, rounding_mode="floor")
    cols = indices.remainder(grid)
    bin_ids = torch.clamp(torch.div(rows * row_bins, grid, rounding_mode="floor"), max=row_bins - 1) * col_bins
    bin_ids = bin_ids + torch.clamp(torch.div(cols * col_bins, grid, rounding_mode="floor"), max=col_bins - 1)
    probs = torch.softmax(keep_logits.float(), dim=1)
    masses = []
    for bin_index in range(row_bins * col_bins):
        masses.append(probs[:, bin_ids == bin_index].sum(dim=1))
    mass = torch.stack(masses, dim=1)
    entropy = -(mass * torch.log(mass + 1e-8)).sum(dim=1)
    max_entropy = torch.log(torch.tensor(float(row_bins * col_bins), device=keep_logits.device))
    return (1.0 - entropy / max_entropy).mean()


def _zscore_rows(torch: Any, values: Any, eps: float = 1e-6) -> Any:
    values = values.float()
    mean = values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, keepdim=True, unbiased=False)
    normalised = (values - mean) / torch.clamp(std, min=eps)
    return torch.where(std < eps, torch.zeros_like(normalised), normalised)


def score_map_distillation_loss(
    torch: Any,
    student_scores: Any,
    teacher_scores: Any,
    has_teacher_score_map: Any,
    *,
    temperature: float = 2.0,
    loss_type: str = "kl",
) -> tuple[Any, float]:
    active = has_teacher_score_map.float().reshape(-1) > 0.5
    if not bool(active.any().item()):
        return student_scores.float().sum() * 0.0, 0.0
    student = student_scores.float()[active]
    teacher = teacher_scores.float()[active]
    if student.ndim != 2 or teacher.ndim != 2 or student.shape != teacher.shape:
        raise ValueError(f"Expected matching [B,N] score maps, got {tuple(student.shape)} and {tuple(teacher.shape)}.")
    loss_name = str(loss_type).lower()
    if loss_name == "kl":
        temp = max(float(temperature), 1e-6)
        log_probs = torch.nn.functional.log_softmax(student / temp, dim=1)
        teacher_probs = torch.nn.functional.softmax(teacher / temp, dim=1)
        loss = torch.nn.functional.kl_div(log_probs, teacher_probs, reduction="batchmean") * (temp * temp)
    elif loss_name == "mse":
        loss = torch.nn.functional.mse_loss(_zscore_rows(torch, student), _zscore_rows(torch, teacher))
    else:
        raise ValueError(f"Unsupported score-map distillation loss: {loss_type}")
    return loss, float(active.float().mean().detach().cpu().item())


def ug_prune_lite_loss(torch: Any, outputs: dict[str, Any], batch: dict[str, Any], weights: UGPruneLiteLossWeights | None = None) -> tuple[Any, dict[str, float]]:
    weights = weights or UGPruneLiteLossWeights()
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    target = bce(outputs["target_logits"], batch["target_labels"])
    context = bce(outputs["context_logits"], batch["context_labels"])
    keep = bce(outputs["keep_logits"], batch["keep_labels"])
    box_bce = bce(outputs["target_logits"], batch["box_labels"])
    box_dice = dice_loss(torch, outputs["target_logits"], batch["box_labels"])
    box = 0.5 * box_bce + 0.5 * box_dice
    diversity = spatial_diversity_loss(torch, outputs["keep_logits"])
    score_distill = outputs["keep_logits"].float().sum() * 0.0
    score_distill_active = 0.0
    if float(weights.score_map_distill) > 0.0:
        score_distill, score_distill_active = score_map_distillation_loss(
            torch,
            outputs.get("score_logits", outputs["keep_logits"]),
            batch.get("teacher_score_map", torch.zeros_like(outputs["keep_logits"])),
            batch.get("has_teacher_score_map", torch.zeros(outputs["keep_logits"].shape[0], device=outputs["keep_logits"].device)),
            temperature=float(weights.score_map_temperature),
            loss_type=str(weights.score_map_loss),
        )
    total = (
        weights.target * target
        + weights.context * context
        + weights.keep * keep
        + weights.gt_box * box
        + weights.diversity * diversity
        + weights.score_map_distill * score_distill
    )
    metrics = {
        "loss/total": float(total.detach().cpu().item()),
        "loss/target_bce": float(target.detach().cpu().item()),
        "loss/context_bce": float(context.detach().cpu().item()),
        "loss/keep_bce": float(keep.detach().cpu().item()),
        "loss/gt_box": float(box.detach().cpu().item()),
        "loss/diversity": float(diversity.detach().cpu().item()),
        "loss/score_map_distill": float(score_distill.detach().cpu().item()),
        "loss/score_map_distill_active": float(score_distill_active),
    }
    return total, metrics


def occupied_bins_for_indices(indices: list[int], *, grid: int = 24, bins: tuple[int, int] = (3, 3)) -> int:
    row_bins, col_bins = bins
    occupied: set[int] = set()
    for index in indices:
        row = int(index) // grid
        col = int(index) % grid
        row_bin = min(row_bins - 1, (row * row_bins) // grid)
        col_bin = min(col_bins - 1, (col * col_bins) // grid)
        occupied.add(row_bin * col_bins + col_bin)
    return len(occupied)


def mean_pairwise_spread_for_indices(indices: list[int], *, grid: int = 24) -> float:
    if len(indices) < 2:
        return 0.0
    coords = [(int(index) // grid, int(index) % grid) for index in indices]
    total = 0.0
    count = 0
    for left in range(len(coords)):
        for right in range(left + 1, len(coords)):
            dr = float(coords[left][0] - coords[right][0])
            dc = float(coords[left][1] - coords[right][1])
            total += (dr * dr + dc * dc) ** 0.5
            count += 1
    return total / float(count)


def selection_metrics_from_indices(
    selected_indices: list[int],
    teacher_indices: list[int],
    role_labels: dict[str, list[int]],
    box_labels: Any,
    *,
    expected_tokens: int = 58,
    grid: int = 24,
) -> dict[str, float]:
    selected = [int(index) for index in selected_indices]
    selected_set = set(selected)
    teacher_set = {int(index) for index in teacher_indices}
    target_set = {int(index) for index in role_labels.get("target", [])}
    context_set = {int(index) for index in role_labels.get("context", [])}
    box_positive = {
        int(index)
        for index, value in enumerate(box_labels.detach().cpu().tolist() if hasattr(box_labels, "detach") else list(box_labels))
        if float(value) > 0.5
    }
    in_range = [index for index in selected if 0 <= index < grid * grid]
    duplicate_count = len(selected) - len(selected_set)
    out_of_range_count = len(selected) - len(in_range)
    teacher_denom = max(1, len(teacher_set))
    target_denom = max(1, len(target_set))
    context_denom = max(1, len(context_set))
    box_denom = max(1, len(box_positive))
    return {
        "selection/teacher_recall_at_58": len(selected_set & teacher_set) / float(teacher_denom),
        "selection/target_recall": len(selected_set & target_set) / float(target_denom),
        "selection/context_recall": len(selected_set & context_set) / float(context_denom),
        "selection/gt_box_patch_coverage_at_58": len(selected_set & box_positive) / float(box_denom),
        "selection/occupied_3x3_bins": float(occupied_bins_for_indices(selected, grid=grid)),
        "selection/mean_pairwise_spread": mean_pairwise_spread_for_indices(selected, grid=grid),
        "selection/unique_token_count": float(len(selected_set)),
        "selection/duplicate_count": float(duplicate_count),
        "selection/out_of_range_count": float(out_of_range_count),
        "selection/exact_budget": float(len(selected) == expected_tokens and duplicate_count == 0 and out_of_range_count == 0),
    }


def ug_prune_lite_selection_metrics(
    torch: Any,
    outputs: dict[str, Any],
    batch: dict[str, Any],
    selector: UGPruneLiteSelector | None = None,
    *,
    expected_tokens: int = 58,
    grid: int = 24,
) -> dict[str, float]:
    selector = selector or UGPruneLiteSelector(torch)
    rows: list[dict[str, float]] = []
    for index, sample_id in enumerate(batch["sample_id"]):
        selection = selector.select(
            outputs["target_logits"][index].detach(),
            outputs["context_logits"][index].detach(),
            outputs["keep_logits"][index].detach(),
            sample_id=str(sample_id),
        )
        rows.append(
            selection_metrics_from_indices(
                selection.selected_indices,
                [int(item) for item in batch["selected_indices"][index]],
                batch["role_labels"][index],
                batch["box_labels"][index],
                expected_tokens=expected_tokens,
                grid=grid,
            )
        )
    if not rows:
        return {}
    keys = sorted(rows[0])
    out = {key: sum(row[key] for row in rows) / float(len(rows)) for key in keys}
    out["selection/collapse_sample_fraction"] = sum(
        1.0 for row in rows if float(row.get("selection/occupied_3x3_bins", 0.0)) < 4.0
    ) / float(len(rows))
    return out
