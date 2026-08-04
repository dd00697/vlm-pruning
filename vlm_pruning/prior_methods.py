"""Small, testable primitives for the pinned FastV and VisionZip adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


FASTV_SOURCE = "pkunlp-icler/FastV"
FASTV_COMMIT = "d1659729b5bf1be225e99ee15783deeea80f63b1"
VISIONZIP_SOURCE = "JIA-Lab-research/VisionZip"
VISIONZIP_COMMIT = "8f86b55c6f000eb033e6912538af2dd7dcb30502"


@dataclass(frozen=True)
class VisionZipBudget:
    dominant: int
    contextual: int

    @property
    def total(self) -> int:
        return self.dominant + self.contextual


def visionzip_budget(total: int) -> VisionZipBudget:
    """Scale the official 54-dominant/10-contextual 64-token recipe."""
    if total < 2:
        raise ValueError("VisionZip requires at least two output tokens.")
    contextual = max(1, round(total * 10 / 64))
    return VisionZipBudget(dominant=total - contextual, contextual=contextual)


def evenly_spaced_targets(torch: Any, available: int, count: int, *, device: Any) -> Any:
    """Match VisionZip's evenly spaced contextual target selection."""
    if count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if available < count:
        raise ValueError(f"Cannot choose {count} contextual targets from {available} tokens.")
    step = max(1, available // count)
    return torch.arange(0, available, step, device=device, dtype=torch.long)[:count]


def visionzip_compress(
    torch: Any,
    patch_hidden: Any,
    cls_attention: Any,
    *,
    output_tokens: int,
    cls_hidden: Any | None = None,
    merge_metric: Any | None = None,
) -> tuple[Any, list[int], dict[str, Any]]:
    """Apply VisionZip dominant selection and contextual token merging.

    ``patch_hidden`` is ``[1, N, D]`` and ``cls_attention`` is either
    ``[1, H, N]`` or ``[H, N]`` from the selected CLIP layer.
    """
    if patch_hidden.ndim != 3 or int(patch_hidden.shape[0]) != 1:
        raise ValueError(f"Expected patch_hidden [1,N,D], got {tuple(patch_hidden.shape)}.")
    original = int(patch_hidden.shape[1])
    if merge_metric is None:
        merge_metric = patch_hidden
        merge_metric_mode = "hidden_state_fallback"
    else:
        if (
            merge_metric.ndim != 3
            or int(merge_metric.shape[0]) != 1
            or int(merge_metric.shape[1]) != original
        ):
            raise ValueError(
                "Expected merge_metric [1,N,D] aligned with patch_hidden, got "
                f"{tuple(merge_metric.shape)}."
            )
        merge_metric_mode = "selected_clip_layer_raw_keys_mean_heads"
    budget = visionzip_budget(output_tokens)
    if cls_hidden is None:
        cls_hidden = patch_hidden[:, :1, :]
        cls_mode = "test_fixture_proxy"
    else:
        if tuple(cls_hidden.shape) != (1, 1, int(patch_hidden.shape[2])):
            raise ValueError(f"Expected cls_hidden [1,1,D], got {tuple(cls_hidden.shape)}.")
        cls_mode = "official_clip_cls_token"
    dominant_patch_count = budget.dominant - 1
    if output_tokens > original:
        raise ValueError(f"Output budget {output_tokens} exceeds {original} input patches.")
    if cls_attention.ndim == 3:
        scores = cls_attention.sum(dim=1)[0]
    elif cls_attention.ndim == 2:
        scores = cls_attention.sum(dim=0)
    else:
        raise ValueError(f"Expected CLS attention [1,H,N] or [H,N], got {tuple(cls_attention.shape)}.")
    if int(scores.numel()) != original:
        raise ValueError(f"Attention has {int(scores.numel())} patches; expected {original}.")

    dominant = torch.topk(scores, dominant_patch_count, largest=True, sorted=False).indices
    dominant = torch.sort(dominant).values
    remaining_mask = torch.ones(original, dtype=torch.bool, device=patch_hidden.device)
    remaining_mask[dominant] = False
    remaining_indices = torch.nonzero(remaining_mask, as_tuple=False).flatten()
    targets_rel = evenly_spaced_targets(
        torch, int(remaining_indices.numel()), budget.contextual, device=patch_hidden.device
    )
    contextual_indices = remaining_indices.index_select(0, targets_rel)

    remaining_hidden = patch_hidden.index_select(1, remaining_indices)
    remaining_metric = merge_metric.index_select(1, remaining_indices)
    target_hidden = patch_hidden.index_select(1, contextual_indices)
    merge_mask = torch.ones(int(remaining_indices.numel()), dtype=torch.bool, device=patch_hidden.device)
    merge_mask[targets_rel] = False
    merge_hidden = remaining_hidden[:, merge_mask, :]
    if int(merge_hidden.shape[1]) > 0:
        target_metric = torch.nn.functional.normalize(
            remaining_metric.index_select(1, targets_rel).float(), p=2, dim=-1
        )
        source_metric = torch.nn.functional.normalize(
            remaining_metric[:, merge_mask, :].float(), p=2, dim=-1
        )
        assignments = torch.bmm(source_metric, target_metric.transpose(1, 2)).argmax(dim=2)
        one_hot = torch.nn.functional.one_hot(
            assignments, num_classes=budget.contextual
        ).to(dtype=merge_hidden.dtype)
        counts = one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
        aggregated = torch.bmm(one_hot.transpose(1, 2), merge_hidden) / counts
        contextual_hidden = target_hidden + aggregated
    else:
        contextual_hidden = target_hidden

    output = torch.cat(
        [cls_hidden, patch_hidden.index_select(1, dominant), contextual_hidden], dim=1
    )
    representative = torch.sort(torch.cat([dominant, contextual_indices])).values
    # The class token has no patch coordinate.  Keep a deterministic central
    # proxy solely for visualisation/schema alignment; it is not fed to the LLM.
    representative_mask = torch.ones(original, dtype=torch.bool, device=patch_hidden.device)
    representative_mask[representative] = False
    candidates = torch.nonzero(representative_mask, as_tuple=False).flatten()
    grid = int(round(original**0.5))
    rows = torch.div(candidates, grid, rounding_mode="floor").float()
    cols = candidates.remainder(grid).float()
    centre = (grid - 1) / 2
    cls_proxy = candidates[((rows - centre) ** 2 + (cols - centre) ** 2).argmin()].reshape(1)
    representative = torch.sort(torch.cat([representative, cls_proxy])).values
    indices = [int(item) for item in representative.detach().cpu().tolist()]
    if int(output.shape[1]) != output_tokens or len(indices) != output_tokens:
        raise ValueError("VisionZip did not produce the exact registered output budget.")
    return output, indices, {
        "visionzip_dominant_tokens": budget.dominant,
        "visionzip_dominant_patch_tokens": dominant_patch_count,
        "visionzip_cls_tokens": 1,
        "visionzip_contextual_tokens": budget.contextual,
        "visionzip_output_tokens": output_tokens,
        "visionzip_scoring": "selected_clip_layer_cls_attention_sum_heads",
        "visionzip_merge": "cosine_assignment_mean_plus_target",
        "visionzip_merge_metric": merge_metric_mode,
        "visionzip_cls_mode": cls_mode,
        "retained_indices_semantics": (
            "39 dominant and 8 contextual representative patches plus one centre proxy "
            "for the coordinate-free CLS token; proxy is not an LLM input"
        ),
    }
