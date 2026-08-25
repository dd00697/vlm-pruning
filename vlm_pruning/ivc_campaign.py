"""Frozen contracts and CPU-only analysis helpers for the IVC-Prune campaign."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

IVC_UPSTREAM_URL = "https://github.com/FireRedTeam/IVC-Prune.git"
IVC_UPSTREAM_COMMIT = "03afd9cee41f60bb30b51a4105839f2b5d467814"
IVC_TRANSFORMERS_VERSION = "4.51.3"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MODEL_REVISION = "a272c74b2481d8aff3aa6fc2c4bf891fe57334fb"
METHODS = ("IVC_STACK_FULL", "IVC_PRUNE")
IVC_PRUNE_LAYER_INDEX = 16
ORIGINAL_VISUAL_TOKENS = 576
DECODER_LAYERS = 32
IMAGE_TOKEN_START_INDEX = 5
MAX_NEW_TOKENS = 48
FIXED_CONTINUATION_TOKEN_IDS = tuple(range(1000, 1032))
REGISTERED_PROMPT = (
    'Identify the target described by this referring expression: "{query}".\n'
    "Return exactly one bounding box in normalized 0-1 image coordinates as "
    "[x_min, y_min, x_max, y_max]. Do not include any other text."
)
NATURAL_ORDER_PLAN = {
    1: METHODS,
    2: tuple(reversed(METHODS)),
    3: METHODS,
    4: tuple(reversed(METHODS)),
    5: METHODS,
}
FIXED32_ORDER_PLAN = {
    1: METHODS,
    2: tuple(reversed(METHODS)),
    3: METHODS,
}


class IVCCampaignError(RuntimeError):
    """Raised when evidence violates the frozen IVC-Prune campaign contract."""


def expected_ivc_config(method: str) -> dict[str, int] | None:
    if method == "IVC_STACK_FULL":
        return None
    if method == "IVC_PRUNE":
        return {
            "ivcp_k": IVC_PRUNE_LAYER_INDEX,
            "image_token_start_index": IMAGE_TOKEN_START_INDEX,
            "image_token_length": ORIGINAL_VISUAL_TOKENS,
        }
    raise IVCCampaignError(f"Unregistered IVC method: {method}")


def timing_order(mode: str, repetition: int) -> tuple[str, ...]:
    plans = NATURAL_ORDER_PLAN if mode == "natural" else FIXED32_ORDER_PLAN
    if mode not in {"natural", "fixed32"} or repetition not in plans:
        raise IVCCampaignError(f"Unregistered timing cell: mode={mode}, repetition={repetition}")
    return plans[repetition]


def validate_visual_trace(method: str, trace: Sequence[int]) -> int:
    values = [int(value) for value in trace]
    if len(values) != DECODER_LAYERS:
        raise IVCCampaignError(f"Expected {DECODER_LAYERS} decoder-layer counts, got {len(values)}")
    if method == "IVC_STACK_FULL":
        if values != [ORIGINAL_VISUAL_TOKENS] * DECODER_LAYERS:
            raise IVCCampaignError(f"Full trace is not dense: {values}")
        return ORIGINAL_VISUAL_TOKENS
    if method != "IVC_PRUNE":
        raise IVCCampaignError(f"Unregistered IVC method: {method}")
    dense = values[:IVC_PRUNE_LAYER_INDEX]
    pruned = values[IVC_PRUNE_LAYER_INDEX:]
    if dense != [ORIGINAL_VISUAL_TOKENS] * IVC_PRUNE_LAYER_INDEX:
        raise IVCCampaignError(f"IVC dense-prefix trace differs: {values}")
    if not pruned or len(set(pruned)) != 1 or not 1 <= pruned[0] < ORIGINAL_VISUAL_TOKENS:
        raise IVCCampaignError(f"IVC adaptive suffix trace differs: {values}")
    return pruned[0]


def visual_trace(method: str, retained_visual_tokens: int = ORIGINAL_VISUAL_TOKENS) -> list[int]:
    if method == "IVC_STACK_FULL":
        trace = [ORIGINAL_VISUAL_TOKENS] * DECODER_LAYERS
    elif method == "IVC_PRUNE":
        if not 1 <= int(retained_visual_tokens) < ORIGINAL_VISUAL_TOKENS:
            raise IVCCampaignError("IVC retained-token count must be in [1, 575]")
        trace = (
            [ORIGINAL_VISUAL_TOKENS] * IVC_PRUNE_LAYER_INDEX
            + [int(retained_visual_tokens)] * (DECODER_LAYERS - IVC_PRUNE_LAYER_INDEX)
        )
    else:
        raise IVCCampaignError(f"Unregistered IVC method: {method}")
    validate_visual_trace(method, trace)
    return trace


def retained_visual_indices(
    keep_indices: Sequence[int], *, image_start: int = IMAGE_TOKEN_START_INDEX
) -> list[int]:
    """Convert IVC's full-sequence indices into zero-based visual-patch indices."""
    keep = [int(value) for value in keep_indices]
    if keep != sorted(set(keep)):
        raise IVCCampaignError("IVC keep indices must be sorted and unique")
    stop = int(image_start) + ORIGINAL_VISUAL_TOKENS
    visual = [value - int(image_start) for value in keep if int(image_start) <= value < stop]
    if not visual or visual != sorted(set(visual)) or visual[-1] >= ORIGINAL_VISUAL_TOKENS:
        raise IVCCampaignError("IVC retained visual indices are invalid")
    return visual


def align_rows(
    rows: Sequence[Mapping[str, Any]], manifest: Sequence[Mapping[str, Any]], *, label: str
) -> list[Mapping[str, Any]]:
    if len(rows) != len(manifest):
        raise IVCCampaignError(f"{label} row count differs from the manifest")
    expected = [str(row["sample_id"]) for row in manifest]
    observed = [str(row.get("sample_id")) for row in rows]
    if observed != expected or len(set(observed)) != len(observed):
        raise IVCCampaignError(f"{label} rows are not an ordered one-to-one manifest alignment")
    return list(rows)


def summarize_quality(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise IVCCampaignError("Cannot summarize empty quality rows")
    ious = np.asarray([float(row["box_iou"]) for row in rows], dtype=np.float64)
    valid = np.asarray([bool(row["valid"]) for row in rows], dtype=np.float64)
    correct = np.asarray([bool(row["success_iou_0_5"]) for row in rows], dtype=np.float64)
    work = np.asarray([float(row["visual_token_layer_work"]) for row in rows], dtype=np.float64)
    final = np.asarray([float(row["final_visual_tokens"]) for row in rows], dtype=np.float64)
    if not all(np.isfinite(array).all() for array in (ious, valid, correct, work, final)):
        raise IVCCampaignError("Quality rows contain non-finite values")
    return {
        "rows": len(rows),
        "parse_rate": float(valid.mean()),
        "acc_at_0_5": float(correct.mean()),
        "mean_iou": float(ious.mean()),
        "mean_visual_token_layer_work": float(work.mean()),
        "mean_average_visual_tokens": float(work.mean() / DECODER_LAYERS),
        "mean_final_visual_tokens": float(final.mean()),
        "min_final_visual_tokens": int(final.min()),
        "max_final_visual_tokens": int(final.max()),
    }


def paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 20_260_825,
) -> dict[str, float | int]:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if (
        left_values.ndim != 1
        or left_values.shape != right_values.shape
        or not left_values.size
        or not np.isfinite(left_values).all()
        or not np.isfinite(right_values).all()
        or resamples <= 0
    ):
        raise IVCCampaignError("Invalid paired bootstrap inputs")
    delta = left_values - right_values
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 100):
        count = min(100, resamples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        estimates[start : start + count] = delta[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "point_delta": float(delta.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "paired_rows": len(delta),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
    }

