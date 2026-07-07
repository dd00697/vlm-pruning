"""Shared helpers for UGround-style adaptive UG-Prune experiments.

This module keeps the experimental action space explicit and small.  The VLM
stays frozen; controller inputs are diagnostics produced by the existing
unrolled-map pruning path.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from vlm_pruning.llava_hooks.mid_layer_pruning import ug_gap58_config_for_prune_layer
from vlm_pruning.selectors import UGPruneConfig, SelectorError


ACTION_GRID_DEFAULT_KS = (8, 10, 11, 12)
ACTION_GRID_DEFAULT_BUDGETS = (48, 58, 72, 96)
ACTION_GRID_DEFAULT_ROLE_PRESETS = ("default", "more_context", "more_anchors")

ROLE_PRESETS: dict[str, tuple[float, float, float]] = {
    "default": (0.50, 0.30, 0.20),
    "current": (0.50, 0.30, 0.20),
    "target_context_anchor": (0.50, 0.30, 0.20),
    "more_context": (0.45, 0.40, 0.15),
    "more-context": (0.45, 0.40, 0.15),
    "more_anchors": (0.45, 0.25, 0.30),
    "more-anchor": (0.45, 0.25, 0.30),
}


@dataclass(frozen=True)
class AdaptiveAction:
    """One fixed pruning action for oracle grids and controller labels."""

    prune_after_layer: int
    target_visual_tokens: int = 58
    map_source: str = "query_max"
    fusion_mode: str = "soft_average"
    role_preset: str = "default"
    selection_strategy: str = "standard"
    zip_alpha: float = 0.20

    def validate(self) -> "AdaptiveAction":
        if int(self.prune_after_layer) <= 0:
            raise SelectorError(f"prune_after_layer must be positive, got {self.prune_after_layer}.")
        if int(self.target_visual_tokens) <= 0:
            raise SelectorError(f"target_visual_tokens must be positive, got {self.target_visual_tokens}.")
        if self.map_source not in {"query_max", "query_pooled"}:
            raise SelectorError(f"Unsupported map_source: {self.map_source}")
        if self.fusion_mode not in {"soft_average", "single_layer", "hard_best"}:
            raise SelectorError(f"Unsupported fusion_mode: {self.fusion_mode}")
        if self.selection_strategy not in {"standard", "coverage", "coverage_zip"}:
            raise SelectorError(f"Unsupported selection_strategy: {self.selection_strategy}")
        if not 0.0 <= float(self.zip_alpha) <= 1.0:
            raise SelectorError(f"zip_alpha must be in [0, 1], got {self.zip_alpha}.")
        coerce_role_preset(self.role_preset)
        return self

    @property
    def name(self) -> str:
        return action_name(self)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class FeatureSpec:
    """Feature names used by the lightweight offline controller."""

    names: tuple[str, ...] = (
        "ug/map_entropy_norm",
        "ug/map_peakiness",
        "ug/map_top_16_mass",
        "ug/map_top_29_mass",
        "ug/map_top_48_mass",
        "ug/map_top_58_mass",
        "ug/map_top_72_mass",
        "ug/map_top_96_mass",
        "ug/map_selected_occupied_bins_3x3",
        "ug/map_selected_mean_pairwise_spread",
        "ug/map_selected_row_span",
        "ug/map_selected_col_span",
        "generated_token_count",
        "latency_s",
    )


@dataclass(frozen=True)
class ControllerPrediction:
    sample_id: str
    predicted_action: str
    confidence: float
    scores: dict[str, float]


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._") or "value"


def action_name(action: AdaptiveAction) -> str:
    action = action.validate()
    name = (
        f"k{int(action.prune_after_layer)}_"
        f"b{int(action.target_visual_tokens)}_"
        f"{_safe_slug(coerce_role_preset(action.role_preset))}_"
        f"{_safe_slug(action.fusion_mode)}_"
        f"{_safe_slug(action.map_source)}"
    )
    if action.selection_strategy != "standard":
        name += f"_{_safe_slug(action.selection_strategy)}"
    return name


def coerce_role_preset(value: str) -> str:
    key = str(value or "default").strip().lower().replace("-", "_")
    if key in {"more_anchor", "more_anchors"}:
        return "more_anchors"
    if key in {"more_context", "context"}:
        return "more_context"
    if key in {"default", "current", "target_context_anchor"}:
        return "default"
    raise SelectorError(f"Unsupported role_preset: {value}")


def role_ratios(role_preset: str) -> tuple[float, float, float]:
    key = coerce_role_preset(role_preset)
    return ROLE_PRESETS[key]


def build_selector_config(action: AdaptiveAction) -> UGPruneConfig:
    action = action.validate()
    target_ratio, context_ratio, anchor_ratio = role_ratios(action.role_preset)
    return ug_gap58_config_for_prune_layer(
        int(action.prune_after_layer),
        map_source=action.map_source,
        fusion_mode=action.fusion_mode,
        target_ratio=target_ratio,
        context_ratio=context_ratio,
        anchor_ratio=anchor_ratio,
        selection_strategy=action.selection_strategy,
        zip_alpha=float(action.zip_alpha),
    )


def parse_csv_ints(value: str | Sequence[int] | None, default: Sequence[int]) -> tuple[int, ...]:
    if value is None:
        return tuple(int(item) for item in default)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            return tuple(int(item) for item in default)
        return tuple(int(part) for part in parts)
    return tuple(int(item) for item in value)


def parse_csv_strings(value: str | Sequence[str] | None, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(str(item) for item in default)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return tuple(parts) if parts else tuple(str(item) for item in default)
    return tuple(str(item) for item in value)


def action_grid(
    *,
    ks: Sequence[int] = ACTION_GRID_DEFAULT_KS,
    budgets: Sequence[int] = ACTION_GRID_DEFAULT_BUDGETS,
    role_presets: Sequence[str] = ACTION_GRID_DEFAULT_ROLE_PRESETS,
    fusion_modes: Sequence[str] = ("soft_average",),
    map_sources: Sequence[str] = ("query_max",),
    selection_strategies: Sequence[str] = ("standard",),
    zip_alpha: float = 0.20,
) -> list[AdaptiveAction]:
    actions: list[AdaptiveAction] = []
    for k in ks:
        for budget in budgets:
            for role in role_presets:
                for fusion in fusion_modes:
                    for source in map_sources:
                        for strategy in selection_strategies:
                            action = AdaptiveAction(
                                prune_after_layer=int(k),
                                target_visual_tokens=int(budget),
                                role_preset=coerce_role_preset(role),
                                fusion_mode=str(fusion),
                                map_source=str(source),
                                selection_strategy=str(strategy),
                                zip_alpha=float(zip_alpha),
                            ).validate()
                            actions.append(action)
    return actions


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def is_reserved_refcoco500_row(row: dict[str, Any]) -> bool:
    metadata = _metadata(row)
    if str(row.get("dataset", "")).lower() != "refcoco":
        return False
    split = str(metadata.get("source_split") or row.get("split") or "").lower()
    if split != "val":
        return False
    source_row = metadata.get("source_row")
    try:
        index = int(source_row)
    except (TypeError, ValueError):
        return False
    return 0 <= index < 500


def assert_allowed_nonfinal_rows(rows: Iterable[dict[str, Any]], *, context: str) -> None:
    blocked = [str(row.get("sample_id")) for row in rows if is_reserved_refcoco500_row(row)]
    if blocked:
        preview = ", ".join(blocked[:5])
        suffix = "" if len(blocked) <= 5 else f", ... (+{len(blocked) - 5})"
        raise ValueError(
            f"{context} includes reserved RefCOCO-500 rows and cannot be used for tuning/model selection: "
            f"{preview}{suffix}"
        )


def extract_feature_dict(row: dict[str, Any], spec: FeatureSpec | None = None) -> dict[str, float]:
    names = (spec or FeatureSpec()).names
    out: dict[str, float] = {}
    for name in names:
        value = row.get(name)
        if value is None and name.startswith("ug/map_"):
            value = row.get(name.replace("ug/map_", ""))
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        if math.isnan(numeric) or math.isinf(numeric):
            numeric = 0.0
        out[name] = numeric
    return out


def oracle_utility(
    *,
    iou: float,
    normalized_latency: float,
    token_fraction: float,
    parse_failed: bool,
    latency_weight: float = 0.05,
    token_weight: float = 0.02,
    parse_failure_penalty: float = 0.25,
) -> float:
    penalty = float(parse_failure_penalty) if parse_failed else 0.0
    return (
        float(iou)
        - float(latency_weight) * float(normalized_latency)
        - float(token_weight) * float(token_fraction)
        - penalty
    )
