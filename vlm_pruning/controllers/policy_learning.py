"""Offline policy-learning helpers for UG-Prune action grids."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PRIMARY_BASELINE_ACTION = "k12_b48_more_context_soft_average_query_max"
K12_B58_REFERENCE_ACTION = "k12_b58_default_soft_average_query_max"

DEFAULT_REWARD_MODE = "default_utility"
ACCURACY_FIRST_REWARD_MODE = "accuracy_first"

NUMERIC_STATE_FEATURES: tuple[str, ...] = (
    "ug/map_entropy_norm",
    "ug/map_peakiness",
    "ug/map_top_16_mass",
    "ug/map_top_29_mass",
    "ug/map_top_48_mass",
    "ug/map_top_58_mass",
    "ug/map_top_72_mass",
    "ug/map_top_96_mass",
    "ug/map_top_16_occupied_bins_3x3",
    "ug/map_top_48_occupied_bins_3x3",
    "ug/map_top_58_occupied_bins_3x3",
    "ug/map_top_72_occupied_bins_3x3",
    "ug/map_top_96_occupied_bins_3x3",
    "ug/map_top_16_spatial_entropy_3x3",
    "ug/map_top_48_spatial_entropy_3x3",
    "ug/map_top_58_spatial_entropy_3x3",
    "ug/map_top_72_spatial_entropy_3x3",
    "ug/map_top_96_spatial_entropy_3x3",
    "ug/map_selected_occupied_bins_3x3",
    "ug/map_selected_mean_pairwise_spread",
    "ug/map_selected_row_span",
    "ug/map_selected_col_span",
)

TASK_STATE_FEATURES: tuple[str, ...] = (
    "task/is_grounding",
    "task/is_open_vqa",
    "task/is_yes_no_vqa",
)

LIST_STATE_FEATURES: tuple[str, ...] = (
    "ug/map_downsample_6x6",
    "ug/map_top48_mask_6x6",
    "ug/map_top58_mask_6x6",
    "ug/map_top72_mask_6x6",
    "ug/map_top96_mask_6x6",
    "ug/map_selected_mask_6x6",
)

ACTION_NUMERIC_FEATURES: tuple[str, ...] = (
    "action/prune_after_layer_norm",
    "action/token_fraction",
    "action/is_default_role",
    "action/is_more_context",
    "action/is_more_anchors",
    "action/is_standard_selection",
    "action/is_coverage_selection",
    "action/is_coverage_zip_selection",
)

LEGACY_ACTION_FEATURE_WIDTH = 5
MASKED_ACTION_REWARD = -10.0

COLOR_WORDS = frozenset(
    "black blue brown gray green grey orange pink purple red silver white yellow".split()
)
SPATIAL_WORDS = frozenset(
    "above below bottom center centre left middle near right top under underneath upper lower front back".split()
)
RELATION_WORDS = frozenset(
    "beside between behind next over wearing holding sitting standing on in with beside".split()
)
PERSON_WORDS = frozenset("man woman boy girl person people child player".split())


@dataclass(frozen=True)
class ParsedAction:
    name: str
    prune_after_layer: int
    target_visual_tokens: int
    role_preset: str
    fusion_mode: str
    map_source: str
    selection_strategy: str = "standard"
    zip_alpha: float = 0.20


@dataclass(frozen=True)
class PolicyConstraintDecision:
    selected_index: int
    raw_best_index: int
    selected_action: str
    raw_best_action: str
    b96_allowed: bool
    b96_margin_vs_best_non_b96: float | None
    block_reasons: tuple[str, ...]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return out


def parse_action_config(row_or_action: dict[str, Any] | str) -> ParsedAction:
    if isinstance(row_or_action, dict):
        action_name = str(row_or_action.get("action") or "")
        config = row_or_action.get("action_config") if isinstance(row_or_action.get("action_config"), dict) else {}
        if config:
            return ParsedAction(
                name=str(config.get("name") or action_name),
                prune_after_layer=int(config.get("prune_after_layer")),
                target_visual_tokens=int(config.get("target_visual_tokens")),
                role_preset=str(config.get("role_preset") or "default"),
                fusion_mode=str(config.get("fusion_mode") or "soft_average"),
                map_source=str(config.get("map_source") or "query_max"),
                selection_strategy=str(config.get("selection_strategy") or "standard"),
                zip_alpha=safe_float(config.get("zip_alpha"), 0.20),
            )
        name = action_name
    else:
        name = str(row_or_action)

    match = re.match(r"^k(?P<k>\d+)_b(?P<b>\d+)_(?P<rest>.+)$", name)
    if not match:
        raise ValueError(f"Cannot parse action name: {name}")
    rest = match.group("rest")
    selection_strategy = "standard"
    for strategy_suffix, strategy in (
        ("coverage_zip", "coverage_zip"),
        ("coverage", "coverage"),
        ("standard", "standard"),
    ):
        marker = f"_{strategy_suffix}"
        if rest.endswith(marker):
            selection_strategy = strategy
            rest = rest[: -len(marker)]
            break

    suffixes = (
        ("soft_average_query_max", "soft_average", "query_max"),
        ("soft_average_query_pooled", "soft_average", "query_pooled"),
        ("single_layer_query_max", "single_layer", "query_max"),
        ("single_layer_query_pooled", "single_layer", "query_pooled"),
        ("hard_best_query_max", "hard_best", "query_max"),
        ("hard_best_query_pooled", "hard_best", "query_pooled"),
    )
    for suffix, fusion, source in suffixes:
        marker = f"_{suffix}"
        if rest.endswith(marker):
            role = rest[: -len(marker)]
            return ParsedAction(
                name=name,
                prune_after_layer=int(match.group("k")),
                target_visual_tokens=int(match.group("b")),
                role_preset=role,
                fusion_mode=fusion,
                map_source=source,
                selection_strategy=selection_strategy,
                zip_alpha=0.20,
            )
    raise ValueError(f"Cannot parse action suffix: {name}")


def action_feature_values(row_or_action: dict[str, Any] | str, *, width: int | None = None) -> list[float]:
    action = parse_action_config(row_or_action)
    role = action.role_preset.replace("-", "_")
    values = [
        action.prune_after_layer / 32.0,
        action.target_visual_tokens / 576.0,
        1.0 if role in {"default", "current", "target_context_anchor"} else 0.0,
        1.0 if role == "more_context" else 0.0,
        1.0 if role == "more_anchors" else 0.0,
        1.0 if action.selection_strategy == "standard" else 0.0,
        1.0 if action.selection_strategy == "coverage" else 0.0,
        1.0 if action.selection_strategy == "coverage_zip" else 0.0,
    ]
    if width is None or width == len(values):
        return values
    if width == LEGACY_ACTION_FEATURE_WIDTH:
        return values[:LEGACY_ACTION_FEATURE_WIDTH]
    raise ValueError(f"Unsupported action feature width {width}; current={len(values)}, legacy={LEGACY_ACTION_FEATURE_WIDTH}")


def action_budget(row_or_action: dict[str, Any] | str) -> int:
    return int(parse_action_config(row_or_action).target_visual_tokens)


def task_family_from_row(row: dict[str, Any] | None) -> str:
    row = row or {}
    task_family = str(row.get("task_family") or "").lower()
    dataset = str(row.get("dataset") or "").lower()
    answer_mode = str(row.get("answer_mode") or "").lower()
    task = str(row.get("task") or "").lower()
    if task_family in {"grounding", "open_vqa", "yes_no_vqa"}:
        return task_family
    if answer_mode == "yes_no" or dataset == "pope":
        return "yes_no_vqa"
    if dataset == "textvqa" or (task == "vqa" and answer_mode != "yes_no"):
        return "open_vqa"
    if task == "grounding" or dataset == "refcoco" or row.get("box_iou") is not None:
        return "grounding"
    return "open_vqa"


def is_core_v6_low_token_action(action: ParsedAction) -> bool:
    role = action.role_preset.replace("-", "_")
    return (
        int(action.prune_after_layer) == 12
        and int(action.target_visual_tokens) in {48, 58}
        and role in {"default", "more_context"}
        and action.fusion_mode == "soft_average"
        and action.map_source == "query_max"
        and action.selection_strategy in {"standard", "coverage_zip"}
    )


def is_v6_rescue_allowed(action_name: str, feature_row: dict[str, Any] | None) -> bool:
    action = parse_action_config(action_name)
    family = task_family_from_row(feature_row)
    role = action.role_preset.replace("-", "_")
    is_standard_more_context_rescue = (
        int(action.prune_after_layer) == 12
        and role == "more_context"
        and action.fusion_mode == "soft_average"
        and action.map_source == "query_max"
        and action.selection_strategy == "standard"
    )
    if is_core_v6_low_token_action(action):
        return True
    if family == "grounding":
        return is_standard_more_context_rescue and int(action.target_visual_tokens) == 144
    if family == "open_vqa":
        return is_standard_more_context_rescue and int(action.target_visual_tokens) in {144, 288}
    if family == "yes_no_vqa":
        return False
    return False


def allowed_action_indices(
    candidate_actions: Sequence[str],
    feature_row: dict[str, Any] | None = None,
    action_constraints: dict[str, Any] | None = None,
) -> list[int]:
    constraints = action_constraints or {}
    task_mask = str(constraints.get("task_action_mask") or "none").lower()
    if task_mask in {"", "none", "allow", "all"}:
        return list(range(len(candidate_actions)))
    if task_mask == "v6_rescue":
        return [
            idx
            for idx, action in enumerate(candidate_actions)
            if is_v6_rescue_allowed(str(action), feature_row)
        ]
    raise ValueError(f"Unsupported task_action_mask: {task_mask}")


def policy_constraint_decision(
    scores: Sequence[float],
    candidate_actions: Sequence[str],
    feature_row: dict[str, Any] | None = None,
    action_constraints: dict[str, Any] | None = None,
) -> PolicyConstraintDecision:
    """Apply transparent deployment constraints to policy scores.

    The main use is the efficiency-policy rescue setting: B96 stays available
    only when the model strongly prefers it and the map looks sufficiently
    uncertain.  If the raw best action is not B96, constraints are a no-op.
    """

    if len(scores) != len(candidate_actions):
        raise ValueError(f"Score/action length mismatch: {len(scores)} scores vs {len(candidate_actions)} actions")
    if not scores:
        raise ValueError("At least one policy score is required.")

    constraints = action_constraints or {}
    b96_policy = str(constraints.get("b96_policy") or "allow").lower()
    raw_best = max(range(len(scores)), key=lambda idx: (float(scores[idx]), -idx))
    raw_action = str(candidate_actions[raw_best])
    allowed = allowed_action_indices(candidate_actions, feature_row, constraints)
    if not allowed:
        raise ValueError("Action constraints leave no valid actions.")
    if raw_best not in allowed:
        selected = max(allowed, key=lambda idx: (float(scores[idx]), -idx))
        return PolicyConstraintDecision(
            selected_index=int(selected),
            raw_best_index=int(raw_best),
            selected_action=str(candidate_actions[selected]),
            raw_best_action=raw_action,
            b96_allowed=action_budget(str(candidate_actions[selected])) != 96,
            b96_margin_vs_best_non_b96=None,
            block_reasons=("task_action_mask",),
        )

    non_b96 = [idx for idx in allowed if action_budget(str(candidate_actions[idx])) != 96]
    if b96_policy == "forbid":
        if not non_b96:
            raise ValueError("b96_policy=forbid leaves no valid actions.")
        selected = max(non_b96, key=lambda idx: (float(scores[idx]), -idx))
        return PolicyConstraintDecision(
            selected_index=int(selected),
            raw_best_index=int(raw_best),
            selected_action=str(candidate_actions[selected]),
            raw_best_action=raw_action,
            b96_allowed=False,
            b96_margin_vs_best_non_b96=None,
            block_reasons=("b96_forbidden",) if action_budget(raw_action) == 96 else (),
        )

    if action_budget(raw_action) != 96 or b96_policy in {"allow", "none", ""}:
        return PolicyConstraintDecision(
            selected_index=int(raw_best),
            raw_best_index=int(raw_best),
            selected_action=raw_action,
            raw_best_action=raw_action,
            b96_allowed=True,
            b96_margin_vs_best_non_b96=None,
            block_reasons=(),
        )

    if not non_b96:
        return PolicyConstraintDecision(
            selected_index=int(raw_best),
            raw_best_index=int(raw_best),
            selected_action=raw_action,
            raw_best_action=raw_action,
            b96_allowed=True,
            b96_margin_vs_best_non_b96=None,
            block_reasons=(),
        )

    best_non_b96 = max(non_b96, key=lambda idx: (float(scores[idx]), -idx))
    margin = float(scores[raw_best]) - float(scores[best_non_b96])
    block_reasons: list[str] = []
    min_margin = constraints.get("b96_min_margin")
    if min_margin is not None and margin < safe_float(min_margin):
        block_reasons.append("q_margin_below_threshold")

    row = feature_row or {}
    entropy_min = constraints.get("b96_entropy_min")
    if entropy_min is not None and safe_float(row.get("ug/map_entropy_norm"), 0.0) < safe_float(entropy_min):
        block_reasons.append("map_entropy_too_low")

    peakiness_max = constraints.get("b96_peakiness_max")
    if peakiness_max is not None and safe_float(row.get("ug/map_peakiness"), 0.0) > safe_float(peakiness_max):
        block_reasons.append("map_peakiness_too_high")

    if block_reasons:
        return PolicyConstraintDecision(
            selected_index=int(best_non_b96),
            raw_best_index=int(raw_best),
            selected_action=str(candidate_actions[best_non_b96]),
            raw_best_action=raw_action,
            b96_allowed=False,
            b96_margin_vs_best_non_b96=margin,
            block_reasons=tuple(block_reasons),
        )

    return PolicyConstraintDecision(
        selected_index=int(raw_best),
        raw_best_index=int(raw_best),
        selected_action=raw_action,
        raw_best_action=raw_action,
        b96_allowed=True,
        b96_margin_vs_best_non_b96=margin,
        block_reasons=(),
    )


def tokenise_query(query: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(query or "").lower())


def query_feature_names() -> list[str]:
    return [
        "query/token_count_norm",
        "query/has_color",
        "query/has_spatial",
        "query/has_relation",
        "query/has_person",
    ]


def query_feature_values(row: dict[str, Any]) -> list[float]:
    query = row.get("query") or row.get("question") or row.get("text") or row.get("prompt") or ""
    tokens = tokenise_query(str(query))
    token_set = set(tokens)
    return [
        min(len(tokens), 32) / 32.0,
        1.0 if token_set & COLOR_WORDS else 0.0,
        1.0 if token_set & SPATIAL_WORDS else 0.0,
        1.0 if token_set & RELATION_WORDS else 0.0,
        1.0 if token_set & PERSON_WORDS else 0.0,
    ]


def task_feature_values(row: dict[str, Any]) -> list[float]:
    task_family = str(row.get("task_family") or "").lower()
    dataset = str(row.get("dataset") or "").lower()
    answer_mode = str(row.get("answer_mode") or "").lower()
    task = str(row.get("task") or "").lower()
    is_grounding = task_family == "grounding" or task == "grounding" or dataset == "refcoco" or row.get("box_iou") is not None
    is_yes_no = task_family == "yes_no_vqa" or answer_mode == "yes_no" or dataset == "pope"
    is_open_vqa = (
        task_family == "open_vqa"
        or (task == "vqa" and not is_yes_no)
        or (dataset in {"textvqa", "ocr_vqa"} and not is_yes_no)
    )
    return [
        1.0 if is_grounding else 0.0,
        1.0 if is_open_vqa else 0.0,
        1.0 if is_yes_no else 0.0,
    ]


def _list_values(value: Any, expected: int = 36) -> list[float]:
    if not isinstance(value, list):
        return [0.0] * expected
    out = [safe_float(item) for item in value[:expected]]
    if len(out) < expected:
        out.extend([0.0] * (expected - len(out)))
    return out


def state_feature_names(*, include_action_features: bool = False) -> list[str]:
    names = list(NUMERIC_STATE_FEATURES)
    for field in LIST_STATE_FEATURES:
        names.extend(f"{field}[{idx:02d}]" for idx in range(36))
    names.extend(query_feature_names())
    names.extend(TASK_STATE_FEATURES)
    if include_action_features:
        names.extend(ACTION_NUMERIC_FEATURES)
    return names


def state_feature_values(row: dict[str, Any], *, include_action_features: bool = False) -> list[float]:
    values: list[float] = [safe_float(row.get(name)) for name in NUMERIC_STATE_FEATURES]
    for field in LIST_STATE_FEATURES:
        values.extend(_list_values(row.get(field), expected=36))
    values.extend(query_feature_values(row))
    values.extend(task_feature_values(row))
    if include_action_features:
        values.extend(action_feature_values(row))
    return values


def state_feature_values_for_names(row: dict[str, Any], feature_names: Sequence[str]) -> list[float]:
    """Return policy features matching a checkpoint's saved feature schema."""

    current_names = state_feature_names(include_action_features=False)
    current_values = state_feature_values(row, include_action_features=False)
    values_by_name = dict(zip(current_names, current_values))
    if all(name in values_by_name for name in feature_names):
        return [values_by_name[name] for name in feature_names]
    return [values_by_name.get(name, safe_float(row.get(name))) for name in feature_names]


def policy_mlp(in_dim: int, out_dim: int, hidden_dim: int):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim // 2),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim // 2, out_dim),
    )


def standardize_values(values: Sequence[float], means: Sequence[float], stds: Sequence[float]) -> list[float]:
    if len(values) != len(means) or len(values) != len(stds):
        raise ValueError(f"Feature width mismatch: values={len(values)} means={len(means)} stds={len(stds)}")
    return [
        (float(value) - float(mean)) / max(float(std), 1e-6)
        for value, mean, std in zip(values, means, stds)
    ]


class LoadedUGPrunePolicy:
    """Small offline-trained policy checkpoint usable in the mid-layer runner."""

    def __init__(self, checkpoint_path: str | Path, *, device: str | Any = "cpu") -> None:
        import torch

        self.path = str(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.payload = payload
        self.policy_name = str(payload["policy_name"])
        self.policy_type = str(payload["policy_type"])
        self.reward_mode = str(payload["reward_mode"])
        self.candidate_actions = [str(action) for action in payload["candidate_actions"]]
        self.feature_action = str(payload.get("feature_action") or PRIMARY_BASELINE_ACTION)
        self.fallback_feature_action = str(payload.get("fallback_feature_action") or K12_B58_REFERENCE_ACTION)
        self.feature_names = [str(name) for name in payload["feature_names"]]
        self.means = [float(value) for value in payload["normalization"]["means"]]
        self.stds = [float(value) for value in payload["normalization"]["stds"]]
        self.hidden_dim = int(payload["hidden_dim"])
        self.input_dim = int(payload["input_dim"])
        self.output_dim = int(payload["output_dim"])
        self.action_constraints = dict(payload.get("action_constraints") or {})
        self.reward_config = dict(payload.get("reward_config") or {})
        self.model = policy_mlp(self.input_dim, self.output_dim, self.hidden_dim).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.device = device

    def predict_action(self, feature_row: dict[str, Any]) -> dict[str, Any]:
        import torch

        state = state_feature_values_for_names(feature_row, self.feature_names)
        if len(state) != len(self.feature_names):
            raise ValueError(f"Policy feature width mismatch: got {len(state)}, expected {len(self.feature_names)}")
        state = standardize_values(state, self.means, self.stds)
        with torch.inference_mode():
            if self.policy_type == "q":
                action_width = int(self.input_dim) - len(state)
                rows = [state + action_feature_values(action, width=action_width) for action in self.candidate_actions]
                scores_tensor = self.model(torch.tensor(rows, dtype=torch.float32, device=self.device)).squeeze(1)
            else:
                scores_tensor = self.model(torch.tensor([state], dtype=torch.float32, device=self.device)).squeeze(0)
            scores = [float(value) for value in scores_tensor.detach().cpu().tolist()]
        constraint_decision = policy_constraint_decision(scores, self.candidate_actions, feature_row, self.action_constraints)
        best_idx = constraint_decision.selected_index
        action_name = self.candidate_actions[best_idx]
        parsed = parse_action_config(action_name)
        return {
            "action": action_name,
            "raw_action": constraint_decision.raw_best_action,
            "action_config": parsed,
            "score": scores[best_idx],
            "raw_score": scores[constraint_decision.raw_best_index],
            "scores": dict(zip(self.candidate_actions, scores)),
            "action_constraints": self.action_constraints,
            "constraint_decision": {
                "b96_allowed": constraint_decision.b96_allowed,
                "b96_margin_vs_best_non_b96": constraint_decision.b96_margin_vs_best_non_b96,
                "block_reasons": list(constraint_decision.block_reasons),
                "raw_best_action": constraint_decision.raw_best_action,
                "selected_action": constraint_decision.selected_action,
            },
            "policy_name": self.policy_name,
            "policy_type": self.policy_type,
            "reward_mode": self.reward_mode,
            "checkpoint": self.path,
        }


def parse_failed(row: dict[str, Any]) -> bool:
    return bool(row.get("invalid") or not row.get("valid", True))


def default_utility_reward(row: dict[str, Any]) -> float:
    if row.get("utility") is not None:
        return safe_float(row.get("utility"))
    iou = safe_float(row.get("box_iou"))
    norm_latency = safe_float(row.get("normalized_latency"), 1.0)
    token_fraction = safe_float(row.get("token_fraction"), safe_float(row.get("actual_llm_visual_tokens"), 576.0) / 576.0)
    penalty = 0.25 if parse_failed(row) else 0.0
    return iou - 0.05 * norm_latency - 0.02 * token_fraction - penalty


def accuracy_first_reward(row: dict[str, Any]) -> float:
    success = 1.0 if row.get("success_iou_0_5") else 0.0
    iou = safe_float(row.get("box_iou"))
    norm_latency = safe_float(row.get("normalized_latency"), 1.0)
    token_fraction = safe_float(row.get("token_fraction"), safe_float(row.get("actual_llm_visual_tokens"), 576.0) / 576.0)
    penalty = 1.0 if parse_failed(row) else 0.0
    return success + 0.25 * iou - 0.05 * norm_latency - 0.02 * token_fraction - penalty


def reward_for_mode(row: dict[str, Any], mode: str) -> float:
    if mode == DEFAULT_REWARD_MODE:
        return default_utility_reward(row)
    if mode == ACCURACY_FIRST_REWARD_MODE:
        return accuracy_first_reward(row)
    raise ValueError(f"Unsupported reward mode: {mode}")


def group_rows_by_sample(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id"))
        action = str(row.get("action"))
        if not sample_id or not action:
            continue
        grouped.setdefault(sample_id, {})[action] = row
    return grouped


def oracle_action_for_sample(rows_by_action: dict[str, dict[str, Any]], *, reward_mode: str) -> str:
    if not rows_by_action:
        raise ValueError("No candidate rows are available.")
    return max(
        rows_by_action,
        key=lambda action: (
            reward_for_mode(rows_by_action[action], reward_mode),
            safe_float(rows_by_action[action].get("box_iou")),
            -safe_float(rows_by_action[action].get("latency_s")),
            action,
        ),
    )
