#!/usr/bin/env python
"""Train offline UG-Prune policies from complete action-grid rewards."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_baseline import BaselineError  # noqa: E402
from run_official_baseline import utc_now, write_json  # noqa: E402
from run_uground_adaptive_grid import evaluate_rule  # noqa: E402
from vlm_pruning.controllers import (  # noqa: E402
    ACCURACY_FIRST_REWARD_MODE,
    DEFAULT_REWARD_MODE,
    K12_B58_REFERENCE_ACTION,
    PRIMARY_BASELINE_ACTION,
    action_feature_values,
    allowed_action_indices,
    policy_constraint_decision,
    group_rows_by_sample,
    parse_action_config,
    policy_mlp,
    reward_for_mode,
    safe_float,
    state_feature_names,
    state_feature_values,
)
from vlm_pruning.controllers.policy_learning import ACTION_NUMERIC_FEATURES  # noqa: E402

TASK_QUALITY_REWARD_MODE = "task_quality"
TASK_ACCURACY_REWARD_MODE = "task_accuracy"


STRICT6_ACTIONS = (
    "k12_b48_default_soft_average_query_max",
    "k12_b48_more_context_soft_average_query_max",
    "k12_b58_default_soft_average_query_max",
    "k12_b58_more_context_soft_average_query_max",
    "k12_b72_default_soft_average_query_max",
    "k12_b72_more_context_soft_average_query_max",
)

PRIMARY6_ACTIONS = (
    "k12_b48_more_context_soft_average_query_max",
    "k12_b58_default_soft_average_query_max",
    "k12_b58_more_context_soft_average_query_max",
    "k12_b72_default_soft_average_query_max",
    "k12_b72_more_context_soft_average_query_max",
    "k12_b96_default_soft_average_query_max",
)

RESCUE7_ACTIONS = (*STRICT6_ACTIONS, "k12_b96_default_soft_average_query_max")

V6_RESCUE_ACTIONS = (
    "k12_b48_default_soft_average_query_max",
    "k12_b48_more_context_soft_average_query_max",
    "k12_b58_default_soft_average_query_max",
    "k12_b58_more_context_soft_average_query_max",
    "k12_b48_default_soft_average_query_max_coverage_zip",
    "k12_b48_more_context_soft_average_query_max_coverage_zip",
    "k12_b58_default_soft_average_query_max_coverage_zip",
    "k12_b58_more_context_soft_average_query_max_coverage_zip",
    "k12_b144_more_context_soft_average_query_max",
    "k12_b288_more_context_soft_average_query_max",
)

ACTION_SETS: dict[str, tuple[str, ...]] = {
    "strict6": STRICT6_ACTIONS,
    "primary6": PRIMARY6_ACTIONS,
    "rescue7": RESCUE7_ACTIONS,
    "v6_rescue": V6_RESCUE_ACTIONS,
}

MASKED_ACTION_REWARD = -10.0


@dataclass(frozen=True)
class RewardRun:
    name: str
    base: str
    latency_weight: float | None = None
    token_weight: float | None = None
    parse_failure_penalty: float | None = None
    constraint_token_target: float | None = None
    constraint_latency_target: float | None = None
    constraint_token_lambda: float = 0.0
    constraint_latency_lambda: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-grid-dir", default=None, type=Path)
    parser.add_argument("--dev-grid-dir", default=None, type=Path)
    parser.add_argument("--train-grid-dirs", default=None, help="Comma-separated train grid dirs. Overrides --train-grid-dir.")
    parser.add_argument("--dev-grid-dirs", default=None, help="Comma-separated dev grid dirs. Overrides --dev-grid-dir.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-actions", default=None, help="Comma-separated action names. Defaults to train/dev intersection.")
    parser.add_argument(
        "--candidate-action-set",
        default=None,
        choices=sorted(ACTION_SETS),
        help="Named compact K=12 action set. Ignored when --candidate-actions is set.",
    )
    parser.add_argument("--feature-action", default=PRIMARY_BASELINE_ACTION)
    parser.add_argument("--fallback-feature-action", default=K12_B58_REFERENCE_ACTION)
    parser.add_argument("--baseline-action", default=PRIMARY_BASELINE_ACTION)
    parser.add_argument("--reference-action", default=K12_B58_REFERENCE_ACTION)
    parser.add_argument(
        "--reward-modes",
        default=f"{DEFAULT_REWARD_MODE},{ACCURACY_FIRST_REWARD_MODE}",
        help="Comma-separated reward modes.",
    )
    parser.add_argument(
        "--reward-sweep-token-weights",
        default=None,
        help="Comma-separated token penalties used to create efficiency IoU rewards, e.g. 0.05,0.10,0.20.",
    )
    parser.add_argument(
        "--reward-sweep-latency-weights",
        default=None,
        help="Comma-separated latency penalties used to create efficiency IoU rewards, e.g. 0.10,0.20.",
    )
    parser.add_argument(
        "--reward-sweep-bases",
        default="iou",
        help="Comma-separated reward bases for token/latency sweeps, e.g. iou,task_quality,task_accuracy.",
    )
    parser.add_argument("--task-balanced", action="store_true", help="Weight samples so each task family contributes equally.")
    parser.add_argument(
        "--lagrangian-token-target",
        type=float,
        default=None,
        help="Optional average-token constraint target. Values >1 are interpreted as visual tokens; <=1 as token fraction.",
    )
    parser.add_argument(
        "--lagrangian-latency-target",
        type=float,
        default=None,
        help="Optional normalized-latency constraint target.",
    )
    parser.add_argument("--lagrangian-token-lambdas", default=None, help="Comma-separated token-constraint multipliers.")
    parser.add_argument("--lagrangian-latency-lambdas", default=None, help="Comma-separated latency-constraint multipliers.")
    parser.add_argument(
        "--policy-types",
        default="supervised_mlp,reward_weighted_mlp,q_mlp,pairwise_rank_mlp",
        help="Comma-separated policy types to train.",
    )
    parser.add_argument(
        "--cql-alphas",
        default="0",
        help="Comma-separated CQL-style Q penalties. 0 keeps the original Q objective.",
    )
    parser.add_argument("--include-bandit-pg", action="store_true", help="Train an offline policy-gradient contextual-bandit MLP.")
    parser.add_argument("--bandit-kl-lambda", type=float, default=0.01)
    parser.add_argument(
        "--b96-policy",
        default="allow",
        choices=["allow", "forbid", "q_margin", "uncertainty"],
        help="Deployment constraint for actions with B=96.",
    )
    parser.add_argument("--b96-min-margin", type=float, default=None)
    parser.add_argument("--b96-entropy-min", type=float, default=None)
    parser.add_argument("--b96-peakiness-max", type=float, default=None)
    parser.add_argument(
        "--task-action-mask",
        default="none",
        choices=["none", "v6_rescue"],
        help="Task-aware action mask used during training, dev evaluation, and saved checkpoint deployment.",
    )
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_float_csv(value: str | None) -> list[float]:
    return [float(part) for part in parse_csv(value)]


def float_slug(value: float) -> str:
    text = f"{float(value):.4g}"
    return text.replace("-", "m").replace(".", "p")


def token_constraint_fraction(target: float | None, row: dict[str, Any]) -> float | None:
    if target is None:
        return None
    if target <= 1.0:
        return float(target)
    original = safe_float(row.get("original_visual_tokens"), 576.0)
    return float(target) / max(original, 1.0)


def row_task_family(row: dict[str, Any]) -> str:
    task_family = str(row.get("task_family") or "").lower()
    dataset = str(row.get("dataset") or "").lower()
    task = str(row.get("task") or "").lower()
    answer_mode = str(row.get("answer_mode") or "").lower()
    if task_family in {"grounding", "open_vqa", "yes_no_vqa"}:
        return task_family
    if answer_mode == "yes_no" or dataset == "pope":
        return "yes_no_vqa"
    if dataset == "textvqa" or (task == "vqa" and answer_mode != "yes_no"):
        return "open_vqa"
    if task == "grounding" or dataset == "refcoco" or row.get("box_iou") is not None:
        return "grounding"
    return "open_vqa"


def row_task_quality(row: dict[str, Any]) -> float:
    if row_task_family(row) == "grounding":
        return safe_float(row.get("box_iou"))
    return safe_float(row.get("score"))


def row_task_accuracy(row: dict[str, Any]) -> float:
    if row_task_family(row) == "grounding":
        return 1.0 if row.get("success_iou_0_5") else 0.0
    return 1.0 if row.get("correct") else 0.0


def custom_reward(row: dict[str, Any], reward_run: RewardRun) -> float:
    if reward_run.base in {DEFAULT_REWARD_MODE, ACCURACY_FIRST_REWARD_MODE}:
        return reward_for_mode(row, reward_run.base)

    iou = safe_float(row.get("box_iou"))
    task_quality = row_task_quality(row)
    task_accuracy = row_task_accuracy(row)
    norm_latency = safe_float(row.get("normalized_latency"), 1.0)
    token_fraction = safe_float(
        row.get("token_fraction"),
        safe_float(row.get("actual_llm_visual_tokens"), 576.0) / 576.0,
    )
    parse_failed = bool(row.get("invalid") or not row.get("valid", True))
    penalty = float(reward_run.parse_failure_penalty or 0.0) if parse_failed else 0.0
    if reward_run.base == "accuracy":
        reward = (
            (1.0 if row.get("success_iou_0_5") else 0.0)
            + 0.25 * iou
            - float(reward_run.latency_weight or 0.0) * norm_latency
            - float(reward_run.token_weight or 0.0) * token_fraction
            - penalty
        )
    elif reward_run.base == "iou":
        reward = (
            iou
            - float(reward_run.latency_weight or 0.0) * norm_latency
            - float(reward_run.token_weight or 0.0) * token_fraction
            - penalty
        )
    elif reward_run.base == TASK_QUALITY_REWARD_MODE:
        reward = (
            task_quality
            - float(reward_run.latency_weight or 0.0) * norm_latency
            - float(reward_run.token_weight or 0.0) * token_fraction
            - penalty
        )
    elif reward_run.base == TASK_ACCURACY_REWARD_MODE:
        reward = (
            task_accuracy
            + 0.25 * task_quality
            - float(reward_run.latency_weight or 0.0) * norm_latency
            - float(reward_run.token_weight or 0.0) * token_fraction
            - penalty
        )
    else:
        raise ValueError(f"Unsupported reward base: {reward_run.base}")

    token_target = token_constraint_fraction(reward_run.constraint_token_target, row)
    if token_target is not None and reward_run.constraint_token_lambda:
        reward -= float(reward_run.constraint_token_lambda) * max(0.0, token_fraction - token_target)
    if reward_run.constraint_latency_target is not None and reward_run.constraint_latency_lambda:
        reward -= float(reward_run.constraint_latency_lambda) * max(0.0, norm_latency - float(reward_run.constraint_latency_target))
    return float(reward)


def reward_run_name(
    *,
    base: str,
    latency_weight: float,
    token_weight: float,
    parse_failure_penalty: float,
    constraint_token_target: float | None = None,
    constraint_latency_target: float | None = None,
    constraint_token_lambda: float = 0.0,
    constraint_latency_lambda: float = 0.0,
) -> str:
    name = (
        f"{base}_l{float_slug(latency_weight)}_"
        f"t{float_slug(token_weight)}_p{float_slug(parse_failure_penalty)}"
    )
    if constraint_token_target is not None or constraint_latency_target is not None:
        name += (
            f"_ctok{float_slug(constraint_token_target or 0.0)}"
            f"_clat{float_slug(constraint_latency_target or 0.0)}"
            f"_ltok{float_slug(constraint_token_lambda)}"
            f"_llat{float_slug(constraint_latency_lambda)}"
        )
    return name


def build_reward_runs(args: argparse.Namespace) -> list[RewardRun]:
    runs: list[RewardRun] = []
    for mode in parse_csv(args.reward_modes):
        runs.append(RewardRun(name=mode, base=mode))

    token_weights = parse_float_csv(args.reward_sweep_token_weights)
    latency_weights = parse_float_csv(args.reward_sweep_latency_weights)
    if token_weights or latency_weights:
        if not token_weights or not latency_weights:
            raise BaselineError("--reward-sweep-token-weights and --reward-sweep-latency-weights must be provided together.")
        sweep_bases = parse_csv(args.reward_sweep_bases) or ["iou"]
        for token_weight in token_weights:
            for latency_weight in latency_weights:
                for base in sweep_bases:
                    runs.append(
                        RewardRun(
                            name=reward_run_name(
                                base=base,
                                latency_weight=latency_weight,
                                token_weight=token_weight,
                                parse_failure_penalty=0.25,
                            ),
                            base=base,
                            latency_weight=latency_weight,
                            token_weight=token_weight,
                            parse_failure_penalty=0.25,
                        )
                    )

    token_lambdas = parse_float_csv(args.lagrangian_token_lambdas)
    latency_lambdas = parse_float_csv(args.lagrangian_latency_lambdas)
    if token_lambdas or latency_lambdas:
        if not token_lambdas or not latency_lambdas:
            raise BaselineError("--lagrangian-token-lambdas and --lagrangian-latency-lambdas must be provided together.")
        base_pairs = [
            run for run in runs
            if run.base in {"iou", "accuracy", TASK_QUALITY_REWARD_MODE, TASK_ACCURACY_REWARD_MODE}
            and run.latency_weight is not None
            and run.token_weight is not None
        ]
        if not base_pairs:
            base_pairs = [
                RewardRun(
                    name=reward_run_name(base="iou", latency_weight=0.05, token_weight=0.02, parse_failure_penalty=0.25),
                    base="iou",
                    latency_weight=0.05,
                    token_weight=0.02,
                    parse_failure_penalty=0.25,
                )
            ]
        for base_run in base_pairs:
            for token_lambda in token_lambdas:
                for latency_lambda in latency_lambdas:
                    runs.append(
                        RewardRun(
                            name=reward_run_name(
                                base=base_run.base,
                                latency_weight=float(base_run.latency_weight or 0.0),
                                token_weight=float(base_run.token_weight or 0.0),
                                parse_failure_penalty=float(base_run.parse_failure_penalty or 0.0),
                                constraint_token_target=args.lagrangian_token_target,
                                constraint_latency_target=args.lagrangian_latency_target,
                                constraint_token_lambda=token_lambda,
                                constraint_latency_lambda=latency_lambda,
                            ),
                            base=base_run.base,
                            latency_weight=base_run.latency_weight,
                            token_weight=base_run.token_weight,
                            parse_failure_penalty=base_run.parse_failure_penalty,
                            constraint_token_target=args.lagrangian_token_target,
                            constraint_latency_target=args.lagrangian_latency_target,
                            constraint_token_lambda=token_lambda,
                            constraint_latency_lambda=latency_lambda,
                        )
                    )

    deduped: dict[str, RewardRun] = {}
    for run in runs:
        deduped[run.name] = run
    out = list(deduped.values())
    if not out:
        raise BaselineError("At least one reward mode/spec is required.")
    return out


def action_constraints_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "b96_policy": args.b96_policy,
        "b96_min_margin": args.b96_min_margin,
        "b96_entropy_min": args.b96_entropy_min,
        "b96_peakiness_max": args.b96_peakiness_max,
        "task_action_mask": args.task_action_mask,
    }


def ensure_nonfinal_grid(grid_dir: Path, *, context: str) -> None:
    config_path = grid_dir / "config.json"
    if not config_path.exists():
        raise BaselineError(f"{context} grid is missing config.json: {grid_dir}")
    config = load_json(config_path)
    if str(config.get("purpose", "")).lower() == "final":
        raise BaselineError(f"{context} grid has purpose=final and cannot be used for policy training/selection: {grid_dir}")


def resolve_grid_dirs(primary: Path | None, many: str | None) -> list[Path]:
    if many:
        return [Path(part) for part in parse_csv(many)]
    if primary is None:
        raise BaselineError("Either the singular grid dir or comma-separated grid dirs must be provided.")
    return [primary]


def load_grid_rows(grid_dirs: list[Path], *, context: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for grid_dir in grid_dirs:
        ensure_nonfinal_grid(grid_dir, context=context)
        path = grid_dir / "per_sample_action_results.jsonl"
        if not path.exists():
            raise BaselineError(f"{context} grid is missing per_sample_action_results.jsonl: {grid_dir}")
        rows.extend(load_jsonl(path))
    return rows


def policy_sample_key(row: dict[str, Any]) -> str:
    if row.get("policy_sample_id"):
        return str(row["policy_sample_id"])
    dataset = str(row.get("dataset") or row.get("task_family") or "unknown")
    sample_id = str(row.get("sample_id") or "")
    category = str(row.get("category") or "")
    source_row = "" if row.get("source_row") is None else str(row.get("source_row"))
    return f"{dataset}:{category}:{sample_id}:{source_row}"


def group_policy_rows_by_sample(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        sample_id = policy_sample_key(row)
        action = str(row.get("action"))
        if not sample_id or not action:
            continue
        grouped.setdefault(sample_id, {})[action] = row
    return grouped


def available_actions(grouped: dict[str, dict[str, dict[str, Any]]]) -> set[str]:
    out: set[str] = set()
    for rows_by_action in grouped.values():
        out.update(rows_by_action)
    return out


def resolve_candidate_actions(args: argparse.Namespace, train_grouped: dict[str, dict[str, dict[str, Any]]], dev_grouped: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    explicit = parse_csv(args.candidate_actions)
    if explicit:
        return explicit
    if args.candidate_action_set:
        return list(ACTION_SETS[str(args.candidate_action_set)])
    train_actions = available_actions(train_grouped)
    dev_actions = available_actions(dev_grouped)
    candidates = sorted(train_actions & dev_actions)
    if not candidates:
        raise BaselineError("Train/dev grids have no action intersection.")
    return candidates


def feature_row(rows_by_action: dict[str, dict[str, Any]], feature_action: str, fallback_action: str) -> dict[str, Any] | None:
    return rows_by_action.get(feature_action) or rows_by_action.get(fallback_action)


def build_dataset(
    grouped: dict[str, dict[str, dict[str, Any]]],
    *,
    candidate_actions: list[str],
    feature_action_name: str,
    fallback_feature_action: str,
    reward_run: RewardRun,
    action_constraints: dict[str, Any],
) -> dict[str, Any]:
    sample_ids: list[str] = []
    features: list[list[float]] = []
    feature_rows: list[dict[str, Any]] = []
    task_families: list[str] = []
    labels: list[int] = []
    rewards: list[list[float]] = []
    rows: list[dict[str, dict[str, Any]]] = []
    for sample_id in sorted(grouped):
        rows_by_action = grouped[sample_id]
        frow = feature_row(rows_by_action, feature_action_name, fallback_feature_action)
        if frow is None:
            continue
        allowed_indices = allowed_action_indices(candidate_actions, frow, action_constraints)
        if not allowed_indices:
            continue
        if any(candidate_actions[idx] not in rows_by_action for idx in allowed_indices):
            continue
        reward_values = [
            custom_reward(rows_by_action[action], reward_run)
            if idx in allowed_indices and action in rows_by_action
            else MASKED_ACTION_REWARD
            for idx, action in enumerate(candidate_actions)
        ]
        best_idx = max(
            allowed_indices,
            key=lambda idx: (
                reward_values[idx],
                row_task_quality(rows_by_action[candidate_actions[idx]]),
                -safe_float(rows_by_action[candidate_actions[idx]].get("latency_s")),
                candidate_actions[idx],
            ),
        )
        sample_ids.append(sample_id)
        features.append(state_feature_values(frow, include_action_features=False))
        feature_rows.append(frow)
        task_families.append(row_task_family(frow))
        labels.append(int(best_idx))
        rewards.append(reward_values)
        rows.append({action: rows_by_action[action] for action in candidate_actions if action in rows_by_action})
    if not sample_ids:
        raise BaselineError("No complete samples found for the requested candidate actions and feature action.")
    return {
        "sample_ids": sample_ids,
        "features": features,
        "feature_rows": feature_rows,
        "task_families": task_families,
        "labels": labels,
        "rewards": rewards,
        "rows": rows,
    }


def standardize(train_x: list[list[float]], *others: list[list[float]]) -> tuple[list[list[float]], list[list[list[float]]], list[float], list[float]]:
    width = len(train_x[0])
    means = [sum(row[i] for row in train_x) / len(train_x) for i in range(width)]
    stds: list[float] = []
    for i, mean in enumerate(means):
        var = sum((row[i] - mean) ** 2 for row in train_x) / len(train_x)
        stds.append(max(math.sqrt(var), 1e-6))

    def apply(rows: list[list[float]]) -> list[list[float]]:
        return [[(row[i] - means[i]) / stds[i] for i in range(width)] for row in rows]

    return apply(train_x), [apply(rows) for rows in others], means, stds


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def mlp(in_dim: int, out_dim: int, hidden_dim: int):
    return policy_mlp(in_dim, out_dim, hidden_dim)


def train_classifier(
    train_x: list[list[float]],
    train_y: list[int],
    *,
    class_count: int,
    sample_weights: list[float] | None,
    args: argparse.Namespace,
):
    import torch

    model = mlp(len(train_x[0]), class_count, args.hidden_dim).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x = torch.tensor(train_x, dtype=torch.float32, device=args.device)
    y = torch.tensor(train_y, dtype=torch.long, device=args.device)
    weights = torch.tensor(sample_weights, dtype=torch.float32, device=args.device) if sample_weights is not None else None
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y, reduction="none")
        if weights is not None:
            loss = loss * weights
        loss.mean().backward()
        opt.step()
    return model


def train_q_model(
    train_x: list[list[float]],
    train_rewards: list[list[float]],
    train_rows: list[dict[str, dict[str, Any]]],
    candidate_actions: list[str],
    *,
    sample_weights: list[float] | None,
    args: argparse.Namespace,
    cql_alpha: float = 0.0,
):
    import torch

    sample_rows: list[list[list[float]]] = []
    for features, reward_values, rows_by_action in zip(train_x, train_rewards, train_rows):
        action_rows: list[list[float]] = []
        for action in candidate_actions:
            action_rows.append(features + action_feature_values(action))
        sample_rows.append(action_rows)
    model = mlp(len(sample_rows[0][0]), 1, args.hidden_dim).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x = torch.tensor(sample_rows, dtype=torch.float32, device=args.device)
    y = torch.tensor(train_rewards, dtype=torch.float32, device=args.device)
    weights = torch.tensor(sample_weights, dtype=torch.float32, device=args.device) if sample_weights is not None else None
    sample_count, action_count, width = x.shape
    x_flat = x.reshape(sample_count * action_count, width)
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        pred = model(x_flat).reshape(sample_count, action_count)
        loss_by_sample = torch.nn.functional.mse_loss(pred, y, reduction="none").mean(dim=1)
        loss = (loss_by_sample * weights).mean() if weights is not None else loss_by_sample.mean()
        if cql_alpha > 0.0:
            conservative = torch.logsumexp(pred, dim=1) - pred.mean(dim=1)
            conservative_loss = (conservative * weights).mean() if weights is not None else conservative.mean()
            loss = loss + float(cql_alpha) * conservative_loss
        loss.backward()
        opt.step()
    return model


def train_bandit_pg_model(
    train_x: list[list[float]],
    train_rewards: list[list[float]],
    *,
    action_count: int,
    args: argparse.Namespace,
):
    import torch

    model = mlp(len(train_x[0]), action_count, args.hidden_dim).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x = torch.tensor(train_x, dtype=torch.float32, device=args.device)
    rewards = torch.tensor(train_rewards, dtype=torch.float32, device=args.device)
    baseline = rewards.mean(dim=1, keepdim=True)
    uniform_log_prob = -math.log(action_count)
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        advantage = rewards - baseline
        expected_advantage = (probs * advantage).sum(dim=1).mean()
        kl_to_uniform = (probs * (log_probs - uniform_log_prob)).sum(dim=1).mean()
        loss = -expected_advantage + float(args.bandit_kl_lambda) * kl_to_uniform
        loss.backward()
        opt.step()
    return model


def train_pairwise_ranker(
    train_x: list[list[float]],
    train_rewards: list[list[float]],
    *,
    action_count: int,
    args: argparse.Namespace,
):
    import torch

    model = mlp(len(train_x[0]), action_count, args.hidden_dim).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x = torch.tensor(train_x, dtype=torch.float32, device=args.device)
    rewards = torch.tensor(train_rewards, dtype=torch.float32, device=args.device)
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        losses = []
        for i in range(action_count):
            for j in range(action_count):
                delta = rewards[:, i] - rewards[:, j]
                mask = delta > 1e-6
                if bool(mask.any()):
                    losses.append(torch.relu(0.05 - (logits[:, i] - logits[:, j]))[mask] * delta[mask])
        if losses:
            loss = torch.cat(losses).mean()
        else:
            loss = torch.tensor(0.0, device=args.device, requires_grad=True)
        loss.backward()
        opt.step()
    return model


def predict_classifier(
    model: Any,
    x_rows: list[list[float]],
    feature_rows: list[dict[str, Any]],
    candidate_actions: list[str],
    action_constraints: dict[str, Any],
) -> list[int]:
    import torch

    preds: list[int] = []
    with torch.inference_mode():
        logits = model(torch.tensor(x_rows, dtype=torch.float32, device=next(model.parameters()).device))
        for scores_tensor, feature_row in zip(logits.detach().cpu(), feature_rows):
            scores = [float(value) for value in scores_tensor.tolist()]
            decision = policy_constraint_decision(scores, candidate_actions, feature_row, action_constraints)
            preds.append(int(decision.selected_index))
    return preds


def predict_q(
    model: Any,
    x_rows: list[list[float]],
    rows: list[dict[str, dict[str, Any]]],
    feature_rows: list[dict[str, Any]],
    candidate_actions: list[str],
    action_constraints: dict[str, Any],
) -> list[int]:
    import torch

    preds: list[int] = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for features, rows_by_action, feature_row in zip(x_rows, rows, feature_rows):
            candidates = [features + action_feature_values(action) for action in candidate_actions]
            scores = model(torch.tensor(candidates, dtype=torch.float32, device=device)).squeeze(1)
            decision = policy_constraint_decision(
                [float(value) for value in scores.detach().cpu().tolist()],
                candidate_actions,
                feature_row,
                action_constraints,
            )
            preds.append(int(decision.selected_index))
    return preds


def save_policy_checkpoint(
    path: Path,
    *,
    model: Any,
    policy_name: str,
    policy_type: str,
    reward_mode: str,
    candidate_actions: list[str],
    feature_names: list[str],
    means: list[float],
    stds: list[float],
    input_dim: int,
    output_dim: int,
    args: argparse.Namespace,
    reward_config: RewardRun,
    action_constraints: dict[str, Any],
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_name": policy_name,
            "policy_type": policy_type,
            "reward_mode": reward_mode,
            "candidate_actions": candidate_actions,
            "feature_action": args.feature_action,
            "fallback_feature_action": args.fallback_feature_action,
            "baseline_action": args.baseline_action,
            "reference_action": args.reference_action,
            "reward_config": asdict(reward_config),
            "action_constraints": action_constraints,
            "feature_names": feature_names,
            "action_feature_names": list(ACTION_NUMERIC_FEATURES),
            "normalization": {"means": means, "stds": stds},
            "hidden_dim": args.hidden_dim,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "state_dict": model.state_dict(),
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        path,
    )


def _summary_from_predictions(dataset: dict[str, Any], candidate_actions: list[str], pred_indices: list[int], reward_mode: str) -> dict[str, Any]:
    chosen: dict[str, dict[str, Any]] = {}
    predictions: list[dict[str, Any]] = []
    for sample_id, rows_by_action, label, pred_idx, reward_values in zip(
        dataset["sample_ids"], dataset["rows"], dataset["labels"], pred_indices, dataset["rewards"]
    ):
        action = candidate_actions[int(pred_idx)]
        chosen[sample_id] = rows_by_action[action]
        best_reward = float(max(reward_values))
        pred_reward = float(reward_values[int(pred_idx)])
        predictions.append(
            {
                "sample_id": sample_id,
                "oracle_action": candidate_actions[int(label)],
                "predicted_action": action,
                "correct_oracle_label": int(pred_idx) == int(label),
                "predicted_reward": pred_reward,
                "oracle_reward": best_reward,
                "regret": best_reward - pred_reward,
            }
        )
    summary = evaluate_rule(chosen)
    summary["reward_mode"] = reward_mode
    summary["oracle_label_accuracy"] = (
        sum(1 for row in predictions if row["correct_oracle_label"]) / len(predictions) if predictions else None
    )
    summary["mean_reward_regret"] = sum(row["regret"] for row in predictions) / len(predictions) if predictions else None
    summary["predictions"] = predictions
    return summary


def fixed_action_summary(dataset: dict[str, Any], action: str, reward_mode: str) -> dict[str, Any] | None:
    if not all(action in rows_by_action for rows_by_action in dataset["rows"]):
        return None
    chosen = {sample_id: rows_by_action[action] for sample_id, rows_by_action in zip(dataset["sample_ids"], dataset["rows"])}
    summary = evaluate_rule(chosen)
    summary["reward_mode"] = reward_mode
    summary["action"] = action
    return summary


def oracle_summary(dataset: dict[str, Any], candidate_actions: list[str], reward_mode: str) -> dict[str, Any]:
    return _summary_from_predictions(dataset, candidate_actions, dataset["labels"], reward_mode)


def reward_gap_weights(rewards: list[list[float]], labels: list[int]) -> list[float]:
    weights: list[float] = []
    for reward_values, label in zip(rewards, labels):
        best = float(reward_values[label])
        others = [float(value) for idx, value in enumerate(reward_values) if idx != label]
        gap = best - max(others) if others else 0.0
        weights.append(max(0.25, 1.0 + 5.0 * gap))
    return weights


def task_balance_weights(task_families: list[str]) -> list[float]:
    counts: dict[str, int] = {}
    for family in task_families:
        counts[family] = counts.get(family, 0) + 1
    if len(counts) <= 1:
        return [1.0 for _ in task_families]
    total = len(task_families)
    task_count = len(counts)
    return [total / (task_count * counts[family]) for family in task_families]


def multiply_weights(left: list[float] | None, right: list[float] | None) -> list[float] | None:
    if left is None:
        return right
    if right is None:
        return left
    return [float(a) * float(b) for a, b in zip(left, right)]


def count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def run_reward_mode(
    args: argparse.Namespace,
    *,
    reward_run: RewardRun,
    candidate_actions: list[str],
    train_grouped: dict[str, dict[str, dict[str, Any]]],
    dev_grouped: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    set_seed(args.seed)
    train_data = build_dataset(
        train_grouped,
        candidate_actions=candidate_actions,
        feature_action_name=args.feature_action,
        fallback_feature_action=args.fallback_feature_action,
        reward_run=reward_run,
        action_constraints=action_constraints_from_args(args),
    )
    dev_data = build_dataset(
        dev_grouped,
        candidate_actions=candidate_actions,
        feature_action_name=args.feature_action,
        fallback_feature_action=args.fallback_feature_action,
        reward_run=reward_run,
        action_constraints=action_constraints_from_args(args),
    )
    train_x, other_x, means, stds = standardize(train_data["features"], dev_data["features"])
    dev_x = other_x[0]
    task_weights = task_balance_weights(train_data["task_families"]) if args.task_balanced else None

    class_count = len(candidate_actions)
    requested_policy_types = set(parse_csv(args.policy_types))
    cql_alphas = parse_float_csv(args.cql_alphas) or [0.0]
    action_constraints = action_constraints_from_args(args)
    trained_models: dict[str, tuple[Any, str, int, int]] = {}

    if "supervised_mlp" in requested_policy_types:
        supervised = train_classifier(
            train_x,
            train_data["labels"],
            class_count=class_count,
            sample_weights=task_weights,
            args=args,
        )
        trained_models["supervised_mlp"] = (supervised, "classifier", len(train_x[0]), class_count)
    if "reward_weighted_mlp" in requested_policy_types:
        weighted = train_classifier(
            train_x,
            train_data["labels"],
            class_count=class_count,
            sample_weights=multiply_weights(task_weights, reward_gap_weights(train_data["rewards"], train_data["labels"])),
            args=args,
        )
        trained_models["reward_weighted_mlp"] = (weighted, "classifier", len(train_x[0]), class_count)
    if "q_mlp" in requested_policy_types:
        for cql_alpha in cql_alphas:
            q_model = train_q_model(
                train_x,
                train_data["rewards"],
                train_data["rows"],
                candidate_actions,
                sample_weights=task_weights,
                args=args,
                cql_alpha=cql_alpha,
            )
            suffix = "" if cql_alpha == 0.0 else f"_cql_a{float_slug(cql_alpha)}"
            trained_models[f"q_mlp{suffix}"] = (
                q_model,
                "q",
                len(train_x[0]) + len(action_feature_values(candidate_actions[0])),
                1,
            )
    if "pairwise_rank_mlp" in requested_policy_types:
        ranker = train_pairwise_ranker(train_x, train_data["rewards"], action_count=class_count, args=args)
        trained_models["pairwise_rank_mlp"] = (ranker, "classifier", len(train_x[0]), class_count)
    if args.include_bandit_pg or "bandit_pg_mlp" in requested_policy_types:
        bandit = train_bandit_pg_model(train_x, train_data["rewards"], action_count=class_count, args=args)
        trained_models["bandit_pg_mlp"] = (bandit, "classifier", len(train_x[0]), class_count)
    if not trained_models:
        raise BaselineError("No policy models requested. Check --policy-types.")

    summaries: dict[str, Any] = {
        "feature_action": args.feature_action,
        "fallback_feature_action": args.fallback_feature_action,
        "feature_names": state_feature_names(include_action_features=False),
        "action_feature_names": list(ACTION_NUMERIC_FEATURES),
        "candidate_actions": candidate_actions,
        "candidate_action_set": args.candidate_action_set,
        "reward_config": asdict(reward_run),
        "action_constraints": action_constraints,
        "train_samples": len(train_data["sample_ids"]),
        "dev_samples": len(dev_data["sample_ids"]),
        "task_balanced": bool(args.task_balanced),
        "train_task_counts": count_values(train_data["task_families"]),
        "dev_task_counts": count_values(dev_data["task_families"]),
        "normalization": {"means": means, "stds": stds},
        "train_oracle": oracle_summary(train_data, candidate_actions, reward_run.name),
        "dev_oracle": oracle_summary(dev_data, candidate_actions, reward_run.name),
        "dev_fixed_primary": fixed_action_summary(dev_data, args.baseline_action, reward_run.name),
        "dev_fixed_reference": fixed_action_summary(dev_data, args.reference_action, reward_run.name),
    }

    feature_names = state_feature_names(include_action_features=False)
    for policy_name, (model, policy_type, input_dim, output_dim) in trained_models.items():
        checkpoint_path = args.output_dir / f"model_{reward_run.name}_{policy_name}.pt"
        save_policy_checkpoint(
            checkpoint_path,
            model=model,
            policy_name=policy_name,
            policy_type=policy_type,
            reward_mode=reward_run.name,
            candidate_actions=candidate_actions,
            feature_names=feature_names,
            means=means,
            stds=stds,
            input_dim=input_dim,
            output_dim=output_dim,
            args=args,
            reward_config=reward_run,
            action_constraints=action_constraints,
        )
        summaries[f"{policy_name}_checkpoint"] = str(checkpoint_path)

    policy_predictions: dict[str, list[int]] = {}
    for name, (model, policy_type, _input_dim, _output_dim) in trained_models.items():
        if policy_type == "q":
            policy_predictions[name] = predict_q(
                model,
                dev_x,
                dev_data["rows"],
                dev_data["feature_rows"],
                candidate_actions,
                action_constraints,
            )
        else:
            policy_predictions[name] = predict_classifier(
                model,
                dev_x,
                dev_data["feature_rows"],
                candidate_actions,
                action_constraints,
            )
    for name, pred_indices in policy_predictions.items():
        summary = _summary_from_predictions(dev_data, candidate_actions, pred_indices, reward_run.name)
        predictions = summary.pop("predictions")
        summaries[f"dev_{name}"] = summary
        write_jsonl(args.output_dir / f"predictions_{reward_run.name}_{name}.jsonl", predictions)
    return summaries


def run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_grid_dirs = resolve_grid_dirs(args.train_grid_dir, args.train_grid_dirs)
    dev_grid_dirs = resolve_grid_dirs(args.dev_grid_dir, args.dev_grid_dirs)
    train_rows = load_grid_rows(train_grid_dirs, context="train")
    dev_rows = load_grid_rows(dev_grid_dirs, context="dev")
    train_grouped = group_policy_rows_by_sample(train_rows)
    dev_grouped = group_policy_rows_by_sample(dev_rows)
    candidate_actions = resolve_candidate_actions(args, train_grouped, dev_grouped)
    reward_runs = build_reward_runs(args)

    report: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "train_grid_dir": str(args.train_grid_dir),
        "dev_grid_dir": str(args.dev_grid_dir),
        "train_grid_dirs": [str(path) for path in train_grid_dirs],
        "dev_grid_dirs": [str(path) for path in dev_grid_dirs],
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "candidate_actions": candidate_actions,
        "candidate_action_set": args.candidate_action_set,
        "baseline_action": args.baseline_action,
        "reference_action": args.reference_action,
        "policy_types": parse_csv(args.policy_types),
        "cql_alphas": parse_float_csv(args.cql_alphas) or [0.0],
        "task_balanced": bool(args.task_balanced),
        "action_constraints": action_constraints_from_args(args),
        "reward_modes": {},
    }
    for reward_run in reward_runs:
        report["reward_modes"][reward_run.name] = run_reward_mode(
            args,
            reward_run=reward_run,
            candidate_actions=candidate_actions,
            train_grouped=train_grouped,
            dev_grouped=dev_grouped,
        )
    write_json(args.output_dir / "policy_summary.json", report)
    print(f"Saved UG-Prune policy-learning summary to {args.output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (BaselineError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
