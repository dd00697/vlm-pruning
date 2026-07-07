#!/usr/bin/env python
"""Select a frozen v6 rescue-policy checkpoint from a policy summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm_pruning.controllers import parse_action_config  # noqa: E402


LANE_DEFAULTS = {
    "rescue68": {
        "token_target": 68.0,
        "max_b288_rate": 0.03,
        "max_rescue_rate": 0.08,
        "min_textvqa_gain": 0.007,
        "max_refcoco_acc_drop": 0.010,
        "max_refcoco_miou_drop": 0.005,
        "max_pope_f1_drop": 0.003,
    },
    "rescue96": {
        "token_target": 96.0,
        "max_b288_rate": 0.10,
        "max_rescue_rate": 0.20,
        "min_textvqa_gain": 0.015,
        "max_refcoco_acc_drop": 0.010,
        "max_refcoco_miou_drop": 0.002,
        "max_pope_f1_drop": 0.003,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-summary", required=True, type=Path)
    parser.add_argument("--lane", required=True, choices=sorted(LANE_DEFAULTS))
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--token-target", type=float, default=None)
    parser.add_argument("--max-b288-rate", type=float, default=None)
    parser.add_argument("--max-rescue-rate", type=float, default=None)
    parser.add_argument("--min-textvqa-gain", type=float, default=None)
    parser.add_argument("--max-refcoco-acc-drop", type=float, default=None)
    parser.add_argument("--max-refcoco-miou-drop", type=float, default=None)
    parser.add_argument("--max-pope-f1-drop", type=float, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    if value is None:
        return float(default)
    return float(value)


def task_metrics(row: dict[str, Any], task: str) -> dict[str, Any]:
    by_task = row.get("by_task")
    if not isinstance(by_task, dict):
        return {}
    value = by_task.get(task)
    return value if isinstance(value, dict) else {}


def rate_for_actions(row: dict[str, Any], predicate) -> float:
    counts = row.get("action_counts")
    if not isinstance(counts, dict):
        return 0.0
    total = sum(int(count) for count in counts.values())
    if total <= 0:
        return 0.0
    selected = 0
    for action, count in counts.items():
        try:
            parsed = parse_action_config(str(action))
        except ValueError:
            continue
        if predicate(parsed):
            selected += int(count)
    return selected / total


def checkpoint_path(output_dir: Path, reward_name: str, eval_name: str) -> Path:
    suffix = eval_name.removeprefix("dev_")
    return output_dir / f"model_{reward_name}_{suffix}.pt"


def predictions_path(output_dir: Path, reward_name: str, eval_name: str) -> Path:
    suffix = eval_name.removeprefix("dev_")
    return output_dir / f"predictions_{reward_name}_{suffix}.jsonl"


def is_policy_eval(name: str) -> bool:
    return name in {
        "dev_q_mlp",
        "dev_q_mlp_cql_a0p05",
        "dev_q_mlp_cql_a0p1",
        "dev_q_mlp_cql_a0p2",
    }


def evaluate_candidate(
    *,
    row: dict[str, Any],
    fixed_primary: dict[str, Any],
    fixed_reference: dict[str, Any],
    token_target: float,
    max_b288_rate: float,
    max_rescue_rate: float,
    min_textvqa_gain: float,
    max_refcoco_acc_drop: float,
    max_refcoco_miou_drop: float,
    max_pope_f1_drop: float,
) -> tuple[bool, list[str], dict[str, float]]:
    b288_rate = rate_for_actions(row, lambda action: action.target_visual_tokens >= 288)
    rescue_rate = rate_for_actions(row, lambda action: action.target_visual_tokens >= 144)
    tokens = metric(row, "tokens/actual_visual_tokens")
    latency = metric(row, "efficiency/mean_latency_s")

    ref = task_metrics(row, "grounding")
    text = task_metrics(row, "open_vqa")
    pope = task_metrics(row, "yes_no_vqa")
    ref_base = task_metrics(fixed_reference, "grounding")
    text_base = task_metrics(fixed_reference, "open_vqa")
    pope_base = task_metrics(fixed_primary, "yes_no_vqa")

    text_gain = metric(text, "quality/task_accuracy") - metric(text_base, "quality/task_accuracy")
    ref_acc_drop = metric(ref_base, "quality/acc_at_0_5") - metric(ref, "quality/acc_at_0_5")
    ref_miou_drop = metric(ref_base, "quality/mean_iou") - metric(ref, "quality/mean_iou")
    pope_f1_drop = metric(pope_base, "quality/f1") - metric(pope, "quality/f1")

    diagnostics = {
        "tokens": tokens,
        "latency": latency,
        "b288_rate": b288_rate,
        "rescue_rate": rescue_rate,
        "textvqa_gain": text_gain,
        "refcoco_acc_drop": ref_acc_drop,
        "refcoco_miou_drop": ref_miou_drop,
        "pope_f1_drop": pope_f1_drop,
        "task_score": metric(row, "quality/task_score"),
        "task_accuracy": metric(row, "quality/task_accuracy"),
        "textvqa_accuracy": metric(text, "quality/task_accuracy"),
        "refcoco_acc_at_0_5": metric(ref, "quality/acc_at_0_5"),
        "refcoco_miou": metric(ref, "quality/mean_iou"),
        "pope_f1": metric(pope, "quality/f1"),
        "invalid_outputs": metric(row, "samples/invalid_outputs"),
        "exact_budget_rate": metric(row, "selection/exact_budget_rate", 1.0),
        "duplicate_count_mean": metric(row, "selection/duplicate_count_mean"),
        "out_of_range_count_mean": metric(row, "selection/out_of_range_count_mean"),
    }

    failures: list[str] = []
    if tokens > token_target:
        failures.append("token_target")
    if b288_rate > max_b288_rate:
        failures.append("b288_rate")
    if rescue_rate > max_rescue_rate:
        failures.append("rescue_rate")
    if text_gain < min_textvqa_gain:
        failures.append("textvqa_gain")
    if ref_acc_drop > max_refcoco_acc_drop:
        failures.append("refcoco_acc_drop")
    if ref_miou_drop > max_refcoco_miou_drop:
        failures.append("refcoco_miou_drop")
    if pope_f1_drop > max_pope_f1_drop:
        failures.append("pope_f1_drop")
    if diagnostics["invalid_outputs"] > 0:
        failures.append("invalid_outputs")
    if diagnostics["exact_budget_rate"] < 1.0:
        failures.append("exact_budget_rate")
    if diagnostics["duplicate_count_mean"] > 0:
        failures.append("duplicate_count_mean")
    if diagnostics["out_of_range_count_mean"] > 0:
        failures.append("out_of_range_count_mean")
    return not failures, failures, diagnostics


def main() -> None:
    args = parse_args()
    defaults = LANE_DEFAULTS[args.lane]
    token_target = args.token_target if args.token_target is not None else defaults["token_target"]
    max_b288_rate = args.max_b288_rate if args.max_b288_rate is not None else defaults["max_b288_rate"]
    max_rescue_rate = args.max_rescue_rate if args.max_rescue_rate is not None else defaults["max_rescue_rate"]
    min_textvqa_gain = args.min_textvqa_gain if args.min_textvqa_gain is not None else defaults["min_textvqa_gain"]
    max_refcoco_acc_drop = (
        args.max_refcoco_acc_drop if args.max_refcoco_acc_drop is not None else defaults["max_refcoco_acc_drop"]
    )
    max_refcoco_miou_drop = (
        args.max_refcoco_miou_drop if args.max_refcoco_miou_drop is not None else defaults["max_refcoco_miou_drop"]
    )
    max_pope_f1_drop = args.max_pope_f1_drop if args.max_pope_f1_drop is not None else defaults["max_pope_f1_drop"]

    summary = load_json(args.policy_summary)
    output_dir = Path(str(summary.get("output_dir") or args.policy_summary.parent))
    reward_modes = summary.get("reward_modes")
    if not isinstance(reward_modes, dict):
        raise SystemExit(f"No reward_modes found in {args.policy_summary}")

    evaluated: list[dict[str, Any]] = []
    for reward_name, payload in reward_modes.items():
        if not isinstance(payload, dict):
            continue
        fixed_reference = payload.get("dev_fixed_reference")
        fixed_primary = payload.get("dev_fixed_primary")
        if not isinstance(fixed_reference, dict) or not isinstance(fixed_primary, dict):
            continue
        for eval_name, row in payload.items():
            if not is_policy_eval(eval_name) or not isinstance(row, dict):
                continue
            passed, failures, diagnostics = evaluate_candidate(
                row=row,
                fixed_primary=fixed_primary,
                fixed_reference=fixed_reference,
                token_target=token_target,
                max_b288_rate=max_b288_rate,
                max_rescue_rate=max_rescue_rate,
                min_textvqa_gain=min_textvqa_gain,
                max_refcoco_acc_drop=max_refcoco_acc_drop,
                max_refcoco_miou_drop=max_refcoco_miou_drop,
                max_pope_f1_drop=max_pope_f1_drop,
            )
            ckpt = checkpoint_path(output_dir, str(reward_name), eval_name)
            pred_path = predictions_path(output_dir, str(reward_name), eval_name)
            evaluated.append(
                {
                    "reward_mode": reward_name,
                    "eval_name": eval_name,
                    "passed": passed,
                    "failures": failures,
                    "checkpoint_path": str(ckpt),
                    "checkpoint_exists": ckpt.exists(),
                    "predictions_path": str(pred_path),
                    "predictions_exist": pred_path.exists(),
                    "action_counts": row.get("action_counts", {}),
                    "diagnostics": diagnostics,
                }
            )

    def rank_key(item: dict[str, Any]) -> tuple[float, ...]:
        diag = item["diagnostics"]
        return (
            diag["textvqa_accuracy"],
            diag["task_score"],
            diag["refcoco_acc_at_0_5"],
            diag["pope_f1"],
            -diag["tokens"],
            -diag["b288_rate"],
            -diag["latency"],
        )

    passing = [item for item in evaluated if item["passed"] and item["checkpoint_exists"]]
    selected = max(passing, key=rank_key) if passing else None
    fallback = max(evaluated, key=rank_key) if evaluated else None

    result = {
        "lane": args.lane,
        "policy_summary": str(args.policy_summary),
        "gates": {
            "token_target": token_target,
            "max_b288_rate": max_b288_rate,
            "max_rescue_rate": max_rescue_rate,
            "min_textvqa_gain": min_textvqa_gain,
            "max_refcoco_acc_drop": max_refcoco_acc_drop,
            "max_refcoco_miou_drop": max_refcoco_miou_drop,
            "max_pope_f1_drop": max_pope_f1_drop,
        },
        "selected": selected,
        "fallback_best": fallback,
        "num_candidates": len(evaluated),
        "num_passing": len(passing),
        "candidates": sorted(evaluated, key=rank_key, reverse=True),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
