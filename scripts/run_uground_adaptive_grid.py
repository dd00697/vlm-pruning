#!/usr/bin/env python
"""Run and aggregate a compact UGround-style UG-Prune action grid."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_baseline import BaselineError  # noqa: E402
from run_official_baseline import DEFAULT_MODEL, load_manifest, safe_slug, utc_now, write_json  # noqa: E402
from vlm_pruning.controllers import (  # noqa: E402
    ACTION_GRID_DEFAULT_BUDGETS,
    ACTION_GRID_DEFAULT_KS,
    ACTION_GRID_DEFAULT_ROLE_PRESETS,
    AdaptiveAction,
    action_grid,
    assert_allowed_nonfinal_rows,
    oracle_utility,
    parse_action_config,
    parse_csv_ints,
    parse_csv_strings,
)


OUTPUT_ROOT = Path("experiments/runs_uground_adaptive_grid")
SINGLE_PASS_OUTPUT_ROOT = Path("experiments/runs_ug_prune_single_pass")
SINGLE_PASS_FIGURES_ROOT = Path("experiments/figures_ug_prune_single_pass")
SINGLE_PASS_TABLES_ROOT = Path("experiments/tables/ug_prune_single_pass")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="refcoco")
    parser.add_argument("--task", default="grounding", choices=["grounding", "vqa"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--grid-name", default=None)
    parser.add_argument("--ks", default=",".join(str(k) for k in ACTION_GRID_DEFAULT_KS))
    parser.add_argument("--budgets", default=",".join(str(b) for b in ACTION_GRID_DEFAULT_BUDGETS))
    parser.add_argument("--role-presets", default=",".join(ACTION_GRID_DEFAULT_ROLE_PRESETS))
    parser.add_argument("--fusion-modes", default="soft_average")
    parser.add_argument("--map-sources", default="query_max")
    parser.add_argument(
        "--selection-strategies",
        default="standard",
        help="Comma-separated UG selection strategies for cartesian grids. Ignored when --actions is set.",
    )
    parser.add_argument(
        "--actions",
        default=None,
        help="Comma-separated exact adaptive action names. Use this for non-cartesian rescue grids.",
    )
    parser.add_argument("--ug-zip-alpha", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--model-path", "--model", dest="model_path", default=DEFAULT_MODEL)
    parser.add_argument("--decode-mode", default="layer_aware_cache", choices=["no_cache_recompute", "layer_aware_cache"])
    parser.add_argument("--diagnostics-mode", default="benchmark", choices=["full", "benchmark"])
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--box-coordinate-mode", default="normalized_0_1", choices=["normalized_0_1", "normalized_0_1000", "absolute_pixels"])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--single-pass-output-root", type=Path, default=SINGLE_PASS_OUTPUT_ROOT)
    parser.add_argument("--single-pass-figures-root", type=Path, default=SINGLE_PASS_FIGURES_ROOT)
    parser.add_argument("--single-pass-tables-root", type=Path, default=SINGLE_PASS_TABLES_ROOT)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--purpose", default="tuning", choices=["tuning", "smoke", "final"])
    parser.add_argument("--allow-final-refcoco500", action="store_true")
    parser.add_argument("--latency-weight", type=float, default=0.05)
    parser.add_argument("--token-weight", type=float, default=0.02)
    parser.add_argument("--parse-failure-penalty", type=float, default=0.25)
    parser.add_argument("--rule-entropy-norm-max", type=float, default=0.965)
    parser.add_argument("--rule-top58-mass-min", type=float, default=0.18)
    parser.add_argument("--rule-peakiness-min", type=float, default=2.0)
    parser.add_argument("--rule-mid-entropy-norm-max", type=float, default=0.985)
    parser.add_argument("--rule-mid-top58-mass-min", type=float, default=0.14)
    parser.add_argument("--rule-mid-peakiness-min", type=float, default=1.5)
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


def grid_run_name(args: argparse.Namespace) -> str:
    if args.grid_name:
        return safe_slug(args.grid_name)
    limit = f"limit{args.limit}_" if args.limit is not None else ""
    return f"{limit}{safe_slug(args.dataset)}_uground_adaptive_{safe_slug(args.purpose)}_{int(time.time())}"


def single_pass_run_name(grid_name: str, action: AdaptiveAction, args: argparse.Namespace) -> str:
    mode = "" if args.decode_mode == "layer_aware_cache" else f"_{safe_slug(args.decode_mode)}"
    diag = "" if args.diagnostics_mode == "benchmark" else f"_{safe_slug(args.diagnostics_mode)}"
    return f"{safe_slug(grid_name)}_{action.name}_seed{args.seed}_{args.precision}{mode}{diag}"


def command_for_action(args: argparse.Namespace, grid_name: str, action: AdaptiveAction) -> tuple[list[str], Path]:
    run_name = single_pass_run_name(grid_name, action, args)
    run_dir = args.single_pass_output_root / run_name
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_ug_prune_single_pass.py"),
        "--dataset",
        args.dataset,
        "--task",
        args.task,
        "--manifest",
        str(args.manifest),
        "--target-visual-tokens",
        str(action.target_visual_tokens),
        "--seed",
        str(args.seed),
        "--precision",
        args.precision,
        "--model-path",
        args.model_path,
        "--run-name",
        run_name,
        "--output-root",
        str(args.single_pass_output_root),
        "--figures-root",
        str(args.single_pass_figures_root),
        "--tables-root",
        str(args.single_pass_tables_root),
        "--prune-after-layer",
        str(action.prune_after_layer),
        "--decode-mode",
        args.decode_mode,
        "--diagnostics-mode",
        args.diagnostics_mode,
        "--ug-map-source",
        action.map_source,
        "--ug-fusion-mode",
        action.fusion_mode,
        "--ug-role-preset",
        action.role_preset,
        "--ug-selection-strategy",
        action.selection_strategy,
        "--ug-zip-alpha",
        str(action.zip_alpha),
        "--box-coordinate-mode",
        args.box_coordinate_mode,
        "--no-wandb",
    ]
    if args.split:
        cmd.extend(["--split", args.split])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.max_new_tokens is not None:
        cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.resume:
        cmd.append("--resume")
    return cmd, run_dir


def action_from_name(name: str, *, default_zip_alpha: float = 0.20) -> AdaptiveAction:
    parsed = parse_action_config(name)
    return AdaptiveAction(
        prune_after_layer=int(parsed.prune_after_layer),
        target_visual_tokens=int(parsed.target_visual_tokens),
        map_source=parsed.map_source,
        fusion_mode=parsed.fusion_mode,
        role_preset=parsed.role_preset,
        selection_strategy=parsed.selection_strategy,
        zip_alpha=float(parsed.zip_alpha if parsed.selection_strategy == "coverage_zip" else default_zip_alpha),
    ).validate()


def actions_from_args(args: argparse.Namespace) -> list[AdaptiveAction]:
    explicit = parse_csv_strings(args.actions, ()) if args.actions else ()
    if explicit:
        actions = [action_from_name(name, default_zip_alpha=float(args.ug_zip_alpha)) for name in explicit]
    else:
        actions = action_grid(
            ks=parse_csv_ints(args.ks, ACTION_GRID_DEFAULT_KS),
            budgets=parse_csv_ints(args.budgets, ACTION_GRID_DEFAULT_BUDGETS),
            role_presets=parse_csv_strings(args.role_presets, ACTION_GRID_DEFAULT_ROLE_PRESETS),
            fusion_modes=parse_csv_strings(args.fusion_modes, ("soft_average",)),
            map_sources=parse_csv_strings(args.map_sources, ("query_max",)),
            selection_strategies=parse_csv_strings(args.selection_strategies, ("standard",)),
            zip_alpha=float(args.ug_zip_alpha),
        )
    names = [action.name for action in actions]
    if len(names) != len(set(names)):
        raise BaselineError("Action list contains duplicate action names.")
    return actions


def result_metric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def task_family(row: dict[str, Any], args: argparse.Namespace | None = None) -> str:
    explicit = str(row.get("task_family") or "").lower()
    if explicit in {"grounding", "open_vqa", "yes_no_vqa"}:
        return explicit
    dataset = str(row.get("dataset") or (args.dataset if args is not None else "") or "").lower()
    task = str(row.get("task") or (args.task if args is not None else "") or "").lower()
    answer_mode = str(row.get("answer_mode") or "").lower()
    if answer_mode == "yes_no" or dataset == "pope":
        return "yes_no_vqa"
    if dataset == "textvqa" or (task == "vqa" and answer_mode != "yes_no"):
        return "open_vqa"
    if task == "grounding" or dataset == "refcoco" or row.get("box_iou") is not None:
        return "grounding"
    return "open_vqa"


def task_quality(row: dict[str, Any]) -> float:
    if task_family(row) == "grounding":
        return result_metric(row, "box_iou")
    return result_metric(row, "score")


def task_accuracy(row: dict[str, Any]) -> float:
    family = task_family(row)
    if family == "grounding":
        return 1.0 if row.get("success_iou_0_5") else 0.0
    return 1.0 if row.get("correct") else 0.0


def yes_no_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    yes_rows = [row for row in rows if str(row.get("answer_mode") or "").lower() == "yes_no" or str(row.get("dataset") or "").lower() == "pope"]
    if not yes_rows:
        return {
            "quality/precision": None,
            "quality/recall": None,
            "quality/f1": None,
            "quality/yes_rate": None,
        }
    tp = sum(1 for row in yes_rows if row.get("prediction") == "yes" and row.get("target") == "yes")
    fp = sum(1 for row in yes_rows if row.get("prediction") == "yes" and row.get("target") == "no")
    fn = sum(1 for row in yes_rows if row.get("prediction") == "no" and row.get("target") == "yes")
    pred_yes = sum(1 for row in yes_rows if row.get("prediction") == "yes")
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and (precision + recall) else None
    return {
        "quality/precision": precision,
        "quality/recall": recall,
        "quality/f1": f1,
        "quality/yes_rate": pred_yes / len(yes_rows) if yes_rows else None,
    }


def summarise_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "samples/evaluated": 0,
            "quality/acc_at_0_5": None,
            "quality/mean_iou": None,
            "efficiency/mean_latency_s": None,
        }
    valid_rows = [row for row in rows if row.get("valid")]
    ious = [result_metric(row, "box_iou") for row in valid_rows]
    summary = {
        "samples/evaluated": len(rows),
        "samples/invalid_outputs": sum(1 for row in rows if row.get("invalid")),
        "quality/task_score": sum(task_quality(row) for row in rows) / len(rows),
        "quality/task_accuracy": sum(task_accuracy(row) for row in rows) / len(rows),
        "quality/acc_at_0_5": sum(1 for row in valid_rows if row.get("success_iou_0_5")) / len(rows),
        "quality/mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "quality/accuracy": sum(task_accuracy(row) for row in rows) / len(rows),
        "quality/parsed_box_rate": sum(1 for row in rows if row.get("raw_parsed_box") is not None or row.get("valid")) / len(rows),
        "efficiency/mean_latency_s": sum(result_metric(row, "latency_s") for row in rows) / len(rows),
        "tokens/actual_visual_tokens": sum(result_metric(row, "actual_llm_visual_tokens") for row in rows) / len(rows),
        "selection/exact_budget_rate": sum(result_metric(row, "selection/exact_budget") for row in rows) / len(rows),
        "selection/duplicate_count_mean": sum(result_metric(row, "selection/duplicate_count") for row in rows) / len(rows),
        "selection/out_of_range_count_mean": sum(result_metric(row, "selection/out_of_range_count") for row in rows) / len(rows),
    }
    summary.update(yes_no_metrics(rows))
    families = sorted({task_family(row) for row in rows})
    if len(families) > 1:
        summary["by_task"] = {
            family: summarise_rows([row for row in rows if task_family(row) == family])
            for family in families
        }
    return summary


def action_summary(action: AdaptiveAction, run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    per_sample_path = run_dir / "per_sample_results.jsonl"
    config_path = run_dir / "config.json"
    if not metrics_path.exists() or not per_sample_path.exists():
        return {"action": action.as_dict(), "run_dir": str(run_dir), "complete": False}
    metrics = load_json(metrics_path)
    rows = load_jsonl(per_sample_path)
    return {
        "action": action.as_dict(),
        "run_dir": str(run_dir),
        "complete": True,
        "metrics": metrics,
        "config": load_json(config_path) if config_path.exists() else {},
        "sample_count": len(rows),
    }


def compact_action_row(action: AdaptiveAction, row: dict[str, Any], latency_reference: float, args: argparse.Namespace) -> dict[str, Any]:
    latency = result_metric(row, "latency_s")
    original_tokens = result_metric(row, "original_visual_tokens", 576.0)
    actual_tokens = result_metric(row, "actual_llm_visual_tokens", float(action.target_visual_tokens))
    normalized_latency = latency / latency_reference if latency_reference > 0 else 0.0
    valid = bool(row.get("valid")) if "valid" in row else not bool(row.get("invalid"))
    parse_failed = bool(row.get("invalid") or not valid)
    quality = task_quality(row)
    category = row.get("category") or ((row.get("metadata") or {}).get("category") if isinstance(row.get("metadata"), dict) else None)
    source_row = (row.get("metadata") or {}).get("source_row") if isinstance(row.get("metadata"), dict) else None
    if source_row is None:
        source_row = row.get("source_row")
    policy_sample_id = row.get("policy_sample_id") or f"{row.get('dataset') or args.dataset}:{category or ''}:{row.get('sample_id')}:{'' if source_row is None else source_row}"
    utility = oracle_utility(
        iou=quality,
        normalized_latency=normalized_latency,
        token_fraction=actual_tokens / original_tokens if original_tokens else 1.0,
        parse_failed=parse_failed,
        latency_weight=args.latency_weight,
        token_weight=args.token_weight,
        parse_failure_penalty=args.parse_failure_penalty,
    )
    feature_keys = [
        "ug/map_entropy_norm",
        "ug/map_peakiness",
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
        "ug/map_downsample_6x6",
        "ug/map_top48_mask_6x6",
        "ug/map_top58_mask_6x6",
        "ug/map_top72_mask_6x6",
        "ug/map_top96_mask_6x6",
        "ug/map_selected_mask_6x6",
    ]
    return {
        "sample_id": str(row.get("sample_id")),
        "policy_sample_id": policy_sample_id,
        "dataset": row.get("dataset") or args.dataset,
        "task": args.task,
        "task_family": task_family(row, args),
        "category": category,
        "source_row": source_row,
        "query": row.get("query") or row.get("question"),
        "question": row.get("question"),
        "action": action.name,
        "action_config": action.as_dict(),
        "box_iou": result_metric(row, "box_iou"),
        "success_iou_0_5": bool(row.get("success_iou_0_5")),
        "score": result_metric(row, "score"),
        "correct": bool(row.get("correct")),
        "policy_quality": quality,
        "policy_accuracy": task_accuracy(row),
        "answer_mode": row.get("answer_mode"),
        "prediction": row.get("prediction"),
        "target": row.get("target"),
        "valid": valid,
        "invalid": bool(row.get("invalid")),
        "invalid_reason": row.get("invalid_reason") or row.get("parse_failure_reason"),
        "latency_s": latency,
        "normalized_latency": normalized_latency,
        "actual_llm_visual_tokens": actual_tokens,
        "token_fraction": actual_tokens / original_tokens if original_tokens else None,
        "utility": utility,
        "selection/exact_budget": result_metric(row, "selection/exact_budget"),
        "selection/duplicate_count": result_metric(row, "selection/duplicate_count"),
        "selection/out_of_range_count": result_metric(row, "selection/out_of_range_count"),
        **{key: row.get(key) for key in feature_keys if key in row},
    }


def reference_latency(action_rows: dict[str, list[dict[str, Any]]]) -> float:
    preferred = "k12_b58_default_soft_average_query_max"
    candidates = action_rows.get(preferred)
    if candidates:
        latencies = [result_metric(row, "latency_s") for row in candidates]
        return sum(latencies) / len(latencies) if latencies else 1.0
    for action_name, rows in sorted(action_rows.items()):
        if action_name.startswith("k12_b58_") and rows:
            latencies = [result_metric(row, "latency_s") for row in rows]
            return sum(latencies) / len(latencies) if latencies else 1.0
    all_latencies = [result_metric(row, "latency_s") for rows in action_rows.values() for row in rows]
    return sum(all_latencies) / len(all_latencies) if all_latencies else 1.0


def evaluate_rule(chosen_by_sample: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(chosen_by_sample.values())
    summary = summarise_rows(rows)
    summary["utility/mean"] = sum(result_metric(row, "utility") for row in rows) / len(rows) if rows else None
    counts: dict[str, int] = {}
    for row in rows:
        action = str(row.get("action"))
        counts[action] = counts.get(action, 0) + 1
    summary["action_counts"] = counts
    return summary


def confidence_passes(row: dict[str, Any], *, entropy: float, top58: float, peakiness: float) -> bool:
    return (
        result_metric(row, "ug/map_entropy_norm", 1.0) <= entropy
        and result_metric(row, "ug/map_top_58_mass", 0.0) >= top58
        and result_metric(row, "ug/map_peakiness", 0.0) >= peakiness
    )


def evaluate_confidence_rules(
    compact_rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    by_action_sample: dict[tuple[str, str], dict[str, Any]] = {
        (str(row["action"]), str(row["sample_id"])): row for row in compact_rows
    }
    sample_ids = sorted({str(row["sample_id"]) for row in compact_rows})
    actions = sorted({str(row["action"]) for row in compact_rows})
    summaries: dict[str, Any] = {}
    for action in actions:
        chosen = {sid: by_action_sample[(action, sid)] for sid in sample_ids if (action, sid) in by_action_sample}
        if len(chosen) == len(sample_ids):
            summaries[f"fixed:{action}"] = evaluate_rule(chosen)

    k10 = "k10_b58_default_soft_average_query_max"
    k11 = "k11_b58_default_soft_average_query_max"
    k12 = "k12_b58_default_soft_average_query_max"
    if all(action in actions for action in (k10, k12)):
        chosen = {}
        for sid in sample_ids:
            probe = by_action_sample.get((k10, sid))
            if probe is None:
                continue
            selected = k10 if confidence_passes(
                probe,
                entropy=args.rule_entropy_norm_max,
                top58=args.rule_top58_mass_min,
                peakiness=args.rule_peakiness_min,
            ) else k12
            if (selected, sid) in by_action_sample:
                chosen[sid] = by_action_sample[(selected, sid)]
        if len(chosen) == len(sample_ids):
            summaries["rule:adaptive_k10_k12"] = evaluate_rule(chosen)

    if all(action in actions for action in (k10, k11, k12)):
        chosen = {}
        for sid in sample_ids:
            probe = by_action_sample.get((k10, sid))
            if probe is None:
                continue
            if confidence_passes(
                probe,
                entropy=args.rule_entropy_norm_max,
                top58=args.rule_top58_mass_min,
                peakiness=args.rule_peakiness_min,
            ):
                selected = k10
            elif confidence_passes(
                probe,
                entropy=args.rule_mid_entropy_norm_max,
                top58=args.rule_mid_top58_mass_min,
                peakiness=args.rule_mid_peakiness_min,
            ):
                selected = k11
            else:
                selected = k12
            if (selected, sid) in by_action_sample:
                chosen[sid] = by_action_sample[(selected, sid)]
        if len(chosen) == len(sample_ids):
            summaries["rule:adaptive_k10_k11_k12"] = evaluate_rule(chosen)
    return {
        "thresholds": {
            "strong": {
                "entropy_norm_max": args.rule_entropy_norm_max,
                "top58_mass_min": args.rule_top58_mass_min,
                "peakiness_min": args.rule_peakiness_min,
            },
            "middle": {
                "entropy_norm_max": args.rule_mid_entropy_norm_max,
                "top58_mass_min": args.rule_mid_top58_mass_min,
                "peakiness_min": args.rule_mid_peakiness_min,
            },
        },
        "rules": summaries,
    }


def aggregate_grid(args: argparse.Namespace, grid_name: str, actions: list[AdaptiveAction], action_run_dirs: dict[str, Path]) -> dict[str, Any]:
    raw_rows_by_action: dict[str, list[dict[str, Any]]] = {}
    action_summaries: list[dict[str, Any]] = []
    for action in actions:
        run_dir = action_run_dirs[action.name]
        summary = action_summary(action, run_dir)
        action_summaries.append(summary)
        per_sample = run_dir / "per_sample_results.jsonl"
        if summary.get("complete") and per_sample.exists():
            raw_rows_by_action[action.name] = load_jsonl(per_sample)

    if not raw_rows_by_action:
        raise BaselineError("No complete action runs were available to aggregate.")

    sample_sets = [set(str(row.get("sample_id")) for row in rows) for rows in raw_rows_by_action.values()]
    complete_sample_ids = sorted(set.intersection(*sample_sets)) if sample_sets else []
    latency_ref = reference_latency(raw_rows_by_action)

    compact_rows: list[dict[str, Any]] = []
    by_sample: dict[str, list[dict[str, Any]]] = {sid: [] for sid in complete_sample_ids}
    action_by_name = {action.name: action for action in actions}
    for action_name, rows in raw_rows_by_action.items():
        action = action_by_name[action_name]
        for row in rows:
            sid = str(row.get("sample_id"))
            if sid not in by_sample:
                continue
            compact = compact_action_row(action, row, latency_ref, args)
            compact_rows.append(compact)
            by_sample[sid].append(compact)

    oracle_rows: list[dict[str, Any]] = []
    for sid, candidates in by_sample.items():
        if len(candidates) != len(raw_rows_by_action):
            continue
        best = max(candidates, key=lambda row: (result_metric(row, "utility"), result_metric(row, "policy_quality"), -result_metric(row, "latency_s")))
        oracle_rows.append({"sample_id": sid, "best_action": best["action"], "best": best, "candidate_count": len(candidates)})

    oracle_chosen = {row["sample_id"]: row["best"] for row in oracle_rows}
    oracle_summary = evaluate_rule(oracle_chosen)
    ref_name = "k12_b58_default_soft_average_query_max"
    ref_rows = {
        row["sample_id"]: row
        for row in compact_rows
        if row["action"] == ref_name and row["sample_id"] in oracle_chosen
    }
    if len(ref_rows) == len(oracle_chosen) and ref_rows:
        ref_summary = evaluate_rule(ref_rows)
        oracle_summary["headroom_vs_k12_b58"] = {
            "acc_at_0_5_delta": (oracle_summary["quality/acc_at_0_5"] or 0.0) - (ref_summary["quality/acc_at_0_5"] or 0.0),
            "mean_iou_delta": (oracle_summary["quality/mean_iou"] or 0.0) - (ref_summary["quality/mean_iou"] or 0.0),
            "mean_latency_delta_s": (oracle_summary["efficiency/mean_latency_s"] or 0.0) - (ref_summary["efficiency/mean_latency_s"] or 0.0),
            "reference_summary": ref_summary,
        }

    confidence_rules = evaluate_confidence_rules(compact_rows, args=args)
    out_dir = args.output_root / grid_name
    write_json(out_dir / "config.json", {
        "created_at_utc": utc_now(),
        "grid_name": grid_name,
        "dataset": args.dataset,
        "task": args.task,
        "manifest": str(args.manifest),
        "split": args.split,
        "limit": args.limit,
        "purpose": args.purpose,
        "actions": [action.as_dict() for action in actions],
        "latency_reference_s": latency_ref,
        "single_pass_output_root": str(args.single_pass_output_root),
    })
    write_json(out_dir / "action_summary.json", action_summaries)
    write_jsonl(out_dir / "per_sample_action_results.jsonl", compact_rows)
    write_jsonl(out_dir / "oracle_per_sample.jsonl", oracle_rows)
    write_json(out_dir / "oracle_summary.json", oracle_summary)
    write_json(out_dir / "confidence_rules_summary.json", confidence_rules)
    return {
        "output_dir": str(out_dir),
        "action_count": len(actions),
        "complete_action_count": len(raw_rows_by_action),
        "complete_sample_count": len(complete_sample_ids),
        "oracle_summary": oracle_summary,
        "confidence_rules": confidence_rules,
    }


def run(args: argparse.Namespace) -> int:
    rows = load_manifest(args.manifest, args.limit)
    contains_reserved = any(
        str(row.get("dataset", "")).lower() == "refcoco"
        and str((row.get("metadata") or {}).get("source_split") or row.get("split") or "").lower() == "val"
        and 0 <= int((row.get("metadata") or {}).get("source_row", 10**9)) < 500
        for row in rows
        if isinstance(row.get("metadata"), dict) and str((row.get("metadata") or {}).get("source_row", "")).lstrip("-").isdigit()
    )
    if args.purpose != "final":
        assert_allowed_nonfinal_rows(rows, context=f"{args.purpose} grid manifest {args.manifest}")
    elif contains_reserved and not args.allow_final_refcoco500:
        raise BaselineError("Final RefCOCO-500 rows require --allow-final-refcoco500.")

    actions = actions_from_args(args)
    if args.purpose == "final" and contains_reserved and len(actions) > 2:
        raise BaselineError("RefCOCO-500 final mode is restricted to frozen candidate/reference actions, not a broad oracle grid.")

    grid_name = grid_run_name(args)
    print(f"Grid {grid_name}: {len(actions)} actions, {len(rows)} samples, purpose={args.purpose}")
    action_run_dirs: dict[str, Path] = {}
    for index, action in enumerate(actions, start=1):
        cmd, run_dir = command_for_action(args, grid_name, action)
        action_run_dirs[action.name] = run_dir
        if args.aggregate_only:
            continue
        print(f"[{index}/{len(actions)}] {action.name}")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        completed = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True)
        if completed.returncode != 0:
            message = f"Action {action.name} failed with exit code {completed.returncode}."
            if args.keep_going:
                print(message, file=sys.stderr)
                continue
            raise BaselineError(message)

    if args.dry_run:
        return 0
    summary = aggregate_grid(args, grid_name, actions, action_run_dirs)
    write_json(args.output_root / grid_name / "run_summary.json", summary)
    print(f"Saved adaptive grid summary to {summary['output_dir']}")
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
