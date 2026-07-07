#!/usr/bin/env python
"""Run the Phase 2B UG-Prune-Gap-58 mid-layer single-pass pilot."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_baseline import (  # noqa: E402
    DEFAULT_GRID,
    DEFAULT_MODEL,
    BaselineError,
    Tee,
    apply_chat_template,
    box_iou,
    cuda_sync,
    eos_token_ids,
    indices_checksum,
    parse_box,
    prepare_compacted_inputs,
    save_grounding_figure,
)
from run_official_baseline import (  # noqa: E402
    add_model_efficiency,
    add_relative_metrics,
    aggregate,
    budget_family,
    env_snapshot,
    keep_ratio,
    load_existing_results,
    load_manifest,
    load_model,
    make_prompt,
    manifest_checksum,
    runtime_estimate,
    safe_slug,
    score_vqa,
    update_runtime_files,
    update_tables,
    utc_now,
    write_json,
)
from run_ug_prune import init_wandb, log_wandb, wandb_run_info  # noqa: E402
from vlm_pruning.analysis.coverage import selection_diagnostics  # noqa: E402
from vlm_pruning.controllers import (  # noqa: E402
    AdaptiveAction,
    LoadedUGPrunePolicy,
    build_selector_config,
    coerce_role_preset,
    parse_action_config,
    role_ratios,
)
from vlm_pruning.features import resolve_query_token_positions  # noqa: E402
from vlm_pruning.geometry.grid import patch_bounds, ring  # noqa: E402
from vlm_pruning.llava_hooks import FROZEN_UG_GAP58_NAME, MidLayerUGPruneForwarder  # noqa: E402
from vlm_pruning.selectors.base import SelectionResult  # noqa: E402


DEFAULT_PROJECT = "vlm-pruning"
METHOD_NAME = "ug_prune_gap58_mid_layer_single_pass"
OUTPUT_ROOT = Path("experiments/runs_ug_prune_single_pass")
FIGURES_ROOT = Path("experiments/figures_ug_prune_single_pass")
TABLES_ROOT = Path("experiments/tables/ug_prune_single_pass")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True, choices=["grounding", "vqa"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--target-visual-tokens", "--token-budget", dest="target_visual_tokens", type=int, default=58)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--model-path", "--model", dest="model_path", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--figures-root", type=Path, default=FIGURES_ROOT)
    parser.add_argument("--tables-root", type=Path, default=TABLES_ROOT)
    parser.add_argument("--image-grid-size", type=int, default=DEFAULT_GRID)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-figures", type=int, default=8)
    parser.add_argument("--box-coordinate-mode", default="normalized_0_1", choices=["normalized_0_1", "normalized_0_1000", "absolute_pixels"])
    parser.add_argument("--prune-after-layer", type=int, default=12)
    parser.add_argument("--position-mode", default="gap_preserving", choices=["gap_preserving"])
    parser.add_argument("--ug-map-source", default="query_max", choices=["query_max", "query_pooled"])
    parser.add_argument("--ug-fusion-mode", default="soft_average", choices=["soft_average", "single_layer"])
    parser.add_argument("--ug-role-preset", default="default", choices=["default", "more_context", "more_anchors"])
    parser.add_argument("--ug-selection-strategy", default="standard", choices=["standard", "coverage", "coverage_zip"])
    parser.add_argument("--ug-zip-alpha", type=float, default=0.20)
    parser.add_argument(
        "--ug-oracle-mode",
        default="none",
        choices=["none", "target_box", "target_context_ring", "target_context_anchor"],
        help="Use RefCOCO ground-truth boxes to preselect retained visual tokens for oracle upper-bound ablations.",
    )
    parser.add_argument("--decode-mode", default="no_cache_recompute", choices=["no_cache_recompute", "layer_aware_cache"])
    parser.add_argument("--diagnostics-mode", default="full", choices=["full", "benchmark"])
    parser.add_argument("--adaptive-confidence-rule", action="store_true")
    parser.add_argument("--adaptive-early-layer", type=int, default=10)
    parser.add_argument("--adaptive-fallback-layer", type=int, default=12)
    parser.add_argument("--adaptive-entropy-norm-max", type=float, default=0.965)
    parser.add_argument("--adaptive-top58-mass-min", type=float, default=0.18)
    parser.add_argument("--adaptive-peakiness-min", type=float, default=2.0)
    parser.add_argument("--ug-policy-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"), choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-job-type", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--suite-total-runs", type=int, default=None)
    parser.add_argument("--suite-completed-runs", type=int, default=None)
    parser.add_argument("--suite-start-time", type=float, default=None)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def target_tokens(args: argparse.Namespace) -> int:
    original = args.image_grid_size * args.image_grid_size
    return max(1, min(original, int(args.target_visual_tokens)))


def selector_config_from_args(args: argparse.Namespace):
    config = build_selector_config(
        AdaptiveAction(
            prune_after_layer=args.prune_after_layer,
            target_visual_tokens=target_tokens(args),
            map_source=args.ug_map_source,
            fusion_mode=args.ug_fusion_mode,
            role_preset=args.ug_role_preset,
        )
    )
    return replace(
        config,
        selection_strategy=args.ug_selection_strategy,
        zip_alpha=float(args.ug_zip_alpha),
    ).validate()


class MidLayerPolicyController:
    """Bridge an offline-trained policy checkpoint into the mid-layer forwarder."""

    def __init__(self, checkpoint_path: Path, *, device: Any) -> None:
        self.policy = LoadedUGPrunePolicy(checkpoint_path, device=device)
        self.feature_action = self.policy.feature_action
        parsed_feature = parse_action_config(self.feature_action)
        self.feature_target_visual_tokens = int(parsed_feature.target_visual_tokens)
        self.feature_selector_config = build_selector_config(
            AdaptiveAction(
                prune_after_layer=int(parsed_feature.prune_after_layer),
                target_visual_tokens=int(parsed_feature.target_visual_tokens),
                map_source=parsed_feature.map_source,
                fusion_mode=parsed_feature.fusion_mode,
                role_preset=parsed_feature.role_preset,
                selection_strategy=parsed_feature.selection_strategy,
                zip_alpha=float(parsed_feature.zip_alpha),
            )
        )

    def choose_action(self, feature_row: dict[str, Any]) -> dict[str, Any]:
        decision = self.policy.predict_action(feature_row)
        parsed = decision["action_config"]
        if int(parsed.prune_after_layer) != 12:
            # The current online hook is intentionally K=12-only.
            raise BaselineError(f"Online policy only supports K=12 actions, got {decision['action']}.")
        selector_config = build_selector_config(
            AdaptiveAction(
                prune_after_layer=int(parsed.prune_after_layer),
                target_visual_tokens=int(parsed.target_visual_tokens),
                map_source=parsed.map_source,
                fusion_mode=parsed.fusion_mode,
                role_preset=parsed.role_preset,
                selection_strategy=parsed.selection_strategy,
                zip_alpha=float(parsed.zip_alpha),
            )
        )
        return {
            **decision,
            "selector_config": selector_config,
            "target_visual_tokens": int(parsed.target_visual_tokens),
        }


def run_name(args: argparse.Namespace, tokens: int) -> str:
    if args.run_name:
        return args.run_name
    suffix = f"limit{args.limit}_" if args.limit is not None else ""
    mode = "" if args.decode_mode == "no_cache_recompute" else f"_{safe_slug(args.decode_mode)}"
    diag = "" if args.diagnostics_mode == "full" else f"_{safe_slug(args.diagnostics_mode)}"
    adaptive = ""
    if args.adaptive_confidence_rule:
        adaptive = f"_adaptive_l{args.adaptive_early_layer}_l{args.adaptive_fallback_layer}"
    variants = []
    if args.ug_map_source != "query_max":
        variants.append(args.ug_map_source)
    if args.ug_fusion_mode != "soft_average":
        variants.append(args.ug_fusion_mode)
    if coerce_role_preset(args.ug_role_preset) != "default":
        variants.append(coerce_role_preset(args.ug_role_preset))
    if args.ug_selection_strategy != "standard":
        variants.append(args.ug_selection_strategy)
    if args.ug_selection_strategy == "coverage_zip" and abs(float(args.ug_zip_alpha) - 0.20) > 1e-9:
        variants.append(f"zipa{str(args.ug_zip_alpha).replace('.', 'p')}")
    if args.ug_oracle_mode != "none":
        variants.append(f"oracle_{args.ug_oracle_mode}")
    variant = "" if not variants else "_" + "_".join(safe_slug(item) for item in variants)
    return f"{suffix}{safe_slug(args.dataset)}_ug_gap58_mid_layer_l{args.prune_after_layer}{adaptive}_tokens{tokens}{variant}_seed{args.seed}_{args.precision}{mode}{diag}"


def query_text_for_sample(args: argparse.Namespace, sample: dict[str, Any]) -> str:
    return str(sample["query"] if args.task == "grounding" else sample["question"])


def _box_area(box: tuple[float, float, float, float] | list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _box_intersection(
    a: tuple[float, float, float, float] | list[float],
    b: tuple[float, float, float, float] | list[float],
) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _patch_center(index: int, width: int, height: int, grid: int) -> tuple[float, float]:
    x0, y0, x1, y1 = patch_bounds(index, width, height, grid)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _normalised_distance_sq(
    index: int,
    center: tuple[float, float],
    width: int,
    height: int,
    grid: int,
) -> float:
    px, py = _patch_center(index, width, height, grid)
    scale_x = max(float(width), 1.0)
    scale_y = max(float(height), 1.0)
    return ((px - center[0]) / scale_x) ** 2 + ((py - center[1]) / scale_y) ** 2


def _append_unique(selected: list[int], candidates: list[int], budget: int) -> list[int]:
    seen = set(selected)
    for index in candidates:
        idx = int(index)
        if idx in seen:
            continue
        selected.append(idx)
        seen.add(idx)
        if len(selected) >= budget:
            break
    return selected


def _fps_order(candidates: list[int], seeds: list[int], width: int, height: int, grid: int) -> list[int]:
    remaining = [int(index) for index in candidates]
    if not remaining:
        return []
    seed_points = [_patch_center(index, width, height, grid) for index in seeds]
    if not seed_points:
        seed_points = [(width / 2.0, height / 2.0)]
    ordered: list[int] = []
    while remaining:
        best_index = remaining[0]
        best_score = -1.0
        all_points = seed_points + [_patch_center(index, width, height, grid) for index in ordered]
        for index in remaining:
            px, py = _patch_center(index, width, height, grid)
            score = min((px - sx) ** 2 + (py - sy) ** 2 for sx, sy in all_points)
            if score > best_score or (score == best_score and index < best_index):
                best_index = index
                best_score = score
        ordered.append(best_index)
        remaining.remove(best_index)
    return ordered


def oracle_selection_for_sample(
    *,
    args: argparse.Namespace,
    sample: dict[str, Any],
    width: int,
    height: int,
    budget: int,
    original_tokens: int,
) -> SelectionResult | None:
    mode = str(args.ug_oracle_mode)
    if mode == "none":
        return None
    if args.task != "grounding" or args.dataset.lower() != "refcoco":
        raise BaselineError("--ug-oracle-mode is only supported for RefCOCO grounding runs.")
    if not sample.get("gt_box"):
        raise BaselineError(f"Oracle mode requires gt_box for sample {sample.get('sample_id')}.")
    grid = int(args.image_grid_size)
    box = [float(value) for value in sample["gt_box"]]
    if _box_area(box) <= 0.0:
        raise BaselineError(f"Oracle mode got an empty gt_box for sample {sample.get('sample_id')}: {box}")

    box_center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    all_indices = list(range(original_tokens))
    overlap_rows = []
    for index in all_indices:
        patch = patch_bounds(index, width, height, grid)
        overlap = _box_intersection(patch, box)
        if overlap > 0.0:
            overlap_rows.append((index, overlap, _normalised_distance_sq(index, box_center, width, height, grid)))
    target_order = [index for index, _, _ in sorted(overlap_rows, key=lambda row: (-row[1], row[2], row[0]))]
    nearest_order = sorted(all_indices, key=lambda index: (_normalised_distance_sq(index, box_center, width, height, grid), index))

    selected: list[int] = []
    roles: dict[str, list[int]] = {"target": [], "context": [], "anchor": [], "fill": []}

    before = len(selected)
    _append_unique(selected, target_order, budget)
    roles["target"] = selected[before:].copy()

    if mode in {"target_context_ring", "target_context_anchor"} and len(selected) < budget:
        context_candidates = sorted(
            ring(set(target_order), radius=1, grid=grid),
            key=lambda index: (_normalised_distance_sq(index, box_center, width, height, grid), index),
        )
        before = len(selected)
        _append_unique(selected, context_candidates, budget)
        roles["context"] = selected[before:].copy()

    if mode == "target_context_anchor" and len(selected) < budget:
        anchor_candidates = [index for index in all_indices if index not in set(selected)]
        before = len(selected)
        _append_unique(selected, _fps_order(anchor_candidates, selected, width, height, grid), budget)
        roles["anchor"] = selected[before:].copy()

    if len(selected) < budget:
        before = len(selected)
        _append_unique(selected, nearest_order, budget)
        roles["fill"] = selected[before:].copy()

    selected = sorted(selected)
    roles = {role: sorted(values) for role, values in roles.items() if values}
    result = SelectionResult(
        selected_indices=selected,
        method=f"oracle_{mode}",
        budget=budget,
        original_tokens=original_tokens,
        grid=grid,
        role_labels=roles,
        metadata={
            "selector_family": "oracle_gt_box",
            "selection_backend": "oracle_gt_box",
            "sample_id": str(sample.get("sample_id")),
            "map_family": "oracle_gt_box",
            "map_source": mode,
            "probe_layers": [],
            "fusion_mode": "oracle",
            "dynamic_layer_metric": None,
            "role_mode": mode,
            "anchor_pattern": "fps" if mode == "target_context_anchor" else None,
            "position_mode": args.position_mode,
            "selection_strategy": "oracle",
            "zip_alpha": None,
            "budget_split": {role: len(values) for role, values in roles.items()},
            "target_patch_count": len(target_order),
            "target_checksum": indices_checksum(roles.get("target", [])),
            "context_checksum": indices_checksum(roles.get("context", [])),
            "anchor_checksum": indices_checksum(roles.get("anchor", [])),
        },
    )
    return result.validate()


def config_payload(args: argparse.Namespace, name: str, original_tokens: int, tokens: int, checksum: str) -> dict[str, Any]:
    split = args.split or "unknown"
    selector_config = selector_config_from_args(args)
    role_preset = coerce_role_preset(args.ug_role_preset)
    target_ratio, context_ratio, anchor_ratio = role_ratios(role_preset)
    return {
        "run_name": name,
        "created_at_utc": utc_now(),
        "model/name": args.model_path,
        "model/stack": "huggingface_llava",
        "model/precision": args.precision,
        "dataset/name": args.dataset,
        "dataset/split": split,
        "dataset/manifest": str(args.manifest),
        "dataset/manifest_checksum": checksum,
        "task": args.task,
        "method/name": METHOD_NAME,
        "method/seed": args.seed,
        "tokens/original_visual_tokens": original_tokens,
        "tokens/target_visual_tokens": tokens,
        "tokens/actual_visual_tokens": tokens,
        "tokens/keep_ratio": keep_ratio(tokens, original_tokens),
        "tokens/budget_family": budget_family(tokens, original_tokens),
        "run/is_limited": args.limit is not None,
        "run/output_dir": None,
        "limit": args.limit,
        "position_mode": args.position_mode,
        "pruning_implementation": "mid_layer_single_pass",
        "pruning_location": f"after_layer_{args.prune_after_layer}",
        "ug/frozen_config": FROZEN_UG_GAP58_NAME,
        "ug/map_family": selector_config.map_family,
        "ug/map_source": selector_config.map_source,
        "ug/probe_layers": ",".join(str(layer) for layer in selector_config.probe_layers),
        "ug/fusion_mode": selector_config.fusion_mode,
        "ug/role_mode": selector_config.role_mode,
        "ug/role_preset": role_preset,
        "ug/target_ratio": target_ratio,
        "ug/context_ratio": context_ratio,
        "ug/anchor_ratio": anchor_ratio,
        "ug/context_radius": selector_config.context_radius,
        "ug/anchor_pattern": selector_config.anchor_pattern,
        "ug/position_mode": selector_config.position_mode,
        "ug/selection_strategy": selector_config.selection_strategy,
        "ug/zip_alpha": float(selector_config.zip_alpha),
        "ug/oracle_mode": args.ug_oracle_mode,
        "ug/prune_after_layer": args.prune_after_layer,
        "ug/decode_mode": args.decode_mode,
        "ug/diagnostics_mode": args.diagnostics_mode,
        "ug/adaptive_enabled": bool(args.adaptive_confidence_rule),
        "ug/adaptive_early_layer": args.adaptive_early_layer if args.adaptive_confidence_rule else None,
        "ug/adaptive_fallback_layer": args.adaptive_fallback_layer if args.adaptive_confidence_rule else None,
        "ug/adaptive_entropy_norm_max": args.adaptive_entropy_norm_max if args.adaptive_confidence_rule else None,
        "ug/adaptive_top58_mass_min": args.adaptive_top58_mass_min if args.adaptive_confidence_rule else None,
        "ug/adaptive_peakiness_min": args.adaptive_peakiness_min if args.adaptive_confidence_rule else None,
        "ug/policy_checkpoint": str(args.ug_policy_checkpoint) if args.ug_policy_checkpoint is not None else None,
        "ug/cache_implementation": (
            "layer_aware_two_bank_dynamic_cache"
            if args.decode_mode == "layer_aware_cache"
            else "none"
        ),
        "ug/cache_limitation": (
            None
            if args.decode_mode == "layer_aware_cache"
            else "layer_specific_kv_cache_not_used_in_v1"
        ),
    }


def completed_run_matches(run_dir: Path, args: argparse.Namespace, tokens: int, row_count: int, checksum: str) -> bool:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    if not metrics_path.exists() or not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    checks = [
        config.get("dataset/manifest_checksum") == checksum,
        int(config.get("dataset/num_samples", -1)) == int(row_count),
        config.get("method/name") == METHOD_NAME,
        int(config.get("method/seed", -1)) == int(args.seed),
        int(config.get("tokens/target_visual_tokens", -1)) == int(tokens),
        int(config.get("tokens/actual_visual_tokens", -1)) == int(tokens),
        int(config.get("ug/prune_after_layer", -1)) == int(args.prune_after_layer),
        config.get("ug/map_source", "query_max") == args.ug_map_source,
        config.get("ug/fusion_mode", "soft_average") == args.ug_fusion_mode,
        config.get("ug/role_preset", "default") == coerce_role_preset(args.ug_role_preset),
        config.get("ug/selection_strategy", "standard") == args.ug_selection_strategy,
        float(config.get("ug/zip_alpha", 0.20)) == float(args.ug_zip_alpha),
        config.get("ug/oracle_mode", "none") == args.ug_oracle_mode,
        config.get("ug/decode_mode") == args.decode_mode,
        config.get("ug/diagnostics_mode", "full") == args.diagnostics_mode,
        bool(config.get("ug/adaptive_enabled", False)) == bool(args.adaptive_confidence_rule),
        int(config.get("ug/adaptive_early_layer") or args.adaptive_early_layer) == int(args.adaptive_early_layer),
        int(config.get("ug/adaptive_fallback_layer") or args.adaptive_fallback_layer) == int(args.adaptive_fallback_layer),
        float(config.get("ug/adaptive_entropy_norm_max") or args.adaptive_entropy_norm_max) == float(args.adaptive_entropy_norm_max),
        float(config.get("ug/adaptive_top58_mass_min") or args.adaptive_top58_mass_min) == float(args.adaptive_top58_mass_min),
        float(config.get("ug/adaptive_peakiness_min") or args.adaptive_peakiness_min) == float(args.adaptive_peakiness_min),
        config.get("model/precision") == args.precision,
        config.get("limit") == args.limit,
        int(metrics.get("samples/evaluated", -1)) == int(row_count),
    ]
    return all(checks)


def wandb_tags(args: argparse.Namespace, config: dict[str, Any], env: dict[str, Any] | None = None) -> list[str]:
    tags = [
        "phase2b",
        "ug-prune",
        "single-pass-pilot",
        safe_slug(args.dataset),
        args.task,
        f"tokens-{config['tokens/actual_visual_tokens']}",
        f"prune-layer-{args.prune_after_layer}",
        f"map-{args.ug_map_source}",
        f"fusion-{args.ug_fusion_mode}",
        f"role-{coerce_role_preset(args.ug_role_preset)}",
        args.precision,
        "gap-preserving",
        safe_slug(args.decode_mode),
        safe_slug(args.diagnostics_mode),
    ]
    if env and env.get("gpu_name"):
        tags.append(safe_slug(str(env["gpu_name"])))
    tags.extend(safe_slug(item) for item in args.wandb_tags.split(",") if item.strip())
    if args.adaptive_confidence_rule:
        tags.append("adaptive-confidence")
    if args.ug_oracle_mode != "none":
        tags.append(f"oracle-{safe_slug(args.ug_oracle_mode)}")
    return tags


def append_generated_token(torch: Any, model: Any, input_ids: Any, inputs_embeds: Any, attention_mask: Any, token: Any) -> tuple[Any, Any, Any]:
    token_embed = model.get_input_embeddings()(token).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    input_ids = torch.cat([input_ids, token.to(device=input_ids.device, dtype=input_ids.dtype)], dim=1)
    inputs_embeds = torch.cat([inputs_embeds, token_embed], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones_like(token, device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)
    return input_ids, inputs_embeds, attention_mask


def generate_mid_layer_answer(
    torch: Any,
    processor: Any,
    model: Any,
    *,
    full_inputs: dict[str, Any],
    prep_diag: dict[str, Any],
    sample_id: str,
    query_text: str,
    query_token_status: str,
    query_token_positions: list[int],
    seed: int,
    max_new_tokens: int,
    original_tokens: int,
    forwarder: MidLayerUGPruneForwarder,
    decode_mode: str,
    policy_context: dict[str, Any] | None = None,
    adaptive_confidence_rule: bool = False,
    adaptive_early_layer: int = 10,
    adaptive_fallback_layer: int = 12,
    adaptive_entropy_norm_max: float = 0.965,
    adaptive_top58_mass_min: float = 0.18,
    adaptive_peakiness_min: float = 2.0,
    precomputed_selection: SelectionResult | None = None,
) -> tuple[str, int, float, Any, dict[str, Any]]:
    if int(full_inputs["input_ids"].shape[0]) != 1:
        raise BaselineError("Phase 2B pilot currently supports batch size 1.")
    if prep_diag.get("actual_llm_visual_tokens") != original_tokens:
        raise BaselineError("Mid-layer UG-Prune must start from all visual tokens.")
    if precomputed_selection is not None and adaptive_confidence_rule:
        raise BaselineError("Precomputed oracle selections are not compatible with adaptive confidence runs.")

    input_ids = full_inputs["input_ids"]
    inputs_embeds = full_inputs["inputs_embeds"]
    attention_mask = full_inputs["attention_mask"]
    eos_ids = eos_token_ids(model, processor)

    if decode_mode == "layer_aware_cache":
        if adaptive_confidence_rule:
            generated_result = forwarder.generate_with_adaptive_layer_aware_cache(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                sample_id=sample_id,
                original_tokens=original_tokens,
                seed=seed,
                query_text=query_text,
                query_token_status=query_token_status,
                query_token_positions=query_token_positions,
                policy_context=policy_context,
                max_new_tokens=max_new_tokens,
                eos_ids=eos_ids,
                early_prune_layer=adaptive_early_layer,
                fallback_prune_layer=adaptive_fallback_layer,
                entropy_norm_max=adaptive_entropy_norm_max,
                top58_mass_min=adaptive_top58_mass_min,
                peakiness_min=adaptive_peakiness_min,
            )
        else:
            generated_result = forwarder.generate_with_layer_aware_cache(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                sample_id=sample_id,
                original_tokens=original_tokens,
                seed=seed,
                query_text=query_text,
                query_token_status=query_token_status,
                query_token_positions=query_token_positions,
                policy_context=policy_context,
                max_new_tokens=max_new_tokens,
                eos_ids=eos_ids,
                selection=precomputed_selection,
            )
        generated = generated_result.generated_ids
        decoded = processor.decode(generated, skip_special_tokens=True).strip()
        retained = generated_result.selection.selected_indices
        compacted_len = int(generated_result.diagnostics["mid_layer/selected_sequence_length"])
        selection_overhead = float(generated_result.diagnostics.get("mid_layer/map_extraction_latency_s") or 0.0) + float(
            generated_result.diagnostics.get("mid_layer/selection_latency_s") or 0.0
        )
        if generated_result.diagnostics.get("ug/adaptive_enabled") and not generated_result.diagnostics.get(
            "ug/adaptive_confident_at_early_layer"
        ):
            selection_overhead += float(generated_result.diagnostics.get("mid_layer/early_map_extraction_latency_s") or 0.0)
            selection_overhead += float(generated_result.diagnostics.get("mid_layer/early_selection_latency_s") or 0.0)
        total_latency = (
            float(prep_diag["vision_encoder_latency_s"])
            + float(prep_diag["projector_latency_s"])
            + float(generated_result.diagnostics["llm_prefill_latency_s"])
            + float(generated_result.diagnostics["decode_latency_s"])
        )
        diagnostics = {
            **generated_result.diagnostics,
            "vision_encoder_latency_s": prep_diag["vision_encoder_latency_s"],
            "projector_latency_s": prep_diag["projector_latency_s"],
            "pruning_selection_latency_s": selection_overhead,
            "total_inference_latency_s": total_latency,
            "original_visual_tokens": original_tokens,
            "requested_keep_ratio": len(retained) / original_tokens,
            "expected_retained_visual_tokens": len(retained),
            "actual_retained_visual_tokens": len(retained),
            "actual_llm_visual_tokens": len(retained),
            "original_input_sequence_length": int(full_inputs["input_ids"].shape[1]),
            "compacted_input_sequence_length": compacted_len,
            "sequence_length_reduction": int(full_inputs["input_ids"].shape[1]) - compacted_len,
            "retained_indices_checksum": indices_checksum(retained),
            "position_ids_mode": "gap_preserving",
            "position_ids_first": int(generated_result.diagnostics["mid_layer/selected_positions_first"]),
            "position_ids_last": int(generated_result.diagnostics["mid_layer/selected_positions_last"]),
            "decode_mode": "layer_aware_cache",
        }
        return decoded, int(generated.numel()), total_latency, generated_result.selection, diagnostics

    if decode_mode != "no_cache_recompute":
        raise BaselineError(f"Unsupported decode mode: {decode_mode}")
    if adaptive_confidence_rule:
        raise BaselineError("Adaptive confidence rule currently requires --decode-mode layer_aware_cache.")

    generated_ids: list[Any] = []
    selection = None
    first_fused = None
    first_diag: dict[str, Any] = {}

    cuda_sync(torch)
    prefill_start = time.perf_counter()
    with torch.inference_mode():
        result = forwarder.forward_once(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            sample_id=sample_id,
            original_tokens=original_tokens,
            seed=seed,
            query_text=query_text,
            query_token_status=query_token_status,
            query_token_positions=query_token_positions,
            selection=precomputed_selection,
        )
        next_token = torch.argmax(result.logits[:, -1, :], dim=-1, keepdim=True)
    cuda_sync(torch)
    prefill_latency = time.perf_counter() - prefill_start
    generated_ids.append(next_token)
    selection = result.selection
    first_fused = result.fused
    first_diag = result.diagnostics
    finished = bool(eos_ids and int(next_token.item()) in eos_ids)

    cuda_sync(torch)
    decode_start = time.perf_counter()
    with torch.inference_mode():
        while len(generated_ids) < max_new_tokens and not finished:
            input_ids, inputs_embeds, attention_mask = append_generated_token(
                torch, model, input_ids, inputs_embeds, attention_mask, next_token
            )
            result = forwarder.forward_once(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                sample_id=sample_id,
                original_tokens=original_tokens,
                seed=seed,
                query_text=query_text,
                query_token_status=query_token_status,
                query_token_positions=query_token_positions,
                selection=selection,
            )
            next_token = torch.argmax(result.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids.append(next_token)
            finished = bool(eos_ids and int(next_token.item()) in eos_ids)
    cuda_sync(torch)
    decode_latency = time.perf_counter() - decode_start

    generated = torch.cat(generated_ids, dim=1)[0] if generated_ids else torch.empty(0, dtype=torch.long)
    decoded = processor.decode(generated, skip_special_tokens=True).strip()
    retained = selection.selected_indices
    compacted_len = int(first_diag["mid_layer/selected_sequence_length"])
    selection_overhead = float(first_diag.get("mid_layer/map_extraction_latency_s") or 0.0) + float(
        first_diag.get("mid_layer/selection_latency_s") or 0.0
    )
    total_latency = (
        float(prep_diag["vision_encoder_latency_s"])
        + float(prep_diag["projector_latency_s"])
        + prefill_latency
        + decode_latency
    )
    diagnostics = {
        **first_diag,
        "vision_encoder_latency_s": prep_diag["vision_encoder_latency_s"],
        "projector_latency_s": prep_diag["projector_latency_s"],
        "pruning_selection_latency_s": selection_overhead,
        "llm_prefill_latency_s": prefill_latency,
        "prefill_latency_s": prefill_latency,
        "time_to_first_token_s": prefill_latency,
        "decode_latency_s": decode_latency,
        "total_inference_latency_s": total_latency,
        "original_visual_tokens": original_tokens,
        "requested_keep_ratio": len(retained) / original_tokens,
        "expected_retained_visual_tokens": len(retained),
        "actual_retained_visual_tokens": len(retained),
        "actual_llm_visual_tokens": len(retained),
        "original_input_sequence_length": int(full_inputs["input_ids"].shape[1]),
        "compacted_input_sequence_length": compacted_len,
        "sequence_length_reduction": int(full_inputs["input_ids"].shape[1]) - compacted_len,
        "retained_indices_checksum": indices_checksum(retained),
        "position_ids_mode": "gap_preserving",
        "position_ids_first": int(first_diag["mid_layer/selected_positions_first"]),
        "position_ids_last": int(first_diag["mid_layer/selected_positions_last"]),
        "decode_mode": "no_cache_recompute",
    }
    if first_fused is not None:
        diagnostics["ug_fused_map_layer_keys"] = list(first_fused.layer_keys)
        diagnostics["ug_selected_layer"] = first_fused.selected_layer
        diagnostics["ug_layer_scores"] = first_fused.layer_scores
    return decoded, int(generated.numel()), total_latency, selection, diagnostics


def build_result(
    args: argparse.Namespace,
    sample: dict[str, Any],
    image_path: Path,
    raw_output: str,
    generated_count: int,
    latency: float,
    selection: Any,
    diagnostics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    retained = selection.selected_indices
    selector_metadata = selection.metadata
    original_visual_tokens = int(config.get("tokens/original_visual_tokens") or selection.original_tokens)
    selected_budget = int(getattr(selection, "budget", config["tokens/target_visual_tokens"]))
    duplicate_count = len(retained) - len(set(int(index) for index in retained))
    out_of_range_count = sum(1 for index in retained if int(index) < 0 or int(index) >= original_visual_tokens)
    exact_budget = len(retained) == selected_budget and duplicate_count == 0 and out_of_range_count == 0
    sample_metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    category = sample.get("category") or sample_metadata.get("category")
    source_row = sample_metadata.get("source_row")
    policy_sample_id = f"{args.dataset}:{category or ''}:{sample.get('sample_id')}:{'' if source_row is None else source_row}"
    result = {
        "sample_id": str(sample["sample_id"]),
        "policy_sample_id": policy_sample_id,
        "dataset": args.dataset,
        "split": sample.get("split") or args.split,
        "category": category,
        "source_row": source_row,
        "image_path": str(image_path.as_posix()),
        "image_id": sample.get("image_id"),
        "method": METHOD_NAME,
        "seed": args.seed,
        "target_visual_tokens": selected_budget,
        "keep_ratio": keep_ratio(selected_budget, original_visual_tokens),
        "raw_model_output": raw_output,
        "latency_s": latency,
        "generated_token_count": generated_count,
        "retained_indices_checksum": indices_checksum(retained),
        "selector_name": selection.method,
        "selection/exact_budget": float(exact_budget),
        "selection/duplicate_count": float(duplicate_count),
        "selection/out_of_range_count": float(out_of_range_count),
        "pruning_implementation": "mid_layer_single_pass",
        "ug_map_key": selector_metadata.get("map_key"),
        "ug_map_family": selector_metadata.get("map_family"),
        "ug_map_source": selector_metadata.get("map_source"),
        "ug_probe_layers": selector_metadata.get("probe_layers"),
        "ug_fusion_mode": selector_metadata.get("fusion_mode"),
        "ug_dynamic_layer_metric": selector_metadata.get("dynamic_layer_metric"),
        "ug_role_mode": selector_metadata.get("role_mode"),
        "ug_anchor_pattern": selector_metadata.get("anchor_pattern"),
        "ug_position_mode": selector_metadata.get("position_mode"),
        "ug_selection_strategy": selector_metadata.get("selection_strategy"),
        "ug_zip_alpha": selector_metadata.get("zip_alpha"),
        "ug_budget_split": selector_metadata.get("budget_split"),
        "ug_query_token_status": selector_metadata.get("query_token_status"),
        "ug_query_token_positions": selector_metadata.get("query_token_positions"),
        **diagnostics,
    }
    if args.diagnostics_mode == "full":
        result.update(
            {
                "retained_indices": retained,
                "selector_metadata": selection.as_metadata(),
                "role_labels": selection.role_labels,
                "ug_fused_map_stats": selector_metadata.get("map_stats"),
            }
        )
    return result


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args.method = METHOD_NAME
    if args.no_wandb:
        args.wandb = False
    if args.ug_policy_checkpoint is not None:
        if args.decode_mode != "layer_aware_cache":
            raise BaselineError("UG policy mode requires --decode-mode layer_aware_cache.")
        if int(args.prune_after_layer) != 12:
            raise BaselineError("UG policy mode currently supports only --prune-after-layer 12.")
    if args.ug_oracle_mode != "none":
        if args.task != "grounding" or args.dataset.lower() != "refcoco":
            raise BaselineError("--ug-oracle-mode is only supported for RefCOCO grounding runs.")
        if args.ug_policy_checkpoint is not None:
            raise BaselineError("--ug-oracle-mode cannot be combined with --ug-policy-checkpoint.")
        if args.adaptive_confidence_rule:
            raise BaselineError("--ug-oracle-mode cannot be combined with --adaptive-confidence-rule.")
    if args.adaptive_confidence_rule:
        if args.decode_mode != "layer_aware_cache":
            raise BaselineError("Adaptive confidence rule requires --decode-mode layer_aware_cache.")
        if int(args.prune_after_layer) != int(args.adaptive_fallback_layer):
            raise BaselineError("--prune-after-layer must match --adaptive-fallback-layer for adaptive runs.")
        if int(args.adaptive_early_layer) >= int(args.adaptive_fallback_layer):
            raise BaselineError("--adaptive-early-layer must be earlier than --adaptive-fallback-layer.")
    original_tokens = args.image_grid_size * args.image_grid_size
    tokens = target_tokens(args)
    selector_config = selector_config_from_args(args)
    name = run_name(args, tokens)
    run_dir = args.output_root / name
    figures_dir = args.figures_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if args.resume else "w"
    with (run_dir / "stdout.log").open(log_mode, encoding="utf-8") as log_handle:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(sys.stdout, log_handle), Tee(sys.stderr, log_handle)
        try:
            rows = load_manifest(args.manifest, args.limit)
            split = args.split or str(rows[0].get("split") or "unknown")
            args.split = split
            checksum = manifest_checksum(args.manifest)
            if args.skip_existing and completed_run_matches(run_dir, args, tokens, len(rows), checksum):
                print(f"Skipping existing completed run: {run_dir}")
                return 0

            config = config_payload(args, name, original_tokens, tokens, checksum)
            config["dataset/num_samples"] = len(rows)
            config["run/output_dir"] = str(run_dir)
            write_json(run_dir / "config.json", jsonable(config))
            with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(jsonable(config), handle, sort_keys=True)

            per_sample_path = run_dir / "per_sample_results.jsonl"
            existing = load_existing_results(per_sample_path, rows) if args.resume else []
            completed = {str(row["sample_id"]) for row in existing}
            pending = [row for row in rows if str(row["sample_id"]) not in completed]
            print(f"Loaded {len(rows)} samples from {args.manifest}; pending={len(pending)} completed={len(existing)}")

            model_load_start = time.perf_counter()
            torch, processor, model, device, model_env = load_model(args.model_path, args.precision)
            model_load_s = time.perf_counter() - model_load_start
            env = env_snapshot(args, model_env)
            env["model_load_latency_s"] = model_load_s
            write_json(run_dir / "run_environment.json", jsonable(env))
            tags = wandb_tags(args, config, env)
            wandb_run = init_wandb(args, config, run_dir, tags)
            write_json(run_dir / "wandb_run.json", jsonable(wandb_run_info(wandb_run, args, config)))
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            image_token_id = int(getattr(model.config, "image_token_index", 32000))
            policy_controller = (
                MidLayerPolicyController(args.ug_policy_checkpoint, device=device)
                if args.ug_policy_checkpoint is not None
                else None
            )
            forwarder = MidLayerUGPruneForwarder(
                torch,
                model,
                image_token_id=image_token_id,
                grid=args.image_grid_size,
                prune_after_layer=args.prune_after_layer,
                target_visual_tokens=tokens,
                selector_config=selector_config,
                diagnostics_mode=args.diagnostics_mode,
                policy_controller=policy_controller,
            )

            tokenizer = getattr(processor, "tokenizer", processor)
            results = list(existing)
            figure_paths: list[Path] = []
            max_new_tokens = args.max_new_tokens or (48 if args.task == "grounding" else 16)
            run_start = time.perf_counter()
            sample_mode = "a" if args.resume and per_sample_path.exists() else "w"
            with per_sample_path.open(sample_mode, encoding="utf-8") as handle:
                for index, sample in enumerate(pending):
                    sample_id = str(sample["sample_id"])
                    image_path = Path(sample["image_path"])
                    image = Image.open(image_path).convert("RGB")
                    width = int(sample.get("width") or image.size[0])
                    height = int(sample.get("height") or image.size[1])
                    prompt = make_prompt(args, sample)
                    text_prompt = apply_chat_template(processor, prompt)
                    query_text = query_text_for_sample(args, sample)

                    full_inputs, prep_diag = prepare_compacted_inputs(
                        torch,
                        processor,
                        model,
                        device,
                        image,
                        prompt,
                        list(range(original_tokens)),
                        original_tokens,
                    )
                    resolution = resolve_query_token_positions(
                        tokenizer,
                        text_prompt,
                        query_text,
                        input_ids=full_inputs["input_ids"][0].detach().cpu().tolist(),
                        image_token_id=image_token_id,
                    )
                    if not resolution.token_positions:
                        raise BaselineError(f"Query-token resolution failed for sample {sample_id}: {resolution.resolver_status}")

                    precomputed_selection = oracle_selection_for_sample(
                        args=args,
                        sample=sample,
                        width=width,
                        height=height,
                        budget=tokens,
                        original_tokens=original_tokens,
                    )
                    raw_output, generated_count, latency, selection, diagnostics = generate_mid_layer_answer(
                        torch,
                        processor,
                        model,
                        full_inputs=full_inputs,
                        prep_diag=prep_diag,
                        sample_id=sample_id,
                        query_text=query_text,
                        query_token_status=resolution.resolver_status,
                        query_token_positions=resolution.token_positions,
                        policy_context={
                            "dataset": args.dataset,
                            "task": args.task,
                            "task_family": "grounding" if args.task == "grounding" else ("yes_no_vqa" if args.dataset.lower() == "pope" else "open_vqa"),
                            "question": sample.get("question"),
                        },
                        seed=args.seed,
                        max_new_tokens=max_new_tokens,
                        original_tokens=original_tokens,
                        forwarder=forwarder,
                        decode_mode=args.decode_mode,
                        adaptive_confidence_rule=args.adaptive_confidence_rule,
                        adaptive_early_layer=args.adaptive_early_layer,
                        adaptive_fallback_layer=args.adaptive_fallback_layer,
                        adaptive_entropy_norm_max=args.adaptive_entropy_norm_max,
                        adaptive_top58_mass_min=args.adaptive_top58_mass_min,
                        adaptive_peakiness_min=args.adaptive_peakiness_min,
                        precomputed_selection=precomputed_selection,
                    )
                    expected_tokens = int(getattr(selection, "budget", tokens))
                    if int(diagnostics.get("actual_llm_visual_tokens") or -1) != expected_tokens:
                        raise BaselineError(
                            f"Actual retained visual tokens {diagnostics.get('actual_llm_visual_tokens')} != selected budget {expected_tokens}."
                        )
                    if args.ug_policy_checkpoint is None and expected_tokens != tokens:
                        raise BaselineError(f"Selected budget {expected_tokens} != target {tokens}.")
                    result = build_result(args, sample, image_path, raw_output, generated_count, latency, selection, diagnostics, config)
                    if args.diagnostics_mode == "full":
                        result["ug_query_token_debug"] = resolution.as_dict()

                    if args.task == "grounding":
                        raw_box, clipped_box, parse_failure = parse_box(raw_output, width, height, args.box_coordinate_mode)
                        iou = box_iou(clipped_box, sample["gt_box"]) if clipped_box is not None else 0.0
                        success = clipped_box is not None and iou >= 0.5
                        retained = selection.selected_indices
                        result.update(
                            {
                                "query": sample.get("query"),
                                "gt_box": sample.get("gt_box"),
                                "raw_parsed_box": raw_box,
                                "predicted_box": clipped_box,
                                "box_iou": iou,
                                "score": iou,
                                "success_iou_0_5": success,
                                "valid": clipped_box is not None,
                                "invalid": clipped_box is None,
                                "parse_failure_reason": parse_failure,
                            }
                        )
                        if args.diagnostics_mode == "full":
                            result.update(selection_diagnostics(retained, sample["gt_box"], (width, height), args.image_grid_size))
                        if args.diagnostics_mode == "full" and len(figure_paths) < args.max_figures:
                            figure_paths.append(
                                save_grounding_figure(
                                    sample,
                                    image,
                                    clipped_box,
                                    retained,
                                    args.image_grid_size,
                                    METHOD_NAME,
                                    keep_ratio(tokens, original_tokens),
                                    iou,
                                    success,
                                    figures_dir,
                                )
                            )
                    else:
                        scored = score_vqa(sample, raw_output, args.dataset)
                        result.update(
                            {
                                "question": sample.get("question"),
                                "choices": sample.get("choices"),
                                "target_answer": sample.get("answer"),
                                **scored,
                            }
                        )

                    result = jsonable(result)
                    results.append(result)
                    handle.write(json.dumps(result, ensure_ascii=True) + "\n")
                    done = len(existing) + index + 1
                    if done % 1 == 0:
                        print(f"Processed {done}/{len(rows)} samples")

            run_duration_s = time.perf_counter() - run_start
            peak_alloc = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
            peak_reserved = torch.cuda.max_memory_reserved() / 1024**2 if torch.cuda.is_available() else None
            metrics = aggregate(results, args, run_duration_s, peak_alloc, peak_reserved)
            add_model_efficiency(metrics, model, torch)
            metrics["selection/exact_budget_rate"] = (
                sum(float(row.get("selection/exact_budget") or 0.0) for row in results) / len(results) if results else None
            )
            metrics["selection/duplicate_count_mean"] = (
                sum(float(row.get("selection/duplicate_count") or 0.0) for row in results) / len(results) if results else None
            )
            metrics["selection/out_of_range_count_mean"] = (
                sum(float(row.get("selection/out_of_range_count") or 0.0) for row in results) / len(results) if results else None
            )
            metrics["model/load_latency_s"] = model_load_s
            metrics["tokens/budget_family"] = config["tokens/budget_family"]
            metrics["position_mode"] = args.position_mode
            metrics["ug/frozen_config"] = FROZEN_UG_GAP58_NAME
            metrics["ug/prune_after_layer"] = args.prune_after_layer
            metrics["ug/probe_layers"] = config["ug/probe_layers"]
            metrics["ug/map_source"] = config["ug/map_source"]
            metrics["ug/fusion_mode"] = config["ug/fusion_mode"]
            metrics["ug/role_preset"] = config["ug/role_preset"]
            metrics["ug/selection_strategy"] = config["ug/selection_strategy"]
            metrics["ug/zip_alpha"] = config["ug/zip_alpha"]
            metrics["ug/oracle_mode"] = config["ug/oracle_mode"]
            metrics["ug/target_ratio"] = config["ug/target_ratio"]
            metrics["ug/context_ratio"] = config["ug/context_ratio"]
            metrics["ug/anchor_ratio"] = config["ug/anchor_ratio"]
            metrics["ug/decode_mode"] = args.decode_mode
            metrics["ug/diagnostics_mode"] = args.diagnostics_mode
            for field in (
                "ug/map_entropy_norm",
                "ug/map_peakiness",
                "ug/map_top_48_mass",
                "ug/map_top_58_mass",
                "ug/map_top_72_mass",
                "ug/map_top_96_mass",
                "ug/map_selected_occupied_bins_3x3",
                "ug/map_selected_mean_pairwise_spread",
                "ug/zip_discarded_tokens",
                "ug/zip_merge_target_tokens",
                "ug/zip_assigned_tokens",
                "ug/zip_nonempty_merge_targets",
            ):
                values = [float(row[field]) for row in results if row.get(field) is not None]
                if values:
                    metrics[f"{field}_mean"] = sum(values) / len(values)
            metrics["ug/adaptive_enabled"] = bool(args.adaptive_confidence_rule)
            metrics["ug/policy_enabled"] = args.ug_policy_checkpoint is not None
            if args.ug_policy_checkpoint is not None:
                policy_counts: dict[str, int] = {}
                policy_raw_counts: dict[str, int] = {}
                policy_budgets: dict[str, int] = {}
                b96_blocked = 0
                for row in results:
                    action = row.get("ug/policy_selected_action")
                    if action is not None:
                        key = str(action)
                        policy_counts[key] = policy_counts.get(key, 0) + 1
                    raw_action = row.get("ug/policy_raw_action")
                    if raw_action is not None:
                        key = str(raw_action)
                        policy_raw_counts[key] = policy_raw_counts.get(key, 0) + 1
                    decision = row.get("ug/policy_constraint_decision")
                    if isinstance(decision, dict) and decision.get("raw_best_action") != decision.get("selected_action"):
                        b96_blocked += 1
                    budget = row.get("ug/policy_selected_budget")
                    if budget is not None:
                        key = str(int(budget))
                        policy_budgets[key] = policy_budgets.get(key, 0) + 1
                metrics["ug/policy_checkpoint"] = str(args.ug_policy_checkpoint)
                metrics["ug/policy_action_counts"] = policy_counts
                metrics["ug/policy_raw_action_counts"] = policy_raw_counts
                metrics["ug/policy_budget_counts"] = policy_budgets
                metrics["ug/policy_b96_blocked_count"] = b96_blocked
            if args.adaptive_confidence_rule:
                chosen_layers: dict[str, int] = {}
                confident_count = 0
                for row in results:
                    chosen = row.get("ug/adaptive_chosen_layer")
                    if chosen is not None:
                        key = str(int(chosen))
                        chosen_layers[key] = chosen_layers.get(key, 0) + 1
                    if row.get("ug/adaptive_confident_at_early_layer"):
                        confident_count += 1
                metrics["ug/adaptive_early_layer"] = args.adaptive_early_layer
                metrics["ug/adaptive_fallback_layer"] = args.adaptive_fallback_layer
                metrics["ug/adaptive_chosen_layer_counts"] = chosen_layers
                metrics["ug/adaptive_confident_rate"] = confident_count / len(results) if results else None
                metrics["ug/adaptive_entropy_norm_max"] = args.adaptive_entropy_norm_max
                metrics["ug/adaptive_top58_mass_min"] = args.adaptive_top58_mass_min
                metrics["ug/adaptive_peakiness_min"] = args.adaptive_peakiness_min
            backends = sorted({str(row.get("ug/selection_backend")) for row in results if row.get("ug/selection_backend")})
            metrics["ug/selection_backend"] = ",".join(backends) if backends else None
            metrics["ug/cache_implementation"] = (
                "layer_aware_two_bank_dynamic_cache"
                if args.decode_mode == "layer_aware_cache"
                else "none"
            )
            metrics["ug/cache_limitation"] = (
                None
                if args.decode_mode == "layer_aware_cache"
                else "layer_specific_kv_cache_not_used_in_v1"
            )
            estimate = runtime_estimate(args, run_duration_s)
            metrics.update({key: value for key, value in estimate.items() if key.startswith("runtime/")})
            add_relative_metrics(metrics, config, args.tables_root)
            write_json(run_dir / "metrics.json", jsonable(metrics))
            update_tables(config, metrics, args)
            update_runtime_files(args.tables_root, estimate)
            log_wandb(wandb_run, metrics, results, run_dir, figure_paths)
            if wandb_run is not None:
                try:
                    wandb_run.finish()
                except Exception:
                    pass
            print(f"Saved Phase 2B mid-layer run outputs to {run_dir}")
            return 0
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except BaselineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
