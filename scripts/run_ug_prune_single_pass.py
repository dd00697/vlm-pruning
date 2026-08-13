#!/usr/bin/env python
"""Run the Phase 2B UG-Prune-Gap-58 mid-layer single-pass pilot."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    default_max_new_tokens,
    env_snapshot,
    evaluator_name,
    keep_ratio,
    load_existing_results,
    load_manifest,
    load_model,
    load_sample_images,
    make_prompt,
    manifest_checksum,
    metric_definition,
    registered_prompt_template,
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
from vlm_pruning.artifacts import (  # noqa: E402
    ArtifactIntegrityError,
    finalize_partial_jsonl,
    load_resume_rows,
    sha256_file,
    source_provenance,
    validate_completed_run,
)
from vlm_pruning.features import resolve_query_token_positions  # noqa: E402
from vlm_pruning.llava_hooks import (  # noqa: E402
    FROZEN_UG_GAP58_NAME,
    MidLayerUGPruneForwarder,
    ug_gap58_config_for_prune_layer,
)
from vlm_pruning.selectors import selector_for_method  # noqa: E402


DEFAULT_PROJECT = "vlm-pruning"
METHOD_NAME = "ug_prune_gap58_mid_layer_single_pass"
OUTPUT_ROOT = Path("experiments/runs_ug_prune_single_pass")
FIGURES_ROOT = Path("experiments/figures_ug_prune_single_pass")
TABLES_ROOT = Path("experiments/tables/ug_prune_single_pass")
ROLE_PRESETS: dict[str, tuple[float, float, float]] = {
    "default": (0.50, 0.30, 0.20),
    "more_context": (0.45, 0.40, 0.15),
    "more_anchors": (0.45, 0.25, 0.30),
}


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
    parser.add_argument(
        "--selection-method",
        default="ug",
        choices=["ug", "random", "uniform"],
        help="Selection rule applied at the configured mid-layer boundary.",
    )
    parser.add_argument("--position-mode", default="gap_preserving", choices=["gap_preserving", "compact"])
    parser.add_argument("--ug-map-source", default="query_max", choices=["query_max", "query_pooled"])
    parser.add_argument("--ug-fusion-mode", default="soft_average", choices=["soft_average", "single_layer"])
    parser.add_argument(
        "--ug-probe-layers",
        default=None,
        help="Optional comma-separated probe layers, for example 8,12. Defaults are derived from the pruning layer.",
    )
    parser.add_argument("--ug-role-preset", default="default", choices=["default", "more_context", "more_anchors"])
    parser.add_argument(
        "--selector-strategy",
        default="structured_tca",
        choices=["query_topk", "target_nms_only", "structured_tca"],
        help="Explicit query-map selector mechanism used when --selection-method=ug.",
    )
    parser.add_argument("--decode-mode", default="no_cache_recompute", choices=["no_cache_recompute", "layer_aware_cache"])
    parser.add_argument("--diagnostics-mode", default="full", choices=["full", "benchmark"])
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


def coerce_role_preset(value: str) -> str:
    key = str(value or "default").strip().lower().replace("-", "_")
    if key in {"default", "current", "target_context_anchor"}:
        return "default"
    if key in {"more_context", "context"}:
        return "more_context"
    if key in {"more_anchor", "more_anchors"}:
        return "more_anchors"
    raise BaselineError(f"Unsupported role preset: {value}")


def role_ratios(role_preset: str) -> tuple[float, float, float]:
    return ROLE_PRESETS[coerce_role_preset(role_preset)]


def selector_config_from_args(args: argparse.Namespace):
    target_ratio, context_ratio, anchor_ratio = role_ratios(args.ug_role_preset)
    probe_layers = None
    if args.ug_probe_layers:
        probe_layers = tuple(int(item.strip()) for item in args.ug_probe_layers.split(",") if item.strip())
    return ug_gap58_config_for_prune_layer(
        args.prune_after_layer,
        map_source=args.ug_map_source,
        fusion_mode=args.ug_fusion_mode,
        probe_layers=probe_layers,
        position_mode=args.position_mode,
        target_ratio=target_ratio,
        context_ratio=context_ratio,
        anchor_ratio=anchor_ratio,
        selector_strategy=args.selector_strategy,
    )


def run_name(args: argparse.Namespace, tokens: int) -> str:
    if args.run_name:
        return args.run_name
    suffix = f"limit{args.limit}_" if args.limit is not None else ""
    mode = "" if args.decode_mode == "no_cache_recompute" else f"_{safe_slug(args.decode_mode)}"
    diag = "" if args.diagnostics_mode == "full" else f"_{safe_slug(args.diagnostics_mode)}"
    variants = []
    if args.selection_method != "ug":
        variants.append(args.selection_method)
    elif args.selector_strategy != "structured_tca":
        variants.append(args.selector_strategy)
    if args.ug_map_source != "query_max":
        variants.append(args.ug_map_source)
    if args.ug_fusion_mode != "soft_average":
        variants.append(args.ug_fusion_mode)
    if args.ug_probe_layers:
        variants.append(f"probes_{args.ug_probe_layers.replace(',', '_')}")
    if args.position_mode != "gap_preserving":
        variants.append(args.position_mode)
    if coerce_role_preset(args.ug_role_preset) != "default":
        variants.append(coerce_role_preset(args.ug_role_preset))
    variant = "" if not variants else "_" + "_".join(safe_slug(item) for item in variants)
    return f"{suffix}{safe_slug(args.dataset)}_ug_gap58_mid_layer_l{args.prune_after_layer}_tokens{tokens}{variant}_seed{args.seed}_{args.precision}{mode}{diag}"


def query_text_for_sample(args: argparse.Namespace, sample: dict[str, Any]) -> str:
    return str(sample["query"] if args.task == "grounding" else sample["question"])


def config_payload(args: argparse.Namespace, name: str, original_tokens: int, tokens: int, checksum: str) -> dict[str, Any]:
    split = args.split or "unknown"
    selector_config = selector_config_from_args(args)
    role_preset = coerce_role_preset(args.ug_role_preset)
    target_ratio, context_ratio, anchor_ratio = role_ratios(role_preset)
    provenance = source_provenance(REPO_ROOT, [
        Path("scripts/run_ug_prune_single_pass.py"),
        Path("vlm_pruning/selectors/ug_prune.py"),
        Path("vlm_pruning/llava_hooks/mid_layer_pruning.py"),
        Path("vlm_pruning/artifacts.py"),
    ])
    return {
        "run_name": name,
        "source/git_commit": provenance["git_commit"],
        "source/git_dirty": provenance["git_dirty"],
        "source/sha256": provenance["source_sha256"],
        "created_at_utc": utc_now(),
        "model/name": args.model_path,
        "model/stack": "huggingface_llava",
        "model/precision": args.precision,
        "model/decoding": "deterministic_greedy",
        "model/max_new_tokens": args.max_new_tokens or default_max_new_tokens(args.dataset, args.task),
        "dataset/name": args.dataset,
        "dataset/split": split,
        "dataset/manifest": str(args.manifest),
        "dataset/manifest_checksum": checksum,
        "task": args.task,
        "method/name": METHOD_NAME if args.selection_method == "ug" else f"{args.selection_method}_mid_layer",
        "method/seed": args.seed,
        "selection/method": args.selection_method,
        "selection/strategy": selector_config.selector_strategy if args.selection_method == "ug" else "not_applicable",
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
        "ug/nms_radius": selector_config.nms_radius,
        "ug/anchor_pattern": selector_config.anchor_pattern,
        "ug/position_mode": selector_config.position_mode,
        "ug/prune_after_layer": args.prune_after_layer,
        "ug/decode_mode": args.decode_mode,
        "ug/diagnostics_mode": args.diagnostics_mode,
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
        "evaluator/name": evaluator_name(args.dataset, args.task),
        "evaluator/metric_definition": metric_definition(args.dataset, args.task),
        "prompt/template": registered_prompt_template(args),
    }


def completed_run_matches(
    run_dir: Path,
    args: argparse.Namespace,
    tokens: int,
    manifest_rows: list[dict[str, Any]],
    checksum: str,
) -> bool:
    if not (run_dir / "COMPLETED.json").is_file():
        return False
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
        int(config.get("dataset/num_samples", -1)) == len(manifest_rows),
        config.get("method/name")
        == (METHOD_NAME if args.selection_method == "ug" else f"{args.selection_method}_mid_layer"),
        config.get("selection/method", "ug") == args.selection_method,
        config.get("selection/strategy", "structured_tca")
        == (args.selector_strategy if args.selection_method == "ug" else "not_applicable"),
        int(config.get("method/seed", -1)) == int(args.seed),
        int(config.get("tokens/target_visual_tokens", -1)) == int(tokens),
        int(config.get("tokens/actual_visual_tokens", -1)) == int(tokens),
        int(config.get("ug/prune_after_layer", -1)) == int(args.prune_after_layer),
        config.get("ug/map_source", "query_max") == args.ug_map_source,
        config.get("ug/fusion_mode", "soft_average") == args.ug_fusion_mode,
        config.get("ug/probe_layers") == ",".join(
            str(layer) for layer in selector_config_from_args(args).probe_layers
        ),
        config.get("ug/role_preset", "default") == coerce_role_preset(args.ug_role_preset),
        config.get("position_mode", "gap_preserving") == args.position_mode,
        config.get("ug/decode_mode") == args.decode_mode,
        config.get("ug/diagnostics_mode", "full") == args.diagnostics_mode,
        config.get("model/precision") == args.precision,
        config.get("limit") == args.limit,
        int(metrics.get("samples/evaluated", -1)) == len(manifest_rows),
    ]
    if not all(checks):
        return False
    report = validate_completed_run(
        run_dir,
        manifest_rows,
        budget=tokens,
        original_tokens=args.image_grid_size * args.image_grid_size,
        task=args.task,
        strategy=args.selector_strategy if args.selection_method == "ug" else None,
    )
    return report["status"] == "pass"


def wandb_tags(args: argparse.Namespace, config: dict[str, Any], env: dict[str, Any] | None = None) -> list[str]:
    tags = [
        "phase2b",
        "ug-prune",
        "single-pass-pilot",
        f"selection-{args.selection_method}",
        safe_slug(args.dataset),
        args.task,
        f"tokens-{config['tokens/actual_visual_tokens']}",
        f"prune-layer-{args.prune_after_layer}",
        f"map-{args.ug_map_source}",
        f"fusion-{args.ug_fusion_mode}",
        f"role-{coerce_role_preset(args.ug_role_preset)}",
        args.precision,
        safe_slug(args.position_mode),
        safe_slug(args.decode_mode),
        safe_slug(args.diagnostics_mode),
    ]
    if env and env.get("gpu_name"):
        tags.append(safe_slug(str(env["gpu_name"])))
    tags.extend(safe_slug(item) for item in args.wandb_tags.split(",") if item.strip())
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
    selection: Any | None = None,
    selection_mode: str = "ug",
) -> tuple[str, int, float, Any, dict[str, Any]]:
    if int(full_inputs["input_ids"].shape[0]) != 1:
        raise BaselineError("Phase 2B pilot currently supports batch size 1.")
    if prep_diag.get("actual_llm_visual_tokens") != original_tokens:
        raise BaselineError("Mid-layer UG-Prune must start from all visual tokens.")

    input_ids = full_inputs["input_ids"]
    inputs_embeds = full_inputs["inputs_embeds"]
    attention_mask = full_inputs["attention_mask"]
    eos_ids = eos_token_ids(model, processor)

    if decode_mode == "layer_aware_cache":
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
            max_new_tokens=max_new_tokens,
            eos_ids=eos_ids,
            selection=selection,
            selection_mode=selection_mode,
        )
        generated = generated_result.generated_ids
        decoded = processor.decode(generated, skip_special_tokens=True).strip()
        retained = generated_result.selection.selected_indices
        compacted_len = int(generated_result.diagnostics["mid_layer/selected_sequence_length"])
        selection_overhead = float(generated_result.diagnostics.get("mid_layer/map_extraction_latency_s") or 0.0) + float(
            generated_result.diagnostics.get("mid_layer/selection_latency_s") or 0.0
        )
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
            "position_ids_mode": str(generated_result.diagnostics.get("ug/position_mode", "gap_preserving")),
            "position_ids_first": int(generated_result.diagnostics["mid_layer/selected_positions_first"]),
            "position_ids_last": int(generated_result.diagnostics["mid_layer/selected_positions_last"]),
            "decode_mode": "layer_aware_cache",
        }
        return decoded, int(generated.numel()), total_latency, generated_result.selection, diagnostics

    if decode_mode != "no_cache_recompute":
        raise BaselineError(f"Unsupported decode mode: {decode_mode}")

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
            selection=selection,
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
        "position_ids_mode": str(first_diag.get("ug/position_mode", "gap_preserving")),
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
    original_visual_tokens = int(selection.original_tokens)
    selected_budget = int(getattr(selection, "budget", config["tokens/target_visual_tokens"]))
    duplicate_count = len(retained) - len(set(int(index) for index in retained))
    out_of_range_count = sum(1 for index in retained if int(index) < 0 or int(index) >= original_visual_tokens)
    exact_budget = len(retained) == selected_budget and duplicate_count == 0 and out_of_range_count == 0
    sample_metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    category = sample.get("category") or sample_metadata.get("category")
    source_row = sample_metadata.get("source_row")
    result = {
        "sample_id": str(sample["sample_id"]),
        "dataset": args.dataset,
        "split": sample.get("split") or args.split,
        "category": category,
        "source_row": source_row,
        "original_sample_id": sample_metadata.get("original_sample_id", str(sample["sample_id"])),
        "image_path": str(image_path.as_posix()),
        "image_paths": [str(value) for value in sample.get("image_paths", [image_path])],
        "image_count": int(sample.get("image_count") or 1),
        "image_id": sample.get("image_id"),
        "mme_pair_id": sample.get("mme_pair_id") or sample_metadata.get("mme_pair_id"),
        "method": config.get("method/name", METHOD_NAME),
        "seed": args.seed,
        "target_visual_tokens": selected_budget,
        "keep_ratio": keep_ratio(selected_budget, original_visual_tokens),
        "raw_model_output": raw_output,
        "latency_s": latency,
        "generated_token_count": generated_count,
        "retained_indices_checksum": indices_checksum(retained),
        # The exact indices are lightweight (48 integers in final runs) and
        # remain essential for budget validation and qualitative overlays even
        # when heavy map diagnostics are disabled.
        "retained_indices": retained,
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
        "ug_budget_split": selector_metadata.get("budget_split"),
        "selector_strategy": selector_metadata.get(
            "selector_strategy", "structured_tca" if getattr(args, "selection_method", "ug") == "ug" else "not_applicable"
        ),
        "ug_nms_radius": selector_metadata.get("nms_radius"),
        "ug_nms_fallback_count": selector_metadata.get("nms_fallback_count", 0),
        "ug_realised_role_counts": selector_metadata.get("realised_role_counts"),
        "ug_role_labels": selection.role_labels,
        "ug_query_token_status": selector_metadata.get("query_token_status"),
        "ug_query_token_positions": selector_metadata.get("query_token_positions"),
        **diagnostics,
    }
    if args.diagnostics_mode == "full":
        result.update(
            {
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
            if args.skip_existing and completed_run_matches(run_dir, args, tokens, rows, checksum):
                print(f"Skipping existing completed run: {run_dir}")
                return 0

            config = config_payload(args, name, original_tokens, tokens, checksum)
            config["dataset/num_samples"] = len(rows)
            config["run/output_dir"] = str(run_dir)
            write_json(run_dir / "config.json", jsonable(config))
            with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(jsonable(config), handle, sort_keys=True)

            per_sample_path = run_dir / "per_sample_results.jsonl"
            partial_path = run_dir / "per_sample_results.partial.jsonl"
            existing = load_resume_rows(partial_path, rows) if args.resume else []
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
            forwarder = MidLayerUGPruneForwarder(
                torch,
                model,
                image_token_id=image_token_id,
                grid=args.image_grid_size,
                prune_after_layer=args.prune_after_layer,
                target_visual_tokens=tokens,
                selector_config=selector_config,
                diagnostics_mode=args.diagnostics_mode,
            )

            tokenizer = getattr(processor, "tokenizer", processor)
            baseline_selector = (
                selector_for_method(args.selection_method)
                if args.selection_method in {"random", "uniform"}
                else None
            )
            results = list(existing)
            figure_paths: list[Path] = []
            max_new_tokens = args.max_new_tokens or default_max_new_tokens(args.dataset, args.task)
            run_start = time.perf_counter()
            sample_mode = "a" if args.resume and partial_path.exists() else "w"
            with partial_path.open(sample_mode, encoding="utf-8") as handle:
                for index, sample in enumerate(pending):
                    sample_id = str(sample["sample_id"])
                    image_paths, images = load_sample_images(sample)
                    image_path = image_paths[0]
                    image = images[0]
                    sample_original_tokens = original_tokens * len(images)
                    width = int(sample.get("width") or image.size[0])
                    height = int(sample.get("height") or image.size[1])
                    prompt = make_prompt(args, sample)
                    text_prompt = apply_chat_template(processor, prompt, len(images))
                    query_text = query_text_for_sample(args, sample)

                    full_inputs, prep_diag = prepare_compacted_inputs(
                        torch,
                        processor,
                        model,
                        device,
                        images,
                        prompt,
                        list(range(sample_original_tokens)),
                        sample_original_tokens,
                    )
                    if baseline_selector is None:
                        resolution = resolve_query_token_positions(
                            tokenizer,
                            text_prompt,
                            query_text,
                            input_ids=full_inputs["input_ids"][0].detach().cpu().tolist(),
                            image_token_id=image_token_id,
                        )
                        if not resolution.token_positions:
                            raise BaselineError(
                                f"Query-token resolution failed for sample {sample_id}: {resolution.resolver_status}"
                            )
                        query_token_status = resolution.resolver_status
                        query_token_positions = resolution.token_positions
                        precomputed_selection = None
                    else:
                        query_token_status = "not_required_for_baseline_control"
                        query_token_positions = []
                        precomputed_selection = baseline_selector.select(
                            sample_id=sample_id,
                            budget=tokens,
                            original_tokens=sample_original_tokens,
                            grid=args.image_grid_size,
                            seed=args.seed,
                        )

                    raw_output, generated_count, latency, selection, diagnostics = generate_mid_layer_answer(
                        torch,
                        processor,
                        model,
                        full_inputs=full_inputs,
                        prep_diag=prep_diag,
                        sample_id=sample_id,
                        query_text=query_text,
                        query_token_status=query_token_status,
                        query_token_positions=query_token_positions,
                        seed=args.seed,
                        max_new_tokens=max_new_tokens,
                        original_tokens=sample_original_tokens,
                        forwarder=forwarder,
                        decode_mode=args.decode_mode,
                        selection=precomputed_selection,
                        selection_mode=args.selection_method,
                    )
                    expected_tokens = int(getattr(selection, "budget", tokens))
                    if int(diagnostics.get("actual_llm_visual_tokens") or -1) != expected_tokens:
                        raise BaselineError(
                            f"Actual retained visual tokens {diagnostics.get('actual_llm_visual_tokens')} != selected budget {expected_tokens}."
                        )
                    if expected_tokens != tokens:
                        raise BaselineError(f"Selected budget {expected_tokens} != target {tokens}.")
                    result = build_result(args, sample, image_path, raw_output, generated_count, latency, selection, diagnostics, config)
                    if args.diagnostics_mode == "full" and baseline_selector is None:
                        result["ug_query_token_debug"] = resolution.as_dict()

                    if args.task == "grounding":
                        raw_box, clipped_box, parse_failure = parse_box(raw_output, width, height, args.box_coordinate_mode)
                        iou = box_iou(clipped_box, sample["gt_box"]) if clipped_box is not None else 0.0
                        success = clipped_box is not None and iou >= 0.5
                        retained = selection.selected_indices
                        result.update(
                            {
                                "query": sample.get("query"),
                                "width": width,
                                "height": height,
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
                                "benchmark_fields": {
                                    key: sample.get(key)
                                    for key in (
                                        "index", "hint", "source", "l2_category", "question_type",
                                        "subfield", "options_raw", "capability", "answer_index",
                                    )
                                    if sample.get(key) is not None
                                },
                                **scored,
                            }
                        )

                    result = jsonable(result)
                    results.append(result)
                    handle.write(json.dumps(result, ensure_ascii=True) + "\n")
                    handle.flush()
                    done = len(existing) + index + 1
                    if done % 50 == 0 or done == len(rows):
                        print(f"Processed {done}/{len(rows)} samples")

            results, final_integrity = finalize_partial_jsonl(
                partial_path,
                per_sample_path,
                rows,
                budget=tokens,
                original_tokens=original_tokens,
                strategy=args.selector_strategy if args.selection_method == "ug" else None,
            )
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
            ):
                values = [float(row[field]) for row in results if row.get(field) is not None]
                if values:
                    metrics[f"{field}_mean"] = sum(values) / len(values)
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
            completion = validate_completed_run(
                run_dir,
                rows,
                budget=tokens,
                original_tokens=original_tokens,
                task=args.task,
                strategy=args.selector_strategy if args.selection_method == "ug" else None,
            )
            completion["finalization"] = final_integrity
            write_json(run_dir / "integrity_report.json", jsonable(completion))
            if completion["status"] != "pass":
                raise ArtifactIntegrityError("Final run integrity failed: " + "; ".join(completion["issues"]))
            write_json(
                run_dir / "COMPLETED.json",
                {
                    "status": "complete",
                    "completed_at_utc": utc_now(),
                    "rows": len(results),
                    "config_sha256": sha256_file(run_dir / "config.json"),
                    "predictions_sha256": sha256_file(per_sample_path),
                    "metrics_sha256": sha256_file(run_dir / "metrics.json"),
                    "integrity_sha256": sha256_file(run_dir / "integrity_report.json"),
                },
            )
            partial_path.unlink(missing_ok=True)
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
