#!/usr/bin/env python
"""Run training-free UG-Prune true-token-removal experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
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
    TRUE_REMOVAL_LABEL,
    TRUE_REMOVAL_LOCATION,
    BaselineError,
    Tee,
    apply_chat_template,
    box_iou,
    generate_answer_true_removal,
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
    git_commit,
    git_dirty,
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
from vlm_pruning.analysis.coverage import selection_diagnostics  # noqa: E402
from vlm_pruning.analysis.heatmaps import save_heatmap_overlay  # noqa: E402
from vlm_pruning.features import (  # noqa: E402
    extract_maps_from_hidden_states,
    fuse_layer_maps,
    parse_bins,
    parse_int_tuple,
    resolve_query_token_positions,
)
from vlm_pruning.selectors import UGPruneConfig, UGPruneInputs, UGPruneSelector  # noqa: E402


DEFAULT_PROJECT = "vlm-pruning"
OUTPUT_ROOT = Path("experiments/runs_ug_prune")
FIGURES_ROOT = Path("experiments/figures_ug_prune")
TABLES_ROOT = Path("experiments/tables/ug_prune")
METHOD_NAME = "ug_prune_tf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True, choices=["grounding", "vqa"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--target-visual-tokens", "--token-budget", dest="target_visual_tokens", type=int)
    parser.add_argument("--keep-ratio", type=float)
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
    parser.add_argument("--max-figures", type=int, default=24)
    parser.add_argument("--box-coordinate-mode", default="normalized_0_1", choices=["normalized_0_1", "normalized_0_1000", "absolute_pixels"])
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
    parser.add_argument("--ug-map-source", default="query_max", choices=["query_max", "query_pooled"])
    parser.add_argument("--ug-probe-layers", default="4,8,12")
    parser.add_argument("--ug-target-ratio", type=float, default=0.50)
    parser.add_argument("--ug-context-ratio", type=float, default=0.30)
    parser.add_argument("--ug-anchor-ratio", type=float, default=0.20)
    parser.add_argument("--ug-context-radius", type=int, default=1)
    parser.add_argument("--ug-anchor-bins", default="3,3")
    parser.add_argument("--ug-nms-radius", type=int, default=1)
    parser.add_argument("--ug-save-map-overlays", action="store_true")
    parser.add_argument("--ug-max-map-overlays", type=int, default=24)
    return parser.parse_args()


def target_tokens(args: argparse.Namespace) -> int:
    original = args.image_grid_size * args.image_grid_size
    if args.target_visual_tokens is not None:
        return max(1, min(original, int(args.target_visual_tokens)))
    if args.keep_ratio is not None:
        return max(1, min(original, int(round(original * args.keep_ratio))))
    return min(original, 58)


def run_name(args: argparse.Namespace, tokens: int) -> str:
    if args.run_name:
        return args.run_name
    suffix = f"limit{args.limit}_" if args.limit is not None else ""
    return f"{suffix}{safe_slug(args.dataset)}_{METHOD_NAME}_tokens{tokens}_seed{args.seed}_{args.precision}"


def config_payload(args: argparse.Namespace, name: str, original_tokens: int, tokens: int, checksum: str) -> dict[str, Any]:
    split = args.split or "unknown"
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
        "pruning_implementation": TRUE_REMOVAL_LABEL,
        "pruning_location": TRUE_REMOVAL_LOCATION,
        "selector/name": METHOD_NAME,
        "selector/interface_version": "phase2",
        "ug/map_source": args.ug_map_source,
        "ug/probe_layers": list(parse_int_tuple(args.ug_probe_layers)),
        "ug/target_ratio": args.ug_target_ratio,
        "ug/context_ratio": args.ug_context_ratio,
        "ug/anchor_ratio": args.ug_anchor_ratio,
        "ug/context_radius": args.ug_context_radius,
        "ug/anchor_bins": list(parse_bins(args.ug_anchor_bins)),
        "ug/nms_radius": args.ug_nms_radius,
        "limit": args.limit,
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    return value


def forward_hidden_states(model: Any, compacted: dict[str, Any]) -> Any:
    kwargs = {
        "inputs_embeds": compacted["inputs_embeds"],
        "attention_mask": compacted["attention_mask"],
        "output_hidden_states": True,
        "use_cache": False,
        "return_dict": True,
    }
    try:
        return model(**kwargs, logits_to_keep=1)
    except TypeError:
        return model(**kwargs)


def query_text_for_sample(args: argparse.Namespace, sample: dict[str, Any]) -> str:
    if args.task == "grounding":
        return str(sample["query"])
    return str(sample["question"])


def completed_run_matches(run_dir: Path, args: argparse.Namespace, tokens: int, row_count: int, checksum: str) -> bool:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    if not metrics_path.exists() or not config_path.exists():
        return False
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    except Exception:
        return False
    checks = [
        config.get("dataset/manifest_checksum") == checksum,
        int(config.get("dataset/num_samples", -1)) == int(row_count),
        config.get("method/name") == METHOD_NAME,
        int(config.get("method/seed", -1)) == int(args.seed),
        int(config.get("tokens/target_visual_tokens", -1)) == int(tokens),
        int(config.get("tokens/actual_visual_tokens", -1)) == int(tokens),
        config.get("model/precision") == args.precision,
        config.get("limit") == args.limit,
        int(metrics.get("samples/evaluated", -1)) == int(row_count),
    ]
    return all(checks)


def init_wandb(args: argparse.Namespace, config: dict[str, Any], run_dir: Path, tags: list[str]) -> Any | None:
    if args.no_wandb or not args.wandb or args.wandb_mode == "disabled":
        return None
    try:
        import wandb

        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_mode:
            os.environ["WANDB_MODE"] = args.wandb_mode
        kwargs = {
            "project": args.wandb_project,
            "name": config["run_name"],
            "config": config,
            "dir": str(run_dir),
            "reinit": True,
            "tags": tags,
            "group": args.wandb_group or f"ug-prune-{safe_slug(args.dataset)}",
            "job_type": args.wandb_job_type or f"ug-prune-budget-{config['tokens/actual_visual_tokens']}",
        }
        if args.wandb_entity:
            kwargs["entity"] = args.wandb_entity
        return wandb.init(**kwargs)
    except Exception as exc:
        print(f"W&B init failed; continuing with local outputs. {type(exc).__name__}: {exc}")
        return None


def wandb_tags(args: argparse.Namespace, config: dict[str, Any], env: dict[str, Any] | None = None) -> list[str]:
    tags = [
        "phase2",
        "ug-prune",
        safe_slug(args.dataset),
        args.task,
        f"tokens-{config['tokens/actual_visual_tokens']}",
        str(config["tokens/budget_family"]),
        f"seed-{args.seed}",
        args.precision,
        "true-token-removal",
        f"map-{args.ug_map_source}",
    ]
    if env and env.get("gpu_name"):
        tags.append(safe_slug(str(env["gpu_name"])))
    tags.extend(safe_slug(item) for item in args.wandb_tags.split(",") if item.strip())
    return tags


def wandb_run_info(run: Any | None, args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    if run is None:
        return {"enabled": False, "project": args.wandb_project, "mode": args.wandb_mode}
    return {
        "enabled": True,
        "project": args.wandb_project,
        "entity": getattr(run, "entity", None),
        "id": getattr(run, "id", None),
        "name": getattr(run, "name", None),
        "url": getattr(run, "url", None),
        "group": args.wandb_group or f"ug-prune-{safe_slug(args.dataset)}",
        "job_type": args.wandb_job_type or f"ug-prune-budget-{config['tokens/actual_visual_tokens']}",
        "mode": args.wandb_mode,
    }


def log_wandb(run: Any | None, metrics: dict[str, Any], results: list[dict[str, Any]], run_dir: Path, figure_paths: list[Path]) -> None:
    if run is None:
        return
    try:
        import wandb

        run.log(metrics)
        for key in [
            "quality/main_score",
            "efficiency/mean_latency_s",
            "efficiency/theoretical_prefill_flops_ratio",
            "tokens/final_visual_tokens",
        ]:
            if key in metrics:
                run.summary[key] = metrics[key]
        table = wandb.Table(
            columns=[
                "sample_id",
                "dataset",
                "split",
                "target_visual_tokens",
                "question_or_query",
                "ground_truth",
                "prediction",
                "is_correct",
                "score",
                "latency_s",
                "query_token_status",
                "retained_checksum",
                "image_path",
            ]
        )
        for row in results:
            table.add_data(
                row.get("sample_id"),
                row.get("dataset"),
                row.get("split"),
                row.get("target_visual_tokens"),
                row.get("question") or row.get("query"),
                row.get("target") or row.get("gt_box"),
                row.get("prediction") or row.get("predicted_box"),
                row.get("correct") if "correct" in row else row.get("success_iou_0_5"),
                row.get("score") if "score" in row else row.get("box_iou"),
                row.get("latency_s"),
                row.get("ug_query_token_status"),
                row.get("retained_indices_checksum"),
                row.get("image_path"),
            )
        run.log({"per_sample_results": table})
        for figure in figure_paths[:8]:
            run.log({"qualitative_ug_prune": wandb.Image(str(figure))})
        artifact = wandb.Artifact(f"{safe_slug(run.name)}-outputs", type="ug-prune-run")
        for path in ["config.yaml", "metrics.json", "run_environment.json", "per_sample_results.jsonl", "wandb_run.json"]:
            candidate = run_dir / path
            if candidate.exists():
                artifact.add_file(str(candidate))
        run.log_artifact(artifact)
    except Exception as exc:
        print(f"W&B logging failed after local outputs were saved. {type(exc).__name__}: {exc}")


def build_result_common(
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
    metadata = selection.metadata
    return {
        "sample_id": str(sample["sample_id"]),
        "dataset": args.dataset,
        "split": sample.get("split") or args.split,
        "image_path": str(image_path.as_posix()),
        "image_id": sample.get("image_id"),
        "method": METHOD_NAME,
        "seed": args.seed,
        "target_visual_tokens": config["tokens/target_visual_tokens"],
        "keep_ratio": config["tokens/keep_ratio"],
        "raw_model_output": raw_output,
        "latency_s": latency,
        "generated_token_count": generated_count,
        "retained_indices_checksum": indices_checksum(retained),
        "retained_indices": retained,
        "selector_name": selection.method,
        "selector_metadata": selection.as_metadata(),
        "role_labels": selection.role_labels,
        "pruning_implementation": TRUE_REMOVAL_LABEL,
        "ug_map_key": metadata.get("map_key"),
        "ug_map_source": metadata.get("map_source"),
        "ug_probe_layers": metadata.get("probe_layers"),
        "ug_budget_split": metadata.get("budget_split"),
        "ug_fused_map_stats": metadata.get("map_stats"),
        "ug_query_token_status": metadata.get("query_token_status"),
        "ug_query_token_positions": metadata.get("query_token_positions"),
        **diagnostics,
    }


def add_role_coverage(result: dict[str, Any], role_labels: dict[str, list[int]], gt_box: list[float], image_size: tuple[int, int], grid: int) -> None:
    for role, indices in role_labels.items():
        diagnostics = selection_diagnostics(indices, gt_box, image_size, grid)
        for key, value in diagnostics.items():
            result[f"role/{role}/{key}"] = value


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args.method = METHOD_NAME
    if args.no_wandb:
        args.wandb = False
    layers = parse_int_tuple(args.ug_probe_layers)
    anchor_bins = parse_bins(args.ug_anchor_bins)
    original_tokens = args.image_grid_size * args.image_grid_size
    tokens = target_tokens(args)
    if tokens >= original_tokens:
        raise BaselineError("UG-Prune should be run with fewer than 576 visual tokens; use the full baseline for 576.")
    selector_config = UGPruneConfig(
        map_source=args.ug_map_source,
        probe_layers=layers,
        target_ratio=args.ug_target_ratio,
        context_ratio=args.ug_context_ratio,
        anchor_ratio=args.ug_anchor_ratio,
        context_radius=args.ug_context_radius,
        anchor_bins=anchor_bins,
        nms_radius=args.ug_nms_radius,
    ).validate()
    selector = UGPruneSelector(selector_config)

    name = run_name(args, tokens)
    run_dir = args.output_root / name
    figures_dir = args.figures_root / name
    map_figures_dir = figures_dir / "ug_maps"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if args.ug_save_map_overlays:
        map_figures_dir.mkdir(parents=True, exist_ok=True)
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
            write_json(run_dir / "run_environment.json", env)
            tags = wandb_tags(args, config, env)
            wandb_run = init_wandb(args, config, run_dir, tags)
            write_json(run_dir / "wandb_run.json", wandb_run_info(wandb_run, args, config))
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            tokenizer = getattr(processor, "tokenizer", processor)
            image_token_id = int(getattr(model.config, "image_token_index", 32000))
            full_retained = list(range(original_tokens))
            results = list(existing)
            figure_paths: list[Path] = []
            max_new_tokens = args.max_new_tokens or (48 if args.task == "grounding" else 16)
            run_start = time.perf_counter()
            map_overlay_count = 0
            sample_mode = "a" if args.resume and per_sample_path.exists() else "w"
            with per_sample_path.open(sample_mode, encoding="utf-8") as handle:
                for index, sample in enumerate(pending):
                    sample_id = str(sample["sample_id"])
                    image_path = Path(sample["image_path"])
                    image = Image.open(image_path).convert("RGB")
                    prompt = make_prompt(args, sample)
                    text_prompt = apply_chat_template(processor, prompt)
                    full_inputs, full_diag = prepare_compacted_inputs(
                        torch,
                        processor,
                        model,
                        device,
                        image,
                        prompt,
                        full_retained,
                        original_tokens,
                    )
                    if full_diag.get("actual_llm_visual_tokens") != original_tokens:
                        raise BaselineError("UG-Prune map extraction must start from all 576 visual tokens.")
                    query_text = query_text_for_sample(args, sample)
                    resolution = resolve_query_token_positions(
                        tokenizer,
                        text_prompt,
                        query_text,
                        input_ids=full_inputs["input_ids"][0].detach().cpu().tolist(),
                        image_token_id=image_token_id,
                    )
                    if not resolution.token_positions:
                        raise BaselineError(f"Query-token resolution failed for sample {sample_id}: {resolution.resolver_status}")
                    with torch.inference_mode():
                        hidden_outputs = forward_hidden_states(model, full_inputs)
                    layer_maps = extract_maps_from_hidden_states(
                        hidden_outputs.hidden_states,
                        full_inputs["input_ids"],
                        image_token_id,
                        resolution.token_positions,
                        layers,
                        grid=args.image_grid_size,
                    )
                    fused = fuse_layer_maps(layer_maps, source=args.ug_map_source, layers=layers, grid=args.image_grid_size)
                    selection_inputs = UGPruneInputs(
                        fused_map=fused.fused_map,
                        map_stats=fused.stats,
                        map_key="+".join(fused.layer_keys),
                        query_text=query_text,
                        query_token_status=resolution.resolver_status,
                        query_token_positions=resolution.token_positions,
                    )
                    selection = selector.select_from_map(
                        sample_id=sample_id,
                        budget=tokens,
                        original_tokens=original_tokens,
                        grid=args.image_grid_size,
                        seed=args.seed,
                        inputs=selection_inputs,
                    )
                    retained = selection.selected_indices
                    raw_output, generated_count, latency, diagnostics = generate_answer_true_removal(
                        torch,
                        processor,
                        model,
                        device,
                        image,
                        prompt,
                        max_new_tokens,
                        retained,
                        original_tokens,
                    )
                    if diagnostics.get("actual_llm_visual_tokens") != tokens:
                        raise BaselineError(f"Actual retained visual tokens {diagnostics.get('actual_llm_visual_tokens')} != target {tokens}.")
                    if diagnostics.get("compacted_input_sequence_length", 0) >= diagnostics.get("original_input_sequence_length", 0):
                        raise BaselineError(
                            "True token removal did not shorten the input sequence: "
                            f"original={diagnostics.get('original_input_sequence_length')} "
                            f"compacted={diagnostics.get('compacted_input_sequence_length')}"
                        )
                    result = build_result_common(args, sample, image_path, raw_output, generated_count, latency, selection, diagnostics, config)
                    result["ug_full_sequence_length_for_map_extraction"] = full_diag.get("compacted_input_sequence_length")
                    result["ug_query_token_debug"] = resolution.as_dict()
                    result["ug_fused_map_layer_keys"] = list(fused.layer_keys)

                    if args.task == "grounding":
                        width = int(sample.get("width") or image.size[0])
                        height = int(sample.get("height") or image.size[1])
                        raw_box, clipped_box, parse_failure = parse_box(raw_output, width, height, args.box_coordinate_mode)
                        iou = box_iou(clipped_box, sample["gt_box"]) if clipped_box is not None else 0.0
                        success = clipped_box is not None and iou >= 0.5
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
                        result.update(selection_diagnostics(retained, sample["gt_box"], (width, height), args.image_grid_size))
                        add_role_coverage(result, selection.role_labels, sample["gt_box"], (width, height), args.image_grid_size)
                        if len(figure_paths) < args.max_figures:
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
                    if args.ug_save_map_overlays and map_overlay_count < args.ug_max_map_overlays:
                        map_path = map_figures_dir / f"{safe_slug(sample_id)}_ug_fused.png"
                        save_heatmap_overlay(
                            image,
                            fused.fused_map,
                            map_path,
                            gt_box=sample.get("gt_box"),
                            top_k=tokens,
                            grid=args.image_grid_size,
                        )
                        figure_paths.append(map_path)
                        map_overlay_count += 1

                    results.append(jsonable(result))
                    handle.write(json.dumps(jsonable(result), ensure_ascii=True) + "\n")
                    done = len(existing) + index + 1
                    if done % 10 == 0 or done == len(rows):
                        print(f"Processed {done}/{len(rows)} samples")

            run_duration_s = time.perf_counter() - run_start
            peak_alloc = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
            peak_reserved = torch.cuda.max_memory_reserved() / 1024**2 if torch.cuda.is_available() else None
            metrics = aggregate(results, args, run_duration_s, peak_alloc, peak_reserved)
            add_model_efficiency(metrics, model, torch)
            metrics["model/load_latency_s"] = model_load_s
            metrics["tokens/budget_family"] = config["tokens/budget_family"]
            metrics["ug/map_source"] = args.ug_map_source
            metrics["ug/probe_layers"] = ",".join(str(layer) for layer in layers)
            metrics["ug/map_overlay_count"] = map_overlay_count
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
            print(f"Saved UG-Prune run outputs to {run_dir}")
            print(
                "Runtime estimate: completed={completed} remaining={remaining} eta={eta}".format(
                    completed=estimate.get("runtime/completed_runs"),
                    remaining=estimate.get("runtime/remaining_runs"),
                    eta=estimate.get("runtime/eta_utc"),
                )
            )
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
