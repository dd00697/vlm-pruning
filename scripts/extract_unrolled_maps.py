#!/usr/bin/env python
"""Extract Phase 1 LLaVA query-image diagnostic maps."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_baseline import DEFAULT_GRID, DEFAULT_MODEL, BaselineError, apply_chat_template, prepare_compacted_inputs  # noqa: E402
from run_official_baseline import (  # noqa: E402
    git_commit,
    git_dirty,
    load_manifest,
    load_model,
    make_prompt,
    manifest_checksum,
    safe_slug,
    utc_now,
    write_json,
)
from vlm_pruning.analysis.coverage import selection_diagnostics  # noqa: E402
from vlm_pruning.analysis.heatmaps import save_heatmap_overlay  # noqa: E402
from vlm_pruning.features import (  # noqa: E402
    extract_maps_from_hidden_states,
    map_statistics,
    pearson_correlation,
    resolve_query_token_positions,
    top_k_indices,
)
from vlm_pruning.geometry.grid import patch_bounds  # noqa: E402


OUTPUT_ROOT = Path("experiments/phase1_unrolled_maps")


def parse_int_list(value: str, *, name: str) -> tuple[int, ...]:
    try:
        out = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated integer list.") from exc
    if not out:
        raise argparse.ArgumentTypeError(f"{name} must not be empty.")
    if any(item <= 0 for item in out):
        raise argparse.ArgumentTypeError(f"{name} values must be positive.")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="refcoco")
    parser.add_argument("--task", default="grounding", choices=["grounding"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-path", "--model", dest="model_path", default=DEFAULT_MODEL)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--probe-layers", default="4,8,12,16,24,32")
    parser.add_argument("--top-ks", default="16,32,48")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--image-grid-size", type=int, default=DEFAULT_GRID)
    parser.add_argument("--max-heatmaps", type=int, default=30)
    parser.add_argument("--skip-heatmaps", action="store_true")
    parser.add_argument(
        "--box-coordinate-mode",
        default="normalized_0_1",
        choices=["normalized_0_1", "normalized_0_1000", "absolute_pixels"],
    )
    return parser.parse_args()


def run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    limit = f"limit{args.limit}" if args.limit is not None else "all"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{safe_slug(args.dataset)}_phase1_{limit}_{args.precision}_{stamp}"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def peak_inside_box(map_values: np.ndarray, gt_box: list[float], image_size: tuple[int, int], grid: int) -> bool:
    peak = int(np.asarray(map_values).reshape(-1).argmax())
    x0, y0, x1, y1 = patch_bounds(peak, image_size[0], image_size[1], grid)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return bool(float(gt_box[0]) <= cx <= float(gt_box[2]) and float(gt_box[1]) <= cy <= float(gt_box[3]))


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    layers = parse_int_list(args.probe_layers, name="probe-layers")
    top_ks = parse_int_list(args.top_ks, name="top-ks")
    if args.image_grid_size != DEFAULT_GRID:
        raise BaselineError(f"Phase 1 currently expects a {DEFAULT_GRID} x {DEFAULT_GRID} visual grid.")

    rows = load_manifest(args.manifest, args.limit)
    run_dir = args.output_root / run_name(args)
    heatmap_dir = run_dir / "heatmaps"
    if run_dir.exists():
        raise BaselineError(f"Output run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    config = {
        "phase": "phase1_unrolled_maps",
        "created_utc": utc_now(),
        "dataset": args.dataset,
        "task": args.task,
        "manifest": str(args.manifest),
        "manifest_checksum": manifest_checksum(args.manifest),
        "limit": args.limit,
        "model_path": args.model_path,
        "precision": args.precision,
        "probe_layers": list(layers),
        "top_ks": list(top_ks),
        "image_grid_size": args.image_grid_size,
        "max_heatmaps": args.max_heatmaps,
        "skip_heatmaps": args.skip_heatmaps,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }
    write_json(run_dir / "config.json", config)

    load_start = time.perf_counter()
    torch, processor, model, device, env = load_model(args.model_path, args.precision)
    model_load_s = time.perf_counter() - load_start
    tokenizer = getattr(processor, "tokenizer", processor)
    image_token_id = int(getattr(model.config, "image_token_index", 32000))
    original_tokens = args.image_grid_size * args.image_grid_size
    retained = list(range(original_tokens))

    query_debug_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    map_store: dict[str, np.ndarray] = {}
    maps_index: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    heatmap_count = 0
    run_start = time.perf_counter()

    with torch.inference_mode():
        for index, sample in enumerate(rows, start=1):
            sample_id = str(sample["sample_id"])
            image = Image.open(sample["image_path"]).convert("RGB")
            width, height = image.size
            prompt = make_prompt(args, sample)
            text_prompt = apply_chat_template(processor, prompt)
            compacted, compact_diag = prepare_compacted_inputs(
                torch,
                processor,
                model,
                device,
                image,
                prompt,
                retained,
                original_tokens,
            )
            if int(compact_diag["actual_llm_visual_tokens"]) != original_tokens:
                raise BaselineError("Phase 1 diagnostics must run with all visual tokens retained.")
            if int(compact_diag["sequence_length_reduction"]) != 0:
                raise BaselineError("Full-token Phase 1 probe unexpectedly shortened the LLM sequence.")

            input_ids = compacted["input_ids"][0].detach().cpu().tolist()
            resolution = resolve_query_token_positions(
                tokenizer,
                text_prompt,
                str(sample["query"]),
                input_ids=input_ids,
                image_token_id=image_token_id,
            )
            debug_row = {
                "sample_id": sample_id,
                "row_index": index - 1,
                "query": sample["query"],
                "input_sequence_length": int(compacted["input_ids"].shape[1]),
                "image_token_count": original_tokens,
                **resolution.as_dict(),
            }
            query_debug_rows.append(jsonable(debug_row))
            if not resolution.token_positions:
                skipped.append({"sample_id": sample_id, "reason": resolution.resolver_status})
                continue

            outputs = forward_hidden_states(model, compacted)
            layer_maps = extract_maps_from_hidden_states(
                outputs.hidden_states,
                compacted["input_ids"],
                image_token_id,
                resolution.token_positions,
                layers,
                grid=args.image_grid_size,
            )

            previous_by_source: dict[str, np.ndarray] = {}
            sample_slug = safe_slug(sample_id)
            for result in layer_maps:
                for source, raw_map in result.maps.items():
                    z_map = result.maps_z[source]
                    previous = previous_by_source.get(source)
                    agreement = pearson_correlation(previous, raw_map) if previous is not None else None
                    previous_by_source[source] = raw_map
                    base_key = f"{sample_slug}__layer{result.layer:02d}__{source}"
                    map_store[f"{base_key}__raw"] = raw_map.astype(np.float32)
                    map_store[f"{base_key}__z"] = z_map.astype(np.float32)
                    maps_index.append(
                        {
                            "sample_id": sample_id,
                            "layer": result.layer,
                            "source": source,
                            "raw_key": f"{base_key}__raw",
                            "z_key": f"{base_key}__z",
                        }
                    )

                    stats = map_statistics(raw_map, top_ks=top_ks, grid=args.image_grid_size)
                    stat_rows.append(
                        jsonable(
                            {
                                "sample_id": sample_id,
                                "dataset": sample.get("dataset", args.dataset),
                                "query": sample["query"],
                                "layer": result.layer,
                                "map_source": source,
                                "inter_layer_pearson_raw_prev": agreement,
                                **stats,
                            }
                        )
                    )

                    for top_k in top_ks:
                        selected = top_k_indices(raw_map, top_k)
                        diagnostics = selection_diagnostics(selected, sample["gt_box"], (width, height), args.image_grid_size)
                        diagnostic_rows.append(
                            jsonable(
                                {
                                    "sample_id": sample_id,
                                    "dataset": sample.get("dataset", args.dataset),
                                    "query": sample["query"],
                                    "layer": result.layer,
                                    "map_source": source,
                                    "top_k": top_k,
                                    "selected_indices": selected,
                                    "peak_center_inside_gt": peak_inside_box(raw_map, sample["gt_box"], (width, height), args.image_grid_size),
                                    **diagnostics,
                                }
                            )
                        )

                    if not args.skip_heatmaps and heatmap_count < args.max_heatmaps:
                        output_path = heatmap_dir / f"{sample_slug}_layer{result.layer:02d}_{source}.png"
                        save_heatmap_overlay(
                            image,
                            z_map,
                            output_path,
                            gt_box=sample["gt_box"],
                            top_k=min(48, original_tokens),
                            grid=args.image_grid_size,
                        )
                        heatmap_count += 1

            if index % 5 == 0 or index == len(rows):
                print(f"Processed {index}/{len(rows)} samples")

    np.savez_compressed(run_dir / "maps.npz", **map_store)
    write_json(run_dir / "maps_index.json", maps_index)
    write_jsonl(run_dir / "query_token_debug.jsonl", query_debug_rows)
    write_csv(run_dir / "map_stats.csv", stat_rows)
    write_csv(run_dir / "map_box_diagnostics.csv", diagnostic_rows)

    elapsed_s = time.perf_counter() - run_start
    summary = {
        "phase": "phase1_unrolled_maps",
        "samples_requested": len(rows),
        "samples_processed": len(rows) - len(skipped),
        "samples_skipped": len(skipped),
        "skipped": skipped,
        "map_count": len(map_store),
        "stats_rows": len(stat_rows),
        "diagnostic_rows": len(diagnostic_rows),
        "heatmaps_saved": heatmap_count,
        "model_load_s": model_load_s,
        "run_elapsed_s": elapsed_s,
        "environment": env,
    }
    write_json(run_dir / "run_summary.json", jsonable(summary))
    print(f"Saved Phase 1 diagnostic outputs to {run_dir}")
    return 0


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
