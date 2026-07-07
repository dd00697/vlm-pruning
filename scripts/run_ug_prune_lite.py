#!/usr/bin/env python
"""Evaluate UG-Prune-Lite with pre-LLM true token removal."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

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
    apply_chat_template,
    box_iou,
    indices_checksum,
    parse_box,
    selected_vision_features,
)
from run_official_baseline import (  # noqa: E402
    aggregate,
    greedy_decode_from_embeds_position_mode,
    keep_ratio,
    load_model,
    make_prompt,
    utc_now,
    write_json,
)
from vlm_pruning.features import resolve_query_token_positions  # noqa: E402
from vlm_pruning.selectors.ug_prune_lite import UGPruneLiteConfig, UGPruneLiteSelector  # noqa: E402
from vlm_pruning.training.ug_prune_lite_dataset import read_jsonl  # noqa: E402
from vlm_pruning.training.ug_prune_lite_features import maybe_apply_projection, projection_matrix  # noqa: E402
from vlm_pruning.training.ug_prune_lite_model import UGPruneLiteModel, UGPruneLiteModelConfig  # noqa: E402


METHOD_NAME = "ug_prune_lite"
OUTPUT_ROOT = Path("experiments/runs_ug_prune_lite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True, choices=["grounding"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target-visual-tokens", type=int, default=58)
    parser.add_argument("--position-mode", default="gap_preserving", choices=["gap_preserving"])
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--model-path", "--model", dest="model_path", default=DEFAULT_MODEL)
    parser.add_argument("--image-grid-size", type=int, default=DEFAULT_GRID)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--box-coordinate-mode", default="normalized_0_1", choices=["normalized_0_1", "normalized_0_1000", "absolute_pixels"])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "offline"), choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "vlm-pruning"))
    return parser.parse_args()


def run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    suffix = f"start{args.start_index}_limit{args.limit}_" if args.limit is not None else ""
    return f"{suffix}{args.dataset}_ug_prune_lite_tok{args.target_visual_tokens}_{args.precision}"


def load_student_bundle(torch: Any, checkpoint: Path, device: Any) -> tuple[Any, dict[str, Any] | None]:
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    config = UGPruneLiteModelConfig(**payload["model_config"])
    model = UGPruneLiteModel.build(torch, config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, (payload.get("train_config") or {}).get("feature_projection")


def load_student(torch: Any, checkpoint: Path, device: Any) -> Any:
    return load_student_bundle(torch, checkpoint, device)[0]


def prepare_lite_inputs(
    torch: Any,
    processor: Any,
    model: Any,
    device: Any,
    image: Image.Image,
    prompt: str,
    image_token_id: int,
    expected_tokens: int,
) -> dict[str, Any]:
    text_prompt = apply_chat_template(processor, prompt)
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    dtype = next(model.parameters()).dtype
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            value = value.to(device=device, dtype=dtype) if key == "pixel_values" else value.to(device=device)
        moved[key] = value
    image_features, vision_latency, projector_latency = selected_vision_features(model, moved["pixel_values"])
    input_ids = moved["input_ids"]
    attention_mask = moved.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    image_positions = torch.nonzero(input_ids[0] == int(image_token_id), as_tuple=False).flatten()
    if int(image_positions.numel()) != expected_tokens:
        raise BaselineError(f"Expected {expected_tokens} image tokens, got {int(image_positions.numel())}.")
    text_embeds = model.get_input_embeddings()(input_ids).to(dtype=dtype).clone()
    text_embeds[0, image_positions, :] = image_features[0].to(device=text_embeds.device, dtype=text_embeds.dtype)
    return {
        "text_prompt": text_prompt,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "inputs_embeds": text_embeds,
        "image_features": image_features,
        "image_positions": image_positions,
        "vision_latency_s": vision_latency,
        "projector_latency_s": projector_latency,
    }


def generate_lite_answer(
    torch: Any,
    processor: Any,
    model: Any,
    student: Any,
    selector: UGPruneLiteSelector,
    *,
    full_inputs: dict[str, Any],
    query_token_positions: list[int],
    feature_projection: dict[str, Any] | None,
    feature_projection_matrix: Any | None,
    sample_id: str,
    max_new_tokens: int,
) -> tuple[str, int, float, Any, dict[str, Any]]:
    query_positions = torch.tensor(query_token_positions, dtype=torch.long, device=full_inputs["inputs_embeds"].device)
    visual_features = full_inputs["image_features"][0]
    query_embedding = full_inputs["inputs_embeds"][0].index_select(0, query_positions).mean(dim=0)
    visual_features = maybe_apply_projection(visual_features, feature_projection_matrix, feature_projection)
    query_embedding = maybe_apply_projection(query_embedding, feature_projection_matrix, feature_projection)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    selector_start = time.perf_counter()
    with torch.inference_mode():
        outputs = student(visual_features.unsqueeze(0), query_embedding.unsqueeze(0))
        selection = selector.select(
            outputs["target_logits"][0],
            outputs["context_logits"][0],
            outputs["keep_logits"][0],
            sample_id=sample_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    selector_latency = time.perf_counter() - selector_start

    retained = selection.selected_indices
    retained_int = [int(index) for index in retained]
    retained_unique = len(set(retained_int))
    retained_duplicate_count = len(retained_int) - retained_unique
    retained_out_of_range_count = sum(1 for index in retained_int if index < 0 or index >= 576)
    keep = torch.tensor(retained, dtype=torch.long, device=full_inputs["input_ids"].device)
    keep_mask = torch.ones(full_inputs["input_ids"].shape[1], dtype=torch.bool, device=full_inputs["input_ids"].device)
    keep_mask[full_inputs["image_positions"]] = False
    keep_mask[full_inputs["image_positions"][keep]] = True
    selected_positions = torch.nonzero(keep_mask, as_tuple=False).flatten()
    compacted = {
        "inputs_embeds": full_inputs["inputs_embeds"][:, selected_positions, :],
        "attention_mask": full_inputs["attention_mask"][:, selected_positions],
        "selected_positions": selected_positions,
        "position_ids": selected_positions.unsqueeze(0).to(device=full_inputs["inputs_embeds"].device, dtype=torch.long),
    }
    raw_output, generated_count, prefill_latency, decode_latency = greedy_decode_from_embeds_position_mode(
        torch,
        processor,
        model,
        compacted,
        max_new_tokens,
        require_position_ids=True,
    )
    total_latency = (
        float(full_inputs["vision_latency_s"])
        + float(full_inputs["projector_latency_s"])
        + selector_latency
        + prefill_latency
        + decode_latency
    )
    diagnostics = {
        "vision_encoder_latency_s": full_inputs["vision_latency_s"],
        "projector_latency_s": full_inputs["projector_latency_s"],
        "student_selector_latency_s": selector_latency,
        "pruning_selection_latency_s": selector_latency,
        "llm_prefill_latency_s": prefill_latency,
        "prefill_latency_s": prefill_latency,
        "time_to_first_token_s": prefill_latency,
        "decode_latency_s": decode_latency,
        "total_inference_latency_s": total_latency,
        "original_visual_tokens": 576,
        "requested_keep_ratio": len(retained) / 576,
        "expected_retained_visual_tokens": 58,
        "actual_retained_visual_tokens": len(retained),
        "actual_llm_visual_tokens": len(retained),
        "retained_unique_visual_tokens": retained_unique,
        "retained_duplicate_count": retained_duplicate_count,
        "retained_out_of_range_count": retained_out_of_range_count,
        "retained_exact_budget": len(retained) == 58 and retained_duplicate_count == 0 and retained_out_of_range_count == 0,
        "original_input_sequence_length": int(full_inputs["input_ids"].shape[1]),
        "compacted_input_sequence_length": int(compacted["inputs_embeds"].shape[1]),
        "sequence_length_reduction": int(full_inputs["input_ids"].shape[1]) - int(compacted["inputs_embeds"].shape[1]),
        "retained_indices_checksum": indices_checksum(retained),
        "position_ids_mode": "gap_preserving",
        "position_ids_first": int(compacted["position_ids"][0, 0].item()),
        "position_ids_last": int(compacted["position_ids"][0, -1].item()),
    }
    return raw_output, generated_count, total_latency, selection, diagnostics


def add_selection_sanity_metrics(metrics: dict[str, Any], results: list[dict[str, Any]], expected_budget: int) -> None:
    if not results:
        metrics.update(
            {
                "selection/exact_budget_rate": 0.0,
                "selection/duplicate_count_mean": 0.0,
                "selection/out_of_range_count_mean": 0.0,
                "selection/unique_token_count_mean": 0.0,
                "selection/retained_token_count_mean": 0.0,
            }
        )
        return
    retained_counts = [float(row.get("actual_retained_visual_tokens", 0.0) or 0.0) for row in results]
    unique_counts = [float(row.get("retained_unique_visual_tokens", 0.0) or 0.0) for row in results]
    duplicate_counts = [float(row.get("retained_duplicate_count", 0.0) or 0.0) for row in results]
    out_of_range_counts = [float(row.get("retained_out_of_range_count", 0.0) or 0.0) for row in results]
    exact = [
        1.0
        if bool(row.get("retained_exact_budget"))
        or (
            int(row.get("actual_retained_visual_tokens", -1) or -1) == expected_budget
            and int(row.get("retained_duplicate_count", 0) or 0) == 0
            and int(row.get("retained_out_of_range_count", 0) or 0) == 0
        )
        else 0.0
        for row in results
    ]
    metrics["selection/exact_budget_rate"] = sum(exact) / float(len(exact))
    metrics["selection/duplicate_count_mean"] = sum(duplicate_counts) / float(len(duplicate_counts))
    metrics["selection/out_of_range_count_mean"] = sum(out_of_range_counts) / float(len(out_of_range_counts))
    metrics["selection/unique_token_count_mean"] = sum(unique_counts) / float(len(unique_counts))
    metrics["selection/retained_token_count_mean"] = sum(retained_counts) / float(len(retained_counts))


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if args.target_visual_tokens != 58:
        raise BaselineError("UG-Prune-Lite v1 is fixed to 58 visual tokens.")
    run_dir = args.output_root / run_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest, start_index=args.start_index, limit=args.limit)
    torch, processor, model, device, _env = load_model(args.model_path, args.precision)
    student, feature_projection = load_student_bundle(torch, args.checkpoint, device)
    feature_projection_mat = projection_matrix(torch, feature_projection, device=device, dtype=torch.float32) if feature_projection else None
    selector = UGPruneLiteSelector(torch, UGPruneLiteConfig())
    image_token_id = int(getattr(model.config, "image_token_index", 32000))
    tokenizer = getattr(processor, "tokenizer", processor)
    config = {
        "created_at_utc": utc_now(),
        "method/name": METHOD_NAME,
        "dataset/name": args.dataset,
        "dataset/manifest": str(args.manifest),
        "dataset/start_index": args.start_index,
        "limit": args.limit,
        "checkpoint": str(args.checkpoint),
        "feature_projection": feature_projection,
        "tokens/target_visual_tokens": args.target_visual_tokens,
        "tokens/keep_ratio": keep_ratio(args.target_visual_tokens, 576),
        "position_mode": args.position_mode,
        "smoke_only_not_final_eval": args.start_index >= 500,
    }
    write_json(run_dir / "config.json", config)
    wandb_run = None
    if args.wandb and args.wandb_mode != "disabled":
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                mode=args.wandb_mode,
                name=run_dir.name,
                group="ug-prune-lite",
                job_type="eval-smoke",
                config=config,
            )
        except Exception as exc:
            print(f"W&B init failed; continuing locally. {type(exc).__name__}: {exc}")
    results: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    with (run_dir / "per_sample_results.jsonl").open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(rows, start=1):
            image = Image.open(sample["image_path"]).convert("RGB")
            prompt = make_prompt(args, sample)
            query_text = str(sample["query"])
            full_inputs = prepare_lite_inputs(
                torch,
                processor,
                model,
                device,
                image,
                prompt,
                image_token_id,
                args.image_grid_size * args.image_grid_size,
            )
            resolution = resolve_query_token_positions(
                tokenizer,
                full_inputs["text_prompt"],
                query_text,
                input_ids=full_inputs["input_ids"][0].detach().cpu().tolist(),
                image_token_id=image_token_id,
            )
            if not resolution.token_positions:
                raise BaselineError(f"Query-token resolution failed for sample {sample['sample_id']}: {resolution.resolver_status}")
            raw_output, generated_count, latency, selection, diagnostics = generate_lite_answer(
                torch,
                processor,
                model,
                student,
                selector,
                full_inputs=full_inputs,
                query_token_positions=resolution.token_positions,
                feature_projection=feature_projection,
                feature_projection_matrix=feature_projection_mat,
                sample_id=str(sample["sample_id"]),
                max_new_tokens=args.max_new_tokens,
            )
            width = int(sample.get("width") or image.size[0])
            height = int(sample.get("height") or image.size[1])
            raw_box, clipped_box, parse_failure = parse_box(raw_output, width, height, args.box_coordinate_mode)
            iou = box_iou(clipped_box, sample["gt_box"]) if clipped_box is not None else 0.0
            row = {
                "sample_id": str(sample["sample_id"]),
                "dataset": args.dataset,
                "split": sample.get("split"),
                "image_id": sample.get("image_id"),
                "image_path": sample.get("image_path"),
                "method": METHOD_NAME,
                "query": query_text,
                "target_visual_tokens": args.target_visual_tokens,
                "raw_model_output": raw_output,
                "generated_token_count": generated_count,
                "latency_s": latency,
                "retained_indices": selection.selected_indices,
                "retained_indices_checksum": indices_checksum(selection.selected_indices),
                "selector_name": selection.method,
                "ug_query_token_status": resolution.resolver_status,
                "raw_parsed_box": raw_box,
                "predicted_box": clipped_box,
                "gt_box": sample.get("gt_box"),
                "box_iou": iou,
                "score": iou,
                "success_iou_0_5": clipped_box is not None and iou >= 0.5,
                "valid": clipped_box is not None,
                "invalid": clipped_box is None,
                "parse_failure_reason": parse_failure,
                **diagnostics,
            }
            results.append(row)
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            print(f"Processed {index}/{len(rows)} samples")
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2 if torch.cuda.is_available() else None
    metrics = aggregate(results, args, time.perf_counter() - run_start, peak_alloc, peak_reserved)
    add_selection_sanity_metrics(metrics, results, args.target_visual_tokens)
    metrics["method/name"] = METHOD_NAME
    metrics["ug_lite/student_selector_latency_s"] = metrics.get("efficiency/pruning_selection_latency_s")
    write_json(run_dir / "metrics.json", metrics)
    if wandb_run is not None:
        wandb_run.log(metrics)
        wandb_run.finish()
    print(f"Saved UG-Prune-Lite run outputs to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
