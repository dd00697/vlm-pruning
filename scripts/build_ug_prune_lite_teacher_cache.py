#!/usr/bin/env python
"""Build a UG-Prune-Lite teacher/feature cache."""

from __future__ import annotations

import argparse
import json
import os
import shutil
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

from run_baseline import DEFAULT_GRID, DEFAULT_MODEL, BaselineError, apply_chat_template, indices_checksum, prepare_compacted_inputs  # noqa: E402
from run_official_baseline import load_model, make_prompt, manifest_checksum, utc_now, write_json  # noqa: E402
from vlm_pruning.features import resolve_query_token_positions  # noqa: E402
from vlm_pruning.llava_hooks import MidLayerUGPruneForwarder, ug_gap58_config_for_prune_layer  # noqa: E402
from vlm_pruning.training.ug_prune_lite_dataset import load_exclude_ids, sample_ids_from_manifest  # noqa: E402
from vlm_pruning.training.ug_prune_lite_features import maybe_apply_projection, projection_matrix, random_projection_spec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True, choices=["grounding"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--exclude-manifest", type=Path, default=None)
    parser.add_argument("--exclude-limit", type=int, default=None)
    parser.add_argument("--exclude-sample-ids", action="append", type=Path, default=[])
    parser.add_argument("--model-path", "--model", dest="model_path", default=DEFAULT_MODEL)
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--box-coordinate-mode", default="normalized_0_1", choices=["normalized_0_1", "normalized_0_1000", "absolute_pixels"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-grid-size", type=int, default=DEFAULT_GRID)
    parser.add_argument("--target-visual-tokens", type=int, default=58)
    parser.add_argument("--teacher-diagnostics-mode", choices=["benchmark", "full"], default="benchmark")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--cache-size-mb-per-example", type=float, default=4.56)
    parser.add_argument("--seconds-per-example-estimate", type=float, default=0.37)
    parser.add_argument("--min-free-disk-gb", type=float, default=10.0)
    parser.add_argument("--feature-projection-dim", type=int, default=None)
    parser.add_argument("--feature-projection-seed", type=int, default=0)
    parser.add_argument("--save-teacher-score-maps", action="store_true")
    parser.add_argument("--teacher-score-map-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--teacher-score-map-key", default="fused")
    parser.add_argument("--teacher-role-score-maps", action="store_true")
    return parser.parse_args()


def query_text_for_sample(sample: dict[str, Any]) -> str:
    return str(sample["query"])


def read_rows_after_exclusion(manifest: Path, *, start_index: int, limit: int | None, exclude: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start_index or not line.strip():
                continue
            row = json.loads(line)
            if str(row["sample_id"]) in exclude:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def cache_plan(
    output_dir: Path,
    *,
    examples: int,
    cache_size_mb_per_example: float,
    seconds_per_example: float,
    min_free_disk_gb: float,
) -> dict[str, Any]:
    usage = shutil.disk_usage(output_dir.parent if output_dir.parent.exists() else Path("."))
    estimated_bytes = int(float(examples) * float(cache_size_mb_per_example) * 1024 * 1024)
    free_after_bytes = int(usage.free - estimated_bytes)
    proceed = free_after_bytes >= int(float(min_free_disk_gb) * 1024**3)
    return {
        "examples": int(examples),
        "output_dir": str(output_dir),
        "cache_size_mb_per_example": float(cache_size_mb_per_example),
        "estimated_cache_size_bytes": estimated_bytes,
        "estimated_cache_size_gib": estimated_bytes / float(1024**3),
        "seconds_per_example_estimate": float(seconds_per_example),
        "estimated_teacher_cache_time_s": float(examples) * float(seconds_per_example),
        "free_disk_bytes_before": int(usage.free),
        "free_disk_gib_before": usage.free / float(1024**3),
        "estimated_free_disk_gib_after": free_after_bytes / float(1024**3),
        "min_free_disk_gb": float(min_free_disk_gb),
        "proceed": bool(proceed),
    }


def validate_teacher_selection(selected: Any, *, sample_id: str, expected: int, original_tokens: int) -> None:
    indices = [int(index) for index in selected.selected_indices]
    if len(indices) != expected:
        raise BaselineError(f"Teacher selected {len(indices)} tokens for {sample_id}; expected {expected}.")
    if len(set(indices)) != expected:
        raise BaselineError(f"Teacher selected duplicate indices for {sample_id}.")
    if indices and (min(indices) < 0 or max(indices) >= original_tokens):
        raise BaselineError(f"Teacher selected out-of-range indices for {sample_id}.")
    roles = selected.role_labels or {}
    for role in ("target", "context", "anchor"):
        if not roles.get(role):
            raise BaselineError(f"Teacher selection for {sample_id} is missing role label: {role}.")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    exclude = load_exclude_ids(args.exclude_sample_ids)
    if args.exclude_manifest is not None:
        exclude |= sample_ids_from_manifest(args.exclude_manifest, limit=args.exclude_limit)
    rows = read_rows_after_exclusion(args.manifest, start_index=args.start_index, limit=args.limit, exclude=exclude)
    if not rows:
        raise BaselineError("No rows remain after applying exclude filters.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_size_mb = float(args.cache_size_mb_per_example)
    if args.feature_projection_dim is not None and cache_size_mb == 4.56:
        cache_size_mb = 4.56 * (float(args.feature_projection_dim) / 4096.0)
    plan = cache_plan(
        args.output_dir,
        examples=len(rows),
        cache_size_mb_per_example=cache_size_mb,
        seconds_per_example=args.seconds_per_example_estimate,
        min_free_disk_gb=args.min_free_disk_gb,
    )
    write_json(args.output_dir / "cache_plan.json", plan)
    if not plan["proceed"]:
        raise BaselineError(
            "Teacher cache plan would violate free-disk threshold: "
            f"estimated_after={plan['estimated_free_disk_gib_after']:.2f} GiB, "
            f"min_required={args.min_free_disk_gb:.2f} GiB."
        )
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    features_dir = args.output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at_utc": utc_now(),
        "method": "ug_prune_lite_teacher_cache",
        "teacher": "optimized_k12_ug_prune_gap58",
        "teacher_prune_after_layer": 12,
        "teacher_diagnostics_mode": args.teacher_diagnostics_mode,
        "model/name": args.model_path,
        "model/precision": args.precision,
        "dataset/name": args.dataset,
        "dataset/manifest": str(args.manifest),
        "dataset/manifest_checksum": manifest_checksum(args.manifest),
        "dataset/start_index": args.start_index,
        "dataset/requested_limit": args.limit,
        "dataset/cache_examples": len(rows),
        "dataset/exclude_manifest": str(args.exclude_manifest) if args.exclude_manifest else None,
        "dataset/exclude_limit": args.exclude_limit,
        "dataset/excluded_count": len(exclude),
        "dataset/reserved_eval_leakage_count": len({str(row["sample_id"]) for row in rows} & exclude),
        "smoke_only_not_final_training": args.manifest.name == "refcoco_val.jsonl" and args.start_index >= 500,
        "tokens/original_visual_tokens": args.image_grid_size * args.image_grid_size,
        "tokens/target_visual_tokens": args.target_visual_tokens,
        "feature_projection": None,
        "teacher_score_maps/enabled": bool(args.save_teacher_score_maps),
        "teacher_score_maps/key": args.teacher_score_map_key if args.save_teacher_score_maps else None,
        "teacher_score_maps/dtype": args.teacher_score_map_dtype if args.save_teacher_score_maps else None,
        "teacher_score_maps/normalization": "zscore",
        "teacher_score_maps/source_map_family": "unrolled" if args.save_teacher_score_maps else None,
        "teacher_score_maps/role_score_maps": bool(args.teacher_role_score_maps),
    }
    write_json(args.output_dir / "metadata.json", metadata)

    torch, processor, model, device, _env = load_model(args.model_path, args.precision)
    projection_spec = None
    projection = None
    if args.feature_projection_dim is not None:
        input_embedding = model.get_input_embeddings()
        text_config = getattr(model.config, "text_config", model.config)
        input_dim = int(getattr(input_embedding, "embedding_dim", 0) or getattr(text_config, "hidden_size", 0))
        projection_spec = random_projection_spec(
            input_dim=input_dim,
            output_dim=int(args.feature_projection_dim),
            seed=int(args.feature_projection_seed),
        )
        projection = projection_matrix(torch, projection_spec, device=device, dtype=torch.float32)
        metadata["feature_projection"] = projection_spec
        write_json(args.output_dir / "metadata.json", metadata)

    image_token_id = int(getattr(model.config, "image_token_index", 32000))
    forwarder = MidLayerUGPruneForwarder(
        torch,
        model,
        image_token_id=image_token_id,
        grid=args.image_grid_size,
        prune_after_layer=12,
        target_visual_tokens=args.target_visual_tokens,
        selector_config=ug_gap58_config_for_prune_layer(12),
        diagnostics_mode=args.teacher_diagnostics_mode,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    examples_path = args.output_dir / "examples.jsonl"
    original_tokens = args.image_grid_size * args.image_grid_size
    start = time.perf_counter()
    role_counts: dict[str, list[int]] = {"target": [], "context": [], "anchor": []}
    checksums: list[str] = []
    with examples_path.open("w", encoding="utf-8") as handle:
        for idx, sample in enumerate(rows, start=1):
            sample_id = str(sample["sample_id"])
            image = Image.open(sample["image_path"]).convert("RGB")
            prompt = make_prompt(args, sample)
            text_prompt = apply_chat_template(processor, prompt)
            query_text = query_text_for_sample(sample)
            full_inputs, _prep = prepare_compacted_inputs(
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

            with torch.inference_mode():
                position_ids = forwarder.position_ids_for(int(full_inputs["inputs_embeds"].shape[1]), full_inputs["inputs_embeds"].device)
                _early, hidden_states = forwarder.run_decoder_layers(
                    full_inputs["inputs_embeds"],
                    attention_mask=full_inputs["attention_mask"],
                    position_ids=position_ids,
                    start_layer=0,
                    end_layer=12,
                    collect_hidden_states=True,
                )
                if hidden_states is None:
                    raise BaselineError("Teacher hidden states were not captured.")
                pipeline = forwarder.select_from_hidden_states(
                    hidden_states=hidden_states,
                    input_ids=full_inputs["input_ids"],
                    sample_id=sample_id,
                    original_tokens=original_tokens,
                    seed=args.seed,
                    query_text=query_text,
                    query_token_status=resolution.resolver_status,
                    query_token_positions=resolution.token_positions,
                )
                selected = pipeline.selection
                validate_teacher_selection(
                    selected,
                    sample_id=sample_id,
                    expected=args.target_visual_tokens,
                    original_tokens=original_tokens,
                )

            image_positions = torch.nonzero(full_inputs["input_ids"][0] == image_token_id, as_tuple=False).flatten()
            visual_features = full_inputs["inputs_embeds"][0, image_positions, :].detach()
            query_positions = torch.tensor(resolution.token_positions, dtype=torch.long, device=full_inputs["inputs_embeds"].device)
            query_embedding = full_inputs["inputs_embeds"][0].index_select(0, query_positions).mean(dim=0).detach()
            visual_features = maybe_apply_projection(visual_features, projection, projection_spec).detach().cpu().to(torch.float16)
            query_embedding = maybe_apply_projection(query_embedding, projection, projection_spec).detach().cpu().to(torch.float16)
            teacher_score_map = None
            if args.save_teacher_score_maps:
                if pipeline.score_map is None:
                    raise BaselineError(f"Teacher score map was not available for sample {sample_id}.")
                if hasattr(pipeline.score_map, "detach"):
                    teacher_score_map = pipeline.score_map.detach().float().cpu().reshape(-1)
                else:
                    teacher_score_map = torch.as_tensor(pipeline.score_map, dtype=torch.float32).reshape(-1)
                if int(teacher_score_map.numel()) != original_tokens:
                    raise BaselineError(
                        f"Teacher score map for {sample_id} has {int(teacher_score_map.numel())} values; "
                        f"expected {original_tokens}."
                    )
                if args.teacher_score_map_dtype == "float16":
                    teacher_score_map = teacher_score_map.to(torch.float16)
            feature_rel = Path("features") / f"{sample_id}.pt"
            feature_payload = {
                "visual_features": visual_features,
                "query_embedding": query_embedding,
                "query_token_positions": [int(pos) for pos in resolution.token_positions],
                "query_token_status": resolution.resolver_status,
            }
            if teacher_score_map is not None:
                feature_payload["teacher_score_map"] = teacher_score_map
                feature_payload["teacher_score_map_key"] = args.teacher_score_map_key
            torch.save(feature_payload, args.output_dir / feature_rel)
            row = {
                "sample_id": sample_id,
                "image_id": sample.get("image_id"),
                "image_path": sample.get("image_path"),
                "query": query_text,
                "gt_box": sample.get("gt_box"),
                "width": int(sample.get("width") or image.size[0]),
                "height": int(sample.get("height") or image.size[1]),
                "selected_indices": selected.selected_indices,
                "role_labels": selected.role_labels,
                "selected_checksum": indices_checksum(selected.selected_indices),
                "feature_path": str(feature_rel.as_posix()),
                "teacher_score_map_key": args.teacher_score_map_key if teacher_score_map is not None else None,
                "teacher_method": selected.method,
                "teacher_metadata": {
                    "map_key": selected.metadata.get("map_key"),
                    "map_source": selected.metadata.get("map_source"),
                    "probe_layers": selected.metadata.get("probe_layers"),
                    "fusion_mode": selected.metadata.get("fusion_mode"),
                    "budget_split": selected.metadata.get("budget_split"),
                },
            }
            checksum = row["selected_checksum"]
            checksums.append(checksum)
            for role in role_counts:
                role_counts[role].append(len(row["role_labels"].get(role) or []))
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            print(f"Cached {idx}/{len(rows)} teacher examples")
    metadata["runtime/cache_wall_time_s"] = time.perf_counter() - start
    metadata["runtime/cache_disk_size_bytes"] = directory_size_bytes(args.output_dir)
    metadata["teacher/selected_checksum_unique_count"] = len(set(checksums))
    metadata["teacher/selected_checksum_count"] = len(checksums)
    for role, counts in role_counts.items():
        metadata[f"teacher/{role}_count_mean"] = sum(counts) / float(len(counts)) if counts else 0.0
        metadata[f"teacher/{role}_count_min"] = min(counts) if counts else 0
        metadata[f"teacher/{role}_count_max"] = max(counts) if counts else 0
    write_json(args.output_dir / "metadata.json", metadata)
    print(f"Saved UG-Prune-Lite teacher cache to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
