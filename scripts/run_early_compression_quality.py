#!/usr/bin/env python
"""Run one registered Phase-5 pre-projector pooling quality cell."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_baseline import apply_chat_template, box_iou, eos_token_ids, parse_box  # noqa: E402
from run_official_baseline import load_manifest, load_model, load_sample_images  # noqa: E402
from vlm_pruning.artifacts import sha256_file, source_provenance  # noqa: E402
from vlm_pruning.llava_hooks import (  # noqa: E402
    ContiguousPoolForwarder,
    EarlyCompressionError,
    prepare_pooled_inputs,
)


CAMPAIGN_ROOT = REPO_ROOT / "experiments/hardware_aware_zero_tail_refcoco_val_20260816"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
)
DEFAULT_OUTPUT_ROOT = CAMPAIGN_ROOT / "early_compression" / "quality"
EXPECTED_MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
MODEL = "llava-hf/llava-1.5-7b-hf"
MAX_NEW_TOKENS = 48
REGISTERED_PROMPT = (
    'Identify the target described by this referring expression: "{query}".\n'
    "Return exactly one bounding box in normalized 0-1 image coordinates as "
    "[x_min, y_min, x_max, y_max]. Do not include any other text."
)
CELL_GRIDS = {"E256_POOL": 16, "E196_POOL": 14}
SOURCE_PATHS = [
    Path("scripts/run_early_compression_quality.py"),
    Path("scripts/validate_early_compression_quality.py"),
    Path("scripts/run_baseline.py"),
    Path("scripts/run_official_baseline.py"),
    Path("vlm_pruning/artifacts.py"),
    Path("vlm_pruning/efficiency.py"),
    Path("vlm_pruning/llava_hooks/early_compression.py"),
]


class EarlyQualityError(RuntimeError):
    """Raised when a quality run violates its registered contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def source_contract() -> dict[str, Any]:
    provenance = source_provenance(REPO_ROOT, SOURCE_PATHS)
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *[str(path) for path in SOURCE_PATHS]],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    provenance["git_dirty"] = bool(status)
    provenance["code_paths_checked"] = [str(path) for path in SOURCE_PATHS]
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", choices=sorted(CELL_GRIDS), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace, rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    source = source_contract()
    if source["git_dirty"]:
        raise EarlyQualityError("Early-compression quality runs require clean committed source")
    manifest_hash = sha256_file(args.manifest)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise EarlyQualityError(f"Frozen manifest hash changed: {manifest_hash}")
    semantic = {
        "cell_id": args.cell_id,
        "input_grid": 24,
        "output_grid": CELL_GRIDS[args.cell_id],
        "pooling": "exact_area_weighted_rectangular_overlap",
        "anchors": "nearest_original_cell_lower_raster_tie",
        "position_mode": "gap_preserving_original_prompt_positions",
        "decoder_execution": "single_contiguous_cache_no_internal_compaction",
        "manifest_sha256": manifest_hash,
        "rows": len(rows),
        "seed": 0,
    }
    return {
        "schema_version": 1,
        "scope": "quality_smoke" if args.limit is not None else "quality_full",
        "cell_id": args.cell_id,
        "created_at_utc": utc_now(),
        "source/git_commit": source["git_commit"],
        "source/git_dirty": False,
        "source/sha256": source["source_sha256"],
        "source/code_paths_checked": source["code_paths_checked"],
        "model/name": MODEL,
        "model/precision": "bf16",
        "model/decoding": "deterministic_greedy",
        "model/max_new_tokens": MAX_NEW_TOKENS,
        "dataset/name": "refcoco",
        "dataset/split": "val",
        "dataset/manifest": str(args.manifest.resolve().relative_to(REPO_ROOT)),
        "dataset/manifest_sha256": manifest_hash,
        "dataset/rows": len(rows),
        "prompt/template": REGISTERED_PROMPT,
        "input_grid": 24,
        "output_grid": CELL_GRIDS[args.cell_id],
        "pooled_visual_tokens": CELL_GRIDS[args.cell_id] ** 2,
        "pooling/mechanism": "exact_area_weighted_rectangular_overlap",
        "pooling/mass_contract": "all_576_patches_contribute_total_mass_576",
        "pooling/anchor_rule": "nearest_original_cell_lower_raster_tie",
        "position_mode": "gap_preserving_original_prompt_positions",
        "decoder_execution": "single_contiguous_cache_no_internal_compaction",
        "query_selector": "none",
        "seed": 0,
        "command/semantic": semantic,
        "command/sha256": hashlib.sha256(
            json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "run/output_dir": str(run_dir.relative_to(REPO_ROOT)),
    }


IMMUTABLE_KEYS = {
    "scope",
    "cell_id",
    "source/git_commit",
    "source/git_dirty",
    "source/sha256",
    "model/name",
    "model/precision",
    "model/decoding",
    "model/max_new_tokens",
    "dataset/name",
    "dataset/split",
    "dataset/manifest",
    "dataset/manifest_sha256",
    "dataset/rows",
    "prompt/template",
    "input_grid",
    "output_grid",
    "pooled_visual_tokens",
    "pooling/mechanism",
    "pooling/mass_contract",
    "pooling/anchor_rule",
    "position_mode",
    "decoder_execution",
    "query_selector",
    "seed",
    "command/semantic",
    "command/sha256",
}


def compatible(existing: dict[str, Any], proposed: dict[str, Any]) -> bool:
    return all(existing.get(key) == proposed.get(key) for key in IMMUTABLE_KEYS)


def load_prefix(path: Path, rows: list[dict[str, Any]], config_sha: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(existing) > len(rows):
        raise EarlyQualityError("Partial output exceeds manifest")
    expected_ids = [str(row["sample_id"]) for row in rows[: len(existing)]]
    actual_ids = [str(row.get("sample_id") or "") for row in existing]
    if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise EarlyQualityError("Partial output is not the exact unique manifest prefix")
    if any(row.get("config_sha256") != config_sha for row in existing):
        raise EarlyQualityError("Partial output belongs to another configuration")
    return existing


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = quantile * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timing_fields = [
        "vision_encoder_latency_s",
        "pooling_latency_s",
        "projector_latency_s",
        "input_assembly_latency_s",
        "preprocessing_other_latency_s",
        "llm_prefill_latency_s",
        "time_to_first_token_s",
        "decode_latency_s",
        "end_to_end_latency_s",
    ]
    metrics: dict[str, Any] = {
        "samples": len(rows),
        "valid_count": sum(bool(row["valid"]) for row in rows),
        "parse_rate": sum(bool(row["valid"]) for row in rows) / len(rows),
        "correct_count": sum(bool(row["success_iou_0_5"]) for row in rows),
        "acc_at_0_5": sum(bool(row["success_iou_0_5"]) for row in rows) / len(rows),
        "mean_iou": sum(float(row["box_iou"]) for row in rows) / len(rows),
        "pooled_visual_tokens": int(rows[0]["pooled_visual_tokens"]),
        "visual_token_layer_work": int(rows[0]["visual_token_layer_work"]),
        "average_visual_tokens": float(rows[0]["average_visual_tokens"]),
        "prompt_kv_total_bytes_mean": sum(int(row["prompt_kv_total_bytes"]) for row in rows) / len(rows),
        "final_kv_total_bytes_mean": sum(int(row["final_kv_total_bytes"]) for row in rows) / len(rows),
        "peak_allocated_gpu_bytes": max(int(row["peak_allocated_gpu_bytes"]) for row in rows),
        "peak_reserved_gpu_bytes": max(int(row["peak_reserved_gpu_bytes"]) for row in rows),
    }
    for field in timing_fields:
        values = [float(row[field]) for row in rows]
        metrics[field] = {
            "mean": sum(values) / len(values),
            "median": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
        }
    metrics["quality_feasible"] = (
        metrics["acc_at_0_5"] >= 0.429
        and metrics["mean_iou"] >= 0.4165978636
        and metrics["parse_rate"] == 1.0
    )
    return metrics


def run(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit <= 0:
        raise EarlyQualityError("--limit must be positive")
    rows = load_manifest(args.manifest, args.limit)
    if not rows or (args.limit is None and len(rows) != 1000):
        raise EarlyQualityError("Full quality cells require exactly 1,000 rows")
    if any(str(row.get("dataset")).lower() != "refcoco" or str(row.get("split")).lower() != "val" for row in rows):
        raise EarlyQualityError("Runner accepts only RefCOCO validation rows")
    run_name = args.run_name or f"{args.cell_id.lower()}_attempt1"
    run_dir = (args.output_root / run_name).resolve()
    completed_path = run_dir / "COMPLETED.json"
    if args.skip_existing and completed_path.is_file():
        from validate_early_compression_quality import validate_run

        validate_run(run_dir, args.manifest, expected_rows=len(rows))
        print(f"Skipping independently validated run: {run_dir}")
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    proposed = build_config(args, rows, run_dir)
    config_path = run_dir / "config.json"
    if config_path.exists():
        if not args.resume:
            raise EarlyQualityError(f"Run directory already exists: {run_dir}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not compatible(config, proposed):
            raise EarlyQualityError("Existing run is incompatible with requested resume")
    else:
        config = proposed
        atomic_write(config_path, canonical_bytes(config))
    config_sha = sha256_file(config_path)
    partial_path = run_dir / "predictions.partial.jsonl"
    predictions_path = run_dir / "predictions.jsonl"
    existing = load_prefix(partial_path, rows, config_sha) if args.resume else []
    if predictions_path.exists() and not completed_path.exists():
        raise EarlyQualityError("Unvalidated final predictions require a fresh attempt")

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    load_started = time.perf_counter()
    torch, processor, model, device, model_env = load_model(MODEL, "bf16")
    model_load_latency = time.perf_counter() - load_started
    forwarder = ContiguousPoolForwarder(torch, model)
    environment = {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "model_load_latency_s": model_load_latency,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        **model_env,
    }
    atomic_write(run_dir / "environment.json", canonical_bytes(jsonable(environment)))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mode = "a" if args.resume and partial_path.exists() else "w"
    completed_ids = {str(row["sample_id"]) for row in existing}
    pending = [row for row in rows if str(row["sample_id"]) not in completed_ids]
    with partial_path.open(mode, encoding="utf-8") as handle:
        for pending_index, sample in enumerate(pending, 1):
            _, images = load_sample_images(sample)
            if len(images) != 1:
                raise EarlyQualityError("Registered protocol requires one image")
            width = int(sample.get("width") or images[0].size[0])
            height = int(sample.get("height") or images[0].size[1])
            prompt = REGISTERED_PROMPT.format(query=str(sample["query"]))
            text_prompt = apply_chat_template(processor, prompt, 1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            end_started = time.perf_counter()
            prep_started = time.perf_counter()
            pooled_inputs, prep = prepare_pooled_inputs(
                torch,
                processor,
                model,
                device,
                images,
                text_prompt,
                output_grid=int(config["output_grid"]),
            )
            prep_total = time.perf_counter() - prep_started
            generated_ids, generation = forwarder.generate_natural(
                pooled_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_ids=eos_token_ids(model, processor),
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_to_end = time.perf_counter() - end_started
            raw_output = processor.decode(generated_ids, skip_special_tokens=True).strip()
            raw_box, predicted_box, parse_failure = parse_box(
                raw_output, width, height, "normalized_0_1"
            )
            iou = box_iou(predicted_box, sample["gt_box"]) if predicted_box is not None else 0.0
            generated_count = int(generated_ids.numel())
            preprocessing_other = max(
                0.0,
                prep_total
                - float(prep["vision_encoder_latency_s"])
                - float(prep["pooling_latency_s"])
                - float(prep["projector_latency_s"])
                - float(prep["input_assembly_latency_s"]),
            )
            compacted_prompt = int(prep["compacted_prompt_length"])
            result = {
                "sample_id": str(sample["sample_id"]),
                "query": sample["query"],
                "image_id": sample["image_id"],
                "image_path": sample["image_path"],
                "width": width,
                "height": height,
                "gt_box": sample["gt_box"],
                "raw_model_output": raw_output,
                "raw_parsed_box": raw_box,
                "predicted_box": predicted_box,
                "parse_failure_reason": parse_failure,
                "valid": predicted_box is not None,
                "box_iou": iou,
                "success_iou_0_5": predicted_box is not None and iou >= 0.5,
                "generated_token_count": generated_count,
                "config_sha256": config_sha,
                "source_git_commit": config["source/git_commit"],
                "cell_id": config["cell_id"],
                "pooled_visual_tokens": int(config["pooled_visual_tokens"]),
                "realized_visual_tokens_by_layer": [int(config["pooled_visual_tokens"])] * 32,
                "visual_token_layer_work": int(config["pooled_visual_tokens"]) * 32,
                "average_visual_tokens": float(config["pooled_visual_tokens"]),
                "full_sequence_lengths_by_layer": [compacted_prompt] * 32,
                "sum_layer_sequence_lengths": compacted_prompt * 32,
                "sum_squared_layer_sequence_lengths": compacted_prompt * compacted_prompt * 32,
                "vision_encoder_latency_s": float(prep["vision_encoder_latency_s"]),
                "pooling_latency_s": float(prep["pooling_latency_s"]),
                "projector_latency_s": float(prep["projector_latency_s"]),
                "input_assembly_latency_s": float(prep["input_assembly_latency_s"]),
                "preprocessing_other_latency_s": preprocessing_other,
                "llm_prefill_latency_s": float(generation["llm_prefill_latency_s"]),
                "time_to_first_token_s": prep_total + float(generation["llm_prefill_latency_s"]),
                "decode_latency_s": float(generation["decode_latency_s"]),
                "decode_time_per_generated_token_s": float(generation["decode_latency_s"])
                / max(generated_count - 1, 1),
                "end_to_end_latency_s": end_to_end,
                "query_scoring_latency_s": 0.0,
                "selector_latency_s": 0.0,
                "gather_latency_s": 0.0,
                "merge_latency_s": 0.0,
                "route_gate_latency_s": 0.0,
                "prompt_kv_total_bytes": int(generation["cache/prompt_total_bytes"]),
                "final_kv_total_bytes": int(generation["cache/final_total_bytes"]),
                "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
                "pooling_evidence": prep,
                "cache_evidence": generation,
                "physical_pool_operations": 1,
                "physical_decoder_compaction_boundaries": 0,
            }
            handle.write(json.dumps(jsonable(result), sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush()
            done = len(existing) + pending_index
            if done % 25 == 0 or done == len(rows):
                print(f"EARLY_QUALITY {args.cell_id}: {done}/{len(rows)}", flush=True)
            for image in images:
                image.close()
            del pooled_inputs, prep, generated_ids, generation
            gc.collect()
    os.replace(partial_path, predictions_path)
    measured = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    atomic_write(run_dir / "metrics.json", canonical_bytes(aggregate(measured)))
    from validate_early_compression_quality import validate_and_finalize

    report = validate_and_finalize(run_dir, args.manifest, expected_rows=len(rows))
    print(json.dumps({"status": "complete", "metrics": report["metrics"]}, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted; exact compatible ordered prefix was preserved.", file=sys.stderr)
        return 130
    except (EarlyQualityError, EarlyCompressionError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"UNEXPECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
