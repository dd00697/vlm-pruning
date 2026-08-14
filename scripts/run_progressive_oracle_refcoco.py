#!/usr/bin/env python
"""Run one registered progressive/oracle RefCOCO-val discovery cell."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
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
    DEFAULT_MODEL,
    BaselineError,
    apply_chat_template,
    box_iou,
    eos_token_ids,
    parse_box,
    prepare_compacted_inputs,
)
from run_official_baseline import load_manifest, load_model, load_sample_images, manifest_checksum  # noqa: E402
from vlm_pruning.artifacts import sha256_file, source_provenance  # noqa: E402
from vlm_pruning.features import resolve_query_token_positions  # noqa: E402
from vlm_pruning.llava_hooks import (  # noqa: E402
    OracleIndexArtifact,
    ProgressivePruneForwarder,
    PruningEvent,
    random_original_indices,
    uniform_original_indices,
    validate_schedule,
)


CAMPAIGN_ROOT = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814"
DEFAULT_MANIFEST = CAMPAIGN_ROOT / "manifests/refcoco_val_discovery1000.jsonl"
DEFAULT_OUTPUT_ROOT = CAMPAIGN_ROOT / "runs"
REGISTERED_PROMPT = (
    'Identify the target described by this referring expression: "{query}".\n'
    "Return exactly one bounding box in normalized 0-1 image coordinates as "
    "[x_min, y_min, x_max, y_max]. Do not include any other text."
)
ORIGINAL_VISUAL_TOKENS = 576
GRID = 24
MAX_NEW_TOKENS = 48


class ProgressiveRunError(RuntimeError):
    """Raised when a run violates its frozen contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def json_checksum(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def parse_schedule(value: str) -> tuple[PruningEvent, ...]:
    if not str(value).strip():
        return ()
    events: list[PruningEvent] = []
    for item in str(value).split(","):
        parts = item.strip().split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "Schedule must be comma-separated boundary:budget:selector events"
            )
        events.append(PruningEvent(int(parts[0]), int(parts[1]), parts[2]))
    try:
        return validate_schedule(events)
    except Exception as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def schedule_string(events: tuple[PruningEvent, ...]) -> str:
    return ",".join(f"{event.boundary}:{event.budget}:{event.selector}" for event in events)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--schedule", type=parse_schedule, default=())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pre-llm-selector", choices=["none", "oracle", "random", "uniform"], default="none")
    parser.add_argument("--pre-llm-budget", type=int)
    parser.add_argument("--oracle-teacher-layer", type=int, choices=[12, 20])
    parser.add_argument(
        "--oracle-index-input",
        type=Path,
        help="Replay a strict index-only oracle JSONL instead of executing the dense teacher.",
    )
    parser.add_argument("--precision", choices=["bf16"], default="bf16")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def validate_cli(args: argparse.Namespace) -> None:
    if args.pre_llm_selector == "none":
        if args.pre_llm_budget is not None or args.oracle_teacher_layer is not None:
            raise ProgressiveRunError("No pre-LLM selector cannot have a budget or oracle teacher")
    else:
        if args.pre_llm_budget is None or not 1 <= int(args.pre_llm_budget) <= ORIGINAL_VISUAL_TOKENS:
            raise ProgressiveRunError("Pre-LLM selection requires a budget in [1, 576]")
        if args.pre_llm_selector == "oracle" and args.oracle_teacher_layer not in {12, 20}:
            raise ProgressiveRunError("Oracle selection requires --oracle-teacher-layer 12 or 20")
        if args.pre_llm_selector != "oracle" and args.oracle_teacher_layer is not None:
            raise ProgressiveRunError("Only oracle selection may specify a teacher layer")
    if args.oracle_index_input is not None:
        if args.pre_llm_selector != "oracle":
            raise ProgressiveRunError("--oracle-index-input is valid only for oracle selection")
        if not args.oracle_index_input.is_file():
            raise ProgressiveRunError(f"Oracle index input does not exist: {args.oracle_index_input}")
    validate_schedule(args.schedule)
    if args.limit is not None and args.limit <= 0:
        raise ProgressiveRunError("--limit must be positive")


def make_prompt(sample: dict[str, Any]) -> str:
    return REGISTERED_PROMPT.format(query=str(sample["query"]))


def source_contract() -> dict[str, Any]:
    paths = [
        Path("scripts/run_progressive_oracle_refcoco.py"),
        Path("scripts/validate_progressive_oracle_run.py"),
        Path("scripts/run_baseline.py"),
        Path("vlm_pruning/artifacts.py"),
        Path("vlm_pruning/features/query_tokens.py"),
        Path("vlm_pruning/llava_hooks/mid_layer_pruning.py"),
        Path("vlm_pruning/llava_hooks/progressive_pruning.py"),
    ]
    provenance = source_provenance(REPO_ROOT, paths)
    # Campaign registry/report updates are intentionally tracked during a long
    # run and must not masquerade as source-code dirt.  Scientific cleanliness
    # is scoped to every executable/hash-bearing source file.
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *[str(path) for path in paths]],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    provenance["git_dirty"] = bool(status)
    provenance["code_paths_checked"] = [str(path) for path in paths]
    return provenance


def build_config(args: argparse.Namespace, rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    provenance = source_contract()
    if provenance["git_dirty"]:
        raise ProgressiveRunError("Discovery/smoke runs require a clean committed source revision")
    manifest_path = args.manifest.resolve()
    try:
        manifest_display = str(manifest_path.relative_to(REPO_ROOT))
    except ValueError:
        manifest_display = str(manifest_path)
    return {
        "schema_version": 1,
        "cell_id": str(args.cell_id),
        "run_name": run_dir.name,
        "created_at_utc": utc_now(),
        "source/git_commit": provenance["git_commit"],
        "source/git_dirty": provenance["git_dirty"],
        "source/sha256": provenance["source_sha256"],
        "model/name": str(args.model_path),
        "model/precision": "bf16",
        "model/decoding": "deterministic_greedy",
        "model/max_new_tokens": MAX_NEW_TOKENS,
        "dataset/name": "refcoco",
        "dataset/split": "val",
        "dataset/manifest": manifest_display,
        "dataset/manifest_checksum": manifest_checksum(args.manifest),
        "dataset/num_samples": len(rows),
        "prompt/template": REGISTERED_PROMPT,
        "evaluator/name": "box_iou_acc_at_0_5",
        "evaluator/coordinate_mode": "normalized_0_1",
        "original_visual_tokens": ORIGINAL_VISUAL_TOKENS,
        "grid": GRID,
        "schedule": [event.as_dict() for event in args.schedule],
        "schedule_string": schedule_string(args.schedule),
        "position_mode": "gap_preserving",
        "query_map": "current_boundary_hidden_query_patch_cosine_query_max",
        "query_tie_break": "descending_relevance_then_lower_surviving_original_raster_index",
        "seed": int(args.seed),
        "pre_llm_selector": str(args.pre_llm_selector),
        "pre_llm_budget": args.pre_llm_budget,
        "oracle_teacher_layer": args.oracle_teacher_layer,
        "oracle_teacher_indices_only": args.pre_llm_selector == "oracle",
        "oracle_candidate_average_excludes_teacher": args.pre_llm_selector == "oracle",
        "oracle_index_input": str(args.oracle_index_input.resolve()) if args.oracle_index_input else None,
        "oracle_index_input_sha256": sha256_file(args.oracle_index_input) if args.oracle_index_input else None,
        "limit": args.limit,
        "scope": "smoke" if args.limit is not None else "discovery",
        "run/output_dir": str(run_dir.relative_to(REPO_ROOT)),
    }


IMMUTABLE_CONFIG_KEYS = {
    "cell_id", "source/git_commit", "source/git_dirty", "source/sha256",
    "model/name", "model/precision", "model/decoding", "model/max_new_tokens",
    "dataset/name", "dataset/split", "dataset/manifest", "dataset/manifest_checksum",
    "dataset/num_samples", "prompt/template", "original_visual_tokens", "grid",
    "schedule", "position_mode", "query_map", "seed", "pre_llm_selector",
    "pre_llm_budget", "oracle_teacher_layer", "limit", "scope",
    "oracle_index_input", "oracle_index_input_sha256",
}


def compatible_existing_config(existing: dict[str, Any], proposed: dict[str, Any]) -> bool:
    return all(existing.get(key) == proposed.get(key) for key in IMMUTABLE_CONFIG_KEYS)


def load_partial_rows(path: Path, manifest_rows: list[dict[str, Any]], config_sha: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ProgressiveRunError(f"Blank partial row at line {line_number}")
            row = json.loads(line)
            rows.append(row)
    if len(rows) > len(manifest_rows):
        raise ProgressiveRunError("Partial artifact exceeds manifest length")
    expected_ids = [str(row["sample_id"]) for row in manifest_rows[: len(rows)]]
    actual_ids = [str(row.get("sample_id") or "") for row in rows]
    if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise ProgressiveRunError("Partial artifact is not an exact unique manifest prefix")
    if any(row.get("config_sha256") != config_sha for row in rows):
        raise ProgressiveRunError("Partial artifact contains another configuration/code epoch")
    return rows


def load_oracle_artifacts(path: Path) -> dict[str, OracleIndexArtifact]:
    artifacts: dict[str, OracleIndexArtifact] = {}
    if not path.exists():
        return artifacts
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ProgressiveRunError(f"Blank oracle index line {line_number}")
            artifact = OracleIndexArtifact.from_dict(json.loads(line))
            if artifact.sample_id in artifacts:
                raise ProgressiveRunError(f"Duplicate oracle artifact {artifact.sample_id}")
            artifacts[artifact.sample_id] = artifact
    return artifacts


def prepare_full_inputs(torch: Any, processor: Any, model: Any, device: Any, images: list[Image.Image], prompt: str):
    return prepare_compacted_inputs(
        torch,
        processor,
        model,
        device,
        images,
        prompt,
        list(range(ORIGINAL_VISUAL_TOKENS)),
        ORIGINAL_VISUAL_TOKENS,
    )


def query_resolution(processor: Any, text_prompt: str, query: str, input_ids: Any, image_token_id: int):
    tokenizer = getattr(processor, "tokenizer", processor)
    resolution = resolve_query_token_positions(
        tokenizer,
        text_prompt,
        query,
        input_ids=input_ids[0].detach().cpu().tolist(),
        image_token_id=image_token_id,
    )
    if not resolution.token_positions:
        raise ProgressiveRunError(f"Query-token resolution failed: {resolution.resolver_status}")
    return resolution


def build_result(
    *,
    sample: dict[str, Any],
    raw_output: str,
    generated_count: int,
    generation: Any,
    config_sha: str,
    source_commit: str,
    prep_diag: dict[str, Any],
    width: int,
    height: int,
    oracle_artifact: OracleIndexArtifact | None,
    initial_random_seed: int | None,
    wall_latency_s: float,
    peak_allocated_mb: float | None,
    peak_reserved_mb: float | None,
) -> dict[str, Any]:
    raw_box, predicted_box, parse_failure = parse_box(raw_output, width, height, "normalized_0_1")
    iou = box_iou(predicted_box, sample["gt_box"]) if predicted_box is not None else 0.0
    success = predicted_box is not None and iou >= 0.5
    diagnostics = jsonable(generation.diagnostics)
    result = {
        "sample_id": str(sample["sample_id"]),
        "original_sample_id": (sample.get("metadata") or {}).get("original_sample_id"),
        "source_row": (sample.get("metadata") or {}).get("source_row"),
        "dataset": "refcoco",
        "split": "val",
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
        "invalid": predicted_box is None,
        "box_iou": iou,
        "score": iou,
        "success_iou_0_5": success,
        "generated_token_count": generated_count,
        "config_sha256": config_sha,
        "source_git_commit": source_commit,
        "original_visual_tokens": ORIGINAL_VISUAL_TOKENS,
        "final_retained_indices": generation.final_retained_indices,
        "final_retained_indices_checksum": json_checksum(generation.final_retained_indices),
        "stage_traces": generation.stage_traces,
        "realized_visual_tokens_by_layer": diagnostics["progressive/realized_visual_tokens_by_layer"],
        "candidate_average_visual_tokens": diagnostics["candidate_average_visual_tokens"],
        "oracle_teacher_visual_token_layer_work": (
            oracle_artifact.teacher_visual_token_layer_work if oracle_artifact is not None else 0
        ),
        "oracle_teacher_layer": oracle_artifact.teacher_layer if oracle_artifact is not None else None,
        "oracle_ranking_checksum": oracle_artifact.ranking_checksum if oracle_artifact is not None else None,
        "oracle_privileged_non_deployable": oracle_artifact is not None,
        "pre_llm_random_derived_seed": initial_random_seed,
        "vision_encoder_latency_s": prep_diag["vision_encoder_latency_s"],
        "projector_latency_s": prep_diag["projector_latency_s"],
        "llm_prefill_latency_s": diagnostics["llm_prefill_latency_s"],
        "decode_latency_s": diagnostics["decode_latency_s"],
        "wall_inference_latency_s": wall_latency_s,
        "peak_allocated_mb": peak_allocated_mb,
        "peak_reserved_mb": peak_reserved_mb,
        **diagnostics,
    }
    return result


def run(args: argparse.Namespace) -> int:
    validate_cli(args)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    rows = load_manifest(args.manifest, args.limit)
    if any(str(row.get("dataset") or "").lower() != "refcoco" or str(row.get("split") or "").lower() != "val" for row in rows):
        raise ProgressiveRunError("Runner accepts RefCOCO validation rows only")
    run_name = args.run_name or f"{args.cell_id.lower()}_attempt1"
    run_dir = (args.output_root / run_name).resolve()
    completed_path = run_dir / "COMPLETED.json"
    if args.skip_existing and completed_path.is_file():
        from validate_progressive_oracle_run import validate_run

        report = validate_run(run_dir, args.manifest, expected_rows=len(rows), finalize=False)
        if report["status"] == "pass":
            print(f"Skipping independently validated completed run: {run_dir}")
            return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    proposed = build_config(args, rows, run_dir)
    config_path = run_dir / "config.json"
    if config_path.exists():
        if not args.resume:
            raise ProgressiveRunError(f"Run directory already initialized: {run_dir}")
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if not compatible_existing_config(existing_config, proposed):
            raise ProgressiveRunError("Existing run config is incompatible with requested resume")
        config = existing_config
    else:
        config = proposed
        atomic_write(config_path, canonical_json_bytes(config))
    config_sha = sha256_file(config_path)
    partial_path = run_dir / "predictions.partial.jsonl"
    predictions_path = run_dir / "predictions.jsonl"
    existing = load_partial_rows(partial_path, rows, config_sha) if args.resume else []
    if predictions_path.exists() and not completed_path.exists():
        raise ProgressiveRunError("Unvalidated final predictions already exist; use a fresh attempt directory")
    completed_ids = {str(row["sample_id"]) for row in existing}
    pending = [row for row in rows if str(row["sample_id"]) not in completed_ids]
    oracle_partial = run_dir / "oracle_indices.partial.jsonl"
    oracle_final = run_dir / "oracle_indices.jsonl"
    oracle_artifacts = load_oracle_artifacts(oracle_partial) if args.resume and args.pre_llm_selector == "oracle" else {}
    if args.oracle_index_input is not None:
        external = load_oracle_artifacts(args.oracle_index_input)
        overlap = set(oracle_artifacts).intersection(external)
        if overlap and any(oracle_artifacts[sample_id] != external[sample_id] for sample_id in overlap):
            raise ProgressiveRunError("Resume oracle indices differ from the registered replay input")
        oracle_artifacts.update(external)

    model_load_start = time.perf_counter()
    torch, processor, model, device, model_env = load_model(args.model_path, args.precision)
    model_load_latency = time.perf_counter() - model_load_start
    image_token_id = int(getattr(model.config, "image_token_index", 32000))
    forwarder = ProgressivePruneForwarder(
        torch,
        model,
        image_token_id=image_token_id,
        schedule=args.schedule,
        grid=GRID,
        initial_visual_tokens=ORIGINAL_VISUAL_TOKENS,
        diagnostics_mode="benchmark",
    )
    environment = {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "model_load_latency_s": model_load_latency,
        **model_env,
    }
    atomic_write(run_dir / "environment.json", canonical_json_bytes(jsonable(environment)))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mode = "a" if args.resume and partial_path.exists() else "w"
    oracle_mode = "a" if args.resume and oracle_partial.exists() else "w"
    oracle_handle = oracle_partial.open(oracle_mode, encoding="utf-8") if args.pre_llm_selector == "oracle" else None
    try:
        with partial_path.open(mode, encoding="utf-8") as prediction_handle:
            for pending_index, sample in enumerate(pending, 1):
                sample_id = str(sample["sample_id"])
                image_paths, images = load_sample_images(sample)
                if len(images) != 1:
                    raise ProgressiveRunError("Registered RefCOCO protocol requires exactly one image")
                width = int(sample.get("width") or images[0].size[0])
                height = int(sample.get("height") or images[0].size[1])
                prompt = make_prompt(sample)
                text_prompt = apply_chat_template(processor, prompt, 1)
                oracle_artifact: OracleIndexArtifact | None = None
                initial_indices: list[int] | None = None
                initial_selector: str | None = None
                initial_checksum: str | None = None
                initial_random_seed: int | None = None

                if args.pre_llm_selector == "oracle":
                    oracle_artifact = oracle_artifacts.get(sample_id)
                    artifact_was_generated = False
                    if oracle_artifact is None:
                        teacher_inputs, teacher_prep = prepare_full_inputs(
                            torch, processor, model, device, images, prompt
                        )
                        teacher_resolution = query_resolution(
                            processor,
                            text_prompt,
                            str(sample["query"]),
                            teacher_inputs["input_ids"],
                            image_token_id,
                        )
                        with torch.inference_mode():
                            oracle_artifact = forwarder.oracle_teacher_indices(
                                inputs_embeds=teacher_inputs["inputs_embeds"],
                                input_ids=teacher_inputs["input_ids"],
                                attention_mask=teacher_inputs["attention_mask"],
                                sample_id=sample_id,
                                teacher_layer=int(args.oracle_teacher_layer),
                                query_token_status=teacher_resolution.resolver_status,
                                query_original_positions=teacher_resolution.token_positions,
                            )
                        # Round-trip through JSON before candidate construction;
                        # only integer indices and scalar provenance survive.
                        oracle_artifact = OracleIndexArtifact.from_dict(
                            json.loads(json.dumps(oracle_artifact.as_dict()))
                        )
                        oracle_handle.write(json.dumps(oracle_artifact.as_dict(), sort_keys=True) + "\n")
                        oracle_handle.flush()
                        oracle_artifacts[sample_id] = oracle_artifact
                        artifact_was_generated = True
                        del teacher_inputs, teacher_prep, teacher_resolution
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    if not artifact_was_generated:
                        oracle_handle.write(json.dumps(oracle_artifact.as_dict(), sort_keys=True) + "\n")
                        oracle_handle.flush()
                    initial_indices = oracle_artifact.selected_indices(int(args.pre_llm_budget))
                    initial_selector = f"oracle_k{oracle_artifact.teacher_layer}"
                    initial_checksum = oracle_artifact.ranking_checksum
                elif args.pre_llm_selector == "random":
                    initial_indices, initial_random_seed = random_original_indices(
                        int(args.pre_llm_budget), sample_id=sample_id, seed=int(args.seed)
                    )
                    initial_selector = "random"
                    initial_checksum = json_checksum(initial_indices)
                elif args.pre_llm_selector == "uniform":
                    initial_indices = uniform_original_indices(int(args.pre_llm_budget))
                    initial_selector = "uniform"
                    initial_checksum = json_checksum(initial_indices)

                # This projector pass is fresh and is the only input to the
                # restarted candidate.  It is deliberately executed after all
                # dense teacher tensors have been destroyed.
                candidate_inputs, prep_diag = prepare_full_inputs(
                    torch, processor, model, device, images, prompt
                )
                candidate_resolution = query_resolution(
                    processor,
                    text_prompt,
                    str(sample["query"]),
                    candidate_inputs["input_ids"],
                    image_token_id,
                )
                wall_start = time.perf_counter()
                generation = forwarder.generate_with_segment_caches(
                    inputs_embeds=candidate_inputs["inputs_embeds"],
                    input_ids=candidate_inputs["input_ids"],
                    attention_mask=candidate_inputs["attention_mask"],
                    sample_id=sample_id,
                    seed=int(args.seed),
                    query_original_positions=candidate_resolution.token_positions,
                    max_new_tokens=MAX_NEW_TOKENS,
                    eos_ids=eos_token_ids(model, processor),
                    initial_retained_indices=initial_indices,
                    initial_selector=initial_selector,
                    initial_selection_checksum=initial_checksum,
                    initial_derived_random_seed=initial_random_seed,
                )
                wall_latency = time.perf_counter() - wall_start
                raw_output = processor.decode(generation.generated_ids, skip_special_tokens=True).strip()
                peak_allocated = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
                peak_reserved = torch.cuda.max_memory_reserved() / 1024**2 if torch.cuda.is_available() else None
                result = build_result(
                    sample=sample,
                    raw_output=raw_output,
                    generated_count=int(generation.generated_ids.numel()),
                    generation=generation,
                    config_sha=config_sha,
                    source_commit=str(config["source/git_commit"]),
                    prep_diag=prep_diag,
                    width=width,
                    height=height,
                    oracle_artifact=oracle_artifact,
                    initial_random_seed=initial_random_seed,
                    wall_latency_s=wall_latency,
                    peak_allocated_mb=peak_allocated,
                    peak_reserved_mb=peak_reserved,
                )
                prediction_handle.write(json.dumps(jsonable(result), ensure_ascii=True, sort_keys=True) + "\n")
                prediction_handle.flush()
                done = len(existing) + pending_index
                if done % 25 == 0 or done == len(rows):
                    print(f"{args.cell_id}: processed {done}/{len(rows)}", flush=True)
                for image in images:
                    image.close()
                del candidate_inputs, prep_diag, candidate_resolution, generation
    finally:
        if oracle_handle is not None:
            oracle_handle.close()

    os.replace(partial_path, predictions_path)
    if args.pre_llm_selector == "oracle":
        os.replace(oracle_partial, oracle_final)
    from validate_progressive_oracle_run import validate_and_finalize_run

    validation = validate_and_finalize_run(run_dir, args.manifest, expected_rows=len(rows))
    if validation["status"] != "pass":
        raise ProgressiveRunError("Independent final validation failed")
    print(json.dumps({"status": "complete", "run_dir": str(run_dir), "metrics": validation["metrics"]}, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted; compatible physical partials were preserved.", file=sys.stderr)
        return 130
    except (ProgressiveRunError, BaselineError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"UNEXPECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
