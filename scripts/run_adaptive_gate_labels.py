#!/usr/bin/env python
"""Generate paired K12/M448 RefCOCO-train labels and frozen K2 gate features."""

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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_baseline import apply_chat_template, eos_token_ids  # noqa: E402
from run_official_baseline import load_manifest, load_model  # noqa: E402
from run_progressive_oracle_refcoco import (  # noqa: E402
    MAX_NEW_TOKENS,
    ORIGINAL_VISUAL_TOKENS,
    REGISTERED_PROMPT,
    build_result,
    prepare_full_inputs,
    query_resolution,
)
from vlm_pruning.artifacts import sha256_file, source_provenance  # noqa: E402
from vlm_pruning.llava_hooks.progressive_pruning import ProgressivePruneForwarder  # noqa: E402
from vlm_pruning.llava_hooks.route_gate import (  # noqa: E402
    AGGRESSIVE_ROUTE,
    AGGRESSIVE_SCHEDULE,
    FEATURE_DIMENSION,
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_SHA256,
    GATE_BOUNDARY,
    SAFE_ROUTE,
    SAFE_SCHEDULE,
    DynamicRouteForwarder,
)


CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
DEFAULT_MANIFEST = CAMPAIGN_ROOT / "gate_data/refcoco_train_gate3000.jsonl"
MANIFEST_AUDIT = CAMPAIGN_ROOT / "gate_data/MANIFEST_AUDIT.json"
DEFAULT_OUTPUT_ROOT = CAMPAIGN_ROOT / "gate_labels"
EXPECTED_MANIFEST_SHA256 = "8f2b567f27ccc05d1fef7b8368f1de7c454127e224c8f25f24944444d2b1d18f"
MODEL = "llava-hf/llava-1.5-7b-hf"


class GateLabelError(RuntimeError):
    """Raised when paired gate-label generation violates its frozen contract."""


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


def json_checksum(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def jsonable(value: Any) -> Any:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def source_contract() -> dict[str, Any]:
    paths = [
        Path("scripts/run_adaptive_gate_labels.py"),
        Path("scripts/validate_adaptive_gate_labels.py"),
        Path("scripts/run_progressive_oracle_refcoco.py"),
        Path("scripts/run_baseline.py"),
        Path("scripts/run_official_baseline.py"),
        Path("vlm_pruning/artifacts.py"),
        Path("vlm_pruning/features/query_tokens.py"),
        Path("vlm_pruning/llava_hooks/mid_layer_pruning.py"),
        Path("vlm_pruning/llava_hooks/progressive_pruning.py"),
        Path("vlm_pruning/llava_hooks/route_gate.py"),
    ]
    provenance = source_provenance(REPO_ROOT, paths)
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *[str(path) for path in paths]],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    provenance["git_dirty"] = bool(status)
    provenance["code_paths_checked"] = [str(path) for path in paths]
    return provenance


def semantic_command(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "program": "scripts/run_adaptive_gate_labels.py",
        "manifest": str(args.manifest.resolve().relative_to(REPO_ROOT)),
        "output_root": str(args.output_root.resolve().relative_to(REPO_ROOT)),
        "run_name": args.run_name,
        "limit": args.limit,
    }


def build_config(args: argparse.Namespace, rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    source = source_contract()
    if source["git_dirty"]:
        raise GateLabelError("Gate-label generation requires clean committed executable source")
    manifest_hash = sha256_file(args.manifest)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise GateLabelError("Gate-label generation requires the frozen 3,000-row train manifest")
    if not MANIFEST_AUDIT.is_file():
        raise GateLabelError("Gate train-manifest audit is missing")
    if args.limit is None and len(rows) != 3000:
        raise GateLabelError("A complete gate-label run requires exactly 3,000 train records")
    command = semantic_command(args)
    return {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "run_name": args.run_name,
        "run/output_dir": str(run_dir.relative_to(REPO_ROOT)),
        "scope": "gate_label_smoke" if args.limit is not None else "gate_labels",
        "limit": args.limit,
        "dataset/name": "refcoco",
        "dataset/split": "train",
        "dataset/manifest": str(args.manifest.resolve().relative_to(REPO_ROOT)),
        "dataset/manifest_checksum": manifest_hash,
        "dataset/manifest_audit_sha256": sha256_file(MANIFEST_AUDIT),
        "dataset/num_samples": len(rows),
        "model/name": MODEL,
        "model/precision": "bf16",
        "model/decoding": "deterministic_greedy",
        "model/max_new_tokens": MAX_NEW_TOKENS,
        "prompt/template": REGISTERED_PROMPT,
        "evaluator/coordinate_mode": "normalized_0_1",
        "original_visual_tokens": ORIGINAL_VISUAL_TOKENS,
        "position_mode": "gap_preserving",
        "seed": 0,
        "gate/boundary": GATE_BOUNDARY,
        "gate/feature_schema": FEATURE_SCHEMA,
        "gate/feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "gate/feature_dimension": FEATURE_DIMENSION,
        "gate/shared_feature_prefix_executions_per_record": 1,
        "gate/physical_route_executions_per_record": 2,
        "safe_route": {
            "name": SAFE_ROUTE,
            "schedule": [event.as_dict() for event in SAFE_SCHEDULE],
        },
        "aggressive_route": {
            "name": AGGRESSIVE_ROUTE,
            "schedule": [event.as_dict() for event in AGGRESSIVE_SCHEDULE],
        },
        "command/semantic": command,
        "command/sha256": json_checksum(command),
        "source/git_commit": source["git_commit"],
        "source/git_dirty": source["git_dirty"],
        "source/sha256": source["source_sha256"],
    }


def compatible(existing: dict[str, Any], proposed: dict[str, Any]) -> bool:
    ignored = {"created_at_utc"}
    return {key: value for key, value in existing.items() if key not in ignored} == {
        key: value for key, value in proposed.items() if key not in ignored
    }


def load_partial(path: Path, rows: list[dict[str, Any]], config_sha: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    parsed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(parsed) > len(rows):
        raise GateLabelError("Gate-label partial contains too many records")
    for index, row in enumerate(parsed):
        if row.get("sample_id") != rows[index]["sample_id"] or row.get("config_sha256") != config_sha:
            raise GateLabelError("Gate-label partial is not an exact compatible manifest prefix")
    return parsed


def route_result(
    *,
    sample: dict[str, Any],
    raw_output: str,
    generation: Any,
    config_sha: str,
    source_commit: str,
    prep_diag: dict[str, Any],
    wall_latency_s: float,
    peak_allocated_mb: float,
    peak_reserved_mb: float,
) -> dict[str, Any]:
    result = build_result(
        sample=sample,
        raw_output=raw_output,
        generated_count=int(generation.generated_ids.numel()),
        generation=generation,
        config_sha=config_sha,
        source_commit=source_commit,
        prep_diag=prep_diag,
        width=int(sample["width"]),
        height=int(sample["height"]),
        oracle_artifact=None,
        initial_random_seed=None,
        wall_latency_s=wall_latency_s,
        peak_allocated_mb=peak_allocated_mb,
        peak_reserved_mb=peak_reserved_mb,
    )
    result["dataset"] = "refcoco"
    result["split"] = "train"
    return result


def run(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit <= 0:
        raise GateLabelError("--limit must be positive")
    rows = load_manifest(args.manifest, args.limit)
    if any(str(row.get("dataset")).lower() != "refcoco" or str(row.get("split")).lower() != "train" for row in rows):
        raise GateLabelError("Gate labels accept RefCOCO train records only")
    run_dir = (args.output_root / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    proposed = build_config(args, rows, run_dir)
    config_path = run_dir / "config.json"
    if config_path.exists():
        if not args.resume:
            raise GateLabelError(f"Run directory already initialized: {run_dir}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not compatible(config, proposed):
            raise GateLabelError("Resume config differs from initialized gate-label run")
    else:
        config = proposed
        atomic_write(config_path, canonical_bytes(config))
    config_sha = sha256_file(config_path)
    partial_path = run_dir / "labels.partial.jsonl"
    final_path = run_dir / "labels.jsonl"
    existing = load_partial(partial_path, rows, config_sha) if args.resume else []
    if final_path.exists() and not (run_dir / "COMPLETED.json").is_file():
        raise GateLabelError("Unvalidated final gate labels already exist; use a fresh attempt")
    pending = rows[len(existing) :]

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    model_load_started = time.perf_counter()
    torch, processor, model, device, model_environment = load_model(MODEL, "bf16")
    image_token_id = int(getattr(model.config, "image_token_index", 32000))
    safe_forwarder = ProgressivePruneForwarder(
        torch,
        model,
        image_token_id=image_token_id,
        schedule=SAFE_SCHEDULE,
        diagnostics_mode="benchmark",
    )
    aggressive_forwarder = ProgressivePruneForwarder(
        torch,
        model,
        image_token_id=image_token_id,
        schedule=AGGRESSIVE_SCHEDULE,
        diagnostics_mode="benchmark",
    )
    feature_forwarder = DynamicRouteForwarder(
        torch,
        model,
        image_token_id=image_token_id,
        diagnostics_mode="benchmark",
        synchronise_subtimings=True,
    )
    environment = {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "model_load_latency_s": time.perf_counter() - model_load_started,
        **model_environment,
    }
    atomic_write(run_dir / "environment.json", canonical_bytes(jsonable(environment)))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mode = "a" if args.resume and partial_path.exists() else "w"
    with partial_path.open(mode, encoding="utf-8") as handle:
        for offset, sample in enumerate(pending, 1):
            from run_official_baseline import load_sample_images

            _, images = load_sample_images(sample)
            if len(images) != 1:
                raise GateLabelError("Registered RefCOCO gate labels require one image")
            prompt = REGISTERED_PROMPT.format(query=str(sample["query"]))
            text_prompt = apply_chat_template(processor, prompt, 1)
            candidate_inputs, prep_diag = prepare_full_inputs(
                torch, processor, model, device, images, prompt
            )
            resolution = query_resolution(
                processor,
                text_prompt,
                str(sample["query"]),
                candidate_inputs["input_ids"],
                image_token_id,
            )
            torch.cuda.synchronize()
            feature_started = time.perf_counter()
            with torch.inference_mode():
                features = feature_forwarder.features_after_k2(
                    inputs_embeds=candidate_inputs["inputs_embeds"],
                    input_ids=candidate_inputs["input_ids"],
                    attention_mask=candidate_inputs["attention_mask"],
                    query_original_positions=resolution.token_positions,
                    width=int(sample["width"]),
                    height=int(sample["height"]),
                )
            torch.cuda.synchronize()
            feature_latency = time.perf_counter() - feature_started

            route_outputs: dict[str, dict[str, Any]] = {}
            for route_name, forwarder in (
                ("safe", safe_forwarder),
                ("aggressive", aggressive_forwarder),
            ):
                torch.cuda.synchronize()
                route_started = time.perf_counter()
                generation = forwarder.generate_with_segment_caches(
                    inputs_embeds=candidate_inputs["inputs_embeds"],
                    input_ids=candidate_inputs["input_ids"],
                    attention_mask=candidate_inputs["attention_mask"],
                    sample_id=str(sample["sample_id"]),
                    seed=0,
                    query_original_positions=resolution.token_positions,
                    max_new_tokens=MAX_NEW_TOKENS,
                    eos_ids=eos_token_ids(model, processor),
                )
                torch.cuda.synchronize()
                route_latency = time.perf_counter() - route_started
                raw_output = processor.decode(
                    generation.generated_ids, skip_special_tokens=True
                ).strip()
                route_outputs[route_name] = route_result(
                    sample=sample,
                    raw_output=raw_output,
                    generation=generation,
                    config_sha=config_sha,
                    source_commit=str(config["source/git_commit"]),
                    prep_diag=prep_diag,
                    wall_latency_s=route_latency,
                    peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2,
                    peak_reserved_mb=torch.cuda.max_memory_reserved() / 1024**2,
                )
                del generation
            record = {
                "sample_id": str(sample["sample_id"]),
                "config_sha256": config_sha,
                "source_git_commit": config["source/git_commit"],
                "training_partition": sample["training_partition"],
                "gate_partition": sample["gate_partition"],
                "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                "feature_dimension": FEATURE_DIMENSION,
                "feature_vector_sha256": features.vector_sha256,
                "feature_vector": [float(value) for value in features.vector.detach().cpu().tolist()],
                "feature_scalars": features.scalar_values,
                "feature_latency_s": feature_latency,
                "shared_feature_prefix_boundary": GATE_BOUNDARY,
                "shared_feature_prefix_executions": 1,
                "physical_route_executions": 2,
                "safe_prediction": route_outputs["safe"],
                "aggressive_prediction": route_outputs["aggressive"],
            }
            handle.write(json.dumps(jsonable(record), sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush()
            completed = len(existing) + offset
            if completed % 25 == 0 or completed == len(rows):
                print(f"GATE_LABELS: processed {completed}/{len(rows)}", flush=True)
            for image in images:
                image.close()
            del candidate_inputs, prep_diag, resolution, features, route_outputs, record
            gc.collect()
    os.replace(partial_path, final_path)
    from validate_adaptive_gate_labels import validate_and_finalize

    report = validate_and_finalize(run_dir, args.manifest, expected_rows=len(rows))
    print(json.dumps({"status": "complete", "run_dir": str(run_dir), "metrics": report["metrics"]}, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted; exact compatible gate-label prefix was preserved.", file=sys.stderr)
        return 130
    except (GateLabelError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
