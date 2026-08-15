#!/usr/bin/env python
"""Run one physical frozen adaptive-route gate seed on RefCOCO validation."""

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

import numpy as np
import torch as checkpoint_torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_baseline import apply_chat_template, eos_token_ids  # noqa: E402
from run_official_baseline import load_manifest, load_model, load_sample_images  # noqa: E402
from run_progressive_oracle_refcoco import (  # noqa: E402
    MAX_NEW_TOKENS,
    ORIGINAL_VISUAL_TOKENS,
    REGISTERED_PROMPT,
    build_result,
    prepare_full_inputs,
    query_resolution,
)
from vlm_pruning.artifacts import sha256_file, source_provenance  # noqa: E402
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
    RouteDecision,
    feature_tensor_sha256,
    route_schedule,
)
from vlm_pruning.llava_hooks.route_gate_model import (  # noqa: E402
    TRAINING_SPEC_SHA256,
    build_gate_model,
    normalization_sha256,
)


CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
DEFAULT_MANIFEST = REPO_ROOT / (
    "experiments/progressive_oracle_refcoco_val_20260814/"
    "manifests/refcoco_val_discovery1000.jsonl"
)
DEFAULT_OUTPUT_ROOT = CAMPAIGN_ROOT / "gate_validation_runs"
DEFAULT_TRAINING_ROOT = CAMPAIGN_ROOT / "gate_training"
EXPECTED_MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
MODEL = "llava-hf/llava-1.5-7b-hf"
SOURCE_PATHS = (
    Path("scripts/run_adaptive_route_gate.py"),
    Path("scripts/validate_adaptive_route_gate.py"),
    Path("scripts/run_progressive_oracle_refcoco.py"),
    Path("scripts/run_baseline.py"),
    Path("scripts/run_official_baseline.py"),
    Path("vlm_pruning/artifacts.py"),
    Path("vlm_pruning/features/query_tokens.py"),
    Path("vlm_pruning/llava_hooks/mid_layer_pruning.py"),
    Path("vlm_pruning/llava_hooks/progressive_pruning.py"),
    Path("vlm_pruning/llava_hooks/route_gate.py"),
    Path("vlm_pruning/llava_hooks/route_gate_model.py"),
)


class AdaptiveGateRunError(RuntimeError):
    """Raised when physical route-gate inference violates its frozen contract."""


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
    parser.add_argument("--seed", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def checked_source() -> dict[str, Any]:
    source = source_provenance(REPO_ROOT, SOURCE_PATHS)
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *[str(path) for path in SOURCE_PATHS]],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if status:
        raise AdaptiveGateRunError("Dynamic route-gate inference requires committed clean executable source")
    source["git_dirty"] = False
    source["code_paths_checked"] = [str(path) for path in SOURCE_PATHS]
    return source


def training_contract(training_root: Path, seed: int) -> dict[str, Any]:
    run_dir = training_root / f"seed_{seed}"
    names = (
        "config.json",
        "checkpoint.pt",
        "training_history.json",
        "calibration.json",
        "calibration_predictions.jsonl",
        "environment.json",
        "integrity_report.json",
        "COMPLETED.json",
    )
    for name in names:
        if not (run_dir / name).is_file():
            raise AdaptiveGateRunError(f"Missing completed gate-training artifact: {run_dir / name}")
    integrity = json.loads((run_dir / "integrity_report.json").read_text(encoding="utf-8"))
    completion = json.loads((run_dir / "COMPLETED.json").read_text(encoding="utf-8"))
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if integrity.get("status") != "pass" or completion.get("status") != "complete":
        raise AdaptiveGateRunError("Gate training has not passed independent validation")
    if int(config.get("seed", -1)) != seed or int(integrity.get("seed", -1)) != seed:
        raise AdaptiveGateRunError("Gate-training seed differs from the requested validation seed")
    if config.get("training_spec_sha256") != TRAINING_SPEC_SHA256:
        raise AdaptiveGateRunError("Gate-training specification differs")
    if sha256_file(run_dir / "checkpoint.pt") != integrity.get("checkpoint_sha256"):
        raise AdaptiveGateRunError("Gate checkpoint hash differs from training integrity")
    if calibration.get("checkpoint_sha256") != integrity.get("checkpoint_sha256"):
        raise AdaptiveGateRunError("Gate calibration does not belong to the checkpoint")
    return {
        "run_dir": str(run_dir.resolve().relative_to(REPO_ROOT)),
        "seed": seed,
        "threshold": float(calibration["threshold"]),
        "calibration_status": calibration["status"],
        "calibration_metrics": calibration["chosen_metrics"],
        "training_spec_sha256": TRAINING_SPEC_SHA256,
        "normalization_sha256": integrity["normalization_sha256"],
        "checkpoint_sha256": integrity["checkpoint_sha256"],
        "config_sha256": sha256_file(run_dir / "config.json"),
        "integrity_sha256": sha256_file(run_dir / "integrity_report.json"),
        "completion_sha256": sha256_file(run_dir / "COMPLETED.json"),
        "train_labels_sha256": config["train_labels_sha256"],
        "train_label_manifest_sha256": "8f2b567f27ccc05d1fef7b8368f1de7c454127e224c8f25f24944444d2b1d18f",
    }


def build_config(args: argparse.Namespace, rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    if sha256_file(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise AdaptiveGateRunError("Dynamic gate requires the frozen 1,000-row validation manifest")
    if args.limit is None and len(rows) != 1000:
        raise AdaptiveGateRunError("A full dynamic gate run requires exactly 1,000 records")
    source = checked_source()
    training = training_contract(args.training_root.resolve(), int(args.seed))
    command = {
        "program": "scripts/run_adaptive_route_gate.py",
        "seed": int(args.seed),
        "run_name": args.run_name,
        "manifest": str(args.manifest.resolve().relative_to(REPO_ROOT)),
        "output_root": str(args.output_root.resolve().relative_to(REPO_ROOT)),
        "training_root": str(args.training_root.resolve().relative_to(REPO_ROOT)),
        "limit": args.limit,
    }
    return {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "run_name": args.run_name,
        "scope": "gate_validation_smoke" if args.limit is not None else "gate_validation_quality",
        "seed": int(args.seed),
        "primary_seed": int(args.seed) == 0,
        "dataset/name": "refcoco",
        "dataset/split": "val",
        "dataset/manifest": str(args.manifest.resolve().relative_to(REPO_ROOT)),
        "dataset/manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "dataset/rows": len(rows),
        "model/name": MODEL,
        "model/precision": "bf16",
        "model/decoding": "deterministic_greedy",
        "model/max_new_tokens": MAX_NEW_TOKENS,
        "prompt/template": REGISTERED_PROMPT,
        "original_visual_tokens": ORIGINAL_VISUAL_TOKENS,
        "position_mode": "gap_preserving",
        "gate/boundary": GATE_BOUNDARY,
        "gate/feature_schema": FEATURE_SCHEMA,
        "gate/feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "gate/feature_dimension": FEATURE_DIMENSION,
        "gate/training": training,
        "safe_route": {"name": SAFE_ROUTE, "schedule": [event.as_dict() for event in SAFE_SCHEDULE]},
        "aggressive_route": {
            "name": AGGRESSIVE_ROUTE,
            "schedule": [event.as_dict() for event in AGGRESSIVE_SCHEDULE],
        },
        "source/git_commit": source["git_commit"],
        "source/git_dirty": False,
        "source/sha256": source["source_sha256"],
        "command/semantic": command,
        "command/sha256": json_checksum(command),
    }


def compatible(existing: dict[str, Any], proposed: dict[str, Any]) -> bool:
    ignored = {"created_at_utc"}
    return {key: value for key, value in existing.items() if key not in ignored} == {
        key: value for key, value in proposed.items() if key not in ignored
    }


def load_partial(path: Path, manifest: list[dict[str, Any]], config_sha: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) > len(manifest):
        raise AdaptiveGateRunError("Dynamic-gate partial contains too many rows")
    for index, row in enumerate(rows):
        if row.get("sample_id") != manifest[index].get("sample_id") or row.get("config_sha256") != config_sha:
            raise AdaptiveGateRunError(f"Dynamic-gate partial is incompatible at row {index}")
    return rows


def load_gate(training_root: Path, seed: int, torch: Any, device: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    run_dir = training_root / f"seed_{seed}"
    try:
        checkpoint = checkpoint_torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = checkpoint_torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    gate_model = build_gate_model(torch).to(device=device, dtype=torch.float32)
    gate_model.load_state_dict(checkpoint["state_dict"], strict=True)
    gate_model.eval()
    mean_array = np.load(training_root / "normalization_mean.npy", allow_pickle=False)
    std_array = np.load(training_root / "normalization_std.npy", allow_pickle=False)
    normalization = json.loads((training_root / "NORMALIZATION.json").read_text(encoding="utf-8"))
    if normalization_sha256(mean_array, std_array) != normalization["normalization_sha256"]:
        raise AdaptiveGateRunError("Gate normalization hash differs")
    mean = torch.from_numpy(mean_array).to(device=device, dtype=torch.float32)
    std = torch.from_numpy(std_array).to(device=device, dtype=torch.float32)
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    return gate_model, mean, std, calibration


def run(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit <= 0:
        raise AdaptiveGateRunError("--limit must be positive")
    rows = load_manifest(args.manifest, args.limit)
    if any(str(row.get("dataset")).lower() != "refcoco" or str(row.get("split")).lower() != "val" for row in rows):
        raise AdaptiveGateRunError("Dynamic gate accepts only RefCOCO validation records")
    run_dir = (args.output_root / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    proposed = build_config(args, rows, run_dir)
    config_path = run_dir / "config.json"
    if config_path.is_file():
        if not args.resume:
            raise AdaptiveGateRunError(f"Run directory already initialized: {run_dir}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not compatible(config, proposed):
            raise AdaptiveGateRunError("Existing dynamic-gate config differs from requested resume")
    else:
        config = proposed
        atomic_write(config_path, canonical_bytes(config))
    config_sha = sha256_file(config_path)
    partial_path = run_dir / "predictions.partial.jsonl"
    predictions_path = run_dir / "predictions.jsonl"
    completed_path = run_dir / "COMPLETED.json"
    if predictions_path.exists() and not completed_path.exists():
        raise AdaptiveGateRunError("Unvalidated final predictions exist; use a fresh attempt")
    existing = load_partial(partial_path, rows, config_sha) if args.resume else []
    pending = rows[len(existing) :]

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    model_started = time.perf_counter()
    torch, processor, model, device, model_environment = load_model(MODEL, "bf16")
    gate_model, gate_mean, gate_std, calibration = load_gate(
        args.training_root.resolve(), int(args.seed), torch, device
    )
    threshold = float(calibration["threshold"])
    image_token_id = int(getattr(model.config, "image_token_index", 32000))
    forwarder = DynamicRouteForwarder(
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
        "model_and_gate_load_latency_s": time.perf_counter() - model_started,
        "gate/device": str(device),
        "gate/dtype": "float32",
        "gate/trainable_parameters": sum(item.numel() for item in gate_model.parameters()),
        **model_environment,
    }
    atomic_write(run_dir / "environment.json", canonical_bytes(jsonable(environment)))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mode = "a" if args.resume and partial_path.exists() else "w"
    with partial_path.open(mode, encoding="utf-8") as handle:
        for offset, sample in enumerate(pending, 1):
            _, images = load_sample_images(sample)
            if len(images) != 1:
                raise AdaptiveGateRunError("Registered RefCOCO gate requires exactly one image")
            width = int(sample.get("width") or images[0].size[0])
            height = int(sample.get("height") or images[0].size[1])
            prompt = REGISTERED_PROMPT.format(query=str(sample["query"]))
            text_prompt = apply_chat_template(processor, prompt, 1)
            candidate_inputs, prep_diag = prepare_full_inputs(torch, processor, model, device, images, prompt)
            resolution = query_resolution(
                processor,
                text_prompt,
                str(sample["query"]),
                candidate_inputs["input_ids"],
                image_token_id,
            )
            capture: dict[str, Any] = {}

            def decide(vector: Any) -> RouteDecision:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    outputs = gate_model(((vector - gate_mean) / gate_std).unsqueeze(0))[0]
                    probability = torch.sigmoid(outputs[0])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                capture["model_latency_s"] = time.perf_counter() - started
                capture["feature_vector"] = vector.detach()
                capture["safe_logit"] = float(outputs[0].item())
                capture["predicted_iou_delta"] = float(outputs[1].item())
                value = float(probability.item())
                capture["probability_safe"] = value
                route_name = SAFE_ROUTE if value >= threshold else AGGRESSIVE_ROUTE
                return RouteDecision(
                    route_name=route_name,
                    probability_safe=value,
                    threshold=threshold,
                    schedule=route_schedule(route_name),
                )

            wall_started = time.perf_counter()
            generation = forwarder.generate_with_route_gate(
                inputs_embeds=candidate_inputs["inputs_embeds"],
                input_ids=candidate_inputs["input_ids"],
                attention_mask=candidate_inputs["attention_mask"],
                sample_id=str(sample["sample_id"]),
                seed=int(args.seed),
                query_original_positions=resolution.token_positions,
                width=width,
                height=height,
                decide=decide,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_ids=eos_token_ids(model, processor),
            )
            wall_latency = time.perf_counter() - wall_started
            raw_output = processor.decode(generation.generated_ids, skip_special_tokens=True).strip()
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
                oracle_artifact=None,
                initial_random_seed=None,
                wall_latency_s=wall_latency,
                peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None,
                peak_reserved_mb=torch.cuda.max_memory_reserved() / 1024**2 if torch.cuda.is_available() else None,
            )
            vector = capture["feature_vector"]
            result.update(
                {
                    "gate_seed": int(args.seed),
                    "gate_checkpoint_sha256": config["gate/training"]["checkpoint_sha256"],
                    "gate_training_spec_sha256": TRAINING_SPEC_SHA256,
                    "gate_normalization_sha256": config["gate/training"]["normalization_sha256"],
                    "gate_threshold": threshold,
                    "gate_probability_safe": capture["probability_safe"],
                    "gate_route_choice": generation.diagnostics["gate/route_name"],
                    "gate_safe_logit": capture["safe_logit"],
                    "gate_predicted_iou_delta": capture["predicted_iou_delta"],
                    "gate_model_latency_s": capture["model_latency_s"],
                    "gate_feature_vector": [float(value) for value in vector.detach().cpu().tolist()],
                    "gate_feature_vector_sha256": feature_tensor_sha256(vector),
                    "gate_physical_routes_executed": 1,
                }
            )
            handle.write(json.dumps(jsonable(result), ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            done = len(existing) + offset
            if done % 25 == 0 or done == len(rows):
                print(f"GATE_SEED_{args.seed}: processed {done}/{len(rows)}", flush=True)
            for image in images:
                image.close()
            del candidate_inputs, prep_diag, resolution, generation, result, capture, vector
            gc.collect()
    os.replace(partial_path, predictions_path)
    from validate_adaptive_route_gate import validate_and_finalize

    report = validate_and_finalize(run_dir, args.manifest, args.training_root, expected_rows=len(rows))
    print(json.dumps({"status": "complete", "seed": args.seed, "metrics": report["metrics"]}, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted; compatible dynamic-gate prefix was preserved.", file=sys.stderr)
        return 130
    except (AdaptiveGateRunError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"UNEXPECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
