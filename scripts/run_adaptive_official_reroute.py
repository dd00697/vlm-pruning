#!/usr/bin/env python
"""Run pinned official Reroute PDrop methods on the frozen RefCOCO manifest."""

from __future__ import annotations

import argparse
import hashlib
import inspect
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

import yaml
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_adaptive_official_sources import recursive_hash  # noqa: E402
from run_baseline import apply_chat_template, box_iou, parse_box  # noqa: E402
from run_official_baseline import load_manifest, load_model, load_sample_images  # noqa: E402
from vlm_pruning.artifacts import sha256_file, source_provenance  # noqa: E402


CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
DEFAULT_MANIFEST = REPO_ROOT / (
    "experiments/progressive_oracle_refcoco_val_20260814/"
    "manifests/refcoco_val_discovery1000.jsonl"
)
DEFAULT_OUTPUT_ROOT = CAMPAIGN_ROOT / "official_runs"
OFFICIAL_ROOT = CAMPAIGN_ROOT / "official_sources"
CHECKOUT = OFFICIAL_ROOT / "checkouts/reroute"
SOURCE_LOCK = OFFICIAL_ROOT / "source_lock.json"
EXPECTED_MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
EXPECTED_REROUTE_COMMIT = "9d422042076de678b2ad67c4a6660e4b97971a1e"
MODEL = "llava-hf/llava-1.5-7b-hf"
MAX_NEW_TOKENS = 48
ORIGINAL_VISUAL_TOKENS = 576
REGISTERED_PROMPT = (
    'Identify the target described by this referring expression: "{query}".\n'
    "Return exactly one bounding box in normalized 0-1 image coordinates as "
    "[x_min, y_min, x_max, y_max]. Do not include any other text."
)
METHOD_CONFIGS = {
    "pdrop": CHECKOUT / "configs/experiment/llava15/avg192/llava15_pdrop_earlyL2_avg192.yaml",
    "pdrop_reroute": CHECKOUT / "configs/experiment/llava15/avg192/llava15_ours_vs_pdrop_avg192.yaml",
}


class OfficialRerouteError(RuntimeError):
    """Raised when the native official contract is not satisfied."""


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


def semantic_command(args: argparse.Namespace) -> dict[str, Any]:
    """Return the resume-stable scientific invocation contract."""
    return {
        "program": "scripts/run_adaptive_official_reroute.py",
        "cell_id": args.cell_id,
        "method": args.method,
        "manifest": str(args.manifest.resolve().relative_to(REPO_ROOT)),
        "output_root": str(args.output_root.resolve().relative_to(REPO_ROOT)),
        "run_name": args.run_name,
        "limit": args.limit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--method", required=True, choices=["none", "pdrop", "pdrop_reroute"])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def source_contract() -> dict[str, Any]:
    paths = [
        Path("scripts/run_adaptive_official_reroute.py"),
        Path("scripts/validate_adaptive_official_reroute.py"),
        Path("scripts/prepare_adaptive_official_sources.py"),
        Path("scripts/run_baseline.py"),
        Path("scripts/run_official_baseline.py"),
        Path("vlm_pruning/artifacts.py"),
    ]
    provenance = source_provenance(REPO_ROOT, paths)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *[str(path) for path in paths]],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    provenance["git_dirty"] = bool(dirty)
    provenance["code_paths_checked"] = [str(path) for path in paths]
    return provenance


def verify_official_source() -> dict[str, Any]:
    if not SOURCE_LOCK.is_file():
        raise OfficialRerouteError("Official source lock is missing")
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    locked = lock["sources"]["reroute"]
    commit = subprocess.check_output(["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(CHECKOUT), "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    recursive, files, size = recursive_hash(CHECKOUT)
    checkout_diff = subprocess.check_output(["git", "-C", str(CHECKOUT), "diff", "--binary"])
    if (
        commit != EXPECTED_REROUTE_COMMIT
        or commit != locked["commit"]
        or dirty
        or recursive != locked["recursive_source_sha256"]
        or files != locked["files"]
        or size != locked["bytes"]
    ):
        raise OfficialRerouteError("Pinned official Reroute source no longer matches source_lock.json")
    return {
        "checkout": str(CHECKOUT.relative_to(REPO_ROOT)),
        "commit": commit,
        "recursive_source_sha256": recursive,
        "source_lock_sha256": sha256_file(SOURCE_LOCK),
        "working_tree_diff_bytes": len(checkout_diff),
        "working_tree_diff_sha256": hashlib.sha256(checkout_diff).hexdigest(),
    }


def method_contract(method: str) -> dict[str, Any]:
    if method == "none":
        return {
            "method": "none",
            "action": "none",
            "drop_layers": [],
            "keep_ratios": [],
            "monotonic": None,
            "official_config": None,
            "realized_visual_tokens_by_layer": [576] * 32,
        }
    path = METHOD_CONFIGS[method]
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    routing = payload["routing"]
    expected_action = "physical_delete" if method == "pdrop" else "compact_route"
    if routing["method"] != "pdrop" or routing["action"] != expected_action:
        raise OfficialRerouteError(f"Official config does not define the registered {method} method")
    layers = [int(value) for value in routing["drop_layers"]]
    ratios = [float(value) for value in routing["keep_ratios"]]
    budgets = [max(1, int(ORIGINAL_VISUAL_TOKENS * ratio)) for ratio in ratios]
    trace: list[int] = []
    start = 0
    budget = ORIGINAL_VISUAL_TOKENS
    for layer, next_budget in zip(layers, budgets, strict=True):
        trace.extend([budget] * (layer - start))
        start = layer
        budget = next_budget
    trace.extend([budget] * (32 - start))
    if len(trace) != 32:
        raise OfficialRerouteError("Official visual-token trace is not 32 layers")
    return {
        "method": "pdrop",
        "action": expected_action,
        "drop_layers": layers,
        "keep_ratios": ratios,
        "budgets": budgets,
        "monotonic": bool(routing["monotonic"]),
        "official_config": str(path.relative_to(REPO_ROOT)),
        "official_config_sha256": sha256_file(path),
        "realized_visual_tokens_by_layer": trace,
    }


def build_config(args: argparse.Namespace, rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    source = source_contract()
    if source["git_dirty"]:
        raise OfficialRerouteError("Official runs require clean committed adapter/validator source")
    official = verify_official_source()
    contract = method_contract(args.method)
    manifest_hash = sha256_file(args.manifest)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise OfficialRerouteError("Official runs require the frozen discovery manifest")
    if args.limit is None and len(rows) != 1000:
        raise OfficialRerouteError("A quality run must contain exactly 1,000 rows")
    command = semantic_command(args)
    return {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "cell_id": args.cell_id,
        "run_name": args.run_name,
        "run/output_dir": str(run_dir.relative_to(REPO_ROOT)),
        "scope": "official_parity_smoke" if args.limit is not None else "official_discovery",
        "limit": args.limit,
        "dataset/name": "refcoco",
        "dataset/split": "val",
        "dataset/manifest": str(args.manifest.resolve().relative_to(REPO_ROOT)),
        "dataset/manifest_checksum": manifest_hash,
        "dataset/num_samples": len(rows),
        "model/name": MODEL,
        "model/precision": "bf16",
        "model/decoding": "deterministic_greedy",
        "model/max_new_tokens": MAX_NEW_TOKENS,
        "prompt/template": REGISTERED_PROMPT,
        "evaluator/coordinate_mode": "normalized_0_1",
        "original_visual_tokens": ORIGINAL_VISUAL_TOKENS,
        "official/source": official,
        "official/method_contract": contract,
        "command/semantic": command,
        "command/sha256": json_checksum(command),
        "source/git_commit": source["git_commit"],
        "source/git_dirty": source["git_dirty"],
        "source/sha256": source["sha256"],
    }


def compatible(existing: dict[str, Any], proposed: dict[str, Any]) -> bool:
    ignored = {"created_at_utc"}
    return {key: value for key, value in existing.items() if key not in ignored} == {
        key: value for key, value in proposed.items() if key not in ignored
    }


def load_partial(path: Path, config_sha: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(existing) > len(rows):
        raise OfficialRerouteError("Partial contains too many rows")
    for index, row in enumerate(existing):
        if row.get("sample_id") != rows[index]["sample_id"] or row.get("config_sha256") != config_sha:
            raise OfficialRerouteError("Partial is not an exact compatible manifest prefix")
    return existing


def move_inputs(torch: Any, inputs: Any, device: Any, dtype: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            result[key] = value
        elif bool(torch.is_floating_point(value)):
            result[key] = value.to(device=device, dtype=dtype)
        else:
            result[key] = value.to(device=device)
    return result


def patch_official(method: str, model: Any, contract: dict[str, Any]) -> tuple[Any | None, dict[str, str]]:
    if method == "none":
        return None, {}
    sys.path.insert(0, str(CHECKOUT))
    from models.dispatcher import TokenDispatcher
    from models.patching import patch_model_for_routing
    from models.router import PDropRouter

    router = PDropRouter(
        drop_layers=list(contract["drop_layers"]),
        keep_ratios=list(contract["keep_ratios"]),
        monotonic=bool(contract["monotonic"]),
    )
    ctx = patch_model_for_routing(
        model,
        router=router,
        dispatcher=TokenDispatcher(),
        action=str(contract["action"]),
        model_family="llava",
    )
    modules = {
        "router": str(Path(inspect.getsourcefile(PDropRouter) or "").resolve().relative_to(REPO_ROOT)),
        "patching": str(Path(inspect.getsourcefile(patch_model_for_routing) or "").resolve().relative_to(REPO_ROOT)),
        "dispatcher": str(Path(inspect.getsourcefile(TokenDispatcher) or "").resolve().relative_to(REPO_ROOT)),
    }
    if any(not value.startswith(str(CHECKOUT.relative_to(REPO_ROOT))) for value in modules.values()):
        raise OfficialRerouteError("Algorithm-defining module did not import from pinned official checkout")
    return ctx, modules


def routing_trace(ctx: Any, contract: dict[str, Any]) -> list[dict[str, Any]]:
    if ctx is None:
        return []
    action = contract["action"]
    incoming = list(range(ORIGINAL_VISUAL_TOKENS))
    traces: list[dict[str, Any]] = []
    for layer, budget in zip(contract["drop_layers"], contract["budgets"], strict=True):
        decision = ctx.routing_log.get(layer)
        if decision is None:
            raise OfficialRerouteError(f"Official source emitted no routing decision at layer {layer}")
        local = [int(value) for value in decision.selected_indices[0].detach().cpu().tolist()]
        if len(local) != budget or local != sorted(local) or len(set(local)) != len(local):
            raise OfficialRerouteError(f"Official routing decision at layer {layer} violates exact budget/order")
        if action == "physical_delete":
            if any(value < 0 or value >= len(incoming) for value in local):
                raise OfficialRerouteError("Physical PDrop selected an invalid local index")
            original = [incoming[value] for value in local]
            incoming_original = list(incoming)
            incoming = original
        else:
            if any(value < 0 or value >= ORIGINAL_VISUAL_TOKENS for value in local):
                raise OfficialRerouteError("Recoverable PDrop selected an invalid original index")
            original = local
            incoming_original = list(range(ORIGINAL_VISUAL_TOKENS))
        traces.append(
            {
                "layer": int(layer),
                "action": action,
                "requested_budget": int(budget),
                "realized_budget": len(original),
                "incoming_original_indices": incoming_original,
                "selected_local_indices": local,
                "selected_original_indices": original,
                "selected_original_indices_checksum": json_checksum(original),
            }
        )
    return traces


def run(args: argparse.Namespace) -> int:
    rows = load_manifest(args.manifest, args.limit)
    if any(str(row.get("dataset")).lower() != "refcoco" or str(row.get("split")).lower() != "val" for row in rows):
        raise OfficialRerouteError("Official adapter accepts RefCOCO validation only")
    run_dir = (args.output_root / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    proposed = build_config(args, rows, run_dir)
    config_path = run_dir / "config.json"
    if config_path.exists():
        if not args.resume:
            raise OfficialRerouteError(f"Run directory already initialized: {run_dir}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not compatible(config, proposed):
            raise OfficialRerouteError("Resume config differs from initialized run")
    else:
        config = proposed
        atomic_write(config_path, canonical_bytes(config))
    config_sha = sha256_file(config_path)
    partial_path = run_dir / "predictions.partial.jsonl"
    final_path = run_dir / "predictions.jsonl"
    existing = load_partial(partial_path, config_sha, rows) if args.resume else []
    pending = rows[len(existing) :]

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    load_start = time.perf_counter()
    torch, processor, model, device, environment = load_model(MODEL, "bf16")
    environment["model_load_latency_s"] = time.perf_counter() - load_start
    environment["python"] = sys.version
    environment["platform"] = platform.platform()
    contract = config["official/method_contract"]
    ctx, official_modules = patch_official(args.method, model, contract)
    environment["official_algorithm_modules"] = official_modules
    environment["official_checkout_commit"] = EXPECTED_REROUTE_COMMIT
    environment["semantic_command_sha256"] = config["command/sha256"]
    atomic_write(run_dir / "environment.json", canonical_bytes(environment))
    dtype = next(model.parameters()).dtype
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mode = "a" if args.resume and partial_path.exists() else "w"
    with partial_path.open(mode, encoding="utf-8") as handle:
        for offset, sample in enumerate(pending, 1):
            _, images = load_sample_images(sample)
            if len(images) != 1:
                raise OfficialRerouteError("Official RefCOCO protocol requires one image")
            prompt = REGISTERED_PROMPT.format(query=str(sample["query"]))
            text_prompt = apply_chat_template(processor, prompt, 1)
            processed = processor(images=images, text=text_prompt, return_tensors="pt")
            inputs = move_inputs(torch, processed, device, dtype)
            input_length = int(inputs["input_ids"].shape[1])
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            latency = time.perf_counter() - started
            generated_ids = generated[:, input_length:]
            raw_output = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
            trace = routing_trace(ctx, contract)
            raw_box, predicted_box, parse_failure = parse_box(
                raw_output, int(sample["width"]), int(sample["height"]), "normalized_0_1"
            )
            iou = box_iou(predicted_box, sample["gt_box"]) if predicted_box is not None else 0.0
            visual_trace = list(contract["realized_visual_tokens_by_layer"])
            result = {
                "sample_id": str(sample["sample_id"]),
                "dataset": "refcoco",
                "split": "val",
                "query": sample["query"],
                "image_id": sample["image_id"],
                "image_path": sample["image_path"],
                "width": int(sample["width"]),
                "height": int(sample["height"]),
                "gt_box": sample["gt_box"],
                "raw_model_output": raw_output,
                "raw_parsed_box": raw_box,
                "predicted_box": predicted_box,
                "parse_failure_reason": parse_failure,
                "valid": predicted_box is not None,
                "invalid": predicted_box is None,
                "box_iou": iou,
                "success_iou_0_5": bool(iou >= 0.5),
                "generated_token_count": int(generated_ids.numel()),
                "wall_inference_latency_s": latency,
                "official_routing_trace": trace,
                "realized_visual_tokens_by_layer": visual_trace,
                "candidate_average_visual_tokens": sum(visual_trace) / 32.0,
                "official_action": contract["action"],
                "official_algorithm_modules": official_modules,
                "official_source_commit": EXPECTED_REROUTE_COMMIT,
                "official_source_recursive_sha256": config["official/source"]["recursive_source_sha256"],
                "config_sha256": config_sha,
                "source_git_commit": config["source/git_commit"],
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
            }
            handle.write(json.dumps(result, sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush()
            completed = len(existing) + offset
            if completed % 25 == 0 or completed == len(rows):
                print(f"{args.cell_id}: processed {completed}/{len(rows)}", flush=True)
            for image in images:
                image.close()
            del inputs, processed, generated, generated_ids
    os.replace(partial_path, final_path)
    from validate_adaptive_official_reroute import validate_and_finalize

    validation = validate_and_finalize(run_dir, args.manifest, expected_rows=len(rows))
    print(json.dumps({"status": "complete", "run_dir": str(run_dir), "metrics": validation["metrics"]}, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted; exact compatible physical prefix was preserved.", file=sys.stderr)
        return 130
    except (OfficialRerouteError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
