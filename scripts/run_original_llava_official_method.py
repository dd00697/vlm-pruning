#!/usr/bin/env python
"""Run official VisionZip or TransPrune on the original LLaVA-1.5 stack.

This file is deliberately only dataset/prompt/artifact glue.  The published
selection and pruning operations are imported from the pinned author
repositories and are never reimplemented here.  Forward pre-hooks observe
sequence lengths without changing tensors, which makes the exact final B=48
claim independently auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_official_baseline import (  # noqa: E402
    BaselineError,
    aggregate,
    add_model_efficiency,
    box_iou,
    default_max_new_tokens,
    load_manifest,
    load_sample_images,
    make_prompt,
    manifest_checksum,
    parse_box,
    score_vqa,
)
from vlm_pruning.artifacts import (  # noqa: E402
    ArtifactIntegrityError,
    finalize_partial_jsonl,
    load_resume_rows,
    sha256_file,
    source_provenance,
    validate_completed_run,
)


METHODS = ("official_llava_full", "official_visionzip", "official_transprune")
VISIONZIP_REPO = REPO_ROOT / "third_party/VisionZip"
VISIONZIP_LLAVA_REPO = Path("/workspace/third_party_official/LLaVA")
TRANSPRUNE_REPO = Path("/workspace/third_party_official/TransPrune")
OFFICIAL_REPOS = {
    "official_llava_full": VISIONZIP_LLAVA_REPO,
    "official_visionzip": VISIONZIP_REPO,
    "official_transprune": TRANSPRUNE_REPO,
}
OFFICIAL_URLS = {
    "official_llava_full": "https://github.com/haotian-liu/LLaVA",
    "official_visionzip": "https://github.com/dvlab-research/VisionZip",
    "official_transprune": "https://github.com/liaolea/TransPrune",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True, choices=["vqa", "grounding"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--box-coordinate-mode", default="normalized_0_1",
                        choices=["normalized_0_1", "normalized_0_1000", "absolute_pixels"])
    parser.add_argument("--prompt-profile", default="registered", choices=["registered", "paper"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def git_clean(path: Path) -> bool:
    return not subprocess.check_output(["git", "status", "--short"], cwd=path, text=True).strip()


def expected_visual_trace(method: str) -> list[int]:
    if method == "official_llava_full":
        return [576] * 32
    if method == "official_visionzip":
        return [48] * 32
    if method == "official_transprune":
        # The official schedule prunes after decoder layers 7, 9 and 12.
        return [576] * 7 + [504] * 2 + [360] * 3 + [48] * 20
    raise ValueError(f"Unknown official method: {method}")


def final_budget(method: str) -> int:
    return 576 if method == "official_llava_full" else 48


def config_fingerprint(payload: dict[str, Any]) -> str:
    scientific = {
        key: payload[key]
        for key in sorted(payload)
        if key not in {"created_at_utc", "run/output_dir", "configuration_fingerprint"}
    }
    return hashlib.sha256(
        json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def imported_under(value: Any, root: Path, label: str) -> str:
    source = Path(inspect.getfile(value)).resolve()
    if root.resolve() not in source.parents:
        raise BaselineError(f"{label} imported from {source}, expected beneath {root.resolve()}")
    return str(source)


def load_official_model(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    try:
        import torch
        import transformers
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
    except Exception as exc:
        raise BaselineError(f"Official LLaVA import failed: {type(exc).__name__}: {exc}") from exc
    if not torch.cuda.is_available():
        raise BaselineError("CUDA is required for official LLaVA baseline execution.")

    expected_llava = TRANSPRUNE_REPO if args.method == "official_transprune" else VISIONZIP_LLAVA_REPO
    builder_source = imported_under(load_pretrained_model, expected_llava, "LLaVA builder")
    model_name = get_model_name_from_path(args.model_path)
    load_kwargs: dict[str, Any] = {}
    if args.method == "official_transprune":
        load_kwargs["method"] = "transprune"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        None,
        model_name,
        **load_kwargs,
    )
    model.eval()

    method_source = builder_source
    if args.method == "official_visionzip":
        try:
            from visionzip import visionzip
        except Exception as exc:
            raise BaselineError(f"Official VisionZip import failed: {type(exc).__name__}: {exc}") from exc
        method_source = imported_under(visionzip, VISIONZIP_REPO, "VisionZip")
        # Official parameter semantics: dominant includes the CLS token, so
        # 40 dominant + 8 contextual produces exactly 48 visual tokens.
        model = visionzip(model, dominant=40, contextual=8)
    elif args.method == "official_transprune":
        model_source = imported_under(model.__class__, TRANSPRUNE_REPO, "TransPrune model")
        method_source = model_source
        model.model.accmulating_layer_list = [7, 8, 9, 10, 11, 12]
        model.model.pruning_layer_list = [7, 9, 12]
        model.model.image_token_ratio_list = [1.0, 0.875, 0.625, 1.0 / 12.0]

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    environment = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_free_memory_gb_after_load": free_bytes / 1024**3,
        "gpu_total_memory_gb": total_bytes / 1024**3,
        "precision": "fp16_official_stack",
        "model_path": str(Path(args.model_path).resolve()),
        "model_name": model_name,
        "context_length": context_len,
        "llava_builder_source": builder_source,
        "official_method_source": method_source,
    }
    return torch, tokenizer, model, image_processor, environment


class SequenceObserver:
    """Read-only hook collection for the visual-token audit."""

    def __init__(self, model: Any):
        layers = list(model.model.layers)
        if len(layers) != 32:
            raise BaselineError(f"Expected 32 LLaVA decoder layers, found {len(layers)}")
        self.layer_sequence_lengths: list[list[int]] = [[] for _ in layers]
        self.handles = []
        for index, layer in enumerate(layers):
            def observe(_module: Any, inputs: tuple[Any, ...], *, layer_index: int = index) -> None:
                if inputs and hasattr(inputs[0], "shape") and len(inputs[0].shape) >= 2:
                    length = int(inputs[0].shape[1])
                    if length > 1:
                        self.layer_sequence_lengths[layer_index].append(length)
            self.handles.append(layer.register_forward_pre_hook(observe))

    def reset(self) -> None:
        self.layer_sequence_lengths = [[] for _ in self.layer_sequence_lengths]

    def prefill_lengths(self) -> list[int]:
        if any(len(values) != 1 for values in self.layer_sequence_lengths):
            counts = [len(values) for values in self.layer_sequence_lengths]
            raise BaselineError(f"Expected one observed prefill per layer, got {counts}")
        return [values[0] for values in self.layer_sequence_lengths]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def conversation_inputs(tokenizer: Any, model: Any, prompt_text: str, image: Any) -> tuple[Any, Any, int]:
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    if model.config.mm_use_im_start_end:
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt_text
    else:
        question = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()
    image_tensor = process_images([image], model.get_vision_tower().image_processor, model.config)[0]
    return input_ids, image_tensor.unsqueeze(0).half().cuda(), int(input_ids.shape[1] - 1)


def generate_one(
    torch: Any,
    tokenizer: Any,
    model: Any,
    observer: SequenceObserver,
    method: str,
    prompt: str,
    image: Any,
    max_new_tokens: int,
) -> tuple[str, int, float, dict[str, Any]]:
    observer.reset()
    input_ids, image_tensor, text_token_count = conversation_inputs(tokenizer, model, prompt, image)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    torch.cuda.synchronize()
    latency = time.perf_counter() - started
    layer_sequence_lengths = observer.prefill_lengths()
    visual_trace = [length - text_token_count for length in layer_sequence_lengths]
    expected = expected_visual_trace(method)
    if visual_trace != expected:
        raise BaselineError(f"Observed visual-token trace {visual_trace} != locked {expected}")
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    generated_count = int(output_ids.shape[-1])
    original_sequence = text_token_count + 576
    compacted_sequence = layer_sequence_lengths[-1]
    diagnostics = {
        "actual_llm_visual_tokens": visual_trace[-1],
        "actual_retained_visual_tokens": visual_trace[-1],
        "layer_visual_token_counts": visual_trace,
        "layer_input_sequence_lengths": layer_sequence_lengths,
        "token_layer_area": sum(visual_trace),
        "original_input_sequence_length": original_sequence,
        "compacted_input_sequence_length": compacted_sequence,
        "sequence_length_reduction": original_sequence - compacted_sequence,
        "requested_keep_ratio": final_budget(method) / 576.0,
        "pruning_selection_latency_s": None,
        "llm_prefill_latency_s": None,
        "decode_latency_s": None,
        "time_to_first_token_s": None,
    }
    return output, generated_count, latency, diagnostics


def base_result(args: argparse.Namespace, sample: dict[str, Any], image_path: Path,
                raw: str, generated_count: int, latency: float,
                diagnostics: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    budget = final_budget(args.method)
    return {
        "sample_id": str(sample["sample_id"]),
        "original_sample_id": metadata.get("original_sample_id", str(sample["sample_id"])),
        "source_row": metadata.get("source_row"),
        "dataset": args.dataset,
        "split": sample.get("split") or args.split,
        "category": sample.get("category"),
        "image_path": image_path.as_posix(),
        "image_paths": [str(value) for value in sample.get("image_paths", [image_path])],
        "image_count": 1,
        "image_id": sample.get("image_id"),
        "mme_pair_id": sample.get("mme_pair_id") or metadata.get("mme_pair_id"),
        "method": args.method,
        "seed": 0,
        "target_visual_tokens": budget,
        "original_visual_tokens": 576,
        "raw_model_output": raw,
        "latency_s": latency,
        "generated_token_count": generated_count,
        "configuration_fingerprint": fingerprint,
        "retained_indices": None,
        "retained_representation": "not_applicable_for_merged_or_internal_official_method",
        "selector_name": args.method,
        "selector_metadata": {
            "official_source_executed": True,
            "exact_final_visual_tokens": diagnostics["actual_llm_visual_tokens"],
        },
        "role_labels": None,
        "pruning_implementation": "author_official_repository",
        **diagnostics,
    }


def build_config(args: argparse.Namespace, rows: list[dict[str, Any]], manifest_sha: str) -> dict[str, Any]:
    repo = OFFICIAL_REPOS[args.method]
    glue = source_provenance(REPO_ROOT, [
        Path("scripts/run_original_llava_official_method.py"),
        Path("scripts/run_official_baseline.py"),
        Path("vlm_pruning/artifacts.py"),
    ])
    budget = final_budget(args.method)
    payload = {
        "run_name": args.run_name,
        "created_at_utc": utc_now(),
        "source/git_commit": glue["git_commit"],
        "source/git_dirty": glue["git_dirty"],
        "source/sha256": glue["source_sha256"],
        "model/name": "liuhaotian/llava-v1.5-7b",
        "model/path": str(Path(args.model_path).resolve()),
        "model/stack": "author_official_original_llava",
        "model/precision": "fp16",
        "model/decoding": "deterministic_greedy",
        "model/max_new_tokens": args.max_new_tokens or default_max_new_tokens(args.dataset, args.task),
        "dataset/name": args.dataset,
        "dataset/split": args.split or rows[0].get("split"),
        "dataset/manifest": str(args.manifest),
        "dataset/manifest_checksum": manifest_sha,
        "dataset/num_samples": len(rows),
        "task": args.task,
        "method/name": args.method,
        "method/seed": 0,
        "tokens/original_visual_tokens": 576,
        "tokens/target_visual_tokens": budget,
        "tokens/actual_visual_tokens": budget,
        "tokens/per_layer_visual_trace": expected_visual_trace(args.method),
        "tokens/budget_family": "full" if budget == 576 else "stress",
        "run/is_limited": args.limit is not None,
        "run/output_dir": str(args.output_root / args.run_name),
        "limit": args.limit,
        "position_mode": "official_original_llava_compact",
        "prompt/profile": args.prompt_profile,
        "box_coordinate_mode": args.box_coordinate_mode,
        "published_baseline/source": OFFICIAL_URLS[args.method],
        "published_baseline/commit": git_revision(repo),
        "published_baseline/repository_clean": git_clean(repo),
        "published_baseline/official_source_executed": True,
        "published_baseline/visionzip_recipe": (
            "official visionzip(model, dominant=40, contextual=8): 40 includes CLS"
            if args.method == "official_visionzip" else None
        ),
        "published_baseline/transprune_accumulating_layers": (
            [7, 8, 9, 10, 11, 12] if args.method == "official_transprune" else None
        ),
        "published_baseline/transprune_pruning_layers": (
            [7, 9, 12] if args.method == "official_transprune" else None
        ),
        "published_baseline/transprune_ratios": (
            [0.875, 0.625, 1.0 / 12.0] if args.method == "official_transprune" else None
        ),
    }
    payload["configuration_fingerprint"] = config_fingerprint(payload)
    return payload


def run(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit <= 0:
        raise BaselineError("--limit must be positive")
    rows = load_manifest(args.manifest, args.limit)
    if any(int(row.get("image_count") or len(row.get("image_paths") or [1])) != 1 for row in rows):
        raise BaselineError("Official original-LLaVA methods require the locked single-image subset.")
    args.split = args.split or str(rows[0].get("split") or "unknown")
    manifest_sha = manifest_checksum(args.manifest)
    config = build_config(args, rows, manifest_sha)
    budget = final_budget(args.method)
    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    completed_path = run_dir / "COMPLETED.json"
    if args.skip_existing and completed_path.is_file():
        report = validate_completed_run(
            run_dir, rows, budget=budget, original_tokens=576, task=args.task,
            require_retained=False,
        )
        if report["status"] == "pass":
            return 0

    write_json(run_dir / "config.json", config)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    write_json(run_dir / "wandb_run.json", {"enabled": False, "reason": "campaign_local_artifacts"})
    partial_path = run_dir / "per_sample_results.partial.jsonl"
    final_path = run_dir / "per_sample_results.jsonl"
    existing = load_resume_rows(partial_path, rows) if args.resume else []
    completed_ids = {str(row["sample_id"]) for row in existing}
    pending = [row for row in rows if str(row["sample_id"]) not in completed_ids]

    model_started = time.perf_counter()
    torch, tokenizer, model, _image_processor, model_env = load_official_model(args)
    model_load_s = time.perf_counter() - model_started
    environment = {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "model_load_latency_s": model_load_s,
        **model_env,
    }
    write_json(run_dir / "run_environment.json", environment)
    torch.cuda.reset_peak_memory_stats()
    observer = SequenceObserver(model)
    results = list(existing)
    max_new_tokens = args.max_new_tokens or default_max_new_tokens(args.dataset, args.task)
    started = time.perf_counter()
    mode = "a" if args.resume and partial_path.exists() else "w"
    try:
        with partial_path.open(mode, encoding="utf-8") as handle:
            for index, sample in enumerate(pending):
                image_paths, images = load_sample_images(sample)
                image = images[0]
                prompt = make_prompt(args, sample)
                raw, generated_count, latency, diagnostics = generate_one(
                    torch, tokenizer, model, observer, args.method, prompt, image, max_new_tokens,
                )
                result = base_result(
                    args, sample, image_paths[0], raw, generated_count, latency,
                    diagnostics, config["configuration_fingerprint"],
                )
                if args.task == "grounding":
                    width = int(sample.get("width") or image.size[0])
                    height = int(sample.get("height") or image.size[1])
                    raw_box, clipped_box, parse_failure = parse_box(raw, width, height, args.box_coordinate_mode)
                    iou = box_iou(clipped_box, sample["gt_box"]) if clipped_box is not None else 0.0
                    result.update({
                        "query": sample.get("query"),
                        "width": width,
                        "height": height,
                        "gt_box": sample.get("gt_box"),
                        "raw_parsed_box": raw_box,
                        "predicted_box": clipped_box,
                        "box_iou": iou,
                        "score": iou,
                        "success_iou_0_5": bool(clipped_box is not None and iou >= 0.5),
                        "valid": clipped_box is not None,
                        "invalid": clipped_box is None,
                        "parse_failure_reason": parse_failure,
                    })
                else:
                    result.update({
                        "question": sample.get("question"),
                        "choices": sample.get("choices"),
                        "target_answer": sample.get("answer"),
                        **score_vqa(sample, raw, args.dataset),
                    })
                results.append(result)
                handle.write(json.dumps(result, ensure_ascii=True, sort_keys=True) + "\n")
                handle.flush()
                done = len(existing) + index + 1
                if done % 10 == 0 or done == len(rows):
                    print(f"Processed {done}/{len(rows)} samples", flush=True)
    finally:
        observer.close()

    results, finalization = finalize_partial_jsonl(
        partial_path, final_path, rows, budget=budget, original_tokens=576,
        require_retained=False,
    )
    run_time_s = time.perf_counter() - started
    metrics = aggregate(
        results, args, run_time_s,
        torch.cuda.max_memory_allocated() / 1024**2,
        torch.cuda.max_memory_reserved() / 1024**2,
    )
    add_model_efficiency(metrics, model, torch)
    metrics.update({
        "model/load_latency_s": model_load_s,
        "tokens/per_layer_visual_trace": expected_visual_trace(args.method),
        "tokens/token_layer_area": sum(expected_visual_trace(args.method)),
        "configuration_fingerprint": config["configuration_fingerprint"],
    })
    write_json(run_dir / "metrics.json", metrics)
    integrity = validate_completed_run(
        run_dir, rows, budget=budget, original_tokens=576, task=args.task,
        require_retained=False,
    )
    integrity["finalization"] = finalization
    integrity["trace_audit"] = {
        "status": "pass" if all(row["layer_visual_token_counts"] == expected_visual_trace(args.method) for row in results) else "fail",
        "expected": expected_visual_trace(args.method),
        "rows_checked": len(results),
    }
    if integrity["trace_audit"]["status"] != "pass":
        integrity["status"] = "fail"
        integrity["issues"].append("per-row visual-token trace mismatch")
    write_json(run_dir / "integrity_report.json", integrity)
    if integrity["status"] != "pass":
        raise ArtifactIntegrityError("Official run integrity failed: " + "; ".join(integrity["issues"]))
    write_json(completed_path, {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "configuration_fingerprint": config["configuration_fingerprint"],
        "rows": len(results),
        "predictions_sha256": sha256_file(final_path),
        "metrics_sha256": sha256_file(run_dir / "metrics.json"),
        "integrity_sha256": sha256_file(run_dir / "integrity_report.json"),
    })
    return 0


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except (BaselineError, ArtifactIntegrityError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
