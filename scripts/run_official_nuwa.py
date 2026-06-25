#!/usr/bin/env python
"""Run official Nuwa through original LLaVA on the local RefCOCO/POPE manifests."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "experiments" / "official_nuwa_env" / "py_overlay"
LLAVA_ROOT = ROOT / "third_party" / "LLaVA"
NUWA_ROOT = ROOT / "third_party" / "Nuwa"

for candidate in (str(OVERLAY), str(LLAVA_ROOT), str(NUWA_ROOT), str(ROOT / "scripts")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

def _early_arg_value(name: str, default: str | None = None) -> str | None:
    prefix = f"{name}="
    for index, value in enumerate(sys.argv):
        if value == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if value.startswith(prefix):
            return value[len(prefix) :]
    return default


EARLY_CACHE_DIR = _early_arg_value("--cache-dir", os.environ.get("HF_HOME") or "D:/piptemp/hf_cache_codex")
if EARLY_CACHE_DIR:
    early_cache_path = Path(EARLY_CACHE_DIR)
    os.environ["HF_HOME"] = str(early_cache_path)
    os.environ["HF_HUB_CACHE"] = str(early_cache_path / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(early_cache_path / "hub")

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import run_baseline as base  # noqa: E402
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402
from nuwa import nuwa as apply_official_nuwa  # noqa: E402


ORIGINAL_VISUAL_TOKENS = 576
STAGE1_CONFIG = {64: 112, 128: 224, 192: 336}
STAGE2_CONFIG = {64: 16, 128: 32, 192: 48}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["grounding", "vqa"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=["official_nuwa", "official_full"])
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--matain-token", type=int, default=64, choices=[64, 128, 192])
    parser.add_argument("--distance", type=int, default=280)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/runs_official_nuwa"))
    parser.add_argument("--figures-root", type=Path, default=Path("experiments/figures_official_nuwa"))
    parser.add_argument("--tables-root", type=Path, default=Path("experiments/tables"))
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME") or "D:/piptemp/hf_cache_codex")
    parser.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-figures", type=int, default=24)
    parser.add_argument("--box-coordinate-mode", default="normalized_0_1", choices=["absolute_pixels", "normalized_0_1", "normalized_0_1000"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def run_name_from_args(args: argparse.Namespace, final_tokens: int) -> str:
    if args.run_name:
        return args.run_name
    model_name = "llava15"
    if args.method == "official_full":
        method = "official_full"
        setting = "full"
    else:
        method = "official_nuwa"
        setting = f"mt{args.matain_token}_d{args.distance}"
    return f"{args.dataset}_{model_name}_{args.task}_{method}_{setting}_tokens{final_tokens}_seed{args.seed}"


def official_prompt(task: str, sample: dict[str, Any], coordinate_mode: str) -> str:
    prompt = base.make_prompt(task, sample, coordinate_mode)
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def load_official_model(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    import torch

    os.environ["HF_HOME"] = args.cache_dir
    disable_torch_init()
    load_4bit = args.quantization == "4bit"
    load_8bit = args.quantization == "8bit"
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        None,
        model_name,
        load_4bit=load_4bit,
        load_8bit=load_8bit,
        device_map="auto",
        use_safetensors=False,
    )
    model.eval()
    if args.method == "official_nuwa":
        model.model.use_nuwa = True
        model = apply_official_nuwa(model, matain_token=args.matain_token, distance=args.distance)
    else:
        model.model.use_nuwa = False
    return torch, tokenizer, model, image_processor


def decode_generated(tokenizer: Any, output_ids: Any, input_len: int) -> tuple[str, int]:
    if output_ids.shape[1] > input_len:
        generated = output_ids[:, input_len:]
    else:
        generated = output_ids
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip(), int(generated.numel())


def tensor_to_list(value: Any) -> list[int]:
    if value is None:
        return []
    try:
        return [int(item) for item in value.detach().cpu().view(-1).tolist()]
    except Exception:
        return [int(item) for item in value]


def generate_one(
    torch: Any,
    tokenizer: Any,
    model: Any,
    image_processor: Any,
    image: Image.Image,
    prompt: str,
    args: argparse.Namespace,
    max_new_tokens: int,
) -> tuple[str, int, float, dict[str, Any]]:
    image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    original_sequence_length = int(input_ids.shape[1] - 1 + ORIGINAL_VISUAL_TOKENS)

    if torch.cuda.is_available():
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
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - started
    text, generated_count = decode_generated(tokenizer, output_ids, int(input_ids.shape[1]))

    if args.method == "official_nuwa":
        stage1_indices = tensor_to_list(getattr(model, "keep_idxs", None))
        stage2_indices = tensor_to_list(getattr(model.model, "_last_nuwa_stage2_source_indices", None))
        stage1_count = int(getattr(model.model, "_last_nuwa_stage1_count", len(stage1_indices)))
        stage2_count = int(getattr(model.model, "_last_nuwa_stage2_count", len(stage2_indices)))
        final_sequence_length = int(getattr(model.model, "_last_nuwa_final_sequence_length", input_ids.shape[1] - 1 + stage2_count))
        retained_indices = stage2_indices or stage1_indices
    else:
        stage1_indices = list(range(ORIGINAL_VISUAL_TOKENS))
        stage2_indices = list(range(ORIGINAL_VISUAL_TOKENS))
        stage1_count = ORIGINAL_VISUAL_TOKENS
        stage2_count = ORIGINAL_VISUAL_TOKENS
        final_sequence_length = original_sequence_length
        retained_indices = list(range(ORIGINAL_VISUAL_TOKENS))

    diagnostics = {
        "original_visual_tokens": ORIGINAL_VISUAL_TOKENS,
        "configured_matain_token": args.matain_token if args.method == "official_nuwa" else None,
        "configured_distance": args.distance if args.method == "official_nuwa" else None,
        "stage1_retained_visual_tokens": stage1_count,
        "stage2_retained_visual_tokens": stage2_count,
        "expected_retained_visual_tokens": stage2_count,
        "actual_retained_visual_tokens": stage2_count,
        "actual_llm_visual_tokens": stage2_count,
        "requested_keep_ratio": stage2_count / ORIGINAL_VISUAL_TOKENS,
        "stage1_keep_ratio": stage1_count / ORIGINAL_VISUAL_TOKENS,
        "original_input_sequence_length": original_sequence_length,
        "compacted_input_sequence_length": final_sequence_length,
        "sequence_length_reduction": original_sequence_length - final_sequence_length,
        "stage1_retained_indices": stage1_indices,
        "stage2_retained_indices": stage2_indices,
        "retained_indices": retained_indices,
        "final_visual_token_count": stage2_count,
        "official_nuwa_stage2_active": bool(args.method == "official_nuwa" and stage2_count < stage1_count),
    }
    return text, generated_count, latency, diagnostics


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def upsert_csv(path: Path, row: dict[str, Any], key: str = "run_name") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows = [item for item in rows if item.get(key) != row.get(key)]
    rows.append({k: "" if v is None else v for k, v in row.items()})
    fieldnames = sorted({field for item in rows for field in item})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_tables(config: dict[str, Any], metrics: dict[str, Any], tables_root: Path) -> None:
    row = dict(config)
    row.update(metrics)
    if config["task"] == "grounding":
        upsert_csv(tables_root / "official_nuwa_grounding.csv", row)
    else:
        upsert_csv(tables_root / "official_nuwa_vqa.csv", row)
    upsert_csv(tables_root / "official_nuwa_efficiency.csv", row)
    if config["method"] == "official_nuwa":
        token_row = {
            "run_name": config["run_name"],
            "task": config["task"],
            "dataset": config["dataset"],
            "matain_token": config["matain_token"],
            "distance": config["distance"],
            "original_visual_tokens": ORIGINAL_VISUAL_TOKENS,
            "official_stage1_expected": STAGE1_CONFIG.get(config["matain_token"]),
            "official_stage2_expected": STAGE2_CONFIG.get(config["matain_token"]),
            "stage1_retained_visual_tokens": metrics.get("stage1_retained_visual_tokens"),
            "stage2_retained_visual_tokens": metrics.get("stage2_retained_visual_tokens"),
            "final_visual_token_count": metrics.get("actual_llm_visual_tokens"),
            "mean_compacted_input_sequence_length": metrics.get("mean_compacted_input_sequence_length"),
        }
        upsert_csv(tables_root / "official_nuwa_token_mapping.csv", token_row)
    write_summary(tables_root)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_summary(tables_root: Path) -> None:
    rows = read_csv(tables_root / "official_nuwa_grounding.csv") + read_csv(tables_root / "official_nuwa_vqa.csv")
    eff = read_csv(tables_root / "official_nuwa_efficiency.csv")
    rows = [row for row in rows if str(row.get("limit")) in {"500", "500.0"} or str(row.get("num_samples")) in {"500", "500.0"}]
    eff = [row for row in eff if str(row.get("limit")) in {"500", "500.0"} or str(row.get("num_samples")) in {"500", "500.0"}]
    quality_cols = ["task", "method", "matain_token", "final_visual_tokens", "mean_iou", "accuracy_at_0.5", "accuracy", "f1"]
    eff_cols = ["task", "method", "matain_token", "actual_llm_visual_tokens", "mean_compacted_input_sequence_length", "mean_latency_s", "gpu_peak_memory_allocated_mb"]
    lines = [
        "# Official Nuwa 500-Example Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Quality",
        "",
        *base.markdown_table([{c: row.get(c, "") for c in quality_cols} for row in rows], quality_cols),
        "",
        "## Efficiency",
        "",
        *base.markdown_table([{c: row.get(c, "") for c in eff_cols} for row in eff], eff_cols),
        "",
    ]
    (tables_root / "official_nuwa_500_summary.md").write_text("\n".join(lines), encoding="utf-8")


def enrich_metrics(metrics: dict[str, Any], results: list[dict[str, Any]], config: dict[str, Any]) -> None:
    metrics["model_stack"] = "official_llava"
    metrics["official_nuwa_overlay"] = str(OVERLAY)
    metrics["official_nuwa_stage2_active_rate"] = base.mean_or_none([row.get("official_nuwa_stage2_active") for row in results])
    metrics["stage1_retained_visual_tokens"] = base.mean_or_none([row.get("stage1_retained_visual_tokens") for row in results])
    metrics["stage2_retained_visual_tokens"] = base.mean_or_none([row.get("stage2_retained_visual_tokens") for row in results])
    metrics["final_visual_tokens"] = metrics.get("actual_llm_visual_tokens")
    metrics["matain_token"] = config.get("matain_token")
    metrics["distance"] = config.get("distance")


def run(args: argparse.Namespace) -> int:
    os.environ["HF_HOME"] = args.cache_dir
    final_tokens = ORIGINAL_VISUAL_TOKENS if args.method == "official_full" else STAGE2_CONFIG[args.matain_token]
    keep_ratio = final_tokens / ORIGINAL_VISUAL_TOKENS
    run_name = run_name_from_args(args, final_tokens)
    run_dir = args.output_root / run_name
    figures_dir = args.figures_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_name": run_name,
        "task": args.task,
        "dataset": args.dataset,
        "method": args.method,
        "model": args.model_path,
        "model_stack": "official_llava",
        "quantization": args.quantization,
        "seed": args.seed,
        "limit": args.limit,
        "matain_token": args.matain_token if args.method == "official_nuwa" else None,
        "distance": args.distance if args.method == "official_nuwa" else None,
        "keep_ratio": keep_ratio,
        "token_budget": final_tokens,
        "box_coordinate_mode": args.box_coordinate_mode if args.task == "grounding" else None,
        "original_visual_tokens": ORIGINAL_VISUAL_TOKENS,
        "official_stage1_expected": STAGE1_CONFIG.get(args.matain_token) if args.method == "official_nuwa" else ORIGINAL_VISUAL_TOKENS,
        "official_stage2_expected": final_tokens,
        "comparison_scope": "official_stack",
        "hf_home": args.cache_dir,
        "llava_source": str(LLAVA_ROOT),
        "nuwa_source": str(NUWA_ROOT),
        "transformers_overlay": str(OVERLAY),
    }
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    rows = base.load_manifest(args.manifest, args.limit)
    existing: list[dict[str, Any]] = []
    per_sample_path = run_dir / "per_sample_results.jsonl"
    if args.resume:
        existing, malformed = base.load_existing_results(per_sample_path, rows)
        if malformed:
            base.rewrite_per_sample_results(per_sample_path, existing)
    completed = {str(row["sample_id"]) for row in existing}
    pending = [row for row in rows if str(row["sample_id"]) not in completed]

    print(f"Loaded {len(rows)} samples; pending {len(pending)}")
    torch, tokenizer, model, image_processor = load_official_model(args)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    max_new_tokens = args.max_new_tokens or (48 if args.task == "grounding" else 8)
    results = list(existing)
    started_all = time.perf_counter()
    mode = "a" if args.resume and per_sample_path.exists() else "w"
    with per_sample_path.open(mode, encoding="utf-8") as handle:
        for idx, sample in enumerate(pending):
            sample_id = str(sample["sample_id"])
            image = Image.open(sample["image_path"]).convert("RGB")
            prompt = official_prompt(args.task, sample, args.box_coordinate_mode)
            raw_output, generated_count, latency, diagnostics = generate_one(
                torch, tokenizer, model, image_processor, image, prompt, args, max_new_tokens
            )
            retained = diagnostics["retained_indices"]
            if args.task == "grounding":
                width = int(sample.get("width") or image.size[0])
                height = int(sample.get("height") or image.size[1])
                raw_box, clipped_box, parse_failure = base.parse_box(raw_output, width, height, args.box_coordinate_mode)
                valid = clipped_box is not None
                iou = base.box_iou(clipped_box, sample["gt_box"]) if valid else 0.0
                success = bool(valid and iou >= 0.5)
                result = {
                    "sample_id": sample_id,
                    "image_path": sample["image_path"],
                    "image_id": sample.get("image_id"),
                    "query": sample.get("query"),
                    "method": args.method,
                    "keep_ratio": keep_ratio,
                    "raw_model_output": raw_output,
                    "raw_parsed_box": raw_box,
                    "predicted_box": clipped_box,
                    "gt_box": sample["gt_box"],
                    "box_iou": iou,
                    "success_iou_0_5": success,
                    "valid": valid,
                    "invalid": not valid,
                    "parse_failure_reason": parse_failure,
                    "latency_s": latency,
                    "generated_token_count": generated_count,
                    "retained_indices": retained,
                    "retained_indices_checksum": base.indices_checksum(retained),
                }
                result.update(diagnostics)
                if idx < args.max_figures:
                    base.save_grounding_figure(sample, image, clipped_box, retained, 24, args.method, keep_ratio, iou, success, figures_dir)
            else:
                norm_answer, extraction_failure = base.normalise_yes_no(raw_output)
                target, target_failed = base.normalise_yes_no(str(sample["answer"]))
                correct = bool(norm_answer is not None and target is not None and norm_answer == target)
                result = {
                    "sample_id": sample_id,
                    "image_path": sample["image_path"],
                    "image_id": sample.get("image_id"),
                    "question": sample.get("question"),
                    "target_answer": sample.get("answer"),
                    "raw_model_answer": raw_output,
                    "normalised_answer": norm_answer,
                    "correct": correct,
                    "invalid": bool(extraction_failure),
                    "extraction_failure": bool(extraction_failure or target_failed),
                    "method": args.method,
                    "keep_ratio": keep_ratio,
                    "latency_s": latency,
                    "generated_token_count": generated_count,
                    "retained_indices": retained,
                    "retained_indices_checksum": base.indices_checksum(retained),
                }
                result.update(diagnostics)
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=True) + "\n")
            if len(results) % 10 == 0 or len(results) == len(rows):
                print(f"Processed {len(results)}/{len(rows)}")

    total_wall = time.perf_counter() - started_all
    if existing:
        total_wall += sum(float(row.get("latency_s") or 0.0) for row in existing)
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None
    metrics = base.aggregate_grounding(results, total_wall, peak_memory) if args.task == "grounding" else base.aggregate_vqa(results, total_wall, peak_memory)
    enrich_metrics(metrics, results, config)
    write_json(run_dir / "metrics.json", metrics)
    run_env = {
        "python": sys.executable,
        "torch": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hf_home": args.cache_dir,
    }
    write_json(run_dir / "run_environment.json", run_env)
    update_tables(config, metrics, args.tables_root)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except base.BaselineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
