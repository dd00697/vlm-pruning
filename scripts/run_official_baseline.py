#!/usr/bin/env python
"""Run official full/random/uniform true-token-removal LLaVA baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
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

from run_baseline import (  # noqa: E402
    DEFAULT_GRID,
    DEFAULT_MODEL,
    TRUE_REMOVAL_LABEL,
    TRUE_REMOVAL_LOCATION,
    BaselineError,
    Tee,
    box_iou,
    generate_answer_true_removal,
    indices_checksum,
    kv_cache_memory_mb,
    llm_prefill_flops,
    parse_box,
    save_grounding_figure,
)
from vlm_pruning.analysis.coverage import selection_diagnostics  # noqa: E402
from vlm_pruning.selectors import SelectionResult, SelectorError, selector_for_method  # noqa: E402


DEFAULT_PROJECT = "vlm-pruning"
OUTPUT_ROOT = Path("experiments/runs_official_baselines")
FIGURES_ROOT = Path("experiments/figures_official_baselines")
TABLES_ROOT = Path("experiments/tables/official_baselines")
REPORTS_ROOT = Path("experiments/reports/official_baselines")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True, choices=["vqa", "grounding"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=["full", "random", "uniform"])
    parser.add_argument("--target-visual-tokens", "--token-budget", dest="target_visual_tokens", type=int)
    parser.add_argument("--keep-ratio", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--model-path", "--model", dest="model_path", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--figures-root", type=Path, default=FIGURES_ROOT)
    parser.add_argument("--tables-root", type=Path, default=TABLES_ROOT)
    parser.add_argument("--reports-root", type=Path, default=REPORTS_ROOT)
    parser.add_argument("--image-grid-size", type=int, default=DEFAULT_GRID)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-figures", type=int, default=24)
    parser.add_argument("--box-coordinate-mode", default="normalized_0_1", choices=["normalized_0_1", "normalized_0_1000", "absolute_pixels"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"), choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-job-type", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--estimate-runtime", action="store_true")
    parser.add_argument("--suite-total-runs", type=int, default=None)
    parser.add_argument("--suite-completed-runs", type=int, default=None)
    parser.add_argument("--suite-start-time", type=float, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        return bool(subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).strip())
    except Exception:
        return None


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "value"


def load_manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise BaselineError(f"Manifest does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise BaselineError(f"Manifest is empty: {path}")
    return rows


def manifest_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def target_tokens(args: argparse.Namespace) -> int:
    original = args.image_grid_size * args.image_grid_size
    if args.method == "full":
        return original
    if args.target_visual_tokens is not None:
        return max(1, min(original, args.target_visual_tokens))
    if args.keep_ratio is not None:
        return max(1, min(original, int(round(original * args.keep_ratio))))
    raise BaselineError("Pruned runs require --target-visual-tokens or --keep-ratio.")


def keep_ratio(target: int, original: int) -> float:
    return float(target) / float(original)


def budget_family(tokens: int, original: int) -> str:
    if tokens >= original:
        return "full"
    if tokens == 432:
        return "gentle"
    if tokens in {48, 32, 16}:
        return "stress"
    return "broad"


def run_name(args: argparse.Namespace, tokens: int) -> str:
    if args.run_name:
        return args.run_name
    seed = args.seed if args.method == "random" else 0
    suffix = f"limit{args.limit}_" if args.limit is not None else ""
    return f"{suffix}{safe_slug(args.dataset)}_{args.method}_tokens{tokens}_seed{seed}_{args.precision}"


def torch_dtype(torch: Any, precision: str) -> Any:
    return torch.bfloat16 if precision == "bf16" else torch.float16


def load_model(model_path: str, precision: str) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    try:
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration
    except Exception as exc:
        raise BaselineError(f"Missing model dependency: {type(exc).__name__}: {exc}") from exc
    if not torch.cuda.is_available():
        raise BaselineError("CUDA GPU is required for full-precision official baseline runs.")
    dtype = torch_dtype(torch, precision)
    device = torch.device("cuda:0")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    env = {
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_gb": total_bytes / 1024**3,
        "gpu_free_memory_gb_at_load_start": free_bytes / 1024**3,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": precision,
        "torch_dtype": str(dtype),
    }
    try:
        processor = AutoProcessor.from_pretrained(model_path)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
    except Exception as exc:
        raise BaselineError(f"Full-precision LLaVA load failed for '{model_path}': {type(exc).__name__}: {exc}") from exc
    return torch, processor, model, device, env


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def canonical_text(value: Any) -> str:
    text = str(value if value is not None else "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    articles = {"a", "an", "the"}
    return " ".join(word for word in text.split() if word not in articles)


def normalise_yes_no(text: str) -> tuple[str | None, bool]:
    cleaned = re.sub(r"[^A-Za-z]+", " ", text.lower()).strip()
    if not cleaned:
        return None, True
    first = cleaned.split()[0]
    if first in {"yes", "no"}:
        return first, False
    if "yes" in cleaned.split():
        return "yes", False
    if "no" in cleaned.split():
        return "no", False
    return None, True


def choice_label(index: int) -> str:
    return chr(ord("A") + index)


def answer_mode(sample: dict[str, Any], dataset: str) -> str:
    explicit = str(sample.get("answer_type") or sample.get("metric") or "").lower()
    if "yes" in explicit or "binary" in explicit:
        return "yes_no"
    if "multi" in explicit or "choice" in explicit:
        return "multiple_choice"
    answer = str(sample.get("answer") or "").strip().lower()
    choices = sample.get("choices") or []
    if dataset.lower() == "pope" or answer in {"yes", "no"}:
        return "yes_no"
    if isinstance(choices, list) and choices:
        return "multiple_choice"
    return "open"


def make_prompt(args: argparse.Namespace, sample: dict[str, Any]) -> str:
    if args.task == "grounding":
        if args.box_coordinate_mode == "normalized_0_1":
            coord = "normalized 0-1 image coordinates"
        elif args.box_coordinate_mode == "normalized_0_1000":
            coord = "normalized 0-1000 image coordinates"
        else:
            coord = "absolute image pixel coordinates"
        return (
            f"Identify the target described by this referring expression: \"{sample['query']}\".\n"
            f"Return exactly one bounding box in {coord} as [x_min, y_min, x_max, y_max]. "
            "Do not include any other text."
        )
    question = str(sample["question"]).strip()
    mode = answer_mode(sample, args.dataset)
    if mode == "yes_no":
        return f"{question}\nAnswer with exactly one word: yes or no."
    choices = sample.get("choices") or []
    if mode == "multiple_choice" and choices:
        lines = [question, ""]
        for idx, choice in enumerate(choices):
            lines.append(f"{choice_label(idx)}. {choice}")
        lines.append("\nAnswer with the option letter only.")
        return "\n".join(lines)
    return f"{question}\nAnswer as briefly as possible."


def extract_choice(raw: str, choices: list[Any]) -> tuple[str | None, str | None]:
    text = str(raw or "").strip()
    match = re.search(r"\b([A-Z])\b", text.upper())
    if match:
        label = match.group(1)
        idx = ord(label) - ord("A")
        if 0 <= idx < len(choices):
            return label, str(choices[idx])
    canonical_raw = canonical_text(text)
    for idx, choice in enumerate(choices):
        if canonical_text(choice) and canonical_text(choice) in canonical_raw:
            return choice_label(idx), str(choice)
    return None, None


def target_choice(answer: Any, choices: list[Any]) -> tuple[str | None, str | None]:
    if answer is None:
        return None, None
    text = str(answer).strip()
    upper = text.upper()
    if len(upper) == 1 and "A" <= upper <= "Z":
        idx = ord(upper) - ord("A")
        if 0 <= idx < len(choices):
            return upper, str(choices[idx])
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(choices):
            return choice_label(idx), str(choices[idx])
        if 1 <= idx <= len(choices):
            return choice_label(idx - 1), str(choices[idx - 1])
    canon = canonical_text(text)
    for idx, choice in enumerate(choices):
        if canon == canonical_text(choice):
            return choice_label(idx), str(choice)
    return None, text


def score_vqa(sample: dict[str, Any], raw: str, dataset: str) -> dict[str, Any]:
    mode = answer_mode(sample, dataset)
    answer = sample.get("answer")
    choices = sample.get("choices") if isinstance(sample.get("choices"), list) else []
    if mode == "yes_no":
        pred, pred_failed = normalise_yes_no(raw)
        target, target_failed = normalise_yes_no(str(answer))
        correct = bool(pred is not None and target is not None and pred == target)
        return {
            "answer_mode": mode,
            "prediction": pred,
            "target": target,
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "invalid": bool(pred_failed or target_failed),
            "invalid_reason": "answer_extraction_failed" if pred_failed else ("target_extraction_failed" if target_failed else None),
        }
    if mode == "multiple_choice":
        pred_label, pred_text = extract_choice(raw, choices)
        target_label, target_text = target_choice(answer, choices)
        correct = bool(pred_label is not None and target_label is not None and pred_label == target_label)
        if not correct and pred_text is not None and target_text is not None:
            correct = canonical_text(pred_text) == canonical_text(target_text)
        return {
            "answer_mode": mode,
            "prediction": pred_label or pred_text,
            "prediction_text": pred_text,
            "target": target_label or target_text,
            "target_text": target_text,
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "invalid": pred_label is None and pred_text is None,
            "invalid_reason": "choice_extraction_failed" if pred_label is None and pred_text is None else None,
        }
    target_values = answer if isinstance(answer, list) else [answer]
    pred_canon = canonical_text(raw)
    target_canons = [canonical_text(value) for value in target_values if value is not None]
    correct = bool(pred_canon and pred_canon in target_canons)
    return {
        "answer_mode": mode,
        "prediction": raw.strip(),
        "target": target_values,
        "correct": correct,
        "score": 1.0 if correct else 0.0,
        "invalid": not bool(pred_canon),
        "invalid_reason": "empty_prediction" if not pred_canon else None,
    }


def median_or_none(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.median(clean) if clean else None


def mean_or_none(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def percentile(values: list[float], p: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * p
    lower = int(pos)
    upper = min(lower + 1, len(clean) - 1)
    frac = pos - lower
    return clean[lower] * (1 - frac) + clean[upper] * frac


def base_metrics(results: list[dict[str, Any]], wall_time_s: float, peak_alloc_mb: float | None, peak_reserved_mb: float | None) -> dict[str, Any]:
    latencies = [row.get("latency_s") for row in results]
    generated = [row.get("generated_token_count") for row in results]
    decode = [row.get("decode_latency_s") for row in results]
    prefill = [row.get("llm_prefill_latency_s") for row in results]
    pruning = [row.get("pruning_selection_latency_s") for row in results]
    metrics = {
        "samples/evaluated": len(results),
        "samples/skipped": 0,
        "samples/invalid_outputs": sum(1 for row in results if row.get("invalid")),
        "quality/valid_output_rate": sum(1 for row in results if not row.get("invalid")) / len(results) if results else 0.0,
        "efficiency/mean_latency_s": mean_or_none(latencies),
        "efficiency/median_latency_s": median_or_none(latencies),
        "efficiency/p95_latency_s": percentile(latencies, 0.95),
        "efficiency/total_wall_time_s": wall_time_s,
        "efficiency/peak_gpu_memory_mb": peak_alloc_mb,
        "efficiency/peak_gpu_memory_reserved_mb": peak_reserved_mb,
        "efficiency/time_to_first_token_s": mean_or_none([row.get("time_to_first_token_s") for row in results]),
        "efficiency/llm_prefill_latency_s": mean_or_none(prefill),
        "efficiency/decode_latency_s": mean_or_none(decode),
        "efficiency/vision_encoder_latency_s": mean_or_none([row.get("vision_encoder_latency_s") for row in results]),
        "efficiency/projector_latency_s": mean_or_none([row.get("projector_latency_s") for row in results]),
        "efficiency/pruning_selection_latency_s": mean_or_none(pruning),
        "efficiency/throughput_examples_per_s": len(results) / wall_time_s if wall_time_s > 0 else None,
        "efficiency/generated_token_count_mean": mean_or_none(generated),
        "efficiency/generated_token_count_total": sum(int(x or 0) for x in generated),
        "tokens/original_visual_tokens": mean_or_none([row.get("original_visual_tokens") for row in results]),
        "tokens/final_visual_tokens": mean_or_none([row.get("actual_llm_visual_tokens") for row in results]),
        "tokens/actual_visual_tokens": mean_or_none([row.get("actual_llm_visual_tokens") for row in results]),
        "tokens/keep_ratio": mean_or_none([row.get("requested_keep_ratio") for row in results]),
        "efficiency/original_sequence_length": mean_or_none([row.get("original_input_sequence_length") for row in results]),
        "efficiency/sequence_length": mean_or_none([row.get("compacted_input_sequence_length") for row in results]),
        "efficiency/sequence_length_reduction": mean_or_none([row.get("sequence_length_reduction") for row in results]),
    }
    gen = metrics["efficiency/generated_token_count_total"]
    decode_total = sum(float(x or 0.0) for x in decode)
    metrics["efficiency/output_tokens_per_s"] = gen / decode_total if decode_total > 0 else None
    metrics["efficiency/time_per_output_token_s"] = decode_total / gen if gen else None
    mean_latency = metrics["efficiency/mean_latency_s"]
    mean_pruning = metrics["efficiency/pruning_selection_latency_s"]
    metrics["efficiency/pruning_overhead_fraction"] = mean_pruning / mean_latency if mean_latency else None
    original_seq = metrics["efficiency/original_sequence_length"]
    compact_seq = metrics["efficiency/sequence_length"]
    reduction = metrics["efficiency/sequence_length_reduction"]
    metrics["efficiency/sequence_length_reduction_percent"] = (reduction / original_seq * 100.0) if original_seq else None
    orig_tokens = metrics["tokens/original_visual_tokens"]
    final_tokens = metrics["tokens/final_visual_tokens"]
    metrics["tokens/token_reduction_percent"] = ((orig_tokens - final_tokens) / orig_tokens * 100.0) if orig_tokens else None
    return metrics


def add_model_efficiency(metrics: dict[str, Any], model: Any, torch: Any) -> None:
    compact = metrics.get("efficiency/sequence_length")
    full = metrics.get("efficiency/original_sequence_length")
    compact_flops = llm_prefill_flops(float(compact), model) if compact else None
    full_flops = llm_prefill_flops(float(full), model) if full else None
    metrics["efficiency/theoretical_prefill_flops"] = compact_flops
    metrics["efficiency/theoretical_prefill_flops_full_reference"] = full_flops
    metrics["efficiency/theoretical_prefill_flops_ratio"] = compact_flops / full_flops if compact_flops and full_flops else None
    metrics["efficiency/theoretical_prefill_flops_reduction_percent"] = (1.0 - compact_flops / full_flops) * 100.0 if compact_flops and full_flops else None
    dtype = next(model.parameters()).dtype
    kv = kv_cache_memory_mb(float(compact), model, dtype, torch) if compact else None
    full_kv = kv_cache_memory_mb(float(full), model, dtype, torch) if full else None
    metrics["efficiency/kv_cache_memory_estimate_mb"] = kv
    metrics["efficiency/kv_cache_memory_full_reference_mb"] = full_kv
    metrics["efficiency/kv_cache_memory_reduction_percent"] = (1.0 - kv / full_kv) * 100.0 if kv and full_kv else None


def aggregate(results: list[dict[str, Any]], args: argparse.Namespace, wall_time_s: float, peak_alloc_mb: float | None, peak_reserved_mb: float | None) -> dict[str, Any]:
    metrics = base_metrics(results, wall_time_s, peak_alloc_mb, peak_reserved_mb)
    if args.task == "grounding":
        valid = [row for row in results if row.get("valid")]
        ious = [float(row.get("box_iou") or 0.0) for row in valid]
        metrics.update(
            {
                "quality/main_score": sum(1 for row in valid if row.get("success_iou_0_5")) / len(results) if results else 0.0,
                "quality/mean_iou": sum(ious) / len(ious) if ious else 0.0,
                "quality/median_iou": statistics.median(ious) if ious else 0.0,
                "quality/acc_at_0_5": sum(1 for row in valid if row.get("success_iou_0_5")) / len(results) if results else 0.0,
                "quality/parsed_box_rate": sum(1 for row in results if row.get("raw_parsed_box") is not None) / len(results) if results else 0.0,
            }
        )
    else:
        correct = sum(1 for row in results if row.get("correct"))
        yes_rows = [row for row in results if row.get("answer_mode") == "yes_no"]
        tp = sum(1 for row in yes_rows if row.get("prediction") == "yes" and row.get("target") == "yes")
        fp = sum(1 for row in yes_rows if row.get("prediction") == "yes" and row.get("target") == "no")
        fn = sum(1 for row in yes_rows if row.get("prediction") == "no" and row.get("target") == "yes")
        pred_yes = sum(1 for row in yes_rows if row.get("prediction") == "yes")
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and (precision + recall) else None
        acc = correct / len(results) if results else 0.0
        metrics.update(
            {
                "quality/main_score": acc,
                "quality/accuracy": acc,
                "quality/exact_match": acc,
                "quality/precision": precision,
                "quality/recall": recall,
                "quality/f1": f1,
                "quality/yes_rate": pred_yes / len(yes_rows) if yes_rows else None,
            }
        )
    return metrics


def load_existing_results(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    wanted = {str(row["sample_id"]) for row in rows}
    by_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(row.get("sample_id"))
            if sid in wanted and sid not in by_id:
                by_id[sid] = row
    return [by_id[str(row["sample_id"])] for row in rows if str(row["sample_id"]) in by_id]


def env_snapshot(args: argparse.Namespace, model_env: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "wandb_project": args.wandb_project,
        "wandb_dir": os.environ.get("WANDB_DIR"),
        "hf_home": os.environ.get("HF_HOME"),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
    }
    if model_env:
        payload.update(model_env)
    return payload


def config_payload(args: argparse.Namespace, name: str, original_tokens: int, tokens: int, checksum: str) -> dict[str, Any]:
    split = args.split or "unknown"
    return {
        "run_name": name,
        "created_at_utc": utc_now(),
        "model/name": args.model_path,
        "model/stack": "huggingface_llava",
        "model/precision": args.precision,
        "dataset/name": args.dataset,
        "dataset/split": split,
        "dataset/manifest": str(args.manifest),
        "dataset/manifest_checksum": checksum,
        "task": args.task,
        "method/name": args.method,
        "method/seed": args.seed,
        "tokens/original_visual_tokens": original_tokens,
        "tokens/target_visual_tokens": tokens,
        "tokens/actual_visual_tokens": tokens,
        "tokens/keep_ratio": keep_ratio(tokens, original_tokens),
        "tokens/budget_family": budget_family(tokens, original_tokens),
        "run/is_limited": args.limit is not None,
        "run/output_dir": None,
        "pruning_implementation": TRUE_REMOVAL_LABEL,
        "pruning_location": TRUE_REMOVAL_LOCATION,
        "evaluator/name": evaluator_name(args.dataset, args.task),
        "evaluator/metric_definition": metric_definition(args.dataset, args.task),
        "limit": args.limit,
    }


def evaluator_name(dataset: str, task: str) -> str:
    if task == "grounding":
        return "box_iou_acc_at_0_5"
    name = dataset.lower()
    if name == "pope":
        return "yes_no_accuracy_f1"
    if name in {"aokvqa", "a-okvqa", "scienceqa_img", "scienceqa", "mmmu", "seed_bench", "seed-bench"}:
        return "multiple_choice_accuracy"
    if name in {"ocr_vqa", "ocr-vqa", "textvqa"}:
        return "normalized_exact_match"
    return "manifest_accuracy"


def metric_definition(dataset: str, task: str) -> str:
    if task == "grounding":
        return "Parse one predicted box and compute mean IoU, median IoU, and Acc@0.5 against manifest gt_box."
    if dataset.lower() == "pope":
        return "Normalize yes/no answers and compute accuracy, precision, recall, F1, and yes-rate."
    return "Dataset manifest driven scoring: multiple-choice accuracy when choices exist, otherwise normalized exact match."


def init_wandb(args: argparse.Namespace, config: dict[str, Any], run_dir: Path, tags: list[str]) -> Any | None:
    if args.no_wandb or not args.wandb or args.wandb_mode == "disabled":
        return None
    try:
        import wandb

        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_mode:
            os.environ["WANDB_MODE"] = args.wandb_mode
        kwargs = {
            "project": args.wandb_project,
            "name": config["run_name"],
            "config": config,
            "dir": str(run_dir),
            "reinit": True,
            "tags": tags,
            "group": args.wandb_group or f"official-baselines-{safe_slug(args.dataset)}",
            "job_type": args.wandb_job_type or job_type(config),
        }
        if args.wandb_entity:
            kwargs["entity"] = args.wandb_entity
        return wandb.init(**kwargs)
    except Exception as exc:
        print(f"W&B init failed; continuing with local outputs. {type(exc).__name__}: {exc}")
        return None


def job_type(config: dict[str, Any]) -> str:
    if config.get("run/is_limited"):
        return "limited-run"
    method = config["method/name"]
    if method == "full":
        return "full-baseline"
    family = config.get("tokens/budget_family")
    if family == "gentle":
        return "gentle-budget"
    if family == "stress":
        return "stress-budget"
    return f"{method}-baseline"


def wandb_tags(args: argparse.Namespace, config: dict[str, Any], env: dict[str, Any] | None = None) -> list[str]:
    tags = [
        "official-baseline",
        safe_slug(args.dataset),
        args.task,
        args.method,
        f"tokens-{config['tokens/actual_visual_tokens']}",
        str(config["tokens/budget_family"]),
        f"seed-{args.seed}",
        args.precision,
        "llava-1.5-7b" if "1.5-7b" in args.model_path.lower() else safe_slug(args.model_path),
        "true-token-removal",
    ]
    if env and env.get("gpu_name"):
        tags.append(safe_slug(str(env["gpu_name"])))
    tags.extend(safe_slug(item) for item in args.wandb_tags.split(",") if item.strip())
    return tags


def wandb_run_info(run: Any | None, args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    if run is None:
        return {"enabled": False, "project": args.wandb_project, "mode": args.wandb_mode}
    return {
        "enabled": True,
        "project": args.wandb_project,
        "entity": getattr(run, "entity", None),
        "id": getattr(run, "id", None),
        "name": getattr(run, "name", None),
        "url": getattr(run, "url", None),
        "group": args.wandb_group or f"official-baselines-{safe_slug(args.dataset)}",
        "job_type": args.wandb_job_type or job_type(config),
        "mode": args.wandb_mode,
    }


def log_wandb(run: Any | None, metrics: dict[str, Any], results: list[dict[str, Any]], run_dir: Path, figure_paths: list[Path]) -> None:
    if run is None:
        return
    try:
        import wandb

        run.log(metrics)
        for key in [
            "quality/main_score",
            "efficiency/mean_latency_s",
            "efficiency/theoretical_prefill_flops_ratio",
            "tokens/final_visual_tokens",
            "runtime/estimated_remaining_wall_time_s",
        ]:
            if key in metrics:
                run.summary[key] = metrics[key]
        table = wandb.Table(
            columns=[
                "sample_id",
                "dataset",
                "split",
                "method",
                "seed",
                "target_visual_tokens",
                "actual_visual_tokens",
                "question_or_query",
                "ground_truth",
                "prediction",
                "is_correct",
                "score",
                "latency_s",
                "invalid_reason",
                "image_path",
            ]
        )
        for row in results:
            table.add_data(
                row.get("sample_id"),
                row.get("dataset"),
                row.get("split"),
                row.get("method"),
                row.get("seed"),
                row.get("target_visual_tokens"),
                row.get("actual_llm_visual_tokens"),
                row.get("question") or row.get("query"),
                row.get("target") or row.get("gt_box"),
                row.get("prediction") or row.get("predicted_box"),
                row.get("correct") if "correct" in row else row.get("success_iou_0_5"),
                row.get("score") if "score" in row else row.get("box_iou"),
                row.get("latency_s"),
                row.get("invalid_reason") or row.get("parse_failure_reason"),
                row.get("image_path"),
            )
        run.log({"per_sample_results": table})
        for figure in figure_paths[:8]:
            run.log({"qualitative_refcoco": wandb.Image(str(figure))})
        artifact = wandb.Artifact(f"{safe_slug(run.name)}-outputs", type="official-baseline-run")
        for path in ["config.yaml", "metrics.json", "run_environment.json", "per_sample_results.jsonl", "wandb_run.json"]:
            candidate = run_dir / path
            if candidate.exists():
                artifact.add_file(str(candidate))
        run.log_artifact(artifact)
    except Exception as exc:
        print(f"W&B logging failed after local outputs were saved. {type(exc).__name__}: {exc}")


def runtime_estimate(args: argparse.Namespace, run_duration_s: float) -> dict[str, Any]:
    now = time.time()
    completed = args.suite_completed_runs
    total = args.suite_total_runs
    start = args.suite_start_time
    if completed is None or total is None:
        completed = None
        remaining = None
    else:
        completed = completed + 1
        remaining = max(0, total - completed)
    elapsed = now - start if start else None
    if remaining is not None:
        if elapsed and completed:
            average = elapsed / completed
        else:
            average = run_duration_s
        remaining_s = remaining * average
    else:
        average = None
        remaining_s = None
    eta = datetime.fromtimestamp(now + remaining_s, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z") if remaining_s is not None else None
    return {
        "runtime/completed_runs": completed,
        "runtime/remaining_runs": remaining,
        "runtime/elapsed_wall_time_s": elapsed,
        "runtime/latest_run_wall_time_s": run_duration_s,
        "runtime/mean_run_wall_time_s": average,
        "runtime/estimated_remaining_wall_time_s": remaining_s,
        "runtime/estimated_total_wall_time_s": (elapsed + remaining_s) if elapsed is not None and remaining_s is not None else None,
        "runtime/eta_utc": eta,
        "updated_at_utc": utc_now(),
    }


def update_runtime_files(tables_root: Path, estimate: dict[str, Any]) -> None:
    write_json(tables_root / "runtime_estimates.json", estimate)
    append_jsonl(tables_root / "runtime_estimates_history.jsonl", estimate)


def reference_key(config: dict[str, Any]) -> tuple[str, str]:
    return str(config["task"]), str(config["dataset/name"])


def read_full_references(tables_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    refs: dict[tuple[str, str], dict[str, Any]] = {}
    for path in [tables_root / "vqa_quality_by_budget.csv", tables_root / "refcoco_quality_by_budget.csv"]:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("method/name") == "full":
                    refs[(row.get("task", ""), row.get("dataset/name", ""))] = row
    return refs


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def add_relative_metrics(metrics: dict[str, Any], config: dict[str, Any], tables_root: Path) -> None:
    refs = read_full_references(tables_root)
    ref = refs.get(reference_key(config))
    if not ref:
        if config["method/name"] == "full":
            metrics["quality/retention_percent_vs_full"] = 100.0
            metrics["quality/drop_vs_full"] = 0.0
            metrics["efficiency/end_to_end_speedup_vs_full"] = 1.0
            metrics["efficiency/prefill_speedup_vs_full"] = 1.0
        return
    quality = metrics.get("quality/main_score")
    full_quality = parse_float(ref.get("quality/main_score"))
    latency = metrics.get("efficiency/mean_latency_s")
    full_latency = parse_float(ref.get("efficiency/mean_latency_s"))
    prefill = metrics.get("efficiency/llm_prefill_latency_s")
    full_prefill = parse_float(ref.get("efficiency/llm_prefill_latency_s"))
    if quality is not None and full_quality:
        metrics["quality/retention_percent_vs_full"] = float(quality) / full_quality * 100.0
        metrics["quality/drop_vs_full"] = float(quality) - full_quality
    if latency and full_latency:
        metrics["efficiency/end_to_end_speedup_vs_full"] = full_latency / float(latency)
    if prefill and full_prefill:
        metrics["efficiency/prefill_speedup_vs_full"] = full_prefill / float(prefill)
    final_tokens = metrics.get("tokens/final_visual_tokens")
    tflops = metrics.get("efficiency/theoretical_prefill_flops")
    if quality is not None and final_tokens:
        metrics["efficiency/quality_per_retained_token"] = float(quality) / float(final_tokens)
    if quality is not None and tflops:
        metrics["efficiency/quality_per_tflop_estimate"] = float(quality) / (float(tflops) / 1e12)
    if quality is not None and latency:
        metrics["efficiency/quality_per_second"] = float(quality) / float(latency)


def update_csv(path: Path, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**config, **metrics}
    key = row["run_name"]
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(item) for item in csv.DictReader(handle)]
        rows = [item for item in rows if item.get("run_name") != key]
    rows.append({k: "" if v is None else v for k, v in row.items() if not isinstance(v, (dict, list))})
    fields = sorted({field for item in rows for field in item.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def update_tables(config: dict[str, Any], metrics: dict[str, Any], args: argparse.Namespace) -> None:
    if config["run/is_limited"]:
        return
    if args.task == "grounding":
        update_csv(args.tables_root / "refcoco_diagnostic.csv", config, metrics)
        update_csv(args.tables_root / "refcoco_quality_by_budget.csv", config, metrics)
        update_csv(args.tables_root / "refcoco_efficiency.csv", config, metrics)
    else:
        update_csv(args.tables_root / "vqa_quality_by_dataset.csv", config, metrics)
        update_csv(args.tables_root / "vqa_quality_by_budget.csv", config, metrics)
        update_csv(args.tables_root / "vqa_efficiency.csv", config, metrics)
        update_csv(args.tables_root / "vqa_budget_retention.csv", config, metrics)


def completed_run_matches(run_dir: Path, args: argparse.Namespace, tokens: int, row_count: int, checksum: str) -> bool:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    if not metrics_path.exists() or not config_path.exists():
        return False
    try:
        config = load_json(config_path)
        metrics = load_json(metrics_path)
    except Exception:
        return False
    expected_seed = args.seed if args.method == "random" else 0
    checks = [
        config.get("dataset/manifest_checksum") == checksum,
        int(config.get("dataset/num_samples", -1)) == int(row_count),
        config.get("method/name") == args.method,
        int(config.get("method/seed", -1)) == int(expected_seed),
        int(config.get("tokens/target_visual_tokens", -1)) == int(tokens),
        int(config.get("tokens/actual_visual_tokens", -1)) == int(tokens),
        config.get("model/precision") == args.precision,
        config.get("limit") == args.limit,
        int(metrics.get("samples/evaluated", -1)) == int(row_count),
    ]
    return all(checks)


def build_result_common(
    args: argparse.Namespace,
    sample: dict[str, Any],
    image_path: Path,
    raw_output: str,
    generated_count: int,
    latency: float,
    selection: SelectionResult,
    diagnostics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    retained = selection.selected_indices
    result = {
        "sample_id": str(sample["sample_id"]),
        "dataset": args.dataset,
        "split": sample.get("split") or args.split,
        "image_path": str(image_path.as_posix()),
        "image_id": sample.get("image_id"),
        "method": args.method,
        "seed": args.seed,
        "target_visual_tokens": config["tokens/target_visual_tokens"],
        "keep_ratio": config["tokens/keep_ratio"],
        "raw_model_output": raw_output,
        "latency_s": latency,
        "generated_token_count": generated_count,
        "retained_indices_checksum": indices_checksum(retained),
        "retained_indices": retained,
        "selector_name": selection.method,
        "selector_metadata": selection.as_metadata(),
        "role_labels": selection.role_labels,
        "pruning_implementation": TRUE_REMOVAL_LABEL,
    }
    result.update(diagnostics)
    return result


def run(args: argparse.Namespace) -> int:
    original_tokens = args.image_grid_size * args.image_grid_size
    tokens = target_tokens(args)
    if args.method == "full" and tokens != original_tokens:
        raise BaselineError("Full-token runs must retain all 576 visual tokens.")
    try:
        selector = selector_for_method(args.method)
    except SelectorError as exc:
        raise BaselineError(str(exc)) from exc
    name = run_name(args, tokens)
    run_dir = args.output_root / name
    figures_dir = args.figures_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if args.resume else "w"
    with (run_dir / "stdout.log").open(log_mode, encoding="utf-8") as log_handle:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(sys.stdout, log_handle), Tee(sys.stderr, log_handle)
        try:
            rows = load_manifest(args.manifest, args.limit)
            split = args.split or str(rows[0].get("split") or "unknown")
            args.split = split
            checksum = manifest_checksum(args.manifest)
            if args.skip_existing and completed_run_matches(run_dir, args, tokens, len(rows), checksum):
                print(f"Skipping existing completed run: {run_dir}")
                return 0
            config = config_payload(args, name, original_tokens, tokens, checksum)
            config["dataset/num_samples"] = len(rows)
            config["run/output_dir"] = str(run_dir)
            config["selector/name"] = selector.name
            config["selector/interface_version"] = "phase0"
            write_json(run_dir / "config.json", config)
            with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=True)

            per_sample_path = run_dir / "per_sample_results.jsonl"
            existing = load_existing_results(per_sample_path, rows) if args.resume else []
            completed = {str(row["sample_id"]) for row in existing}
            pending = [row for row in rows if str(row["sample_id"]) not in completed]
            print(f"Loaded {len(rows)} samples from {args.manifest}; pending={len(pending)} completed={len(existing)}")

            model_load_start = time.perf_counter()
            torch, processor, model, device, model_env = load_model(args.model_path, args.precision)
            model_load_s = time.perf_counter() - model_load_start
            env = env_snapshot(args, model_env)
            env["model_load_latency_s"] = model_load_s
            write_json(run_dir / "run_environment.json", env)
            tags = wandb_tags(args, config, env)
            wandb_run = init_wandb(args, config, run_dir, tags)
            write_json(run_dir / "wandb_run.json", wandb_run_info(wandb_run, args, config))
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            results = list(existing)
            figure_paths: list[Path] = []
            max_new_tokens = args.max_new_tokens or (48 if args.task == "grounding" else 16)
            run_start = time.perf_counter()
            sample_mode = "a" if args.resume and per_sample_path.exists() else "w"
            with per_sample_path.open(sample_mode, encoding="utf-8") as handle:
                for index, sample in enumerate(pending):
                    sample_id = str(sample["sample_id"])
                    image_path = Path(sample["image_path"])
                    image = Image.open(image_path).convert("RGB")
                    try:
                        selection = selector.select(
                            sample_id=sample_id,
                            budget=tokens,
                            original_tokens=original_tokens,
                            grid=args.image_grid_size,
                            seed=args.seed,
                        )
                    except SelectorError as exc:
                        raise BaselineError(str(exc)) from exc
                    retained = selection.selected_indices
                    prompt = make_prompt(args, sample)
                    raw_output, generated_count, latency, diagnostics = generate_answer_true_removal(
                        torch,
                        processor,
                        model,
                        device,
                        image,
                        prompt,
                        max_new_tokens,
                        retained,
                        original_tokens,
                    )
                    if diagnostics.get("actual_llm_visual_tokens") != tokens:
                        raise BaselineError(f"Actual retained visual tokens {diagnostics.get('actual_llm_visual_tokens')} != target {tokens}.")
                    if args.method != "full" and diagnostics.get("compacted_input_sequence_length", 0) >= diagnostics.get("original_input_sequence_length", 0):
                        raise BaselineError(
                            "True token removal did not shorten the input sequence: "
                            f"original={diagnostics.get('original_input_sequence_length')} "
                            f"compacted={diagnostics.get('compacted_input_sequence_length')}"
                        )
                    result = build_result_common(args, sample, image_path, raw_output, generated_count, latency, selection, diagnostics, config)
                    if args.task == "grounding":
                        width = int(sample.get("width") or image.size[0])
                        height = int(sample.get("height") or image.size[1])
                        raw_box, clipped_box, parse_failure = parse_box(raw_output, width, height, args.box_coordinate_mode)
                        iou = box_iou(clipped_box, sample["gt_box"]) if clipped_box is not None else 0.0
                        success = clipped_box is not None and iou >= 0.5
                        result.update(
                            {
                                "query": sample.get("query"),
                                "gt_box": sample.get("gt_box"),
                                "raw_parsed_box": raw_box,
                                "predicted_box": clipped_box,
                                "box_iou": iou,
                                "score": iou,
                                "success_iou_0_5": success,
                                "valid": clipped_box is not None,
                                "invalid": clipped_box is None,
                                "parse_failure_reason": parse_failure,
                            }
                        )
                        result.update(selection_diagnostics(retained, sample["gt_box"], (width, height), args.image_grid_size))
                        if len(figure_paths) < args.max_figures:
                            figure_paths.append(
                                save_grounding_figure(
                                    sample,
                                    image,
                                    clipped_box,
                                    retained,
                                    args.image_grid_size,
                                    args.method,
                                    keep_ratio(tokens, original_tokens),
                                    iou,
                                    success,
                                    figures_dir,
                                )
                            )
                    else:
                        scored = score_vqa(sample, raw_output, args.dataset)
                        result.update(
                            {
                                "question": sample.get("question"),
                                "choices": sample.get("choices"),
                                "target_answer": sample.get("answer"),
                                **scored,
                            }
                        )
                    results.append(result)
                    handle.write(json.dumps(result, ensure_ascii=True) + "\n")
                    done = len(existing) + index + 1
                    if done % 10 == 0 or done == len(rows):
                        print(f"Processed {done}/{len(rows)} samples")

            run_duration_s = time.perf_counter() - run_start
            peak_alloc = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
            peak_reserved = torch.cuda.max_memory_reserved() / 1024**2 if torch.cuda.is_available() else None
            metrics = aggregate(results, args, run_duration_s, peak_alloc, peak_reserved)
            add_model_efficiency(metrics, model, torch)
            metrics["model/load_latency_s"] = model_load_s
            metrics["tokens/budget_family"] = config["tokens/budget_family"]
            estimate = runtime_estimate(args, run_duration_s)
            metrics.update({key: value for key, value in estimate.items() if key.startswith("runtime/")})
            add_relative_metrics(metrics, config, args.tables_root)
            write_json(run_dir / "metrics.json", metrics)
            update_tables(config, metrics, args)
            update_runtime_files(args.tables_root, estimate)
            log_wandb(wandb_run, metrics, results, run_dir, figure_paths)
            if wandb_run is not None:
                try:
                    wandb_run.finish()
                except Exception:
                    pass
            print(f"Saved run outputs to {run_dir}")
            print(
                "Runtime estimate: completed={completed} remaining={remaining} eta={eta}".format(
                    completed=estimate.get("runtime/completed_runs"),
                    remaining=estimate.get("runtime/remaining_runs"),
                    eta=estimate.get("runtime/eta_utc"),
                )
            )
            return 0
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if args.no_wandb:
        args.wandb = False
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
