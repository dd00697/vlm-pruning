#!/usr/bin/env python
"""Authoritatively rescore one completed K=12/B=48 run from raw outputs.

The scorer never trusts the convenience score stored during inference.  It
executes the pinned lmms-eval parsing code where that is the benchmark's
official static evaluator, records exact evaluator-source hashes, and emits
both aggregate and per-sample scoring artifacts.  MM-Vet predictions are
exported but deliberately remain externally blocked without the official
judge credentials.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OFFICIAL_ROOT = Path("/workspace/third_party_official/lmms-eval")
OFFICIAL_TASKS = OFFICIAL_ROOT / "lmms_eval/tasks"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_baseline import box_iou, parse_box  # noqa: E402
from vlm_pruning.mme import score_mme  # noqa: E402


class ScoringError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", help="Override dataset/name from config.json")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    temporary.replace(path)


def load_source_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ScoringError(f"Cannot load official evaluator source {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ast_functions(path: Path, names: set[str], namespace: dict[str, Any]) -> dict[str, Callable[..., Any]]:
    """Execute exact selected function definitions without importing lmms-eval's heavy task registry."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    found = {node.name for node in selected}
    if found != names:
        raise ScoringError(f"Missing official functions in {path}: {sorted(names - found)}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def field(row: dict[str, Any], name: str, default: Any = None) -> Any:
    benchmark = row.get("benchmark_fields") if isinstance(row.get("benchmark_fields"), dict) else {}
    return benchmark.get(name, row.get(name, default))


def raw_output(row: dict[str, Any]) -> str:
    return str(row.get("raw_model_output") or row.get("prediction") or "").strip()


def gqa_normalize(value: Any) -> str:
    text = str(value if value is not None else "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def yes_no(value: Any) -> str:
    text = str(value or "").lower().strip().replace(".", "")
    if text in {"yes", "no"}:
        return text
    if len(text) == 1:
        return {"y": "yes", "n": "no"}.get(text, "other")
    prefix = text[:4]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return "other"


def common_metrics(scored: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["official_score"]) for row in scored if row.get("official_score") is not None]
    return {
        "samples": len(scored),
        "scored_samples": len(values),
        "mean_score": sum(values) / len(values) if values else None,
        "invalid_outputs": sum(bool(row.get("official_invalid")) for row in scored),
    }


def score_grounding(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored = []
    for row in rows:
        width, height = row.get("width"), row.get("height")
        if not width or not height:
            image_path = Path(str(row.get("image_path") or ""))
            if not image_path.is_absolute():
                image_path = ROOT / image_path
            if image_path.is_file():
                from PIL import Image

                with Image.open(image_path) as image:
                    width, height = image.size
        predicted = None
        if width and height:
            _, predicted, _ = parse_box(raw_output(row), int(width), int(height), "normalized_0_1")
        if predicted is None:
            predicted = row.get("predicted_box")
        target = row.get("gt_box")
        value = box_iou(predicted, target) if isinstance(predicted, list) and isinstance(target, list) else 0.0
        scored.append({"sample_id": row["sample_id"], "official_score": value, "official_correct": value >= 0.5, "official_invalid": predicted is None})
    metrics = common_metrics(scored)
    values = [item["official_score"] for item in scored]
    metrics.update({"metric": "box_iou_acc_at_0_5", "mean_iou": sum(values) / len(values), "accuracy_at_0_5": sum(item["official_correct"] for item in scored) / len(scored)})
    return scored, metrics


def score_gqa(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored = []
    for row in rows:
        pred = gqa_normalize(raw_output(row))
        target = gqa_normalize(row.get("target_answer"))
        correct = bool(pred and pred == target)
        scored.append({"sample_id": row["sample_id"], "official_prediction": pred, "official_target": target, "official_score": float(correct), "official_correct": correct, "official_invalid": not pred})
    metrics = common_metrics(scored)
    metrics["metric"] = "exact_match_ignore_case_punctuation"
    metrics["accuracy"] = metrics["mean_score"]
    return scored, metrics


def score_vqa(rows: list[dict[str, Any]], source_paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = OFFICIAL_TASKS / "_task_utils/vqa_eval_metric.py"
    processor = load_source_module(source, "pinned_vqa_eval_metric").EvalAIAnswerProcessor()
    scored = []
    for row in rows:
        answers = row.get("target_answer")
        if not isinstance(answers, list):
            raise ScoringError(f"{row['sample_id']}: VQA target is not an annotator answer list")
        prediction = processor(raw_output(row))
        normalized_answers = [processor(str(answer)) for answer in answers]
        leave_one_out = []
        for index in range(len(normalized_answers)):
            matches = sum(answer == prediction for other, answer in enumerate(normalized_answers) if other != index)
            leave_one_out.append(min(1.0, matches / 3.0))
        value = sum(leave_one_out) / len(leave_one_out) if leave_one_out else 0.0
        scored.append({"sample_id": row["sample_id"], "official_score": value, "official_correct": value == 1.0, "official_invalid": not bool(raw_output(row))})
    metrics = common_metrics(scored)
    metrics["metric"] = "evalai_vqa_leave_one_annotator_out"
    metrics["accuracy"] = metrics["mean_score"]
    source_paths.append(source)
    return scored, metrics


def score_scienceqa(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored = []
    for row in rows:
        target = str(row.get("target_answer") or "").strip().lower()
        pred = raw_output(row)
        correct = pred.lower() == target or (len(pred) >= 2 and pred[0].isupper() and pred[1] == "." and pred[0].lower() == target)
        scored.append({"sample_id": row["sample_id"], "official_prediction": pred, "official_target": target, "official_score": float(correct), "official_correct": correct, "official_invalid": not pred})
    metrics = common_metrics(scored)
    metrics["metric"] = "scienceqa_img_exact_match"
    metrics["accuracy"] = metrics["mean_score"]
    return scored, metrics


def binary_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(item["prediction"] == "yes" and item["target"] == "yes" for item in items)
    fp = sum(item["prediction"] == "yes" and item["target"] == "no" for item in items)
    tn = sum(item["prediction"] == "no" and item["target"] == "no" for item in items)
    fn = sum(item["prediction"] == "no" and item["target"] == "yes" for item in items)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "samples": len(items), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / len(items) if items else 0.0,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "yes_ratio": sum(item["prediction"] == "yes" for item in items) / len(items) if items else 0.0,
        "invalid_outputs": sum(item["prediction"] == "other" for item in items),
    }


def score_pope(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored, by_subset = [], defaultdict(list)
    for row in rows:
        pred, target = yes_no(raw_output(row)), yes_no(row.get("target_answer"))
        subset = str(row.get("category") or "unknown").lower()
        item = {"sample_id": row["sample_id"], "subset": subset, "prediction": pred, "target": target, "official_score": float(pred == target), "official_correct": pred == target, "official_invalid": pred == "other"}
        scored.append(item)
        by_subset[subset].append(item)
    metrics = common_metrics(scored)
    metrics.update({"metric": "pope_binary_classification", "subsets": {key: binary_metrics(value) for key, value in sorted(by_subset.items())}, "overall": binary_metrics(scored)})
    return scored, metrics


def score_mme_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored = []
    for row in rows:
        pred, target = yes_no(raw_output(row)), yes_no(row.get("target_answer"))
        scored.append({"sample_id": row["sample_id"], "category": row.get("category"), "mme_pair_id": row.get("mme_pair_id"), "prediction": pred, "target": target, "official_score": float(pred == target), "official_correct": pred == target, "official_invalid": pred == "other"})
    metrics = score_mme(scored)
    metrics["metric"] = "mme_acc_plus_acc_plus"
    return scored, metrics


def score_mmbench(rows: list[dict[str, Any]], source_paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluator_module = load_source_module(OFFICIAL_TASKS / "mmbench/mmbench_evals.py", "pinned_mmbench_evals")
    mcq_module = load_source_module(OFFICIAL_TASKS / "_task_utils/mcq_extract.py", "pinned_mcq_extract")
    evaluator = evaluator_module.MMBench_Evaluator()
    parsed: dict[int, dict[str, Any]] = {}
    answer_map: dict[int, str] = {}
    for row in rows:
        idx = int(field(row, "index"))
        choices = list(row.get("choices") or [])
        choice_map = {chr(ord("A") + i): str(value) for i, value in enumerate(choices)}
        prediction = raw_output(row)
        inferred = evaluator.can_infer(prediction, dict(choice_map))
        if not inferred:
            inferred = mcq_module.extract_mcq_answer(prediction, choices=sorted(choice_map))
        target = str(row.get("target_answer") or "").strip().upper()
        parsed[idx] = {"sample_id": row["sample_id"], "prediction": inferred or "", "target": target, "category": row.get("category"), "l2_category": field(row, "l2_category")}
        answer_map[idx] = target
    base_indices = sorted(idx for idx in parsed if idx < 1_000_000)
    scored = []
    for base in base_indices:
        variants = [parsed[idx] for idx in sorted(parsed) if idx % 1_000_000 == base]
        correct = bool(variants) and all(item["prediction"] == answer_map[idx] for idx, item in ((idx, parsed[idx]) for idx in sorted(parsed) if idx % 1_000_000 == base))
        scored.append({"sample_id": parsed[base]["sample_id"], "base_index": base, "circular_variants": len(variants), "variant_predictions": [item["prediction"] for item in variants], "official_score": float(correct), "official_correct": correct, "official_invalid": any(not item["prediction"] for item in variants), "category": parsed[base]["category"], "l2_category": parsed[base]["l2_category"]})
    if len(scored) * 3 != len(rows):
        raise ScoringError(f"MMBench circular grouping expected 3 variants per base row; got {len(rows)} rows/{len(scored)} groups")
    metrics = common_metrics(scored)
    metrics.update({"metric": "mmbench_en_dev_static_circular", "accuracy": metrics["mean_score"], "base_questions": len(scored), "circular_rows": len(rows)})
    source_paths.extend([OFFICIAL_TASKS / "mmbench/mmbench_evals.py", OFFICIAL_TASKS / "_task_utils/mcq_extract.py"])
    return scored, metrics


def score_mmmu(rows: list[dict[str, Any]], source_paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mcq_path = OFFICIAL_TASKS / "_task_utils/mmmu_mcq_utils.py"
    utils_path = OFFICIAL_TASKS / "mmmu/utils.py"
    mcq = load_source_module(mcq_path, "pinned_mmmu_mcq_utils")
    functions = load_ast_functions(utils_path, {"extract_numbers", "check_is_number", "normalize_str", "parse_open_response", "eval_open", "eval_multi_choice"}, {"re": re})
    random.seed(0)
    np.random.seed(0)
    scored = []
    for row in rows:
        question_type = str(field(row, "question_type") or "")
        prediction = raw_output(row)
        target = row.get("target_answer")
        choices = list(row.get("choices") or [])
        if question_type == "multiple-choice":
            index2ans, all_choices = mcq.get_multi_choice_info(choices)
            parsed = mcq.parse_mmmu_multi_choice_response(prediction, all_choices, index2ans)
            correct = functions["eval_multi_choice"](target, parsed)
        else:
            parsed = functions["parse_open_response"](prediction)
            correct = functions["eval_open"](target, parsed)
        scored.append({"sample_id": row["sample_id"], "image_count": int(row.get("image_count") or 1), "question_type": question_type, "official_prediction": parsed, "official_target": target, "official_score": float(correct), "official_correct": bool(correct), "official_invalid": not bool(prediction)})
    metrics = common_metrics(scored)
    subsets = {}
    for name, subset in (("all_900", scored), ("single_image_857", [x for x in scored if x["image_count"] == 1]), ("multi_image_43", [x for x in scored if x["image_count"] > 1])):
        subsets[name] = {"samples": len(subset), "accuracy": sum(x["official_score"] for x in subset) / len(subset) if subset else None}
    metrics.update({"metric": "mmmu_acc", "accuracy": metrics["mean_score"], "subsets": subsets})
    source_paths.extend([mcq_path, utils_path])
    return scored, metrics


def mmvet_export(rows: list[dict[str, Any]], run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = {str(row["sample_id"]): raw_output(row) for row in rows}
    atomic_json(run_dir / "official_predictions.json", predictions)
    scored = [{"sample_id": row["sample_id"], "official_score": None, "official_invalid": not bool(raw_output(row))} for row in rows]
    return scored, {"metric": "mmvet_gpt4_judge", "samples": len(rows), "status": "blocked_external_scoring", "judge_model": "gpt-4o-2024-11-20", "reason": "Official judge credentials are unavailable; predictions were exported and no proxy score was substituted."}


def score(dataset: str, rows: list[dict[str, Any]], run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[Path]]:
    name = dataset.lower()
    sources: list[Path] = []
    if name.startswith("refcoco"):
        scored, metrics = score_grounding(rows)
    elif name == "gqa":
        scored, metrics = score_gqa(rows)
    elif name in {"vqav2_val", "textvqa", "textvqa_val"}:
        scored, metrics = score_vqa(rows, sources)
    elif name == "pope_all":
        scored, metrics = score_pope(rows)
        sources.append(OFFICIAL_TASKS / "pope/utils.py")
    elif name == "mme":
        scored, metrics = score_mme_rows(rows)
        sources.append(OFFICIAL_TASKS / "mme/utils.py")
    elif name == "scienceqa_img_test":
        scored, metrics = score_scienceqa(rows)
        sources.append(OFFICIAL_TASKS / "scienceqa/utils.py")
    elif name == "mmbench_en_dev":
        scored, metrics = score_mmbench(rows, sources)
    elif name == "mmmu_val":
        scored, metrics = score_mmmu(rows, sources)
    elif name == "mmvet_test":
        scored, metrics = mmvet_export(rows, run_dir)
    else:
        raise ScoringError(f"Unsupported dataset: {dataset}")
    return scored, metrics, sources


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    result_path = run_dir / "per_sample_results.jsonl"
    config_path = run_dir / "config.json"
    if not result_path.is_file() or not config_path.is_file():
        raise ScoringError(f"Run is missing config/results: {run_dir}")
    output_path = run_dir / "official_metrics.json"
    if output_path.exists() and not args.force:
        raise ScoringError(f"Refusing to overwrite {output_path}; pass --force after validating the run")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = args.dataset or config.get("dataset/name")
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ScoringError("No result rows")
    ids = [str(row.get("sample_id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ScoringError("Result sample IDs are not unique")
    scored, metrics, sources = score(str(dataset), rows, run_dir)
    if len(scored) == 0:
        raise ScoringError("Official scorer emitted no per-sample rows")
    provenance = {
        "scored_at_utc": utc_now(),
        "dataset": dataset,
        "run_dir": str(run_dir),
        "input_results_sha256": sha256(result_path),
        "input_config_sha256": sha256(config_path),
        "scorer_sha256": sha256(Path(__file__)),
        "lmms_eval_root": str(OFFICIAL_ROOT),
        "lmms_eval_commit": "75571cfe0fa36ca5ec2e6aa2bb268f9d38295af3",
        "official_source_sha256": {str(path.relative_to(OFFICIAL_ROOT)): sha256(path) for path in sorted(set(sources))},
    }
    metrics["status"] = metrics.get("status", "complete")
    metrics["provenance"] = provenance
    atomic_jsonl(run_dir / "official_per_sample_scores.jsonl", scored)
    atomic_json(output_path, metrics)
    print(json.dumps({"dataset": dataset, "samples": len(rows), "scored_rows": len(scored), "status": metrics["status"], "official_metrics": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
