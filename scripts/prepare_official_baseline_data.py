#!/usr/bin/env python
"""Prepare official-baseline JSONL manifests from Hugging Face or local JSONL data."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


GROUNDING_DATASETS = {"refcoco", "refcoco_plus", "refcocog"}
VQA_DATASETS = {
    "pope",
    "aokvqa",
    "a-okvqa",
    "scienceqa_img",
    "scienceqa",
    "ocr_vqa",
    "ocr-vqa",
    "textvqa",
    "mmmu",
    "mme",
    "seed_bench",
    "seed-bench",
    "mmvet",
    "gqa",
}

HF_CANDIDATES: dict[str, list[tuple[str, str | None]]] = {
    "refcoco": [("lmms-lab/RefCOCO", None)],
    "refcoco_plus": [("lmms-lab/RefCOCO+", None), ("lmms-lab/RefCOCOplus", None)],
    "refcocog": [("lmms-lab/RefCOCOg", None)],
    "pope": [("lmms-lab/POPE", None)],
    "aokvqa": [("lmms-lab/A-OKVQA", None), ("HuggingFaceM4/A-OKVQA", None)],
    "a-okvqa": [("lmms-lab/A-OKVQA", None), ("HuggingFaceM4/A-OKVQA", None)],
    "scienceqa_img": [("lmms-lab/ScienceQA-IMG", None)],
    "scienceqa": [("lmms-lab/ScienceQA-IMG", None)],
    "ocr_vqa": [("lmms-lab/OCR-VQA", None), ("howard-hou/OCR-VQA", None)],
    "ocr-vqa": [("lmms-lab/OCR-VQA", None), ("howard-hou/OCR-VQA", None)],
    "textvqa": [("lmms-lab/textvqa", None)],
    "mmmu": [("lmms-lab/MMMU", None), ("MMMU/MMMU", "Accounting")],
    "mme": [("lmms-lab/MME", None)],
    "seed_bench": [("lmms-lab/SEED-Bench", None), ("lmms-lab/SEED-Bench-Image", None)],
    "seed-bench": [("lmms-lab/SEED-Bench", None), ("lmms-lab/SEED-Bench-Image", None)],
    "mmvet": [("lmms-lab/MMVet", None)],
    "gqa": [("lmms-lab/GQA", None)],
}


class DataPreparationError(RuntimeError):
    pass


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
IMAGE_FIELD_CANDIDATES = [
    "image",
    "img",
    "images",
    "image_file",
    "image_path",
    "decoded_image",
    "image_1",
    "image_2",
    "image_3",
    "image_4",
    "image_5",
    "image_6",
    "image_7",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(GROUNDING_DATASETS | VQA_DATASETS))
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Materialize Hugging Face datasets instead of streaming the selected split.",
    )
    parser.add_argument("--bbox-format", choices=["xywh", "xyxy"], default="xywh")
    parser.add_argument("--all-expressions", action="store_true")
    return parser.parse_args()


def normalize_dataset_name(value: str) -> str:
    return value.lower().replace("-", "_")


def choose_split(dataset: Any, requested: str) -> tuple[Any, str]:
    if not hasattr(dataset, "keys"):
        return dataset, requested
    keys = list(dataset.keys())
    candidates = [requested]
    if requested == "val":
        candidates += ["validation", "valid", "dev", "test"]
    elif requested == "validation":
        candidates += ["val", "valid", "dev", "test"]
    elif requested == "dev":
        candidates += ["validation", "val", "test"]
    for candidate in candidates:
        if candidate in dataset:
            return dataset[candidate], candidate
    if len(keys) == 1:
        return dataset[keys[0]], keys[0]
    raise DataPreparationError(f"Requested split '{requested}' not found. Available splits: {keys}")


def load_hf_dataset(dataset: str, split: str, *, streaming: bool = False) -> tuple[Iterable[Any], str, str]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise DataPreparationError(f"Missing datasets package: {exc}") from exc
    errors: list[str] = []
    for dataset_id, config_name in HF_CANDIDATES[normalize_dataset_name(dataset)]:
        try:
            if config_name:
                loaded = load_dataset(dataset_id, config_name, streaming=streaming)
            else:
                loaded = load_dataset(dataset_id, streaming=streaming)
            selected, actual_split = choose_split(loaded, split)
            return selected, f"{dataset_id}:{config_name}" if config_name else dataset_id, actual_split
        except Exception as exc:
            errors.append(f"{dataset_id}{('/' + config_name) if config_name else ''}: {type(exc).__name__}: {exc}")
    raise DataPreparationError("Unable to load dataset candidates. " + " | ".join(errors))


def sanitize_name(value: Any, default: str) -> str:
    text = str(value if value is not None else default)
    text = text.replace("\\", "/").split("/")[-1]
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return text or default


def ensure_image_filename(name: str) -> str:
    path = Path(name)
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return name
    if suffix:
        return f"{path.with_suffix('').name}.jpg"
    return f"{name}.jpg"


def looks_like_local_path(value: Any) -> bool:
    if not isinstance(value, (str, os.PathLike)):
        return False
    text = str(value).strip()
    if not text or len(text) > 240 or any(char in text for char in "\n\r\t"):
        return False
    suffix = Path(text).suffix.lower()
    return suffix in IMAGE_SUFFIXES or "/" in text or "\\" in text


def safe_existing_path(value: Any) -> Path | None:
    if not looks_like_local_path(value):
        return None
    try:
        path = Path(value)
        return path if path.exists() else None
    except OSError:
        return None


def as_pil_image(value: Any) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (list, tuple)):
        for item in value:
            image = as_pil_image(item)
            if image is not None:
                return image
    if isinstance(value, dict):
        path = value.get("path") or value.get("filename") or value.get("image_path")
        existing = safe_existing_path(path)
        if existing is not None:
            return Image.open(existing).convert("RGB")
        data = value.get("bytes")
        if data:
            import io

            return Image.open(io.BytesIO(data)).convert("RGB")
    existing = safe_existing_path(value)
    if existing is not None:
        return Image.open(existing).convert("RGB")
    return None


def find_image(row: dict[str, Any]) -> Image.Image | None:
    for key in IMAGE_FIELD_CANDIDATES:
        if key in row:
            image = as_pil_image(row[key])
            if image is not None:
                return image
    for key, value in row.items():
        if key.startswith("image_") or key.startswith("img_"):
            image = as_pil_image(value)
            if image is not None:
                return image
    for value in row.values():
        image = as_pil_image(value)
        if image is not None:
            return image
    return None


def row_id(row: dict[str, Any], index: int) -> str:
    for key in ["sample_id", "question_id", "id", "annotation_id", "ann_id", "image_id", "index"]:
        if row.get(key) is not None:
            return sanitize_name(row[key], f"sample_{index:06d}")
    return f"sample_{index:06d}"


def image_name(row: dict[str, Any], sample_id: str) -> str:
    for key in ["file_name", "filename", "image_path", "path"]:
        value = row.get(key)
        if isinstance(value, dict):
            value = value.get("path") or value.get("filename")
        if isinstance(value, (str, os.PathLike)) and looks_like_local_path(value):
            name = sanitize_name(value, f"{sample_id}.jpg")
            return ensure_image_filename(name)
    image_value = row.get("image")
    if isinstance(image_value, dict):
        value = image_value.get("path") or image_value.get("filename")
        if isinstance(value, (str, os.PathLike)) and looks_like_local_path(value):
            name = sanitize_name(value, f"{sample_id}.jpg")
            return ensure_image_filename(name)
    for key in ["image_id", "data_id", "question_id", "sample_id", "id", "index"]:
        value = row.get(key)
        if isinstance(value, (str, int)):
            name = sanitize_name(value, f"{sample_id}.jpg")
            return ensure_image_filename(name)
    return f"{sample_id}.jpg"


def listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def extract_bbox(row: dict[str, Any]) -> list[float] | None:
    for key in ["bbox", "box", "gt_box", "bounding_box", "answer_bbox"]:
        value = row.get(key)
        if value is None:
            continue
        values = listify(value)
        if len(values) >= 4:
            try:
                return [float(values[0]), float(values[1]), float(values[2]), float(values[3])]
            except Exception:
                return None
    return None


def bbox_to_xyxy(bbox: list[float], fmt: str, width: int, height: int) -> list[float]:
    x0, y0, a, b = bbox
    x1, y1 = (x0 + a, y0 + b) if fmt == "xywh" else (a, b)
    clipped = [
        max(0.0, min(float(width), x0)),
        max(0.0, min(float(height), y0)),
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise DataPreparationError(f"Invalid bbox after clipping: {bbox} -> {clipped}")
    return clipped


def ref_queries(row: dict[str, Any]) -> list[str]:
    for key in ["sentences", "expressions", "expression", "caption", "query", "text", "answer"]:
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, list):
            out = []
            for item in value:
                if isinstance(item, dict):
                    item = item.get("sent") or item.get("raw") or item.get("text") or item.get("caption")
                if item is not None and str(item).strip():
                    out.append(str(item).strip())
            if out:
                return out
        if value is not None and str(value).strip():
            return [str(value).strip()]
    return []


def extract_question(row: dict[str, Any]) -> str | None:
    for key in ["question", "query", "text", "prompt", "instruction", "questions"]:
        value = row.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def parse_container(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("[", "{", "(")):
            try:
                return ast.literal_eval(text)
            except Exception:
                return value
    return value


def extract_choices(row: dict[str, Any]) -> list[str]:
    for key in ["choices", "options", "all_choices", "candidate_answers"]:
        value = parse_container(row.get(key))
        if isinstance(value, list) and value:
            return [str(item.get("text") if isinstance(item, dict) and "text" in item else item) for item in value]
        if isinstance(value, dict):
            ordered = []
            for label in ["A", "B", "C", "D", "E", "F"]:
                if label in value:
                    ordered.append(str(value[label]))
                elif label.lower() in value:
                    ordered.append(str(value[label.lower()]))
            if ordered:
                return ordered
    alpha = []
    for label in "abcdefghijklmnopqrstuvwxyz":
        found = False
        for key in [f"choice_{label}", f"option_{label}", f"choice{label}", f"option{label}", label.upper(), label]:
            value = row.get(key)
            if value is not None:
                alpha.append(str(value))
                found = True
                break
        if not found and alpha:
            break
    if alpha:
        return alpha
    # ScienceQA-style separate option fields sometimes appear as option1/option2.
    numbered = []
    for idx in range(10):
        for key in [f"option{idx}", f"choice{idx}"]:
            if row.get(key) is not None:
                numbered.append(str(row[key]))
                break
    return numbered


def extract_answer(row: dict[str, Any], choices: list[str]) -> Any:
    for key in [
        "answer_idx",
        "answer_index",
        "label_idx",
        "correct_choice_idx",
        "correct_answer_idx",
        "correct_index",
    ]:
        value = row.get(key)
        if value is not None:
            try:
                idx = int(value)
                if 0 <= idx < len(choices):
                    return chr(ord("A") + idx)
                if 1 <= idx <= len(choices):
                    return chr(ord("A") + idx - 1)
            except Exception:
                return value
    for key in ["answer", "label", "gt_answer", "correct_answer", "target", "response", "answers"]:
        value = row.get(key)
        if value is not None:
            return parse_container(value)
    return None


def question_answer_pairs(row: dict[str, Any], choices: list[str]) -> list[tuple[str, Any, str]]:
    questions = row.get("questions")
    if isinstance(questions, list) and questions and row.get("question") is None:
        raw_answers = parse_container(row.get("answers") if row.get("answers") is not None else row.get("answer"))
        answers = raw_answers if isinstance(raw_answers, list) else [raw_answers] * len(questions)
        pairs = []
        for idx, question in enumerate(questions):
            if question is None or not str(question).strip():
                continue
            if len(answers) == len(questions):
                answer = answers[idx]
            elif len(answers) == 1:
                answer = answers[0]
            else:
                answer = answers
            if answer is None:
                continue
            pairs.append((str(question).strip(), answer, f"q{idx}"))
        return pairs
    question = extract_question(row)
    if not question:
        return []
    answer = extract_answer(row, choices)
    if answer is None:
        return []
    return [(question, answer, "")]


def answer_type(dataset: str, answer: Any, choices: list[str]) -> str:
    name = normalize_dataset_name(dataset)
    if name == "pope" or str(answer).lower() in {"yes", "no"}:
        return "yes_no"
    if choices:
        return "multiple_choice"
    return "open"


def is_supported_vqa_row(dataset: str, row: dict[str, Any]) -> bool:
    """Keep image-only rows for image-only VLM evaluation."""
    if normalize_dataset_name(dataset) != "seed_bench":
        return True
    data_type = row.get("data_type")
    if data_type is not None and str(data_type).strip().lower() not in {"image", "img"}:
        return False
    for key in ["data_id", "image_id", "video_id", "file_name", "filename"]:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip().lower()
        if text.endswith((".mp4", ".webm", ".avi", ".mov", ".mkv")):
            return False
    return True


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        image.save(path, quality=95)


def prepare_grounding(args: argparse.Namespace, rows: Iterable[Any], source: str, actual_split: str) -> int:
    image_root = args.raw_root / normalize_dataset_name(args.dataset) / actual_split / "images"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for index, raw_row in enumerate(rows):
            row = dict(raw_row)
            image = find_image(row)
            if image is None:
                raise DataPreparationError(f"Image missing or unreadable for row {index}")
            bbox = extract_bbox(row)
            if bbox is None:
                raise DataPreparationError(f"Ground-truth bbox missing for row {index}")
            width, height = image.size
            gt_box = bbox_to_xyxy(bbox, args.bbox_format, width, height)
            queries = ref_queries(row)
            if not queries:
                raise DataPreparationError(f"Referring expression missing for row {index}")
            base_id = row_id(row, index)
            path = image_root / image_name(row, base_id)
            save_image(image, path)
            for q_idx, query in enumerate(queries if args.all_expressions else queries[:1]):
                sample_id = f"{base_id}_{q_idx}" if len(queries) > 1 else base_id
                record = {
                    "sample_id": sample_id,
                    "dataset": normalize_dataset_name(args.dataset),
                    "split": args.split,
                    "image_id": str(row.get("image_id") or row.get("file_name") or base_id),
                    "image_path": str(path.as_posix()),
                    "query": query,
                    "gt_box": gt_box,
                    "width": width,
                    "height": height,
                    "metadata": {"source": source, "source_split": actual_split, "source_row": index},
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                count += 1
                if args.limit is not None and count >= args.limit:
                    return count
    return count


def prepare_vqa(args: argparse.Namespace, rows: Iterable[Any], source: str, actual_split: str) -> int:
    image_root = args.raw_root / normalize_dataset_name(args.dataset) / actual_split / "images"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    stop_at_first_unsupported = normalize_dataset_name(args.dataset) == "seed_bench"
    with args.output.open("w", encoding="utf-8") as handle:
        for index, raw_row in enumerate(rows):
            row = dict(raw_row)
            if not is_supported_vqa_row(args.dataset, row):
                if stop_at_first_unsupported and count:
                    break
                continue
            image = find_image(row)
            if image is None:
                raise DataPreparationError(f"Image missing or unreadable for row {index}")
            choices = extract_choices(row)
            pairs = question_answer_pairs(row, choices)
            if not pairs:
                if not extract_question(row):
                    raise DataPreparationError(f"Question missing for row {index}")
                raise DataPreparationError(f"Answer missing for row {index}")
            base_id = row_id(row, index)
            path = image_root / image_name(row, base_id)
            save_image(image, path)
            category = row.get("category") or row.get("type") or row.get("subject") or row.get("domain") or ""
            for question, answer, suffix in pairs:
                sample_id = f"{base_id}_{suffix}" if suffix else base_id
                record = {
                    "sample_id": sample_id,
                    "dataset": normalize_dataset_name(args.dataset),
                    "split": args.split,
                    "image_id": str(row.get("image_id") or row.get("data_id") or row.get("file_name") or base_id),
                    "image_path": str(path.as_posix()),
                    "question": question,
                    "answer": answer,
                    "choices": choices,
                    "answer_type": answer_type(args.dataset, answer, choices),
                    "category": str(category),
                    "metadata": {
                        "source": source,
                        "source_split": actual_split,
                        "source_row": index,
                        "source_question_suffix": suffix,
                        "source_data_type": row.get("data_type"),
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                count += 1
                if args.limit is not None and count >= args.limit:
                    return count
    return count


def load_local_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        name = normalize_dataset_name(args.dataset)
        if args.raw_root.is_file() and args.raw_root.suffix.lower() == ".jsonl":
            rows = load_local_jsonl(args.raw_root)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as handle:
                for index, row in enumerate(rows):
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                    if args.limit is not None and index + 1 >= args.limit:
                        break
            print(f"Wrote manifest from local JSONL: {args.output}")
            return 0
        if args.no_download:
            raise DataPreparationError(
                f"No local loader for {args.raw_root}; provide a JSONL --raw-root or allow --download."
            )
        rows, source, actual_split = load_hf_dataset(name, args.split, streaming=not args.no_streaming)
        if name in GROUNDING_DATASETS:
            count = prepare_grounding(args, rows, source, actual_split)
        else:
            count = prepare_vqa(args, rows, source, actual_split)
        if count <= 0:
            raise DataPreparationError("No records written.")
        print(f"Wrote {count} records to {args.output}")
        return 0
    except KeyboardInterrupt:
        return 130
    except DataPreparationError as exc:
        print(f"Error: {args.dataset}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Some Hugging Face streaming readers leave background state that can crash
    # during interpreter finalization. Exit after explicit flushing so the real
    # preparation status is preserved for sweep orchestration.
    os._exit(code)
