#!/usr/bin/env python
"""Prepare RefCOCO-family and VQA manifests for baseline runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


GROUNDING_DATASETS = {"refcoco", "refcoco_plus", "refcocog"}
VQA_DATASETS = {"pope", "scienceqa_img", "gqa", "textvqa"}

HF_CANDIDATES = {
    "refcoco": ["lmms-lab/RefCOCO"],
    "refcoco_plus": ["lmms-lab/RefCOCO+", "lmms-lab/RefCOCOplus"],
    "refcocog": ["lmms-lab/RefCOCOg"],
    "pope": ["lmms-lab/POPE"],
    "scienceqa_img": ["lmms-lab/ScienceQA-IMG"],
    "gqa": ["lmms-lab/GQA"],
    "textvqa": ["lmms-lab/textvqa"],
}


class DataPreparationError(RuntimeError):
    """Raised when a dataset cannot be converted to the required manifest."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(GROUNDING_DATASETS | VQA_DATASETS))
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true", help="Download from Hugging Face datasets when needed.")
    mode.add_argument("--no-download", action="store_true", help="Only use local files.")
    parser.add_argument(
        "--bbox-format",
        default="xywh",
        choices=["xywh", "xyxy"],
        help="Source RefCOCO bbox format before conversion to manifest xyxy.",
    )
    parser.add_argument(
        "--all-expressions",
        action="store_true",
        help="Emit one manifest row per referring expression instead of one row per annotation.",
    )
    return parser.parse_args()


def fail(message: str, *, affected: str | None = None) -> None:
    prefix = f"{affected}: " if affected else ""
    print(f"Error: {prefix}{message}", file=sys.stderr)


def ensure_dependencies() -> None:
    missing = []
    try:
        import datasets  # noqa: F401
    except Exception:
        missing.append("datasets")
    try:
        import PIL  # noqa: F401
    except Exception:
        missing.append("pillow")
    if missing:
        raise DataPreparationError(
            "Missing Python package(s): "
            + ", ".join(missing)
            + ". Install with: python -m pip install datasets pillow"
        )


def sanitize_name(value: Any, default: str) -> str:
    text = str(value if value is not None else default)
    text = text.replace("\\", "/").split("/")[-1]
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return text or default


def choose_split(dataset_dict: Any, requested: str) -> tuple[Any, str]:
    if not hasattr(dataset_dict, "keys"):
        return dataset_dict, requested
    keys = list(dataset_dict.keys())
    candidates = [requested]
    if requested == "val":
        candidates.extend(["validation", "valid", "test"])
    elif requested == "validation":
        candidates.extend(["val", "valid", "test"])
    for candidate in candidates:
        if candidate in dataset_dict:
            return dataset_dict[candidate], candidate
    if len(keys) == 1:
        only = keys[0]
        print(f"Requested split '{requested}' not found; using only available split '{only}'.")
        return dataset_dict[only], only
    raise DataPreparationError(f"Requested split '{requested}' not found. Available splits: {keys}")


def load_hf_dataset(dataset: str, split: str) -> tuple[Any, str, str]:
    from datasets import load_dataset

    errors: list[str] = []
    for dataset_id in HF_CANDIDATES[dataset]:
        try:
            loaded = load_dataset(dataset_id)
            selected, actual_split = choose_split(loaded, split)
            return selected, dataset_id, actual_split
        except Exception as exc:  # keep trying candidates, then report all errors
            errors.append(f"{dataset_id}: {type(exc).__name__}: {exc}")
    raise DataPreparationError("Unable to load Hugging Face dataset candidates. " + " | ".join(errors))


def as_pil_image(value: Any) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        path = value.get("path") or value.get("filename")
        if path and Path(path).exists():
            return Image.open(path).convert("RGB")
        bytes_value = value.get("bytes")
        if bytes_value:
            import io

            return Image.open(io.BytesIO(bytes_value)).convert("RGB")
    if isinstance(value, (str, os.PathLike)) and Path(value).exists():
        return Image.open(value).convert("RGB")
    return None


def find_image(row: dict[str, Any]) -> Image.Image | None:
    for key in ["image", "img", "image_file", "image_path"]:
        if key in row:
            image = as_pil_image(row[key])
            if image is not None:
                return image
    for value in row.values():
        image = as_pil_image(value)
        if image is not None:
            return image
    return None


def image_name(row: dict[str, Any], sample_id: str) -> str:
    for key in ["file_name", "filename", "image_id", "image", "image_path"]:
        value = row.get(key)
        if isinstance(value, dict):
            value = value.get("path") or value.get("filename")
        if isinstance(value, (str, int)):
            name = sanitize_name(value, f"{sample_id}.jpg")
            if not Path(name).suffix:
                name += ".jpg"
            return name
    return f"{sanitize_name(sample_id, 'sample')}.jpg"


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95)


def listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def extract_bbox(row: dict[str, Any]) -> list[float] | None:
    for key in ["bbox", "box", "gt_box", "bounding_box"]:
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
    if fmt == "xywh":
        x1, y1 = x0 + a, y0 + b
    else:
        x1, y1 = a, b
    clipped = [
        max(0.0, min(float(width), x0)),
        max(0.0, min(float(height), y0)),
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise DataPreparationError(f"Invalid or empty bbox after conversion/clipping: source={bbox}, xyxy={clipped}")
    return clipped


def refcoco_queries(row: dict[str, Any]) -> list[str]:
    for key in ["answer", "answers", "sentences", "expressions", "expression", "caption", "query", "text"]:
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, list):
            queries = []
            for item in value:
                if isinstance(item, dict):
                    item = item.get("sent") or item.get("raw") or item.get("text") or item.get("caption")
                if item is not None and str(item).strip():
                    queries.append(str(item).strip())
            if queries:
                return queries
        if value is not None and str(value).strip():
            return [str(value).strip()]
    return []


def vqa_question_answer(row: dict[str, Any]) -> tuple[str | None, str | None, list[str]]:
    question = None
    for key in ["question", "query", "text", "prompt"]:
        if row.get(key) is not None:
            question = str(row[key]).strip()
            break
    answer = None
    for key in ["answer", "label", "gt_answer", "answers"]:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            answer = str(value[0]).strip() if value else None
        else:
            answer = str(value).strip()
        break
    choices: list[str] = []
    for key in ["choices", "options"]:
        if isinstance(row.get(key), list):
            choices = [str(item) for item in row[key]]
            break
    return question, answer, choices


def row_sample_id(row: dict[str, Any], index: int) -> str:
    for key in ["question_id", "sample_id", "id", "annotation_id", "ann_id"]:
        if row.get(key) is not None:
            return sanitize_name(row[key], f"sample_{index:06d}")
    return f"sample_{index:06d}"


def prepare_grounding(args: argparse.Namespace, rows: Iterable[dict[str, Any]], source_id: str, actual_split: str) -> int:
    image_root = args.raw_root / args.dataset / actual_split / "images"
    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            row = dict(row)
            image = find_image(row)
            if image is None:
                raise DataPreparationError(f"Image missing or unreadable for row {row_index}.")
            bbox = extract_bbox(row)
            if bbox is None:
                raise DataPreparationError(f"Ground-truth bbox missing for row {row_index}.")
            width, height = image.size
            gt_box = bbox_to_xyxy(bbox, args.bbox_format, width, height)
            queries = refcoco_queries(row)
            if not queries:
                raise DataPreparationError(f"Referring expression/query missing for row {row_index}.")
            base_id = row_sample_id(row, row_index)
            file_name = image_name(row, base_id)
            image_path = image_root / file_name
            if not image_path.exists():
                save_image(image, image_path)
            selected_queries = queries if args.all_expressions else queries[:1]
            for query_index, query in enumerate(selected_queries):
                sample_id = f"{base_id}_{query_index}" if len(queries) > 1 else base_id
                record = {
                    "sample_id": sample_id,
                    "dataset": args.dataset,
                    "split": args.split,
                    "image_id": str(row.get("image_id") or row.get("file_name") or base_id),
                    "image_path": str(image_path.as_posix()),
                    "query": query,
                    "gt_box": gt_box,
                    "width": width,
                    "height": height,
                    "metadata": {
                        "source": source_id,
                        "source_split": actual_split,
                        "source_row": row_index,
                        "source_bbox": bbox,
                        "source_bbox_format": args.bbox_format,
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                count += 1
                if args.limit is not None and count >= args.limit:
                    return count
    return count


def prepare_vqa(args: argparse.Namespace, rows: Iterable[dict[str, Any]], source_id: str, actual_split: str) -> int:
    image_root = args.raw_root / args.dataset / actual_split / "images"
    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            row = dict(row)
            image = find_image(row)
            if image is None:
                raise DataPreparationError(f"Image missing or unreadable for row {row_index}.")
            question, answer, choices = vqa_question_answer(row)
            if not question:
                raise DataPreparationError(f"Question missing for row {row_index}.")
            if answer is None:
                raise DataPreparationError(f"Answer annotation missing for row {row_index}.")
            sample_id = row_sample_id(row, row_index)
            file_name = image_name(row, sample_id)
            image_path = image_root / file_name
            if not image_path.exists():
                save_image(image, image_path)
            record = {
                "sample_id": sample_id,
                "dataset": args.dataset,
                "split": args.split,
                "image_id": str(row.get("image_id") or row.get("file_name") or sample_id),
                "image_path": str(image_path.as_posix()),
                "question": question,
                "answer": answer,
                "choices": choices,
                "category": str(row.get("category") or row.get("type") or ""),
                "metadata": {"source": source_id, "source_split": actual_split, "source_row": row_index},
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1
            if args.limit is not None and count >= args.limit:
                return count
    return count


def load_local_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    args = parse_args()
    try:
        if args.raw_root.is_file() and args.raw_root.suffix.lower() == ".jsonl":
            records = load_local_manifest(args.raw_root)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as handle:
                for index, record in enumerate(records):
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                    if args.limit is not None and index + 1 >= args.limit:
                        break
            print(f"Wrote manifest from local JSONL: {args.output}")
            return 0

        if args.no_download:
            raise DataPreparationError(
                f"No local loader found for raw root '{args.raw_root}'. Provide a manifest JSONL as --raw-root or allow --download."
            )

        ensure_dependencies()
        rows, source_id, actual_split = load_hf_dataset(args.dataset, args.split)
        if args.dataset in GROUNDING_DATASETS:
            count = prepare_grounding(args, rows, source_id, actual_split)
        else:
            count = prepare_vqa(args, rows, source_id, actual_split)
        if count <= 0:
            raise DataPreparationError("No manifest records were written.")
        print(f"Wrote {count} records to {args.output}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except DataPreparationError as exc:
        fail(str(exc), affected=args.dataset)
        return 2
    except Exception as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
