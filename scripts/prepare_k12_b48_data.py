#!/usr/bin/env python
"""Prepare pinned, lossless manifests for the K=12/B=48 campaign.

Unlike the legacy preparer, this entry point never aliases split names, never
uses an unpinned dataset revision, and never re-encodes a source JPEG.  The
manifest is published atomically only after source-row, ID, image, and schema
checks pass.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import itertools
import io
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image as PILImage

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_pruning.artifacts import atomic_write_jsonl, sha256_file
from vlm_pruning.final_campaign import DATASETS, DatasetProtocol


DEFAULT_OUTPUT_ROOT = Path("data/k12_b48_expanded")
DATASET_BY_KEY = {item.key: item for item in DATASETS}
_IMAGE_RECORD_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


class PreparationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *sorted(DATASET_BY_KEY)], default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, help="Compatibility smoke only; limited manifests are never final.")
    parser.add_argument("--hf-home", type=Path, default=Path("/workspace/cache/huggingface"))
    return parser.parse_args()


def safe_id(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return text or "missing"


def _images_as_bytes(dataset: Any, image_type: Any) -> Any:
    for column in dataset.column_names:
        if column == "image" or re.fullmatch(r"image_[1-9][0-9]*", column):
            dataset = dataset.cast_column(column, image_type(decode=False))
    return dataset


def load_split(protocol: DatasetProtocol, *, streaming: bool = False) -> tuple[Iterable[dict[str, Any]], int]:
    from datasets import Image as HFImage
    from datasets import concatenate_datasets, load_dataset

    kwargs: dict[str, Any] = {"revision": protocol.revision, "streaming": streaming}
    if protocol.config:
        kwargs["name"] = protocol.config
    if protocol.key == "pope_all":
        parts = [_images_as_bytes(load_dataset(protocol.dataset_id, split=split, **kwargs), HFImage)
                 for split in ("random", "popular", "adversarial")]
        if streaming:
            return itertools.chain.from_iterable(parts), protocol.expected_rows
        dataset = concatenate_datasets(parts)
    else:
        dataset = load_dataset(protocol.dataset_id, split=protocol.split, **kwargs)
    if not streaming and len(dataset) != protocol.expected_rows:
        raise PreparationError(
            f"{protocol.key}: source row count {len(dataset)} != locked {protocol.expected_rows}"
        )
    dataset = _images_as_bytes(dataset, HFImage)
    return dataset, protocol.expected_rows if streaming else len(dataset)


def load_gqa(protocol: DatasetProtocol) -> tuple[Iterable[dict[str, Any]], int]:
    from datasets import Image as HFImage
    from datasets import load_dataset

    instructions = load_dataset(
        protocol.dataset_id,
        protocol.config,
        split=protocol.split,
        revision=protocol.revision,
    )
    images = load_dataset(
        protocol.dataset_id,
        protocol.image_dataset_config,
        split=protocol.split,
        revision=protocol.revision,
    ).cast_column("image", HFImage(decode=False))
    if len(instructions) != protocol.expected_rows:
        raise PreparationError(
            f"gqa: instruction count {len(instructions)} != locked {protocol.expected_rows}"
        )
    if len(images) != protocol.image_dataset_rows:
        raise PreparationError(
            f"gqa: image count {len(images)} != locked {protocol.image_dataset_rows}"
        )
    by_id = {str(row["id"]): row["image"] for row in images}

    def joined() -> Iterable[dict[str, Any]]:
        for row in instructions:
            item = dict(row)
            image_id = str(item["imageId"])
            if image_id not in by_id:
                raise PreparationError(f"gqa: no image for imageId={image_id}")
            item["image"] = by_id[image_id]
            yield item

    return joined(), len(instructions)


def source_bytes(value: Any) -> tuple[bytes, str]:
    if isinstance(value, dict):
        data = value.get("bytes")
        source_path = value.get("path")
        if data is not None:
            return bytes(data), Path(str(source_path or "image")).suffix.lower()
        if source_path and Path(str(source_path)).is_file():
            path = Path(str(source_path))
            return path.read_bytes(), path.suffix.lower()
    if isinstance(value, PILImage.Image):
        buffer = io.BytesIO()
        value.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue(), ".png"
    raise PreparationError(f"Unsupported or missing image payload: {type(value).__name__}")


def image_record(value: Any, image_root: Path) -> dict[str, Any]:
    source_hint = None
    if isinstance(value, dict) and value.get("path"):
        source_hint = str(value["path"])
        cached = _IMAGE_RECORD_CACHE.get((str(image_root), source_hint))
        if cached is not None:
            return dict(cached)
    data, suffix = source_bytes(value)
    byte_sha = hashlib.sha256(data).hexdigest()
    try:
        with PILImage.open(io.BytesIO(data)) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            pixel_digest = hashlib.sha256()
            pixel_digest.update(f"{width}x{height}:RGB:".encode("ascii"))
            pixel_digest.update(rgb.tobytes())
            pixel_sha = pixel_digest.hexdigest()
            detected = (image.format or "PNG").lower()
    except Exception as exc:
        raise PreparationError(f"Image decode failed: {type(exc).__name__}: {exc}") from exc
    allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if suffix not in allowed:
        suffix = ".jpg" if detected in {"jpg", "jpeg"} else f".{detected}"
    path = image_root / f"{byte_sha}{suffix}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
    if path.stat().st_size != len(data) or sha256_file(path) != byte_sha:
        raise PreparationError(f"Stored image content mismatch: {path}")
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        relative = path
    record = {
        "image_path": relative.as_posix(),
        "image_byte_sha256": byte_sha,
        "image_pixel_sha256": pixel_sha,
        "width": width,
        "height": height,
    }
    if source_hint is not None:
        _IMAGE_RECORD_CACHE[(str(image_root), source_hint)] = dict(record)
    return record


def image_values(row: dict[str, Any], protocol: DatasetProtocol) -> list[Any]:
    if protocol.key == "mmmu_val":
        return [row[f"image_{index}"] for index in range(1, 8) if row.get(f"image_{index}") is not None]
    value = row.get("image")
    return [] if value is None else [value]


def parse_choices(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = value
    else:
        parsed = value
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    return []


def mmbench_choices(row: dict[str, Any]) -> list[str]:
    choices: list[str] = []
    for label in "ABCDE":
        value = row.get(label)
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        choices.append(text)
    return choices


def answer_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return [str(value)] if value is not None else []
    result: list[str] = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("answer")
        if item is not None:
            result.append(str(item))
    return result


def base_record(
    protocol: DatasetProtocol,
    source_row: int,
    sample_id: str,
    images: list[dict[str, Any]],
) -> dict[str, Any]:
    if not images:
        raise PreparationError(f"{protocol.key} row {source_row}: no images")
    return {
        "sample_id": sample_id,
        "dataset": protocol.key,
        "split": protocol.split,
        "image_path": images[0]["image_path"],
        "image_paths": [item["image_path"] for item in images],
        "image_count": len(images),
        "image_byte_sha256": [item["image_byte_sha256"] for item in images],
        "image_pixel_sha256": [item["image_pixel_sha256"] for item in images],
        "metadata": {
            "source_dataset": protocol.dataset_id,
            "source_revision": protocol.revision,
            "source_config": protocol.config,
            "source_split": protocol.split,
            "source_row": source_row,
        },
    }


def grounding_records(
    protocol: DatasetProtocol,
    row: dict[str, Any],
    source_row: int,
    images: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(images) != 1:
        raise PreparationError(f"{protocol.key} row {source_row}: grounding expects one image")
    bbox = row.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise PreparationError(f"{protocol.key} row {source_row}: invalid bbox")
    x, y, width, height = [float(value) for value in bbox]
    image_width, image_height = images[0]["width"], images[0]["height"]
    gt_box = [
        max(0.0, min(float(image_width), x)),
        max(0.0, min(float(image_height), y)),
        max(0.0, min(float(image_width), x + width)),
        max(0.0, min(float(image_height), y + height)),
    ]
    if gt_box[2] <= gt_box[0] or gt_box[3] <= gt_box[1]:
        raise PreparationError(f"{protocol.key} row {source_row}: empty bbox after clipping")
    expressions = answer_strings(row.get("answer"))
    if not expressions:
        raise PreparationError(f"{protocol.key} row {source_row}: no referring expressions")
    base = safe_id(row.get("question_id", source_row))
    records: list[dict[str, Any]] = []
    for expression_index, query in enumerate(expressions):
        record = base_record(protocol, source_row, f"{base}__e{expression_index}", images)
        record.update(
            {
                "image_id": str(row.get("file_name") or base),
                "query": query,
                "gt_box": gt_box,
                "width": image_width,
                "height": image_height,
                "expression_index": expression_index,
                "expressions_per_object": len(expressions),
            }
        )
        records.append(record)
    return records


def vqa_record(
    protocol: DatasetProtocol,
    row: dict[str, Any],
    source_row: int,
    images: list[dict[str, Any]],
) -> dict[str, Any]:
    key = protocol.key
    if key == "gqa":
        sample_id = safe_id(row["id"])
        question, answer, choices = row["question"], row["answer"], []
        category = str((row.get("types") or {}).get("detailed") or "")
        image_id = row["imageId"]
    elif key == "mmbench_en_dev":
        sample_id = safe_id(row["index"])
        choices = mmbench_choices(row)
        question = str(row["question"])
        answer, category, image_id = row["answer"], row.get("category", ""), row["index"]
    elif key == "mmmu_val":
        sample_id = safe_id(row["id"])
        question, answer = row["question"], row["answer"]
        choices = parse_choices(row.get("options"))
        category, image_id = row.get("subfield", ""), row["id"]
    elif key == "mme":
        sample_id = f"mme_source_{source_row:06d}__{safe_id(row['question_id'])}"
        question, answer, choices = row["question"], row["answer"], []
        category, image_id = row.get("category", ""), row["question_id"]
    elif key == "vqav2_val":
        sample_id = safe_id(row["question_id"])
        question, answer, choices = row["question"], answer_strings(row["answers"]), []
        category, image_id = row.get("answer_type", ""), row["image_id"]
    elif key == "textvqa_val":
        sample_id = safe_id(row["question_id"])
        question, answer, choices = row["question"], answer_strings(row["answers"]), []
        category, image_id = "", row["image_id"]
    elif key == "pope_all":
        sample_id = f"{safe_id(row.get('category', 'unknown'))}__{safe_id(row.get('question_id') or row['id'])}"
        question, answer, choices = row["question"], row["answer"], []
        category, image_id = row.get("category", ""), row.get("id")
    elif key == "scienceqa_img_test":
        sample_id = f"scienceqa_test_{source_row:06d}"
        choices = [str(value) for value in row["choices"]]
        answer = chr(ord("A") + int(row["answer"]))
        question = str(row["question"])
        category, image_id = row.get("subject", ""), sample_id
    elif key == "mmvet_test":
        sample_id = safe_id(row["question_id"])
        question, answer, choices = row["question"], row["answer"], []
        category, image_id = row.get("capability", ""), row["question_id"]
    else:
        raise PreparationError(f"No VQA conversion for {key}")
    record = base_record(protocol, source_row, sample_id, images)
    record.update(
        {
            "image_id": str(image_id),
            "question": str(question),
            "answer": answer,
            "choices": choices,
            "answer_type": (
                "multiple_choice"
                if choices
                else "yes_no"
                if str(answer).strip().lower() in {"yes", "no"}
                else "open"
            ),
            "category": str(category),
        }
    )
    if key == "mmmu_val":
        record["question_type"] = row.get("question_type")
        record["subfield"] = row.get("subfield")
        record["options_raw"] = row.get("options")
        record["metadata"]["multi_image_extension"] = len(images) > 1
    if key == "mmbench_en_dev":
        hint = str(row.get("hint") or "").strip()
        record.update(
            {
                "hint": "" if hint.lower() == "nan" else hint,
                "index": row.get("index"),
                "source": row.get("source"),
                "l2_category": row.get("L2-category"),
            }
        )
    if key == "scienceqa_img_test":
        record["hint"] = str(row.get("hint") or "")
        record["answer_index"] = int(row["answer"])
        record["subject"] = row.get("subject")
    if key == "mme":
        record["mme_pair_id"] = f"{str(row.get('category') or '').lower()}::{row['question_id']}"
    if key == "mmvet_test":
        record["capability"] = row.get("capability")
    return record


def validate_manifest(
    protocol: DatasetProtocol,
    records: list[dict[str, Any]],
    source_rows: int,
    *,
    limited: bool,
) -> dict[str, Any]:
    issues: list[str] = []
    ids = [row["sample_id"] for row in records]
    if len(ids) != len(set(ids)):
        issues.append("duplicate sample IDs")
    if not limited and source_rows != protocol.expected_rows:
        issues.append(f"source rows {source_rows} != {protocol.expected_rows}")
    expected_manifest_rows = protocol.expected_manifest_rows or protocol.expected_rows
    if not limited and len(records) != expected_manifest_rows:
        issues.append(f"manifest rows {len(records)} != {expected_manifest_rows}")
    if not records:
        issues.append("manifest has no records")
    missing_images = []
    for row in records:
        for path in row.get("image_paths", []):
            resolved = Path(path)
            if not resolved.is_absolute():
                resolved = ROOT / resolved
            if not resolved.is_file():
                missing_images.append(path)
    if missing_images:
        issues.append(f"missing images: {missing_images[:5]}")
    if protocol.task == "grounding" and any(
        not row.get("query") or len(row.get("gt_box", [])) != 4 for row in records
    ):
        issues.append("grounding schema failure")
    if protocol.task == "vqa" and any(
        not row.get("question") or row.get("answer") is None for row in records
    ):
        issues.append("VQA schema failure")
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "source_rows": source_rows,
        "manifest_rows": len(records),
        "unique_sample_ids": len(set(ids)),
        "unique_images": len({path for row in records for path in row.get("image_paths", [])}),
        "image_count_histogram": {
            str(count): sum(int(row.get("image_count", 0)) == count for row in records)
            for count in sorted({int(row.get("image_count", 0)) for row in records})
        },
        "limited": limited,
    }


def prepare(protocol: DatasetProtocol, output_root: Path, limit: int | None) -> dict[str, Any]:
    dataset_root = ROOT / output_root / protocol.key
    image_root = dataset_root / "images"
    rows, source_rows = (
        load_gqa(protocol)
        if protocol.key == "gqa"
        else load_split(
            protocol,
            streaming=protocol.key in {"mmmu_val", "vqav2_val", "textvqa_val"},
        )
    )
    records: list[dict[str, Any]] = []
    processed_sources = 0
    for source_row, raw in enumerate(rows):
        row = dict(raw)
        stored_images = [image_record(value, image_root) for value in image_values(row, protocol)]
        converted = (
            grounding_records(protocol, row, source_row, stored_images)
            if protocol.task == "grounding"
            else [vqa_record(protocol, row, source_row, stored_images)]
        )
        records.extend(converted)
        processed_sources += 1
        if processed_sources % 1000 == 0:
            print(f"{protocol.key}: prepared {processed_sources}/{source_rows} source rows", flush=True)
        if limit is not None and processed_sources >= limit:
            break
    limited = limit is not None
    if not limited and processed_sources != protocol.expected_rows:
        raise PreparationError(
            f"{protocol.key}: streamed source rows {processed_sources} != locked {protocol.expected_rows}"
        )
    if protocol.key == "mme" and not limited:
        from vlm_pruning.mme import unique_manifest_rows

        records = unique_manifest_rows(records)
    report = validate_manifest(protocol, records, processed_sources if limited else source_rows, limited=limited)
    if report["status"] != "pass":
        raise PreparationError(f"{protocol.key}: " + "; ".join(report["issues"]))
    manifest = dataset_root / ("manifest.smoke.jsonl" if limited else "manifest.jsonl")
    atomic_write_jsonl(manifest, records)
    reopened = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    second = validate_manifest(protocol, reopened, processed_sources if limited else source_rows, limited=limited)
    if second["status"] != "pass" or len(reopened) != len(records):
        raise PreparationError(f"{protocol.key}: atomic manifest reopen failed")
    report.update(
        {
            "protocol": asdict(protocol),
            "manifest": str(manifest.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest),
        }
    )
    provenance = dataset_root / ("provenance.smoke.json" if limited else "provenance.json")
    provenance.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    os.environ["HF_HOME"] = str(args.hf_home)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    selected = DATASETS if args.dataset == "all" else (DATASET_BY_KEY[args.dataset],)
    reports: list[dict[str, Any]] = []
    try:
        for protocol in selected:
            reports.append(prepare(protocol, args.output_root, args.limit))
    except PreparationError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "status": "pass",
                "datasets": [
                    {
                        "key": report["protocol"]["key"],
                        "source_rows": report["source_rows"],
                        "manifest_rows": report["manifest_rows"],
                        "manifest_sha256": report["manifest_sha256"],
                    }
                    for report in reports
                ],
            },
            indent=2,
        )
    )
    if args.limit is not None:
        # Streaming uses background HTTP/Arrow resources. Give their finalizers
        # a deterministic window before interpreter shutdown; otherwise some
        # pyarrow builds can abort after the validated smoke manifest is safely
        # published.
        gc.collect()
        time.sleep(2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
