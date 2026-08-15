#!/usr/bin/env python
"""Freeze the train-only RefCOCO manifest used by the adaptive route gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pickle
import tempfile
import time
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
DEFAULT_ARCHIVE = CAMPAIGN_ROOT / "gate_data/sources/refcoco.zip"
DEFAULT_OUTPUT = CAMPAIGN_ROOT / "gate_data/refcoco_train_gate3000.jsonl"
DEFAULT_IMAGE_ROOT = CAMPAIGN_ROOT / "gate_data/images"
VALIDATION_MANIFEST = REPO_ROOT / (
    "experiments/progressive_oracle_refcoco_val_20260814/"
    "manifests/refcoco_val_discovery1000.jsonl"
)
SOURCE_URL = (
    "https://web.archive.org/web/20220413011718id_/"
    "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip"
)
ARCHIVE_SHA256 = "7f924bb7ed8dc4568058e4ff626281918d56e5206f4c868c5a80f088f38c8bf0"
REFER_REPOSITORY = "https://github.com/lichengunc/refer"
REFER_COMMIT = "e3bbaa30d2ca41cf0e5c0d3819d7e4ed9fd38fff"
# The COCO S3 endpoint's current TLS certificate does not cover the historical
# images.cocodataset.org alias.  Its documented HTTP object endpoint serves the
# same immutable JPEG objects without redirecting through that invalid TLS name.
COCO_IMAGE_URL = "http://images.cocodataset.org/train2014/{filename}"
SELECTION_SEED = "adaptive-route-gate-refcoco-train-v1"
TOTAL_RECORDS = 3000
DEVELOPMENT_RECORDS = 2400
CALIBRATION_RECORDS = 600
FIT_RECORDS = 2160
EARLY_STOP_RECORDS = 240


class GateManifestError(RuntimeError):
    """Raised when the frozen train-only gate-data contract is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def stable_hash(namespace: str, value: int | str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}|{namespace}|{value}".encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise GateManifestError(f"Non-object JSONL row in {path}")
    return rows


def validation_original_ids(rows: Iterable[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for row in rows:
        metadata = row.get("metadata") or {}
        original = metadata.get("original_sample_id")
        if not isinstance(original, str) or not original:
            raise GateManifestError("Frozen validation row lacks original_sample_id")
        result.add(original)
    return result


def load_source(archive: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if sha256_file(archive) != ARCHIVE_SHA256:
        raise GateManifestError("RefCOCO source archive hash differs from the frozen archive")
    with zipfile.ZipFile(archive) as source:
        expected_members = {
            "refcoco/instances.json",
            "refcoco/refs(google).p",
            "refcoco/refs(unc).p",
        }
        if set(source.namelist()) != expected_members:
            raise GateManifestError("RefCOCO source archive members differ from the frozen layout")
        refs_payload = source.read("refcoco/refs(unc).p")
        instances_payload = source.read("refcoco/instances.json")
        # The official 2016 release stores plain builtin containers in a
        # Python-2 pickle.  The archive hash above authenticates those bytes.
        refs = pickle.loads(refs_payload, encoding="latin1")  # noqa: S301
        instances = json.loads(instances_payload)
        member_evidence = {
            info.filename: {
                "bytes": info.file_size,
                "crc32": f"{info.CRC:08x}",
                "sha256": hashlib.sha256(source.read(info.filename)).hexdigest(),
            }
            for info in source.infolist()
        }
    if not isinstance(refs, list) or len(refs) != 50000:
        raise GateManifestError("Official UNC reference table does not contain 50,000 references")
    if not isinstance(instances, dict):
        raise GateManifestError("Official instance table is malformed")
    return refs, instances, member_evidence


def choose_records(
    refs: list[dict[str, Any]],
    *,
    excluded_original_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose one reference per image, then hash-rank 3,000 unique images."""
    train_refs: list[tuple[int, dict[str, Any]]] = []
    excluded = 0
    for source_index, ref in enumerate(refs):
        if ref.get("split") != "train":
            continue
        original_id = f"{int(ref['ann_id'])}_0"
        if original_id in excluded_original_ids:
            excluded += 1
            continue
        train_refs.append((source_index, ref))
    by_image: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for item in train_refs:
        by_image[int(item[1]["image_id"])].append(item)
    representatives: list[tuple[int, dict[str, Any]]] = []
    for image_id, image_refs in by_image.items():
        representatives.append(
            min(
                image_refs,
                key=lambda item: (
                    stable_hash("reference", int(item[1]["ref_id"])),
                    int(item[1]["ref_id"]),
                ),
            )
        )
    representatives.sort(
        key=lambda item: (
            stable_hash("image", int(item[1]["image_id"])),
            int(item[1]["image_id"]),
        )
    )
    if len(representatives) < TOTAL_RECORDS:
        raise GateManifestError("Fewer than 3,000 eligible unique RefCOCO-train images")
    selected = representatives[:TOTAL_RECORDS]

    calibration_keys = {
        int(item[1]["image_id"])
        for item in sorted(
            selected,
            key=lambda item: (
                stable_hash("development-calibration", int(item[1]["image_id"])),
                int(item[1]["image_id"]),
            ),
        )[:CALIBRATION_RECORDS]
    }
    development = [item for item in selected if int(item[1]["image_id"]) not in calibration_keys]
    early_stop_keys = {
        int(item[1]["image_id"])
        for item in sorted(
            development,
            key=lambda item: (
                stable_hash("fit-early-stop", int(item[1]["image_id"])),
                int(item[1]["image_id"]),
            ),
        )[:EARLY_STOP_RECORDS]
    }
    annotated: list[dict[str, Any]] = []
    for source_index, ref in selected:
        image_id = int(ref["image_id"])
        if image_id in calibration_keys:
            gate_partition = "calibration"
            training_partition = "calibration"
        elif image_id in early_stop_keys:
            gate_partition = "development"
            training_partition = "early_stop"
        else:
            gate_partition = "development"
            training_partition = "fit"
        annotated.append(
            {
                "source_index": source_index,
                "ref": ref,
                "gate_partition": gate_partition,
                "training_partition": training_partition,
                "selection_hash": stable_hash("image", image_id),
                "split_hash": stable_hash("development-calibration", image_id),
                "subsplit_hash": stable_hash("fit-early-stop", image_id),
            }
        )
    counts: dict[str, int] = defaultdict(int)
    for item in annotated:
        counts[item["training_partition"]] += 1
    expected_counts = {"fit": FIT_RECORDS, "early_stop": EARLY_STOP_RECORDS, "calibration": CALIBRATION_RECORDS}
    if dict(counts) != expected_counts:
        raise GateManifestError(f"Gate split counts {dict(counts)} != {expected_counts}")
    evidence = {
        "official_unc_references": len(refs),
        "official_unc_train_references": sum(ref.get("split") == "train" for ref in refs),
        "validation_original_ids_excluded": excluded,
        "eligible_train_references": len(train_refs),
        "eligible_unique_images": len(by_image),
        "sampling_unit": "unique_image_with_one_hash_selected_reference",
        "records": TOTAL_RECORDS,
        "development_records": DEVELOPMENT_RECORDS,
        "calibration_records": CALIBRATION_RECORDS,
        "fit_records": FIT_RECORDS,
        "early_stop_records": EARLY_STOP_RECORDS,
    }
    return annotated, evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def validate_image(path: Path, width: int, height: int) -> str:
    with Image.open(path) as image:
        image.load()
        if image.size != (width, height):
            raise GateManifestError(
                f"Image dimensions for {path} are {image.size}, expected {(width, height)}"
            )
        if image.mode not in {"RGB", "L"}:
            image.convert("RGB")
    return sha256_file(path)


def download_image(path: Path, *, filename: str, width: int, height: int) -> str:
    if path.is_file():
        return validate_image(path, width, height)
    path.parent.mkdir(parents=True, exist_ok=True)
    url = COCO_IMAGE_URL.format(filename=filename)
    failure: Exception | None = None
    for attempt in range(4):
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{attempt}.part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "vlm-pruning-gate-data/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
            validate_image(temporary, width, height)
            os.replace(temporary, path)
            return sha256_file(path)
        except Exception as exc:  # pragma: no cover - network failures are environment-specific
            failure = exc
            if temporary.exists():
                temporary.unlink()
            if attempt < 3:
                time.sleep(2**attempt)
    raise GateManifestError(f"Failed to download {url}: {failure}")


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def build_manifest_rows(
    selected: list[dict[str, Any]],
    instances: dict[str, Any],
    image_root: Path,
    *,
    workers: int,
    download: bool,
) -> list[dict[str, Any]]:
    image_by_id = {int(row["id"]): row for row in instances["images"]}
    annotation_by_id = {int(row["id"]): row for row in instances["annotations"]}
    specifications: list[dict[str, Any]] = []
    for item in selected:
        ref = item["ref"]
        image_id = int(ref["image_id"])
        ann_id = int(ref["ann_id"])
        image = image_by_id.get(image_id)
        annotation = annotation_by_id.get(ann_id)
        if image is None or annotation is None or int(annotation["image_id"]) != image_id:
            raise GateManifestError(f"Missing or mismatched instance evidence for annotation {ann_id}")
        sentences = sorted(ref["sentences"], key=lambda row: int(row["sent_id"]))
        if not sentences:
            raise GateManifestError(f"Reference {ref['ref_id']} has no expressions")
        sentence = sentences[0]
        filename = str(image["file_name"])
        path = image_root / filename
        specifications.append(
            {
                "item": item,
                "image": image,
                "annotation": annotation,
                "sentence": sentence,
                "filename": filename,
                "path": path,
            }
        )
    image_hashes: dict[int, str] = {}
    if download:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_spec = {
                executor.submit(
                    download_image,
                    spec["path"],
                    filename=spec["filename"],
                    width=int(spec["image"]["width"]),
                    height=int(spec["image"]["height"]),
                ): spec
                for spec in specifications
            }
            for completed, future in enumerate(concurrent.futures.as_completed(future_to_spec), 1):
                spec = future_to_spec[future]
                image_hashes[int(spec["image"]["id"])] = future.result()
                if completed % 100 == 0 or completed == len(specifications):
                    print(f"gate manifest images: prepared {completed}/{len(specifications)}", flush=True)
    else:
        for spec in specifications:
            if not spec["path"].is_file():
                raise GateManifestError(f"Missing image under --no-download: {spec['path']}")
            image_hashes[int(spec["image"]["id"])] = validate_image(
                spec["path"], int(spec["image"]["width"]), int(spec["image"]["height"])
            )

    rows: list[dict[str, Any]] = []
    for spec in specifications:
        item = spec["item"]
        ref = item["ref"]
        image = spec["image"]
        annotation = spec["annotation"]
        sentence = spec["sentence"]
        x, y, width, height = [float(value) for value in annotation["bbox"]]
        image_width, image_height = int(image["width"]), int(image["height"])
        gt_box = [
            max(0.0, min(float(image_width), x)),
            max(0.0, min(float(image_height), y)),
            max(0.0, min(float(image_width), x + width)),
            max(0.0, min(float(image_height), y + height)),
        ]
        if gt_box[2] <= gt_box[0] or gt_box[3] <= gt_box[1]:
            raise GateManifestError(f"Empty bounding box for annotation {ref['ann_id']}")
        rows.append(
            {
                "sample_id": f"refcoco_train_{int(ref['ann_id']):012d}_0",
                "dataset": "refcoco",
                "split": "train",
                "image_id": str(ref["file_name"]),
                "image_path": relative(spec["path"]),
                "image_byte_sha256": image_hashes[int(image["id"])],
                "query": str(sentence.get("raw") or sentence.get("sent") or "").strip(),
                "gt_box": gt_box,
                "width": image_width,
                "height": image_height,
                "gate_partition": item["gate_partition"],
                "training_partition": item["training_partition"],
                "metadata": {
                    "source": "lichengunc/refer RefCOCO UNC",
                    "source_archive_sha256": ARCHIVE_SHA256,
                    "source_repository_commit": REFER_COMMIT,
                    "source_split": "train",
                    "source_row": int(item["source_index"]),
                    "ref_id": int(ref["ref_id"]),
                    "ann_id": int(ref["ann_id"]),
                    "coco_image_id": int(ref["image_id"]),
                    "sent_id": int(sentence["sent_id"]),
                    "expression_index": 0,
                    "original_sample_id": f"{int(ref['ann_id'])}_0",
                    "selection_hash": item["selection_hash"],
                    "split_hash": item["split_hash"],
                    "subsplit_hash": item["subsplit_hash"],
                    "one_reference_per_image": True,
                },
            }
        )
    return rows


def validate_rows(rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != TOTAL_RECORDS or len({row["sample_id"] for row in rows}) != TOTAL_RECORDS:
        raise GateManifestError("Gate manifest must contain exactly 3,000 unique sample IDs")
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["training_partition"])] += 1
    expected = {"fit": FIT_RECORDS, "early_stop": EARLY_STOP_RECORDS, "calibration": CALIBRATION_RECORDS}
    if dict(counts) != expected:
        raise GateManifestError(f"Final gate split counts {dict(counts)} != {expected}")
    image_ids = [int(row["metadata"]["coco_image_id"]) for row in rows]
    if len(set(image_ids)) != TOTAL_RECORDS:
        raise GateManifestError("Gate manifest image grouping is not unique")
    train_original = {str(row["metadata"]["original_sample_id"]) for row in rows}
    validation_original = validation_original_ids(validation_rows)
    record_overlap = sorted(train_original.intersection(validation_original))
    if record_overlap:
        raise GateManifestError(f"Gate/validation record-ID leakage: {record_overlap[:5]}")
    validation_images = {
        int((row.get("image_id") or "").split("_")[-2])
        for row in validation_rows
        if len((row.get("image_id") or "").split("_")) >= 2
    }
    image_overlap = sorted(set(image_ids).intersection(validation_images))
    return {
        "record_id_overlap_with_frozen_validation": 0,
        "coco_image_id_overlap_with_frozen_validation": len(image_overlap),
        "overlapping_coco_image_ids": image_overlap,
        "unique_gate_images": len(set(image_ids)),
        "partition_counts": dict(counts),
    }


def main() -> int:
    args = parse_args()
    if args.workers <= 0 or args.workers > 64:
        raise GateManifestError("--workers must be in [1, 64]")
    archive = args.archive.resolve()
    output = args.output.resolve()
    image_root = args.image_root.resolve()
    validation_rows = load_jsonl(VALIDATION_MANIFEST)
    if len(validation_rows) != 1000:
        raise GateManifestError("Frozen validation manifest no longer contains 1,000 rows")
    refs, instances, member_evidence = load_source(archive)
    selected, selection_evidence = choose_records(
        refs,
        excluded_original_ids=validation_original_ids(validation_rows),
    )
    rows = build_manifest_rows(
        selected,
        instances,
        image_root,
        workers=args.workers,
        download=not args.no_download,
    )
    leakage = validate_rows(rows, validation_rows)
    payload = b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8") for row in rows
    )
    atomic_write(output, payload)
    audit = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "manifest": relative(output),
        "manifest_sha256": sha256_file(output),
        "image_root": relative(image_root),
        "source": {
            "archive": relative(archive),
            "archive_sha256": sha256_file(archive),
            "archive_url": SOURCE_URL,
            "archive_members": member_evidence,
            "official_repository": REFER_REPOSITORY,
            "official_repository_commit": REFER_COMMIT,
            "split_definition": "RefCOCO UNC train",
        },
        "selection": {
            "algorithm": "sha256_rank_unique_image_then_sha256_rank_reference_v1",
            "seed": SELECTION_SEED,
            **selection_evidence,
        },
        "leakage_audit": leakage,
        "frozen_validation_manifest": relative(VALIDATION_MANIFEST),
        "frozen_validation_manifest_sha256": sha256_file(VALIDATION_MANIFEST),
    }
    audit_path = output.with_name("MANIFEST_AUDIT.json")
    atomic_write(audit_path, canonical_bytes(audit))
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateManifestError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
