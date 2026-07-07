#!/usr/bin/env python
"""Build leakage-safe manifests for UG-Prune-Lite training and dev checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm_pruning.training.ug_prune_lite_dataset import read_jsonl  # noqa: E402


DEFAULT_VAL = Path("data/manifests/official_baselines/refcoco_val.jsonl")
DEFAULT_OUTPUT = Path("data/manifests/ug_prune_lite")


class ManifestBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class IndexedRow:
    source_index: int
    row: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train-manifest", type=Path, default=None)
    parser.add_argument("--source-dev-manifest", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--reserved-eval-manifest", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--reserved-eval-limit", type=int, default=500)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=1000)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--use-all-allowed", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-missing-train", action="store_true")
    parser.add_argument("--allow-nonfinal-val-fallback", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_indices(path: Path, rows: list[IndexedRow]) -> None:
    path.write_text("\n".join(str(item.source_index) for item in rows) + "\n", encoding="utf-8")


def checksum_ids(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(str(row["sample_id"]) for row in rows):
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def candidate_train_manifests(output_dir: Path) -> list[Path]:
    return [
        Path("data/manifests/official_baselines/refcoco_train.jsonl"),
        Path("data/manifests/official_baselines/refcoco_train_ug_prune_lite_source.jsonl"),
        output_dir / "refcoco_train_source.jsonl",
    ]


def prepare_train_manifest(train_limit: int | None) -> tuple[Path | None, str | None]:
    prepared = Path("data/manifests/official_baselines/refcoco_train_ug_prune_lite_source.jsonl")
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "prepare_official_baseline_data.py"),
        "--dataset",
        "refcoco",
        "--split",
        "train",
        "--output",
        str(prepared),
    ]
    if train_limit is not None:
        cmd.extend(["--limit", str(train_limit)])
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return prepared, None


def load_indexed(path: Path, *, start_index: int = 0) -> list[IndexedRow]:
    rows: list[IndexedRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start_index or not line.strip():
                continue
            rows.append(IndexedRow(index, json.loads(line)))
    return rows


def load_reserved(path: Path, limit: int) -> tuple[list[dict[str, Any]], set[str]]:
    rows = read_jsonl(path, limit=limit)
    return rows, {str(row["sample_id"]) for row in rows}


def select_indexed(
    rows: list[IndexedRow],
    *,
    exclude: set[str],
    limit: int | None,
    used: set[str] | None = None,
) -> list[IndexedRow]:
    used = used or set()
    out: list[IndexedRow] = []
    for item in rows:
        sample_id = str(item.row["sample_id"])
        if sample_id in exclude or sample_id in used:
            continue
        out.append(item)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def split_allowed_pool(
    eligible: list[IndexedRow],
    *,
    reserved_ids: set[str],
    train_limit: int | None,
    dev_limit: int | None,
    train_fraction: float | None,
    use_all_allowed: bool,
    seed: int,
    shuffle: bool,
) -> tuple[list[IndexedRow], list[IndexedRow]]:
    pool = []
    seen: set[str] = set()
    for item in eligible:
        sample_id = str(item.row["sample_id"])
        if sample_id in reserved_ids or sample_id in seen:
            continue
        seen.add(sample_id)
        pool.append(item)
    if shuffle:
        pool = list(pool)
        random.Random(seed).shuffle(pool)
    if train_fraction is not None:
        if not 0.0 < train_fraction < 1.0:
            raise ManifestBuildError("--train-fraction must be between 0 and 1.")
        train_count = int(len(pool) * float(train_fraction))
        if train_limit is not None:
            train_count = min(train_count, int(train_limit))
        dev_count = len(pool) - train_count if dev_limit is None else min(int(dev_limit), max(0, len(pool) - train_count))
        return pool[:train_count], pool[train_count : train_count + dev_count]

    dev_count = min(int(dev_limit or 0), len(pool))
    if use_all_allowed:
        train_count = max(0, len(pool) - dev_count)
    else:
        train_count = min(int(train_limit) if train_limit is not None else max(0, len(pool) - dev_count), max(0, len(pool) - dev_count))
    train_rows = pool[:train_count]
    dev_rows = pool[train_count : train_count + dev_count]
    return train_rows, dev_rows


def overlap_count(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    return len({str(row["sample_id"]) for row in left} & {str(row["sample_id"]) for row in right})


def index_summary(rows: list[IndexedRow]) -> dict[str, Any]:
    indices = [item.source_index for item in rows]
    return {
        "count": len(indices),
        "min": min(indices) if indices else None,
        "max": max(indices) if indices else None,
        "first_10": indices[:10],
        "last_10": indices[-10:],
    }


def output_name(prefix: str, count: int, *, all_allowed: bool) -> str:
    return f"{prefix}_allowed.jsonl" if all_allowed else f"{prefix}_{count}.jsonl"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reserved_rows, reserved_ids = load_reserved(args.reserved_eval_manifest, args.reserved_eval_limit)
    reserved_ids_path = args.output_dir / "refcoco_reserved_eval_first500_sample_ids.txt"
    reserved_ids_path.write_text("\n".join(sorted(reserved_ids)) + "\n", encoding="utf-8")

    source_train = args.source_train_manifest
    prepare_error = None
    if source_train is None:
        for candidate in candidate_train_manifests(args.output_dir):
            if candidate.exists():
                source_train = candidate
                break
    if source_train is None and args.prepare_missing_train:
        source_train, prepare_error = prepare_train_manifest(args.train_limit)

    split_kind = "official_train"
    fallback_reason = None
    if source_train is not None and source_train.exists():
        train_indexed = select_indexed(
            load_indexed(source_train),
            exclude=reserved_ids,
            limit=args.train_limit,
        )
        train_ids = {str(item.row["sample_id"]) for item in train_indexed}
        dev_indexed = select_indexed(
            load_indexed(args.source_dev_manifest, start_index=args.reserved_eval_limit),
            exclude=reserved_ids,
            limit=args.dev_limit,
            used=train_ids,
        )
        if not train_indexed:
            raise ManifestBuildError(f"No usable train rows found in {source_train}")
    else:
        if not args.allow_nonfinal_val_fallback:
            raise ManifestBuildError(
                "No real RefCOCO train manifest found. Re-run with --prepare-missing-train or "
                "--allow-nonfinal-val-fallback for explicitly non-final splits."
            )
        split_kind = "nonfinal_val_after_reserved_fallback"
        fallback_reason = prepare_error or "No real RefCOCO train manifest found locally."
        train_indexed, dev_indexed = split_allowed_pool(
            load_indexed(args.source_dev_manifest, start_index=args.reserved_eval_limit),
            reserved_ids=reserved_ids,
            train_limit=args.train_limit,
            dev_limit=args.dev_limit,
            train_fraction=args.train_fraction,
            use_all_allowed=args.use_all_allowed,
            seed=args.seed,
            shuffle=args.shuffle,
        )

    if not train_indexed:
        raise ManifestBuildError("No usable train rows were found after applying reserved-eval exclusions.")
    if not dev_indexed:
        raise ManifestBuildError("No usable dev rows were found after applying reserved-eval exclusions.")

    train_rows = [item.row for item in train_indexed]
    dev_rows = [item.row for item in dev_indexed]
    train_path = args.output_dir / output_name("refcoco_train", len(train_rows), all_allowed=args.use_all_allowed and split_kind != "official_train")
    dev_path = args.output_dir / output_name("refcoco_dev", len(dev_rows), all_allowed=args.use_all_allowed and split_kind != "official_train")
    train_indices_path = args.output_dir / f"{train_path.stem}_source_indices.txt"
    dev_indices_path = args.output_dir / f"{dev_path.stem}_source_indices.txt"
    write_jsonl(train_path, train_rows)
    write_jsonl(dev_path, dev_rows)
    write_indices(train_indices_path, train_indexed)
    write_indices(dev_indices_path, dev_indexed)

    train_eval_overlap = len({str(row["sample_id"]) for row in train_rows} & reserved_ids)
    dev_eval_overlap = len({str(row["sample_id"]) for row in dev_rows} & reserved_ids)
    train_dev_overlap = overlap_count(train_rows, dev_rows)
    if train_eval_overlap or dev_eval_overlap or train_dev_overlap:
        raise ManifestBuildError(
            f"Leakage detected: train_eval={train_eval_overlap}, dev_eval={dev_eval_overlap}, train_dev={train_dev_overlap}"
        )

    metadata = {
        "method": "ug_prune_lite_manifest_hygiene",
        "split_kind": split_kind,
        "fallback_warning": split_kind == "nonfinal_val_after_reserved_fallback",
        "fallback_reason": fallback_reason,
        "source_train_manifest": str(source_train) if source_train else None,
        "source_dev_manifest": str(args.source_dev_manifest),
        "reserved_eval_manifest": str(args.reserved_eval_manifest),
        "reserved_eval_limit": args.reserved_eval_limit,
        "reserved_eval_count": len(reserved_rows),
        "reserved_eval_rows": {"start": 0, "end_inclusive": args.reserved_eval_limit - 1},
        "reserved_eval_ids_path": str(reserved_ids_path),
        "train_manifest": str(train_path),
        "dev_manifest": str(dev_path),
        "train_source_indices_path": str(train_indices_path),
        "dev_source_indices_path": str(dev_indices_path),
        "train_source_index_summary": index_summary(train_indexed),
        "dev_source_index_summary": index_summary(dev_indexed),
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "allowed_pool_count": sum(1 for _ in load_indexed(args.source_dev_manifest, start_index=args.reserved_eval_limit)),
        "train_checksum": checksum_ids(train_rows),
        "dev_checksum": checksum_ids(dev_rows),
        "reserved_eval_checksum": checksum_ids(reserved_rows),
        "overlap/train_dev": train_dev_overlap,
        "overlap/train_reserved_eval": train_eval_overlap,
        "overlap/dev_reserved_eval": dev_eval_overlap,
        "prepare_missing_train": bool(args.prepare_missing_train),
        "prepare_error": prepare_error,
        "use_all_allowed": bool(args.use_all_allowed),
        "train_limit": args.train_limit,
        "dev_limit": args.dev_limit,
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "shuffle": bool(args.shuffle),
    }
    write_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
