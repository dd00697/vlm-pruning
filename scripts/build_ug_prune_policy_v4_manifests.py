#!/usr/bin/env python
"""Build deterministic held-out manifests for v4 UG-Prune policy learning."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_official_baseline import write_json  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("data/manifests/ug_prune_policy_v4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textvqa-manifest", type=Path, default=Path("data/manifests/official_baselines/textvqa_validation.jsonl"))
    parser.add_argument("--pope-manifest", type=Path, default=Path("data/manifests/official_baselines/pope_val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--textvqa-counts", default="2000,1000,2000", help="train,dev,final counts.")
    parser.add_argument("--pope-counts", default="3000,2000,4000", help="train,dev,final counts.")
    parser.add_argument("--seed", default="ug-prune-policy-v4-low68")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def parse_counts(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("Counts must be train,dev,final.")
    return parts[0], parts[1], parts[2]


def stable_hash(row: dict[str, Any], *, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{row_key(row)}".encode("utf-8")).hexdigest()


def row_key(row: dict[str, Any]) -> str:
    dataset = str(row.get("dataset") or "")
    sample_id = str(row.get("sample_id") or row.get("id") or row.get("image_id") or "")
    category = str(row.get("category") or "")
    raw_source_row = (row.get("metadata") or {}).get("source_row")
    source_row = "" if raw_source_row is None else str(raw_source_row)
    return f"{dataset}:{category}:{sample_id}:{source_row}"


def stratum_key(row: dict[str, Any]) -> str:
    dataset = str(row.get("dataset") or "").lower()
    if dataset == "pope":
        category = str(row.get("category") or (row.get("metadata") or {}).get("category") or "unknown")
        answer = str(row.get("answer") or "unknown").lower()
        return f"{category}|{answer}"
    category = str(row.get("category") or "unknown")
    answer_type = str(row.get("answer_type") or "unknown")
    return f"{category}|{answer_type}"


def allocate(total: int, strata: dict[str, list[dict[str, Any]]], *, target: int) -> dict[str, int]:
    if target < 0 or target > total:
        raise ValueError(f"Target {target} is outside available row count {total}.")
    exact = [(key, len(rows) * target / total) for key, rows in strata.items()]
    allocations = {key: int(value) for key, value in exact}
    remaining = target - sum(allocations.values())
    order = sorted(exact, key=lambda item: (item[1] - int(item[1]), len(strata[item[0]]), item[0]), reverse=True)
    for key, _value in order[:remaining]:
        allocations[key] += 1
    return allocations


def split_rows(rows: list[dict[str, Any]], *, counts: tuple[int, int, int], seed: str) -> dict[str, list[dict[str, Any]]]:
    train_target, dev_target, final_target = counts
    if train_target + dev_target + final_target != len(rows):
        raise ValueError(f"Requested counts {counts} do not sum to {len(rows)} rows.")

    strata: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        strata.setdefault(stratum_key(row), []).append(row)
    for key, bucket in strata.items():
        bucket.sort(key=lambda row: stable_hash(row, seed=f"{seed}:{key}"))

    train_alloc = allocate(len(rows), strata, target=train_target)
    remaining_strata: dict[str, list[dict[str, Any]]] = {}
    train_rows: list[dict[str, Any]] = []
    for key, bucket in strata.items():
        train_n = train_alloc[key]
        train_rows.extend(bucket[:train_n])
        remaining_strata[key] = bucket[train_n:]

    remaining_total = sum(len(bucket) for bucket in remaining_strata.values())
    dev_alloc = allocate(remaining_total, remaining_strata, target=dev_target)
    dev_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for key, bucket in remaining_strata.items():
        dev_n = dev_alloc[key]
        dev_rows.extend(bucket[:dev_n])
        final_rows.extend(bucket[dev_n:])

    for split_rows_out in (train_rows, dev_rows, final_rows):
        split_rows_out.sort(key=lambda row: stable_hash(row, seed=f"{seed}:output"))

    if len(final_rows) != final_target:
        raise ValueError(f"Final split has {len(final_rows)} rows, expected {final_target}.")
    return {"train": train_rows, "dev": dev_rows, "final": final_rows}


def row_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {row_key(row) for row in rows}


def build_one(name: str, source: Path, counts: tuple[int, int, int], output_dir: Path, seed: str) -> dict[str, Any]:
    rows = load_jsonl(source)
    splits = split_rows(rows, counts=counts, seed=f"{seed}:{name}")
    overlap = row_ids(splits["train"]) & row_ids(splits["dev"]) | row_ids(splits["train"]) & row_ids(splits["final"]) | row_ids(splits["dev"]) & row_ids(splits["final"])
    if overlap:
        raise ValueError(f"{name} split overlap detected: {sorted(overlap)[:3]}")

    outputs: dict[str, str] = {}
    for split_name, split_rows_out in splits.items():
        path = output_dir / f"{name}_{split_name}.jsonl"
        write_jsonl(path, split_rows_out)
        outputs[split_name] = str(path)
    return {
        "source": str(source),
        "counts": {split_name: len(split_rows_out) for split_name, split_rows_out in splits.items()},
        "paths": outputs,
        "strata": sorted({stratum_key(row) for row in rows}),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": args.seed,
        "textvqa": build_one("textvqa", args.textvqa_manifest, parse_counts(args.textvqa_counts), args.output_dir, args.seed),
        "pope": build_one("pope", args.pope_manifest, parse_counts(args.pope_counts), args.output_dir, args.seed),
    }
    write_json(args.output_dir / "metadata.json", summary)
    print(f"Wrote v4 policy manifests to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
