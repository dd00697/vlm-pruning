#!/usr/bin/env python
"""Freeze the RefCOCO-val discovery manifest for the progressive/oracle campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814"
SOURCE_MANIFEST = REPO_ROOT / "data/manifests/final_evaluation/refcoco_val_unique.jsonl"
OUTPUT_MANIFEST = CAMPAIGN_ROOT / "manifests/refcoco_val_discovery1000.jsonl"
LOCK_PATH = CAMPAIGN_ROOT / "manifests/discovery_manifest_lock.json"
SAMPLING_NAMESPACE = "progressive-oracle-refcoco-val-20260814-v1"
SAMPLING_RULE = 'SHA256("progressive-oracle-refcoco-val-20260814-v1\\0" + sample_id)'
EXPECTED_SOURCE_ROWS = 8811
EXPECTED_SELECTED_ROWS = 1000


class CampaignPreparationError(RuntimeError):
    """Raised when the discovery manifest cannot be frozen safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CampaignPreparationError(f"Manifest does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise CampaignPreparationError(f"Blank JSONL line at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CampaignPreparationError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise CampaignPreparationError(f"Row {line_number} is not a JSON object")
            rows.append(row)
    return rows


def validate_source_rows(
    rows: list[dict[str, Any]],
    *,
    repo_root: Path,
    expected_rows: int,
) -> list[str]:
    if len(rows) != expected_rows:
        raise CampaignPreparationError(f"Expected {expected_rows} source rows, found {len(rows)}")
    sample_ids: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise CampaignPreparationError(f"Source row {index} has no sample_id")
        if sample_id in seen:
            raise CampaignPreparationError(f"Duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        sample_ids.append(sample_id)
        if str(row.get("dataset") or "").lower() != "refcoco":
            raise CampaignPreparationError(f"Non-RefCOCO source row {sample_id}")
        if str(row.get("split") or "").lower() != "val":
            raise CampaignPreparationError(f"Non-validation source row {sample_id}")
        image_path = Path(str(row.get("image_path") or ""))
        if not image_path.as_posix():
            raise CampaignPreparationError(f"Source row {sample_id} has no image_path")
        resolved = image_path if image_path.is_absolute() else repo_root / image_path
        if not resolved.is_file():
            raise CampaignPreparationError(f"Missing image for {sample_id}: {resolved}")
    return sample_ids


def selection_digest(sample_id: str) -> str:
    payload = SAMPLING_NAMESPACE.encode("utf-8") + b"\0" + sample_id.encode("utf-8")
    return sha256_bytes(payload)


def select_discovery_rows(
    rows: list[dict[str, Any]],
    *,
    selected_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if selected_rows <= 0 or selected_rows > len(rows):
        raise CampaignPreparationError(
            f"selected_rows must be in [1, {len(rows)}], got {selected_rows}"
        )
    ranked = sorted(
        (
            {"sample_id": str(row["sample_id"]), "selection_digest": selection_digest(str(row["sample_id"]))}
            for row in rows
        ),
        key=lambda item: (item["selection_digest"], item["sample_id"].encode("utf-8")),
    )
    selected_ranked = ranked[:selected_rows]
    selected_ids = {item["sample_id"] for item in selected_ranked}
    selected_in_source_order = [row for row in rows if str(row["sample_id"]) in selected_ids]
    if len(selected_in_source_order) != selected_rows:
        raise CampaignPreparationError("Selection did not produce the requested number of unique rows")
    return selected_in_source_order, selected_ranked


def canonical_jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8") for row in rows
    )


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def git_output(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()


def prepare_campaign(
    *,
    repo_root: Path = REPO_ROOT,
    source_manifest: Path = SOURCE_MANIFEST,
    output_manifest: Path = OUTPUT_MANIFEST,
    lock_path: Path = LOCK_PATH,
    expected_source_rows: int = EXPECTED_SOURCE_ROWS,
    selected_rows: int = EXPECTED_SELECTED_ROWS,
) -> dict[str, Any]:
    rows = read_jsonl_strict(source_manifest)
    source_ids = validate_source_rows(rows, repo_root=repo_root, expected_rows=expected_source_rows)
    selected, selected_ranked = select_discovery_rows(rows, selected_rows=selected_rows)
    output_bytes = canonical_jsonl_bytes(selected)
    selected_ids_source_order = [str(row["sample_id"]) for row in selected]
    selected_id_bytes = ("\n".join(selected_ids_source_order) + "\n").encode("utf-8")
    ranked_bytes = canonical_jsonl_bytes(selected_ranked)
    generator_path = Path(__file__).resolve()
    try:
        generator_display_path = str(generator_path.relative_to(repo_root))
    except ValueError:
        generator_display_path = str(generator_path)
    lock = {
        "created_at_utc": utc_now(),
        "expected_source_rows": expected_source_rows,
        "generation_git_commit": git_output(repo_root, "rev-parse", "HEAD"),
        "generation_git_dirty": bool(git_output(repo_root, "status", "--porcelain", "--untracked-files=no")),
        "generator_path": generator_display_path,
        "generator_sha256": sha256_file(generator_path),
        "output_manifest": str(output_manifest.relative_to(repo_root)),
        "output_manifest_sha256": sha256_bytes(output_bytes),
        "sampling_namespace": SAMPLING_NAMESPACE,
        "sampling_rule": SAMPLING_RULE,
        "selected_id_count": len(selected_ids_source_order),
        "selected_ids_sha256": sha256_bytes(selected_id_bytes),
        "selected_ranked_id_digest_sha256": sha256_bytes(ranked_bytes),
        "selected_rows": selected_rows,
        "selection_output_order": "original_source_manifest_order",
        "source_id_count": len(source_ids),
        "source_ids_sha256": sha256_bytes(("\n".join(source_ids) + "\n").encode("utf-8")),
        "source_manifest": str(source_manifest.relative_to(repo_root)),
        "source_manifest_sha256": sha256_file(source_manifest),
        "status": "frozen",
    }

    if output_manifest.exists() or lock_path.exists():
        if not output_manifest.is_file() or not lock_path.is_file():
            raise CampaignPreparationError("Discovery manifest and lock must either both exist or both be absent")
        if sha256_file(output_manifest) != lock["output_manifest_sha256"]:
            raise CampaignPreparationError("Existing discovery manifest differs from deterministic selection")
        existing_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        invariant_keys = {
            "expected_source_rows",
            "generator_sha256",
            "output_manifest",
            "output_manifest_sha256",
            "sampling_namespace",
            "sampling_rule",
            "selected_id_count",
            "selected_ids_sha256",
            "selected_ranked_id_digest_sha256",
            "selected_rows",
            "selection_output_order",
            "source_id_count",
            "source_ids_sha256",
            "source_manifest",
            "source_manifest_sha256",
            "status",
        }
        mismatches = {
            key: {"existing": existing_lock.get(key), "computed": lock.get(key)}
            for key in invariant_keys
            if existing_lock.get(key) != lock.get(key)
        }
        if mismatches:
            raise CampaignPreparationError(f"Existing discovery lock differs: {mismatches}")
        return existing_lock

    atomic_write_bytes(output_manifest, output_bytes)
    atomic_write_bytes(lock_path, canonical_json_bytes(lock))
    return lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--output-manifest", type=Path, default=OUTPUT_MANIFEST)
    parser.add_argument("--lock-path", type=Path, default=LOCK_PATH)
    parser.add_argument("--expected-source-rows", type=int, default=EXPECTED_SOURCE_ROWS)
    parser.add_argument("--selected-rows", type=int, default=EXPECTED_SELECTED_ROWS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        lock = prepare_campaign(
            repo_root=args.repo_root.resolve(),
            source_manifest=args.source_manifest.resolve(),
            output_manifest=args.output_manifest.resolve(),
            lock_path=args.lock_path.resolve(),
            expected_source_rows=args.expected_source_rows,
            selected_rows=args.selected_rows,
        )
    except (CampaignPreparationError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
