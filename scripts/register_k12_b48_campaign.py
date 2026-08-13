#!/usr/bin/env python
"""Register the outcome-blind K=12/B=48 final campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_pruning.final_campaign import CAMPAIGN_ROOT, MODEL_REVISION, protocol_payload, validate_protocol


MODEL_SNAPSHOT = (
    Path("/workspace/cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots")
    / MODEL_REVISION
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def checkpoint_files(*, hash_weights: bool) -> list[dict[str, Any]]:
    if not MODEL_SNAPSHOT.is_dir():
        raise SystemExit(f"Pinned checkpoint snapshot is missing: {MODEL_SNAPSHOT}")
    rows: list[dict[str, Any]] = []
    for path in sorted(MODEL_SNAPSHOT.iterdir()):
        if not path.is_file():
            continue
        is_weight = path.suffix == ".safetensors"
        resolved = path.resolve()
        blob_name = resolved.name if resolved.parent.name == "blobs" else None
        digest = sha256(path) if hash_weights or not is_weight else blob_name
        digest_source = "computed_sha256" if hash_weights or not is_weight else "huggingface_lfs_sha256_blob_name"
        rows.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "sha256_source": digest_source,
                "resolved_blob": str(resolved),
            }
        )
    required = {"config.json", "model.safetensors.index.json"}
    missing = sorted(required - {row["name"] for row in rows})
    if missing:
        raise SystemExit(f"Checkpoint snapshot is incomplete: {missing}")
    return rows


def run_registry(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in payload["datasets"]:
        for method in payload["mandatory_core_methods"]:
            rows.append(
                {
                    "run_id": f"{dataset['key']}__{method}",
                    "dataset": dataset["key"],
                    "method": method,
                    "status": "pending_manifest",
                    "mandatory": True,
                    "target_visual_tokens": 576 if method == "full" else 48,
                    "note": "full is an unpruned reference" if method == "full" else "exact final B=48 required",
                }
            )
    for method in payload["methods"]:
        for dataset in payload["datasets"]:
            status = "pending_admission_gate" if method["status"] == "gate_pending" else method["status"]
            rows.append(
                {
                    "run_id": f"{dataset['key']}__official_{method['key']}",
                    "dataset": dataset["key"],
                    "method": method["key"],
                    "status": status,
                    "mandatory": False,
                    "target_visual_tokens": 48,
                    "note": method.get("reason") or method.get("exact_b48_plan"),
                }
            )
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hash-weights",
        action="store_true",
        help="Recompute all large checkpoint SHA-256 values instead of using immutable HF LFS blob names.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Development-only registration. Full inference must use a later clean experiment epoch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = protocol_payload()
    issues = validate_protocol(payload)
    if issues:
        raise SystemExit("Invalid protocol: " + "; ".join(issues))
    status = git("status", "--porcelain")
    if status and not args.allow_dirty:
        raise SystemExit("Refusing to register from a dirty worktree; commit protocol code first.")
    revision = git("rev-parse", "HEAD")
    payload.update(
        {
            "registered_at_utc": utc_now(),
            "source_git_commit": revision,
            "source_git_clean": not bool(status),
            "checkpoint_snapshot": str(MODEL_SNAPSHOT),
            "checkpoint_files": checkpoint_files(hash_weights=args.hash_weights),
            "protocol_validation": {"status": "pass", "issues": []},
        }
    )
    root = ROOT / CAMPAIGN_ROOT
    write_json(root / "protocol_lock.json", payload)
    write_json(
        root / "run_registry.json",
        {
            "campaign_id": payload["campaign_id"],
            "registered_at_utc": payload["registered_at_utc"],
            "source_git_commit": revision,
            "runs": run_registry(payload),
        },
    )
    write_json(
        root / "method_admission.json",
        {
            "campaign_id": payload["campaign_id"],
            "policy": "official repository plus parity plus exact per-example final B=48",
            "methods": payload["methods"],
        },
    )
    print(json.dumps({"status": "registered", "root": str(root), "runs": len(run_registry(payload))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
