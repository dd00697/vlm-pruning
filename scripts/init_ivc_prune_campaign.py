#!/usr/bin/env python3
"""Freeze the IVC-Prune replacement campaign before any GPU execution."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from vlm_pruning.artifacts import sha256_file
from vlm_pruning.ivc_campaign import (
    FIXED32_ORDER_PLAN,
    IMAGE_TOKEN_START_INDEX,
    IVC_PRUNE_LAYER_INDEX,
    IVC_TRANSFORMERS_VERSION,
    IVC_UPSTREAM_COMMIT,
    MAX_NEW_TOKENS,
    METHODS,
    MODEL_ID,
    MODEL_REVISION,
    NATURAL_ORDER_PLAN,
    ORIGINAL_VISUAL_TOKENS,
    REGISTERED_PROMPT,
    IVCCampaignError,
)

CAMPAIGN = REPO_ROOT / "experiments/ivc_prune_refcoco_val_20260825"
QUALITY_MANIFEST = REPO_ROOT / "data/manifests/final_evaluation/refcoco_val_unique.jsonl"
TIMING_MANIFEST = (
    REPO_ROOT
    / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
)
QUALITY_SHA256 = "f26d7b954189e86f55b50f9506bb670d3a0cf847d3152779af6c08dbf094e3d7"
TIMING_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, default=CAMPAIGN / "SOURCE_LOCK.json")
    parser.add_argument("--model-snapshot", type=Path, required=True)
    return parser.parse_args()


def initialize(args: argparse.Namespace) -> int:
    checks = (
        (QUALITY_MANIFEST, QUALITY_SHA256, 8811),
        (TIMING_MANIFEST, TIMING_SHA256, 1000),
    )
    for path, digest, rows in checks:
        if not path.is_file() or sha256_file(path) != digest or count_jsonl(path) != rows:
            raise IVCCampaignError(f"Frozen manifest differs: {path}")
    source_lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
    if (
        source_lock.get("status") != "SOURCE_LOCKED"
        or source_lock.get("upstream_commit") != IVC_UPSTREAM_COMMIT
    ):
        raise IVCCampaignError("Source lock is absent or does not match the frozen IVC commit")
    model = args.model_snapshot.resolve()
    if not (model / "config.json").is_file():
        raise IVCCampaignError(f"Model snapshot is incomplete: {model}")
    if MODEL_REVISION not in model.parts and MODEL_REVISION not in str(model):
        raise IVCCampaignError("Model snapshot path does not contain the registered HF revision")
    protocol = {
        "schema_version": 1,
        "frozen_at_utc": utc_now(),
        "status": "PROTOCOL_FROZEN_NO_ACCEPTED_GPU_RUNS_YET",
        "scientific_role": (
            "post-hoc external validation reference replacing FEATHER; it cannot change the "
            "frozen Z_M448_0 candidate or the completed held-out decision"
        ),
        "methods": list(METHODS),
        "paired_control": "IVC_STACK_FULL",
        "external_method": "IVC_PRUNE",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "snapshot": str(model)},
        "source": {
            "ivc_commit": IVC_UPSTREAM_COMMIT,
            "transformers_version": IVC_TRANSFORMERS_VERSION,
            "source_lock_sha256": sha256_file(args.source_lock),
        },
        "ivc_configuration": {
            "ivcp_k_zero_based": IVC_PRUNE_LAYER_INDEX,
            "selection_signal_layer_zero_based": IVC_PRUNE_LAYER_INDEX - 1,
            "first_pruned_decoder_layer_zero_based": IVC_PRUNE_LAYER_INDEX,
            "image_token_start_index": IMAGE_TOKEN_START_INDEX,
            "image_token_length": ORIGINAL_VISUAL_TOKENS,
            "retained_budget": "adaptive and recorded per example",
        },
        "quality": {
            "manifest": str(QUALITY_MANIFEST.relative_to(REPO_ROOT)),
            "manifest_sha256": QUALITY_SHA256,
            "rows": 8811,
            "metrics": ["parse_rate", "Acc@0.5", "mIoU"],
            "paired_bootstrap_resamples": 10000,
            "paired_bootstrap_seed": 20260825,
            "held_out_splits_forbidden": True,
        },
        "timing": {
            "manifest": str(TIMING_MANIFEST.relative_to(REPO_ROOT)),
            "manifest_sha256": TIMING_SHA256,
            "rows_per_cell": 1000,
            "warmups_per_cell": 20,
            "natural_repetitions": 5,
            "natural_order": {str(k): list(v) for k, v in NATURAL_ORDER_PLAN.items()},
            "fixed32_repetitions_if_triggered": 3,
            "fixed32_order": {str(k): list(v) for k, v in FIXED32_ORDER_PLAN.items()},
            "fixed32_trigger": (
                "run only if natural timing fails to show a stable direction or component totals "
                "do not reconcile with synchronized end-to-end latency"
            ),
            "batch_size": 1,
            "precision": "fp16_native",
            "decoding": "greedy",
            "max_new_tokens": MAX_NEW_TOKENS,
            "selector_overhead_included": True,
        },
        "prompt": REGISTERED_PROMPT,
        "comparison_rule": (
            "quality may be shown descriptively beside canonical rows with stack labels; speed and "
            "relative quality claims for IVC must use IVC_STACK_FULL as the paired reference"
        ),
        "prohibitions": [
            "no IVC hyperparameter tuning",
            "no IVC testA/testB execution",
            "no candidate reselection",
            "no deletion or modification of FEATHER evidence",
            "no cross-stack paired significance or speedup claim",
        ],
    }
    protocol_path = CAMPAIGN / "PROTOCOL.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        comparable = dict(protocol)
        comparable["frozen_at_utc"] = existing.get("frozen_at_utc")
        if existing != comparable:
            raise IVCCampaignError("Existing frozen protocol differs; refusing to overwrite it")
    else:
        atomic_json(protocol_path, protocol)
    print(json.dumps({"status": "frozen", "protocol": str(protocol_path)}, indent=2))
    return 0


def main() -> int:
    try:
        return initialize(parse_args())
    except (IVCCampaignError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
