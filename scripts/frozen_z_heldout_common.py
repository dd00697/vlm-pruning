"""Frozen constants and fail-closed helpers for the Z_M448_0 held-out campaign."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "experiments/frozen_z_m448_0_refcoco_testab_20260820"
PYTHON = Path("/workspace/.followup_venv/bin/python")
MODEL = "llava-hf/llava-1.5-7b-hf"
MODEL_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
MODEL_CONFIG_SHA256 = "0bde54495c54bcc346064a0d314b010d1c4d3ca7f7e583b6f711949a35352c0f"
MODEL_INDEX_SHA256 = "73af906b5ac12aea6eddf104d482d2a50bca30b20d61ee9f6c5269d3427a6991"
MODEL_SHARD_SHA256 = {
    "model-00001-of-00003.safetensors": "c11dbf016ee7d35ee130c19b67e20eb04873996006f424b7bcbd453b6517ee66",
    "model-00002-of-00003.safetensors": "46df6c6e5fad297fe7fbca4963dcf180cb1d0528cabc884d1f603311fed01328",
    "model-00003-of-00003.safetensors": "4f06177c37ca13944e73b57dd6d051767b950dffc12400c700dfa2509c24c697",
}
MODEL_SNAPSHOT = (
    Path("/workspace/cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots")
    / MODEL_REVISION
)
FROZEN_DECISION = (
    REPO / "experiments/final_val_efficiency_baselines_20260818/analysis/FROZEN_FINAL_CANDIDATE.json"
)
FROZEN_DECISION_SHA256 = "e8e0186d998eb3ddd94da0a9fac38f760cae7ba3351e633ae9d136ff65963b0a"
ACCEPTED_Z_CONFIG = (
    REPO / "experiments/recent_methods_full_refcoco_val_20260817/runs/v5_z_m448_0_attempt1/config.json"
)
ACCEPTED_Z_CONFIG_SHA256 = "5a503b6cd6740400d167b903bcbf12d85e163fc440f7a04781c5612ecc474adb"
PROMPT = (
    'Identify the target described by this referring expression: "{query}".\n'
    "Return exactly one bounding box in normalized 0-1 image coordinates as "
    "[x_min, y_min, x_max, y_max]. Do not include any other text."
)
Z_SCHEDULE = "4:448:query_topk_merge,12:48:query_topk,20:0:none"
Z_TRACE = [576] * 4 + [448] * 8 + [48] * 8 + [0] * 12
FULL_TRACE = [576] * 32
MANIFESTS = {
    "testA": {
        "path": REPO / "data/manifests/final_evaluation/refcoco_testA_unique.jsonl",
        "rows": 5657,
        "sha256": "01332dfc67c59c9077f14325dc11cb301b9a01a55eccdcc815f92f3fe587fe23",
    },
    "testB": {
        "path": REPO / "data/manifests/final_evaluation/refcoco_testB_unique.jsonl",
        "rows": 5095,
        "sha256": "0f0fd8ff86faf8ac48446c27e71c350b14aed2a8d86d6a59b4f7599def2d40c9",
    },
}
HISTORICAL_FULL = {
    "testA": REPO / "experiments/final_evaluation/runs/official/final_refcoco_testA_full_b576_seed0",
    "testB": REPO / "experiments/final_evaluation/runs/official/final_refcoco_testB_full_b576_seed0",
}
THRESHOLDS = {
    "minimum_valid_parsed_box_rate": 0.995,
    "maximum_accuracy_loss_absolute": 0.03,
    "maximum_miou_loss_absolute": 0.015,
}
BOOTSTRAP = {
    "resamples": 10000,
    "seed": 20260820,
    "confidence": 0.95,
    "unit": "paired_manifest_row_within_split",
    "pool_splits": False,
}
SOURCE_PATHS = [
    "scripts/run_progressive_oracle_refcoco.py",
    "scripts/validate_progressive_oracle_run.py",
    "scripts/run_baseline.py",
    "scripts/frozen_z_heldout_common.py",
    "scripts/init_frozen_z_heldout_campaign.py",
    "scripts/audit_frozen_z_full_controls.py",
    "scripts/run_frozen_z_heldout_campaign.py",
    "scripts/analyse_frozen_z_heldout.py",
    "vlm_pruning/artifacts.py",
    "vlm_pruning/features/query_tokens.py",
    "vlm_pruning/llava_hooks/mid_layer_pruning.py",
    "vlm_pruning/llava_hooks/progressive_pruning.py",
    "vlm_pruning/llava_hooks/runtime_repairs.py",
]


class HeldoutError(RuntimeError):
    """A physical or registered held-out contract was violated."""


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HeldoutError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise HeldoutError(f"Blank JSONL row {number}: {path}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise HeldoutError(f"Non-object JSONL row {number}: {path}")
            rows.append(row)
    return rows


def verify_manifest(split: str) -> list[dict[str, Any]]:
    spec = MANIFESTS[split]
    path = Path(spec["path"])
    if sha256(path) != spec["sha256"]:
        raise HeldoutError(f"{split} manifest SHA-256 differs")
    rows = read_jsonl(path)
    if len(rows) != spec["rows"]:
        raise HeldoutError(f"{split} manifest row count differs")
    ids = [str(row.get("sample_id") or "") for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(rows):
        raise HeldoutError(f"{split} IDs are not unique and non-empty")
    for index, row in enumerate(rows):
        if str(row.get("dataset") or "").lower() != "refcoco" or str(row.get("split") or "").lower() != split.lower():
            raise HeldoutError(f"{split} row {index} has the wrong dataset/split")
        if not Path(str(row["image_path"])).is_file():
            raise HeldoutError(f"{split} row {index} image is missing")
        box = row.get("gt_box")
        width, height = int(row["width"]), int(row["height"])
        if not isinstance(box, list) or len(box) != 4 or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise HeldoutError(f"{split} row {index} has an invalid box")
    return rows
