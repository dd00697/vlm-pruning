#!/usr/bin/env python3
"""Fail-closed terminal audit for the frozen Z_M448_0 held-out campaign.

The audit has two phases.  ``collect_requirements`` is read-only and raises on
the first missing or contradictory item.  Only after it returns successfully
does ``finalize_terminal_state`` atomically publish terminal state.  This is
deliberately independent of the campaign controller and analysis writers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
CAMPAIGN = REPO / "experiments/frozen_z_m448_0_refcoco_testab_20260820"
DISSERTATION = Path("/workspace/Dissertation_Draft")
DERIVED = DISSERTATION / "analysis/derived/frozen_z_m448_0_refcoco_testab"
SOURCE_PDF = DISSERTATION / "Dissertation_Template.pdf"
USER_PDF = Path("/workspace/dissertation_final.pdf")
PROTOCOL = CAMPAIGN / "FROZEN_PROTOCOL.json"
PRELOCK = CAMPAIGN / "PRE_INFERENCE_LOCK.json"
REGISTRY = CAMPAIGN / "EXPERIMENT_REGISTRY.json"
CONTROLLER = CAMPAIGN / "CONTROLLER_STATE.json"
REPORT = CAMPAIGN / "RESEARCH_REPORT.md"
FINAL = CAMPAIGN / "FINAL_COMPLETION_REPORT.json"
FINAL_TEST_LOG = CAMPAIGN / "analysis/FINAL_TEST_SUITE.log"
FINAL_TEST_JSON = CAMPAIGN / "analysis/FINAL_TEST_SUITE.json"
TERMINAL_STATE = "GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED"
INFERENCE_COMMIT = "eca952596765199b95e602bb59a6cc80582f0924"

LOCKED_HASHES = {
    "FROZEN_PROTOCOL.json": "d683b3a559925617517e5f617bb70c243d8e02e1b1d0eaa073bf5bd40420019e",
    "ANALYSIS_SCHEMA.json": "90fa3b4c11a1d85d1ba3fca5196d8e6b081a5ec9d507e0e44b7bbcda43173084",
    "DECISION_THRESHOLDS.json": "3faba2342a64c43c6f1a42da2e77c49b7cc3650760591f770ee4e5459199b6a3",
    "COMMANDS.json": "0aec088fa2a87eeb2ad572c6333d88f6b406539172d4dade96b8118b837cd0f4",
    "SOURCE_ENVIRONMENT_LOCK.json": "23095d6928498571bf1c636ee03187c48321d480bf03f54c37845ec36eb65b9f",
}
PRELOCK_SHA256 = "c3c172683b34b8ad48f9d0a4b5df473507ac3bce3151d1e07864264b18d1ae1e"
FULL_AUDIT_SHA256 = "7ff52dabe9817bceefc01115ed4fe8a870535e6051520e554764a7ab1459753e"
SMOKE_LEDGER_SHA256 = "e11a57e26be8a1f98817a09e3b2bd88166bf0798aae89257eabd3c614e31ff40"
PRE_CANDIDATE_LEDGER_ENTRIES = 4
PRE_CANDIDATE_LEDGER_PREFIX_SHA256 = "7b9d05cbe7387a2a2d32ba6a1230e3d41d49d3ea0720b44a694d47f8f83ebba6"
PRE_CANDIDATE_LEDGER_HEAD_SHA256 = "5b3bb588fa7e2afb5f9c7a11a3dff9f481b769b2e68b56bed2558332bf50cc06"
GOAL_SHA256 = "9f9ab62c0a224ea289e18a659f74fd0710e16e4dd0499034beaf8df2ad901f35"
AGENTS_SHA256 = "2b33e9a8f9f474e0e29f912ccd0fe82621344865a5e1a5460fa2400aed4be5c2"
PROTECTED_HASHES = {
    "PRO_REASONING_RESEARCH_HANDOFF.md": "c64abee8234d7cc17ce92b29f019f58ce963c2ff67e7b659e827db443a0b3aaa",
    "legacy_goal2.md": "8b6bbb28e08182e94f490f72a30f2972e7f9d1be6564ae733b5be32f50010023",
}
EXPECTED_CELLS = ("Full_testA", "Z_M448_0_testA", "Full_testB", "Z_M448_0_testB")
EXPECTED_QUALITY_ORDER = ("Full_testA", "Full_testB", "Z_M448_0_testA", "Z_M448_0_testB")
EXPECTED_QUALITY_DIRS = {
    "Full_testA": "quality/full_testa_attempt1",
    "Z_M448_0_testA": "quality/z_m448_0_testa_attempt1",
    "Full_testB": "quality/full_testb_attempt1",
    "Z_M448_0_testB": "quality/z_m448_0_testb_attempt1",
}
EXPECTED_SMOKE_DIRS = {
    "Full_testA": "smokes/full_testa_smoke2_attempt1",
    "Z_M448_0_testA": "smokes/z_m448_0_testa_smoke2_attempt1",
    "Full_testB": "smokes/full_testb_smoke2_attempt1",
    "Z_M448_0_testB": "smokes/z_m448_0_testb_smoke2_attempt1",
}
OUTCOME_FIELDS = {
    "raw_parsed_box", "predicted_box", "parse_failure_reason", "valid", "invalid",
    "box_iou", "score", "success_iou_0_5", "outcome_embargoed",
}
REQUIRED_ROW_FIELDS = {
    "sample_id", "manifest_index", "dataset", "split", "query", "image_id", "image_path",
    "width", "height", "gt_box", "raw_model_output", "generated_token_count", "method",
    "config_sha256", "source_git_commit", "checkpoint_revision",
    "checkpoint_config_sha256", "checkpoint_tensor_inventory_sha256", "environment_sha256",
    "realized_visual_tokens_by_layer", "candidate_average_visual_tokens", "stage_traces",
    "invariant_results",
}


class TerminalAuditError(RuntimeError):
    """A terminal requirement lacks exact passing evidence."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalAuditError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_time(value: Any, label: str) -> datetime:
    require(isinstance(value, str) and value, f"Missing timestamp: {label}")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TerminalAuditError(f"Malformed timestamp {label}: {value!r}") from exc


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"Missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def json_checksum(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TerminalAuditError(f"Invalid JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"Missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            require(bool(line.strip()), f"Blank JSONL line: {path}:{number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TerminalAuditError(f"Malformed JSONL {path}:{number}: {exc}") from exc
            require(isinstance(value, dict), f"Non-object JSONL line: {path}:{number}")
            rows.append(value)
    return rows


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def close(actual: Any, expected: Any, label: str, tolerance: float = 1e-12) -> None:
    require(math.isfinite(float(actual)), f"Non-finite value: {label}")
    require(math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance),
            f"Value differs for {label}: {actual} != {expected}")


def audit_preregistration() -> tuple[dict[str, Any], dict[str, Any]]:
    require(sha256_file(PRELOCK) == PRELOCK_SHA256, "Pre-inference lock changed")
    lock = read_json(PRELOCK)
    require(lock.get("status") == "locked" and lock.get("candidate_outputs_present") is False,
            "Pre-inference lock is not a clean pre-output lock")
    for name, expected in LOCKED_HASHES.items():
        path = CAMPAIGN / name
        require(sha256_file(path) == expected, f"Frozen preregistration changed: {name}")
        require(lock.get("artifacts", {}).get(name) == expected, f"Prelock does not bind {name}")
    protocol = read_json(PROTOCOL)
    require(protocol.get("source", {}).get("git_commit") == INFERENCE_COMMIT, "Inference epoch differs")
    require(protocol.get("exact_logical_cells") == list(EXPECTED_CELLS), "Logical cell inventory differs")
    require(protocol.get("candidate_attempts") == {"testA": 1, "testB": 1}, "Candidate attempt limit differs")
    require(protocol.get("outcome_embargo") == "seal_both_candidate_splits_before_any_metric_analysis_or_display",
            "Outcome embargo text differs")
    require(protocol.get("methods", {}).get("Full", {}).get("trace") == [576] * 32, "Full trace differs")
    z_trace = [576] * 4 + [448] * 8 + [48] * 8 + [0] * 12
    require(protocol.get("methods", {}).get("Z_M448_0", {}).get("trace") == z_trace, "Z trace differs")
    require(protocol["methods"]["Full"].get("work") == 18432, "Full work differs")
    require(protocol["methods"]["Z_M448_0"].get("work") == 6272, "Z work differs")
    require(protocol["bootstrap"] == {
        "confidence": 0.95, "pool_splits": False, "resamples": 10000,
        "seed": 20260820, "unit": "paired_manifest_row_within_split",
    }, "Bootstrap lock differs")
    require(protocol["thresholds"] == {
        "maximum_accuracy_loss_absolute": 0.03,
        "maximum_miou_loss_absolute": 0.015,
        "minimum_valid_parsed_box_rate": 0.995,
    }, "Decision thresholds differ")
    return protocol, {"status": "pass", "prelock_sha256": PRELOCK_SHA256, "locked_artifacts": LOCKED_HASHES}


def audit_protected_and_source(protocol: dict[str, Any]) -> dict[str, Any]:
    require(sha256_file(REPO / "GOAL.md") == GOAL_SHA256, "Active GOAL.md changed")
    require(sha256_file(REPO / "AGENTS.md") == AGENTS_SHA256, "AGENTS.md changed")
    protected = {}
    for relative, expected in PROTECTED_HASHES.items():
        actual = sha256_file(REPO / relative)
        require(actual == expected, f"Protected evidence changed: {relative}")
        protected[relative] = actual
    source = protocol["source"]
    require(source.get("git_dirty") is False and source.get("git_commit") == INFERENCE_COMMIT,
            "Frozen source lock is malformed")
    for relative, expected in source.get("sha256", {}).items():
        path = REPO / relative
        require(sha256_file(path) == expected, f"Inference source changed: {relative}")
        blob = subprocess.check_output(["git", "show", f"{INFERENCE_COMMIT}:{relative}"], cwd=REPO)
        require(sha256_bytes(blob) == expected, f"Inference commit blob differs: {relative}")
    return {"status": "pass", "inference_commit": INFERENCE_COMMIT,
            "source_files": len(source["sha256"]), "protected_sha256": protected}


def audit_manifests_checkpoint(protocol: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifests: dict[str, list[dict[str, Any]]] = {}
    evidence: dict[str, Any] = {}
    for split in ("testA", "testB"):
        spec = protocol["manifests"][split]
        path = REPO / spec["path"]
        require(sha256_file(path) == spec["sha256"], f"{split} manifest hash differs")
        rows = read_jsonl(path)
        ids = [str(row.get("sample_id") or "") for row in rows]
        require(len(rows) == int(spec["rows"]), f"{split} row count differs")
        require(len(set(ids)) == len(ids) and all(ids), f"{split} IDs are not unique/non-empty")
        require(ids[0] == spec["first_id"] and ids[-1] == spec["last_id"], f"{split} order endpoints differ")
        for index, row in enumerate(rows):
            require(str(row.get("dataset", "")).lower() == "refcoco", f"{split} row {index} dataset differs")
            require(str(row.get("split", "")).lower() == split.lower(), f"{split} row {index} split differs")
            require(split.lower() in ids[index].lower(), f"{split} row {index} ID is not split-qualified")
            image = Path(str(row.get("image_path") or ""))
            require(image.is_file(), f"{split} row {index} image missing")
            width, height = int(row["width"]), int(row["height"])
            box = row.get("gt_box")
            require(isinstance(box, list) and len(box) == 4, f"{split} row {index} GT box malformed")
            require(0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height,
                    f"{split} row {index} GT box invalid")
        manifests[split] = rows
        evidence[split] = {"path": spec["path"], "sha256": spec["sha256"],
                           "rows": len(rows), "unique_ids": len(set(ids))}
    checkpoint = protocol["checkpoint"]
    snapshot = Path(checkpoint["snapshot"])
    require(snapshot.is_dir(), "Checkpoint snapshot missing")
    reference = snapshot.parent.parent / "refs/main"
    require(reference.read_text(encoding="utf-8").strip() == checkpoint["revision"], "Checkpoint ref differs")
    require(sha256_file(snapshot / "config.json") == checkpoint["config_sha256"], "Checkpoint config differs")
    require(sha256_file(snapshot / "model.safetensors.index.json") == checkpoint["index_sha256"],
            "Checkpoint index differs")
    for shard in checkpoint["shards"]:
        path = snapshot / shard["name"]
        require(path.stat().st_size == int(shard["bytes"]), f"Checkpoint shard size differs: {path.name}")
        require(sha256_file(path) == shard["sha256"], f"Checkpoint shard hash differs: {path.name}")
    return manifests, {"status": "pass", "manifests": evidence, "checkpoint_revision": checkpoint["revision"],
                       "checkpoint_shards": len(checkpoint["shards"])}


def audit_inventory_scope(protocol: dict[str, Any]) -> dict[str, Any]:
    quality_root = CAMPAIGN / "quality"
    smoke_root = CAMPAIGN / "smokes"
    quality_dirs = {str(path.relative_to(CAMPAIGN)) for path in quality_root.glob("*") if path.is_dir()}
    smoke_dirs = {str(path.relative_to(CAMPAIGN)) for path in smoke_root.glob("*") if path.is_dir()}
    require(quality_dirs == set(EXPECTED_QUALITY_DIRS.values()), f"Quality directory inventory differs: {quality_dirs}")
    require(smoke_dirs == set(EXPECTED_SMOKE_DIRS.values()), f"Smoke directory inventory differs: {smoke_dirs}")
    allowed_ids = set(EXPECTED_CELLS)
    configs = sorted(CAMPAIGN.glob("quality/*/config.json")) + sorted(CAMPAIGN.glob("smokes/*/config.json"))
    require(len(configs) == 8, f"Expected eight physical configs, found {len(configs)}")
    for path in configs:
        config = read_json(path)
        require(config.get("cell_id") in allowed_ids, f"Unauthorized cell config: {path}")
        require(config.get("dataset/name") == "refcoco", f"Unauthorized dataset: {path}")
        require(config.get("dataset/split") in {"testA", "testB"}, f"Unauthorized split: {path}")
        require(config.get("method/name") == "progressive", f"Unauthorized runner method: {path}")
        require(config.get("pre_llm_selector") == "none", f"Unauthorized selector: {path}")
    forbidden_roots = ("timing", "validation", "transfer", "training", "checkpoints", "official_sources")
    forbidden = [name for name in forbidden_roots if (CAMPAIGN / name).exists()]
    require(not forbidden, f"Forbidden campaign branches exist: {forbidden}")
    campaign_bytes = sum(path.stat().st_size for path in CAMPAIGN.rglob("*") if path.is_file())
    require(campaign_bytes <= int(protocol["storage"]["campaign_cap_bytes"]), "Campaign size cap exceeded")
    free = shutil.disk_usage("/workspace").free
    require(free >= int(protocol["storage"]["minimum_runtime_free_bytes"]), "Workspace reserve violated")
    return {"status": "pass", "quality_dirs": sorted(quality_dirs), "smoke_dirs": sorted(smoke_dirs),
            "configs": len(configs), "campaign_bytes": campaign_bytes, "workspace_free_bytes": free}


def audit_full_decisions(registry: dict[str, Any]) -> dict[str, Any]:
    path = CAMPAIGN / "FULL_REUSE_AUDIT.json"
    require(sha256_file(path) == FULL_AUDIT_SHA256, "Full reuse audit changed")
    audit = read_json(path)
    require(audit.get("status") == "complete", "Full reuse audit incomplete")
    audit_time = parse_time(audit.get("audited_at_utc"), "Full audit")
    evidence = {}
    for split in ("testA", "testB"):
        record = audit.get("splits", {}).get(split, {})
        require(record.get("decision") == "rerun" and record.get("passed") is False,
                f"Historical Full {split} was not correctly rejected")
        failed = set(record.get("failed_gates") or [])
        required_failed = {"atomic_completed_marker", "checkpoint_revision_recorded",
                           "full_manifest_sha256_recorded", "row_config_sha256_bound",
                           "source_clean", "source_file_hashes_recorded"}
        require(required_failed <= failed, f"Historical Full {split} rejection evidence differs")
        for item in record.get("immutable_files", {}).values():
            historical = Path(record["historical_path"]) / item.get("name", "") if item.get("name") else None
            if historical is not None:
                require(sha256_file(historical) == item["sha256"], f"Historical Full changed: {historical}")
        # The current audit stores immutable files keyed by filename.
        for name, item in record.get("immutable_files", {}).items():
            historical = Path(record["historical_path"]) / name
            require(historical.stat().st_size == int(item["bytes"]), f"Historical Full bytes changed: {historical}")
            require(sha256_file(historical) == item["sha256"], f"Historical Full hash changed: {historical}")
        cell = registry.get("cells", {}).get(f"Full_{split}", {})
        require(cell.get("status") == "complete_rerun", f"Full {split} is not an accepted rerun")
        require(str(cell.get("run_dir", "")).startswith(str(CAMPAIGN.relative_to(REPO))),
                f"Full {split} does not point into the new campaign")
        z_config = CAMPAIGN / EXPECTED_QUALITY_DIRS[f"Z_M448_0_{split}"] / "config.json"
        require(audit_time <= parse_time(read_json(z_config)["created_at_utc"], f"Z {split} config"),
                f"Full decision occurred after Z {split} began")
        evidence[split] = {"decision": "rerun", "failed_gates": sorted(failed)}
    return {"status": "pass", "audit_sha256": FULL_AUDIT_SHA256, "splits": evidence}


def normalized_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    count = len(rows)
    require(count > 0, "Cannot score an empty cell")
    return {
        "samples": count,
        "parsed_box_rate": sum(bool(row["valid"]) for row in rows) / count,
        "acc_at_0_5": sum(bool(row["success_iou_0_5"]) for row in rows) / count,
        "mean_iou": sum(float(row["box_iou"]) for row in rows) / count,
        "average_visual_tokens": sum(float(row["candidate_average_visual_tokens"]) for row in rows) / count,
        "token_layer_work": sum(sum(int(value) for value in row["realized_visual_tokens_by_layer"])
                                for row in rows) / count,
    }


def artifact_ledger_cells() -> dict[str, Any]:
    ledger = read_json(CAMPAIGN / "QUALITY_ARTIFACT_LEDGER.json")
    require(ledger.get("schema_version") == 1 and ledger.get("status") == "pass",
            "Quality artifact ledger incomplete")
    require(ledger.get("exact_cells") == list(EXPECTED_CELLS), "Artifact ledger exact-cell order differs")
    require(ledger.get("quality_run_ledger_sha256") == sha256_file(CAMPAIGN / "QUALITY_RUN_LEDGER.json"),
            "Artifact ledger quality-run binding differs")
    require(ledger.get("outcome_embargo_release_sha256") == sha256_file(CAMPAIGN / "OUTCOME_EMBARGO_RELEASE.json"),
            "Artifact ledger embargo-release binding differs")
    cells = ledger.get("cells")
    require(isinstance(cells, dict) and set(cells) == set(EXPECTED_CELLS), "Artifact ledger cells differ")
    return cells


def verify_ledger_artifacts(cell_id: str, run_dir: Path, record: Mapping[str, Any], required: set[str]) -> None:
    require(set(record) == {"path", "files"}, f"Ledger cell schema differs: {cell_id}")
    require(record.get("path") == str(run_dir.relative_to(REPO)), f"Ledger run path differs: {cell_id}")
    files = record.get("files")
    require(isinstance(files, dict), f"Ledger files missing: {cell_id}")
    prediction_names = {
        name for name in ("predictions.jsonl", "predictions.raw.jsonl", "predictions.scored.jsonl")
        if (run_dir / name).is_file()
    }
    environment_names = {path.name for path in run_dir.glob("environment*.json")}
    optional_names = {name for name in ("SEALED_RAW.json", "integrity_raw.json") if (run_dir / name).is_file()}
    expected_names = {
        "config.json", "metrics.json", "integrity_report.json", "COMPLETED.json",
        *prediction_names, *environment_names, *optional_names,
    }
    require(required <= expected_names, f"Required artifact inventory is internally incomplete: {cell_id}")
    require(set(files) == expected_names, f"Ledger artifact coverage differs: {cell_id}")
    for relative, expected in files.items():
        require(isinstance(expected, dict) and set(expected) == {"bytes", "sha256"},
                f"Ledger file schema differs: {cell_id}/{relative}")
        path = run_dir / relative
        require(path.is_file() and path.stat().st_size == int(expected["bytes"]),
                f"Artifact ledger bytes differ: {cell_id}/{relative}")
        require(sha256_file(path) == expected["sha256"], f"Artifact ledger hash differs: {cell_id}/{relative}")


def audit_cell(
    cell_id: str,
    manifest: list[dict[str, Any]],
    protocol: dict[str, Any],
    ledger_record: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from scripts.validate_progressive_oracle_run import parse_box, validate_run
    from scripts.run_baseline import box_iou

    method, split = ("Z_M448_0", cell_id.rsplit("_", 1)[1]) if cell_id.startswith("Z_M448_0") else ("Full", cell_id.rsplit("_", 1)[1])
    run_dir = CAMPAIGN / EXPECTED_QUALITY_DIRS[cell_id]
    config = read_json(run_dir / "config.json")
    expected_trace = protocol["methods"][method]["trace"]
    expected_schedule = protocol["methods"][method]["schedule"]
    expected_rows = len(manifest)
    checks = {
        "cell_id": cell_id, "dataset/name": "refcoco", "dataset/split": split,
        "dataset/manifest_checksum": protocol["manifests"][split]["sha256"],
        "dataset/num_samples": expected_rows, "model/name": protocol["checkpoint"]["name"],
        "model/revision": protocol["checkpoint"]["revision"], "model/precision": "bf16",
        "model/decoding": "deterministic_greedy", "model/max_new_tokens": 48,
        "prompt/template": protocol["inference"]["prompt"], "position_mode": "gap_preserving",
        "runtime_implementation": "original", "seed": 0, "pre_llm_selector": "none",
        "schedule_string": expected_schedule, "limit": None,
    }
    for key, expected in checks.items():
        require(config.get(key) == expected, f"Config differs {cell_id}/{key}")
    require(config.get("source/git_commit") == INFERENCE_COMMIT and config.get("source/git_dirty") is False,
            f"Source epoch differs: {cell_id}")
    outcome_embargo = method == "Z_M448_0"
    require(bool(config.get("outcome_embargo")) is outcome_embargo, f"Embargo config differs: {cell_id}")
    validation = validate_run(run_dir, REPO / protocol["manifests"][split]["path"], expected_rows=expected_rows)
    require(validation.get("status") == "pass", f"Independent validator failed: {cell_id}")
    environment_paths = sorted(run_dir.glob("environment*.json"))
    require(environment_paths and (run_dir / "environment.json").is_file(), f"Environment missing: {cell_id}")
    environment_hashes = {sha256_file(path) for path in environment_paths}
    raw_path = run_dir / ("predictions.raw.jsonl" if outcome_embargo else "predictions.jsonl")
    raw_rows = read_jsonl(raw_path)
    scored_rows = read_jsonl(run_dir / "predictions.scored.jsonl") if outcome_embargo else raw_rows
    require(len(raw_rows) == len(scored_rows) == expected_rows, f"Physical row count differs: {cell_id}")
    config_hash = sha256_file(run_dir / "config.json")
    recomputed: list[dict[str, Any]] = []
    for index, (source, raw, scored) in enumerate(zip(manifest, raw_rows, scored_rows, strict=True)):
        require(REQUIRED_ROW_FIELDS <= set(raw), f"Required row fields missing: {cell_id}/{index}")
        require(raw.get("sample_id") == source["sample_id"] and raw.get("manifest_index") == index,
                f"Manifest identity differs: {cell_id}/{index}")
        for key in ("dataset", "split", "query", "image_id", "image_path", "width", "height", "gt_box"):
            require(raw.get(key) == source.get(key), f"Manifest field differs: {cell_id}/{index}/{key}")
        require(raw.get("method") == method, f"Method differs: {cell_id}/{index}")
        require(raw.get("config_sha256") == config_hash, f"Config hash differs: {cell_id}/{index}")
        require(raw.get("source_git_commit") == INFERENCE_COMMIT, f"Source hash differs: {cell_id}/{index}")
        require(raw.get("checkpoint_revision") == protocol["checkpoint"]["revision"],
                f"Checkpoint revision differs: {cell_id}/{index}")
        require(raw.get("checkpoint_config_sha256") == protocol["checkpoint"]["config_sha256"],
                f"Checkpoint config differs: {cell_id}/{index}")
        require(raw.get("environment_sha256") in environment_hashes, f"Environment epoch differs: {cell_id}/{index}")
        invariants = raw.get("invariant_results")
        require(isinstance(invariants, dict), f"Invariant results missing: {cell_id}/{index}")
        for name in ("text_tokens_preserved", "gap_positions_preserved", "exact_trace_recorded", "physical_segment_cache"):
            require(invariants.get(name) is True, f"Invariant false: {cell_id}/{index}/{name}")
        require(invariants.get("zero_tail_no_score") is outcome_embargo,
                f"Zero-tail invariant differs: {cell_id}/{index}")
        require(raw.get("realized_visual_tokens_by_layer") == expected_trace,
                f"Trace differs: {cell_id}/{index}")
        close(raw.get("candidate_average_visual_tokens"), sum(expected_trace) / 32, f"average {cell_id}/{index}")
        require(sum(int(value) for value in raw["realized_visual_tokens_by_layer"]) == protocol["methods"][method]["work"],
                f"Token-layer work differs: {cell_id}/{index}")
        predicted, raw_box, reason = parse_box(str(raw.get("raw_model_output") or ""), int(source["width"]), int(source["height"]))
        iou = box_iou(predicted, source["gt_box"]) if predicted is not None else 0.0
        if outcome_embargo:
            require(raw.get("outcome_embargoed") is True, f"Raw row not embargoed: {cell_id}/{index}")
            require(all(raw.get(key) is None for key in OUTCOME_FIELDS - {"outcome_embargoed"}),
                    f"Raw outcome leaked before seal: {cell_id}/{index}")
            for key in set(raw) - OUTCOME_FIELDS:
                require(scored.get(key) == raw.get(key), f"Scoring changed raw evidence: {cell_id}/{index}/{key}")
            require(scored.get("outcome_embargoed") is False, f"Scored row remains embargoed: {cell_id}/{index}")
        require(scored.get("parse_failure_reason") == reason, f"Parse reason differs: {cell_id}/{index}")
        require(bool(scored.get("valid")) is (predicted is not None), f"Validity differs: {cell_id}/{index}")
        require(bool(scored.get("invalid")) is (predicted is None), f"Invalidity differs: {cell_id}/{index}")
        require(bool(scored.get("success_iou_0_5")) is bool(iou >= 0.5), f"Correctness differs: {cell_id}/{index}")
        close(scored.get("box_iou"), iou, f"IoU {cell_id}/{index}")
        if raw_box is not None:
            require(scored.get("raw_parsed_box") == raw_box, f"Raw parsed box differs: {cell_id}/{index}")
        if predicted is not None:
            require(scored.get("predicted_box") == predicted, f"Predicted box differs: {cell_id}/{index}")
        recomputed.append(scored)
    required_files = {"config.json", "environment.json", "metrics.json", "integrity_report.json", "COMPLETED.json"}
    if outcome_embargo:
        required_files |= {"predictions.raw.jsonl", "predictions.scored.jsonl", "integrity_raw.json", "SEALED_RAW.json"}
    else:
        required_files.add("predictions.jsonl")
    required_files |= {path.name for path in environment_paths}
    verify_ledger_artifacts(cell_id, run_dir, ledger_record, required_files)
    completion = read_json(run_dir / "COMPLETED.json")
    require(completion.get("status") == "complete" and completion.get("rows") == expected_rows,
            f"Completion sentinel differs: {cell_id}")
    require(completion.get("config_sha256") == config_hash, f"Completion config hash differs: {cell_id}")
    require(completion.get("metrics_sha256") == sha256_file(run_dir / "metrics.json"),
            f"Completion metrics hash differs: {cell_id}")
    require(completion.get("integrity_sha256") == sha256_file(run_dir / "integrity_report.json"),
            f"Completion integrity hash differs: {cell_id}")
    if outcome_embargo:
        require(completion.get("raw_predictions_sha256") == sha256_file(run_dir / "predictions.raw.jsonl"),
                f"Completion raw hash differs: {cell_id}")
        require(completion.get("scored_predictions_sha256") == sha256_file(run_dir / "predictions.scored.jsonl"),
                f"Completion scored hash differs: {cell_id}")
    else:
        require(completion.get("predictions_sha256") == sha256_file(run_dir / "predictions.jsonl"),
                f"Completion predictions hash differs: {cell_id}")
    expected_metrics = normalized_metrics(recomputed)
    stored = read_json(run_dir / "metrics.json")
    if outcome_embargo:
        for key, value in expected_metrics.items():
            close(stored.get(key), value, f"stored metric {cell_id}/{key}")
    else:
        mapping = {
            "samples": "samples/evaluated", "parsed_box_rate": "quality/parsed_box_rate",
            "acc_at_0_5": "quality/acc_at_0_5", "mean_iou": "quality/mean_iou",
            "average_visual_tokens": "tokens/candidate_average_visual_tokens",
        }
        for key, stored_key in mapping.items():
            close(stored.get(stored_key), expected_metrics[key], f"stored metric {cell_id}/{stored_key}")
    return recomputed, {"status": "pass", "run_dir": str(run_dir.relative_to(REPO)), "rows": expected_rows,
                        "config_sha256": config_hash, "environment_epochs": len(environment_paths),
                        "completion_sha256": sha256_file(run_dir / "COMPLETED.json")}


def audit_smokes() -> dict[str, Any]:
    ledger_path = CAMPAIGN / "SMOKE_LEDGER.json"
    require(sha256_file(ledger_path) == SMOKE_LEDGER_SHA256, "Smoke ledger changed")
    ledger = read_json(ledger_path)
    records = ledger.get("records")
    require(ledger.get("status") == "complete" and ledger.get("serial_execution") is True,
            "Smoke ledger is not serial/complete")
    require(isinstance(records, list) and len(records) == 4, "Smoke inventory differs")
    previous: datetime | None = None
    observed = []
    for record in records:
        require(record.get("limit") == 2 and record.get("exit_code") == 0, "Smoke did not pass exactly two rows")
        require(record.get("pids_before") == [] and record.get("pids_after") == [], "Smoke GPU isolation differs")
        start = parse_time(record.get("started_at_utc"), "smoke start")
        finish = parse_time(record.get("finished_at_utc"), "smoke finish")
        require(start <= finish and (previous is None or previous <= start), "Smoke intervals overlap")
        previous = finish
        run_dir = REPO / record["run_dir"]
        require(sha256_file(run_dir / "config.json") == record["config_sha256"], "Smoke config hash differs")
        prediction = run_dir / ("predictions.raw.jsonl" if record["method"] == "Z_M448_0" else "predictions.jsonl")
        require(sha256_file(prediction) == record["predictions_sha256"], "Smoke prediction hash differs")
        observed.append(f"{record['method']}_{record['split']}")
    require(observed == list(EXPECTED_CELLS), f"Smoke order/cells differ: {observed}")
    return {"status": "pass", "ledger_sha256": SMOKE_LEDGER_SHA256, "records": len(records), "serial": True}


def audit_attempts_seriality() -> tuple[dict[str, Any], dict[str, Any]]:
    ledger_path = CAMPAIGN / "QUALITY_RUN_LEDGER.json"
    ledger = read_json(ledger_path)
    require(ledger.get("status") == "raw_sealed_both_splits" and ledger.get("serial_execution") is True,
            "Quality ledger is not raw-sealed/serial")
    records = ledger.get("records")
    require(isinstance(records, list) and len(records) == 4, "Quality ledger must contain exactly four physical cells")
    expected = list(EXPECTED_QUALITY_ORDER)
    observed: list[str] = []
    previous: datetime | None = None
    for record in records:
        cell = f"{record.get('method')}_{record.get('split')}"
        observed.append(cell)
        require(record.get("limit") is None and record.get("exit_code") == 0, f"Quality record malformed: {cell}")
        require(record.get("pids_before") == [] and record.get("pids_after") == [], f"GPU isolation differs: {cell}")
        start = parse_time(record.get("started_at_utc"), f"{cell} start")
        finish = parse_time(record.get("finished_at_utc"), f"{cell} finish")
        require(start <= finish and (previous is None or previous <= start), f"Quality intervals overlap: {cell}")
        previous = finish
        expected_run_dir = str((CAMPAIGN / EXPECTED_QUALITY_DIRS[cell]).relative_to(REPO))
        require(record.get("run_dir") == expected_run_dir, f"Quality path differs: {cell}")
    require(observed == expected, f"Quality execution order differs: {observed}")
    require(not (CAMPAIGN / "QUALITY_RUN_LEDGER.partial.json").exists(), "Partial quality ledger remains")
    require(not list((CAMPAIGN / "quality").glob("**/predictions.partial.jsonl")), "Unresolved physical partial remains")
    require(not list((CAMPAIGN / "quality").glob("**/*attempt[2-9]*")), "Additional quality attempt exists")
    release = read_json(CAMPAIGN / "OUTCOME_EMBARGO_RELEASE.json")
    require(release.get("quality_ledger_sha256") == sha256_file(ledger_path), "Release does not bind quality ledger")
    return ledger, {"status": "pass", "records": observed, "serial_execution": True,
                    "ledger_sha256": sha256_file(ledger_path)}


def audit_embargo(registry: dict[str, Any], quality_ledger: dict[str, Any]) -> dict[str, Any]:
    release_path = CAMPAIGN / "OUTCOME_EMBARGO_RELEASE.json"
    release = read_json(release_path)
    require(release.get("status") == "outcome_embargo_released", "Outcome embargo was not formally released")
    release_time = parse_time(release.get("released_at_utc"), "embargo release")
    seals = {}
    for split in ("testA", "testB"):
        run_dir = CAMPAIGN / EXPECTED_QUALITY_DIRS[f"Z_M448_0_{split}"]
        seal_path = run_dir / "SEALED_RAW.json"
        seal = read_json(seal_path)
        require(seal.get("status") == "raw_sealed_outcome_embargoed", f"Raw seal differs: {split}")
        require(seal.get("quality_metrics_present") is False, f"Raw seal contains quality metrics: {split}")
        require(seal.get("raw_predictions_sha256") == sha256_file(run_dir / "predictions.raw.jsonl"),
                f"Raw seal prediction hash differs: {split}")
        require(seal.get("integrity_raw_sha256") == sha256_file(run_dir / "integrity_raw.json"),
                f"Raw seal integrity hash differs: {split}")
        sealed_time = parse_time(seal.get("sealed_at_utc"), f"{split} raw seal")
        require(sealed_time <= release_time, f"Embargo released before {split} was sealed")
        require(release.get("candidate_seals", {}).get(split) == sha256_file(seal_path),
                f"Release seal hash differs: {split}")
        completion = read_json(run_dir / "COMPLETED.json")
        require(parse_time(completion.get("completed_at_utc"), f"{split} completion") >= release_time,
                f"{split} scored completion predates embargo release")
        for name in ("predictions.scored.jsonl", "metrics.json", "integrity_report.json", "COMPLETED.json"):
            require((run_dir / name).stat().st_mtime + 1 >= release_time.timestamp(),
                    f"Scored artifact predates embargo release: {split}/{name}")
        seals[split] = {"sealed_at_utc": seal["sealed_at_utc"], "sha256": sha256_file(seal_path)}
    require(registry.get("phase_status", {}).get("quality") == "complete", "Registry quality phase incomplete")
    require(registry.get("phase_status", {}).get("analysis") == "pass", "Registry analysis phase incomplete")
    require(read_json(CONTROLLER).get("candidate_outcome_embargo") is False, "Controller embargo flag remains set")
    return {"status": "pass", "release_sha256": sha256_file(release_path),
            "released_at_utc": release["released_at_utc"], "candidate_seals": seals,
            "quality_records": len(quality_ledger["records"])}


def paired_bootstrap(z_rows: Sequence[Mapping[str, Any]], full_rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, Any]:
    require([row["sample_id"] for row in z_rows] == [row["sample_id"] for row in full_rows],
            "Paired bootstrap inputs differ in order")
    z_acc = np.asarray([bool(row["success_iou_0_5"]) for row in z_rows], dtype=np.float64)
    f_acc = np.asarray([bool(row["success_iou_0_5"]) for row in full_rows], dtype=np.float64)
    z_iou = np.asarray([float(row["box_iou"]) for row in z_rows], dtype=np.float64)
    f_iou = np.asarray([float(row["box_iou"]) for row in full_rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    acc_parts: list[np.ndarray] = []
    iou_parts: list[np.ndarray] = []
    remaining = 10000
    while remaining:
        batch = min(200, remaining)
        indices = rng.integers(0, len(z_rows), size=(batch, len(z_rows)))
        acc_parts.append((z_acc[indices] - f_acc[indices]).mean(axis=1))
        iou_parts.append((z_iou[indices] - f_iou[indices]).mean(axis=1))
        remaining -= batch
    output = {}
    for name, point, distribution in (
        ("acc_at_0_5", float((z_acc - f_acc).mean()), np.concatenate(acc_parts)),
        ("mean_iou", float((z_iou - f_iou).mean()), np.concatenate(iou_parts)),
    ):
        output[name] = {
            "direction": "Z_M448_0_minus_Full", "point_delta": point,
            "lower_95": float(np.quantile(distribution, 0.025)),
            "upper_95": float(np.quantile(distribution, 0.975)),
            "resamples": 10000, "seed": seed,
        }
    return output


def split_gates(z_metrics: Mapping[str, Any], full_metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "parsed_box_rate": float(z_metrics["parsed_box_rate"]) >= float(thresholds["minimum_valid_parsed_box_rate"]),
        "accuracy_loss": float(full_metrics["acc_at_0_5"]) - float(z_metrics["acc_at_0_5"]) <= float(thresholds["maximum_accuracy_loss_absolute"]),
        "miou_loss": float(full_metrics["mean_iou"]) - float(z_metrics["mean_iou"]) <= float(thresholds["maximum_miou_loss_absolute"]),
    }
    return {"passed": all(gates.values()), "gates": gates}


def classify_heldout(gate_results: Mapping[str, Mapping[str, Any]]) -> str:
    require(set(gate_results) == {"testA", "testB"}, "Held-out classification requires exactly testA/testB")
    passes = [bool(gate_results[split]["passed"]) for split in ("testA", "testB")]
    return ("HELDOUT_CONFIRMED_BOTH_SPLITS" if all(passes)
            else "HELDOUT_MIXED_SPLIT_RESULT" if any(passes)
            else "HELDOUT_NOT_CONFIRMED")


def compare_nested_floats(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    require(set(actual) == set(expected), f"Keys differ: {label}")
    for key, value in expected.items():
        item = actual[key]
        if isinstance(value, Mapping):
            require(isinstance(item, Mapping), f"Object differs: {label}/{key}")
            compare_nested_floats(item, value, f"{label}/{key}")
        elif isinstance(value, float):
            close(item, value, f"{label}/{key}")
        else:
            require(item == value, f"Value differs: {label}/{key}")


def audit_analysis(rows: dict[str, list[dict[str, Any]]], protocol: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    results_path = CAMPAIGN / "analysis/HELDOUT_RESULTS.json"
    contrasts_path = CAMPAIGN / "analysis/PAIRED_CONTRASTS.json"
    decision_path = CAMPAIGN / "analysis/HELDOUT_DECISION.json"
    heterogeneity_path = CAMPAIGN / "analysis/SPLIT_HETEROGENEITY.json"
    results = read_json(results_path)
    contrasts = read_json(contrasts_path)
    decision = read_json(decision_path)
    heterogeneity = read_json(heterogeneity_path)
    release_time = parse_time(read_json(CAMPAIGN / "OUTCOME_EMBARGO_RELEASE.json")["released_at_utc"], "release")
    require(parse_time(results.get("analyzed_at_utc"), "analysis") >= release_time, "Analysis predates embargo release")
    recomputed_metrics: dict[str, Any] = {}
    recomputed_contrasts: dict[str, Any] = {}
    gate_results: dict[str, Any] = {}
    for split_index, split in enumerate(("testA", "testB")):
        full = rows[f"Full_{split}"]
        z = rows[f"Z_M448_0_{split}"]
        recomputed_metrics[split] = {"Full": normalized_metrics(full), "Z_M448_0": normalized_metrics(z)}
        # The prelocked analysis source derives a deterministic child seed by split index.
        recomputed_contrasts[split] = paired_bootstrap(z, full, int(protocol["bootstrap"]["seed"]) + split_index)
        z_metrics, f_metrics = recomputed_metrics[split]["Z_M448_0"], recomputed_metrics[split]["Full"]
        gate_results[split] = split_gates(z_metrics, f_metrics, protocol["thresholds"])
    status = classify_heldout(gate_results)
    compare_nested_floats(results.get("metrics", {}), recomputed_metrics, "heldout metrics")
    compare_nested_floats(contrasts.get("contrasts", {}), recomputed_contrasts, "paired contrasts")
    require(results.get("paired_contrasts") == contrasts.get("contrasts"), "Results/contrast artifacts disagree")
    require(results.get("gate_results") == gate_results, "Stored gate results differ")
    require(results.get("splits_pooled") is False and results.get("bootstrap") == protocol["bootstrap"],
            "Analysis pooled splits or changed bootstrap")
    require(decision.get("status") == status and decision.get("gate_results") == gate_results,
            "Held-out decision differs")
    require(decision.get("thresholds") == protocol["thresholds"], "Decision thresholds differ")
    delta = {
        metric: recomputed_contrasts["testA"][metric]["point_delta"] - recomputed_contrasts["testB"][metric]["point_delta"]
        for metric in ("acc_at_0_5", "mean_iou")
    }
    require(heterogeneity.get("status") == "complete", "Split heterogeneity artifact incomplete")
    require(heterogeneity.get("direction") == "testA_minus_testB_of_Z_minus_Full",
            "Split heterogeneity direction differs")
    compare_nested_floats(heterogeneity.get("split_difference_in_z_minus_full", {}), delta, "heterogeneity")
    csv_rows = list(csv.DictReader((CAMPAIGN / "analysis/HELDOUT_RESULTS.csv").open(encoding="utf-8")))
    require([(row["split"], row["method"]) for row in csv_rows] == [
        ("testA", "Full"), ("testA", "Z_M448_0"), ("testB", "Full"), ("testB", "Z_M448_0")
    ], "Held-out CSV inventory/order differs")
    for row in csv_rows:
        expected = recomputed_metrics[row["split"]][row["method"]]
        for key in ("samples", "parsed_box_rate", "acc_at_0_5", "mean_iou", "average_visual_tokens", "token_layer_work"):
            close(row[key], expected[key], f"CSV {row['split']}/{row['method']}/{key}")
    return status, {"status": "pass", "heldout_status": status,
                    "results_sha256": sha256_file(results_path), "contrasts_sha256": sha256_file(contrasts_path),
                    "decision_sha256": sha256_file(decision_path),
                    "heterogeneity_sha256": sha256_file(heterogeneity_path),
                    "bootstrap_child_seeds": {"testA": 20260820, "testB": 20260821}, "splits_pooled": False}


def audit_decision_ledger_and_report(status: str) -> dict[str, Any]:
    ledger_path = CAMPAIGN / "DECISION_LEDGER.jsonl"
    rows = read_jsonl(ledger_path)
    require(rows, "Decision ledger is empty")
    previous = None
    events = []
    previous_time: datetime | None = None
    for index, row in enumerate(rows):
        require(row.get("previous_entry_sha256") == previous, f"Decision ledger chain differs: {index}")
        payload = {key: value for key, value in row.items() if key != "entry_sha256"}
        expected = json_checksum(payload)
        require(row.get("entry_sha256") == expected, f"Decision ledger entry hash differs: {index}")
        recorded = parse_time(row.get("recorded_at_utc"), f"decision ledger row {index}")
        require(previous_time is None or previous_time <= recorded, f"Decision ledger time order differs: {index}")
        previous_time = recorded
        evidence_relative = row.get("evidence_path")
        require(isinstance(evidence_relative, str) and evidence_relative,
                f"Decision ledger evidence path missing: {index}")
        evidence = (CAMPAIGN / evidence_relative).resolve()
        require(evidence.is_relative_to(CAMPAIGN.resolve()), f"Decision ledger evidence escapes campaign: {index}")
        require(row.get("evidence_sha256") == sha256_file(evidence),
                f"Decision ledger evidence hash differs: {index}")
        previous = expected
        events.append(row.get("event"))
    required_events = (
        "ledger_initialized_from_immutable_pre_z_evidence",
        "historical_full_reuse_decision",
        "physical_smokes_passed",
        "preregistration_report_snapshot_preserved",
        "quality_cells_complete_and_bound",
        "outcome_embargo_released_after_both_raw_seals",
        "preregistered_heldout_rule_applied",
        "cumulative_report_finalized",
        "dissertation_and_pdf_qa_complete",
    )
    require(events == list(required_events), f"Decision ledger event inventory/order differs: {events}")
    require(rows[6].get("heldout_status") == status, "Decision-ledger held-out status differs")
    lock = read_json(CAMPAIGN / "DECISION_LEDGER_PRE_CANDIDATE_LOCK.json")
    prefix_count = int(lock.get("entries", -1))
    require(lock.get("status") == "locked" and prefix_count == PRE_CANDIDATE_LEDGER_ENTRIES,
            "Pre-candidate decision lock malformed")
    require(lock.get("last_entry_sha256") == PRE_CANDIDATE_LEDGER_HEAD_SHA256 == rows[prefix_count - 1]["entry_sha256"],
            "Pre-candidate decision-ledger head differs")
    prefix_payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
        for row in rows[:prefix_count]
    )
    prefix_sha256 = sha256_bytes(prefix_payload)
    require(lock.get("prefix_sha256") == PRE_CANDIDATE_LEDGER_PREFIX_SHA256 == prefix_sha256,
            "Decision ledger pre-candidate prefix changed")
    z_start = parse_time(read_json(CAMPAIGN / EXPECTED_QUALITY_DIRS["Z_M448_0_testA"] / "config.json")["created_at_utc"], "Z start")
    lock_time = parse_time(lock.get("locked_at_utc"), "decision ledger lock")
    require(parse_time(rows[prefix_count - 1]["recorded_at_utc"], "last pre-candidate event") <= lock_time <= z_start,
            "Decision ledger was not locked before the candidate")
    report = REPORT.read_text(encoding="utf-8")
    required_text = ("preregister", "Full", "testA", "testB", "outcome embargo", "paired",
                     "95%", "heterogeneity", status, "equivalence", "timing")
    missing = [text for text in required_text if text.lower() not in report.lower()]
    require(not missing, f"Cumulative research report omits required content: {missing}")
    forbidden = ("candidate was switched", "selected a replacement", "held-out speedup", "proves equivalence")
    hits = [text for text in forbidden if text in report.lower()]
    require(not hits, f"Research report contains forbidden overclaim: {hits}")
    return {"status": "pass", "entries": len(rows), "head_sha256": previous,
            "ledger_sha256": sha256_file(ledger_path), "pre_candidate_lock_sha256": sha256_file(CAMPAIGN / "DECISION_LEDGER_PRE_CANDIDATE_LOCK.json"),
            "report_sha256_before_terminal": sha256_file(REPORT)}


def iter_evidence_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        for item in value.values():
            yield from iter_evidence_records(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_evidence_records(item)


def resolve_evidence_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for root in (DISSERTATION, REPO):
        candidate = root / path
        if candidate.exists():
            return candidate
    return DISSERTATION / path


def audit_dissertation_pdf(status: str) -> dict[str, Any]:
    asset_path = DERIVED / "ASSET_MANIFEST.json"
    qa_path = CAMPAIGN / "PDF_QA.json"
    prepared_path = CAMPAIGN / "PDF_QA_PREPARED.json"
    visual_path = CAMPAIGN / "VISUAL_INSPECTION.json"
    manifest = read_json(asset_path)
    require(manifest.get("status") == "complete", "Held-out dissertation asset manifest incomplete")
    require(manifest.get("heldout_status") == status, "Dissertation asset status differs")
    source_names = {
        "HELDOUT_RESULTS.json", "PAIRED_CONTRASTS.json", "HELDOUT_DECISION.json",
        "SPLIT_HETEROGENEITY.json",
    }
    sources = manifest.get("sources")
    require(isinstance(sources, dict) and set(sources) == source_names, "Dissertation source inventory differs")
    for name, record in sources.items():
        source = CAMPAIGN / "analysis" / name
        derived = DERIVED / name
        require(record == {"source_sha256": sha256_file(source), "derived_sha256": sha256_file(derived)},
                f"Dissertation copied-source binding differs: {name}")
        require(source.read_bytes() == derived.read_bytes(), f"Dissertation copied source is not byte-identical: {name}")
    generated = manifest.get("generated")
    require(isinstance(generated, dict) and set(generated) == {
        "macros.tex", "main.tex", "CLAIM_EVIDENCE_LEDGER.json",
    },
            "Generated dissertation asset inventory differs")
    for name, record in generated.items():
        require(isinstance(record, dict) and set(record) == {"path", "sha256"},
                f"Generated asset schema differs: {name}")
        require(sha256_file(Path(record["path"])) == record["sha256"], f"Generated dissertation asset changed: {name}")
    claims = read_json(DERIVED / "CLAIM_EVIDENCE_LEDGER.json")
    require(claims.get("status") == "complete" and claims.get("heldout_status") == status,
            "Held-out claim-evidence ledger differs")
    require(claims.get("source_sha256") == {
        name: sha256_file(CAMPAIGN / "analysis" / name) for name in source_names
    }, "Claim-evidence source bindings differ")
    claim_rows = claims.get("claims", {})
    require(claim_rows.get("candidate_frozen_before_heldout", {}).get("supported") is True,
            "Frozen-candidate claim is unsupported")
    require(claim_rows.get("splitwise_quality_confirmation", {}).get("supported") is True,
            "Splitwise-quality claim is unsupported")
    require(claim_rows.get("heldout_timing_claim", {}).get("supported") is False,
            "Claim ledger incorrectly supports held-out timing")
    require(claim_rows.get("population_equivalence_claim", {}).get("supported") is False,
            "Claim ledger incorrectly supports population equivalence")
    change = read_json(CAMPAIGN / "DISSERTATION_CHANGE_RECORD.json")
    require(change.get("status") == "complete", "Dissertation change record incomplete")
    backup_root = Path(change["backup_root"])
    require(backup_root.is_dir(), "Dissertation backup directory missing")
    for name, item in change.get("baseline_files", {}).items():
        backup = backup_root / "Dissertation_Draft" / name
        require(item == {"bytes": backup.stat().st_size, "sha256": sha256_file(backup)},
                f"Dissertation backup changed: {name}")
    backed_user = backup_root / "dissertation_final.pdf"
    require(change.get("baseline_user_pdf") == {
        "bytes": backed_user.stat().st_size, "sha256": sha256_file(backed_user),
    }, "Backed-up user PDF changed")
    for name, item in change.get("final_files", {}).items():
        path = DISSERTATION / name
        require(item == {"bytes": path.stat().st_size, "sha256": sha256_file(path)},
                f"Final dissertation file changed: {name}")
    qa = read_json(qa_path)
    prepared, visual = read_json(prepared_path), read_json(visual_path)
    require(qa.get("status") == "pass" and qa.get("heldout_status") == status, "PDF QA did not pass")
    require(qa == read_json(DERIVED / "PDF_QA.json"), "Campaign/derived PDF-QA records differ")
    require(qa.get("prepared_sha256") == sha256_file(prepared_path), "Prepared PDF-QA record changed")
    require(qa.get("visual_inspection_sha256") == sha256_file(visual_path), "Visual-inspection record changed")
    page_count = int(prepared.get("pages", 0))
    expected_pages = list(range(1, page_count + 1))
    require(page_count > 0 and prepared.get("pages_to_render") == expected_pages,
            "Prepared QA does not cover every PDF page")
    require(visual.get("status") == "pass" and visual.get("inspected_pages") == expected_pages,
            "Visual QA does not cover every PDF page")
    require(qa.get("all_pages_visually_inspected") is True and qa.get("pages") == page_count,
            "Final QA does not certify all pages")
    rendered = prepared.get("rendered_pages")
    visual_pages = visual.get("pages")
    require(isinstance(rendered, dict) and set(rendered) == {str(page) for page in expected_pages},
            "Rendered-page inventory differs")
    require(isinstance(visual_pages, dict) and set(visual_pages) == set(rendered),
            "Visual page-note inventory differs")
    for page, record in rendered.items():
        image_hash = sha256_file(Path(record["path"]))
        require(record.get("sha256") == image_hash, f"Rendered page changed: {page}")
        inspection = visual_pages[page]
        require(inspection.get("status") == "pass" and len(str(inspection.get("note") or "").strip()) >= 20,
                f"Visual note is not substantive: page {page}")
        require(inspection.get("rendered_sha256") == image_hash, f"Visual note binds a different page: {page}")
    require(SOURCE_PDF.is_file() and USER_PDF.is_file(), "Validated/user PDF missing")
    source_hash = sha256_file(SOURCE_PDF)
    require(source_hash == sha256_file(USER_PDF) == qa.get("pdf_sha256") == qa.get("published_pdf_sha256"),
            "Published PDF is not byte-identical to QA PDF")
    require(change.get("pdf_qa_sha256") == sha256_file(qa_path), "Change record does not bind final PDF QA")
    require(change.get("final_draft_pdf") == {"bytes": SOURCE_PDF.stat().st_size, "sha256": source_hash},
            "Change record does not bind the final draft PDF")
    require(change.get("final_user_pdf") == {"bytes": USER_PDF.stat().st_size, "sha256": source_hash},
            "Change record does not bind the published PDF")
    log = DISSERTATION / "Dissertation_Template.log"
    build_log = log.read_text(encoding="utf-8", errors="replace")
    broken = re.findall(r"There were undefined (?:references|citations)|multiply defined|Fatal error occurred|Emergency stop", build_log, re.I)
    require(not broken, f"LaTeX build has terminal errors: {broken}")
    subprocess.run(["pdfinfo", str(USER_PDF)], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pdf_text = subprocess.check_output(["pdftotext", str(USER_PDF), "-"], text=True)
    normalized_pdf_text = " ".join(pdf_text.split())
    status_text = {
        "HELDOUT_CONFIRMED_BOTH_SPLITS": "confirmed on both frozen splits",
        "HELDOUT_MIXED_SPLIT_RESULT": "mixed split result",
        "HELDOUT_NOT_CONFIRMED": "not confirmed under",
    }[status]
    markers = ("testA", "testB", "Full", status_text)
    require(all(marker in normalized_pdf_text for marker in markers), "Final PDF omits held-out status/evidence markers")
    require("Z_M448_0" in normalized_pdf_text or "Z-M448-0" in normalized_pdf_text, "Final PDF omits Z_M448_0")
    require("descriptive" in normalized_pdf_text.lower() and "timing" in normalized_pdf_text.lower(), "Final PDF omits descriptive/timing caveat")
    publication_path = CAMPAIGN / "PDF_PUBLICATION_RECORD.json"
    derived_publication_path = DERIVED / "PDF_PUBLICATION_RECORD.json"
    publication = read_json(publication_path)
    require(publication.get("status") == "published_after_visual_qa" and publication.get("byte_identical") is True,
            "PDF publication record incomplete")
    require(publication.get("source_pdf_sha256") == source_hash == publication.get("published_pdf_sha256"),
            "PDF publication identity differs")
    require(publication.get("pdf_qa_sha256") == sha256_file(qa_path), "Publication does not bind final PDF QA")
    require(publication == read_json(derived_publication_path), "Campaign/derived publication records differ")
    postpublication = manifest.get("postpublication")
    require(isinstance(postpublication, dict) and set(postpublication) == {"PDF_QA.json", "PDF_PUBLICATION_RECORD.json"},
            "Postpublication asset inventory differs")
    require(postpublication["PDF_QA.json"] == {"sha256": sha256_file(DERIVED / "PDF_QA.json")},
            "Asset manifest PDF-QA binding differs")
    require(postpublication["PDF_PUBLICATION_RECORD.json"] == {"sha256": sha256_file(derived_publication_path)},
            "Asset manifest publication binding differs")
    return {"status": "pass", "asset_manifest_sha256": sha256_file(asset_path),
            "claim_evidence_ledger_sha256": sha256_file(DERIVED / "CLAIM_EVIDENCE_LEDGER.json"),
            "pdf_qa_sha256": sha256_file(qa_path), "prepared_qa_sha256": sha256_file(prepared_path),
            "visual_inspection_sha256": sha256_file(visual_path),
            "source_pdf_sha256": source_hash, "user_pdf_sha256": source_hash, "pages": expected_pages,
            "change_record_sha256": sha256_file(CAMPAIGN / "DISSERTATION_CHANGE_RECORD.json"),
            "publication_record_sha256": sha256_file(publication_path)}


def audit_process_git_tests(protocol: dict[str, Any]) -> dict[str, Any]:
    current = git("rev-parse", "HEAD")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", INFERENCE_COMMIT, current], cwd=REPO)
    require(ancestor.returncode == 0, "Final source is not a descendant of the inference epoch")
    status = git("status", "--porcelain", "--untracked-files=all").splitlines()
    allowed = {"?? PRO_REASONING_RESEARCH_HANDOFF.md", "?? legacy_goal2.md"}
    require(set(status) <= allowed, f"Unexpected terminal Git status: {status}")
    diff = subprocess.run(["git", "diff", "--check"], cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    require(diff.returncode == 0, f"git diff --check failed: {diff.stdout}")
    compile_targets = sorted({
        *REPO.glob("scripts/*frozen_z*.py"), *REPO.glob("tests/test_*frozen_z*.py"),
        *DISSERTATION.glob("analysis/*frozen_z*.py"),
    })
    compile_failures = []
    for path in compile_targets:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (OSError, SyntaxError) as exc:
            compile_failures.append(f"{path}: {exc}")
    require(not compile_failures and compile_targets, f"Changed Python compile failed: {compile_failures}")
    test = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=REPO,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    match = re.search(r"Ran (\d+) tests?", test.stdout)
    tests_run = int(match.group(1)) if match else 0
    require(test.returncode == 0 and tests_run > 0, f"Full unit suite failed: {test.stdout[-6000:]}")
    controller = read_json(CONTROLLER)
    require(controller.get("active_workers") == [], "Controller still records active workers")
    lines = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines()
    runner_names = (
        "run_progressive_oracle_refcoco.py", "run_frozen_z_heldout_campaign.py",
        "analyse_frozen_z_heldout.py", "latexmk", "pdflatex",
    )
    active = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            pid_text, command = stripped.split(maxsplit=1)
        except ValueError:
            continue
        if int(pid_text) in {os.getpid(), os.getppid()}:
            continue
        if any(name in command for name in runner_names):
            active.append(stripped)
    require(not active, f"Campaign/build processes remain active: {active}")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True
    ).strip()
    require(not gpu, f"GPU compute process remains active: {gpu}")
    campaign_bytes = sum(path.stat().st_size for path in CAMPAIGN.rglob("*") if path.is_file())
    require(campaign_bytes <= int(protocol["storage"]["campaign_cap_bytes"]), "Final campaign cap exceeded")
    require(shutil.disk_usage("/workspace").free >= int(protocol["storage"]["minimum_runtime_free_bytes"]),
            "Final storage reserve violated")
    return {"status": "pass", "git_commit": current, "git_status": status, "git_diff_check": "pass",
            "inference_epoch_is_ancestor": True, "compile_targets": [str(path) for path in compile_targets],
            "tests": tests_run, "test_output_sha256": sha256_bytes(test.stdout.encode()),
            "test_output_tail": test.stdout[-8000:], "_test_output": test.stdout,
            "active_workers": [], "gpu_compute": []}


def collect_requirements() -> tuple[str, dict[str, Any]]:
    protocol, prereg = audit_preregistration()
    registry = read_json(REGISTRY)
    require(registry.get("phase_status", {}).get("analysis") == "pass", "Analysis phase is incomplete")
    require(registry.get("phase_status", {}).get("dissertation") == "pass", "Dissertation phase is incomplete")
    requirements: dict[str, Any] = {"preregistration": prereg}
    requirements["protected_inputs_and_source"] = audit_protected_and_source(protocol)
    manifests, requirements["manifests_and_checkpoint"] = audit_manifests_checkpoint(protocol)
    requirements["inventory_scope_storage"] = audit_inventory_scope(protocol)
    requirements["historical_full_decisions"] = audit_full_decisions(registry)
    requirements["physical_smokes"] = audit_smokes()
    quality_ledger, requirements["attempts_and_seriality"] = audit_attempts_seriality()
    ledger_cells = artifact_ledger_cells()
    scored: dict[str, list[dict[str, Any]]] = {}
    cell_evidence = {}
    for cell_id in EXPECTED_CELLS:
        split = cell_id.rsplit("_", 1)[1]
        scored[cell_id], cell_evidence[cell_id] = audit_cell(cell_id, manifests[split], protocol, ledger_cells[cell_id])
    requirements["quality_cells_and_row_invariants"] = {"status": "pass", "cells": cell_evidence}
    requirements["outcome_embargo"] = audit_embargo(registry, quality_ledger)
    status, requirements["analysis_bootstrap_and_decision"] = audit_analysis(scored, protocol)
    require(registry.get("heldout_status") == status, "Registry held-out status differs")
    requirements["decision_ledger_and_cumulative_report"] = audit_decision_ledger_and_report(status)
    requirements["dissertation_and_pdf"] = audit_dissertation_pdf(status)
    requirements["source_tests_git_processes"] = audit_process_git_tests(protocol)
    return status, requirements


def finalize_terminal_state(status: str, requirements: dict[str, Any]) -> dict[str, Any]:
    require(not FINAL.exists(), "FINAL_COMPLETION_REPORT.json already exists; refusing to overwrite")
    completed = utc_now()
    test_evidence = requirements.get("source_tests_git_processes")
    require(isinstance(test_evidence, dict) and isinstance(test_evidence.get("_test_output"), str),
            "Successful in-memory full-test evidence is missing")
    test_output = str(test_evidence.pop("_test_output"))
    require(sha256_bytes(test_output.encode()) == test_evidence["test_output_sha256"],
            "In-memory test output changed before terminal publication")
    # The cumulative report is immutable once its decision-ledger event is appended.
    # Terminal evidence belongs in FINAL_COMPLETION_REPORT.json rather than mutating
    # the report out from under that hash-chain entry.
    report_payload = REPORT.read_bytes()
    registry = read_json(REGISTRY)
    registry["phase_status"]["terminal_audit"] = "pass"
    registry["status"] = TERMINAL_STATE
    registry["terminal_state"] = TERMINAL_STATE
    registry["heldout_status"] = status
    registry["final_completion_report"] = str(FINAL.relative_to(REPO))
    registry["updated_at_utc"] = completed
    controller = read_json(CONTROLLER)
    controller.update({
        "phase": "complete", "status": TERMINAL_STATE, "heldout_status": status,
        "active_workers": [], "candidate_outcome_embargo": False,
        "next_action": "human_review_only", "updated_at_utc": completed,
    })
    registry_payload = canonical_bytes(registry)
    controller_payload = canonical_bytes(controller)
    final = {
        "schema_version": 1,
        "status": TERMINAL_STATE,
        "terminal_state": TERMINAL_STATE,
        "heldout_status": status,
        "completed_at_utc": completed,
        "campaign": CAMPAIGN.name,
        "requirements": requirements,
        "issues": [],
        "next_action": "human_review_only",
        "artifacts": {
            "registry_sha256": sha256_bytes(registry_payload),
            "controller_sha256": sha256_bytes(controller_payload),
            "report_sha256": sha256_bytes(report_payload),
            "heldout_decision_sha256": sha256_file(CAMPAIGN / "analysis/HELDOUT_DECISION.json"),
            "heldout_results_sha256": sha256_file(CAMPAIGN / "analysis/HELDOUT_RESULTS.json"),
            "paired_contrasts_sha256": sha256_file(CAMPAIGN / "analysis/PAIRED_CONTRASTS.json"),
            "split_heterogeneity_sha256": sha256_file(CAMPAIGN / "analysis/SPLIT_HETEROGENEITY.json"),
            "decision_ledger_sha256": sha256_file(CAMPAIGN / "DECISION_LEDGER.jsonl"),
            "quality_artifact_ledger_sha256": sha256_file(CAMPAIGN / "QUALITY_ARTIFACT_LEDGER.json"),
            "dissertation_asset_manifest_sha256": sha256_file(DERIVED / "ASSET_MANIFEST.json"),
            "dissertation_claim_evidence_sha256": sha256_file(DERIVED / "CLAIM_EVIDENCE_LEDGER.json"),
            "dissertation_change_record_sha256": sha256_file(CAMPAIGN / "DISSERTATION_CHANGE_RECORD.json"),
            "pdf_qa_sha256": sha256_file(CAMPAIGN / "PDF_QA.json"),
            "pdf_publication_record_sha256": sha256_file(CAMPAIGN / "PDF_PUBLICATION_RECORD.json"),
            "source_pdf_sha256": sha256_file(SOURCE_PDF),
            "user_pdf_sha256": sha256_file(USER_PDF),
        },
    }
    # No terminal file is touched before every requirement above has passed.
    atomic_write(FINAL_TEST_LOG, test_output.encode())
    atomic_write(FINAL_TEST_JSON, canonical_bytes({
        "schema_version": 1,
        "status": "pass",
        "completed_at_utc": completed,
        "tests": test_evidence["tests"],
        "log_sha256": sha256_file(FINAL_TEST_LOG),
    }))
    final["artifacts"]["final_test_log_sha256"] = sha256_file(FINAL_TEST_LOG)
    final["artifacts"]["final_test_summary_sha256"] = sha256_file(FINAL_TEST_JSON)
    atomic_write(REGISTRY, registry_payload)
    atomic_write(CONTROLLER, controller_payload)
    atomic_write(FINAL, canonical_bytes(final))
    require(sha256_file(REPORT) == final["artifacts"]["report_sha256"], "Cumulative report changed during finalization")
    require(sha256_file(REGISTRY) == final["artifacts"]["registry_sha256"], "Final registry write verification failed")
    require(sha256_file(CONTROLLER) == final["artifacts"]["controller_sha256"], "Final controller write verification failed")
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Run every audit check without terminal-state writes.")
    args = parser.parse_args()
    status, requirements = collect_requirements()
    if args.check_only:
        print(json.dumps({"status": "all_checks_pass", "heldout_status": status,
                          "requirements": sorted(requirements)}, indent=2, sort_keys=True))
        return 0
    final = finalize_terminal_state(status, requirements)
    print(json.dumps({"status": final["status"], "heldout_status": status,
                      "output": str(FINAL.relative_to(REPO))}, indent=2, sort_keys=True))
    print(TERMINAL_STATE)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TerminalAuditError, OSError, KeyError, ValueError, json.JSONDecodeError,
            subprocess.SubprocessError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
