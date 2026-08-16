#!/usr/bin/env python
"""Create the zero-GPU Phase 0 record for the hardware-aware campaign.

Historical campaigns are immutable inputs.  This entry point verifies their
registered artifacts and writes only beneath the new campaign root.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT.parent
CAMPAIGN_ROOT = REPO_ROOT / "experiments/hardware_aware_zero_tail_refcoco_val_20260816"
VALIDATION_MANIFEST = (
    REPO_ROOT
    / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
)
TRAIN_MANIFEST = (
    REPO_ROOT
    / "experiments/adaptive_efficiency_refcoco_val_20260815/gate_data/refcoco_train_gate3000.jsonl"
)
PROGRESSIVE_ROOT = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814"
ADAPTIVE_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
TIMING_ANALYSIS = ADAPTIVE_ROOT / "analysis/MATCHED_TIMING_RESULTS.json"
K12_RESCORED = ADAPTIVE_ROOT / "analysis/rescored/K12_ORIGINAL.jsonl"
K12_SOURCE = (
    REPO_ROOT
    / "experiments/final_selector_ablation_and_repair_20260811/runs/repair_val_k12_b48_single_layer_more_context/per_sample_results.jsonl"
)
CHECKPOINT_SNAPSHOT = (
    WORKSPACE
    / "cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
)
VALIDATION_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
TRAIN_SHA256 = "8f2b567f27ccc05d1fef7b8368f1de7c454127e224c8f25f24944444d2b1d18f"
EXPECTED_PROGRESSIVE_HASHES = {
    "FINAL_COMPLETION_REPORT.json": "baeef380d40b6095ed4f804bf321b71d9b70e54bd28651f81e94d80494a75e20",
    "EXPERIMENT_REGISTRY.json": "9f4bdbef5979e301ec9a78eff113925eb6084cd16d4483a5b2160af8a6c5214a",
    "RESEARCH_REPORT.md": "b0ff5376d4d6616c74b2a7bb03c37113af9c380fc180679f5eb3f9d87a9f0eb3",
}
EXPECTED_ADAPTIVE_HASHES = {
    "FINAL_COMPLETION_REPORT.json": None,
    "EXPERIMENT_REGISTRY.json": "0865efd0e4ecc69e37b8fb49cf2200977e2be8c6bd1ef1ea8942320711680111",
    "CONTROLLER_STATE.json": "fa01c8fddcb11882af2abdbaa1d64dace5cae2d82cb402ce7e7620b8a3b70332",
    "RESEARCH_REPORT.md": "810eec271797515d9b45e74f9e19150d558182f4813e63b11d227f39fc477eed",
    "analysis/MATCHED_TIMING_RESULTS.json": "c7cc9aa8fe329949127d703cdaa1dfb3bc7cb16169f262fb2d91d8c5919c6a2d",
    "analysis/GATE_RESULTS_AND_SELECTION.json": "3bcd742f5af0bb3b18c651ce1cc3c631197bb33902a0e471f61429b14039da36",
}
EXPECTED_K12_HASHES = {
    "rescored": "6a40fe8125fa4003fcdbe2ac33e73acd353755603d554af193fa833d525b53f2",
    "source": "1817cd703648031d9035d3547faa835839bcf441f4c14515b6a6571a1f36366d",
}
EXPECTED_M448_METRICS = {
    "acc_at_0_5": 0.43,
    "mean_iou": 0.4284360666246254,
    "parse_rate": 1.0,
    "average_visual_tokens": 214.0,
}
TIMING_FIELDS = (
    "average_visual_tokens",
    "visual_token_layer_work",
    "main_prefill_with_overhead_latency_s",
    "end_to_end_latency_s",
    "decode_latency_s",
    "prompt_kv_total_bytes",
    "final_kv_total_bytes",
    "query_scoring_latency_s",
    "selector_latency_s",
    "gather_latency_s",
    "merge_latency_s",
    "generated_token_count",
)
SOURCE_INVENTORY = (
    "vlm_pruning/llava_hooks/progressive_pruning.py",
    "vlm_pruning/llava_hooks/mid_layer_pruning.py",
    "vlm_pruning/llava_hooks/route_gate.py",
    "scripts/run_progressive_oracle_refcoco.py",
    "scripts/validate_progressive_oracle_run.py",
    "scripts/run_adaptive_matched_timing_trial.py",
    "scripts/validate_adaptive_matched_timing_trial.py",
    "scripts/analyse_adaptive_matched_timing.py",
)
CELLS = {
    "P_K12": {"phase": "profiling", "kind": "mandatory", "method": "exact_original_k12"},
    "P_M448": {"phase": "profiling", "kind": "mandatory", "method": "exact_accepted_m448"},
    "P_M448_REPLAY": {"phase": "profiling", "kind": "mandatory_nondeployable_upper_bound"},
    "R_FUSED_GATHER_MERGE": {"phase": "runtime_repairs", "kind": "conditional"},
    "R_NATIVE_CACHE": {"phase": "runtime_repairs", "kind": "conditional"},
    "R_SCORE_REUSE": {"phase": "runtime_repairs", "kind": "conditional"},
    "R_COMPILED_TORCH": {"phase": "runtime_repairs", "kind": "conditional_after_exact_repair"},
    "R_COMPILED_CUDA_GRAPH": {"phase": "runtime_repairs", "kind": "conditional_after_exact_repair"},
    "Z_K12_0": {"phase": "zero_tail_quality", "kind": "mandatory", "average_visual_tokens": 228.0},
    "Z_M448_0": {"phase": "zero_tail_quality", "kind": "mandatory", "average_visual_tokens": 196.0},
    "Z_M448_16": {"phase": "zero_tail_quality", "kind": "conditional_quality_fallback", "average_visual_tokens": 202.0},
    "E256_POOL": {"phase": "early_compression", "kind": "conditional_phase_mandatory", "average_visual_tokens": 256.0},
    "E196_POOL": {"phase": "early_compression", "kind": "conditional_on_e256_quality", "average_visual_tokens": 196.0},
    "E256_LORA": {"phase": "early_compression", "kind": "strictly_conditional_projector_only_training", "average_visual_tokens": 256.0},
}


class Phase0Error(RuntimeError):
    """Raised when frozen provenance cannot be established."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise Phase0Error(f"Missing evidence file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> tuple[str | None, int, int]:
    if not root.is_dir():
        return None, 0, 0
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative_path = path.relative_to(root).as_posix()
        file_hash = sha256_file(path)
        size = path.stat().st_size
        digest.update(f"{relative_path}\0{file_hash}\0{size}\n".encode())
        files += 1
        total_bytes += size
    return digest.hexdigest(), files, total_bytes


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase0Error(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise Phase0Error(f"Blank JSONL row: {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise Phase0Error(f"Non-object JSONL row: {path}:{line_number}")
            rows.append(row)
    return rows


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_bytes(value))


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def run(command: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=check)


def order_sha256(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def validate_ordered_ids(rows: list[dict[str, Any]], expected_ids: list[str], path: Path) -> dict[str, Any]:
    ids = [str(row.get("sample_id") or "") for row in rows]
    if ids != expected_ids:
        raise Phase0Error(f"Frozen row order differs: {path}")
    if len(ids) != len(set(ids)) or any(not sample_id for sample_id in ids):
        raise Phase0Error(f"Missing or duplicate sample IDs: {path}")
    return {"rows": len(ids), "unique_ids": len(set(ids)), "sample_id_order_sha256": order_sha256(ids)}


def validate_manifest(path: Path, expected_hash: str, expected_rows: int, *, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise Phase0Error(f"Manifest SHA-256 mismatch: {actual_hash} != {expected_hash}")
    rows = read_jsonl(path)
    ids = [str(row.get("sample_id") or "") for row in rows]
    if len(rows) != expected_rows or len(set(ids)) != expected_rows or any(not value for value in ids):
        raise Phase0Error(f"Invalid manifest rows/IDs: {path}")
    if any(str(row.get("dataset", "")).lower() != "refcoco" for row in rows):
        raise Phase0Error(f"Non-RefCOCO row in {path}")
    if any(str(row.get("split", "")).lower() != split for row in rows):
        raise Phase0Error(f"Unexpected split in {path}")
    missing_images = [str(row.get("image_path")) for row in rows if not (REPO_ROOT / str(row.get("image_path"))).is_file()]
    if missing_images:
        raise Phase0Error(f"Manifest has {len(missing_images)} missing images")
    return rows, {
        "path": relative(path),
        "sha256": actual_hash,
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "sample_id_order_sha256": order_sha256(ids),
        "all_images_exist": True,
        "first_sample_id": ids[0],
        "last_sample_id": ids[-1],
    }


def validate_train_partitions(rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gate: defaultdict[str, int] = defaultdict(int)
    training: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        gate[str(row.get("gate_partition"))] += 1
        training[str(row.get("training_partition"))] += 1
    expected_gate = {"development": 2400, "calibration": 600}
    expected_training = {"fit": 2160, "early_stop": 240, "calibration": 600}
    if dict(gate) != expected_gate or dict(training) != expected_training:
        raise Phase0Error(f"Train partition mismatch: gate={dict(gate)}, training={dict(training)}")
    validation_ids = {str(row["sample_id"]) for row in validation_rows}
    train_ids = {str(row["sample_id"]) for row in rows}
    overlap = validation_ids & train_ids
    if overlap:
        raise Phase0Error(f"Train/validation sample-ID leakage: {len(overlap)} rows")
    validation_images = {str(row.get("image_id")) for row in validation_rows}
    train_images = {str(row.get("image_id")) for row in rows}
    return {
        "gate_partition_counts": expected_gate,
        "training_partition_counts": expected_training,
        "validation_sample_id_overlap": 0,
        "validation_image_id_overlap": len(validation_images & train_images),
        "validation_image_overlap_note": "Historical manifest mixes source COCO splits; record overlap is zero and image overlap is diagnostic only.",
    }


def verify_file_hashes(root: Path, expected: dict[str, str | None]) -> dict[str, str]:
    output: dict[str, str] = {}
    for name, registered_hash in expected.items():
        path = root / name
        actual = sha256_file(path)
        if registered_hash is not None and actual != registered_hash:
            raise Phase0Error(f"Historical hash mismatch for {path}: {actual} != {registered_hash}")
        output[relative(path)] = actual
    return output


def validate_m448(registry: dict[str, Any], validation_ids: list[str]) -> dict[str, Any]:
    cell = ((registry.get("cells") or {}).get("M448") or {})
    if cell.get("status") != "accepted" or cell.get("schedule") != "4:448:query_topk_merge,12:48:query_topk":
        raise Phase0Error("Adaptive registry does not contain the accepted exact M448 cell")
    accepted = cell.get("accepted_run") or {}
    run_dir = REPO_ROOT / str(accepted.get("run_dir"))
    names = {
        "config_sha256": "config.json",
        "integrity_sha256": "integrity_report.json",
        "metrics_sha256": "metrics.json",
        "predictions_sha256": "predictions.jsonl",
        "completion_sha256": "COMPLETED.json",
    }
    hashes: dict[str, str] = {}
    for key, filename in names.items():
        actual = sha256_file(run_dir / filename)
        if actual != accepted.get(key):
            raise Phase0Error(f"M448 {filename} hash mismatch")
        hashes[filename] = actual
    prediction_rows = read_jsonl(run_dir / "predictions.jsonl")
    ordering = validate_ordered_ids(prediction_rows, validation_ids, run_dir / "predictions.jsonl")
    metrics = read_json(run_dir / "metrics.json")
    normalized_metrics = {
        "acc_at_0_5": float(metrics["quality/acc_at_0_5"]),
        "mean_iou": float(metrics["quality/mean_iou"]),
        "parse_rate": float(metrics["quality/parsed_box_rate"]),
        "average_visual_tokens": float(metrics["tokens/candidate_average_visual_tokens"]),
    }
    for key, expected in EXPECTED_M448_METRICS.items():
        if abs(normalized_metrics[key] - expected) > 1e-12:
            raise Phase0Error(f"M448 metric mismatch for {key}")
    attempts = cell.get("attempts") or []
    source_commit = attempts[-1].get("source_commit") if attempts else None
    if source_commit != "747168fa52fd8fe659ae5252c00470086b2ad79d":
        raise Phase0Error("M448 source commit mismatch")
    return {
        "run_dir": relative(run_dir),
        "source_commit": source_commit,
        "schedule": cell["schedule"],
        "hashes": hashes,
        "ordering": ordering,
        "metrics": normalized_metrics,
    }


def validate_k12(validation_ids: list[str]) -> dict[str, Any]:
    hashes = {"rescored": sha256_file(K12_RESCORED), "source": sha256_file(K12_SOURCE)}
    if hashes != EXPECTED_K12_HASHES:
        raise Phase0Error(f"K12 hash mismatch: {hashes}")
    rescored = read_jsonl(K12_RESCORED)
    ordering = validate_ordered_ids(rescored, validation_ids, K12_RESCORED)
    source_rows = read_jsonl(K12_SOURCE)
    source_ids = [str(row.get("sample_id") or "") for row in source_rows]
    if len(source_ids) != len(set(source_ids)) or not set(validation_ids).issubset(source_ids):
        raise Phase0Error("K12 source does not uniquely cover the frozen validation manifest")
    accuracy = sum(bool(row["success_iou_0_5"]) for row in rescored) / len(rescored)
    mean_iou = sum(float(row["box_iou"]) for row in rescored) / len(rescored)
    parse_rate = sum(row.get("parse_failure_reason") is None for row in rescored) / len(rescored)
    expected = (0.439, 0.4215978635808666, 1.0)
    actual = (accuracy, mean_iou, parse_rate)
    if any(abs(left - right) > 1e-12 for left, right in zip(actual, expected, strict=True)):
        raise Phase0Error(f"K12 metric mismatch: {actual}")
    return {
        "rescored_path": relative(K12_RESCORED),
        "source_path": relative(K12_SOURCE),
        "hashes": hashes,
        "ordering": ordering,
        "source_rows": len(source_rows),
        "frozen_rows_covered": len(validation_ids),
        "metrics": {"acc_at_0_5": accuracy, "mean_iou": mean_iou, "parse_rate": parse_rate, "average_visual_tokens": 246.0},
    }


def validate_timing(validation_ids: list[str]) -> dict[str, Any]:
    analysis = read_json(TIMING_ANALYSIS)
    if analysis.get("status") != "complete" or analysis.get("selected_project_candidate") != "M448":
        raise Phase0Error("Accepted timing analysis is not complete for M448")
    artifacts = ((analysis.get("natural") or {}).get("artifacts") or [])
    if len(artifacts) != 15:
        raise Phase0Error(f"Expected 15 natural timing artifacts, found {len(artifacts)}")
    rows_by_method: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    artifact_evidence: list[dict[str, Any]] = []
    for artifact in artifacts:
        run_dir = REPO_ROOT / str(artifact["run_dir"])
        hashes = {
            "timing_sha256": sha256_file(run_dir / "timing.jsonl"),
            "integrity_sha256": sha256_file(run_dir / "integrity_report.json"),
            "completion_sha256": sha256_file(run_dir / "COMPLETED.json"),
        }
        for key, actual in hashes.items():
            if actual != artifact[key]:
                raise Phase0Error(f"Timing artifact hash mismatch: {run_dir}/{key}")
        rows = read_jsonl(run_dir / "timing.jsonl")
        ordering = validate_ordered_ids(rows, validation_ids, run_dir / "timing.jsonl")
        method = str(artifact["method"])
        if any(str(row.get("method")) != method for row in rows):
            raise Phase0Error(f"Timing method label mismatch: {run_dir}")
        rows_by_method[method].extend(rows)
        artifact_evidence.append({"run_dir": relative(run_dir), "method": method, **hashes, **ordering})
    recomputed: dict[str, dict[str, float]] = {}
    recorded = ((analysis.get("natural") or {}).get("methods") or {})
    for method, rows in sorted(rows_by_method.items()):
        recomputed[method] = {}
        for field in TIMING_FIELDS:
            mean = sum(float(row[field]) for row in rows) / len(rows)
            recomputed[method][field] = mean
            recorded_mean = float(recorded[method][field]["mean"])
            if abs(mean - recorded_mean) > 1e-12:
                raise Phase0Error(f"Timing recomputation mismatch: {method}/{field}")
    return {
        "analysis_path": relative(TIMING_ANALYSIS),
        "analysis_sha256": sha256_file(TIMING_ANALYSIS),
        "natural_artifacts": artifact_evidence,
        "rows": sum(len(rows) for rows in rows_by_method.values()),
        "recomputed_component_table": recomputed,
        "all_registered_hashes_match": True,
        "all_rows_match_frozen_order": True,
    }


def git_audit() -> dict[str, Any]:
    tracked = run(["git", "status", "--porcelain=v1", "--untracked-files=no"], check=True).stdout.splitlines()
    all_status = run(["git", "status", "--porcelain=v1", "--untracked-files=all"], check=True).stdout.splitlines()
    if tracked:
        raise Phase0Error(f"Tracked source is dirty: {tracked}")
    if all_status != ["?? PRO_REASONING_RESEARCH_HANDOFF.md"]:
        raise Phase0Error(f"Unexpected Git status: {all_status}")
    handoff = REPO_ROOT / "PRO_REASONING_RESEARCH_HANDOFF.md"
    return {
        "source_commit": run(["git", "rev-parse", "HEAD"], check=True).stdout.strip(),
        "tracked_source_clean": True,
        "status": all_status,
        "user_handoff_path": relative(handoff),
        "user_handoff_sha256": sha256_file(handoff),
    }


def environment_audit() -> dict[str, Any]:
    import torch
    import transformers

    gpu = run([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    compute = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    process_lines = [line for line in compute.stdout.splitlines() if line.strip()]
    if gpu.returncode or compute.returncode or process_lines:
        raise Phase0Error(f"GPU preflight failed or GPU is occupied: {process_lines}")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": [line for line in gpu.stdout.splitlines() if line.strip()],
        "active_compute_processes": process_lines,
    }


def checkpoint_audit() -> dict[str, Any]:
    if not CHECKPOINT_SNAPSHOT.is_dir():
        raise Phase0Error(f"Pinned checkpoint snapshot is absent: {CHECKPOINT_SNAPSHOT}")
    required = ["config.json", "model.safetensors.index.json", "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]
    missing = [name for name in required if not (CHECKPOINT_SNAPSHOT / name).is_file()]
    if missing:
        raise Phase0Error(f"Pinned checkpoint cache is incomplete: {missing}")
    links = sorted(path for path in CHECKPOINT_SNAPSHOT.iterdir() if path.is_symlink())
    broken = [path.name for path in links if not path.resolve().is_file()]
    if broken:
        raise Phase0Error(f"Broken checkpoint cache links: {broken}")
    return {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "snapshot_revision": CHECKPOINT_SNAPSHOT.name,
        "snapshot_path": str(CHECKPOINT_SNAPSHOT),
        "config_sha256": sha256_file(CHECKPOINT_SNAPSHOT / "config.json"),
        "model_index_sha256": sha256_file(CHECKPOINT_SNAPSHOT / "model.safetensors.index.json"),
        "symlinks": len(links),
        "broken_symlinks": broken,
        "required_files_present": required,
    }


def active_campaign_processes() -> list[str]:
    result = run(["ps", "-eo", "pid=,args="])
    marker = CAMPAIGN_ROOT.name
    own = {os.getpid(), os.getppid()}
    found: list[str] = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) not in own and marker in fields[1]:
            found.append(line.strip())
    return found


def storage_audit() -> dict[str, Any]:
    usage = shutil.disk_usage(WORKSPACE)
    if usage.free < 100 * 1024**3:
        raise Phase0Error(f"Less than 100 GiB free: {usage.free}")
    return {
        "filesystem": str(WORKSPACE),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "available_bytes": usage.free,
        "available_gib": usage.free / 1024**3,
        "campaign_cap_bytes": 250 * 1024**3,
        "campaign_cap_gib": 250,
        "launch_floor_bytes": 100 * 1024**3,
        "launch_floor_gib": 100,
        "initial_campaign_bytes": 0,
        "status": "pass",
    }


def report_markdown(audit: dict[str, Any]) -> str:
    timing = audit["historical_evidence"]["timing"]["recomputed_component_table"]
    return "\n".join([
        "# Hardware-aware zero-tail RefCOCO validation campaign",
        "",
        "## Scope and frozen protocol",
        "",
        "This discovery campaign is restricted to the frozen 1,000-row RefCOCO validation manifest. No 8,811-row, held-out, transfer, learned-router, or dissertation work is authorized.",
        "",
        f"- Manifest SHA-256: `{audit['validation_manifest']['sha256']}`",
        f"- Source epoch: `{audit['git']['source_commit']}`",
        f"- Checkpoint revision: `{audit['checkpoint']['snapshot_revision']}`",
        "",
        "## Phase 0 — provenance and restart audit",
        "",
        "Passed. Historical M448, original K12, progressive completion, and all 15 accepted natural timing artifacts independently match their registered hashes and frozen row order.",
        "",
        "| Historical method | Avg visual tokens | Token-layer work | Prefill (s) | E2E (s) | Prompt KV bytes | Selector (s) | Gather (s) | Merge (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| K12 original | {timing['k12_original']['average_visual_tokens']:.1f} | {timing['k12_original']['visual_token_layer_work']:.1f} | {timing['k12_original']['main_prefill_with_overhead_latency_s']:.6f} | {timing['k12_original']['end_to_end_latency_s']:.6f} | {timing['k12_original']['prompt_kv_total_bytes']:.1f} | {timing['k12_original']['selector_latency_s']:.6f} | {timing['k12_original']['gather_latency_s']:.6f} | {timing['k12_original']['merge_latency_s']:.6f} |",
        f"| M448 | {timing['candidate']['average_visual_tokens']:.1f} | {timing['candidate']['visual_token_layer_work']:.1f} | {timing['candidate']['main_prefill_with_overhead_latency_s']:.6f} | {timing['candidate']['end_to_end_latency_s']:.6f} | {timing['candidate']['prompt_kv_total_bytes']:.1f} | {timing['candidate']['selector_latency_s']:.6f} | {timing['candidate']['gather_latency_s']:.6f} | {timing['candidate']['merge_latency_s']:.6f} |",
        "",
        "## Instrumentation correctness",
        "",
        "Pending.",
        "",
        "## Runtime attribution and repairs",
        "",
        "Pending.",
        "",
        "## Zero-tail quality",
        "",
        "Pending mandatory Z_K12_0 and Z_M448_0 cells.",
        "",
        "## Matched timing and strict-win decision",
        "",
        "Pending.",
        "",
        "## Conditional early compression and adaptation",
        "",
        "Pending the decoder strict-win branch.",
        "",
        "## Analysis, limitations, and final recommendation",
        "",
        "Pending completion of every triggered branch.",
        "",
        "## Failures and rejected attempts",
        "",
        "None in Phase 0.",
        "",
    ])


def main() -> int:
    if CAMPAIGN_ROOT.exists() and any(CAMPAIGN_ROOT.iterdir()):
        raise Phase0Error(f"Campaign root is not empty; refusing to overwrite: {CAMPAIGN_ROOT}")

    validation_rows, validation_evidence = validate_manifest(VALIDATION_MANIFEST, VALIDATION_SHA256, 1000, split="val")
    train_rows, train_evidence = validate_manifest(TRAIN_MANIFEST, TRAIN_SHA256, 3000, split="train")
    partitions = validate_train_partitions(train_rows, validation_rows)
    validation_ids = [str(row["sample_id"]) for row in validation_rows]

    progressive_hashes = verify_file_hashes(PROGRESSIVE_ROOT, EXPECTED_PROGRESSIVE_HASHES)
    progressive_final = read_json(PROGRESSIVE_ROOT / "FINAL_COMPLETION_REPORT.json")
    if progressive_final.get("status") != "pass":
        raise Phase0Error("Progressive campaign final audit is not passing")

    adaptive_hashes = verify_file_hashes(ADAPTIVE_ROOT, EXPECTED_ADAPTIVE_HASHES)
    adaptive_final = read_json(ADAPTIVE_ROOT / "FINAL_COMPLETION_REPORT.json")
    if adaptive_final.get("status") != "complete" or adaptive_final.get("stopping_condition") != "GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED":
        raise Phase0Error("Adaptive campaign is not terminally complete")
    adaptive_registry = read_json(ADAPTIVE_ROOT / "EXPERIMENT_REGISTRY.json")

    process_evidence = active_campaign_processes()
    if process_evidence:
        raise Phase0Error(f"Active campaign processes found: {process_evidence}")
    dissertation_tree_hash, dissertation_files, dissertation_bytes = tree_sha256(WORKSPACE / "Dissertation_Draft")
    dissertation_pdf = WORKSPACE / "dissertation_final.pdf"

    audit = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ROOT.name,
        "status": "pass",
        "completed_at_utc": utc_now(),
        "validation_manifest": validation_evidence,
        "conditional_train_manifest": {**train_evidence, **partitions},
        "git": git_audit(),
        "storage": storage_audit(),
        "environment": environment_audit(),
        "checkpoint": checkpoint_audit(),
        "active_campaign_processes": process_evidence,
        "historical_evidence": {
            "progressive": {"status": progressive_final["status"], "hashes": progressive_hashes},
            "adaptive": {"status": adaptive_final["status"], "stopping_condition": adaptive_final["stopping_condition"], "hashes": adaptive_hashes},
            "m448": validate_m448(adaptive_registry, validation_ids),
            "k12_original": validate_k12(validation_ids),
            "timing": validate_timing(validation_ids),
        },
        "source_inventory": {name: sha256_file(REPO_ROOT / name) for name in SOURCE_INVENTORY},
        "new_root_forbidden_output_scan": {"root_existed_before_initialization": CAMPAIGN_ROOT.exists(), "forbidden_paths": []},
        "dissertation_read_only_snapshot": {
            "source_path": str(WORKSPACE / "Dissertation_Draft"),
            "source_tree_sha256": dissertation_tree_hash,
            "source_files": dissertation_files,
            "source_bytes": dissertation_bytes,
            "pdf_path": str(dissertation_pdf),
            "pdf_exists": dissertation_pdf.is_file(),
            "pdf_sha256": sha256_file(dissertation_pdf) if dissertation_pdf.is_file() else None,
        },
        "constraints": {
            "validation_only_1000": True,
            "no_8811_testA_testB_transfer": True,
            "no_learned_router_or_patch_selector": True,
            "dissertation_read_only": True,
            "timing_serial_no_cuda_overlap": True,
        },
    }

    for directory in ("profiling", "runtime_repairs", "zero_tail", "early_compression", "timing", "analysis", "rejected_runs"):
        (CAMPAIGN_ROOT / directory).mkdir(parents=True, exist_ok=True)
    atomic_json(CAMPAIGN_ROOT / "PREFLIGHT_AUDIT.json", audit)
    atomic_json(CAMPAIGN_ROOT / "STORAGE_BUDGET.json", audit["storage"])

    phases = {
        "phase0": {"status": "complete", "evidence_path": relative(CAMPAIGN_ROOT / "PREFLIGHT_AUDIT.json")},
        "instrumentation": {"status": "pending"},
        "runtime_attribution": {"status": "pending"},
        "runtime_repairs": {"status": "pending"},
        "zero_tail_quality": {"status": "pending"},
        "decoder_timing": {"status": "pending"},
        "early_compression": {"status": "pending"},
        "analysis": {"status": "pending"},
        "final_audit": {"status": "pending"},
    }
    thresholds = {
        "k12_like": {"acc_at_0_5_gte": 0.429, "mean_iou_gte": 0.4165978636, "parse_rate_eq": 1.0},
        "runtime_repair_attribution_share_gte": 0.20,
        "strict_win": {"prompt_kv_lower": True, "mean_prefill_lower": True, "mean_natural_e2e_lower": True, "paired_hierarchical_cis_exclude_zero_in_faster_direction": True},
        "e256_lora_near_quality": {"acc_at_0_5_gte": 0.419, "mean_iou_gte": 0.4115978636},
        "e256_lora_speed": {"prefill_relative_reduction_gte": 0.10, "or_e2e_relative_reduction_gte": 0.02},
    }
    registry = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ROOT.name,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "goal_sha256": sha256_file(REPO_ROOT / "GOAL.md"),
        "agents_sha256": sha256_file(REPO_ROOT / "AGENTS.md"),
        "source_commit": audit["git"]["source_commit"],
        "manifest": validation_evidence,
        "conditional_train_manifest": train_evidence,
        "thresholds": thresholds,
        "phases": phases,
        "cells": {cell_id: {**spec, "status": "pending", "attempts": []} for cell_id, spec in CELLS.items()},
        "m448_opt": {"status": "pending"},
        "branch_decisions": [],
        "failures": [],
        "terminal_outcome": None,
    }
    atomic_json(CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json", registry)
    state = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ROOT.name,
        "updated_at_utc": utc_now(),
        "state": "PHASE0_COMPLETE_INSTRUMENTATION_PENDING",
        "current_phase": "instrumentation",
        "active_workers": [],
        "accepted_prefixes": {},
        "next_required_action": "implement_and_validate_instrumentation_then_run_k12_m448_smokes",
        "stop_reason": None,
        "decision_history": [{"phase": "phase0", "status": "complete", "evidence_path": relative(CAMPAIGN_ROOT / "PREFLIGHT_AUDIT.json")}],
    }
    atomic_json(CAMPAIGN_ROOT / "CONTROLLER_STATE.json", state)
    atomic_write(CAMPAIGN_ROOT / "RESEARCH_REPORT.md", report_markdown(audit).encode())
    completion = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "source_commit": audit["git"]["source_commit"],
        "manifest_sha256": VALIDATION_SHA256,
        "preflight_sha256": sha256_file(CAMPAIGN_ROOT / "PREFLIGHT_AUDIT.json"),
        "storage_sha256": sha256_file(CAMPAIGN_ROOT / "STORAGE_BUDGET.json"),
        "registry_sha256": sha256_file(CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"),
        "controller_sha256": sha256_file(CAMPAIGN_ROOT / "CONTROLLER_STATE.json"),
        "report_sha256": sha256_file(CAMPAIGN_ROOT / "RESEARCH_REPORT.md"),
        "next_state": state["state"],
    }
    atomic_json(CAMPAIGN_ROOT / "analysis/PHASE0_COMPLETED.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
