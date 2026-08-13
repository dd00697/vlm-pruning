#!/usr/bin/env python3
"""Register, smoke, soak, execute, and validate the mandatory K=12/B=48 matrix."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "experiments/k12_b48_expanded_evaluation_20260813"
PYTHON = Path("/workspace/.followup_venv/bin/python")
MODEL = "llava-hf/llava-1.5-7b-hf"
MODEL_SNAPSHOT = Path(
    "/workspace/cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/"
    "snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_pruning.artifacts import read_jsonl_strict, sha256_file, validate_completed_run  # noqa: E402
from vlm_pruning.final_campaign import DATASETS  # noqa: E402
from vlm_pruning.gpu_scheduler import admissible_widths  # noqa: E402


_MANIFEST_ROWS_CACHE: dict[Path, list[dict[str, Any]]] = {}
_MANIFEST_SHA_CACHE: dict[Path, str] = {}


@dataclass(frozen=True)
class Cell:
    run_id: str
    dataset: str
    task: str
    split: str
    manifest: Path
    method: str
    stage: str = "broad"
    seed: int = 0

    @property
    def budget(self) -> int:
        return 576 if self.method == "full" else 48


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_event(value: dict[str, Any]) -> None:
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    with (CAMPAIGN / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": utc_now(), **value}, sort_keys=True) + "\n")


def execution_env() -> dict[str, str]:
    if not MODEL_SNAPSHOT.is_dir():
        raise RuntimeError(f"Canonical model snapshot is missing: {MODEL_SNAPSHOT}")
    env = os.environ.copy()
    env.update({
        "VLM_LOCAL_MODEL_PATH": str(MODEL_SNAPSHOT),
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_VERBOSITY": "error",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    return env


def broad_cells() -> list[Cell]:
    result = []
    for protocol in DATASETS:
        manifest = ROOT / "data/k12_b48_expanded" / protocol.key / "manifest.jsonl"
        for method in ("full", "random", "uniform", "structured_tca"):
            result.append(Cell(f"{protocol.key}__{method}", protocol.key, protocol.task, protocol.split, manifest, method))
    return result


def refcoco_repair_cells() -> list[Cell]:
    manifest = ROOT / "data/manifests/final_evaluation/refcoco_val_unique.jsonl"
    return [
        Cell("refcoco_val__full", "refcoco", "grounding", "val", manifest, "full", "refcoco_completion"),
        Cell("refcoco_val__random", "refcoco", "grounding", "val", manifest, "random", "refcoco_completion"),
        Cell("refcoco_val__uniform", "refcoco", "grounding", "val", manifest, "uniform", "refcoco_completion"),
        Cell("refcoco_val__structured_compact", "refcoco", "grounding", "val", manifest, "structured_compact", "position_control"),
    ]


def random_variance_cells() -> list[Cell]:
    result = []
    for protocol in DATASETS:
        manifest = ROOT / "data/k12_b48_expanded" / protocol.key / "manifest.jsonl"
        for seed in (1, 2):
            result.append(Cell(
                f"{protocol.key}__random_seed{seed}", protocol.key, protocol.task,
                protocol.split, manifest, "random", "random_variance", seed,
            ))
    manifest = ROOT / "data/manifests/final_evaluation/refcoco_val_unique.jsonl"
    for seed in (1, 2):
        result.append(Cell(
            f"refcoco_val__random_seed{seed}", "refcoco", "grounding", "val",
            manifest, "random", "random_variance", seed,
        ))
    return result


def cells() -> list[Cell]:
    return refcoco_repair_cells() + broad_cells() + random_variance_cells()


def output_root(phase: str) -> Path:
    return CAMPAIGN / ("runs/core" if phase == "full" else f"smoke/{phase}")


def run_dir(cell: Cell, phase: str, run_id: str | None = None) -> Path:
    return output_root(phase) / (run_id or cell.run_id)


def command(cell: Cell, phase: str, *, run_id: str | None = None, limit: int | None = None) -> list[str]:
    common = [
        "--dataset", cell.dataset, "--task", cell.task, "--manifest", str(cell.manifest.relative_to(ROOT)),
        "--split", cell.split, "--target-visual-tokens", str(cell.budget), "--seed", str(cell.seed), "--precision", "bf16",
        "--model-path", MODEL, "--run-name", run_id or cell.run_id,
        "--output-root", str(output_root(phase).relative_to(ROOT)),
        "--figures-root", str((CAMPAIGN / "figures").relative_to(ROOT)),
        "--tables-root", str((CAMPAIGN / "tables").relative_to(ROOT)),
        "--max-figures", "0", "--resume", "--skip-existing", "--no-wandb",
    ]
    if limit is not None:
        common += ["--limit", str(limit)]
    if cell.method == "full":
        return [str(PYTHON), "scripts/run_official_baseline.py", *common, "--method", "full",
                "--position-mode", "gap_preserving", "--prompt-profile", "registered",
                "--reports-root", str((CAMPAIGN / "reports").relative_to(ROOT))]
    selection = cell.method if cell.method in {"random", "uniform"} else "ug"
    position = "compact" if cell.method == "structured_compact" else "gap_preserving"
    return [
        str(PYTHON), "scripts/run_ug_prune_single_pass.py", *common,
        "--prune-after-layer", "12", "--selection-method", selection,
        "--selector-strategy", "structured_tca", "--ug-map-source", "query_max",
        "--ug-fusion-mode", "single_layer", "--ug-probe-layers", "12",
        "--ug-role-preset", "more_anchors", "--position-mode", position,
        "--decode-mode", "layer_aware_cache", "--diagnostics-mode", "benchmark",
    ]


def manifest_rows(cell: Cell, limit: int | None = None) -> list[dict[str, Any]]:
    rows = _MANIFEST_ROWS_CACHE.get(cell.manifest)
    if rows is None:
        rows = read_jsonl_strict(cell.manifest)
        _MANIFEST_ROWS_CACHE[cell.manifest] = rows
    return rows if limit is None else rows[:limit]


def manifest_sha(path: Path) -> str:
    value = _MANIFEST_SHA_CACHE.get(path)
    if value is None:
        value = sha256_file(path)
        _MANIFEST_SHA_CACHE[path] = value
    return value


def integrity(cell: Cell, phase: str, *, run_id: str | None = None, limit: int | None = None, score: bool = True) -> dict[str, Any]:
    directory = run_dir(cell, phase, run_id)
    strategy = "structured_tca" if cell.method in {"structured_tca", "structured_compact"} else None
    report = validate_completed_run(directory, manifest_rows(cell, limit), budget=cell.budget,
                                    original_tokens=576, task=cell.task, strategy=strategy)
    if report.get("status") != "pass":
        return report
    if not (directory / "COMPLETED.json").is_file():
        report["status"] = "fail"
        report.setdefault("issues", []).append("completion sentinel missing")
        return report
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    expected = {
        "source/git_dirty": False, "model/name": MODEL, "model/precision": "bf16",
        "model/decoding": "deterministic_greedy", "dataset/name": cell.dataset,
        "dataset/split": cell.split, "dataset/manifest_checksum": manifest_sha(cell.manifest),
        "dataset/num_samples": len(manifest_rows(cell, limit)), "task": cell.task,
        "tokens/target_visual_tokens": cell.budget, "method/seed": cell.seed, "limit": limit,
    }
    if cell.method != "full":
        expected.update({
            "ug/prune_after_layer": 12, "ug/map_source": "query_max", "ug/fusion_mode": "single_layer",
            "ug/probe_layers": "12", "ug/role_preset": "more_anchors",
            "ug/position_mode": "compact" if cell.method == "structured_compact" else "gap_preserving",
            "ug/decode_mode": "layer_aware_cache", "ug/diagnostics_mode": "benchmark",
            "selection/method": cell.method if cell.method in {"random", "uniform"} else "ug",
            "selection/strategy": "not_applicable" if cell.method in {"random", "uniform"} else "structured_tca",
        })
    mismatches = {key: {"expected": value, "actual": config.get(key)} for key, value in expected.items() if config.get(key) != value}
    trace_issues = []
    for row in read_jsonl_strict(directory / "per_sample_results.jsonl"):
        expected_tokens = int(row.get("original_visual_tokens", 576)) if cell.method == "full" else 48
        if int(row.get("actual_llm_visual_tokens", -1)) != expected_tokens:
            trace_issues.append(f"{row.get('sample_id')}: visual-token count")
        if cell.method in {"structured_tca", "structured_compact"}:
            roles = row.get("ug_realised_role_counts") or {}
            if any(int(roles.get(name, -1)) != count for name, count in {"target": 21, "context": 12, "anchor": 15}.items()):
                trace_issues.append(f"{row.get('sample_id')}: roles={roles}")
        if len(trace_issues) >= 20:
            break
    if trace_issues:
        mismatches["token_traces"] = {"expected": "exact per-sample contract", "actual": trace_issues}
    if score and phase == "full" and not (directory / "official_metrics.json").is_file():
        proc = subprocess.run([str(PYTHON), "scripts/score_k12_b48_official.py", "--run-dir", str(directory)],
                              cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        report["official_scorer"] = {"returncode": proc.returncode, "output": proc.stdout[-8000:]}
        if proc.returncode:
            mismatches["official_scorer"] = report["official_scorer"]
    if phase == "full" and not (directory / "official_metrics.json").is_file():
        mismatches["official_metrics"] = {"expected": "present", "actual": "missing"}
    report["config_and_trace_contract"] = {"status": "pass" if not mismatches else "fail", "mismatches": mismatches}
    if mismatches:
        report["status"] = "fail"
        report.setdefault("issues", []).append(f"config/trace mismatch: {mismatches}")
    return report


def register() -> dict[str, Any]:
    missing = sorted({str(cell.manifest) for cell in cells() if not cell.manifest.is_file()})
    if missing:
        raise RuntimeError(f"Missing canonical manifests; finish data preparation first: {missing[:10]}")
    rows = []
    for cell in cells():
        rows.append({
            "run_id": cell.run_id, "dataset": cell.dataset, "task": cell.task, "split": cell.split,
            "manifest": str(cell.manifest.relative_to(ROOT)), "manifest_sha256": manifest_sha(cell.manifest),
            "expected_rows": len(manifest_rows(cell)), "method": cell.method, "stage": cell.stage,
            "seed": cell.seed, "budget": cell.budget, "command": command(cell, "full"),
            "shell": shlex.join(command(cell, "full")),
        })
    payload = {"campaign_id": "k12_b48_expanded_evaluation_20260813", "registered_at_utc": utc_now(), "cells": rows}
    write_json(CAMPAIGN / "core_run_manifests.json", payload)
    return {"status": "registered", "cells": len(rows)}


def run_processes(jobs: list[tuple[Cell, str, str | None, int | None]], width: int) -> list[dict[str, Any]]:
    pending, active, completed = list(jobs), [], []
    logs = CAMPAIGN / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    while pending or active:
        while pending and len(active) < width:
            cell, phase, run_id, limit = pending.pop(0)
            name = run_id or cell.run_id
            handle = (logs / f"{phase}_{name}.log").open("a", encoding="utf-8")
            cmd = command(cell, phase, run_id=run_id, limit=limit)
            append_event({"event": "run_started", "run_id": name, "phase": phase, "command": cmd})
            process = subprocess.Popen(
                cmd, cwd=ROOT, env=execution_env(), text=True,
                stdout=handle, stderr=subprocess.STDOUT,
            )
            active.append((process, cell, phase, run_id, limit, handle, time.time()))
        time.sleep(2)
        remaining = []
        for process, cell, phase, run_id, limit, handle, started in active:
            code = process.poll()
            if code is None:
                remaining.append((process, cell, phase, run_id, limit, handle, started)); continue
            handle.close()
            validation = integrity(cell, phase, run_id=run_id, limit=limit, score=phase == "full") if code == 0 else {"status": "fail", "issues": [f"process exit {code}"]}
            row = {"run_id": run_id or cell.run_id, "phase": phase, "returncode": code,
                   "duration_s": time.time() - started, "validation": validation}
            completed.append(row); append_event({"event": "run_finished", **row})
        active = remaining
    return completed


def serial_smoke(limit: int) -> dict[str, Any]:
    results = run_processes([(cell, "serial", f"serial_{cell.run_id}", limit) for cell in cells()], 1)
    payload = {"status": "pass" if all(x["validation"].get("status") == "pass" for x in results) else "fail",
               "created_at_utc": utc_now(), "limit": limit, "runs": results}
    write_json(CAMPAIGN / "serial_smoke_report.json", payload); return payload


def gpu_snapshot() -> str:
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version", "--format=csv,noheader,nounits"], text=True).strip()
    except Exception as exc:
        return str(exc)


def gpu_total_memory_mb() -> float:
    value = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True,
    ).strip().splitlines()
    if len(value) != 1:
        raise RuntimeError(f"Expected exactly one campaign GPU, got {value}")
    return float(value[0])


def gpu_free_memory_mb() -> float:
    values = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True,
    ).strip().splitlines()
    if len(values) != 1:
        raise RuntimeError(f"Expected exactly one campaign GPU, got {values}")
    return float(values[0])


def serial_peak_memory_mb() -> float:
    peaks = []
    for cell in cells():
        metrics = run_dir(cell, "serial", f"serial_{cell.run_id}") / "metrics.json"
        if not metrics.is_file():
            raise RuntimeError(f"Missing serial memory evidence: {metrics}")
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        peak = payload.get("efficiency/peak_gpu_memory_reserved_mb")
        if peak is None:
            peak = payload.get("efficiency/peak_gpu_memory_mb")
        if peak is None:
            raise RuntimeError(f"Serial peak memory missing: {metrics}")
        peaks.append(float(peak))
    return max(peaks)


def soak(limit: int) -> dict[str, Any]:
    reps = [x for x in cells() if x.dataset == "gqa" and x.method in {"random", "uniform", "structured_tca"}]
    attempts, safe = [], 0
    total_mb = gpu_total_memory_mb()
    peak_mb = serial_peak_memory_mb()
    widths, headroom_mb = admissible_widths(total_mb, peak_mb)
    skipped = [width for width in (3, 2, 1) if width not in widths]
    for width in widths:
        required_mb = width * peak_mb + headroom_mb
        free_mb = gpu_free_memory_mb()
        if free_mb < required_mb:
            attempts.append({"width": width, "passed": False, "launched": False,
                             "reason": f"current free {free_mb} MiB < required {required_mb} MiB"})
            continue
        before = gpu_snapshot()
        results = run_processes([(cell, "soak", f"soak{width}_{cell.run_id}", limit) for cell in reps[:width]], width)
        passed = all(x["validation"].get("status") == "pass" for x in results)
        attempts.append({"width": width, "before": before, "after": gpu_snapshot(), "passed": passed, "runs": results})
        if passed:
            safe = width; break
    payload = {"status": "pass" if safe else "fail", "created_at_utc": utc_now(),
               "safe_parallelism": safe, "limit": limit, "attempts": attempts,
               "memory_gate": {"gpu_total_mb": total_mb, "serial_peak_reserved_mb": peak_mb,
                               "required_headroom_mb": headroom_mb, "admissible_widths": widths,
                               "skipped_without_launch": skipped}}
    write_json(CAMPAIGN / "gpu_soak_report.json", payload); return payload


def full(max_parallel: int | None) -> dict[str, Any]:
    smoke = json.loads((CAMPAIGN / "serial_smoke_report.json").read_text())
    soak_report = json.loads((CAMPAIGN / "gpu_soak_report.json").read_text())
    if smoke.get("status") != "pass" or soak_report.get("status") != "pass":
        raise RuntimeError("Serial smoke and GPU soak must pass before full execution")
    safe = int(soak_report["safe_parallelism"])
    width = min(safe, max_parallel) if max_parallel else safe
    gate = soak_report["memory_gate"]
    required_mb = width * float(gate["serial_peak_reserved_mb"]) + float(gate["required_headroom_mb"])
    free_mb = gpu_free_memory_mb()
    if free_mb < required_mb:
        raise RuntimeError(f"Current GPU free memory {free_mb} MiB < required {required_mb} MiB")
    results = run_processes([(cell, "full", None, None) for cell in cells()], width)
    payload = {"status": "pass" if all(x["validation"].get("status") == "pass" for x in results) else "fail",
               "created_at_utc": utc_now(), "parallelism": width, "runs": results}
    write_json(CAMPAIGN / "core_completion.json", payload); return payload


def status() -> dict[str, Any]:
    rows = []
    for cell in cells():
        directory, state = run_dir(cell, "full"), "missing"
        if (directory / "per_sample_results.jsonl").is_file():
            try: state = integrity(cell, "full", score=False).get("status", "fail")
            except Exception as exc: state = f"fail:{exc}"
        elif (directory / "per_sample_results.partial.jsonl").is_file():
            state = f"partial:{sum(bool(line.strip()) for line in (directory / 'per_sample_results.partial.jsonl').open())}"
        rows.append({"run_id": cell.run_id, "state": state})
    return {"status": "pass", "runs": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("register")
    p = sub.add_parser("serial-smoke"); p.add_argument("--limit", type=int, default=2)
    p = sub.add_parser("soak"); p.add_argument("--limit", type=int, default=20)
    p = sub.add_parser("full"); p.add_argument("--max-parallel", type=int, choices=(1, 2, 3))
    sub.add_parser("status")
    args = parser.parse_args()
    if args.action == "register": payload = register()
    elif args.action == "serial-smoke": payload = serial_smoke(args.limit)
    elif args.action == "soak": payload = soak(args.limit)
    elif args.action == "full": payload = full(args.max_parallel)
    else: payload = status()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") in {"pass", "registered"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
