#!/usr/bin/env python3
"""Admit and execute exact-B48 VisionZip/TransPrune official baselines."""

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
OUTPUT_ROOT = CAMPAIGN / "runs/official_original_llava"
SMOKE_ROOT = CAMPAIGN / "smoke/official_original_llava"
ORIGINAL_CACHE = Path("/dev/shm/vlm_original_llava_cache")
ORIGINAL_REVISION = "4481d270cc22fd5c4d1bb5df129622006ccd9234"
VISION_REVISION = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
ORIGINAL_SNAPSHOT = ORIGINAL_CACHE / "models--liuhaotian--llava-v1.5-7b" / "snapshots" / ORIGINAL_REVISION
CANONICAL_SUMMARY = CAMPAIGN / "canonical_checkpoint_tensor_summary.json"
EQUIVALENCE_REPORT = CAMPAIGN / "original_llava_checkpoint_equivalence.json"
VISIONZIP_REPO = ROOT / "third_party/VisionZip"
VISIONZIP_LLAVA_REPO = Path("/workspace/third_party_official/LLaVA")
TRANSPRUNE_REPO = Path("/workspace/third_party_official/TransPrune")
CANONICAL_PYTHON = Path("/workspace/.followup_venv/bin/python")
METHODS = {
    "official_visionzip": {
        "python": Path("/tmp/vlm_official_visionzip_env/bin/python"),
        "repo": VISIONZIP_REPO,
        "commit": "8f86b55c6f000eb033e6912538af2dd7dcb30502",
        "pythonpath": [VISIONZIP_LLAVA_REPO, VISIONZIP_REPO, ROOT],
    },
    "official_transprune": {
        "python": Path("/tmp/vlm_official_transprune_env/bin/python"),
        "repo": TRANSPRUNE_REPO,
        "commit": "c9e8284f5808f7692c40590ffc99537e9220a01d",
        "pythonpath": [TRANSPRUNE_REPO, ROOT],
    },
}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_original_llava_official_method import expected_visual_trace  # noqa: E402
from vlm_pruning.artifacts import read_jsonl_strict, sha256_file, validate_completed_run  # noqa: E402
from vlm_pruning.final_campaign import DATASETS  # noqa: E402
from vlm_pruning.gpu_scheduler import admissible_widths  # noqa: E402


@dataclass(frozen=True)
class Cell:
    dataset: str
    task: str
    split: str
    manifest: Path
    expected_rows: int
    method: str
    subset_note: str | None = None

    @property
    def run_id(self) -> str:
        return f"{self.dataset}__{self.method}"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def single_snapshot(model_prefix: str) -> Path:
    roots = sorted((ORIGINAL_CACHE / model_prefix / "snapshots").glob("*"))
    if len(roots) != 1 or not roots[0].is_dir():
        raise RuntimeError(f"Expected one complete snapshot for {model_prefix}, found {roots}")
    return roots[0]


def vision_snapshot() -> Path:
    snapshot = single_snapshot("models--openai--clip-vit-large-patch14-336")
    if snapshot.name != VISION_REVISION:
        raise RuntimeError(f"CLIP snapshot revision {snapshot.name} != locked {VISION_REVISION}")
    return snapshot


def official_env(method: str) -> dict[str, str]:
    info = METHODS[method]
    env = os.environ.copy()
    prior = env.get("PYTHONPATH")
    pieces = [str(path) for path in info["pythonpath"]]
    if prior:
        pieces.append(prior)
    env.update({
        "PYTHONPATH": os.pathsep.join(pieces),
        "HF_HOME": "/dev/shm/vlm_original_llava_home",
        "HF_HUB_CACHE": str(ORIGINAL_CACHE),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    return env


def prepare_mmmu_single_image_manifest() -> Path:
    source = ROOT / "data/k12_b48_expanded/mmmu_val/manifest.jsonl"
    output = ROOT / "data/k12_b48_expanded/mmmu_val/manifest_single_image.jsonl"
    rows = [row for row in read_jsonl_strict(source) if int(row.get("image_count") or 1) == 1]
    if len(rows) != 857 or len({str(row["sample_id"]) for row in rows}) != 857:
        raise RuntimeError(f"Locked MMMU single-image subset must be 857 unique rows, got {len(rows)}")
    temporary = output.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return output


def cells() -> list[Cell]:
    result = []
    for protocol in DATASETS:
        manifest = ROOT / "data/k12_b48_expanded" / protocol.key / "manifest.jsonl"
        expected = int(protocol.expected_manifest_rows or protocol.expected_rows)
        note = None
        if protocol.key == "mmmu_val":
            manifest = prepare_mmmu_single_image_manifest()
            expected = 857
            note = "Official original-LLaVA method uses prospectively locked single-image MMMU subset (857/900)."
        for method in METHODS:
            result.append(Cell(protocol.key, protocol.task, protocol.split, manifest, expected, method, note))
    return result


def command(cell: Cell, phase: str, limit: int | None) -> list[str]:
    info = METHODS[cell.method]
    root = OUTPUT_ROOT if phase == "full" else SMOKE_ROOT / phase
    name = cell.run_id if phase == "full" else f"{phase}_{cell.run_id}"
    cmd = [
        str(info["python"]), "scripts/run_original_llava_official_method.py",
        "--dataset", cell.dataset, "--task", cell.task,
        "--manifest", str(cell.manifest.relative_to(ROOT)), "--split", cell.split,
        "--method", cell.method, "--model-path", str(ORIGINAL_SNAPSHOT),
        "--output-root", str(root.relative_to(ROOT)), "--run-name", name,
        "--prompt-profile", "registered", "--resume", "--skip-existing",
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return cmd


def run_dir(cell: Cell, phase: str) -> Path:
    root = OUTPUT_ROOT if phase == "full" else SMOKE_ROOT / phase
    name = cell.run_id if phase == "full" else f"{phase}_{cell.run_id}"
    return root / name


def expected_rows(cell: Cell, limit: int | None) -> list[dict[str, Any]]:
    rows = read_jsonl_strict(cell.manifest)
    if len(rows) != cell.expected_rows:
        raise RuntimeError(f"{cell.dataset} manifest rows {len(rows)} != {cell.expected_rows}")
    return rows if limit is None else rows[:limit]


def validate(cell: Cell, phase: str, limit: int | None, *, score: bool) -> dict[str, Any]:
    directory = run_dir(cell, phase)
    rows = expected_rows(cell, limit)
    report = validate_completed_run(
        directory, rows, budget=48, original_tokens=576, task=cell.task, require_retained=False,
    )
    issues = list(report.get("issues") or [])
    if report.get("status") == "pass":
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        expected_config = {
            "source/git_dirty": False,
            "model/name": "liuhaotian/llava-v1.5-7b",
            "model/precision": "official_mixed_fp16_llm_fp32_vision",
            "dataset/name": cell.dataset,
            "dataset/split": cell.split,
            "dataset/manifest_checksum": sha256_file(cell.manifest),
            "dataset/num_samples": len(rows),
            "method/name": cell.method,
            "method/seed": 0,
            "tokens/target_visual_tokens": 48,
            "tokens/actual_visual_tokens": 48,
            "tokens/per_layer_visual_trace": expected_visual_trace(cell.method),
            "published_baseline/commit": METHODS[cell.method]["commit"],
            "published_baseline/official_source_executed": True,
            "limit": limit,
        }
        for key, value in expected_config.items():
            if config.get(key) != value:
                issues.append(f"config {key}: expected {value!r}, got {config.get(key)!r}")
        for row in read_jsonl_strict(directory / "per_sample_results.jsonl"):
            if row.get("layer_visual_token_counts") != expected_visual_trace(cell.method):
                issues.append(f"{row.get('sample_id')}: exact token trace failed")
                break
        if not (directory / "COMPLETED.json").is_file():
            issues.append("completion sentinel missing")
    if not issues and phase == "full" and score:
        scoring = subprocess.run(
            [str(CANONICAL_PYTHON), "scripts/score_k12_b48_official.py", "--run-dir", str(directory)],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if scoring.returncode:
            issues.append(f"official scorer failed ({scoring.returncode}): {scoring.stdout[-4000:]}")
    if phase == "full" and not (directory / "official_metrics.json").is_file():
        issues.append("official_metrics.json missing")
    report["issues"] = issues
    report["status"] = "pass" if not issues else "fail"
    return report


def run_command(cmd: list[str], method: str, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        return subprocess.run(cmd, cwd=ROOT, env=official_env(method), stdout=handle, stderr=subprocess.STDOUT).returncode


def admission() -> dict[str, Any]:
    if not ORIGINAL_SNAPSHOT.is_dir() or not CANONICAL_SUMMARY.is_file():
        raise RuntimeError("Original checkpoint or canonical tensor summary is missing.")
    for method, info in METHODS.items():
        if not info["python"].is_file():
            raise RuntimeError(f"Environment missing for {method}: {info['python']}")
        if git_value(info["repo"], "rev-parse", "HEAD") != info["commit"]:
            raise RuntimeError(f"Official commit changed for {method}")
        if git_value(info["repo"], "status", "--short"):
            raise RuntimeError(f"Official repository dirty for {method}")
    if git_value(VISIONZIP_LLAVA_REPO, "status", "--short"):
        raise RuntimeError("Official LLaVA dependency repository is dirty.")

    comparison = subprocess.run([
        str(METHODS["official_visionzip"]["python"]),
        "scripts/audit_llava_checkpoint_equivalence.py", "compare-original",
        "--canonical-summary", str(CANONICAL_SUMMARY),
        "--original-checkpoint", str(ORIGINAL_SNAPSHOT),
        "--vision-checkpoint", str(vision_snapshot()),
        "--output", str(EQUIVALENCE_REPORT),
    ], cwd=ROOT, env=official_env("official_visionzip"), text=True,
       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if comparison.returncode:
        raise RuntimeError(f"Checkpoint equivalence failed: {comparison.stdout[-4000:]}")

    gqa = next(cell for cell in cells() if cell.dataset == "gqa" and cell.method == "official_visionzip")
    parity_name = "parity_original_llava_full_gqa_one"
    parity_cmd = command(gqa, "admission", 1)
    parity_cmd[parity_cmd.index("official_visionzip")] = "official_llava_full"
    parity_cmd[parity_cmd.index("admission_gqa__official_visionzip")] = parity_name
    if run_command(parity_cmd, "official_visionzip", CAMPAIGN / "logs/admission_original_full.log"):
        raise RuntimeError("Original LLaVA full parity execution failed.")
    parity_dir = SMOKE_ROOT / "admission" / parity_name
    canonical_dir = CAMPAIGN / "smoke/official_methods/parity_canonical_full_gqa_one"
    original_row = read_jsonl_strict(parity_dir / "per_sample_results.jsonl")[0]
    canonical_row = read_jsonl_strict(canonical_dir / "per_sample_results.jsonl")[0]
    if original_row["raw_model_output"] != canonical_row["raw_model_output"]:
        raise RuntimeError(
            f"Original/HF deterministic output parity failed: {original_row['raw_model_output']!r} != {canonical_row['raw_model_output']!r}"
        )

    method_evidence = {}
    for method in METHODS:
        cell = next(cell for cell in cells() if cell.dataset == "gqa" and cell.method == method)
        code = run_command(command(cell, "admission", 1), method, CAMPAIGN / f"logs/admission_{method}.log")
        if code:
            raise RuntimeError(f"Official minimal execution failed for {method} (exit {code}).")
        report = validate(cell, "admission", 1, score=False)
        if report["status"] != "pass":
            raise RuntimeError(f"Admission validation failed for {method}: {report['issues']}")
        method_evidence[method] = {
            "status": "matched_b48_eligible",
            "artifact": str(run_dir(cell, "admission").relative_to(ROOT)),
            "trace": expected_visual_trace(method),
        }
    payload = {
        "status": "pass",
        "checked_at_utc": utc_now(),
        "tensor_equivalence": json.loads(EQUIVALENCE_REPORT.read_text(encoding="utf-8")),
        "deterministic_full_parity": {
            "status": "pass",
            "canonical_artifact": str(canonical_dir.relative_to(ROOT)),
            "original_artifact": str(parity_dir.relative_to(ROOT)),
            "raw_output": original_row["raw_model_output"],
        },
        "methods": method_evidence,
    }
    write_json(CAMPAIGN / "original_official_admission_report.json", payload)
    return payload


def register() -> dict[str, Any]:
    gate = json.loads((CAMPAIGN / "original_official_admission_report.json").read_text(encoding="utf-8"))
    if gate.get("status") != "pass":
        raise RuntimeError("Official original-LLaVA admission must pass before registration.")
    rows = []
    for cell in cells():
        expected_rows(cell, None)
        rows.append({
            "run_id": cell.run_id,
            "dataset": cell.dataset,
            "split": cell.split,
            "method": cell.method,
            "manifest": str(cell.manifest.relative_to(ROOT)),
            "manifest_sha256": sha256_file(cell.manifest),
            "expected_rows": cell.expected_rows,
            "subset_note": cell.subset_note,
            "final_visual_tokens": 48,
            "command": command(cell, "full", None),
            "shell": shlex.join(command(cell, "full", None)),
        })
    payload = {"status": "registered", "registered_at_utc": utc_now(), "cells": rows}
    write_json(CAMPAIGN / "original_official_run_manifest.json", payload)
    return {"status": "registered", "cells": len(rows)}


def run_processes(jobs: list[tuple[Cell, str, int | None]], width: int) -> list[dict[str, Any]]:
    pending, active, completed = list(jobs), [], []
    while pending or active:
        while pending and len(active) < width:
            cell, phase, limit = pending.pop(0)
            log = CAMPAIGN / "logs" / f"{phase}_{cell.run_id}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a", encoding="utf-8")
            process = subprocess.Popen(command(cell, phase, limit), cwd=ROOT, env=official_env(cell.method),
                                       text=True, stdout=handle, stderr=subprocess.STDOUT)
            active.append((process, cell, phase, limit, handle, time.time()))
        time.sleep(2)
        remaining = []
        for process, cell, phase, limit, handle, started in active:
            code = process.poll()
            if code is None:
                remaining.append((process, cell, phase, limit, handle, started))
                continue
            handle.close()
            report = validate(cell, phase, limit, score=phase == "full") if code == 0 else {
                "status": "fail", "issues": [f"process exited {code}"],
            }
            completed.append({"run_id": cell.run_id, "phase": phase, "returncode": code,
                              "duration_s": time.time() - started, "validation": report})
        active = remaining
    return completed


def serial_smoke(limit: int) -> dict[str, Any]:
    results = run_processes([(cell, "serial", limit) for cell in cells()], 1)
    payload = {"status": "pass" if all(x["validation"]["status"] == "pass" for x in results) else "fail",
               "created_at_utc": utc_now(), "limit": limit, "runs": results}
    write_json(CAMPAIGN / "original_official_serial_smoke.json", payload)
    return payload


def gpu_snapshot() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    except Exception as exc:
        return str(exc)


def gpu_total_memory_mb() -> float:
    values = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True,
    ).strip().splitlines()
    if len(values) != 1:
        raise RuntimeError(f"Expected one campaign GPU, got {values}")
    return float(values[0])


def gpu_free_memory_mb() -> float:
    values = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True,
    ).strip().splitlines()
    if len(values) != 1:
        raise RuntimeError(f"Expected one campaign GPU, got {values}")
    return float(values[0])


def serial_peak_memory_mb() -> float:
    peaks = []
    for cell in cells():
        path = run_dir(cell, "serial") / "metrics.json"
        if not path.is_file():
            raise RuntimeError(f"Missing serial memory evidence: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        peak = metrics.get("efficiency/peak_gpu_memory_reserved_mb")
        if peak is None:
            peak = metrics.get("efficiency/peak_gpu_memory_mb")
        if peak is None:
            raise RuntimeError(f"Serial peak memory missing: {path}")
        peaks.append(float(peak))
    return max(peaks)


def soak(limit: int) -> dict[str, Any]:
    representatives = [cell for cell in cells() if cell.dataset in {"gqa", "mme", "textvqa_val"}]
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
        results = run_processes([(cell, f"soak{width}", limit) for cell in representatives[:width]], width)
        passed = all(x["validation"]["status"] == "pass" for x in results)
        attempts.append({"width": width, "before": before, "after": gpu_snapshot(), "passed": passed, "runs": results})
        if passed:
            safe = width
            break
    payload = {"status": "pass" if safe else "fail", "safe_parallelism": safe,
               "created_at_utc": utc_now(), "limit": limit, "attempts": attempts,
               "memory_gate": {"gpu_total_mb": total_mb, "serial_peak_reserved_mb": peak_mb,
                               "required_headroom_mb": headroom_mb, "admissible_widths": widths,
                               "skipped_without_launch": skipped}}
    write_json(CAMPAIGN / "original_official_gpu_soak.json", payload)
    return payload


def full(max_parallel: int | None) -> dict[str, Any]:
    smoke = json.loads((CAMPAIGN / "original_official_serial_smoke.json").read_text(encoding="utf-8"))
    soak_report = json.loads((CAMPAIGN / "original_official_gpu_soak.json").read_text(encoding="utf-8"))
    if smoke.get("status") != "pass" or soak_report.get("status") != "pass":
        raise RuntimeError("Serial smoke and GPU soak must pass before full official execution.")
    width = int(soak_report["safe_parallelism"])
    if max_parallel is not None:
        width = min(width, max_parallel)
    gate = soak_report["memory_gate"]
    required_mb = width * float(gate["serial_peak_reserved_mb"]) + float(gate["required_headroom_mb"])
    free_mb = gpu_free_memory_mb()
    if free_mb < required_mb:
        raise RuntimeError(f"Current GPU free memory {free_mb} MiB < required {required_mb} MiB")
    results = run_processes([(cell, "full", None) for cell in cells()], width)
    payload = {"status": "pass" if all(x["validation"]["status"] == "pass" for x in results) else "fail",
               "created_at_utc": utc_now(), "parallelism": width, "runs": results}
    write_json(CAMPAIGN / "original_official_completion.json", payload)
    return payload


def status() -> dict[str, Any]:
    rows = []
    for cell in cells():
        directory = run_dir(cell, "full")
        state = "missing"
        if (directory / "per_sample_results.jsonl").is_file():
            try:
                state = validate(cell, "full", None, score=False)["status"]
            except Exception as exc:
                state = f"fail:{exc}"
        elif (directory / "per_sample_results.partial.jsonl").is_file():
            state = f"partial:{sum(bool(line.strip()) for line in (directory / 'per_sample_results.partial.jsonl').open())}"
        rows.append({"run_id": cell.run_id, "state": state})
    return {"status": "pass", "runs": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("admit")
    sub.add_parser("register")
    smoke = sub.add_parser("serial-smoke")
    smoke.add_argument("--limit", type=int, default=2)
    soak_parser = sub.add_parser("soak")
    soak_parser.add_argument("--limit", type=int, default=20)
    full_parser = sub.add_parser("full")
    full_parser.add_argument("--max-parallel", type=int, choices=[1, 2, 3])
    sub.add_parser("status")
    args = parser.parse_args()
    try:
        if args.action == "admit":
            payload = admission()
        elif args.action == "register":
            payload = register()
        elif args.action == "serial-smoke":
            payload = serial_smoke(args.limit)
        elif args.action == "soak":
            payload = soak(args.limit)
        elif args.action == "full":
            payload = full(args.max_parallel)
        else:
            payload = status()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload.get("status") in {"pass", "registered"} else 2
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
