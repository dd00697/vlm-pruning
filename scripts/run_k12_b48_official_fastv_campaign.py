#!/usr/bin/env python3
"""Gate, register, smoke, and execute the official FastV exact-B48 campaign."""

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
OUTPUT_ROOT = CAMPAIGN / "runs/official_fastv"
SMOKE_ROOT = CAMPAIGN / "smoke/official_fastv_campaign"
OFFICIAL_REPO = ROOT / "third_party/FastV"
OFFICIAL_SOURCE = OFFICIAL_REPO / "src/FastV/llava-hf/transformers/src"
OFFICIAL_PYTHON = Path("/tmp/vlm_official_fastv_env/bin/python")
CANONICAL_PYTHON = Path("/workspace/.followup_venv/bin/python")
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MODEL_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
MODEL_SNAPSHOT = (
    Path("/workspace/cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots")
    / MODEL_REVISION
)
FASTV_COMMIT = "d1659729b5bf1be225e99ee15783deeea80f63b1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_pruning.artifacts import read_jsonl_strict, sha256_file, validate_completed_run  # noqa: E402
from vlm_pruning.final_campaign import DATASETS  # noqa: E402
from vlm_pruning.gpu_scheduler import admissible_widths  # noqa: E402


@dataclass(frozen=True)
class Cell:
    key: str
    task: str
    split: str
    manifest: Path
    expected_rows: int
    subset_note: str | None = None

    @property
    def run_id(self) -> str:
        return f"{self.key}__official_fastv"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def official_env() -> dict[str, str]:
    env = os.environ.copy()
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(OFFICIAL_SOURCE) + (os.pathsep + prior if prior else "")
    env.update({
        "VLM_LOCAL_MODEL_PATH": str(MODEL_SNAPSHOT),
        "TRANSFORMERS_VERBOSITY": "error",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    return env


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def prepare_mmmu_single_image_manifest() -> Path:
    source = ROOT / "data/k12_b48_expanded/mmmu_val/manifest.jsonl"
    output = ROOT / "data/k12_b48_expanded/mmmu_val/manifest_single_image.jsonl"
    rows = [row for row in read_jsonl_strict(source) if int(row.get("image_count") or 1) == 1]
    if len(rows) != 857 or len({str(row["sample_id"]) for row in rows}) != 857:
        raise RuntimeError(f"Locked MMMU one-image subset must contain 857 unique rows, found {len(rows)}.")
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
            note = "Official FastV is single-image only; prospectively locked MMMU one-image subset (857/900)."
        result.append(Cell(protocol.key, protocol.task, protocol.split, manifest, expected, note))
    return result


def command(cell: Cell, phase: str, limit: int | None = None) -> list[str]:
    root = OUTPUT_ROOT if phase == "full" else SMOKE_ROOT / phase
    name = cell.run_id if phase == "full" else f"{phase}_{cell.run_id}"
    cmd = [
        str(OFFICIAL_PYTHON), "scripts/run_official_baseline.py",
        "--dataset", cell.key, "--task", cell.task,
        "--manifest", str(cell.manifest.relative_to(ROOT)), "--split", cell.split,
        "--method", "official_fastv", "--target-visual-tokens", "48",
        "--seed", "0", "--precision", "bf16", "--model-path", MODEL_ID,
        "--run-name", name, "--output-root", str(root.relative_to(ROOT)),
        "--figures-root", str((CAMPAIGN / "figures").relative_to(ROOT)),
        "--tables-root", str((CAMPAIGN / "tables").relative_to(ROOT)),
        "--reports-root", str((CAMPAIGN / "reports").relative_to(ROOT)),
        "--max-figures", "0", "--position-mode", "gap_preserving",
        "--prompt-profile", "registered", "--resume", "--skip-existing", "--no-wandb",
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
        raise RuntimeError(f"{cell.key} manifest has {len(rows)} rows; expected {cell.expected_rows}.")
    return rows if limit is None else rows[:limit]


def validate(cell: Cell, phase: str, limit: int | None, score: bool = True) -> dict[str, Any]:
    directory = run_dir(cell, phase)
    rows = expected_rows(cell, limit)
    report = validate_completed_run(
        directory, rows, budget=48, original_tokens=576, task=cell.task, require_retained=True,
    )
    issues = list(report.get("issues") or [])
    if report.get("status") == "pass":
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        environment = json.loads((directory / "run_environment.json").read_text(encoding="utf-8"))
        expected_config = {
            "source/git_dirty": False,
            "model/name": MODEL_ID,
            "model/precision": "bf16",
            "dataset/name": cell.key,
            "dataset/split": cell.split,
            "dataset/manifest_checksum": sha256_file(cell.manifest),
            "dataset/num_samples": len(rows),
            "method/name": "official_fastv",
            "method/seed": 0,
            "tokens/target_visual_tokens": 48,
            "tokens/actual_visual_tokens": 48,
            "published_baseline/commit": FASTV_COMMIT,
            "published_baseline/official_source_executed": True,
            "published_baseline/fastv_k": 3,
            "limit": limit,
        }
        for key, value in expected_config.items():
            if config.get(key) != value:
                issues.append(f"config {key}: expected {value!r}, got {config.get(key)!r}")
        source_path = str(environment.get("transformers_source") or "")
        if not bool(environment.get("fastv_transformers_source")) or str(OFFICIAL_SOURCE) not in source_path:
            issues.append(f"official FastV Transformers source was not active: {source_path}")
        exact_trace = [576, 576, 576] + [48] * 29
        for sample in read_jsonl_strict(directory / "per_sample_results.jsonl"):
            retained = sample.get("retained_indices") or []
            if sample.get("per_layer_visual_tokens") != exact_trace:
                issues.append(f"{sample.get('sample_id')}: non-exact layer trace")
            if len(retained) != 48 or len(set(retained)) != 48:
                issues.append(f"{sample.get('sample_id')}: invalid observed official selection")
            if issues and len(issues) >= 20:
                break
    if not issues and phase == "full" and score:
        scoring = subprocess.run(
            [str(CANONICAL_PYTHON), "scripts/score_k12_b48_official.py", "--run-dir", str(directory)],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if scoring.returncode:
            issues.append(f"official scorer failed ({scoring.returncode}): {scoring.stdout[-4000:]}")
    if phase == "full" and not (directory / "official_metrics.json").is_file():
        issues.append("official_metrics.json is missing")
    report["issues"] = issues
    report["status"] = "pass" if not issues else "fail"
    return report


def admission_gate() -> dict[str, Any]:
    if git_value(OFFICIAL_REPO, "rev-parse", "HEAD") != FASTV_COMMIT:
        raise RuntimeError("FastV official repository commit changed.")
    if git_value(OFFICIAL_REPO, "status", "--short"):
        raise RuntimeError("FastV official repository is dirty.")
    if not OFFICIAL_PYTHON.is_file() or not MODEL_SNAPSHOT.is_dir():
        raise RuntimeError("Official FastV environment or canonical checkpoint is missing.")
    evidence = CAMPAIGN / "smoke/official_methods"
    active = evidence / "admission_official_fastv_gqa_one"
    canonical = evidence / "parity_canonical_full_gqa_one"
    disabled = evidence / "parity_official_fastv_disabled_gqa_one"
    for directory in (active, canonical, disabled):
        if json.loads((directory / "integrity_report.json").read_text())["status"] != "pass":
            raise RuntimeError(f"Admission evidence did not pass: {directory}")
        if json.loads((directory / "config.json").read_text()).get("source/git_dirty") is not False:
            raise RuntimeError(f"Admission evidence was produced from a dirty harness: {directory}")
    canonical_row = read_jsonl_strict(canonical / "per_sample_results.jsonl")[0]
    disabled_row = read_jsonl_strict(disabled / "per_sample_results.jsonl")[0]
    active_row = read_jsonl_strict(active / "per_sample_results.jsonl")[0]
    parity = canonical_row["raw_model_output"] == disabled_row["raw_model_output"]
    exact = active_row.get("per_layer_visual_tokens") == [576, 576, 576] + [48] * 29
    if not parity or not exact:
        raise RuntimeError(f"FastV admission failed: disabled parity={parity}, exact B48={exact}.")
    freeze = subprocess.check_output([str(OFFICIAL_PYTHON), "-m", "pip", "freeze", "--all"], text=True).splitlines()
    payload = {
        "status": "pass", "checked_at_utc": utc_now(), "official_repo": str(OFFICIAL_REPO),
        "official_commit": FASTV_COMMIT, "official_repo_clean": True,
        "canonical_model": MODEL_ID, "canonical_revision": MODEL_REVISION,
        "minimal_example": {"status": "pass", "artifact": str(active.relative_to(ROOT))},
        "exact_b48": {"status": "pass", "trace": active_row["per_layer_visual_tokens"]},
        "disabled_method_parity": {
            "status": "pass", "canonical_artifact": str(canonical.relative_to(ROOT)),
            "official_disabled_artifact": str(disabled.relative_to(ROOT)),
            "raw_output": canonical_row["raw_model_output"],
        },
        "environment": {"python": str(OFFICIAL_PYTHON), "pip_freeze": freeze},
    }
    write_json(CAMPAIGN / "fastv_admission_report.json", payload)
    return payload


def register() -> dict[str, Any]:
    gate = admission_gate()
    rows = []
    for cell in cells():
        expected_rows(cell, None)
        rows.append({
            "run_id": cell.run_id, "dataset": cell.key, "split": cell.split,
            "manifest": str(cell.manifest.relative_to(ROOT)), "manifest_sha256": sha256_file(cell.manifest),
            "expected_rows": cell.expected_rows, "subset_note": cell.subset_note,
            "method": "official_fastv", "final_visual_tokens": 48,
            "command": command(cell, "full"), "shell": shlex.join(command(cell, "full")),
        })
    payload = {"status": "registered", "registered_at_utc": utc_now(), "admission": gate, "cells": rows}
    write_json(CAMPAIGN / "official_fastv_run_manifest.json", payload)
    return {"status": "registered", "cells": len(rows)}


def run_processes(jobs: list[tuple[Cell, str, int | None]], width: int) -> list[dict[str, Any]]:
    pending, active, completed = list(jobs), [], []
    log_root = CAMPAIGN / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    while pending or active:
        while pending and len(active) < width:
            cell, phase, limit = pending.pop(0)
            handle = (log_root / f"{phase}_{cell.run_id}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(
                command(cell, phase, limit), cwd=ROOT, env=official_env(),
                text=True, stdout=handle, stderr=subprocess.STDOUT,
            )
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
            completed.append({
                "run_id": cell.run_id, "phase": phase, "returncode": code,
                "duration_s": time.time() - started, "validation": report,
            })
        active = remaining
    return completed


def serial_smoke(limit: int) -> dict[str, Any]:
    results = run_processes([(cell, "serial", limit) for cell in cells()], 1)
    payload = {"status": "pass" if all(x["validation"]["status"] == "pass" for x in results) else "fail",
               "created_at_utc": utc_now(), "limit": limit, "runs": results}
    write_json(CAMPAIGN / "official_fastv_serial_smoke.json", payload)
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
    representatives = [cell for cell in cells() if cell.key in {"gqa", "mme", "textvqa_val"}]
    attempts, safe = [], 0
    total_mb = gpu_total_memory_mb()
    peak_mb = serial_peak_memory_mb()
    widths, headroom_mb = admissible_widths(total_mb, peak_mb)
    skipped = [width for width in (3, 2, 1) if width not in widths]
    for width in widths:
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
    write_json(CAMPAIGN / "official_fastv_gpu_soak.json", payload)
    return payload


def full(max_parallel: int | None) -> dict[str, Any]:
    smoke = json.loads((CAMPAIGN / "official_fastv_serial_smoke.json").read_text())
    soak_report = json.loads((CAMPAIGN / "official_fastv_gpu_soak.json").read_text())
    if smoke.get("status") != "pass" or soak_report.get("status") != "pass":
        raise RuntimeError("Official FastV serial smoke and GPU soak must pass before full execution.")
    width = int(soak_report["safe_parallelism"])
    if max_parallel is not None:
        width = min(width, max_parallel)
    results = run_processes([(cell, "full", None) for cell in cells()], width)
    payload = {"status": "pass" if all(x["validation"]["status"] == "pass" for x in results) else "fail",
               "created_at_utc": utc_now(), "parallelism": width, "runs": results}
    write_json(CAMPAIGN / "official_fastv_completion.json", payload)
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
    sub.add_parser("gate")
    sub.add_parser("register")
    smoke = sub.add_parser("serial-smoke"); smoke.add_argument("--limit", type=int, default=1)
    soaking = sub.add_parser("soak"); soaking.add_argument("--limit", type=int, default=10)
    full_parser = sub.add_parser("full"); full_parser.add_argument("--max-parallel", type=int, choices=(1, 2, 3))
    sub.add_parser("status")
    args = parser.parse_args()
    if args.action == "gate": payload = admission_gate()
    elif args.action == "register": payload = register()
    elif args.action == "serial-smoke": payload = serial_smoke(args.limit)
    elif args.action == "soak": payload = soak(args.limit)
    elif args.action == "full": payload = full(args.max_parallel)
    else: payload = status()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") in {"pass", "registered"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
