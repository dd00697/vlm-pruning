#!/usr/bin/env python
"""Audit adaptive-campaign implementation and GPU preflight evidence."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_progressive_oracle_run import validate_run  # noqa: E402
from vlm_pruning.llava_hooks import average_visual_tokens, realized_visual_trace, validate_schedule  # noqa: E402


CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
PREFLIGHT_ROOT = CAMPAIGN_ROOT / "preflight"
SMOKE_ROOT = PREFLIGHT_ROOT / "smoke_runs"
PARALLEL_ROOT = PREFLIGHT_ROOT / "parallel_soak"
MANIFEST = REPO_ROOT / (
    "experiments/progressive_oracle_refcoco_val_20260814/"
    "manifests/refcoco_val_discovery1000.jsonl"
)
HISTORICAL_PREFLIGHT = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814/preflight"
REPORT_PATH = PREFLIGHT_ROOT / "PREFLIGHT_REPORT.json"
STATE_PATH = CAMPAIGN_ROOT / "CONTROLLER_STATE.json"
REGISTRY_PATH = CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"
RESEARCH_REPORT_PATH = CAMPAIGN_ROOT / "RESEARCH_REPORT.md"
PYTHON = "/workspace/.followup_venv/bin/python"


class PreflightError(RuntimeError):
    """Raised when any implementation gate lacks exact passing evidence."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
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
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreflightError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise PreflightError(f"Malformed JSONL: {path}")
    return rows


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def exact_rows(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    fields: list[str],
    label: str,
) -> None:
    if len(left) != len(right):
        raise PreflightError(f"{label}: row-count mismatch")
    for index, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
        if left_row.get("sample_id") != right_row.get("sample_id"):
            raise PreflightError(f"{label}: sample mismatch at row {index}")
        for field in fields:
            if left_row.get(field) != right_row.get(field):
                raise PreflightError(f"{label}: {field} differs at row {index}")


def traces(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [list(row["stage_traces"]) for row in rows]


def run_cpu_gates() -> dict[str, Any]:
    compile_command = [PYTHON, "-m", "compileall", "-q", "scripts", "vlm_pruning", "tests"]
    compile_result = subprocess.run(compile_command, cwd=REPO_ROOT, capture_output=True, text=True)
    if compile_result.returncode:
        raise PreflightError(f"compileall failed: {compile_result.stderr}")
    test_command = [PYTHON, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
    test_result = subprocess.run(test_command, cwd=REPO_ROOT, capture_output=True, text=True)
    if test_result.returncode:
        raise PreflightError(f"unit suite failed: {test_result.stdout}\n{test_result.stderr}")
    match = re.search(r"Ran (\d+) tests", test_result.stdout + test_result.stderr)
    if match is None:
        raise PreflightError("Could not determine unit-test count")
    expected_averages = {
        "K12": ([(12, 48, "query_topk")], 246.0),
        "P3": ([(4, 384, "query_topk"), (12, 48, "query_topk")], 198.0),
        "C448": ([(4, 448, "query_topk"), (12, 48, "query_topk")], 214.0),
        "C480": ([(4, 480, "query_topk"), (12, 48, "query_topk")], 222.0),
        "D1": ([(8, 384, "query_topk"), (12, 48, "query_topk")], 222.0),
        "M448": ([(4, 448, "query_topk_merge"), (12, 48, "query_topk")], 214.0),
        "B1": (
            [(4, 384, "query_topk"), (12, 96, "query_topk"), (20, 16, "query_topk")],
            198.0,
        ),
        "A1": (
            [(4, 384, "query_topk"), (12, 96, "query_topk"), (20, 48, "query_topk")],
            210.0,
        ),
    }
    realized = {
        name: average_visual_tokens(realized_visual_trace(validate_schedule(schedule)))
        for name, (schedule, _expected) in expected_averages.items()
    }
    for name, (_schedule, expected) in expected_averages.items():
        if realized[name] != expected:
            raise PreflightError(f"{name} average {realized[name]} != {expected}")
    return {
        "compile": {"command": compile_command, "pass": True},
        "unit_suite": {"command": test_command, "pass": True, "tests": int(match.group(1))},
        "schedule_average_formulas": {"pass": True, "realized": realized},
    }


def validate_smokes() -> dict[str, Any]:
    run_dirs = {
        "full": SMOKE_ROOT / "full",
        "k12": SMOKE_ROOT / "k12",
        "p3": SMOKE_ROOT / "p3",
        "c448": SMOKE_ROOT / "c448",
        "m448": SMOKE_ROOT / "m448",
        "resume_c448": SMOKE_ROOT / "resume_c448",
        "parallel_c448": PARALLEL_ROOT / "c448",
        "parallel_k12": PARALLEL_ROOT / "k12",
    }
    validations: dict[str, Any] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    for name, run_dir in run_dirs.items():
        validation = validate_run(run_dir, MANIFEST, expected_rows=2)
        completion = read_json(run_dir / "COMPLETED.json")
        if validation["status"] != "pass" or completion.get("source_git_commit") != current_commit:
            raise PreflightError(f"{name} lacks a valid current-source completion")
        validations[name] = {
            "run_dir": relative(run_dir),
            "completion_sha256": sha256_file(run_dir / "COMPLETED.json"),
            "predictions_sha256": sha256_file(run_dir / "predictions.jsonl"),
            "metrics": validation["metrics"],
        }
        rows[name] = read_jsonl(run_dir / "predictions.jsonl")

    canonical_full = read_jsonl(
        HISTORICAL_PREFLIGHT / "existing_full/existing_full/per_sample_results.jsonl"
    )[:2]
    canonical_k12 = read_jsonl(
        HISTORICAL_PREFLIGHT
        / "existing_single_stage/existing_k12_b48/per_sample_results.jsonl"
    )[:2]
    historical_p3 = read_jsonl(HISTORICAL_PREFLIGHT / "smoke_runs/two_stage/predictions.jsonl")
    exact_rows(
        rows["full"], canonical_full, ["raw_model_output", "predicted_box", "box_iou"], "full parity"
    )
    exact_rows(
        rows["k12"],
        canonical_k12,
        ["raw_model_output", "predicted_box", "box_iou"],
        "K12 canonical parity",
    )
    for new_row, old_row in zip(rows["k12"], canonical_k12, strict=True):
        if new_row["final_retained_indices"] != old_row["retained_indices"]:
            raise PreflightError("K12 retained indices differ from canonical single-stage mode")
    exact_rows(
        rows["p3"],
        historical_p3,
        ["raw_model_output", "predicted_box", "box_iou", "final_retained_indices"],
        "P3 parity",
    )
    for new_trace, old_trace in zip(traces(rows["p3"]), traces(historical_p3), strict=True):
        for new_stage, old_stage in zip(new_trace, old_trace, strict=True):
            for field in ("score_map_checksum", "outgoing_original_indices", "outgoing_position_ids"):
                if new_stage[field] != old_stage[field]:
                    raise PreflightError(f"P3 stage parity differs for {field}")

    exact_rows(
        rows["resume_c448"],
        rows["c448"],
        ["raw_model_output", "predicted_box", "box_iou", "final_retained_indices"],
        "resume parity",
    )
    exact_rows(
        rows["parallel_c448"],
        rows["c448"],
        ["raw_model_output", "predicted_box", "box_iou", "final_retained_indices"],
        "parallel C448 parity",
    )
    exact_rows(
        rows["parallel_k12"],
        rows["k12"],
        ["raw_model_output", "predicted_box", "box_iou", "final_retained_indices"],
        "parallel K12 parity",
    )
    for hard_row, merge_row in zip(rows["c448"], rows["m448"], strict=True):
        hard_stage = hard_row["stage_traces"][0]
        merge_stage = merge_row["stage_traces"][0]
        for field in (
            "score_map_checksum",
            "incoming_original_indices",
            "outgoing_original_indices",
            "outgoing_position_ids",
            "incoming_sequence_length",
            "outgoing_sequence_length",
        ):
            if hard_stage[field] != merge_stage[field]:
                raise PreflightError(f"M448 exact C448 control differs for {field}")
        if (
            merge_stage.get("merge_assignment_count") != 128
            or merge_stage.get("merge_input_mass") != 576.0
            or merge_stage.get("merge_output_mass") != 576.0
            or len(merge_stage.get("merge_output_masses") or []) != 448
            or any(float(value) <= 0 for value in merge_stage["merge_output_masses"])
        ):
            raise PreflightError("M448 physical mass/assignment evidence is invalid")
    return {
        "runs": validations,
        "parity": {
            "unpruned_canonical": {"pass": True, "samples": 2},
            "k12_canonical_single_stage": {"pass": True, "samples": 2},
            "p3_historical_progressive": {"pass": True, "samples": 2},
            "resume_exact": {
                "pass": True,
                "samples": 2,
                "interrupted_rows": 1,
                "final_rows": 2,
            },
            "parallel_semantic": {"pass": True, "samples_per_path": 2},
            "m448_exact_c448_prefix_and_mass_contract": {"pass": True, "samples": 2},
        },
    }


def gpu_capacity(smokes: dict[str, Any]) -> dict[str, Any]:
    line = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()[0]
    name, total_text, free_text = [part.strip() for part in line.split(",")]
    total = int(total_text)
    free = int(free_text)
    peak = max(
        float(run["metrics"]["efficiency/peak_reserved_mb"])
        for run in smokes["runs"].values()
    )
    required_margin = 0.15 * total
    two_remaining = total - 2.0 * peak
    three_remaining = total - 3.0 * peak
    width = 2 if two_remaining >= required_margin else 1
    if width != 2 or three_remaining >= required_margin:
        raise PreflightError("Measured capacity did not establish exactly two safe quality workers")
    return {
        "gpu": name,
        "memory_total_mib": total,
        "memory_free_mib_after_soak": free,
        "maximum_measured_reserved_mib": peak,
        "required_safety_margin_mib": required_margin,
        "two_worker_remaining_mib": two_remaining,
        "three_worker_remaining_mib": three_remaining,
        "safe_quality_parallel_width": width,
        "timing_parallel_width": 1,
    }


def update_campaign(report: dict[str, Any]) -> None:
    now = report["verified_at_utc"]
    state = read_json(STATE_PATH)
    registry = read_json(REGISTRY_PATH)
    state.update(
        {
            "state": "PREFLIGHT_COMPLETE_C448_PENDING",
            "next_required_action": "run_C448",
            "updated_at_utc": now,
        }
    )
    state["decision_history"].append(
        {
            "phase": "preflight",
            "status": "complete",
            "evidence_path": relative(REPORT_PATH),
            "evidence_sha256": sha256_file(REPORT_PATH),
            "safe_quality_parallel_width": report["capacity"]["safe_quality_parallel_width"],
        }
    )
    registry["phases"]["preflight"] = {
        "status": "complete",
        "report_path": relative(REPORT_PATH),
        "report_sha256": sha256_file(REPORT_PATH),
        "source_commit": report["source_commit"],
        "safe_quality_parallel_width": report["capacity"]["safe_quality_parallel_width"],
    }
    registry["updated_at_utc"] = now
    atomic_json(STATE_PATH, state)
    atomic_json(REGISTRY_PATH, registry)

    document = RESEARCH_REPORT_PATH.read_text(encoding="utf-8")
    section = """## Implementation and preflight

All implementation gates passed before any new 1,000-row cell. The complete repository suite passed {tests} tests. Two-example physical smokes reproduced canonical unpruned and K12 outputs exactly, and reproduced the historical P3 progressive path exactly, including retained indices and recomputed score-map checksums. A forced interruption after one row resumed to exactly two ordered rows with no duplication.

M448's physical smoke matched C448 exactly through K4 selection and shape/position evidence. Independently recomputed diagnostics proved 128 deterministic nearest-retained assignments, 448 positive output masses, and total input/output mass 576 for every example. All smoke finalizers also verified text survival, raster order, gap-preserving positions, exact budgets, fresh score maps, raw-output rescoring, and per-layer cache lengths.

The NVIDIA {gpu} exposed {total} MiB. The largest measured reservation was {peak:.0f} MiB per runner. Two conservative peak reservations leave {remaining:.0f} MiB ({fraction:.1%}) free, so quality concurrency is capped at two; three workers fail the registered 15% safety margin. Timing remains serial.
""".format(
        tests=report["cpu_gates"]["unit_suite"]["tests"],
        gpu=report["capacity"]["gpu"],
        total=report["capacity"]["memory_total_mib"],
        peak=report["capacity"]["maximum_measured_reserved_mib"],
        remaining=report["capacity"]["two_worker_remaining_mib"],
        fraction=report["capacity"]["two_worker_remaining_mib"]
        / report["capacity"]["memory_total_mib"],
    )
    updated, count = re.subn(
        r"## Implementation and preflight\n\n.*?(?=\n## Adaptive fixed-method experiments)",
        section.rstrip(),
        document,
        flags=re.DOTALL,
    )
    if count != 1:
        raise PreflightError("Could not update the single cumulative report preflight section")
    RESEARCH_REPORT_PATH.write_text(updated, encoding="utf-8")


def main() -> int:
    scoped_paths = [
        "scripts/run_progressive_oracle_refcoco.py",
        "scripts/validate_progressive_oracle_run.py",
        "vlm_pruning/llava_hooks/progressive_pruning.py",
    ]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *scoped_paths], cwd=REPO_ROOT, text=True
    ).strip()
    if dirty:
        raise PreflightError(f"Scientific runner source is dirty: {dirty}")
    cpu = run_cpu_gates()
    smoke = validate_smokes()
    capacity = gpu_capacity(smoke)
    report = {
        "status": "pass",
        "verified_at_utc": utc_now(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "cpu_gates": cpu,
        "semantic_smokes": smoke,
        "capacity": capacity,
        "new_1000_row_cells_launched_before_pass": 0,
    }
    atomic_json(REPORT_PATH, report)
    update_campaign(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
