#!/usr/bin/env python
"""Resumable orchestration for the registered final dissertation evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = Path("experiments/final_evaluation")
UG_RUN_ROOT = FINAL_ROOT / "runs" / "ug"
VAL_MANIFEST = "data/manifests/final_evaluation/refcoco_val_unique.jsonl"
EXPECTED_VAL_ROWS = 8811
FACTORIAL = (
    ("query_max", "more_context"),
    ("query_max", "more_anchors"),
    ("query_pooled", "more_context"),
    ("query_pooled", "more_anchors"),
)


@dataclass(frozen=True)
class RunEntry:
    id: str
    phase: str
    expected_rows: int
    run_dir: str
    command: tuple[str, ...]

    def record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = shlex.join(self.command)
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["soak", "selection"], required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "vlm-pruning"))
    parser.add_argument("--logs-root", type=Path)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_event(root: Path, payload: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": utc_now(), **payload}, sort_keys=True) + "\n")


def ug_command(
    *,
    name: str,
    map_source: str,
    role: str,
    output_root: Path,
    limit: int | None,
    args: argparse.Namespace,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "scripts/run_ug_prune_single_pass.py",
        "--dataset",
        "refcoco",
        "--task",
        "grounding",
        "--manifest",
        VAL_MANIFEST,
        "--split",
        "val",
        "--target-visual-tokens",
        "48",
        "--prune-after-layer",
        "20",
        "--position-mode",
        "gap_preserving",
        "--decode-mode",
        "layer_aware_cache",
        "--diagnostics-mode",
        "full",
        "--ug-role-preset",
        role,
        "--ug-map-source",
        map_source,
        "--ug-fusion-mode",
        "single_layer",
        "--precision",
        "bf16",
        "--run-name",
        name,
        "--output-root",
        str(output_root),
        "--figures-root",
        str(FINAL_ROOT / "figures"),
        "--tables-root",
        str(FINAL_ROOT / "tables"),
        "--max-figures",
        "0",
        "--resume",
        "--skip-existing",
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if args.wandb_mode == "disabled" or limit is not None:
        command.append("--no-wandb")
    else:
        command.extend(
            [
                "--wandb",
                "--wandb-project",
                args.wandb_project,
                "--wandb-mode",
                args.wandb_mode,
                "--wandb-group",
                "dissertation-final-selection-20260729",
                "--wandb-job-type",
                "final-method-selection",
                "--wandb-tags",
                f"dissertation,final,selection,refcoco,k20,b48,{map_source},{role}",
            ]
        )
    return tuple(command)


def entries_for(args: argparse.Namespace) -> list[RunEntry]:
    if args.phase == "soak":
        output_root = FINAL_ROOT / "soak" / "ug"
        return [
            RunEntry(
                id=f"final_soak_k20_b48_{map_source}_{role}",
                phase="soak",
                expected_rows=20,
                run_dir=str(output_root / f"final_soak_k20_b48_{map_source}_{role}"),
                command=ug_command(
                    name=f"final_soak_k20_b48_{map_source}_{role}",
                    map_source=map_source,
                    role=role,
                    output_root=output_root,
                    limit=20,
                    args=args,
                ),
            )
            for map_source, role in FACTORIAL[:3]
        ]
    return [
        RunEntry(
            id=f"final_selection_refcoco_val_k20_b48_{map_source}_{role}_bf16",
            phase="selection",
            expected_rows=EXPECTED_VAL_ROWS,
            run_dir=str(UG_RUN_ROOT / f"final_selection_refcoco_val_k20_b48_{map_source}_{role}_bf16"),
            command=ug_command(
                name=f"final_selection_refcoco_val_k20_b48_{map_source}_{role}_bf16",
                map_source=map_source,
                role=role,
                output_root=UG_RUN_ROOT,
                limit=None,
                args=args,
            ),
        )
        for map_source, role in FACTORIAL
    ]


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                return []
    return rows


def complete(entry: RunEntry) -> bool:
    run_dir = Path(entry.run_dir)
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    rows = read_rows(run_dir / "per_sample_results.jsonl")
    if not metrics_path.exists() or not config_path.exists() or len(rows) != entry.expected_rows:
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    ids = [str(row.get("sample_id")) for row in rows]
    manifest_rows = read_rows(Path(str(config.get("dataset/manifest") or "")))
    manifest_ids = [str(row.get("sample_id")) for row in manifest_rows[: entry.expected_rows]]
    exact_budget = all(
        int(row.get("target_visual_tokens", -1)) == 48
        and int(row.get("actual_llm_visual_tokens", -1)) == 48
        and len(row.get("retained_indices") or []) == 48
        and float(row.get("selection/exact_budget") or 0.0) == 1.0
        for row in rows
    )
    manifest_path = Path(VAL_MANIFEST)
    recorded_checksum = str(config.get("dataset/manifest_checksum") or "")
    manifest_checksum_ok = False
    if manifest_path.exists() and recorded_checksum:
        import hashlib

        manifest_checksum_ok = hashlib.sha256(manifest_path.read_bytes()).hexdigest().startswith(
            recorded_checksum
        )
    return all(
        (
            int(metrics.get("samples/evaluated", -1)) == entry.expected_rows,
            len(ids) == len(set(ids)),
            ids == manifest_ids,
            exact_budget,
            config.get("dataset/manifest") == VAL_MANIFEST,
            manifest_checksum_ok,
            int(config.get("ug/prune_after_layer", -1)) == 20,
            int(config.get("tokens/target_visual_tokens", -1)) == 48,
            config.get("ug/fusion_mode") == "single_layer",
            config.get("position_mode") == "gap_preserving",
            config.get("model/name") == "llava-hf/llava-1.5-7b-hf",
            config.get("model/precision") == "bf16",
        )
    )


def gpu_memory() -> dict[str, int] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()[0]
        total, used, free = [int(item.strip()) for item in output.split(",")]
        return {"total_mib": total, "used_mib": used, "free_mib": free}
    except Exception:
        return None


def verify_soak(entries: list[RunEntry], logs_root: Path, minimum_free_mib: int = 2048) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for entry in entries:
        suffix = entry.id.removeprefix("final_soak_")
        serial_dir = FINAL_ROOT / "smoke" / "ug" / f"final_smoke_{suffix}"
        serial = read_rows(serial_dir / "per_sample_results.jsonl")
        concurrent = read_rows(Path(entry.run_dir) / "per_sample_results.jsonl")
        serial_signature = [
            (
                row.get("sample_id"),
                row.get("raw_model_output"),
                row.get("retained_indices_checksum"),
                row.get("box_iou"),
            )
            for row in serial
        ]
        concurrent_signature = [
            (
                row.get("sample_id"),
                row.get("raw_model_output"),
                row.get("retained_indices_checksum"),
                row.get("box_iou"),
            )
            for row in concurrent
        ]
        comparisons.append(
            {
                "id": entry.id,
                "serial_rows": len(serial),
                "concurrent_rows": len(concurrent),
                "identical": serial_signature == concurrent_signature and len(serial_signature) == 20,
            }
        )
    samples_path = logs_root / "gpu_memory.jsonl"
    samples = read_rows(samples_path)
    min_free = min((int(row["free_mib"]) for row in samples if row.get("free_mib") is not None), default=-1)
    passed = all(item["identical"] for item in comparisons) and min_free >= minimum_free_mib
    report = {
        "passed": passed,
        "minimum_required_free_mib": minimum_free_mib,
        "observed_minimum_free_mib": min_free,
        "comparisons": comparisons,
        "recommended_quality_parallelism": 3 if passed else 1,
    }
    write_json(logs_root / "soak_report.json", report)
    return report


def run(entries: list[RunEntry], args: argparse.Namespace, logs_root: Path) -> None:
    state_path = logs_root / "state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"phase": args.phase, "runs": {}}
    )
    rows = state.setdefault("runs", {})
    for entry in entries:
        row = rows.setdefault(entry.id, {})
        row.update(
            {
                "phase": entry.phase,
                "expected_rows": entry.expected_rows,
                "run_dir": entry.run_dir,
                "command": shlex.join(entry.command),
            }
        )
        row.setdefault("attempts", 0)
        if complete(entry):
            row["status"] = "complete"
    write_json(state_path, state)

    pending = [entry for entry in entries if rows[entry.id].get("status") != "complete"]
    active: dict[str, tuple[RunEntry, subprocess.Popen[Any], Any, Path]] = {}
    while pending or active:
        while pending and len(active) < max(1, int(args.max_parallel)):
            entry = pending.pop(0)
            row = rows[entry.id]
            row["attempts"] = int(row.get("attempts", 0)) + 1
            row["status"] = "running"
            row["started_at"] = utc_now()
            log_path = logs_root / f"{entry.id}.attempt{row['attempts']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            environment = dict(os.environ)
            environment.setdefault("HF_HOME", "/workspace/cache/huggingface")
            environment["HF_HUB_DISABLE_XET"] = "1"
            process = subprocess.Popen(
                entry.command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active[entry.id] = (entry, process, handle, log_path)
            append_event(
                logs_root,
                {
                    "event": "start",
                    "id": entry.id,
                    "attempt": row["attempts"],
                    "pid": process.pid,
                    "command": shlex.join(entry.command),
                },
            )
            write_json(state_path, state)

        memory = gpu_memory()
        if memory is not None:
            with (logs_root / "gpu_memory.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"time": utc_now(), **memory}, sort_keys=True) + "\n")
        if not active:
            break
        time.sleep(max(1.0, float(args.poll_seconds)))
        for run_id, (entry, process, handle, log_path) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            row = rows[run_id]
            row["finished_at"] = utc_now()
            row["returncode"] = code
            row["last_log"] = str(log_path)
            if code == 0 and complete(entry):
                row["status"] = "complete"
                append_event(logs_root, {"event": "complete", "id": run_id})
            elif int(row["attempts"]) < int(args.max_attempts):
                row["status"] = "retry_pending"
                pending.append(entry)
                append_event(logs_root, {"event": "retry", "id": run_id, "returncode": code})
            else:
                row["status"] = "failed"
                append_event(logs_root, {"event": "failed", "id": run_id, "returncode": code})
            del active[run_id]
            write_json(state_path, state)

    failures = [run_id for run_id, row in rows.items() if row.get("status") == "failed"]
    write_json(
        logs_root / "summary.json",
        {
            "phase": args.phase,
            "registered": len(entries),
            "complete": sum(row.get("status") == "complete" for row in rows.values()),
            "failed": failures,
            "updated_at": utc_now(),
        },
    )
    if failures:
        raise SystemExit(f"Failed runs: {', '.join(failures)}")


def main() -> int:
    args = parse_args()
    logs_root = args.logs_root or (FINAL_ROOT / "logs" / args.phase)
    entries = entries_for(args)
    logs_root.mkdir(parents=True, exist_ok=True)
    with (logs_root / "commands.jsonl").open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.record(), sort_keys=True) + "\n")
    print(f"Registered {len(entries)} {args.phase} runs in {logs_root}")
    if not args.execute:
        return 0
    run(entries, args, logs_root)
    if args.phase == "soak":
        report = verify_soak(entries, logs_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["passed"]:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
