#!/usr/bin/env python
"""Run the focused RefCOCO ablations requested for the final dissertation."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_pruning.artifacts import validate_completed_run


MANIFEST = "data/manifests/official_baselines/refcoco_val.jsonl"
RUNS_ROOT = Path("experiments/runs_ug_prune_single_pass")
DEFAULT_LOGS_ROOT = Path("experiments/logs/strong_dissertation_ablations_20260728")
EXPECTED_FULL_ROWS = 8811


@dataclass(frozen=True)
class Run:
    name: str
    phase: str
    expected_rows: int
    command: tuple[str, ...]

    @property
    def run_dir(self) -> Path:
        return RUNS_ROOT / self.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--latency-only", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--latency-limit", type=int, default=1000)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "vlm-pruning"))
    parser.add_argument("--wandb-group", default="dissertation-strong-ablations-20260728")
    return parser.parse_args()


def base_command(
    *,
    name: str,
    budget: int,
    prune_layer: int,
    fusion: str = "single_layer",
    probe_layers: str | None = None,
    map_source: str = "query_max",
    role: str = "more_context",
    position_mode: str = "gap_preserving",
    limit: int | None = None,
    wandb_mode: str,
    wandb_project: str,
    wandb_group: str,
    phase: str,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "scripts/run_ug_prune_single_pass.py",
        "--dataset",
        "refcoco",
        "--task",
        "grounding",
        "--manifest",
        MANIFEST,
        "--split",
        "val",
        "--target-visual-tokens",
        str(budget),
        "--precision",
        "bf16",
        "--run-name",
        name,
        "--prune-after-layer",
        str(prune_layer),
        "--position-mode",
        position_mode,
        "--decode-mode",
        "layer_aware_cache",
        "--diagnostics-mode",
        "benchmark",
        "--ug-role-preset",
        role,
        "--ug-map-source",
        map_source,
        "--ug-fusion-mode",
        fusion,
        "--resume",
        "--skip-existing",
    ]
    if probe_layers:
        command.extend(["--ug-probe-layers", probe_layers])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if wandb_mode == "disabled":
        command.append("--no-wandb")
    else:
        command.extend(
            [
                "--wandb",
                "--wandb-project",
                wandb_project,
                "--wandb-mode",
                wandb_mode,
                "--wandb-group",
                wandb_group,
                "--wandb-job-type",
                phase,
                "--wandb-tags",
                f"dissertation,strong-ablation,{phase},refcoco,budget-{budget},layer-{prune_layer}",
            ]
        )
    return tuple(command)


def quality_runs(args: argparse.Namespace) -> list[Run]:
    runs: list[Run] = []

    # Depth-by-budget sweep. L8/B48 already exists as a complete aligned run.
    for layer, budget in (
        (12, 48),
        (16, 48),
        (20, 48),
        (8, 128),
        (12, 128),
        (16, 128),
        (20, 128),
    ):
        name = f"strong_ablation_refcoco_depth_l{layer}_b{budget}_single_layer_bf16"
        runs.append(
            Run(
                name,
                "depth_quality",
                EXPECTED_FULL_ROWS,
                base_command(
                    name=name,
                    budget=budget,
                    prune_layer=layer,
                    wandb_mode=args.wandb_mode,
                    wandb_project=args.wandb_project,
                    wandb_group=args.wandb_group,
                    phase="depth-quality",
                ),
            )
        )

    name = "strong_ablation_refcoco_position_compact_k12_b48_more_context_bf16"
    runs.append(
        Run(
            name,
            "position_quality",
            EXPECTED_FULL_ROWS,
            base_command(
                name=name,
                budget=48,
                prune_layer=12,
                fusion="soft_average",
                position_mode="compact",
                wandb_mode=args.wandb_mode,
                wandb_project=args.wandb_project,
                wandb_group=args.wandb_group,
                phase="position-quality",
            ),
        )
    )

    name = "strong_ablation_refcoco_query_pooled_more_anchors_k12_b128_bf16"
    runs.append(
        Run(
            name,
            "aggregation_allocation_quality",
            EXPECTED_FULL_ROWS,
            base_command(
                name=name,
                budget=128,
                prune_layer=12,
                fusion="soft_average",
                map_source="query_pooled",
                role="more_anchors",
                wandb_mode=args.wandb_mode,
                wandb_project=args.wandb_project,
                wandb_group=args.wandb_group,
                phase="aggregation-allocation-quality",
            ),
        )
    )

    for budget in (48, 128):
        name = f"strong_ablation_refcoco_fusion_8_12_b{budget}_more_context_bf16"
        runs.append(
            Run(
                name,
                "fusion_quality",
                EXPECTED_FULL_ROWS,
                base_command(
                    name=name,
                    budget=budget,
                    prune_layer=12,
                    fusion="soft_average",
                    probe_layers="8,12",
                    wandb_mode=args.wandb_mode,
                    wandb_project=args.wandb_project,
                    wandb_group=args.wandb_group,
                    phase="fusion-quality",
                ),
            )
        )
    return runs


def latency_runs(args: argparse.Namespace) -> list[Run]:
    limit = int(args.latency_limit)
    runs: list[Run] = []
    for layer in (8, 12, 16, 20):
        for budget in (48, 128):
            name = f"strong_ablation_latency_refcoco_depth_l{layer}_b{budget}_limit{limit}_bf16"
            runs.append(
                Run(
                    name,
                    "depth_latency_isolated",
                    limit,
                    base_command(
                        name=name,
                        budget=budget,
                        prune_layer=layer,
                        limit=limit,
                        wandb_mode=args.wandb_mode,
                        wandb_project=args.wandb_project,
                        wandb_group=args.wandb_group,
                        phase="depth-latency-isolated",
                    ),
                )
            )
    return runs


def run_integrity(run: Run) -> dict[str, Any]:
    config_path = run.run_dir / "config.json"
    if not config_path.exists():
        return {"status": "fail", "issues": ["missing config.json"]}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        manifest_path = Path(str(config["dataset/manifest"]))
        manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if config.get("limit") is not None:
            manifest_rows = manifest_rows[: int(config["limit"])]
        return validate_completed_run(
            run.run_dir,
            manifest_rows,
            budget=int(config["tokens/target_visual_tokens"]),
            original_tokens=int(config["tokens/original_visual_tokens"]),
            task=str(config["task"]),
            strategy=config.get("selection/strategy") if config.get("selection/method", "ug") == "ug" else None,
        )
    except Exception as exc:
        return {"status": "fail", "issues": [str(exc)]}


def is_complete(run: Run) -> bool:
    report = run_integrity(run)
    return report.get("status") == "pass" and report.get("rows", {}).get("actual_rows") == run.expected_rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_event(root: Path, payload: dict[str, Any]) -> None:
    with (root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": time.time(), **payload}, sort_keys=True) + "\n")


def run_phase(runs: list[Run], args: argparse.Namespace, *, parallel: int) -> None:
    state_path = args.logs_root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"runs": {}}
    rows = state.setdefault("runs", {})
    pending = [run for run in runs if not is_complete(run)]
    active: dict[str, tuple[Run, subprocess.Popen[Any], Any, Path]] = {}

    for run in runs:
        row = rows.setdefault(run.name, {})
        row.update({"phase": run.phase, "run_dir": str(run.run_dir), "expected_rows": run.expected_rows})
        row.setdefault("attempts", 0)
        if is_complete(run):
            row["status"] = "complete"
    write_json(state_path, state)

    while pending or active:
        while pending and len(active) < max(1, int(parallel)):
            run = pending.pop(0)
            if is_complete(run):
                rows[run.name]["status"] = "complete"
                continue
            row = rows[run.name]
            row["attempts"] = int(row.get("attempts", 0)) + 1
            row["status"] = "running"
            row["started_at"] = time.time()
            log_path = args.logs_root / f"{run.name}.attempt{row['attempts']}.log"
            handle = log_path.open("a", encoding="utf-8")
            env = dict(os.environ)
            env.setdefault("HF_HOME", "/workspace/cache/huggingface")
            env.setdefault("HF_HUB_OFFLINE", "1")
            process = subprocess.Popen(run.command, stdout=handle, stderr=subprocess.STDOUT, env=env)
            active[run.name] = (run, process, handle, log_path)
            append_event(args.logs_root, {"event": "start", "name": run.name, "command": shlex.join(run.command)})
            write_json(state_path, state)

        if not active:
            break
        time.sleep(float(args.poll_seconds))
        for name, (run, process, handle, log_path) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            row = rows[name]
            row.update({"finished_at": time.time(), "returncode": code, "last_log": str(log_path)})
            if code == 0 and is_complete(run):
                row["status"] = "complete"
                append_event(args.logs_root, {"event": "complete", "name": name})
            elif int(row["attempts"]) < int(args.max_attempts):
                row["status"] = "retry_pending"
                pending.append(run)
                append_event(args.logs_root, {"event": "retry", "name": name, "returncode": code})
            else:
                row["status"] = "failed"
                append_event(args.logs_root, {"event": "failed", "name": name, "returncode": code})
            del active[name]
            write_json(state_path, state)

    failed = [run.name for run in runs if rows.get(run.name, {}).get("status") == "failed"]
    if failed:
        raise SystemExit("Failed runs: " + ", ".join(failed))


def write_summary(root: Path, runs: list[Run]) -> None:
    summary: list[dict[str, Any]] = []
    for run in runs:
        row: dict[str, Any] = {
            "name": run.name,
            "phase": run.phase,
            "run_dir": str(run.run_dir),
            "expected_rows": run.expected_rows,
            "complete": is_complete(run),
        }
        metrics_path = run.run_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            for key in (
                "samples/evaluated",
                "quality/acc_at_0_5",
                "quality/mean_iou",
                "efficiency/mean_latency_s",
                "efficiency/median_latency_s",
                "efficiency/p95_latency_s",
                "ug/prune_after_layer",
                "ug/probe_layers",
                "ug/fusion_mode",
                "ug/map_source",
                "ug/role_preset",
                "position_mode",
            ):
                if key in metrics:
                    row[key] = metrics[key]
        summary.append(row)
    write_json(root / "summary.json", summary)


def main() -> int:
    args = parse_args()
    if args.quality_only and args.latency_only:
        raise SystemExit("--quality-only and --latency-only are mutually exclusive.")
    quality = [] if args.latency_only else quality_runs(args)
    latency = [] if args.quality_only else latency_runs(args)
    runs = [*quality, *latency]
    args.logs_root.mkdir(parents=True, exist_ok=True)
    with (args.logs_root / "commands.jsonl").open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(
                json.dumps(
                    {
                        "name": run.name,
                        "phase": run.phase,
                        "expected_rows": run.expected_rows,
                        "run_dir": str(run.run_dir),
                        "command": shlex.join(run.command),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    write_summary(args.logs_root, runs)
    print(f"Prepared {len(quality)} quality and {len(latency)} isolated-latency runs.")
    if not args.execute:
        return 0
    try:
        if quality:
            run_phase(quality, args, parallel=args.max_parallel)
        if latency:
            run_phase(latency, args, parallel=1)
    finally:
        write_summary(args.logs_root, runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
