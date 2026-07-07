#!/usr/bin/env python
"""Generate and run the full-dataset dissertation ablation campaign."""

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


DATASETS: dict[str, dict[str, str]] = {
    "refcoco": {
        "task": "grounding",
        "manifest": "data/manifests/official_baselines/refcoco_val.jsonl",
        "split": "val",
    },
    "pope": {
        "task": "vqa",
        "manifest": "data/manifests/official_baselines/pope_val.jsonl",
        "split": "val",
    },
    "textvqa": {
        "task": "vqa",
        "manifest": "data/manifests/official_baselines/textvqa_validation.jsonl",
        "split": "validation",
    },
}

POLICY_CHECKPOINT = "experiments/controllers/ug_prune_policy_v2_train200_strict6_sweep/model_iou_l0p2_t0p2_p0p25_q_mlp_cql_a0p1.pt"
UG_ROOT = Path("experiments/runs_ug_prune_single_pass")
BASELINE_ROOT = Path("experiments/runs_official_baselines")


@dataclass(frozen=True)
class RunEntry:
    id: str
    phase: str
    dataset: str
    method: str
    tokens: int | str
    expected_rows: int
    run_dir: str
    cmd: list[str]

    @property
    def command(self) -> str:
        return shlex.join(self.cmd)

    def as_record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = self.command
        return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run commands after writing commands.jsonl.")
    parser.add_argument("--logs-root", type=Path, default=Path("experiments/logs/full_dataset_ablation_campaign"))
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--latency-max-parallel", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "vlm-pruning"))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"), choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--latency-only", action="store_true")
    parser.add_argument("--latency-limit", type=int, default=1000)
    return parser.parse_args(argv)


def count_rows(path: str, limit: int | None = None) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for _ in handle:
            count += 1
            if limit is not None and count >= limit:
                return count
    return count


def wandb_group(args: argparse.Namespace) -> str:
    if args.wandb_group:
        return args.wandb_group
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"dissertation-ablation-campaign-{stamp}"


def with_wandb(cmd: list[str], args: argparse.Namespace, *, group: str, job_type: str, tags: str) -> list[str]:
    if args.no_wandb or args.wandb_mode == "disabled":
        return [*cmd, "--no-wandb"]
    return [
        *cmd,
        "--wandb",
        "--wandb-project",
        args.wandb_project,
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-group",
        group,
        "--wandb-job-type",
        job_type,
        "--wandb-tags",
        tags,
    ]


def baseline_entry(
    *,
    args: argparse.Namespace,
    dataset: str,
    method: str,
    tokens: int,
    phase: str,
    run_name: str,
    limit: int | None = None,
) -> RunEntry:
    spec = DATASETS[dataset]
    cmd = [
        sys.executable,
        "scripts/run_official_baseline.py",
        "--dataset",
        dataset,
        "--task",
        spec["task"],
        "--manifest",
        spec["manifest"],
        "--split",
        spec["split"],
        "--method",
        method,
        "--precision",
        args.precision,
        "--run-name",
        run_name,
        "--position-mode",
        "compact",
        "--resume",
        "--skip-existing",
    ]
    if method != "full":
        cmd.extend(["--target-visual-tokens", str(tokens), "--seed", str(args.seed)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    cmd = with_wandb(
        cmd,
        args,
        group=wandb_group(args),
        job_type=f"{phase}-{method}-tokens-{tokens}",
        tags=f"ablation,full-dataset,{phase},{dataset},{method},tokens-{tokens}",
    )
    return RunEntry(
        id=run_name,
        phase=phase,
        dataset=dataset,
        method=method,
        tokens=tokens,
        expected_rows=count_rows(spec["manifest"], limit),
        run_dir=str(BASELINE_ROOT / run_name),
        cmd=cmd,
    )


def ug_entry(
    *,
    args: argparse.Namespace,
    dataset: str,
    tokens: int,
    phase: str,
    run_name: str,
    method: str,
    role: str = "more_context",
    map_source: str = "query_max",
    fusion: str = "soft_average",
    prune_layer: int = 12,
    selection_strategy: str = "standard",
    oracle_mode: str = "none",
    policy: bool = False,
    limit: int | None = None,
) -> RunEntry:
    spec = DATASETS[dataset]
    cmd = [
        sys.executable,
        "scripts/run_ug_prune_single_pass.py",
        "--dataset",
        dataset,
        "--task",
        spec["task"],
        "--manifest",
        spec["manifest"],
        "--split",
        spec["split"],
        "--target-visual-tokens",
        str(tokens),
        "--precision",
        args.precision,
        "--run-name",
        run_name,
        "--prune-after-layer",
        str(prune_layer),
        "--decode-mode",
        "layer_aware_cache",
        "--diagnostics-mode",
        "benchmark",
        "--resume",
        "--skip-existing",
    ]
    if policy:
        cmd.extend(["--ug-policy-checkpoint", POLICY_CHECKPOINT])
    else:
        cmd.extend(
            [
                "--ug-role-preset",
                role,
                "--ug-map-source",
                map_source,
                "--ug-fusion-mode",
                fusion,
                "--ug-selection-strategy",
                selection_strategy,
            ]
        )
        if oracle_mode != "none":
            cmd.extend(["--ug-oracle-mode", oracle_mode])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    cmd = with_wandb(
        cmd,
        args,
        group=wandb_group(args),
        job_type=f"{phase}-{method}-tokens-{tokens}",
        tags=f"ablation,full-dataset,{phase},{dataset},{method},tokens-{tokens}",
    )
    return RunEntry(
        id=run_name,
        phase=phase,
        dataset=dataset,
        method=method,
        tokens="policy" if policy else tokens,
        expected_rows=count_rows(spec["manifest"], limit),
        run_dir=str(UG_ROOT / run_name),
        cmd=cmd,
    )


def add_unique(entries: list[RunEntry], seen: set[str], entry: RunEntry) -> None:
    if entry.id in seen:
        return
    seen.add(entry.id)
    entries.append(entry)


def build_main_entries(args: argparse.Namespace) -> list[RunEntry]:
    entries: list[RunEntry] = []
    seen: set[str] = set()

    for dataset in DATASETS:
        for method in ("random", "uniform"):
            add_unique(
                entries,
                seen,
                baseline_entry(
                    args=args,
                    dataset=dataset,
                    method=method,
                    tokens=72,
                    phase="matched_b72",
                    run_name=f"ablation_fullmatrix_{dataset}_{method}_tokens72_seed{args.seed}_{args.precision}",
                ),
            )
        add_unique(
            entries,
            seen,
            ug_entry(
                args=args,
                dataset=dataset,
                tokens=72,
                phase="matched_b72",
                run_name=f"ablation_fullmatrix_{dataset}_ug_k12_b72_more_context_{args.precision}",
                method="ug_k12_b72_more_context",
            ),
        )

    for dataset in ("refcoco", "pope", "textvqa"):
        for budget in (48, 128):
            for role in ("default", "more_context", "more_anchors"):
                existing_name = f"official_fullmatrix_{dataset}_ug_k12_b{budget}_more_context_{args.precision}"
                run_name = existing_name if role == "more_context" else f"ablation_fullmatrix_{dataset}_ug_k12_b{budget}_{role}_{args.precision}"
                add_unique(
                    entries,
                    seen,
                    ug_entry(
                        args=args,
                        dataset=dataset,
                        tokens=budget,
                        phase="role_preset",
                        run_name=run_name,
                        method=f"ug_k12_b{budget}_{role}",
                        role=role,
                    ),
                )

    for budget in (48, 128):
        for map_source in ("query_max", "query_pooled"):
            existing_name = f"official_fullmatrix_refcoco_ug_k12_b{budget}_more_context_{args.precision}"
            run_name = (
                existing_name
                if map_source == "query_max"
                else f"ablation_fullmatrix_refcoco_ug_k12_b{budget}_more_context_{map_source}_{args.precision}"
            )
            add_unique(
                entries,
                seen,
                ug_entry(
                    args=args,
                    dataset="refcoco",
                    tokens=budget,
                    phase="map_source",
                    run_name=run_name,
                    method=f"ug_k12_b{budget}_more_context_{map_source}",
                    map_source=map_source,
                ),
            )

    add_unique(
        entries,
        seen,
        ug_entry(
            args=args,
            dataset="refcoco",
            tokens=48,
            phase="layer_fusion",
            run_name=f"official_fullmatrix_refcoco_ug_k12_b48_more_context_{args.precision}",
            method="ug_k12_b48_more_context_soft_average",
        ),
    )
    for layer in (4, 8, 12):
        add_unique(
            entries,
            seen,
            ug_entry(
                args=args,
                dataset="refcoco",
                tokens=48,
                phase="layer_fusion",
                run_name=f"ablation_fullmatrix_refcoco_ug_k{layer}_b48_more_context_single_layer_{args.precision}",
                method=f"ug_k{layer}_b48_more_context_single_layer",
                fusion="single_layer",
                prune_layer=layer,
            ),
        )

    for strategy in ("standard", "coverage", "coverage_zip"):
        run_name = (
            f"official_fullmatrix_refcoco_ug_k12_b48_more_context_{args.precision}"
            if strategy == "standard"
            else f"ablation_fullmatrix_refcoco_ug_k12_b48_more_context_{strategy}_{args.precision}"
        )
        add_unique(
            entries,
            seen,
            ug_entry(
                args=args,
                dataset="refcoco",
                tokens=48,
                phase="selection_strategy",
                run_name=run_name,
                method=f"ug_k12_b48_more_context_{strategy}",
                selection_strategy=strategy,
            ),
        )

    for oracle_mode in ("target_box", "target_context_ring", "target_context_anchor"):
        add_unique(
            entries,
            seen,
            ug_entry(
                args=args,
                dataset="refcoco",
                tokens=48,
                phase="oracle",
                run_name=f"ablation_fullmatrix_refcoco_oracle_{oracle_mode}_b48_{args.precision}",
                method=f"oracle_{oracle_mode}_b48",
                oracle_mode=oracle_mode,
            ),
        )

    return entries


def build_latency_entries(args: argparse.Namespace) -> list[RunEntry]:
    limit = int(args.latency_limit)
    entries = [
        baseline_entry(
            args=args,
            dataset="refcoco",
            method="full",
            tokens=576,
            phase="latency_isolated",
            run_name=f"latency_iso_refcoco_full_tokens576_limit{limit}_seed{args.seed}_{args.precision}",
            limit=limit,
        ),
        baseline_entry(
            args=args,
            dataset="refcoco",
            method="random",
            tokens=48,
            phase="latency_isolated",
            run_name=f"latency_iso_refcoco_random_tokens48_limit{limit}_seed{args.seed}_{args.precision}",
            limit=limit,
        ),
        baseline_entry(
            args=args,
            dataset="refcoco",
            method="uniform",
            tokens=48,
            phase="latency_isolated",
            run_name=f"latency_iso_refcoco_uniform_tokens48_limit{limit}_seed{args.seed}_{args.precision}",
            limit=limit,
        ),
        ug_entry(
            args=args,
            dataset="refcoco",
            tokens=48,
            phase="latency_isolated",
            run_name=f"latency_iso_refcoco_ug_k12_b48_more_context_limit{limit}_{args.precision}",
            method="ug_k12_b48_more_context",
            limit=limit,
        ),
        ug_entry(
            args=args,
            dataset="refcoco",
            tokens=192,
            phase="latency_isolated",
            run_name=f"latency_iso_refcoco_ug_k12_b192_more_context_limit{limit}_{args.precision}",
            method="ug_k12_b192_more_context",
            limit=limit,
        ),
        ug_entry(
            args=args,
            dataset="refcoco",
            tokens=72,
            phase="latency_isolated",
            run_name=f"latency_iso_refcoco_v2_strict_policy_b72_limit{limit}_{args.precision}",
            method="v2_strict_policy_b72",
            policy=True,
            limit=limit,
        ),
    ]
    return entries


def metrics_complete(entry: RunEntry) -> bool:
    metrics_path = Path(entry.run_dir) / "metrics.json"
    if not metrics_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return int(metrics.get("samples/evaluated", -1)) == int(entry.expected_rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_event(logs_root: Path, event: dict[str, Any]) -> None:
    payload = {"time": time.time(), **event}
    with (logs_root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def load_state(logs_root: Path) -> dict[str, Any]:
    path = logs_root / "state.json"
    if not path.exists():
        return {"runs": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(logs_root: Path, state: dict[str, Any]) -> None:
    write_json(logs_root / "state.json", state)


def write_commands(logs_root: Path, entries: list[RunEntry]) -> None:
    logs_root.mkdir(parents=True, exist_ok=True)
    with (logs_root / "commands.jsonl").open("w", encoding="utf-8") as handle:
        for index, entry in enumerate(entries, start=1):
            record = {"index": index, **entry.as_record()}
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def run_entries(entries: list[RunEntry], *, args: argparse.Namespace, max_parallel: int, label: str) -> None:
    logs_root = args.logs_root
    state = load_state(logs_root)
    runs = state.setdefault("runs", {})
    queue = list(entries)
    active: dict[str, tuple[subprocess.Popen[Any], Any, Path]] = {}
    max_parallel = max(1, int(max_parallel))

    for entry in queue:
        row = runs.setdefault(entry.id, {})
        row.setdefault("attempts", 0)
        row.update(
            {
                "phase": entry.phase,
                "dataset": entry.dataset,
                "method": entry.method,
                "tokens": entry.tokens,
                "run_dir": entry.run_dir,
                "expected_rows": entry.expected_rows,
            }
        )
        if metrics_complete(entry):
            row["status"] = "complete"
    save_state(logs_root, state)

    pending = [entry for entry in queue if runs.get(entry.id, {}).get("status") != "complete"]
    while pending or active:
        while pending and len(active) < max_parallel:
            entry = pending.pop(0)
            row = runs.setdefault(entry.id, {})
            if metrics_complete(entry):
                row["status"] = "complete"
                save_state(logs_root, state)
                continue
            attempts = int(row.get("attempts", 0)) + 1
            row["attempts"] = attempts
            row["status"] = "running"
            row["started_at"] = time.time()
            log_path = logs_root / f"{entry.id}.attempt{attempts}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            append_event(logs_root, {"event": "start", "label": label, "id": entry.id, "attempt": attempts, "command": entry.command})
            process = subprocess.Popen(entry.cmd, stdout=log_handle, stderr=subprocess.STDOUT)
            active[entry.id] = (process, log_handle, log_path)
            save_state(logs_root, state)

        if not active:
            break

        time.sleep(float(args.poll_seconds))
        for run_id, (process, log_handle, log_path) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log_handle.close()
            entry = next(item for item in entries if item.id == run_id)
            row = runs.setdefault(run_id, {})
            row["finished_at"] = time.time()
            row["returncode"] = code
            row["last_log"] = str(log_path)
            if code == 0 and metrics_complete(entry):
                row["status"] = "complete"
                append_event(logs_root, {"event": "complete", "label": label, "id": run_id, "returncode": code})
            elif int(row.get("attempts", 0)) < int(args.max_attempts):
                row["status"] = "retry_pending"
                pending.append(entry)
                append_event(logs_root, {"event": "retry", "label": label, "id": run_id, "returncode": code})
            else:
                row["status"] = "failed"
                append_event(logs_root, {"event": "failed", "label": label, "id": run_id, "returncode": code})
            del active[run_id]
            save_state(logs_root, state)

    failed = [run_id for run_id, row in runs.items() if row.get("status") == "failed" and any(entry.id == run_id for entry in entries)]
    if failed:
        raise SystemExit(f"{label} failed runs: {', '.join(failed)}")


def metric_row(entry: RunEntry) -> dict[str, Any]:
    metrics_path = Path(entry.run_dir) / "metrics.json"
    row: dict[str, Any] = {
        "id": entry.id,
        "phase": entry.phase,
        "dataset": entry.dataset,
        "method": entry.method,
        "tokens": entry.tokens,
        "expected_rows": entry.expected_rows,
        "run_dir": entry.run_dir,
        "complete": metrics_complete(entry),
    }
    if not metrics_path.exists():
        return row
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in (
        "samples/evaluated",
        "quality/acc_at_0_5",
        "quality/mean_iou",
        "quality/accuracy",
        "quality/f1",
        "efficiency/mean_latency_s",
        "efficiency/median_latency_s",
        "efficiency/p95_latency_s",
        "efficiency/pruning_selection_latency_s",
        "tokens/actual_visual_tokens",
        "ug/oracle_mode",
        "ug/role_preset",
        "ug/map_source",
        "ug/fusion_mode",
        "ug/selection_strategy",
        "ug/prune_after_layer",
    ):
        if key in metrics:
            row[key] = metrics[key]
    return row


def write_summary(logs_root: Path, entries: list[RunEntry]) -> None:
    rows = [metric_row(entry) for entry in entries]
    write_json(logs_root / "summary.json", rows)
    with (logs_root / "summary.tsv").open("w", encoding="utf-8") as handle:
        columns = [
            "complete",
            "phase",
            "dataset",
            "method",
            "tokens",
            "samples/evaluated",
            "quality/acc_at_0_5",
            "quality/mean_iou",
            "quality/accuracy",
            "quality/f1",
            "efficiency/mean_latency_s",
            "run_dir",
        ]
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.main_only and args.latency_only:
        raise SystemExit("--main-only and --latency-only cannot both be set.")
    main_entries = [] if args.latency_only else build_main_entries(args)
    latency_entries = [] if args.main_only else build_latency_entries(args)
    entries = [*main_entries, *latency_entries]
    write_commands(args.logs_root, entries)
    print(f"Wrote {len(entries)} commands to {args.logs_root / 'commands.jsonl'}")
    write_summary(args.logs_root, entries)
    if not args.execute:
        return 0
    try:
        if main_entries:
            run_entries(main_entries, args=args, max_parallel=args.max_parallel, label="main")
        if latency_entries:
            run_entries(latency_entries, args=args, max_parallel=args.latency_max_parallel, label="latency")
    finally:
        write_summary(args.logs_root, entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
