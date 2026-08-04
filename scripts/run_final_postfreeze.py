#!/usr/bin/env python
"""Run resumable held-out and generalisation campaigns after method freezing."""

from __future__ import annotations

import argparse
import hashlib
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
FINAL = Path("experiments/final_evaluation")


@dataclass(frozen=True)
class Entry:
    id: str
    rows: int
    tokens: int
    manifest: str
    run_dir: str
    command: tuple[str, ...]
    category_counts: dict[str, int] | None = None
    expected_config: dict[str, Any] | None = None
    limit: int | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": utc_now(), **payload}, sort_keys=True) + "\n")


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except Exception:
        return []


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


def common_wandb(args: argparse.Namespace, group: str, job_type: str, tags: str) -> list[str]:
    if args.wandb_mode == "disabled" or getattr(args, "limit", None) is not None:
        return ["--no-wandb"]
    return [
        "--wandb",
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-project",
        args.wandb_project,
        "--wandb-group",
        group,
        "--wandb-job-type",
        job_type,
        "--wandb-tags",
        tags,
    ]


def official_entry(
    args: argparse.Namespace,
    *,
    dataset: str,
    task: str,
    split: str,
    manifest: str,
    rows: int,
    method: str,
    seed: int,
) -> Entry:
    limit = getattr(args, "limit", None)
    tokens = 576 if method == "full" else 48
    base_name = f"final_{dataset}_{split}_{method}_b{tokens}_seed{seed}"
    name = f"smoke{limit}_{base_name}" if limit is not None else base_name
    output = (
        FINAL / "smoke/final_matrix" / args.phase / "official"
        if limit is not None
        else FINAL / "runs" / "official"
    )
    command = [
        sys.executable,
        "scripts/run_official_baseline.py",
        "--dataset",
        dataset,
        "--task",
        task,
        "--manifest",
        manifest,
        "--split",
        split,
        "--method",
        method,
        "--target-visual-tokens",
        str(tokens),
        "--seed",
        str(seed),
        "--precision",
        "bf16",
        "--position-mode",
        "gap_preserving" if method != "visionzip" else "compact",
        "--run-name",
        name,
        "--output-root",
        str(output),
        "--figures-root",
        str(FINAL / "figures"),
        "--tables-root",
        str(FINAL / "tables"),
        "--reports-root",
        str(FINAL / "reports"),
        "--max-figures",
        "0",
        "--resume",
        "--skip-existing",
        *(["--limit", str(limit)] if limit is not None else []),
        *common_wandb(
            args,
            f"dissertation-final-{dataset}-{split}-20260729",
            "final-heldout" if dataset == "refcoco" else "final-generalisation",
            f"dissertation,final,{dataset},{split},{method},b{tokens},seed{seed}",
        ),
    ]
    return Entry(
        id=name,
        rows=min(rows, limit) if limit is not None else rows,
        tokens=tokens,
        manifest=manifest,
        run_dir=str(output / name),
        command=tuple(command),
        category_counts={"random": 3000, "popular": 3000, "adversarial": 3000}
        if dataset == "pope" and limit is None
        else None,
        expected_config={
            "method/name": method,
            "method/seed": seed,
            "position_mode": "compact" if method == "visionzip" else "gap_preserving",
            **(
                {
                    "published_baseline/commit": "d1659729b5bf1be225e99ee15783deeea80f63b1",
                    "published_baseline/fastv_k": 3,
                }
                if method == "fastv"
                else {
                    "published_baseline/commit": "8f86b55c6f000eb033e6912538af2dd7dcb30502"
                }
                if method == "visionzip"
                else {}
            ),
            "limit": limit,
        },
        limit=limit,
    )


def final_method_entry(
    args: argparse.Namespace,
    *,
    dataset: str,
    task: str,
    split: str,
    manifest: str,
    rows: int,
    frozen: dict[str, Any],
) -> Entry:
    limit = getattr(args, "limit", None)
    method = frozen["method"]
    base_name = f"final_{dataset}_{split}_frozen_k20_b48"
    name = f"smoke{limit}_{base_name}" if limit is not None else base_name
    output = (
        FINAL / "smoke/final_matrix" / args.phase / "frozen"
        if limit is not None
        else FINAL / "runs" / "frozen"
    )
    command = [
        sys.executable,
        "scripts/run_ug_prune_single_pass.py",
        "--dataset",
        dataset,
        "--task",
        task,
        "--manifest",
        manifest,
        "--split",
        split,
        "--target-visual-tokens",
        "48",
        "--prune-after-layer",
        "20",
        "--position-mode",
        "gap_preserving",
        "--decode-mode",
        "layer_aware_cache",
        "--diagnostics-mode",
        "benchmark",
        "--ug-role-preset",
        method["role_allocation"],
        "--ug-map-source",
        method["query_scoring"],
        "--ug-fusion-mode",
        "single_layer",
        "--precision",
        "bf16",
        "--run-name",
        name,
        "--output-root",
        str(output),
        "--figures-root",
        str(FINAL / "figures"),
        "--tables-root",
        str(FINAL / "tables"),
        "--max-figures",
        "0",
        "--resume",
        "--skip-existing",
        *(["--limit", str(limit)] if limit is not None else []),
        *common_wandb(
            args,
            f"dissertation-final-{dataset}-{split}-20260729",
            "final-heldout" if dataset == "refcoco" else "final-generalisation",
            f"dissertation,final,frozen,{dataset},{split},k20,b48",
        ),
    ]
    return Entry(
        id=name,
        rows=min(rows, limit) if limit is not None else rows,
        tokens=48,
        manifest=manifest,
        run_dir=str(output / name),
        command=tuple(command),
        category_counts={"random": 3000, "popular": 3000, "adversarial": 3000}
        if dataset == "pope" and limit is None
        else None,
        expected_config={
            "method/seed": 0,
            "ug/prune_after_layer": 20,
            "ug/map_source": method["query_scoring"],
            "ug/role_preset": method["role_allocation"],
            "ug/fusion_mode": "single_layer",
            "ug/probe_layers": "20",
            "ug/position_mode": "gap_preserving",
            "ug/decode_mode": "layer_aware_cache",
            "limit": limit,
        },
        limit=limit,
    )


def entries(args: argparse.Namespace) -> list[Entry]:
    frozen_path = FINAL / "frozen_method.json"
    if not frozen_path.exists():
        raise SystemExit("frozen_method.json does not exist; finish and freeze validation first.")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    protocol_path = FINAL / "protocol/protocol.json"
    if (
        not protocol_path.exists()
        or hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        != frozen.get("registered_protocol_sha256")
    ):
        raise SystemExit(
            "The registered protocol differs from the frozen method; do not run held-out data "
            "until the freeze/protocol mismatch is resolved."
        )
    if args.phase == "heldout":
        result: list[Entry] = []
        for split, count in (("testA", 5657), ("testB", 5095)):
            manifest = f"data/manifests/final_evaluation/refcoco_{split}_unique.jsonl"
            result.append(
                final_method_entry(
                    args,
                    dataset="refcoco",
                    task="grounding",
                    split=split,
                    manifest=manifest,
                    rows=count,
                    frozen=frozen,
                )
            )
            for method, seeds in (
                ("full", (0,)),
                ("random", (0, 1, 2)),
                ("uniform", (0,)),
                ("fastv", (0,)),
                ("visionzip", (0,)),
            ):
                for seed in seeds:
                    result.append(
                        official_entry(
                            args,
                            dataset="refcoco",
                            task="grounding",
                            split=split,
                            manifest=manifest,
                            rows=count,
                            method=method,
                            seed=seed,
                        )
                    )
        return result

    result = []
    for dataset, split, manifest, count in (
        ("pope", "test", "data/manifests/final_evaluation/pope_all_unique.jsonl", 9000),
        (
            "textvqa",
            "validation",
            "data/manifests/official_baselines/textvqa_validation.jsonl",
            5000,
        ),
    ):
        result.append(
            final_method_entry(
                args,
                dataset=dataset,
                task="vqa",
                split=split,
                manifest=manifest,
                rows=count,
                frozen=frozen,
            )
        )
        for method in ("full", "fastv", "visionzip"):
            result.append(
                official_entry(
                    args,
                    dataset=dataset,
                    task="vqa",
                    split=split,
                    manifest=manifest,
                    rows=count,
                    method=method,
                    seed=0,
                )
            )
    return result


def complete(entry: Entry) -> bool:
    run_dir = Path(entry.run_dir)
    rows = read_rows(run_dir / "per_sample_results.jsonl")
    ids = [str(row.get("sample_id")) for row in rows]
    all_manifest_rows = read_rows(Path(entry.manifest))
    manifest_rows = all_manifest_rows[: entry.rows]
    manifest_ids = [str(row.get("sample_id")) for row in manifest_rows]
    if (
        len(rows) != entry.rows
        or len(set(ids)) != entry.rows
        or len(all_manifest_rows) < entry.rows
        or ids != manifest_ids
    ):
        return False
    if not all(
        (run_dir / name).exists()
        for name in ("metrics.json", "config.json", "run_environment.json", "wandb_run.json")
    ):
        return False
    if not all(
        int(row.get("target_visual_tokens", -1)) == entry.tokens
        and len(row.get("retained_indices") or []) == entry.tokens
        and len(set(int(value) for value in (row.get("retained_indices") or [])))
        == entry.tokens
        and all(
            0 <= int(value) < 576 for value in (row.get("retained_indices") or [])
        )
        and int(row.get("actual_llm_visual_tokens", -1)) == entry.tokens
        and int(row.get("seed", -1))
        == int((entry.expected_config or {}).get("method/seed", 0))
        for row in rows
    ):
        return False
    if entry.category_counts:
        counts = {
            category: sum(str(row.get("category")).lower() == category for row in rows)
            for category in entry.category_counts
        }
        if counts != entry.category_counts:
            return False
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest_digest = hashlib.sha256(Path(entry.manifest).read_bytes()).hexdigest()
    recorded_digest = str(config.get("dataset/manifest_checksum") or "")
    if "refcoco" in entry.id:
        expected_prompt = (
            'Identify the target described by this referring expression: "{query}".\n'
            "Return exactly one bounding box in normalized 0-1 image coordinates as "
            "[x_min, y_min, x_max, y_max]. Do not include any other text."
        )
        expected_evaluator = "box_iou_acc_at_0_5"
    elif "pope" in entry.id:
        expected_prompt = "{question}\nAnswer with exactly one word: yes or no."
        expected_evaluator = "yes_no_accuracy_f1"
    else:
        expected_prompt = "{question}\nAnswer as briefly as possible."
        expected_evaluator = "vqa_soft_accuracy"
    wandb_record = json.loads(
        (run_dir / "wandb_run.json").read_text(encoding="utf-8")
    )
    wandb_ok = entry.limit is not None or (
        bool(wandb_record.get("enabled"))
        and wandb_record.get("mode") in {"offline", "online"}
        and bool(wandb_record.get("id"))
        and (
            wandb_record.get("mode") == "online"
            or bool(list((run_dir / "wandb").glob("offline-run-*")))
        )
    )
    expected_config_ok = all(
        config.get(key) == value
        for key, value in (entry.expected_config or {}).items()
    )
    return (
        config.get("dataset/manifest") == entry.manifest
        and bool(recorded_digest)
        and manifest_digest.startswith(recorded_digest)
        and config.get("model/name") == "llava-hf/llava-1.5-7b-hf"
        and config.get("model/precision") == "bf16"
        and config.get("model/decoding") == "deterministic_greedy"
        and int(config.get("model/max_new_tokens", -1))
        == (48 if "refcoco" in entry.id else 16)
        and config.get("prompt/template") == expected_prompt
        and config.get("evaluator/name") == expected_evaluator
        and bool(config.get("evaluator/metric_definition"))
        and expected_config_ok
        and int(config.get("tokens/target_visual_tokens", -1)) == entry.tokens
        and int(metrics.get("samples/evaluated", -1)) == entry.rows
        and wandb_ok
    )


def run_campaign(items: list[Entry], args: argparse.Namespace) -> None:
    logs = (
        FINAL / "logs" / f"smoke_{args.phase}_{args.limit}"
        if args.limit is not None
        else FINAL / "logs" / args.phase
    )
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "commands.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(
                json.dumps(
                    {
                        "id": item.id,
                        "expected_rows": item.rows,
                        "tokens": item.tokens,
                        "run_dir": item.run_dir,
                        "limit": item.limit,
                        "command": shlex.join(item.command),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    state_path = logs / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"runs": {}}
    records = state["runs"]
    pending: list[Entry] = []
    for item in items:
        record = records.setdefault(item.id, {})
        record.update(
            {
                "run_dir": item.run_dir,
                "rows": item.rows,
                "tokens": item.tokens,
                "limit": item.limit,
                "command": shlex.join(item.command),
            }
        )
        record.setdefault("attempts", 0)
        record["status"] = "complete" if complete(item) else record.get("status", "pending")
        if record["status"] != "complete":
            pending.append(item)
    write_json(state_path, state)
    if not args.execute:
        print(f"Registered {len(items)} runs; {len(pending)} pending.")
        return

    active: dict[str, tuple[Entry, subprocess.Popen[Any], Any]] = {}
    while pending or active:
        while pending and len(active) < args.max_parallel:
            item = pending.pop(0)
            record = records[item.id]
            record["attempts"] += 1
            record["status"] = "running"
            record["started_at"] = utc_now()
            handle = (logs / f"{item.id}.attempt{record['attempts']}.log").open("a", encoding="utf-8")
            environment = dict(os.environ)
            environment.setdefault("HF_HOME", "/workspace/cache/huggingface")
            environment["HF_HUB_DISABLE_XET"] = "1"
            process = subprocess.Popen(
                item.command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active[item.id] = item, process, handle
            append_event(
                logs / "events.jsonl",
                {"event": "start", "id": item.id, "attempt": record["attempts"], "pid": process.pid},
            )
            write_json(state_path, state)
        memory = gpu_memory()
        if memory is not None:
            with (logs / "gpu_memory.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"time": utc_now(), **memory}, sort_keys=True) + "\n")
        time.sleep(args.poll_seconds)
        for run_id, (item, process, handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            record = records[run_id]
            record["finished_at"] = utc_now()
            record["returncode"] = code
            if code == 0 and complete(item):
                record["status"] = "complete"
                append_event(logs / "events.jsonl", {"event": "complete", "id": run_id})
            elif record["attempts"] < args.max_attempts:
                record["status"] = "retry_pending"
                pending.append(item)
                append_event(logs / "events.jsonl", {"event": "retry", "id": run_id, "returncode": code})
            else:
                record["status"] = "failed"
                append_event(logs / "events.jsonl", {"event": "failed", "id": run_id, "returncode": code})
            del active[run_id]
            write_json(state_path, state)
    failed = [name for name, record in records.items() if record["status"] == "failed"]
    write_json(
        logs / "summary.json",
        {
            "phase": args.phase,
            "registered": len(items),
            "complete": sum(record["status"] == "complete" for record in records.values()),
            "failed": failed,
            "updated_at": utc_now(),
        },
    )
    if failed:
        raise SystemExit(f"Failed runs: {', '.join(failed)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["heldout", "generalisation"])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "vlm-pruning"))
    parser.add_argument(
        "--limit",
        type=int,
        choices=(5, 20),
        help="Run a separate 5- or 20-example smoke matrix before full execution.",
    )
    args = parser.parse_args()
    if not 1 <= args.max_parallel <= 3:
        raise SystemExit("Quality parallelism is capped at the validated safe value of three.")
    if args.execute:
        soak_path = FINAL / "logs/soak/soak_report.json"
        if not soak_path.exists():
            raise SystemExit(
                "Quality execution requires the registered three-process GPU soak report."
            )
        soak = json.loads(soak_path.read_text(encoding="utf-8"))
        if not soak.get("passed"):
            raise SystemExit("The registered GPU soak did not pass.")
        safe_parallel = int(soak.get("recommended_quality_parallelism", 1))
        safe_parallel = max(1, min(3, safe_parallel))
        if args.max_parallel > safe_parallel:
            print(
                f"Reducing quality parallelism from {args.max_parallel} to "
                f"the soak-validated value {safe_parallel}."
            )
            args.max_parallel = safe_parallel
        if args.limit is None:
            missing_smokes: list[str] = []
            for smoke_phase, expected in (
                ("heldout", 16),
                ("generalisation", 8),
            ):
                passed = False
                for smoke_limit in (5, 20):
                    path = (
                        FINAL
                        / "logs"
                        / f"smoke_{smoke_phase}_{smoke_limit}"
                        / "state.json"
                    )
                    if not path.exists():
                        continue
                    smoke_state = json.loads(path.read_text(encoding="utf-8"))
                    smoke_runs = smoke_state.get("runs", {})
                    if (
                        len(smoke_runs) == expected
                        and all(
                            row.get("status") == "complete"
                            for row in smoke_runs.values()
                        )
                    ):
                        passed = True
                        break
                if not passed:
                    missing_smokes.append(smoke_phase)
            if missing_smokes:
                raise SystemExit(
                    "Full post-freeze execution requires a completed 5- or "
                    "20-example smoke matrix for: "
                    + ", ".join(missing_smokes)
                )
    run_campaign(entries(args), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
