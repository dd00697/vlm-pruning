#!/usr/bin/env python
"""Run the 25 registered efficiency trials strictly serially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_pruning.efficiency import balanced_latin_square  # noqa: E402


OUTPUT = Path("experiments/final_evaluation/timing")
LOGS = Path("experiments/final_evaluation/logs/timing")
METHODS = ["full", "k12", "final", "fastv", "visionzip"]
MANIFEST = Path(
    "data/manifests/final_evaluation/refcoco_testA_timing1000_seed20260729.jsonl"
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def complete(
    repetition: int,
    position: int,
    method: str,
    *,
    output: Path | None = None,
    expected_rows: int = 1000,
    minimum_warmup: int = 20,
    require_wandb: bool = True,
) -> bool:
    output = OUTPUT if output is None else output
    run_dir = output / f"rep{repetition}_pos{position}_{method}"
    metrics_path = run_dir / "metrics.json"
    timing_path = run_dir / "per_sample_timing.jsonl"
    config_path = run_dir / "config.json"
    provenance_path = run_dir / "wandb_run.json"
    environment_path = run_dir / "run_environment.json"
    if not all(
        path.exists()
        for path in (
            metrics_path,
            timing_path,
            config_path,
            provenance_path,
            environment_path,
        )
    ):
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in timing_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except Exception:
        return False
    expected_tokens = 576 if method == "full" else 48
    try:
        manifest_ids = [
            str(json.loads(line)["sample_id"])
            for line in MANIFEST.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except Exception:
        return False
    manifest_ids = manifest_ids[:expected_rows]
    row_ids = [str(row.get("sample_id")) for row in rows]
    provenance_ok = not require_wandb or (
        bool(provenance.get("enabled"))
        and provenance.get("mode") in {"offline", "online"}
        and bool(provenance.get("id"))
        and (
            provenance.get("mode") == "online"
            or bool(list((run_dir / "wandb").glob("offline-run-*")))
        )
    )
    return (
        int(metrics.get("samples", -1)) == expected_rows
        and len(rows) == expected_rows
        and row_ids == manifest_ids
        and len(set(row_ids)) == expected_rows
        and config.get("method") == method
        and int(config.get("repetition", -1)) == repetition
        and int(config.get("order_position", -1)) == position
        and int(config.get("warmup", -1)) >= minimum_warmup
        and int(config.get("limit", -1)) == expected_rows
        and config.get("checkpoint") == "llava-hf/llava-1.5-7b-hf"
        and config.get("precision") == "bf16"
        and config.get("manifest_sha256") == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
        and provenance_ok
        and all(
            int(row.get("requested_visual_tokens", -1)) == expected_tokens
            and int(row.get("actual_visual_tokens", -1)) == expected_tokens
            and int(row.get("repetition", -1)) == repetition
            and int(row.get("order_position", -1)) == position
            and row.get("peak_allocated_gpu_bytes") is not None
            and row.get("peak_reserved_gpu_bytes") is not None
            and all(
                row.get(field) is not None
                for field in (
                    "vision_encoder_latency_s",
                    "projector_latency_s",
                    "selector_latency_s",
                    "llm_prefill_latency_s",
                    "decode_latency_s",
                    "decode_time_per_token_s",
                    "end_to_end_latency_s",
                    "prompt_kv_cache_bytes",
                    "final_kv_cache_bytes",
                )
            )
            and (
                method not in {"k12", "final", "fastv"}
                or all(
                    row.get(field) is not None
                    for field in (
                        "prompt_early_bank_bytes",
                        "prompt_late_bank_bytes",
                        "final_early_bank_bytes",
                        "final_late_bank_bytes",
                    )
                )
            )
            for row in rows
        )
    )


def gpu_processes() -> list[str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ],
            text=True,
        )
        return [line.strip() for line in output.splitlines() if line.strip()]
    except Exception:
        return []


def entries_for_mode(smoke: bool) -> tuple[Path, Path, list[dict[str, Any]]]:
    """Build the registered full or five-method serial smoke campaign."""
    orders = balanced_latin_square(METHODS)
    output = (
        Path("experiments/final_evaluation/timing_smoke")
        if smoke
        else OUTPUT
    )
    logs = (
        Path("experiments/final_evaluation/logs/timing_smoke")
        if smoke
        else LOGS
    )
    selected_orders = orders[:1] if smoke else orders
    entries = [
        {
            "id": f"rep{repetition}_pos{position}_{method}",
            "repetition": repetition,
            "position": position,
            "method": method,
            "command": (
                sys.executable,
                "scripts/run_final_timing_trial.py",
                "--method",
                method,
                "--repetition",
                str(repetition),
                "--order-position",
                str(position),
                *(
                    (
                        "--output-root",
                        str(output),
                        "--limit",
                        "5",
                        "--warmup-examples",
                        "5",
                        "--wandb-mode",
                        "disabled",
                    )
                    if smoke
                    else ()
                ),
            ),
        }
        for repetition, order in enumerate(selected_orders, start=1)
        for position, method in enumerate(order, start=1)
    ]
    return output, logs, entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run five serial 5-example preflight trials in separate directories.",
    )
    args = parser.parse_args()
    output, logs, entries = entries_for_mode(args.smoke)
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "commands.jsonl").open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(
                json.dumps({"id": entry["id"], "command": shlex.join(entry["command"])}, sort_keys=True)
                + "\n"
            )
    state_path = logs / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"runs": {}}
    for entry in entries:
        record = state["runs"].setdefault(entry["id"], {"attempts": 0})
        record["command"] = shlex.join(entry["command"])
        record["status"] = (
            "complete"
            if complete(
                entry["repetition"],
                entry["position"],
                entry["method"],
                output=output,
                expected_rows=5 if args.smoke else 1000,
                minimum_warmup=5 if args.smoke else 20,
                require_wandb=not args.smoke,
            )
            else record.get("status", "pending")
        )
    write_json(state_path, state)
    if not args.execute:
        pending = sum(record["status"] != "complete" for record in state["runs"].values())
        print(
            f"Registered {len(entries)} serial "
            f"{'timing smoke' if args.smoke else 'timing'} trials; {pending} pending."
        )
        return 0

    if not args.smoke:
        smoke_state_path = Path(
            "experiments/final_evaluation/logs/timing_smoke/state.json"
        )
        if not smoke_state_path.exists():
            raise SystemExit("Full timing requires the five-trial timing smoke gate.")
        smoke_runs = json.loads(
            smoke_state_path.read_text(encoding="utf-8")
        ).get("runs", {})
        if len(smoke_runs) != 5 or any(
            row.get("status") != "complete" for row in smoke_runs.values()
        ):
            raise SystemExit("The five-trial timing smoke gate is incomplete.")

    for entry in entries:
        record = state["runs"][entry["id"]]
        if record["status"] == "complete":
            continue
        for _ in range(args.max_attempts - int(record["attempts"])):
            active = gpu_processes()
            if active:
                raise SystemExit(
                    "Timing campaign requires exclusive CUDA access; active GPU processes: "
                    + "; ".join(active)
                )
            record["attempts"] += 1
            record["status"] = "running"
            with (logs / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "time": datetime.now(timezone.utc)
                            .replace(microsecond=0)
                            .isoformat(),
                            "event": "start",
                            "id": entry["id"],
                            "attempt": record["attempts"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            write_json(state_path, state)
            environment = dict(os.environ)
            environment.setdefault("HF_HOME", "/workspace/cache/huggingface")
            environment.update(
                {
                    "HF_HUB_DISABLE_XET": "1",
                    "WANDB_MODE": "offline",
                }
            )
            log_path = logs / f"{entry['id']}.attempt{record['attempts']}.log"
            with log_path.open("a", encoding="utf-8") as handle:
                code = subprocess.call(
                    entry["command"],
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            if code == 0 and complete(
                entry["repetition"],
                entry["position"],
                entry["method"],
                output=output,
                expected_rows=5 if args.smoke else 1000,
                minimum_warmup=5 if args.smoke else 20,
                require_wandb=not args.smoke,
            ):
                record["status"] = "complete"
                with (logs / "events.jsonl").open("a", encoding="utf-8") as event_handle:
                    event_handle.write(
                        json.dumps(
                            {
                                "time": datetime.now(timezone.utc)
                                .replace(microsecond=0)
                                .isoformat(),
                                "event": "complete",
                                "id": entry["id"],
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                write_json(state_path, state)
                break
            record["status"] = "failed"
            record["returncode"] = code
            with (logs / "events.jsonl").open("a", encoding="utf-8") as event_handle:
                event_handle.write(
                    json.dumps(
                        {
                            "time": datetime.now(timezone.utc)
                            .replace(microsecond=0)
                            .isoformat(),
                            "event": "failed",
                            "id": entry["id"],
                            "returncode": code,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            write_json(state_path, state)
        if record["status"] != "complete":
            raise SystemExit(f"Timing trial failed: {entry['id']}")
    write_json(
        logs / "summary.json",
        {
            "registered": len(entries),
            "complete": sum(record["status"] == "complete" for record in state["runs"].values()),
            "serial": True,
            "smoke": args.smoke,
        },
    )
    print(
        f"All {len(entries)} {'timing smoke' if args.smoke else 'timing'} "
        "trials are complete."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
