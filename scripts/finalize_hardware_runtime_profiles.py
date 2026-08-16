#!/usr/bin/env python
"""Independently aggregate the five counterbalanced hardware-profile trials."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for entry in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from run_hardware_runtime_profile import (  # noqa: E402
    CAMPAIGN_ROOT,
    CELL_IDS,
    MANIFEST,
    PROFILE_ORDER_PLAN,
    aggregate,
)
from validate_hardware_runtime_profile import read_json, read_jsonl, validate_run  # noqa: E402
from vlm_pruning.artifacts import sha256_file  # noqa: E402


class ProfileCampaignError(RuntimeError):
    """Raised when the counterbalanced profile campaign is incomplete."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def accepted_trial(repetition: int, position: int, cell_id: str) -> Path:
    pattern = f"rep{repetition}_pos{position}_{cell_id.lower()}_attempt*"
    candidates = sorted((CAMPAIGN_ROOT / "profiling/trials").glob(pattern))
    accepted = [path for path in candidates if (path / "COMPLETED.json").is_file()]
    if len(accepted) != 1:
        raise ProfileCampaignError(
            f"Expected exactly one accepted trial for repetition {repetition}, "
            f"position {position}, {cell_id}; found {len(accepted)}"
        )
    for candidate in candidates:
        if candidate not in accepted and not (candidate / "REJECTION.json").is_file():
            raise ProfileCampaignError(f"Nonaccepted trial lacks REJECTION.json: {candidate}")
    return accepted[0]


def validate_completion_hashes(run_dir: Path) -> dict[str, Any]:
    completion = read_json(run_dir / "COMPLETED.json")
    for filename, key in (
        ("config.json", "config_sha256"),
        ("environment.json", "environment_sha256"),
        ("profiles.jsonl", "profiles_sha256"),
        ("metrics.json", "metrics_sha256"),
        ("integrity_report.json", "integrity_sha256"),
    ):
        if completion.get(key) != sha256_file(run_dir / filename):
            raise ProfileCampaignError(f"Completion hash differs for {run_dir / filename}")
    return completion


def per_row_component_total(
    row: dict[str, Any],
    names: set[str],
    *,
    clock: str = "cuda_event_s",
    phase: str | None = None,
) -> float:
    total = 0.0
    for record in row["runtime_profile"]["records"]:
        if record["component"] not in names:
            continue
        if phase is not None and (record.get("metadata") or {}).get("phase") != phase:
            continue
        value = record.get(clock)
        if value is not None:
            total += float(value)
    return total


def component_groups(rows: list[dict[str, Any]]) -> dict[str, float]:
    definitions = {
        "score_rank": {"score_construction", "ranking_and_index_materialization"},
        "gather_merge": {
            "index_position_mask_gather",
            "merge_assignment",
            "merge_weighted_reduction",
        },
        "cache_segment_dispatch": {
            "cache_construction",
            "segment_cache_transition",
            "cache_proxy_dispatch",
        },
    }
    result = {
        name: sum(per_row_component_total(row, components) for row in rows) / len(rows)
        for name, components in definitions.items()
    }
    result["decoder_prefill"] = sum(
        per_row_component_total(row, {"decoder_segment"}, phase="prefill") for row in rows
    ) / len(rows)
    result["decoder_decode"] = sum(
        per_row_component_total(row, {"decoder_segment"}, phase="decode") for row in rows
    ) / len(rows)
    return result


def run() -> int:
    output_dir = CAMPAIGN_ROOT / "profiling/aggregate"
    if output_dir.exists():
        raise ProfileCampaignError(f"Refusing to overwrite profile aggregate: {output_dir}")
    trials: list[dict[str, Any]] = []
    rows_by_cell: dict[str, list[dict[str, Any]]] = {cell_id: [] for cell_id in CELL_IDS}
    for repetition, order in PROFILE_ORDER_PLAN.items():
        for position, cell_id in enumerate(order, 1):
            run_dir = accepted_trial(repetition, position, cell_id)
            completion = validate_completion_hashes(run_dir)
            independent = validate_run(run_dir, MANIFEST)
            config = read_json(run_dir / "config.json")
            if int(config.get("global_repetition", -1)) != repetition:
                raise ProfileCampaignError(f"Trial repetition differs: {run_dir}")
            if int(config.get("counterbalanced_order_position", -1)) != position:
                raise ProfileCampaignError(f"Trial order position differs: {run_dir}")
            rows = read_jsonl(run_dir / "profiles.jsonl")
            rows_by_cell[cell_id].extend(rows)
            trials.append(
                {
                    "repetition": repetition,
                    "order_position": position,
                    "cell_id": cell_id,
                    "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                    "source_git_commit": config["source_git_commit"],
                    "config_sha256": completion["config_sha256"],
                    "profiles_sha256": completion["profiles_sha256"],
                    "metrics_sha256": completion["metrics_sha256"],
                    "completion_sha256": sha256_file(run_dir / "COMPLETED.json"),
                    "independent_validation": independent,
                }
            )
    aggregates: dict[str, Any] = {}
    groups: dict[str, dict[str, float]] = {}
    torch_profiles: dict[str, Any] = {}
    for cell_id, rows in rows_by_cell.items():
        if len(rows) != 500 or len({row["sample_id"] for row in rows}) != 100:
            raise ProfileCampaignError(f"{cell_id} does not contain five repetitions of 100 samples")
        if {int(row["repetition"]) for row in rows} != set(PROFILE_ORDER_PLAN):
            raise ProfileCampaignError(f"{cell_id} repetition set differs")
        aggregates[cell_id] = aggregate(rows)
        groups[cell_id] = component_groups(rows)
        rep1_trial = next(
            Path(trial["run_dir"])
            for trial in trials
            if trial["cell_id"] == cell_id and trial["repetition"] == 1
        )
        summary_path = REPO_ROOT / rep1_trial / "torch_profiler_summary.json"
        if not summary_path.is_file():
            raise ProfileCampaignError(f"Missing PyTorch-profiler summary for {cell_id}")
        torch_profiles[cell_id] = {
            "path": str(summary_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(summary_path),
            "summary": read_json(summary_path),
        }

    k12 = aggregates["P_K12"]["means"]
    live = aggregates["P_M448"]["means"]
    replay = aggregates["P_M448_REPLAY"]["means"]
    prefill_gap = float(live["prefill_latency_s"]) - float(k12["prefill_latency_s"])
    group_deltas = {
        name: groups["P_M448"][name] - groups["P_K12"][name]
        for name in groups["P_M448"]
    }
    replay_selection_effect = float(live["prefill_latency_s"]) - float(
        replay["prefill_latency_s"]
    )
    decode_gap = float(live["decode_latency_s"]) - float(k12["decode_latency_s"])
    threshold = 0.2 * max(prefill_gap, 0.0)
    triggers = {
        "R_FUSED_GATHER_MERGE": prefill_gap > 0.0
        and max(group_deltas["gather_merge"], 0.0) >= threshold,
        "R_NATIVE_CACHE": prefill_gap > 0.0
        and max(group_deltas["cache_segment_dispatch"], decode_gap, 0.0) >= threshold,
        "R_SCORE_REUSE": prefill_gap > 0.0
        and max(replay_selection_effect, 0.0) >= threshold,
    }
    attribution = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "clock_for_component_attribution": "synchronized_cuda_events",
        "overlap_warning": (
            "Component scopes include nested intervals and are not summed into an additive latency model. "
            "Branch triggers use only their registered isolated or incremental comparison."
        ),
        "m448_minus_k12_prefill_wall_s": prefill_gap,
        "m448_minus_k12_decode_wall_s": decode_gap,
        "live_minus_replay_prefill_wall_s": replay_selection_effect,
        "component_group_means_s": groups,
        "m448_minus_k12_component_group_s": group_deltas,
        "twenty_percent_positive_prefill_gap_s": threshold,
        "trigger_formula": {
            "R_FUSED_GATHER_MERGE": "positive M448-minus-K12 gather/merge CUDA delta >= 20% positive wall prefill gap",
            "R_NATIVE_CACHE": "max(positive cache/dispatch CUDA delta, positive natural decode wall delta) >= 20% positive wall prefill gap",
            "R_SCORE_REUSE": "positive live-minus-replay wall prefill delta >= 20% positive wall prefill gap",
        },
        "repair_triggers": triggers,
    }
    output_dir.mkdir(parents=True)
    atomic_write(output_dir / "TRIALS.json", canonical_bytes({"trials": trials}))
    atomic_write(output_dir / "PROFILE_RESULTS.json", canonical_bytes(aggregates))
    atomic_write(output_dir / "PYTORCH_PROFILES.json", canonical_bytes(torch_profiles))
    atomic_write(output_dir / "ATTRIBUTION.json", canonical_bytes(attribution))
    completion = {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "rows_per_cell": 500,
        "unique_samples_per_cell": 100,
        "repetitions": list(PROFILE_ORDER_PLAN),
        "trial_count": len(trials),
        "trials_sha256": sha256_file(output_dir / "TRIALS.json"),
        "profile_results_sha256": sha256_file(output_dir / "PROFILE_RESULTS.json"),
        "pytorch_profiles_sha256": sha256_file(output_dir / "PYTORCH_PROFILES.json"),
        "attribution_sha256": sha256_file(output_dir / "ATTRIBUTION.json"),
    }
    atomic_write(output_dir / "COMPLETED.json", canonical_bytes(completion))
    print(json.dumps({"status": "complete", "attribution": attribution}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    try:
        return run()
    except (ProfileCampaignError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
