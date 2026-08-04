#!/usr/bin/env python
"""Verify the selection code epoch and complete factual run metadata.

The first long-running selection wave may have started immediately before
provenance-only config fields were added. This script never changes predictions
or metrics. It requires an exact current-code smoke match, preserves the
original config, verifies the recorded execution command, and only then adds
the missing factual metadata to ``config.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "experiments/final_evaluation"
RUN_ROOT = FINAL / "runs/ug"
DEFAULT_SMOKE_ROOT = FINAL / "selection_equivalence"
EXPECTED_ROWS = 8811
CELLS = (
    ("query_max", "more_context"),
    ("query_max", "more_anchors"),
    ("query_pooled", "more_context"),
    ("query_pooled", "more_anchors"),
)
GROUNDING_PROMPT = (
    'Identify the target described by this referring expression: "{query}".\n'
    "Return exactly one bounding box in normalized 0-1 image coordinates as "
    "[x_min, y_min, x_max, y_max]. Do not include any other text."
)
METRIC_DEFINITION = (
    "Parse one predicted box and compute mean IoU, median IoU, and Acc@0.5 "
    "against manifest gt_box."
)
METADATA = {
    "model/decoding": "deterministic_greedy",
    "model/max_new_tokens": 48,
    "prompt/template": GROUNDING_PROMPT,
    "evaluator/name": "box_iou_acc_at_0_5",
    "evaluator/metric_definition": METRIC_DEFINITION,
}


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing required artefact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Missing required artefact: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_name(query_scoring: str, role: str) -> str:
    return (
        "final_selection_refcoco_val_k20_b48_"
        f"{query_scoring}_{role}_bf16"
    )


def smoke_name(query_scoring: str, role: str) -> str:
    return f"current_code_{query_scoring}_{role}"


def option_value(command: str, option: str) -> str | None:
    tokens = shlex.split(command)
    positions = [index for index, token in enumerate(tokens) if token == option]
    if len(positions) != 1:
        return None
    index = positions[0]
    return tokens[index + 1] if index + 1 < len(tokens) else None


def semantic_signature(row: dict[str, Any]) -> dict[str, Any]:
    """Fields that must be identical for a provenance-only code change."""
    return {
        "sample_id": str(row.get("sample_id")),
        "raw_model_output": row.get("raw_model_output"),
        "generated_token_count": row.get("generated_token_count"),
        "retained_indices": row.get("retained_indices"),
        "retained_indices_checksum": row.get("retained_indices_checksum"),
        "raw_parsed_box": row.get("raw_parsed_box"),
        "predicted_box": row.get("predicted_box"),
        "box_iou": row.get("box_iou"),
        "success_iou_0_5": row.get("success_iou_0_5"),
        "actual_llm_visual_tokens": row.get("actual_llm_visual_tokens"),
    }


def smoke_commands(smoke_root: Path, examples: int) -> list[str]:
    commands = []
    for query_scoring, role in CELLS:
        commands.append(
            shlex.join(
                [
                    "/usr/bin/python",
                    "scripts/run_ug_prune_single_pass.py",
                    "--dataset",
                    "refcoco",
                    "--task",
                    "grounding",
                    "--manifest",
                    "data/manifests/final_evaluation/refcoco_val_unique.jsonl",
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
                    query_scoring,
                    "--ug-fusion-mode",
                    "single_layer",
                    "--precision",
                    "bf16",
                    "--limit",
                    str(examples),
                    "--run-name",
                    smoke_name(query_scoring, role),
                    "--output-root",
                    str(smoke_root),
                    "--figures-root",
                    str(smoke_root / "figures"),
                    "--tables-root",
                    str(smoke_root / "tables"),
                    "--max-figures",
                    "0",
                    "--no-wandb",
                ]
            )
        )
    return commands


def verify_command(command: str, query_scoring: str, role: str) -> None:
    expected = {
        "--dataset": "refcoco",
        "--task": "grounding",
        "--manifest": "data/manifests/final_evaluation/refcoco_val_unique.jsonl",
        "--split": "val",
        "--target-visual-tokens": "48",
        "--prune-after-layer": "20",
        "--position-mode": "gap_preserving",
        "--decode-mode": "layer_aware_cache",
        "--ug-role-preset": role,
        "--ug-map-source": query_scoring,
        "--ug-fusion-mode": "single_layer",
        "--precision": "bf16",
    }
    wrong = {
        option: (option_value(command, option), value)
        for option, value in expected.items()
        if option_value(command, option) != value
    }
    if wrong:
        raise RuntimeError(
            f"{run_name(query_scoring, role)}: recorded command mismatch: {wrong}"
        )


def verify_cell(
    state: dict[str, Any],
    smoke_root: Path,
    query_scoring: str,
    role: str,
    examples: int,
) -> dict[str, Any]:
    name = run_name(query_scoring, role)
    record = state.get("runs", {}).get(name)
    if not record or record.get("status") != "complete":
        raise RuntimeError(f"{name}: selection run is not complete.")
    command = str(record.get("command") or "")
    verify_command(command, query_scoring, role)

    run_dir = RUN_ROOT / name
    smoke_dir = smoke_root / smoke_name(query_scoring, role)
    result_path = run_dir / "per_sample_results.jsonl"
    smoke_path = smoke_dir / "per_sample_results.jsonl"
    rows = read_jsonl(result_path)
    smoke_rows = read_jsonl(smoke_path)
    if len(rows) != EXPECTED_ROWS or len(smoke_rows) != examples:
        raise RuntimeError(
            f"{name}: expected {EXPECTED_ROWS} full and {examples} smoke rows; "
            f"found {len(rows)} and {len(smoke_rows)}."
        )
    mismatches = [
        index
        for index, (full_row, smoke_row) in enumerate(
            zip(rows[:examples], smoke_rows, strict=True)
        )
        if semantic_signature(full_row) != semantic_signature(smoke_row)
    ]
    if mismatches:
        raise RuntimeError(
            f"{name}: current-code smoke differs at row indices {mismatches[:10]}."
        )

    config_path = run_dir / "config.json"
    config = read_json(config_path)
    smoke_config = read_json(smoke_dir / "config.json")
    for key, expected in METADATA.items():
        smoke_value = smoke_config.get(key)
        if smoke_value != expected:
            raise RuntimeError(
                f"{smoke_dir}: current-code {key}={smoke_value!r}, expected {expected!r}."
            )
        existing = config.get(key)
        if existing is not None and existing != expected:
            raise RuntimeError(
                f"{run_dir}: conflicting historical {key}={existing!r}, expected {expected!r}."
            )

    original_hash = sha256(config_path)
    preserved = run_dir / "config.pre_provenance_annotation.json"
    yaml_path = run_dir / "config.yaml"
    preserved_yaml = run_dir / "config.pre_provenance_annotation.yaml"
    if not preserved.exists():
        write_json(preserved, config)
    elif not config.get("provenance/selection_code_epoch_verified"):
        if read_json(preserved) != config:
            raise RuntimeError(f"{run_dir}: preserved config does not match current config.")
    if yaml_path.exists() and not preserved_yaml.exists():
        shutil.copyfile(yaml_path, preserved_yaml)

    config.update(METADATA)
    config.update(
        {
            "provenance/selection_code_epoch_verified": True,
            "provenance/selection_equivalence_examples": examples,
            "provenance/selection_equivalence_smoke": str(
                smoke_dir.relative_to(ROOT)
            ),
            "provenance/metadata_annotation_reason": (
                "Provenance-only fields verified from the recorded command, task defaults, "
                "and exact current-code semantic smoke; predictions and metrics are unchanged."
            ),
            "provenance/recorded_execution_command": command,
        }
    )
    write_json(config_path, config)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    return {
        "cell": f"{query_scoring}+{role}",
        "run": name,
        "run_results_sha256": sha256(result_path),
        "smoke_results_sha256": sha256(smoke_path),
        "semantic_examples_compared": examples,
        "semantic_match": True,
        "config_original_sha256": original_hash,
        "config_preserved_sha256": sha256(preserved),
        "config_yaml_preserved_sha256": (
            sha256(preserved_yaml) if preserved_yaml.exists() else None
        ),
        "config_annotated_sha256": sha256(config_path),
        "recorded_command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument(
        "--print-smoke-commands",
        action="store_true",
        help="Print the four current-code smoke commands and do not verify or annotate.",
    )
    args = parser.parse_args()
    if not 5 <= args.examples <= 100:
        raise SystemExit("--examples must be between 5 and 100.")
    if args.print_smoke_commands:
        print("\n".join(smoke_commands(args.smoke_root, args.examples)))
        return 0

    state = read_json(FINAL / "logs/selection/state.json")
    cells = [
        verify_cell(
            state,
            args.smoke_root,
            query_scoring,
            role,
            args.examples,
        )
        for query_scoring, role in CELLS
    ]
    sources = {
        relative: sha256(ROOT / relative)
        for relative in (
            "scripts/run_ug_prune_single_pass.py",
            "vlm_pruning/llava_hooks/mid_layer_pruning.py",
            "vlm_pruning/selectors/ug_prune.py",
        )
    }
    report = {
        "status": "passed",
        "verified_at_utc": utc_now(),
        "examples_per_cell": args.examples,
        "metadata_added": METADATA,
        "source_sha256": sources,
        "cells": cells,
        "prediction_or_metric_files_modified": False,
    }
    write_json(FINAL / "protocol/selection_code_epoch_equivalence.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
