#!/usr/bin/env python
"""Validate the registered RefCOCO factorial and freeze its winner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "experiments/final_evaluation/runs/ug"
OUTPUT = ROOT / "experiments/final_evaluation"
EXPECTED = 8811
EXPECTED_METADATA = {
    "model/decoding": "deterministic_greedy",
    "model/max_new_tokens": 48,
    "prompt/template": (
        'Identify the target described by this referring expression: "{query}".\n'
        "Return exactly one bounding box in normalized 0-1 image coordinates as "
        "[x_min, y_min, x_max, y_max]. Do not include any other text."
    ),
    "evaluator/name": "box_iou_acc_at_0_5",
    "evaluator/metric_definition": (
        "Parse one predicted box and compute mean IoU, median IoU, and Acc@0.5 "
        "against manifest gt_box."
    ),
}
CELLS = (
    ("query_max", "more_context"),
    ("query_max", "more_anchors"),
    ("query_pooled", "more_context"),
    ("query_pooled", "more_anchors"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def interval(values: np.ndarray, rng: np.random.Generator, replicates: int = 10_000) -> list[float]:
    count = len(values)
    samples = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 250):
        size = min(250, replicates - start)
        indices = rng.integers(0, count, size=(size, count))
        samples[start : start + size] = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return [float(low), float(high)]


def main() -> int:
    equivalence_path = (
        OUTPUT / "protocol/selection_code_epoch_equivalence.json"
    )
    equivalence = read_json(equivalence_path)
    if equivalence.get("status") != "passed":
        raise RuntimeError("Selection code-epoch equivalence has not passed.")
    for relative, expected_hash in equivalence.get("source_sha256", {}).items():
        source = ROOT / relative
        if not source.exists() or sha256(source) != expected_hash:
            raise RuntimeError(
                f"Selection equivalence source changed after verification: {relative}."
            )
    manifest_rows = read_jsonl(
        ROOT / "data/manifests/final_evaluation/refcoco_val_unique.jsonl"
    )
    manifest_ids = [str(row["sample_id"]) for row in manifest_rows]
    rows_by_cell: dict[str, list[dict[str, Any]]] = {}
    table: list[dict[str, Any]] = []
    for map_source, role in CELLS:
        name = f"final_selection_refcoco_val_k20_b48_{map_source}_{role}_bf16"
        run_dir = RUN_ROOT / name
        config = read_json(run_dir / "config.json")
        metrics = read_json(run_dir / "metrics.json")
        rows = read_jsonl(run_dir / "per_sample_results.jsonl")
        ids = [str(row["sample_id"]) for row in rows]
        if len(rows) != EXPECTED or len(set(ids)) != EXPECTED or ids != manifest_ids:
            raise RuntimeError(f"{name}: expected {EXPECTED} unique rows, found {len(rows)}/{len(set(ids))}.")
        if not all(
            int(row.get("target_visual_tokens", -1)) == 48
            and int(row.get("actual_llm_visual_tokens", -1)) == 48
            and len(row.get("retained_indices") or []) == 48
            and float(row.get("selection/exact_budget") or 0.0) == 1.0
            for row in rows
        ):
            raise RuntimeError(f"{name}: exact 48-token assertion failed.")
        manifest_path = ROOT / str(config.get("dataset/manifest") or "")
        if (
            config.get("dataset/manifest") != "data/manifests/final_evaluation/refcoco_val_unique.jsonl"
            or not manifest_path.exists()
            or not sha256(manifest_path).startswith(
                str(config.get("dataset/manifest_checksum") or "")
            )
        ):
            raise RuntimeError(f"{name}: registered manifest path/checksum validation failed.")
        expected_config = {
            "ug/map_source": map_source,
            "ug/role_preset": role,
            "ug/prune_after_layer": 20,
            "ug/fusion_mode": "single_layer",
            "ug/probe_layers": "20",
            "position_mode": "gap_preserving",
            "tokens/target_visual_tokens": 48,
            "model/name": "llava-hf/llava-1.5-7b-hf",
            "model/precision": "bf16",
            **EXPECTED_METADATA,
            "provenance/selection_code_epoch_verified": True,
        }
        for key, expected in expected_config.items():
            if config.get(key) != expected:
                raise RuntimeError(f"{name}: {key}={config.get(key)!r}, expected {expected!r}.")
        cell = f"{map_source}+{role}"
        rows_by_cell[cell] = rows
        acc = sum(bool(row.get("success_iou_0_5")) for row in rows) / len(rows)
        mean_iou = sum(float(row.get("box_iou") or 0.0) for row in rows) / len(rows)
        table.append(
            {
                "cell": cell,
                "query_scoring": map_source,
                "role_allocation": role,
                "acc_at_0_5": acc,
                "mean_iou": mean_iou,
                "full_run_selector_time_diagnostic_s": float(
                    metrics["efficiency/pruning_selection_latency_s"]
                ),
                "samples": len(rows),
                "run_dir": str(run_dir.relative_to(ROOT)),
                "per_sample_sha256": sha256(run_dir / "per_sample_results.jsonl"),
            }
        )

    table.sort(key=lambda row: (-row["acc_at_0_5"], -row["mean_iou"], row["cell"]))
    best_acc = table[0]["acc_at_0_5"]
    tied = [row for row in table if row["acc_at_0_5"] == best_acc]
    best_iou = max(row["mean_iou"] for row in tied)
    tied = [row for row in tied if row["mean_iou"] == best_iou]
    tie_break_report = None
    if len(tied) > 1:
        tie_path = OUTPUT / "protocol/selection_tie_break_overheads.json"
        if not tie_path.exists():
            raise RuntimeError(
                "Accuracy and mIoU are exactly tied. Run the registered common-subset "
                "selector-overhead diagnostic and write selection_tie_break_overheads.json."
            )
        tie_break_report = read_json(tie_path)
        overheads = tie_break_report.get("selector_overhead_s", {})
        if any(row["cell"] not in overheads for row in tied):
            raise RuntimeError("Tie-break overhead file does not contain every tied cell.")
        minimum = min(float(overheads[row["cell"]]) for row in tied)
        tied = [
            row for row in tied if float(overheads[row["cell"]]) == minimum
        ]
    winner = next(
        (
            row
            for row in tied
            if row["cell"] == "query_max+more_context"
        ),
        tied[0],
    )
    table.sort(
        key=lambda row: (
            0 if row["cell"] == winner["cell"] else 1,
            -row["acc_at_0_5"],
            -row["mean_iou"],
            row["cell"],
        )
    )
    winner_rows = rows_by_cell[winner["cell"]]
    winner_ids = [str(row["sample_id"]) for row in winner_rows]
    rng = np.random.default_rng(20260729)
    paired: list[dict[str, Any]] = []
    winner_by_id = {str(row["sample_id"]): row for row in winner_rows}
    for contender in table[1:]:
        other = {str(row["sample_id"]): row for row in rows_by_cell[contender["cell"]]}
        if set(other) != set(winner_ids):
            raise RuntimeError(f"{contender['cell']}: paired IDs do not align with winner.")
        acc_delta = np.asarray(
            [
                float(bool(winner_by_id[sid]["success_iou_0_5"]))
                - float(bool(other[sid]["success_iou_0_5"]))
                for sid in winner_ids
            ]
        )
        iou_delta = np.asarray(
            [float(winner_by_id[sid]["box_iou"]) - float(other[sid]["box_iou"]) for sid in winner_ids]
        )
        paired.append(
            {
                "winner_minus": contender["cell"],
                "acc_delta": float(acc_delta.mean()),
                "acc_delta_95_interval": interval(acc_delta, rng),
                "miou_delta": float(iou_delta.mean()),
                "miou_delta_95_interval": interval(iou_delta, rng),
            }
        )

    winner_config = read_json(ROOT / winner["run_dir"] / "config.json")
    protocol_path = ROOT / "experiments/final_evaluation/protocol/protocol.json"
    protocol = read_json(protocol_path)
    if (
        protocol["repository"].get(
            "selection_code_epoch_equivalence_sha256"
        )
        != sha256(equivalence_path)
    ):
        raise RuntimeError(
            "Registered protocol does not contain the verified selection "
            "code-epoch report hash."
        )
    frozen = {
        "status": "frozen",
        "selection_dataset": "RefCOCO validation",
        "selection_rule": [
            "highest Acc@0.5",
            "then highest mIoU",
            "then lower selector overhead",
            "then query-max + more-context",
        ],
        "winner": winner,
        "method": {
            "prune_after_layer": 20,
            "visual_token_budget": 48,
            "query_scoring": winner_config["ug/map_source"],
            "role_allocation": winner_config["ug/role_preset"],
            "target_ratio": winner_config["ug/target_ratio"],
            "context_ratio": winner_config["ug/context_ratio"],
            "anchor_ratio": winner_config["ug/anchor_ratio"],
            "fusion": "single_layer",
            "probe_layer": 20,
            "position_mode": "gap_preserving",
            "decode_mode": "layer_aware_cache",
            "precision": "bf16",
            "checkpoint": winner_config["model/name"],
            "prompt": protocol["prompts"]["grounding"],
            "max_new_tokens": protocol["max_new_tokens"]["grounding"],
            "decoding": "deterministic greedy",
        },
        "manifest": winner_config["dataset/manifest"],
        "manifest_checksum": winner_config["dataset/manifest_checksum"],
        "source_config_sha256": sha256(ROOT / winner["run_dir"] / "config.json"),
        "source_results_sha256": winner["per_sample_sha256"],
        "registered_protocol_sha256": sha256(protocol_path),
        "registered_source_sha256": protocol["repository"]["source_sha256"],
        "selection_code_epoch_equivalence_sha256": sha256(equivalence_path),
    }
    (OUTPUT / "frozen_method.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "selection_rule_applied": True,
        "cells_ranked": table,
        "paired_intervals": paired,
        "winner": winner["cell"],
        "tie_break_report": tie_break_report,
    }
    (OUTPUT / "selection_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
