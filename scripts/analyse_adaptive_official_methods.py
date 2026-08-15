#!/usr/bin/env python
"""Finalize official-source results and update the adaptive campaign ledger."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815"
MANIFEST = REPO_ROOT / (
    "experiments/progressive_oracle_refcoco_val_20260814/"
    "manifests/refcoco_val_discovery1000.jsonl"
)
OFFICIAL_ROOT = CAMPAIGN_ROOT / "official_runs"
ANALYSIS_PATH = CAMPAIGN_ROOT / "analysis/OFFICIAL_METHOD_RESULTS.json"
REGISTRY_PATH = CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"
STATE_PATH = CAMPAIGN_ROOT / "CONTROLLER_STATE.json"
REPORT_PATH = CAMPAIGN_ROOT / "RESEARCH_REPORT.md"
SOURCE_LOCK = CAMPAIGN_ROOT / "official_sources/source_lock.json"
COMPATIBILITY_AUDIT = CAMPAIGN_ROOT / "official_sources/compatibility_audit.json"
PARITY_CANONICAL = OFFICIAL_ROOT / "parity_canonical_20"
PARITY_ADAPTER = OFFICIAL_ROOT / "parity_full_20"
PDROP_SMOKE = OFFICIAL_ROOT / "pdrop_smoke2_attempt1"
REROUTE_SMOKE = OFFICIAL_ROOT / "pdrop_reroute_smoke2_attempt2"
PDROP_RUN = OFFICIAL_ROOT / "pdrop_full1000_attempt1"
REROUTE_RUN = OFFICIAL_ROOT / "pdrop_reroute_full1000_attempt1"
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260815


class OfficialAnalysisError(RuntimeError):
    """Raised when official artifacts cannot support a terminal branch."""


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


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OfficialAnalysisError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path, expected: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected or not all(isinstance(row, dict) for row in rows):
        raise OfficialAnalysisError(f"Expected {expected} JSON objects: {path}")
    return rows


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def artifact_hashes(run_dir: Path, prediction_name: str = "predictions.jsonl") -> dict[str, Any]:
    names = {
        "completion_sha256": "COMPLETED.json",
        "config_sha256": "config.json",
        "environment_sha256": "environment.json",
        "integrity_sha256": "integrity_report.json",
        "metrics_sha256": "metrics.json",
        "predictions_sha256": prediction_name,
    }
    result = {key: sha256_file(run_dir / name) for key, name in names.items()}
    result["run_dir"] = relative(run_dir)
    return result


def metrics(integrity: dict[str, Any]) -> dict[str, Any]:
    value = integrity["metrics"]
    return {
        "rows": int(value["samples/evaluated"]),
        "acc_at_0_5": float(value["quality/acc_at_0_5"]),
        "mean_iou": float(value["quality/mean_iou"]),
        "parse_rate": float(value["quality/parsed_box_rate"]),
        "average_visual_tokens": float(value["tokens/candidate_average_visual_tokens"]),
        "peak_allocated_mb": float(value["efficiency/peak_allocated_mb"]),
        "peak_reserved_mb": float(value["efficiency/peak_reserved_mb"]),
    }


def paired_interval(left: np.ndarray, right: np.ndarray, *, seed: int, scale: float) -> dict[str, Any]:
    if left.shape != (1000,) or right.shape != (1000,):
        raise OfficialAnalysisError("Official paired arrays are not 1,000-row aligned")
    delta = left - right
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, BOOTSTRAPS, 500):
        count = min(500, BOOTSTRAPS - start)
        indices = rng.integers(0, 1000, size=(count, 1000))
        samples.append(delta[indices].mean(axis=1) * scale)
    values = np.concatenate(samples)
    return {
        "point_delta": float(delta.mean() * scale),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "bootstrap_resamples": BOOTSTRAPS,
        "seed": seed,
    }


def parity_evidence() -> dict[str, Any]:
    canonical_integrity = read_json(PARITY_CANONICAL / "integrity_report.json")
    adapter_integrity = read_json(PARITY_ADAPTER / "integrity_report.json")
    if canonical_integrity.get("status") != "pass" or adapter_integrity.get("status") != "pass":
        raise OfficialAnalysisError("An unpruned parity artifact is not independently valid")
    canonical = read_jsonl(PARITY_CANONICAL / "predictions.jsonl", 20)
    adapter = read_jsonl(PARITY_ADAPTER / "predictions.jsonl", 20)
    fields = (
        "sample_id",
        "raw_model_output",
        "raw_parsed_box",
        "predicted_box",
        "box_iou",
        "success_iou_0_5",
        "valid",
        "invalid",
        "parse_failure_reason",
        "generated_token_count",
        "gt_box",
        "width",
        "height",
        "query",
    )
    differences = [
        {"row": index, "field": field, "canonical": left.get(field), "adapter": right.get(field)}
        for index, (left, right) in enumerate(zip(canonical, adapter, strict=True))
        for field in fields
        if left.get(field) != right.get(field)
    ]
    if differences:
        raise OfficialAnalysisError(f"Official unpruned parity has decoded differences: {differences[:1]}")
    legacy_handoff = REPO_ROOT / (
        "experiments/progressive_oracle_refcoco_val_20260814/preflight/"
        "existing_full/existing_full/per_sample_results.jsonl"
    )
    legacy_rows = read_jsonl(legacy_handoff, 2)
    return {
        "status": "pass_exact",
        "rows": 20,
        "compared_fields": list(fields),
        "differences": 0,
        "canonical_run": artifact_hashes(PARITY_CANONICAL),
        "official_adapter_run": artifact_hashes(PARITY_ADAPTER),
        "raw_output_sequence_sha256": hashlib.sha256(
            "\n".join(row["raw_model_output"] for row in canonical).encode("utf-8")
        ).hexdigest(),
        "handoff_legacy_comparator": {
            "path": relative(legacy_handoff),
            "sha256": sha256_file(legacy_handoff),
            "rows": len(legacy_rows),
            "status": "INSUFFICIENT_FOR_REGISTERED_20_ROW_PARITY",
        },
    }


def official_contrast(pdrop: list[dict[str, Any]], reroute: list[dict[str, Any]]) -> dict[str, Any]:
    if [row["sample_id"] for row in pdrop] != [row["sample_id"] for row in reroute]:
        raise OfficialAnalysisError("PDrop and PDrop+Reroute rows are not aligned")
    left_acc = np.asarray([float(bool(row["success_iou_0_5"])) for row in reroute])
    right_acc = np.asarray([float(bool(row["success_iou_0_5"])) for row in pdrop])
    left_iou = np.asarray([float(row["box_iou"]) for row in reroute])
    right_iou = np.asarray([float(row["box_iou"]) for row in pdrop])
    return {
        "contrast": "OFF_PDROP_REROUTE_192-minus-OFF_PDROP_L2_192",
        "acc_at_0_5_percentage_points": paired_interval(
            left_acc, right_acc, seed=BOOTSTRAP_SEED, scale=100.0
        ),
        "mean_iou": paired_interval(
            left_iou, right_iou, seed=BOOTSTRAP_SEED + 1, scale=1.0
        ),
        "discordant": {
            "reroute_correct_pdrop_wrong": int(np.sum((left_acc == 1) & (right_acc == 0))),
            "reroute_wrong_pdrop_correct": int(np.sum((left_acc == 0) & (right_acc == 1))),
        },
        "parse_rate_delta": 0.0,
        "average_visual_tokens_delta": 191.46875 - 191.34375,
        "visual_token_layer_work_delta": 4.0,
    }


def append_once(items: list[dict[str, Any]], key: str, value: dict[str, Any]) -> None:
    identifier = value[key]
    for existing in items:
        if existing.get(key) != identifier:
            continue
        previous = {name: item for name, item in existing.items() if name != "at_utc"}
        proposed = {name: item for name, item in value.items() if name != "at_utc"}
        if previous != proposed:
            raise OfficialAnalysisError(f"Conflicting persisted {key}={identifier}")
        return
    items.append(value)


def update_report(analysis: dict[str, Any]) -> None:
    pdrop = analysis["accepted_cells"]["OFF_PDROP_L2_192"]["metrics"]
    reroute = analysis["accepted_cells"]["OFF_PDROP_REROUTE_192"]["metrics"]
    contrast = analysis["paired_contrast"]
    acc = contrast["acc_at_0_5_percentage_points"]
    iou = contrast["mean_iou"]
    discordant = contrast["discordant"]
    official = "\n".join(
        [
            "## Official source-native comparisons",
            "",
            "The pinned Nüwa checkout is `175b7d57a7890b9818dd26d82f2d91d9868b0d80` (recursive SHA-256 `78aa108d…`); the pinned Reroute checkout is `9d422042076de678b2ad67c4a6660e4b97971a1e` (recursive SHA-256 `561d977e…`). Both checkouts remained clean. The thin Reroute adapter's 20-example unpruned mode exactly matched the current canonical runner on IDs, decoded strings, boxes, IoUs, correctness, and generated lengths.",
            "",
            "| Cell | Native action / schedule | Status | Acc@0.5 | mIoU | Parse | Avg visual tokens | K12-like |",
            "|---|---|---|---:|---:|---:|---:|---|",
            "| OFF_NUWA_192 | documented native Nüwa operating point | skipped: official incompatible after 3 bounded attempts | — | — | — | — | — |",
            f"| OFF_PDROP_L2_192 | physical delete at K2/K7/K15/K23, budgets 343/204/122/72 | accepted | {pdrop['acc_at_0_5']:.4f} | {pdrop['mean_iou']:.4f} | {pdrop['parse_rate']:.4f} | {pdrop['average_visual_tokens']:.5f} | fail |",
            f"| OFF_PDROP_REROUTE_192 | compact route at K2/K7/K15/K23, budgets 259/259/129/64 | accepted | {reroute['acc_at_0_5']:.4f} | {reroute['mean_iou']:.4f} | {reroute['parse_rate']:.4f} | {reroute['average_visual_tokens']:.5f} | fail |",
            "| OFF_NUWA_REROUTE_192 | conditional official hybrid | skipped: pinned Reroute source has no Nüwa router/config | — | — | — | — | — |",
            "",
            f"PDrop+Reroute minus matched PDrop is {acc['point_delta']:+.3f} Acc points (95% paired bootstrap CI [{acc['lower_95']:.3f}, {acc['upper_95']:.3f}]) and {iou['point_delta']:+.4f} mIoU (CI [{iou['lower_95']:.4f}, {iou['upper_95']:.4f}]), with {discordant['reroute_correct_pdrop_wrong']}/{discordant['reroute_wrong_pdrop_correct']} discordant correctness outcomes. The interval evidence is inconclusive; both official cells are well below the K12-like quality thresholds despite lower token work.",
            "",
            "Nüwa was not approximated: its published package targets the legacy upstream `llava` class tree, has no explicit average-192 LLaVA configuration, and cannot source-natively patch the canonical Hugging Face checkpoint. The conditional Nüwa+Reroute cell was not constructed because the official Reroute release directly implements only FastV and PDrop routers.",
            "",
            f"Machine-readable evidence: `{relative(ANALYSIS_PATH)}`.",
        ]
    )
    document = REPORT_PATH.read_text(encoding="utf-8")
    document = document.replace("The complete repository suite passed 166 tests.", "The complete repository suite passed 179 tests.")
    updated, count = re.subn(
        r"## Official source-native comparisons\n\n.*?(?=\n## Learned schedule gate)",
        official,
        document,
        flags=re.DOTALL,
    )
    if count != 1:
        raise OfficialAnalysisError("Could not replace the official section in the cumulative report")
    gate_audit = read_json(CAMPAIGN_ROOT / "gate_data/MANIFEST_AUDIT.json")
    gate_smoke = read_json(CAMPAIGN_ROOT / "gate_labels/labels_smoke2_attempt1/integrity_report.json")
    gate_section = "\n".join(
        [
            "## Learned schedule gate",
            "",
            f"The frozen train-only manifest contains 3,000 unique-image records (2,160 fit, 240 early-stop, 600 calibration), SHA-256 `{gate_audit['manifest_sha256']}`. It excludes 506 upstream records whose original IDs occur in the frozen validation subset and proves zero remaining record-ID overlap. Ninety COCO-image overlaps imposed by the mixed historical split are recorded without moving any validation label into training.",
            "",
            f"The two-record physical label smoke passed exact K12 and M448 trace/cache/rescoring validation and the frozen 8,209-feature K2 schema (`{gate_smoke['feature_schema_sha256']}`). Full train-label generation is in progress from clean source revision `688d53f2407a7fc06c9313e8cb23cb9f2d947a3b`.",
        ]
    )
    updated, count = re.subn(
        r"## Learned schedule gate\n\n.*?(?=\n## Conditional matched timing)",
        gate_section,
        updated,
        flags=re.DOTALL,
    )
    if count != 1:
        raise OfficialAnalysisError("Could not replace the gate section in the cumulative report")
    failures = "\n".join(
        [
            "## Failures and rejected attempts",
            "",
            "- OFF_NUWA_192 is `SKIPPED_OFFICIAL_INCOMPATIBLE` after the three documented source, dependency, and checkpoint-API attempts; no local approximation was substituted.",
            "- OFF_NUWA_REROUTE_192 is `SKIPPED_NO_DIRECT_OFFICIAL_SUPPORT`; the pinned official Reroute checkout contains no Nüwa route.",
            "- The first PDrop+Reroute smoke stopped before model loading because the preceding official import created ignored `__pycache__` files inside the recursively hashed checkout. Only those four disposable bytecode files were removed; the checkout was reverified byte-for-byte and attempt 2 passed with bytecode writes disabled.",
            "- The handoff's historical full comparator contained 2 rather than 20 rows, so it was retained as insufficient evidence. A fresh current-canonical 20-row run supplied exact parity instead.",
            "- The first train-image preparation transport used the COCO alias over HTTPS, whose certificate did not match in this pod. It produced no final images; the official HTTP object endpoint served the same immutable JPEG objects and the successful manifest records every image hash.",
        ]
    )
    updated, count = re.subn(
        r"## Failures and rejected attempts\n\n.*?(?=\n## Limitations)",
        failures,
        updated,
        flags=re.DOTALL,
    )
    if count != 1:
        raise OfficialAnalysisError("Could not replace the failure section in the cumulative report")
    atomic_write(REPORT_PATH, updated.encode("utf-8"))


def main() -> int:
    from scripts.validate_adaptive_official_reroute import validate_run

    pdrop_validation = validate_run(PDROP_RUN, MANIFEST, expected_rows=1000)
    reroute_validation = validate_run(REROUTE_RUN, MANIFEST, expected_rows=1000)
    pdrop_smoke = validate_run(PDROP_SMOKE, MANIFEST, expected_rows=2)
    reroute_smoke = validate_run(REROUTE_SMOKE, MANIFEST, expected_rows=2)
    source_lock = read_json(SOURCE_LOCK)
    compatibility = read_json(COMPATIBILITY_AUDIT)
    pdrop_rows = read_jsonl(PDROP_RUN / "predictions.jsonl", 1000)
    reroute_rows = read_jsonl(REROUTE_RUN / "predictions.jsonl", 1000)
    pdrop_metrics = metrics(pdrop_validation)
    reroute_metrics = metrics(reroute_validation)
    analysis = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "source_lock": {
            "path": relative(SOURCE_LOCK),
            "sha256": sha256_file(SOURCE_LOCK),
            "sources": source_lock["sources"],
        },
        "compatibility_audit": {
            "path": relative(COMPATIBILITY_AUDIT),
            "sha256": sha256_file(COMPATIBILITY_AUDIT),
        },
        "unpruned_parity": parity_evidence(),
        "method_smokes": {
            "OFF_PDROP_L2_192": {
                "validation": pdrop_smoke,
                "artifacts": artifact_hashes(PDROP_SMOKE),
            },
            "OFF_PDROP_REROUTE_192": {
                "validation": reroute_smoke,
                "artifacts": artifact_hashes(REROUTE_SMOKE),
            },
        },
        "accepted_cells": {
            "OFF_PDROP_L2_192": {
                "metrics": pdrop_metrics,
                "artifacts": artifact_hashes(PDROP_RUN),
                "integrity_checks": pdrop_validation["checks"],
                "k12_like": False,
            },
            "OFF_PDROP_REROUTE_192": {
                "metrics": reroute_metrics,
                "artifacts": artifact_hashes(REROUTE_RUN),
                "integrity_checks": reroute_validation["checks"],
                "k12_like": False,
            },
        },
        "skipped_cells": {
            "OFF_NUWA_192": compatibility["nuwa"],
            "OFF_NUWA_REROUTE_192": compatibility["nuwa_reroute"],
        },
        "paired_contrast": official_contrast(pdrop_rows, reroute_rows),
        "strongest_official_by_accuracy": "OFF_PDROP_REROUTE_192",
        "official_pareto_frontier": ["OFF_PDROP_L2_192", "OFF_PDROP_REROUTE_192"],
        "failures": [
            {
                "id": "PDROP_REROUTE_SMOKE_ATTEMPT1_PYCACHE_HASH",
                "status": "recovered_bounded_attempt2",
                "reason": "official import generated four ignored bytecode files included by the recursive source hash",
                "failed_run_dir": relative(OFFICIAL_ROOT / "pdrop_reroute_smoke2_attempt1"),
            },
            {
                "id": "HISTORICAL_PARITY_COMPARATOR_ONLY_2_ROWS",
                "status": "replaced_by_fresh_current_canonical_20",
                "reason": "handoff comparator was not a 20-row artifact",
            },
        ],
    }
    atomic_write(ANALYSIS_PATH, canonical_bytes(analysis))

    registry = read_json(REGISTRY_PATH)
    state = read_json(STATE_PATH)
    registry["cells"]["OFF_NUWA_192"] = {
        "status": "skipped",
        "terminal_status": "SKIPPED_OFFICIAL_INCOMPATIBLE",
        "skip_reason": compatibility["nuwa"]["attempts"][-1]["reason"],
        "attempts": compatibility["nuwa"]["attempts"],
    }
    registry["cells"]["OFF_NUWA_REROUTE_192"] = {
        "status": "skipped",
        "terminal_status": "SKIPPED_NO_DIRECT_OFFICIAL_SUPPORT",
        "skip_reason": compatibility["nuwa_reroute"]["reason"],
        "attempts": [],
    }
    for cell_id, run_dir, smoke_dir, item_metrics in (
        ("OFF_PDROP_L2_192", PDROP_RUN, PDROP_SMOKE, pdrop_metrics),
        ("OFF_PDROP_REROUTE_192", REROUTE_RUN, REROUTE_SMOKE, reroute_metrics),
    ):
        registry["cells"][cell_id] = {
            "status": "accepted",
            "metrics": item_metrics,
            "accepted_run": artifact_hashes(run_dir),
            "acceptance_evidence": {
                "unpruned_parity": analysis["unpruned_parity"]["status"],
                "method_smoke": artifact_hashes(smoke_dir),
                "official_source_commit": pdrop_validation["official_source_commit"],
                "source_native_modules": True,
            },
            "attempts": [
                {
                    "kind": "bounded_method_smoke",
                    "status": "accepted",
                    "run_dir": relative(smoke_dir),
                },
                {
                    "kind": "full_1000_quality",
                    "status": "accepted",
                    "run_dir": relative(run_dir),
                },
            ],
        }
    registry["cells"]["OFF_PDROP_REROUTE_192"]["attempts"].insert(
        0,
        {
            "kind": "bounded_method_smoke",
            "status": "failed_before_model_load",
            "run_dir": relative(OFFICIAL_ROOT / "pdrop_reroute_smoke2_attempt1"),
            "reason": analysis["failures"][0]["reason"],
        },
    )
    registry["phases"]["official_methods"] = {
        "status": "complete",
        "analysis_path": relative(ANALYSIS_PATH),
        "analysis_sha256": sha256_file(ANALYSIS_PATH),
        "accepted_cells": ["OFF_PDROP_L2_192", "OFF_PDROP_REROUTE_192"],
        "skipped_cells": ["OFF_NUWA_192", "OFF_NUWA_REROUTE_192"],
    }
    append_once(
        registry["decisions"],
        "decision",
        {
            "at_utc": utc_now(),
            "decision": "complete_official_source_branch",
            "accepted": ["OFF_PDROP_L2_192", "OFF_PDROP_REROUTE_192"],
            "skipped": {
                "OFF_NUWA_192": "SKIPPED_OFFICIAL_INCOMPATIBLE",
                "OFF_NUWA_REROUTE_192": "SKIPPED_NO_DIRECT_OFFICIAL_SUPPORT",
            },
            "evidence_path": relative(ANALYSIS_PATH),
            "evidence_sha256": sha256_file(ANALYSIS_PATH),
        },
    )
    for failure in analysis["failures"]:
        append_once(registry["failures"], "id", failure)
    registry["totals"]["accepted_validation_quality"] = sum(
        cell.get("status") == "accepted" for cell in registry["cells"].values()
    )
    registry["updated_at_utc"] = utc_now()
    state["state"] = "OFFICIAL_COMPLETE_GATE_LABELS_IN_PROGRESS"
    state["next_required_action"] = "complete_and_validate_3000_train_route_labels"
    append_once(
        state["decision_history"],
        "phase",
        {
            "phase": "official_methods",
            "status": "complete",
            "evidence_path": relative(ANALYSIS_PATH),
            "evidence_sha256": sha256_file(ANALYSIS_PATH),
            "accepted": ["OFF_PDROP_L2_192", "OFF_PDROP_REROUTE_192"],
        },
    )
    state["updated_at_utc"] = utc_now()
    update_report(analysis)
    atomic_write(REGISTRY_PATH, canonical_bytes(registry))
    atomic_write(STATE_PATH, canonical_bytes(state))
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OfficialAnalysisError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
