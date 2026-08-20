#!/usr/bin/env python3
"""Build immutable acceptance ledgers and a cumulative outcome-faithful held-out report."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
for value in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from frozen_z_heldout_common import CAMPAIGN, REPO, HeldoutError, atomic_json, now, read_json, read_jsonl, sha256


def artifact_record(run_dir: Path) -> dict[str, Any]:
    required = ["config.json", "metrics.json", "integrity_report.json", "COMPLETED.json"]
    prediction_names = [name for name in ("predictions.jsonl", "predictions.raw.jsonl", "predictions.scored.jsonl") if (run_dir / name).is_file()]
    environment_names = sorted(path.name for path in run_dir.glob("environment*.json"))
    names = required + prediction_names + environment_names
    if any(not (run_dir / name).is_file() for name in names) or not prediction_names or not environment_names:
        raise HeldoutError(f"Accepted cell artifact set is incomplete: {run_dir}")
    optional = [name for name in ("SEALED_RAW.json", "integrity_raw.json") if (run_dir / name).is_file()]
    names.extend(optional)
    return {
        "path": str(run_dir.relative_to(REPO)),
        "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": sha256(run_dir / name)} for name in sorted(names)},
    }


def verify_chain(rows: list[dict[str, Any]]) -> str | None:
    previous = None
    for index, row in enumerate(rows):
        if row.get("previous_entry_sha256") != previous:
            raise HeldoutError(f"Decision-ledger chain differs at row {index}")
        payload = dict(row)
        claimed = str(payload.pop("entry_sha256"))
        actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
        if claimed != actual:
            raise HeldoutError(f"Decision-ledger entry hash differs at row {index}")
        previous = claimed
    return previous


def append_event(rows: list[dict[str, Any]], event: str, evidence_path: Path, note: str, extra: dict[str, Any] | None = None) -> None:
    if any(row.get("event") == event for row in rows):
        raise HeldoutError(f"Decision-ledger event already exists: {event}")
    relative = str(evidence_path.relative_to(CAMPAIGN))
    entry: dict[str, Any] = {
        "event": event, "recorded_at_utc": now(), "evidence_path": relative,
        "evidence_sha256": sha256(evidence_path), "note": note,
        "previous_entry_sha256": verify_chain(rows),
    }
    if extra:
        entry.update(extra)
    digest = hashlib.sha256(json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    entry["entry_sha256"] = digest
    rows.append(entry)


def write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n" for row in rows).encode()
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def finalize() -> dict[str, Any]:
    registry = read_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json")
    results_path = CAMPAIGN / "analysis/HELDOUT_RESULTS.json"
    decision_path = CAMPAIGN / "analysis/HELDOUT_DECISION.json"
    results, decision = read_json(results_path), read_json(decision_path)
    if registry["phase_status"].get("analysis") != "pass" or decision.get("status") != registry.get("heldout_status"):
        raise HeldoutError("Held-out analysis is not complete and registry-consistent")
    cells: dict[str, Any] = {}
    for split in ("testA", "testB"):
        for method in ("Full", "Z_M448_0"):
            run_dir = REPO / registry["cells"][f"{method}_{split}"]["run_dir"]
            cells[f"{method}_{split}"] = artifact_record(run_dir)
    quality_ledger = {
        "schema_version": 1, "status": "pass", "created_at_utc": now(),
        "exact_cells": ["Full_testA", "Z_M448_0_testA", "Full_testB", "Z_M448_0_testB"],
        "cells": cells, "quality_run_ledger_sha256": sha256(CAMPAIGN / "QUALITY_RUN_LEDGER.json"),
        "outcome_embargo_release_sha256": sha256(CAMPAIGN / "OUTCOME_EMBARGO_RELEASE.json"),
    }
    quality_path = CAMPAIGN / "QUALITY_ARTIFACT_LEDGER.json"
    atomic_json(quality_path, quality_ledger)
    ledger_path = CAMPAIGN / "DECISION_LEDGER.jsonl"
    rows = read_jsonl(ledger_path)
    verify_chain(rows)
    append_event(rows, "quality_cells_complete_and_bound", quality_path, "Both rerun Full controls and both sole-attempt Z cells are accepted and byte-bound.")
    append_event(rows, "outcome_embargo_released_after_both_raw_seals", CAMPAIGN / "OUTCOME_EMBARGO_RELEASE.json", "Candidate outcomes were scored only after both raw split artifacts were sealed.")
    append_event(rows, "preregistered_heldout_rule_applied", decision_path, "Split-wise descriptive gates applied without pooling, tuning or candidate reselection.", {"heldout_status": decision["status"]})
    write_ledger(ledger_path, rows)
    metrics, contrasts, gates = results["metrics"], results["paired_contrasts"], results["gate_results"]
    heterogeneity_path = CAMPAIGN / "analysis/SPLIT_HETEROGENEITY.json"
    heterogeneity = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": now(),
        "direction": "testA_minus_testB_of_Z_minus_Full",
        "split_difference_in_z_minus_full": {
            "acc_at_0_5": contrasts["testA"]["acc_at_0_5"]["point_delta"] - contrasts["testB"]["acc_at_0_5"]["point_delta"],
            "mean_iou": contrasts["testA"]["mean_iou"]["point_delta"] - contrasts["testB"]["mean_iou"]["point_delta"],
        },
        "splits_pooled": False,
    }
    atomic_json(heterogeneity_path, heterogeneity)
    report = [
        "## Final one-shot held-out evidence", "",
        "## Preregistered design and provenance", "",
        "Z_M448_0 was frozen from complete RefCOCO validation before either held-out split was executed. The exact testA/testB manifests, checkpoint tensors, source epoch, commands, paired-bootstrap procedure and split-wise thresholds were locked before candidate inference. Historical Full rows were internally sound but failed the stricter source/checkpoint provenance gate, so both Full controls were rerun. Four serial physical smokes passed before quality inference.", "",
        "The two Z candidate runs were each executed once. Raw testA and testB generations were independently validated and sealed without parsing or quality fields; only after both seals existed was the outcome embargo released and both splits scored.", "",
        "## Split-wise results", "",
        "| Split | Method | N | Parsed | Acc@0.5 | mIoU | Avg visual tokens | Work |", "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("testA", "testB"):
        for method in ("Full", "Z_M448_0"):
            row = metrics[split][method]
            report.append(f"| {split} | {method} | {row['samples']} | {row['parsed_box_rate']:.4f} | {row['acc_at_0_5']:.4f} | {row['mean_iou']:.4f} | {row['average_visual_tokens']:.0f} | {row['token_layer_work']:.0f} |")
    report.extend(["", "## Paired uncertainty and frozen gates", "", "Contrasts below are Z_M448_0 minus Full. Accuracy deltas are percentage points. Each split uses 10,000 deterministic paired manifest-row bootstrap resamples; testA uses child seed 20260820 and testB uses the preregistered deterministic child seed 20260821. NumPy percentile quantiles use the default linear method. Splits are never pooled.", "", "| Split | Acc delta pp [95% CI] | mIoU delta [95% CI] | Parsed gate | Acc-loss gate | mIoU-loss gate | Split result |", "|---|---:|---:|---|---|---|---|"])
    for split in ("testA", "testB"):
        acc, iou = contrasts[split]["acc_at_0_5"], contrasts[split]["mean_iou"]
        check = gates[split]
        g = check["gates"]
        report.append(f"| {split} | {100*acc['point_delta']:+.2f} [{100*acc['lower_95']:+.2f}, {100*acc['upper_95']:+.2f}] | {iou['point_delta']:+.4f} [{iou['lower_95']:+.4f}, {iou['upper_95']:+.4f}] | {'Pass' if g['parsed_box_rate'] else 'Fail'} | {'Pass' if g['accuracy_loss'] else 'Fail'} | {'Pass' if g['miou_loss'] else 'Fail'} | {'Pass' if check['passed'] else 'Fail'} |")
    pass_names = [split for split in ("testA", "testB") if gates[split]["passed"]]
    fail_names = [split for split in ("testA", "testB") if not gates[split]["passed"]]
    hetero = heterogeneity["split_difference_in_z_minus_full"]
    report.extend(["", "## Registered interpretation", "", f"Final descriptive status: `{decision['status']}`. Passing splits: {', '.join(pass_names) or 'none'}. Failing splits: {', '.join(fail_names) or 'none'}.", "", f"Descriptive split heterogeneity (testA minus testB in the Z-minus-Full contrast) is {100*hetero['acc_at_0_5']:+.2f} accuracy percentage points and {hetero['mean_iou']:+.4f} mIoU. This heterogeneity is retained rather than averaged into a pooled decision.", "", "The result applies only to the frozen RefCOCO testA/testB quality protocol. It is not an equivalence or non-inferiority test, population-level proof, transfer claim, SOTA claim, held-out timing result, new speed/energy claim, or permission to tune, recover, switch candidates or inspect another method.", ""])
    report_path = CAMPAIGN / "RESEARCH_REPORT.md"
    if not report_path.is_file():
        raise HeldoutError("The preregistration research report is missing")
    prereg_path = CAMPAIGN / "RESEARCH_REPORT_PREREGISTERED.md"
    if not prereg_path.is_file():
        raise HeldoutError("The pre-outcome research-report snapshot is missing")
    prereg_report = prereg_path.read_text(encoding="utf-8").rstrip()
    if "Preregistered 2026-08-20" not in prereg_report:
        raise HeldoutError("The report snapshot lacks its preregistration entry")
    analysis_report = report_path.read_text(encoding="utf-8").rstrip()
    write_text_atomic(
        report_path,
        prereg_report + "\n\n## Locked analysis handoff\n\n" + analysis_report + "\n\n" + "\n".join(report),
    )
    rows = read_jsonl(ledger_path)
    append_event(rows, "cumulative_report_finalized", report_path, "Report records preregistration, provenance, smokes, embargo, both split estimates, paired intervals, all gates, heterogeneity and limitations.")
    write_ledger(ledger_path, rows)
    registry["quality_artifact_ledger_sha256"] = sha256(quality_path)
    registry["decision_ledger_sha256"] = sha256(ledger_path)
    registry["research_report_sha256"] = sha256(report_path)
    registry["split_heterogeneity_sha256"] = sha256(heterogeneity_path)
    atomic_json(CAMPAIGN / "EXPERIMENT_REGISTRY.json", registry)
    return {"status": "pass", "heldout_status": decision["status"], "quality_artifact_ledger_sha256": sha256(quality_path), "decision_ledger_sha256": sha256(ledger_path), "report_sha256": sha256(report_path)}


if __name__ == "__main__":
    try:
        print(json.dumps(finalize(), indent=2, sort_keys=True))
    except (HeldoutError, OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
