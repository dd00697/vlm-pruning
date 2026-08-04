#!/usr/bin/env python
"""Fail-loud final audit and write the dissertation goal completion report."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "experiments/final_evaluation"
DISS = Path("/workspace/Dissertation_Draft")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing required artefact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Missing required artefact: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_page_count(path: Path) -> int:
    """Return the PDF page count without requiring Poppler to be installed."""
    if shutil.which("pdfinfo"):
        info = subprocess.check_output(["pdfinfo", str(path)], text=True)
        for line in info.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":", 1)[1])
        raise RuntimeError(f"pdfinfo did not report a page count for {path}")

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PDF page-count audit requires either pdfinfo or PyMuPDF"
        ) from exc
    with fitz.open(path) as document:
        return document.page_count


def campaign(path: Path, expected: int) -> tuple[dict[str, Any], list[str]]:
    state = read_json(path)
    rows = state.get("runs", {})
    issues = [
        f"{name}:status={row.get('status')}"
        for name, row in rows.items()
        if row.get("status") != "complete"
    ]
    if len(rows) != expected:
        issues.append(f"registered_count={len(rows)} expected={expected}")
    return rows, issues


def provenance_issues(run_id: str, run_dir: Path) -> list[str]:
    path = run_dir / "wandb_run.json"
    if not path.exists():
        return [f"{run_id}: missing {path}"]
    try:
        record = read_json(path)
    except Exception as exc:
        return [f"{run_id}: malformed W&B provenance: {exc}"]
    issues: list[str] = []
    if not record.get("enabled"):
        issues.append(f"{run_id}: W&B was not initialised")
    if record.get("mode") not in {"offline", "online"}:
        issues.append(f"{run_id}: unsupported W&B mode {record.get('mode')!r}")
    if not record.get("id"):
        issues.append(f"{run_id}: W&B run ID is missing")
    if record.get("mode") == "offline" and not list((run_dir / "wandb").glob("offline-run-*")):
        issues.append(f"{run_id}: syncable offline W&B directory is missing")
    return issues


def main() -> int:
    selection_runs, selection_bad = campaign(FINAL / "logs/selection/state.json", 4)
    heldout_smoke_runs, heldout_smoke_bad = campaign(
        FINAL / "logs/smoke_heldout_5/state.json", 16
    )
    general_smoke_runs, general_smoke_bad = campaign(
        FINAL / "logs/smoke_generalisation_5/state.json", 8
    )
    timing_smoke_runs, timing_smoke_bad = campaign(
        FINAL / "logs/timing_smoke/state.json", 5
    )
    heldout_runs, heldout_bad = campaign(FINAL / "logs/heldout/state.json", 16)
    general_runs, general_bad = campaign(FINAL / "logs/generalisation/state.json", 8)
    timing_runs, timing_bad = campaign(FINAL / "logs/timing/state.json", 25)
    run_groups = {
        "selection": selection_runs,
        "heldout_smoke": heldout_smoke_runs,
        "generalisation_smoke": general_smoke_runs,
        "timing_smoke": timing_smoke_runs,
        "heldout_refcoco": heldout_runs,
        "generalisation": general_runs,
        "serial_timing": timing_runs,
    }

    protocol_path = FINAL / "protocol/protocol.json"
    protocol = read_json(protocol_path)
    frozen = read_json(FINAL / "frozen_method.json")
    selection = read_json(FINAL / "selection_report.json")
    source_issues = []
    for relative, expected_hash in protocol["repository"]["source_sha256"].items():
        source = ROOT / relative
        if not source.exists() or sha256(source) != expected_hash:
            source_issues.append(f"{relative}: current source differs from registered protocol")
    fidelity_path = FINAL / "protocol/published_adapter_fidelity.md"
    if (
        not fidelity_path.exists()
        or sha256(fidelity_path)
        != protocol["repository"].get("published_adapter_fidelity_sha256")
    ):
        source_issues.append("published adapter fidelity note differs from registered protocol")
    equivalence_path = FINAL / "protocol/selection_code_epoch_equivalence.json"
    if (
        not equivalence_path.exists()
        or sha256(equivalence_path)
        != protocol["repository"].get(
            "selection_code_epoch_equivalence_sha256"
        )
        or read_json(equivalence_path).get("status") != "passed"
    ):
        source_issues.append(
            "selection code-epoch equivalence report is missing, stale or failed"
        )
    if frozen.get("registered_protocol_sha256") != sha256(protocol_path):
        source_issues.append("frozen method does not reference the current registered protocol")
    for name, record in protocol["manifests"].items():
        manifest = ROOT / record["path"]
        if not manifest.exists() or sha256(manifest) != record["sha256"]:
            source_issues.append(f"{name}: registered manifest checksum mismatch")
    timing_manifest = ROOT / protocol["timing"]["subset_manifest"]
    if (
        not timing_manifest.exists()
        or sha256(timing_manifest) != protocol["timing"]["subset_sha256"]
    ):
        source_issues.append("registered timing-subset checksum mismatch")

    analysis_paths = (
        DISS / "analysis/derived/final_evaluation/heldout_refcoco.csv",
        DISS / "analysis/derived/final_evaluation/heldout_paired_intervals.csv",
        DISS / "analysis/derived/final_evaluation/pope_by_category.csv",
        DISS / "analysis/derived/final_evaluation/textvqa.csv",
        DISS / "analysis/derived/final_evaluation/generalisation_paired_intervals.csv",
        DISS / "analysis/derived/final_evaluation/timing_runs.csv",
        DISS / "analysis/derived/final_evaluation/timing_summary.csv",
        DISS / "analysis/derived/final_evaluation/timing_paired_intervals.csv",
        DISS / "analysis/derived/final_evaluation/error_analysis.csv",
        DISS / "analysis/derived/final_evaluation/qualitative_selection_ledger.json",
        DISS / "analysis/derived/final_evaluation/claim_to_artifact_ledger.json",
        DISS / "images/final_evaluation/testA_accuracy_latency_pareto.pdf",
        DISS / "images/final_evaluation/box_centre_error_heatmap.pdf",
    )
    missing_analysis = [str(path) for path in analysis_paths if not path.exists()]
    analysis_hashes = {
        str(path.relative_to(DISS)): sha256(path)
        for path in analysis_paths
        if path.exists()
    }
    ledger_path = (
        DISS
        / "analysis/derived/final_evaluation/claim_to_artifact_ledger.json"
    )
    ledger_issues: list[str] = []
    ledger = read_json(ledger_path) if ledger_path.exists() else {}
    if ledger.get("protocol_sha256") != sha256(protocol_path):
        ledger_issues.append(
            "claim ledger does not reference the current registered protocol"
        )

    def resolve_ledger_path(value: str) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        dissertation_candidate = DISS / candidate
        return (
            dissertation_candidate
            if dissertation_candidate.exists()
            else ROOT / candidate
        )

    ledger_records: list[dict[str, Any]] = []
    for top_level in ("protocol", "frozen_method", "analysis_source"):
        record = ledger.get(top_level)
        if isinstance(record, dict):
            ledger_records.append(record)
        else:
            ledger_issues.append(f"claim ledger is missing {top_level}")
    claims = ledger.get("claims")
    if not isinstance(claims, dict) or not claims:
        ledger_issues.append("claim ledger has no claim evidence")
    else:
        for claim, records in claims.items():
            if not isinstance(records, list) or not records:
                ledger_issues.append(f"claim ledger has no evidence for {claim}")
                continue
            for record in records:
                if isinstance(record, dict):
                    ledger_records.append(record)
                else:
                    ledger_issues.append(
                        f"claim ledger has malformed evidence for {claim}"
                    )
    for record in ledger_records:
        path_value = record.get("path")
        expected_hash = record.get("sha256")
        if not path_value or not expected_hash:
            ledger_issues.append(
                f"claim ledger contains an incomplete evidence record: {record}"
            )
            continue
        path = resolve_ledger_path(str(path_value))
        if not path.exists() or sha256(path) != expected_hash:
            ledger_issues.append(
                f"claim-ledger evidence is missing or changed: {path_value}"
            )

    provenance_bad: list[str] = []
    run_inventory: list[dict[str, Any]] = []
    for group, rows in run_groups.items():
        for run_id, record in sorted(rows.items()):
            run_dir = (
                FINAL / "timing" / run_id
                if group == "serial_timing"
                else FINAL / "timing_smoke" / run_id
                if group == "timing_smoke"
                else ROOT / str(record.get("run_dir") or "")
            )
            if "smoke" not in group:
                provenance_bad.extend(provenance_issues(run_id, run_dir))
            run_inventory.append(
                {
                    "group": group,
                    "id": run_id,
                    "status": record.get("status"),
                    "run_dir": str(run_dir),
                    "metrics_sha256": (
                        sha256(run_dir / "metrics.json")
                        if (run_dir / "metrics.json").exists()
                        else None
                    ),
                    "results_sha256": (
                        sha256(run_dir / "per_sample_results.jsonl")
                        if (run_dir / "per_sample_results.jsonl").exists()
                        else sha256(run_dir / "per_sample_timing.jsonl")
                        if (run_dir / "per_sample_timing.jsonl").exists()
                        else None
                    ),
                }
            )

    code_checks = read_json(FINAL / "verification/final_code_checks.json")
    qa = read_json(DISS / "FINAL_PDF_QA.json")
    qa_bad = []
    if any(qa.get("latex_checks", {}).values()):
        qa_bad.append(f"LaTeX checks failed: {qa.get('latex_checks')}")
    if qa.get("missing_required_text"):
        qa_bad.append(f"Required PDF content missing: {qa['missing_required_text']}")
    visual = qa.get("visual_inspection", {})
    if (
        visual.get("status") != "passed"
        or sorted(visual.get("inspected_pages", []))
        != sorted(visual.get("pages_to_render", []))
    ):
        qa_bad.append("new PDF pages have not all been visually inspected")

    pdf = DISS / "Dissertation_Template.pdf"
    pdf_pages = None
    if pdf.exists():
        pdf_pages = pdf_page_count(pdf)

    result_tables = {}
    for name, path in (
        ("heldout_refcoco", analysis_paths[0]),
        ("pope", analysis_paths[2]),
        ("textvqa", analysis_paths[3]),
        ("timing", analysis_paths[6]),
        ("error_analysis", analysis_paths[8]),
    ):
        result_tables[name] = read_csv(path) if path.exists() else []

    all_issues = (
        selection_bad
        + heldout_smoke_bad
        + general_smoke_bad
        + timing_smoke_bad
        + heldout_bad
        + general_bad
        + timing_bad
        + source_issues
        + missing_analysis
        + ledger_issues
        + provenance_bad
        + qa_bad
    )
    if frozen.get("status") != "frozen":
        all_issues.append("final method is not frozen")
    if code_checks.get("status") != "passed":
        all_issues.append("final code checks did not pass")
    if not pdf.exists():
        all_issues.append("Dissertation_Template.pdf is missing")

    caveats = [
        "Three selection jobs began before provenance-only prompt/evaluator config fields were added; recorded commands and exact 20-example current-code semantic matches were verified before the preserved configs were annotated. Prediction and metric files were not changed.",
        "FastV uses the pinned K=3 selection rule adapted to current Hugging Face LLaVA with a static selected set during cached decoding.",
        "VisionZip uses the pinned dominant/contextual rule with 1 CLS, 39 dominant patch and 8 contextual tokens at B=48.",
        "All results use one LLaVA-1.5-7B checkpoint and one NVIDIA A40; hardware generality is not established.",
        "Validation selects the method; RefCOCO testA/testB remain held-out reporting data.",
    ]
    report = {
        "status": "complete" if not all_issues else "incomplete",
        "issues": all_issues,
        "run_counts": {
            name: {
                "complete": sum(row.get("status") == "complete" for row in rows.values()),
                "required": {"selection": 4, "heldout_refcoco": 16, "generalisation": 8, "serial_timing": 25}[name],
            }
            for name, rows in run_groups.items()
            if name
            in {
                "selection",
                "heldout_refcoco",
                "generalisation",
                "serial_timing",
            }
        },
        "smoke_run_counts": {
            "heldout": {
                "complete": sum(
                    row.get("status") == "complete"
                    for row in heldout_smoke_runs.values()
                ),
                "required": 16,
            },
            "generalisation": {
                "complete": sum(
                    row.get("status") == "complete"
                    for row in general_smoke_runs.values()
                ),
                "required": 8,
            },
            "timing": {
                "complete": sum(
                    row.get("status") == "complete"
                    for row in timing_smoke_runs.values()
                ),
                "required": 5,
            },
        },
        "runs": run_inventory,
        "frozen_winner": frozen.get("winner"),
        "selection_ranking": selection.get("cells_ranked"),
        "results": result_tables,
        "analysis_artifacts_sha256": analysis_hashes,
        "protocol_sha256": sha256(protocol_path),
        "code_checks": code_checks,
        "pdf": {
            "path": str(pdf),
            "sha256": sha256(pdf) if pdf.exists() else None,
            "pages": pdf_pages,
            "qa": qa,
        },
        "caveats_and_adapter_deviations": caveats,
    }
    output = FINAL / "FINAL_COMPLETION_REPORT.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = [
        "# Final dissertation evaluation completion report",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Run completion",
        "",
        *[
            f"- {name}: {value['complete']}/{value['required']}"
            for name, value in report["run_counts"].items()
        ],
        *[
            f"- {name} smoke: {value['complete']}/{value['required']}"
            for name, value in report["smoke_run_counts"].items()
        ],
        "",
        "## Frozen method",
        "",
        f"- Winner: `{report['frozen_winner']}`",
        "",
        "## Verification",
        "",
        f"- Code checks: {code_checks.get('status')}",
        f"- PDF pages: {pdf_pages}",
        f"- Visual QA: {visual.get('status')}",
        f"- Open issues: {len(all_issues)}",
        "",
        "## Caveats and adapter notes",
        "",
        *[f"- {item}" for item in caveats],
        "",
        f"Machine-readable report: `{output}`",
    ]
    (FINAL / "FINAL_COMPLETION_REPORT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
