"""Strict, manifest-aligned experiment artefact handling.

Canonical JSONLs are never used as append targets.  Jobs append to a partial
file and this module atomically publishes a canonical, manifest-ordered file
only after every registered row passes validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class ArtifactIntegrityError(RuntimeError):
    """Raised when a run artefact violates the registered contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance(repo_root: Path, paths: Sequence[Path]) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()

    try:
        revision = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", None
    hashes: dict[str, str] = {}
    for path in paths:
        resolved = path if path.is_absolute() else repo_root / path
        if resolved.is_file():
            hashes[str(resolved.relative_to(repo_root))] = sha256_file(resolved)
    return {"git_commit": revision, "git_dirty": dirty, "source_sha256": hashes}


def read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ArtifactIntegrityError(f"Missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactIntegrityError(f"Malformed JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ArtifactIntegrityError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _ids(rows: Sequence[Mapping[str, Any]], *, label: str) -> list[str]:
    result: list[str] = []
    for index, row in enumerate(rows):
        if row.get("sample_id") is None:
            raise ArtifactIntegrityError(f"{label} row {index} has no sample_id")
        result.append(str(row["sample_id"]))
    return result


def load_resume_rows(path: Path, manifest_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Load a partial file strictly, returning its rows in manifest order."""
    if not path.exists():
        return []
    rows = read_jsonl_strict(path)
    manifest_ids = _ids(manifest_rows, label="manifest")
    wanted = set(manifest_ids)
    row_ids = _ids(rows, label="partial")
    counts: dict[str, int] = {}
    for sid in row_ids:
        counts[sid] = counts.get(sid, 0) + 1
    duplicates = sorted(sid for sid, count in counts.items() if count > 1)
    extras = sorted(set(row_ids) - wanted)
    if duplicates or extras:
        raise ArtifactIntegrityError(f"Invalid resume file {path}: duplicates={duplicates[:10]} extras={extras[:10]}")
    by_id = {str(row["sample_id"]): row for row in rows}
    return [by_id[sid] for sid in manifest_ids if sid in by_id]


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    budget: int | None = None,
    original_tokens: int | None = None,
    strategy: str | None = None,
    require_order: bool = True,
    require_complete: bool = True,
    require_retained: bool = True,
) -> dict[str, Any]:
    manifest_ids = _ids(manifest_rows, label="manifest")
    row_ids = _ids(rows, label="prediction")
    counts: dict[str, int] = {}
    for sid in row_ids:
        counts[sid] = counts.get(sid, 0) + 1
    duplicates = sorted(sid for sid, count in counts.items() if count > 1)
    manifest_set = set(manifest_ids)
    row_set = set(row_ids)
    missing = [sid for sid in manifest_ids if sid not in row_set]
    extras = [sid for sid in row_ids if sid not in manifest_set]
    order_equal = row_ids == manifest_ids
    issues: list[str] = []
    if duplicates:
        issues.append(f"duplicate sample IDs: {duplicates[:10]}")
    if extras:
        issues.append(f"unexpected sample IDs: {extras[:10]}")
    if require_complete and missing:
        issues.append(f"missing sample IDs: {missing[:10]} ({len(missing)} total)")
    if require_complete and len(rows) != len(manifest_rows):
        issues.append(f"row count {len(rows)} != expected {len(manifest_rows)}")
    if require_order and not order_equal:
        issues.append("prediction order does not equal manifest order")

    budget_failures: list[str] = []
    for row in rows:
        sid = str(row.get("sample_id"))
        row_original_tokens = row.get("original_visual_tokens", original_tokens)
        try:
            row_original_tokens = int(row_original_tokens) if row_original_tokens is not None else None
        except (TypeError, ValueError):
            row_original_tokens = None
        if budget is not None:
            expected_budget = int(budget)
            if str(row.get("method") or "").lower() == "full" and row_original_tokens is not None:
                expected_budget = row_original_tokens
            requested = row.get("target_visual_tokens")
            actual = row.get("actual_llm_visual_tokens", row.get("actual_retained_visual_tokens"))
            if requested is not None and int(requested) != expected_budget:
                budget_failures.append(f"{sid}: requested={requested}")
            if actual is not None and int(actual) != expected_budget:
                budget_failures.append(f"{sid}: actual={actual}")
            retained = row.get("retained_indices")
            if require_retained:
                if not isinstance(retained, list):
                    budget_failures.append(f"{sid}: retained_indices missing")
                else:
                    values = [int(value) for value in retained]
                    if len(values) != expected_budget or len(set(values)) != len(values):
                        budget_failures.append(f"{sid}: invalid retained count/duplicates")
                    if row_original_tokens is not None and any(
                        value < 0 or value >= row_original_tokens for value in values
                    ):
                        budget_failures.append(f"{sid}: retained index out of range")
        if strategy is not None and row.get("selector_strategy") != strategy:
            budget_failures.append(f"{sid}: selector_strategy={row.get('selector_strategy')!r}")
    if budget_failures:
        issues.append(f"token/config failures: {budget_failures[:10]} ({len(budget_failures)} total)")
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "expected_rows": len(manifest_rows),
        "actual_rows": len(rows),
        "unique_ids": len(row_set),
        "duplicate_ids": duplicates,
        "missing_ids": missing,
        "unexpected_ids": extras,
        "manifest_order_equal": order_equal,
        "budget_failure_count": len(budget_failures),
    }


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def finalize_partial_jsonl(
    partial_path: Path,
    final_path: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    original_tokens: int,
    strategy: str | None = None,
    require_retained: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_resume_rows(partial_path, manifest_rows)
    report = validate_rows(
        rows,
        manifest_rows,
        budget=budget,
        original_tokens=original_tokens,
        strategy=strategy,
        require_retained=require_retained,
    )
    if report["status"] != "pass":
        raise ArtifactIntegrityError("Cannot finalize partial JSONL: " + "; ".join(report["issues"]))
    atomic_write_jsonl(final_path, rows)
    reopened = read_jsonl_strict(final_path)
    second = validate_rows(
        reopened,
        manifest_rows,
        budget=budget,
        original_tokens=original_tokens,
        strategy=strategy,
        require_retained=require_retained,
    )
    if second["status"] != "pass":
        raise ArtifactIntegrityError("Atomic final JSONL failed reopen validation: " + "; ".join(second["issues"]))
    second["sha256"] = sha256_file(final_path)
    return reopened, second


def recompute_core_metrics(rows: Sequence[Mapping[str, Any]], *, task: str) -> dict[str, float | int]:
    result: dict[str, float | int] = {"samples/evaluated": len(rows)}
    if task == "grounding":
        count = len(rows)
        ious = [float(row.get("box_iou") or 0.0) for row in rows]
        accuracy = sum(bool(row.get("success_iou_0_5")) for row in rows) / count if count else 0.0
        result.update({
            "quality/main_score": accuracy,
            "quality/acc_at_0_5": accuracy,
            "quality/mean_iou": sum(ious) / count if count else 0.0,
            "quality/parsed_box_rate": sum(row.get("raw_parsed_box") is not None for row in rows) / count if count else 0.0,
        })
    return result


def validate_metric_agreement(
    rows: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any], *, task: str, tolerance: float = 1e-9
) -> dict[str, Any]:
    recomputed = recompute_core_metrics(rows, task=task)
    mismatches: dict[str, Any] = {}
    for key, expected in recomputed.items():
        actual = metrics.get(key)
        if actual is None or abs(float(actual) - float(expected)) > tolerance:
            mismatches[key] = {"stored": actual, "recomputed": expected}
    return {"status": "pass" if not mismatches else "fail", "tolerance": tolerance, "mismatches": mismatches, "recomputed": recomputed}


def validate_completed_run(
    run_dir: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    original_tokens: int,
    task: str,
    strategy: str | None = None,
    require_retained: bool = True,
) -> dict[str, Any]:
    issues: list[str] = []
    required = ["config.json", "config.yaml", "run_environment.json", "metrics.json", "per_sample_results.jsonl", "wandb_run.json"]
    missing_files = [name for name in required if not (run_dir / name).is_file()]
    if missing_files:
        return {"status": "fail", "issues": [f"missing files: {missing_files}"], "run_dir": str(run_dir)}
    try:
        rows = read_jsonl_strict(run_dir / "per_sample_results.jsonl")
        row_report = validate_rows(rows, manifest_rows, budget=budget, original_tokens=original_tokens, strategy=strategy, require_retained=require_retained)
        issues.extend(row_report["issues"])
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        metric_report = validate_metric_agreement(rows, metrics, task=task)
        if metric_report["status"] != "pass":
            issues.append(f"aggregate mismatch: {metric_report['mismatches']}")
    except (OSError, ValueError, ArtifactIntegrityError) as exc:
        return {"status": "fail", "issues": [str(exc)], "run_dir": str(run_dir)}
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "run_dir": str(run_dir),
        "rows": row_report,
        "metrics": metric_report,
        "prediction_sha256": sha256_file(run_dir / "per_sample_results.jsonl"),
    }
