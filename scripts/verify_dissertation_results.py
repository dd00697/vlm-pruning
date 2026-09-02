#!/usr/bin/env python3
"""Verify the compact evidence bundle and README-facing dissertation values."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results/dissertation/summary.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(actual: float, expected: float, label: str, tolerance: float = 1e-6) -> None:
    require(math.isclose(actual, expected, abs_tol=tolerance), f"{label}: {actual} != {expected}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    schedule = data["method"]["schedule"]
    work = sum(int(item["layers"]) * int(item["visual_tokens"]) for item in schedule)
    require(work == 7296, f"unexpected K12 token-layer work: {work}")
    require(sum(int(item["layers"]) for item in schedule) == 32, "schedule must cover 32 layers")

    validation = data["refcoco"]["validation"]
    close(validation["full"]["acc_at_0_5"], 0.449325, "validation Full accuracy")
    close(validation["k12_z0"]["acc_at_0_5"], 0.419476, "validation K12 accuracy")
    timing = data["timing"]
    calculated_reduction = 100 * (
        timing["full_end_to_end_seconds"] - timing["k12_z0_end_to_end_seconds"]
    ) / timing["full_end_to_end_seconds"]
    close(calculated_reduction, timing["latency_reduction_percent"], "latency reduction", 0.001)

    with (ROOT / "results/dissertation/refcoco.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["split"]: row for row in csv.DictReader(handle)}
    require(set(rows) == {"validation", "testA", "testB"}, "RefCOCO CSV split set differs")
    for split in rows:
        record = data["refcoco"][split]
        close(float(rows[split]["full_acc_at_0_5"]), record["full"]["acc_at_0_5"], f"{split} Full")
        close(float(rows[split]["k12_z0_acc_at_0_5"]), record["k12_z0"]["acc_at_0_5"], f"{split} K12")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for required in ("41.95%", "44.93%", "0.814", "6.26%", "22.50%", "K12-QueryTopK-Z0"):
        require(required in readme, f"README is missing {required!r}")
    figures = [
        ROOT / "assets/readme/k12_schedule.png",
        ROOT / "assets/readme/quality_vs_work.png",
        ROOT / "assets/readme/cross_benchmark_transfer.png",
        ROOT / "assets/readme/textvqa_no_tail.png",
    ]
    require(all(path.is_file() and path.stat().st_size > 1000 for path in figures), "README figure missing")

    provenance = json.loads((ROOT / "results/dissertation/provenance.json").read_text(encoding="utf-8"))
    checked_sources = 0
    for source in provenance["sources"]:
        source_path = (ROOT / source["path"]).resolve()
        if source_path.is_file():
            require(sha256(source_path) == source["sha256"], f"source hash differs: {source_path}")
            checked_sources += 1

    ignored_dirs = {".git", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".venv", "venv", "env"}
    markdown = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.md")
        if not ignored_dirs.intersection(path.relative_to(ROOT).parts)
    )
    require(markdown == ["README.md"], f"unexpected Markdown files: {markdown}")
    print(json.dumps({"status": "pass", "markdown": markdown, "figures": len(figures), "sources_checked": checked_sources}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
