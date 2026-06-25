#!/usr/bin/env python
"""Summarize official baseline CSV outputs into Markdown report artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TABLES_ROOT = Path("experiments/tables/official_baselines")
REPORTS_ROOT = Path("experiments/reports/official_baselines")
FIGURES_ROOT = Path("experiments/figures_official_baselines/summary")

VQA_DATASETS = [
    {"dataset": "pope", "task": "vqa", "stress": True},
    {"dataset": "aokvqa", "task": "vqa", "stress": True},
    {"dataset": "scienceqa_img", "task": "vqa", "stress": True},
    {"dataset": "textvqa", "task": "vqa", "stress": True},
    {"dataset": "mmmu", "task": "vqa", "stress": False},
    {"dataset": "mme", "task": "vqa", "stress": False},
    {"dataset": "seed_bench", "task": "vqa", "stress": False},
    {"dataset": "mmvet", "task": "vqa", "stress": False},
]
REFCOCO = {"dataset": "refcoco", "task": "grounding", "stress": True}
BROAD_TOKENS = [432, 288, 144, 58]
STRESS_TOKENS = [48, 32, 16]
RANDOM_SEEDS = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-root", type=Path, default=TABLES_ROOT)
    parser.add_argument("--reports-root", type=Path, default=REPORTS_ROOT)
    parser.add_argument("--figures-root", type=Path, default=FIGURES_ROOT)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def fmt(value: Any) -> str:
    number = as_float(value)
    if number is not None:
        if abs(number) >= 1000:
            return f"{number:.1f}"
        return f"{number:.4f}"
    return "" if value is None else str(value)


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["_No completed rows._"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return lines


def compact_rows(rows: list[dict[str, str]], task: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "dataset": row.get("dataset/name"),
                "task": task,
                "method": row.get("method/name"),
                "tokens": row.get("tokens/actual_visual_tokens") or row.get("tokens/final_visual_tokens"),
                "seed": row.get("method/seed"),
                "score": row.get("quality/main_score"),
                "accuracy": row.get("quality/accuracy") or row.get("quality/acc_at_0_5"),
                "mean_iou": row.get("quality/mean_iou"),
                "latency_s": row.get("efficiency/mean_latency_s"),
                "ttft_s": row.get("efficiency/time_to_first_token_s"),
                "tpot_s": row.get("efficiency/time_per_output_token_s"),
                "flops_ratio": row.get("efficiency/theoretical_prefill_flops_ratio"),
                "peak_mem_mb": row.get("efficiency/peak_gpu_memory_mb"),
            }
        )
    return out


def random_seed_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        if row.get("method/name") != "random":
            continue
        key = (row.get("dataset/name", ""), row.get("tokens/actual_visual_tokens") or row.get("tokens/final_visual_tokens") or "", row.get("task", ""))
        score = as_float(row.get("quality/main_score"))
        if score is not None:
            grouped.setdefault(key, []).append(score)
    out = []
    for (dataset, tokens, task), values in sorted(grouped.items()):
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
        out.append({"dataset": dataset, "task": task, "tokens": tokens, "seeds": len(values), "mean_score": mean, "std_score": var**0.5})
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = sorted({field for row in rows for field in row.keys()})
    if not fields:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_blockers(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def intish(value: Any) -> int | None:
    number = as_float(value)
    return int(round(number)) if number is not None else None


def expected_matrix(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    completed_datasets = {row.get("dataset/name") for row in rows}
    datasets = list(VQA_DATASETS)
    if "ocr_vqa" in completed_datasets and "textvqa" not in completed_datasets:
        datasets = [{**item, "dataset": "ocr_vqa"} if item["dataset"] == "textvqa" else item for item in datasets]
    datasets.append(REFCOCO)

    expected: list[dict[str, Any]] = []
    for spec in datasets:
        expected.append(
            {
                "dataset": spec["dataset"],
                "task": spec["task"],
                "method": "full",
                "tokens": 576,
                "seed": 0,
                "budget_family": "full",
            }
        )
        tokens = BROAD_TOKENS + (STRESS_TOKENS if spec["stress"] else [])
        for token in tokens:
            family = "gentle" if token == 432 else ("stress" if token in STRESS_TOKENS else "broad")
            expected.append(
                {
                    "dataset": spec["dataset"],
                    "task": spec["task"],
                    "method": "uniform",
                    "tokens": token,
                    "seed": 0,
                    "budget_family": family,
                }
            )
            for seed in RANDOM_SEEDS:
                expected.append(
                    {
                        "dataset": spec["dataset"],
                        "task": spec["task"],
                        "method": "random",
                        "tokens": token,
                        "seed": seed,
                        "budget_family": family,
                    }
                )
    return expected


def completion_matrix(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    completed_keys = set()
    completed_scores: dict[tuple[str, str, str, int, int], dict[str, str]] = {}
    for row in rows:
        dataset = row.get("dataset/name") or ""
        task = row.get("task") or ("grounding" if dataset == "refcoco" else "vqa")
        method = row.get("method/name") or ""
        tokens = intish(row.get("tokens/actual_visual_tokens") or row.get("tokens/final_visual_tokens"))
        seed = intish(row.get("method/seed"))
        if not dataset or not method or tokens is None or seed is None:
            continue
        key = (dataset, task, method, tokens, seed)
        completed_keys.add(key)
        completed_scores[key] = row

    matrix = []
    for item in expected_matrix(rows):
        key = (item["dataset"], item["task"], item["method"], item["tokens"], item["seed"])
        completed = key in completed_keys
        row = completed_scores.get(key, {})
        matrix.append(
            {
                **item,
                "status": "completed" if completed else "missing",
                "quality/main_score": row.get("quality/main_score", ""),
                "efficiency/mean_latency_s": row.get("efficiency/mean_latency_s", ""),
                "run_name": row.get("run_name", ""),
            }
        )
    return matrix


def completion_summary(matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in matrix:
        key = (row["task"], row["dataset"], row["method"])
        target = grouped.setdefault(
            key,
            {
                "task": row["task"],
                "dataset": row["dataset"],
                "method": row["method"],
                "expected_runs": 0,
                "completed_runs": 0,
                "missing_runs": 0,
            },
        )
        target["expected_runs"] += 1
        if row["status"] == "completed":
            target["completed_runs"] += 1
        else:
            target["missing_runs"] += 1
    return sorted(grouped.values(), key=lambda item: (item["task"], item["dataset"], item["method"]))


def numeric_compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        tokens = as_float(row.get("tokens"))
        score = as_float(row.get("score"))
        latency = as_float(row.get("latency_s"))
        flops = as_float(row.get("flops_ratio"))
        if tokens is None or score is None:
            continue
        out.append(
            {
                "dataset": row.get("dataset"),
                "task": row.get("task"),
                "method": row.get("method"),
                "tokens": tokens,
                "seed": intish(row.get("seed")),
                "score": score,
                "latency_s": latency,
                "flops_ratio": flops,
                "mean_iou": as_float(row.get("mean_iou")),
            }
        )
    return out


def load_runtime_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def save_plot(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")


def plot_summary_figures(
    figures_root: Path,
    vqa_compact: list[dict[str, Any]],
    ref_compact: list[dict[str, Any]],
    matrix: list[dict[str, Any]],
    runtime_history: list[dict[str, Any]],
) -> list[dict[str, str]]:
    figures_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [{"path": "", "title": "Plot generation skipped", "description": f"matplotlib unavailable: {type(exc).__name__}: {exc}"}]

    compact = numeric_compact_rows(vqa_compact + ref_compact)
    full_rows = [row for row in compact if row["method"] == "full"]
    if full_rows:
        labels = [str(row["dataset"]) for row in full_rows]
        scores = [float(row["score"]) for row in full_rows]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4.8))
        ax.bar(labels, scores, color="#2f6f8f")
        ax.set_ylabel("Main score")
        ax.set_title("Full-token baseline score by dataset")
        ax.set_ylim(0, max(1.0, max(scores) * 1.1))
        ax.tick_params(axis="x", rotation=35)
        path = figures_root / "full_token_scores.png"
        save_plot(fig, path)
        plt.close(fig)
        manifest.append({"path": str(path), "title": "Full-token scores", "description": "Main full-token score for each completed dataset."})

    if matrix:
        summary = completion_summary(matrix)
        labels = [f"{row['dataset']}:{row['method']}" for row in summary]
        completed = [float(row["completed_runs"]) for row in summary]
        missing = [float(row["missing_runs"]) for row in summary]
        fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.42), 5.5))
        ax.bar(labels, completed, label="completed", color="#2f6f8f")
        ax.bar(labels, missing, bottom=completed, label="missing", color="#c75050")
        ax.set_ylabel("Runs")
        ax.set_title("Official baseline completion matrix")
        ax.legend()
        ax.tick_params(axis="x", rotation=70)
        path = figures_root / "completion_matrix.png"
        save_plot(fig, path)
        plt.close(fig)
        manifest.append({"path": str(path), "title": "Completion matrix", "description": "Completed and missing final rows by dataset and method."})

    if compact:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for key in sorted({(row["dataset"], row["method"]) for row in compact}):
            rows = sorted([row for row in compact if (row["dataset"], row["method"]) == key], key=lambda item: item["tokens"])
            ax.plot([row["tokens"] for row in rows], [row["score"] for row in rows], marker="o", label=f"{key[0]}:{key[1]}")
        ax.set_xlabel("Retained visual tokens")
        ax.set_ylabel("Main score")
        ax.set_title("Quality versus visual-token budget")
        ax.invert_xaxis()
        if len(compact) <= 18:
            ax.legend(fontsize=7, loc="best")
        path = figures_root / "quality_vs_tokens.png"
        save_plot(fig, path)
        plt.close(fig)
        manifest.append({"path": str(path), "title": "Quality vs tokens", "description": "Main score against retained visual tokens for completed rows."})

        tradeoff = [row for row in compact if row.get("latency_s") is not None]
        if tradeoff:
            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            methods = sorted({str(row["method"]) for row in tradeoff})
            markers = {"full": "o", "uniform": "s", "random": "^"}
            for method in methods:
                rows = [row for row in tradeoff if row["method"] == method]
                ax.scatter(
                    [float(row["latency_s"]) for row in rows],
                    [float(row["score"]) for row in rows],
                    label=method,
                    marker=markers.get(method, "o"),
                    s=[max(25.0, float(row["tokens"]) / 4.0) for row in rows],
                    alpha=0.8,
                )
            ax.set_xlabel("Mean latency per sample (s)")
            ax.set_ylabel("Main score")
            ax.set_title("Quality and latency tradeoff")
            ax.legend()
            path = figures_root / "quality_latency_tradeoff.png"
            save_plot(fig, path)
            plt.close(fig)
            manifest.append({"path": str(path), "title": "Quality-latency tradeoff", "description": "Main score versus mean latency; marker size reflects retained tokens."})

    ref_numeric = [row for row in numeric_compact_rows(ref_compact) if row["dataset"] == "refcoco"]
    if ref_numeric:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for method in sorted({row["method"] for row in ref_numeric}):
            rows = sorted([row for row in ref_numeric if row["method"] == method], key=lambda item: item["tokens"])
            ax.plot([row["tokens"] for row in rows], [row["score"] for row in rows], marker="o", label=f"{method} Acc@0.5")
            if any(row.get("mean_iou") is not None for row in rows):
                ax.plot([row["tokens"] for row in rows], [row.get("mean_iou") or 0.0 for row in rows], marker="x", linestyle="--", label=f"{method} mean IoU")
        ax.set_xlabel("Retained visual tokens")
        ax.set_ylabel("Score")
        ax.set_title("RefCOCO localization diagnostic")
        ax.invert_xaxis()
        ax.legend(fontsize=8)
        path = figures_root / "refcoco_quality_vs_tokens.png"
        save_plot(fig, path)
        plt.close(fig)
        manifest.append({"path": str(path), "title": "RefCOCO quality vs tokens", "description": "RefCOCO Acc@0.5 and mean IoU against retained visual tokens."})

    runtime_points = [
        (idx + 1, as_float(row.get("runtime/estimated_remaining_wall_time_s")))
        for idx, row in enumerate(runtime_history)
        if as_float(row.get("runtime/estimated_remaining_wall_time_s")) is not None
    ]
    if runtime_points:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.plot([idx for idx, _value in runtime_points], [value / 3600.0 for _idx, value in runtime_points], marker="o", color="#3b7f5f")
        ax.set_xlabel("Runtime estimate update")
        ax.set_ylabel("Estimated remaining time (h)")
        ax.set_title("Rolling suite ETA history")
        path = figures_root / "runtime_eta_history.png"
        save_plot(fig, path)
        plt.close(fig)
        manifest.append({"path": str(path), "title": "Runtime ETA history", "description": "Saved rolling remaining-time estimate after completed runs."})

    write_csv(
        figures_root / "figure_manifest.csv",
        manifest,
        ["path", "title", "description"],
    )
    return manifest


def main() -> int:
    args = parse_args()
    args.tables_root.mkdir(parents=True, exist_ok=True)
    args.reports_root.mkdir(parents=True, exist_ok=True)
    args.figures_root.mkdir(parents=True, exist_ok=True)
    vqa_rows = read_rows(args.tables_root / "vqa_quality_by_budget.csv")
    ref_rows = read_rows(args.tables_root / "refcoco_quality_by_budget.csv")
    vqa_compact = compact_rows(vqa_rows, "vqa")
    ref_compact = compact_rows(ref_rows, "grounding")
    seed_summary = random_seed_summary(vqa_rows + ref_rows)
    all_quality_rows = vqa_rows + ref_rows
    matrix = completion_matrix(all_quality_rows)
    missing = [row for row in matrix if row["status"] == "missing"]
    matrix_fields = [
        "task",
        "dataset",
        "method",
        "tokens",
        "seed",
        "budget_family",
        "status",
        "quality/main_score",
        "efficiency/mean_latency_s",
        "run_name",
    ]
    write_csv(args.tables_root / "official_baselines_completion_matrix.csv", matrix, matrix_fields)
    write_csv(args.tables_root / "official_baselines_missing_runs.csv", missing, matrix_fields)
    write_csv(
        args.tables_root / "vqa_random_seed_summary.csv",
        [row for row in seed_summary if row["task"] != "grounding"],
        ["dataset", "task", "tokens", "seeds", "mean_score", "std_score"],
    )
    write_csv(
        args.tables_root / "refcoco_random_seed_summary.csv",
        [row for row in seed_summary if row["task"] == "grounding"],
        ["dataset", "task", "tokens", "seeds", "mean_score", "std_score"],
    )
    blockers = read_blockers(args.tables_root / "dataset_blockers.jsonl")
    runtime_latest = read_json(args.tables_root / "runtime_estimates.json")
    runtime_history = load_runtime_history(args.tables_root / "runtime_estimates_history.jsonl")
    figure_manifest = plot_summary_figures(args.figures_root, vqa_compact, ref_compact, matrix, runtime_history)
    lines = [
        "# Official Baselines Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')}",
        "",
        "This report only summarizes rows present in saved official-baseline tables. Missing rows are not treated as zero.",
        "",
        "## VQA Quality And Efficiency",
        "",
        *md_table(vqa_compact, ["dataset", "method", "tokens", "seed", "score", "accuracy", "latency_s", "ttft_s", "tpot_s", "flops_ratio", "peak_mem_mb"]),
        "",
        "## RefCOCO Diagnostic",
        "",
        *md_table(ref_compact, ["dataset", "method", "tokens", "seed", "score", "accuracy", "mean_iou", "latency_s", "ttft_s", "tpot_s", "flops_ratio", "peak_mem_mb"]),
        "",
        "## Random Seed Aggregates",
        "",
        *md_table(seed_summary, ["dataset", "task", "tokens", "seeds", "mean_score", "std_score"]),
        "",
        "## Completion Matrix",
        "",
        *md_table(completion_summary(matrix), ["task", "dataset", "method", "expected_runs", "completed_runs", "missing_runs"]),
        "",
        "## Runtime Estimate",
        "",
        *md_table([runtime_latest] if runtime_latest else [], ["runtime/completed_runs", "runtime/remaining_runs", "runtime/elapsed_wall_time_s", "runtime/estimated_remaining_wall_time_s", "runtime/eta_utc"]),
        "",
        "## Figures",
        "",
        *md_table(figure_manifest, ["title", "path", "description"]),
        "",
        "## Blockers",
        "",
        *md_table(blockers, ["dataset", "limit", "note", "exit_code"]),
        "",
    ]
    summary_path = args.tables_root / "official_baselines_summary.md"
    report_path = args.reports_root / "OFFICIAL_BASELINES_RUN_REPORT.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
