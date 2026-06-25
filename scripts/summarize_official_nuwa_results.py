#!/usr/bin/env python
"""Build official-Nuwa matched and combined result tables from saved runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OFFICIAL_TOKENS = [16, 32, 48]
MATCHED_METHODS = ["random", "uniform", "visionzip", "fastv"]
OFFICIAL_METHODS = ["official_full", "official_nuwa"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--tables-dir", type=Path, default=Path("experiments/tables"))
    parser.add_argument("--allow-missing", action="store_true", help="Write partial tables when documented matched-baseline blockers remain.")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def completed_500(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "num_samples" in df.columns:
        return df[pd.to_numeric(df["num_samples"], errors="coerce") == 500].copy()
    if "limit" in df.columns:
        return df[pd.to_numeric(df["limit"], errors="coerce") == 500].copy()
    return df.copy()


def ensure_scope(df: pd.DataFrame, scope: str, stack: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = completed_500(df).copy()
    if "comparison_scope" not in out.columns:
        out["comparison_scope"] = scope
    else:
        out["comparison_scope"] = out["comparison_scope"].fillna(scope).replace("", scope)
    if "model_stack" not in out.columns:
        out["model_stack"] = stack
    else:
        out["model_stack"] = out["model_stack"].fillna(stack).replace("", stack)
    return out


def token_series(df: pd.DataFrame) -> pd.Series:
    for column in ["actual_llm_visual_tokens", "final_visual_tokens", "retained_visual_tokens", "token_budget"]:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def normalize_table(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["task"] = out.get("task", task)
    out["tokens"] = token_series(out)
    out["tokens"] = out["tokens"].round().astype("Int64")
    if "actual_llm_visual_tokens" not in out.columns:
        out["actual_llm_visual_tokens"] = out["tokens"]
    return out


def combined_matched_table(tables_dir: Path, task: str) -> pd.DataFrame:
    matched_dir = tables_dir / "official_nuwa_matched_source"
    suffix = "grounding" if task == "grounding" else "vqa"
    official = ensure_scope(read_csv(tables_dir / f"official_nuwa_{suffix}.csv"), "official_stack", "official_llava")
    matched_true = ensure_scope(read_csv(matched_dir / f"true_removal_{suffix}.csv"), "budget_matched_hf_stack", "hf_llava")
    matched_prior = ensure_scope(read_csv(matched_dir / f"prior_methods_{suffix}.csv"), "budget_matched_hf_stack", "hf_llava")
    combined = pd.concat([official, matched_true, matched_prior], ignore_index=True, sort=False)
    combined = normalize_table(completed_500(combined), task)
    combined = combined[combined["method"].isin(OFFICIAL_METHODS + MATCHED_METHODS)].copy()
    combined = combined[(combined["method"].isin(OFFICIAL_METHODS)) | (combined["tokens"].isin(OFFICIAL_TOKENS))].copy()
    return add_matched_deltas(combined, metric_for_task(task))


def combined_efficiency_table(tables_dir: Path) -> pd.DataFrame:
    matched_dir = tables_dir / "official_nuwa_matched_source"
    official = ensure_scope(read_csv(tables_dir / "official_nuwa_efficiency.csv"), "official_stack", "official_llava")
    matched_true = ensure_scope(read_csv(matched_dir / "true_removal_efficiency.csv"), "budget_matched_hf_stack", "hf_llava")
    matched_prior = ensure_scope(read_csv(matched_dir / "prior_methods_efficiency.csv"), "budget_matched_hf_stack", "hf_llava")
    combined = pd.concat([official, matched_true, matched_prior], ignore_index=True, sort=False)
    combined = normalize_table(completed_500(combined), "mixed")
    combined = combined[combined["method"].isin(OFFICIAL_METHODS + MATCHED_METHODS)].copy()
    combined = combined[(combined["method"].isin(OFFICIAL_METHODS)) | (combined["tokens"].isin(OFFICIAL_TOKENS))].copy()
    return combined.sort_values(["task", "tokens", "method"], na_position="last")


def metric_for_task(task: str) -> str:
    return "accuracy_at_0.5" if task == "grounding" else "accuracy"


def add_matched_deltas(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return df
    out = df.copy()
    out[metric] = pd.to_numeric(out[metric], errors="coerce")
    official_full = out[out["method"] == "official_full"]
    full_metric = float(official_full[metric].iloc[0]) if not official_full.empty else np.nan
    out["delta_vs_official_full"] = out[metric] - full_metric
    official_by_tokens = {}
    for _, row in out[out["method"] == "official_nuwa"].iterrows():
        if not pd.isna(row["tokens"]):
            official_by_tokens[int(row["tokens"])] = float(row[metric])
    out["delta_vs_official_nuwa_same_tokens"] = out.apply(
        lambda row: row[metric] - official_by_tokens.get(int(row["tokens"]), np.nan)
        if not pd.isna(row.get("tokens"))
        else np.nan,
        axis=1,
    )
    for method in MATCHED_METHODS:
        refs = {}
        for _, ref_row in out[out["method"] == method].iterrows():
            if not pd.isna(ref_row["tokens"]):
                refs[int(ref_row["tokens"])] = float(ref_row[metric])
        out[f"delta_vs_{method}_same_tokens"] = out.apply(
            lambda row: row[metric] - refs.get(int(row["tokens"]), np.nan)
            if not pd.isna(row.get("tokens"))
            else np.nan,
            axis=1,
        )
    return out.sort_values(["tokens", "method"], na_position="last")


def build_all_methods_table(tables_dir: Path, task: str) -> pd.DataFrame:
    suffix = "grounding" if task == "grounding" else "vqa"
    base = ensure_scope(read_csv(tables_dir / f"all_methods_with_nuwa_{suffix}.csv"), "previous_hf_stack", "hf_llava")
    official = combined_matched_table(tables_dir, task)
    combined = pd.concat([base, official], ignore_index=True, sort=False)
    combined = normalize_table(combined, task)
    if "run_name" in combined.columns:
        combined = combined.drop_duplicates(subset=["task", "method", "run_name"], keep="last")
    else:
        combined = combined.drop_duplicates(subset=["task", "method", "keep_ratio", "tokens", "comparison_scope"], keep="last")
    return combined


def build_all_methods_efficiency(tables_dir: Path) -> pd.DataFrame:
    base = ensure_scope(read_csv(tables_dir / "all_methods_with_nuwa_efficiency.csv"), "previous_hf_stack", "hf_llava")
    official = combined_efficiency_table(tables_dir)
    combined = pd.concat([base, official], ignore_index=True, sort=False)
    combined = normalize_table(combined, "mixed")
    if "run_name" in combined.columns:
        combined = combined.drop_duplicates(subset=["task", "method", "run_name"], keep="last")
    else:
        combined = combined.drop_duplicates(subset=["task", "method", "keep_ratio", "tokens", "comparison_scope"], keep="last")
    return combined


def fmt(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df.empty:
        return ["No rows available."]
    present = [column for column in columns if column in df.columns]
    lines = ["| " + " | ".join(present) + " |", "| " + " | ".join(["---"] * len(present)) + " |"]
    for _, row in df[present].iterrows():
        values = []
        for column in present:
            value = row[column]
            values.append(fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_summary(tables_dir: Path, grounding: pd.DataFrame, vqa: pd.DataFrame, efficiency: pd.DataFrame) -> None:
    missing_ground = missing_expected(grounding)
    missing_vqa = missing_expected(vqa)
    lines = [
        "# Official Nuwa Matched 500-Sample Summary",
        "",
        "Official Nuwa rows use original LLaVA/Nuwa. Random, uniform, VisionZip, and FastV rows are matched to the same final visual-token counts but use the existing HF LLaVA stack.",
        "",
        "## Missing Matched Rows",
        "",
        "The rows below are missing only when local loader blockers prevented completion. Do not interpret absent rows as zero performance.",
        "",
        "Grounding: " + (", ".join(f"{m}@{t}" for m, t in missing_ground) if missing_ground else "none"),
        "",
        "VQA: " + (", ".join(f"{m}@{t}" for m, t in missing_vqa) if missing_vqa else "none"),
        "",
        "## RefCOCO Grounding",
        "",
        *markdown_table(grounding, ["method", "tokens", "mean_iou", "accuracy_at_0.5", "delta_vs_official_full", "delta_vs_official_nuwa_same_tokens", "comparison_scope"]),
        "",
        "## POPE VQA",
        "",
        *markdown_table(vqa, ["method", "tokens", "accuracy", "f1", "delta_vs_official_full", "delta_vs_official_nuwa_same_tokens", "comparison_scope"]),
        "",
        "## Efficiency",
        "",
        *markdown_table(efficiency, ["task", "method", "tokens", "mean_latency_s", "llm_prefill_latency_s", "gpu_peak_memory_allocated_mb", "comparison_scope"]),
        "",
    ]
    (tables_dir / "official_nuwa_matched_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_all_methods_summary(tables_dir: Path, grounding: pd.DataFrame, vqa: pd.DataFrame, efficiency: pd.DataFrame) -> None:
    lines = [
        "# All Methods With Official Nuwa Summary",
        "",
        "This table set combines prior HF-stack results, official Nuwa, and official-token-budget matched baselines. Use `comparison_scope` and `model_stack` when interpreting fairness.",
        "",
        "## Official-token RefCOCO Rows",
        "",
        *markdown_table(grounding[grounding["tokens"].isin(OFFICIAL_TOKENS)], ["method", "tokens", "mean_iou", "accuracy_at_0.5", "comparison_scope"]),
        "",
        "## Official-token POPE Rows",
        "",
        *markdown_table(vqa[vqa["tokens"].isin(OFFICIAL_TOKENS)], ["method", "tokens", "accuracy", "f1", "comparison_scope"]),
        "",
        "## Official-token Efficiency Rows",
        "",
        *markdown_table(efficiency[efficiency["tokens"].isin(OFFICIAL_TOKENS)], ["task", "method", "tokens", "mean_latency_s", "gpu_peak_memory_allocated_mb", "comparison_scope"]),
        "",
    ]
    (tables_dir / "all_methods_with_official_nuwa_summary.md").write_text("\n".join(lines), encoding="utf-8")


def verify_expected(df: pd.DataFrame, task: str) -> None:
    missing = missing_expected(df)
    if missing:
        raise RuntimeError(f"Missing {task} matched rows: {missing}")
    if df[df["method"] == "official_full"].empty:
        raise RuntimeError(f"Missing {task} official_full row")


def missing_expected(df: pd.DataFrame) -> list[tuple[str, int]]:
    missing: list[tuple[str, int]] = []
    for method in MATCHED_METHODS + ["official_nuwa"]:
        for tokens in OFFICIAL_TOKENS:
            if df[(df["method"] == method) & (df["tokens"] == tokens)].empty:
                missing.append((method, tokens))
    return missing


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    tables_dir = root / args.tables_dir
    tables_dir.mkdir(parents=True, exist_ok=True)

    matched_ground = combined_matched_table(tables_dir, "grounding")
    matched_vqa = combined_matched_table(tables_dir, "vqa")
    matched_eff = combined_efficiency_table(tables_dir)
    if not args.allow_missing:
        verify_expected(matched_ground, "grounding")
        verify_expected(matched_vqa, "vqa")
    elif matched_ground[matched_ground["method"] == "official_full"].empty or matched_vqa[matched_vqa["method"] == "official_full"].empty:
        raise RuntimeError("official_full rows are required even in partial matched mode")

    matched_ground.to_csv(tables_dir / "official_nuwa_matched_grounding.csv", index=False)
    matched_vqa.to_csv(tables_dir / "official_nuwa_matched_vqa.csv", index=False)
    matched_eff.to_csv(tables_dir / "official_nuwa_matched_efficiency.csv", index=False)
    write_summary(tables_dir, matched_ground, matched_vqa, matched_eff)

    all_ground = build_all_methods_table(tables_dir, "grounding")
    all_vqa = build_all_methods_table(tables_dir, "vqa")
    all_eff = build_all_methods_efficiency(tables_dir)
    all_ground.to_csv(tables_dir / "all_methods_with_official_nuwa_grounding.csv", index=False)
    all_vqa.to_csv(tables_dir / "all_methods_with_official_nuwa_vqa.csv", index=False)
    all_eff.to_csv(tables_dir / "all_methods_with_official_nuwa_efficiency.csv", index=False)
    write_all_methods_summary(tables_dir, all_ground, all_vqa, all_eff)

    print(f"Wrote official Nuwa matched and combined tables to {tables_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
