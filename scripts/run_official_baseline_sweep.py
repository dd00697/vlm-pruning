#!/usr/bin/env python
"""Prepare data and run the official baseline sweep sequentially."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = Path("data/manifests/official_baselines")
TABLES_ROOT = Path("experiments/tables/official_baselines")
BACKGROUND_LOG_ROOT = Path("experiments/background_logs_official_baselines")

VQA_DATASETS = [
    {"dataset": "pope", "split": "val", "task": "vqa", "stress": True},
    {"dataset": "aokvqa", "split": "validation", "task": "vqa", "stress": True},
    {"dataset": "scienceqa_img", "split": "validation", "task": "vqa", "stress": True},
    {"dataset": "ocr_vqa", "split": "test", "task": "vqa", "stress": True, "fallback": {"dataset": "textvqa", "split": "validation"}},
    {"dataset": "mmmu", "split": "validation", "task": "vqa", "stress": False},
    {"dataset": "mme", "split": "test", "task": "vqa", "stress": False},
    {"dataset": "seed_bench", "split": "test", "task": "vqa", "stress": False},
    {"dataset": "mmvet", "split": "test", "task": "vqa", "stress": False},
]

REFCOCO = {"dataset": "refcoco", "split": "val", "task": "grounding", "stress": True}
BROAD_TOKENS = [432, 288, 144, 58]
STRESS_TOKENS = [48, 32, 16]
RANDOM_SEEDS = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset names or all.")
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--model-path", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "vlm-pruning"))
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--ocr-dataset",
        choices=["ocr_vqa", "textvqa"],
        default="ocr_vqa",
        help="Dataset to use for the OCR/text-in-image slot.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest-root", type=Path, default=MANIFEST_ROOT)
    parser.add_argument("--tables-root", type=Path, default=TABLES_ROOT)
    parser.add_argument("--background-log-root", type=Path, default=BACKGROUND_LOG_ROOT)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for local checks.")
    parser.add_argument(
        "--methods",
        default="full,uniform,random",
        help="Comma-separated methods to schedule: full,uniform,random. Defaults to all.",
    )
    parser.add_argument(
        "--tokens",
        default="all",
        help="Comma-separated pruned token budgets, or all. Applies to uniform/random only.",
    )
    parser.add_argument(
        "--random-seeds",
        default="0,1,2",
        help="Comma-separated random seeds for random pruning.",
    )
    parser.add_argument(
        "--stop-after-full-token-baselines",
        action="store_true",
        help="Run only the 576-token full baselines.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_datasets(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = []
    for spec in VQA_DATASETS:
        if spec["dataset"] == "ocr_vqa" and args.ocr_dataset == "textvqa":
            replacement = dict(spec)
            replacement.update(spec["fallback"])
            replacement.pop("fallback", None)
            datasets.append(replacement)
        else:
            datasets.append(spec)
    datasets.append(REFCOCO)
    if args.datasets == "all":
        return datasets
    wanted = {item.strip().lower().replace("-", "_") for item in args.datasets.split(",") if item.strip()}
    return [item for item in datasets if item["dataset"].lower().replace("-", "_") in wanted]


def manifest_path(args: argparse.Namespace, dataset: str, split: str, limit: int | None) -> Path:
    suffix = f"_limit{limit}" if limit is not None else ""
    return args.manifest_root / f"{dataset}_{split}{suffix}.jsonl"


def run_cmd(cmd: list[str], log_path: Path | None, dry_run: bool) -> int:
    printable = " ".join(cmd)
    print(printable)
    if dry_run:
        return 0
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(printable + "\n\n")
            process = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
        return process.returncode
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def prepare_dataset(args: argparse.Namespace, spec: dict[str, Any], limit: int | None) -> tuple[dict[str, Any], Path | None, str | None]:
    dataset = spec["dataset"]
    split = spec["split"]
    output = manifest_path(args, dataset, split, limit)
    if output.exists() and output.stat().st_size > 0:
        return spec, output, None
    cmd = [
        sys.executable,
        "scripts/prepare_official_baseline_data.py",
        "--dataset",
        dataset,
        "--split",
        split,
        "--output",
        str(output),
        "--raw-root",
        str(args.raw_root),
        "--download",
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    code = run_cmd(cmd, args.background_log_root / "prepare" / f"{dataset}_{split}.log", args.dry_run)
    if code == 0:
        return spec, output, None
    fallback = spec.get("fallback")
    if fallback:
        fallback_spec = dict(spec)
        fallback_spec.update(fallback)
        fallback_spec.pop("fallback", None)
        fallback_output = manifest_path(args, fallback_spec["dataset"], fallback_spec["split"], limit)
        cmd = [
            sys.executable,
            "scripts/prepare_official_baseline_data.py",
            "--dataset",
            fallback_spec["dataset"],
            "--split",
            fallback_spec["split"],
            "--output",
            str(fallback_output),
            "--raw-root",
            str(args.raw_root),
            "--download",
        ]
        if limit is not None:
            cmd += ["--limit", str(limit)]
        fallback_code = run_cmd(cmd, args.background_log_root / "prepare" / f"{fallback_spec['dataset']}_{fallback_spec['split']}.log", args.dry_run)
        if fallback_code == 0:
            return fallback_spec, fallback_output, f"{dataset} blocked; used {fallback_spec['dataset']} fallback"
    return spec, None, f"prepare failed for {dataset} split {split}"


def parse_seed_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_method_list(value: str) -> set[str]:
    methods = {item.strip().lower() for item in value.split(",") if item.strip()}
    allowed = {"full", "uniform", "random"}
    unknown = methods - allowed
    if unknown:
        raise ValueError(f"Unknown method(s) in --methods: {sorted(unknown)}. Allowed: {sorted(allowed)}")
    if not methods:
        raise ValueError("--methods must include at least one of full,uniform,random")
    return methods


def parse_token_filter(value: str) -> set[int] | None:
    if value.strip().lower() == "all":
        return None
    tokens = {int(item.strip()) for item in value.split(",") if item.strip()}
    if not tokens:
        raise ValueError("--tokens must be all or a comma-separated list of integers")
    return tokens


def pruned_tokens_for_spec(spec: dict[str, Any], token_filter: set[int] | None) -> list[int]:
    tokens = BROAD_TOKENS + (STRESS_TOKENS if spec.get("stress") else [])
    if token_filter is not None:
        tokens = [token for token in tokens if token in token_filter]
    return tokens


def run_specs(dataset_specs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    methods = parse_method_list(args.methods)
    token_filter = parse_token_filter(args.tokens)
    random_seeds = parse_seed_list(args.random_seeds)
    if "full" in methods:
        for spec in dataset_specs:
            specs.append({**spec, "method": "full", "tokens": 576, "seed": 0})
    if args.stop_after_full_token_baselines:
        return specs
    pruned_specs: list[dict[str, Any]] = []
    for spec in dataset_specs:
        for token in pruned_tokens_for_spec(spec, token_filter):
            pruned_specs.append({**spec, "tokens": token})
    if "uniform" in methods:
        for item in pruned_specs:
            specs.append({**item, "method": "uniform", "seed": 0})
    if "random" in methods:
        for spec in dataset_specs:
            for token in pruned_tokens_for_spec(spec, token_filter):
                for seed in random_seeds:
                    specs.append({**spec, "method": "random", "tokens": token, "seed": seed})
    return specs


def run_experiment(args: argparse.Namespace, spec: dict[str, Any], manifest: Path, idx: int, total: int, suite_start: float, limit: int | None) -> int:
    dataset = spec["dataset"]
    tokens = int(spec["tokens"])
    method = spec["method"]
    seed = int(spec["seed"])
    cmd = [
        sys.executable,
        "scripts/run_official_baseline.py",
        "--dataset",
        dataset,
        "--task",
        spec["task"],
        "--manifest",
        str(manifest),
        "--split",
        spec["split"],
        "--method",
        method,
        "--target-visual-tokens",
        str(tokens),
        "--precision",
        args.precision,
        "--model-path",
        args.model_path,
        "--wandb-project",
        args.wandb_project,
        "--wandb-mode",
        args.wandb_mode,
        "--suite-total-runs",
        str(total),
        "--suite-completed-runs",
        str(idx),
        "--suite-start-time",
        str(suite_start),
        "--estimate-runtime",
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    if method == "random":
        cmd += ["--seed", str(seed)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if args.resume:
        cmd.append("--resume")
    if args.no_wandb or args.wandb_mode == "disabled":
        cmd.append("--no-wandb")
    else:
        cmd += ["--wandb", "--wandb-project", args.wandb_project]
    log_name = f"{dataset}_{method}_tokens{tokens}_seed{seed}.log"
    return run_cmd(cmd, args.background_log_root / "runs" / log_name, args.dry_run)


def write_blockers(args: argparse.Namespace, blockers: list[dict[str, Any]]) -> None:
    if not blockers:
        return
    path = args.tables_root / "dataset_blockers.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for blocker in blockers:
            handle.write(json.dumps(blocker, ensure_ascii=True) + "\n")


def main() -> int:
    args = parse_args()
    try:
        parse_method_list(args.methods)
        parse_token_filter(args.tokens)
        parse_seed_list(args.random_seeds)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    limit = args.limit
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    args.tables_root.mkdir(parents=True, exist_ok=True)
    dataset_specs = selected_datasets(args)
    manifests: dict[str, Path] = {}
    effective_specs: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    if not args.skip_prepare:
        for spec in dataset_specs:
            effective, manifest, note = prepare_dataset(args, spec, limit)
            if note:
                blockers.append({"dataset": spec["dataset"], "limit": limit, "note": note, "time": time.time()})
            if manifest is not None:
                manifests[effective["dataset"]] = manifest
                effective_specs.append(effective)
    else:
        for spec in dataset_specs:
            manifest = manifest_path(args, spec["dataset"], spec["split"], limit)
            if manifest.exists():
                manifests[spec["dataset"]] = manifest
                effective_specs.append(spec)
            else:
                blockers.append({"dataset": spec["dataset"], "limit": limit, "note": f"manifest missing: {manifest}", "time": time.time()})
    write_blockers(args, blockers)
    if args.prepare_only:
        return 0 if effective_specs else 2
    specs = run_specs(effective_specs, args)
    total = len(specs)
    suite_start = time.time()
    failures = []
    for idx, spec in enumerate(specs):
        manifest = manifests.get(spec["dataset"])
        if manifest is None:
            continue
        code = run_experiment(args, spec, manifest, idx, total, suite_start, limit)
        if code != 0:
            failures.append({**spec, "exit_code": code})
            write_blockers(args, [{"dataset": spec["dataset"], "limit": limit, "note": f"run failed: {spec}", "exit_code": code, "time": time.time()}])
            print(f"Run failed but sweep will continue: {spec} exit={code}", file=sys.stderr)
    if failures:
        print(f"Completed sweep with {len(failures)} failed run(s). See {args.tables_root / 'dataset_blockers.jsonl'}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
