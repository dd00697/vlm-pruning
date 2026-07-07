#!/usr/bin/env python
"""Generate or run UG-Prune ablation commands."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass


DATASETS = {
    "refcoco": {
        "task": "grounding",
        "manifest": "data/manifests/official_baselines/refcoco_val.jsonl",
        "smoke_manifest": "data/manifests/official_baselines/refcoco_val_limit5.jsonl",
        "default_limit": 500,
    },
    "pope": {
        "task": "vqa",
        "manifest": "data/manifests/official_baselines/pope_val.jsonl",
        "smoke_manifest": "data/manifests/official_baselines/pope_val_limit5.jsonl",
        "default_limit": 2500,
    },
    "textvqa": {
        "task": "vqa",
        "manifest": "data/manifests/official_baselines/textvqa_validation.jsonl",
        "smoke_manifest": "data/manifests/official_baselines/textvqa_validation_limit5.jsonl",
        "default_limit": 2500,
    },
}


@dataclass(frozen=True)
class Ablation:
    name: str
    family: str
    args: tuple[str, ...]
    core: bool = True


ABLATIONS = [
    Ablation("role_target_only", "role", ("--ug-role-mode", "target_only")),
    Ablation("role_target_context", "role", ("--ug-role-mode", "target_context")),
    Ablation("role_target_context_anchor", "role", ("--ug-role-mode", "target_context_anchor")),
    Ablation("map_query_max", "map_source", ("--ug-map-source", "query_max")),
    Ablation("map_query_pooled", "map_source", ("--ug-map-source", "query_pooled")),
    Ablation("layer_4", "layer", ("--ug-fusion-mode", "single_layer", "--ug-probe-layers", "4")),
    Ablation("layer_8", "layer", ("--ug-fusion-mode", "single_layer", "--ug-probe-layers", "8")),
    Ablation("layer_12", "layer", ("--ug-fusion-mode", "single_layer", "--ug-probe-layers", "12")),
    Ablation("layer_soft_4_8_12", "layer", ("--ug-fusion-mode", "soft_average", "--ug-probe-layers", "4,8,12")),
    Ablation("radius_0", "context_radius", ("--ug-context-radius", "0")),
    Ablation("radius_1", "context_radius", ("--ug-context-radius", "1")),
    Ablation("radius_2", "context_radius", ("--ug-context-radius", "2")),
    Ablation("anchor_none", "anchor", ("--ug-anchor-pattern", "none")),
    Ablation("anchor_fps", "anchor", ("--ug-anchor-pattern", "fps")),
    Ablation("map_family_unrolled", "map_family", ("--ug-map-family", "unrolled"), core=False),
    Ablation("map_family_clip", "map_family", ("--ug-map-family", "clip"), core=False),
    Ablation("position_compact", "position", ("--position-mode", "compact"), core=False),
    Ablation("position_raster_compact", "position", ("--position-mode", "raster_compact"), core=False),
    Ablation("position_gap_preserving", "position", ("--position-mode", "gap_preserving"), core=False),
    Ablation("dynamic_hard_best", "dynamic_layer", ("--ug-fusion-mode", "hard_best", "--ug-probe-layers", "4,8,12"), core=False),
    Ablation("anchor_corners", "anchor_pattern", ("--ug-anchor-pattern", "corners"), core=False),
    Ablation("anchor_grid", "anchor_pattern", ("--ug-anchor-pattern", "grid"), core=False),
    Ablation("anchor_pattern_fps", "anchor_pattern", ("--ug-anchor-pattern", "fps"), core=False),
    Ablation("fusion_hard_best", "fusion", ("--ug-fusion-mode", "hard_best", "--ug-probe-layers", "4,8,12"), core=False),
    Ablation("fusion_soft_average", "fusion", ("--ug-fusion-mode", "soft_average", "--ug-probe-layers", "4,8,12"), core=False),
]

FOCUSED_ABLATION_NAMES = (
    "role_target_only",
    "role_target_context",
    "role_target_context_anchor",
    "map_query_max",
    "map_query_pooled",
    "layer_4",
    "layer_8",
    "layer_12",
    "layer_soft_4_8_12",
    "fusion_hard_best",
    "position_compact",
    "position_raster_compact",
    "position_gap_preserving",
    "anchor_none",
    "anchor_fps",
)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="refcoco,pope,textvqa")
    parser.add_argument("--budgets", default="288,58")
    parser.add_argument("--ablation-set", default="core", choices=["core", "focused", "all"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-default-limits", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="vlm-pruning")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-job-type", default=None)
    parser.add_argument("--extra-args", default="")
    return parser.parse_args(argv)


def selected_ablations(ablation_set: str) -> list[Ablation]:
    if ablation_set == "core":
        return [item for item in ABLATIONS if item.core]
    if ablation_set == "focused":
        by_name = {item.name: item for item in ABLATIONS}
        return [by_name[name] for name in FOCUSED_ABLATION_NAMES]
    return list(ABLATIONS)


def build_command(dataset: str, budget: int, ablation: Ablation, args: argparse.Namespace) -> list[str]:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    spec = DATASETS[dataset]
    manifest = spec["smoke_manifest"] if args.smoke else spec["manifest"]
    if args.limit is not None:
        limit = args.limit
    elif args.smoke:
        limit = 5
    elif args.no_default_limits:
        limit = None
    else:
        limit = int(spec["default_limit"])
    run_name = f"{dataset}_ug_ablate_{ablation.name}_tok{budget}_{args.precision}"
    if args.smoke:
        run_name = f"smoke_{run_name}"
    command = [
        sys.executable,
        "scripts/run_ug_prune.py",
        "--dataset",
        dataset,
        "--task",
        str(spec["task"]),
        "--manifest",
        manifest,
        "--target-visual-tokens",
        str(budget),
        "--precision",
        args.precision,
        "--run-name",
        run_name,
        "--ablation-name",
        ablation.name,
        "--ablation-family",
        ablation.family,
        *ablation.args,
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if args.resume:
        command.append("--resume")
    if args.skip_existing:
        command.append("--skip-existing")
    if args.smoke or args.no_wandb:
        command.append("--no-wandb")
    elif args.wandb:
        command.extend(["--wandb", "--wandb-project", args.wandb_project, "--wandb-mode", args.wandb_mode])
        if args.wandb_entity:
            command.extend(["--wandb-entity", args.wandb_entity])
        if args.wandb_group:
            command.extend(["--wandb-group", args.wandb_group])
        if args.wandb_job_type:
            command.extend(["--wandb-job-type", args.wandb_job_type])
        command.extend(["--wandb-tags", f"ug-prune,ablation,{ablation.family},{ablation.name},budget{budget}"])
    if args.extra_args.strip():
        command.extend(shlex.split(args.extra_args))
    return command


def main() -> int:
    args = parse_args()
    datasets = parse_csv(args.datasets)
    budgets = [int(item) for item in parse_csv(args.budgets)]
    commands = [
        build_command(dataset, budget, ablation, args)
        for dataset in datasets
        for budget in budgets
        for ablation in selected_ablations(args.ablation_set)
    ]
    for command in commands:
        print(shlex.join(command))
        if args.execute:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
