#!/usr/bin/env python
"""Generate or run the focused UG-Prune 58-token add-on ablations."""

from __future__ import annotations

import argparse
import shlex
import subprocess

from run_ug_prune_ablation_sweep import Ablation, build_command


ADDON_RUNS = (
    ("refcoco", Ablation("map_family_clip", "map_family", ("--ug-map-family", "clip"), core=False)),
    ("pope", Ablation("map_family_clip", "map_family", ("--ug-map-family", "clip"), core=False)),
    ("textvqa", Ablation("map_family_clip", "map_family", ("--ug-map-family", "clip"), core=False)),
    ("refcoco", Ablation("radius_0", "context_radius", ("--ug-context-radius", "0"))),
    ("refcoco", Ablation("radius_2", "context_radius", ("--ug-context-radius", "2"))),
    ("refcoco", Ablation("anchor_corners", "anchor_pattern", ("--ug-anchor-pattern", "corners"), core=False)),
    ("refcoco", Ablation("anchor_grid", "anchor_pattern", ("--ug-anchor-pattern", "grid"), core=False)),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--budget", type=int, default=58)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-default-limits", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="vlm-pruning")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-group", default="ug-prune-ablate-58-addon")
    parser.add_argument("--wandb-job-type", default="ablation-58-addon")
    parser.add_argument("--extra-args", default="")
    return parser.parse_args(argv)


def command_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        smoke=False,
        precision=args.precision,
        limit=args.limit,
        no_default_limits=args.no_default_limits,
        resume=not args.no_resume,
        skip_existing=not args.no_skip_existing,
        wandb=args.wandb,
        no_wandb=not args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        wandb_group=args.wandb_group,
        wandb_job_type=args.wandb_job_type,
        extra_args=args.extra_args,
    )


def main() -> int:
    args = parse_args()
    base_args = command_args(args)
    for dataset, ablation in ADDON_RUNS:
        run_name = f"{dataset}_ug_addon_{ablation.name}_tok{args.budget}_{args.precision}"
        command = build_command(dataset, args.budget, ablation, base_args)
        command[command.index("--run-name") + 1] = run_name
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
