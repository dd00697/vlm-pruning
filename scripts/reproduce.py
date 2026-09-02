#!/usr/bin/env python3
"""Plan or execute the curated dissertation reproduction workflow."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "reproduction.yaml"


class ReproductionError(RuntimeError):
    pass


def load_registry() -> dict[str, Any]:
    value = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != 2:
        raise ReproductionError("reproduction.yaml must be a version 2 mapping")
    if not isinstance(value.get("stages"), dict):
        raise ReproductionError("reproduction.yaml has no stages mapping")
    return value


def render(value: Any, context: dict[str, Any]) -> str:
    try:
        return str(value).format_map(context)
    except KeyError as exc:
        raise ReproductionError(f"unknown template field: {exc.args[0]}") from exc


def expand(registry: dict[str, Any], requested: str, context: dict[str, Any]) -> list[tuple[str, list[str]]]:
    pipelines = registry.get("pipelines", {})
    stages = pipelines.get(requested, [requested])
    if not isinstance(stages, list) or not stages:
        raise ReproductionError(f"empty pipeline: {requested}")
    result: list[tuple[str, list[str]]] = []
    for stage_name in stages:
        if stage_name not in registry["stages"]:
            raise ReproductionError(f"unknown stage: {stage_name}")
        for specification in registry["stages"][stage_name].get("commands", []):
            for case in specification.get("foreach", [{}]):
                local = {**context, **case}
                command = [render(item, local) for item in specification["argv"]]
                if not command or not all(command):
                    raise ReproductionError(f"empty command in stage {stage_name}")
                result.append((stage_name, command))
    return result


def preflight(commands: list[tuple[str, list[str]]]) -> None:
    missing = []
    for _, command in commands:
        if len(command) > 1 and command[1].endswith(".py") and not (ROOT / command[1]).is_file():
            missing.append(command[1])
    if missing:
        raise ReproductionError("missing registered scripts: " + ", ".join(sorted(set(missing))))


def parse_args(registry: dict[str, Any]) -> argparse.Namespace:
    choices = sorted({*registry["stages"], *registry.get("pipelines", {})})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=choices, default="all")
    parser.add_argument("--execute", action="store_true", help="execute commands; default prints the plan")
    parser.add_argument("--max-workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ivc-python", default=os.environ.get("IVC_PYTHON", "python3.10"))
    parser.add_argument("--ivc-source", default=os.environ.get("IVC_SOURCE", "third_party/IVC-Prune"))
    parser.add_argument("--ivc-model", default=os.environ.get("IVC_MODEL_SNAPSHOT", "MODEL_SNAPSHOT_REQUIRED"))
    return parser.parse_args()


def main() -> int:
    try:
        registry = load_registry()
        args = parse_args(registry)
        context = {
            "python": os.environ.get("PYTHON_BIN", sys.executable),
            "max_workers": args.max_workers,
            "ivc_python": args.ivc_python,
            "ivc_source": args.ivc_source,
            "ivc_model": args.ivc_model,
        }
        commands = expand(registry, args.stage, context)
        preflight(commands)
        print(f"Dissertation reproduction: {args.stage} ({'execute' if args.execute else 'plan only'})")
        current = None
        for stage, command in commands:
            if stage != current:
                print(f"\n[{stage}] {registry['stages'][stage]['description']}")
                current = stage
            print("  + " + shlex.join(command), flush=True)
            if args.execute:
                completed = subprocess.run(command, cwd=ROOT, check=False)
                if completed.returncode:
                    return completed.returncode
        return 0
    except (OSError, ValueError, yaml.YAMLError, ReproductionError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
