#!/usr/bin/env python3
"""Plan or execute the registered dissertation reproduction workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "reproduction.yaml"
FINAL = ROOT / "experiments/final_evaluation"


class ReproductionError(RuntimeError):
    """Raised when the reproduction registry or environment is invalid."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReproductionError(f"missing required file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReproductionError(f"expected a JSON object: {path}")
    return value


def load_registry(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise ReproductionError(f"missing reproduction registry: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ReproductionError("reproduction.yaml must be a version 1 mapping")
    if not isinstance(value.get("stages"), dict):
        raise ReproductionError("reproduction.yaml must define a stages mapping")
    return value


def _render(value: Any, context: dict[str, Any]) -> str:
    try:
        return str(value).format_map(context)
    except KeyError as exc:
        raise ReproductionError(f"unknown command template field: {exc.args[0]}") from exc


def expand_stage(
    registry: dict[str, Any],
    stage_name: str,
    context: dict[str, Any],
) -> list[list[str]]:
    stages = registry["stages"]
    if stage_name not in stages:
        raise ReproductionError(f"unknown reproduction stage: {stage_name}")
    stage = stages[stage_name]
    if not isinstance(stage, dict) or not isinstance(stage.get("commands"), list):
        raise ReproductionError(f"stage {stage_name!r} must define a command list")

    expanded: list[list[str]] = []
    for spec in stage["commands"]:
        if not isinstance(spec, dict) or not isinstance(spec.get("argv"), list):
            raise ReproductionError(f"invalid command in stage {stage_name!r}")
        condition = spec.get("when")
        if condition is not None and not bool(context.get(str(condition), False)):
            continue
        cases = spec.get("foreach", [{}])
        if not isinstance(cases, list) or not cases:
            raise ReproductionError(f"empty foreach list in stage {stage_name!r}")
        for case in cases:
            if not isinstance(case, dict):
                raise ReproductionError(f"foreach cases in {stage_name!r} must be mappings")
            command_context = {**context, **case}
            command = [_render(item, command_context) for item in spec["argv"]]
            if not command or not all(command):
                raise ReproductionError(f"empty command argument in stage {stage_name!r}")
            expanded.append(command)
    return expanded


def selected_stages(registry: dict[str, Any], requested: str) -> list[str]:
    pipelines = registry.get("pipelines", {})
    names = pipelines.get(requested, [requested]) if isinstance(pipelines, dict) else [requested]
    if not isinstance(names, list) or not names:
        raise ReproductionError(f"pipeline {requested!r} must contain at least one stage")
    for name in names:
        if name not in registry["stages"]:
            raise ReproductionError(f"pipeline {requested!r} references unknown stage {name!r}")
    return [str(name) for name in names]


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_record() -> dict[str, str] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None
    if not output:
        return None
    name, memory_mib, driver = [item.strip() for item in output[0].split(",", 2)]
    return {"name": name, "memory_mib": memory_mib, "driver": driver}


def preflight(args: argparse.Namespace, *, skip_gpu: bool) -> bool:
    protocol = read_json(FINAL / "protocol/protocol.json")
    frozen = read_json(FINAL / "frozen_method.json")
    errors: list[str] = []
    warnings: list[str] = []

    def drift(message: str, allowed: bool) -> None:
        (warnings if allowed else errors).append(message)

    expected_python = str(protocol["environment"]["python"]).split()[0]
    actual_python = platform.python_version()
    if actual_python != expected_python:
        drift(
            f"Python is {actual_python}; registered version is {expected_python}",
            args.allow_environment_drift,
        )

    for package in ("torch", "transformers", "datasets", "wandb"):
        expected = protocol["environment"].get(package)
        actual = package_version(package)
        if actual != expected:
            drift(
                f"{package} is {actual or 'not installed'}; registered version is {expected}",
                args.allow_environment_drift,
            )

    source_hashes = frozen.get("registered_source_sha256", {})
    source_mismatches: list[str] = []
    for relative, expected in source_hashes.items():
        path = (ROOT / relative).resolve()
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            source_mismatches.append(relative)
    if source_mismatches:
        drift(
            "registered source hash mismatch: " + ", ".join(source_mismatches),
            args.allow_source_drift,
        )

    for name, record in protocol.get("manifests", {}).items():
        path = ROOT / str(record["path"])
        actual = sha256(path) if path.is_file() else None
        if actual != record.get("sha256"):
            errors.append(f"{name} manifest is missing or has the wrong SHA-256: {path}")

    timing = protocol.get("timing", {})
    timing_path = ROOT / str(timing.get("subset_manifest", ""))
    if not timing_path.is_file() or sha256(timing_path) != timing.get("subset_sha256"):
        errors.append("registered timing subset is missing or has the wrong SHA-256")

    expected_revisions = {
        "FastV": (
            ROOT / "third_party/FastV",
            protocol["held_out_methods"]["fastv"]["revision"],
        ),
        "VisionZip": (
            ROOT / "third_party/VisionZip",
            protocol["held_out_methods"]["visionzip"]["revision"],
        ),
    }
    for name, (path, expected) in expected_revisions.items():
        actual = git_revision(path)
        if actual != expected:
            errors.append(f"{name} revision is {actual or 'unavailable'}; expected {expected}")

    hf_home = Path(os.environ.get("HF_HOME", "/workspace/cache/huggingface"))
    checkpoint_root = hf_home / "hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
    snapshots = sorted(path.name for path in checkpoint_root.glob("*") if path.is_dir())
    if not snapshots:
        errors.append(
            "LLaVA checkpoint is not cached under "
            f"{checkpoint_root}; set HF_HOME or download llava-hf/llava-1.5-7b-hf"
        )
    warnings.append(
        "The registered artifacts record the LLaVA model name but not an immutable Hub "
        "revision; checkpoint identity cannot be proved from the run metadata alone."
    )

    gpu = None if skip_gpu else gpu_record()
    if not skip_gpu:
        if gpu is None:
            drift("no CUDA GPU was visible through nvidia-smi", args.allow_hardware_drift)
        elif gpu["name"] != "NVIDIA A40":
            drift(
                f"GPU is {gpu['name']}; registered timing hardware is NVIDIA A40",
                args.allow_hardware_drift,
            )

    report = {
        "status": "passed" if not errors else "failed",
        "repository": str(ROOT),
        "python": {"actual": actual_python, "expected": expected_python},
        "source_hashes": {"checked": len(source_hashes), "mismatches": source_mismatches},
        "checkpoint_snapshots_found": snapshots,
        "gpu": gpu,
        "warnings": warnings,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return not errors


def parse_args(registry: dict[str, Any]) -> argparse.Namespace:
    stage_choices = sorted(
        {"preflight", *registry["stages"].keys(), *registry.get("pipelines", {}).keys()}
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=stage_choices, default="all")
    parser.add_argument("--execute", action="store_true", help="run commands; default is plan only")
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--wandb-mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--full-audit", action="store_true", help="include strict PDF QA in verify")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true", help="omit GPU checks in preflight")
    parser.add_argument("--allow-source-drift", action="store_true")
    parser.add_argument("--allow-environment-drift", action="store_true")
    parser.add_argument("--allow-hardware-drift", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        registry = load_registry()
        args = parse_args(registry)
        if args.full_audit and args.allow_source_drift:
            raise ReproductionError("--full-audit cannot be combined with --allow-source-drift")

        if args.stage == "preflight":
            return 0 if preflight(args, skip_gpu=args.skip_gpu) else 3

        stages = selected_stages(registry, args.stage)
        stage_limits = [int(registry["stages"][name].get("max_parallel", 3)) for name in stages]
        if args.max_parallel is None:
            args.max_parallel = min(stage_limits)
        if not 1 <= args.max_parallel <= 3:
            raise ReproductionError("--max-parallel must be between 1 and 3")
        for stage_name, maximum in zip(stages, stage_limits, strict=True):
            if args.max_parallel > maximum:
                raise ReproductionError(
                    f"stage {stage_name!r} permits at most --max-parallel {maximum}"
                )

        dissertation_root = Path(
            os.environ.get("DISSERTATION_ROOT", str(ROOT.parent / "Dissertation_Draft"))
        ).resolve()
        context = {
            "python": os.environ.get("PYTHON_BIN", sys.executable),
            "max_parallel": args.max_parallel,
            "wandb_mode": args.wandb_mode,
            "dissertation_root": str(dissertation_root),
            "full_audit": args.full_audit,
        }

        commands: list[tuple[str, list[str]]] = []
        for stage_name in stages:
            for command in expand_stage(registry, stage_name, context):
                commands.append((stage_name, command))

        if args.execute and "verify" in stages:
            analysis = dissertation_root / "analysis/build_final_evaluation.py"
            if not analysis.is_file():
                raise ReproductionError(
                    f"missing {analysis}; set DISSERTATION_ROOT to the dissertation source"
                )

        needs_gpu = any(bool(registry["stages"][name].get("gpu")) for name in stages)
        if args.execute and not args.skip_preflight:
            if not preflight(args, skip_gpu=not needs_gpu):
                return 3

        mode = "execute" if args.execute else "plan only; add --execute to run"
        print(f"Dissertation reproduction: {args.stage} ({mode})")
        current_stage: str | None = None
        for stage_name, command in commands:
            if stage_name != current_stage:
                description = registry["stages"][stage_name].get("description", stage_name)
                print(f"\n[{stage_name}] {description}")
                current_stage = stage_name
            print(f"  + {shlex.join(command)}", flush=True)
            if args.execute:
                completed = subprocess.run(command, cwd=ROOT, check=False)
                if completed.returncode != 0:
                    return completed.returncode
        return 0
    except (OSError, ValueError, yaml.YAMLError, ReproductionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
