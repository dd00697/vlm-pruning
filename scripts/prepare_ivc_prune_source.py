#!/usr/bin/env python3
"""Prepare and lock the pinned official IVC-Prune source checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from vlm_pruning.artifacts import sha256_file
from vlm_pruning.ivc_campaign import (
    IVC_TRANSFORMERS_VERSION,
    IVC_UPSTREAM_COMMIT,
    IVC_UPSTREAM_URL,
    IVCCampaignError,
)

PATCH = REPO_ROOT / "third_party_patches/ivc_prune_remove_debug_print.patch"
CRITICAL_FILES = (
    "transformers/src/transformers/models/llama/modeling_llama.py",
    "transformers/src/transformers/models/llava/modeling_llava.py",
    "VLMEvalKit/vlmeval/vlm/ivcp/llava_ivcp.py",
    "VLMEvalKit/vlmeval/config.py",
    "README.md",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkout",
        type=Path,
        default=REPO_ROOT / "third_party/IVC-Prune",
        help="Dedicated checkout path; must be absent or already match the pinned source.",
    )
    parser.add_argument(
        "--lock-output",
        type=Path,
        default=REPO_ROOT / "experiments/ivc_prune_refcoco_val_20260825/SOURCE_LOCK.json",
    )
    return parser.parse_args()


def prepare(args: argparse.Namespace) -> int:
    checkout = args.checkout.resolve()
    if not PATCH.is_file():
        raise IVCCampaignError(f"Missing registered source patch: {PATCH}")
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", IVC_UPSTREAM_URL, str(checkout)], check=True
        )
        subprocess.run(["git", "checkout", "--detach", IVC_UPSTREAM_COMMIT], cwd=checkout, check=True)
    if not (checkout / ".git").exists():
        raise IVCCampaignError(f"IVC source path is not a Git checkout: {checkout}")
    if git("rev-parse", "HEAD", cwd=checkout) != IVC_UPSTREAM_COMMIT:
        raise IVCCampaignError("Existing IVC checkout is not at the registered upstream commit")
    status = git("status", "--porcelain", cwd=checkout)
    if not status:
        subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=checkout, check=True)
        subprocess.run(["git", "apply", str(PATCH)], cwd=checkout, check=True)
    actual_diff = subprocess.check_output(["git", "diff", "--binary"], cwd=checkout)
    # Git may rewrite abbreviated blob headers, so bind the semantic patch by reverse applicability.
    reverse = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(PATCH)],
        cwd=checkout,
        check=False,
        capture_output=True,
    )
    if reverse.returncode != 0:
        raise IVCCampaignError("Checkout changes do not contain the registered debug-print patch")
    changed = git("diff", "--name-only", cwd=checkout).splitlines()
    expected_changed = ["transformers/src/transformers/models/llava/modeling_llava.py"]
    if changed != expected_changed:
        raise IVCCampaignError(f"IVC checkout contains unregistered changes: {changed}")
    untracked = git("ls-files", "--others", "--exclude-standard", cwd=checkout).splitlines()
    if untracked:
        raise IVCCampaignError(f"IVC checkout contains unregistered files: {untracked}")
    version_file = checkout / "transformers/src/transformers/__init__.py"
    if f'__version__ = "{IVC_TRANSFORMERS_VERSION}"' not in version_file.read_text(encoding="utf-8"):
        raise IVCCampaignError("Pinned IVC Transformers version differs from the registered contract")
    critical = {name: sha256_file(checkout / name) for name in CRITICAL_FILES}
    lock = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "upstream_url": IVC_UPSTREAM_URL,
        "upstream_commit": IVC_UPSTREAM_COMMIT,
        "checkout": str(checkout),
        "transformers_version": IVC_TRANSFORMERS_VERSION,
        "registered_patch": str(PATCH.relative_to(REPO_ROOT)),
        "registered_patch_sha256": sha256_file(PATCH),
        "post_patch_diff_sha256": hashlib.sha256(actual_diff).hexdigest(),
        "changed_files": changed,
        "untracked_files": untracked,
        "critical_file_sha256": critical,
        "status": "SOURCE_LOCKED",
    }
    atomic_json(args.lock_output.resolve(), lock)
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


def main() -> int:
    try:
        return prepare(parse_args())
    except (IVCCampaignError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
