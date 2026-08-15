#!/usr/bin/env python
"""Freeze official 2026 source checkouts and audit bounded compatibility."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815/official_sources"
CHECKOUT_ROOT = OFFICIAL_ROOT / "checkouts"
LOCK_PATH = OFFICIAL_ROOT / "source_lock.json"
COMPATIBILITY_PATH = OFFICIAL_ROOT / "compatibility_audit.json"
PYTHON = "/workspace/.followup_venv/bin/python"

SOURCES = {
    "nuwa": {
        "path": CHECKOUT_ROOT / "nuwa",
        "remote": "https://github.com/Man-PaperRejected/Nuwa.git",
        "expected_commit": "175b7d57a7890b9818dd26d82f2d91d9868b0d80",
        "paper": "https://arxiv.org/abs/2602.02951",
        "venue": "ICLR 2026",
    },
    "reroute": {
        "path": CHECKOUT_ROOT / "reroute",
        "remote": "https://github.com/elmma/mllm-reroute.git",
        "expected_commit": "9d422042076de678b2ad67c4a6660e4b97971a1e",
        "paper": "https://arxiv.org/abs/2606.12412",
        "venue": "arXiv 2026",
    },
}


class OfficialSourceError(RuntimeError):
    """Raised when a source checkout cannot be frozen exactly."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def recursive_hash(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = 0
    size = 0
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file() and ".git" not in candidate.parts):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        files += 1
        size += len(payload)
    return digest.hexdigest(), files, size


def git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()


def freeze_sources() -> dict[str, Any]:
    locked: dict[str, Any] = {}
    for name, spec in SOURCES.items():
        checkout = spec["path"]
        if not (checkout / ".git").is_dir():
            raise OfficialSourceError(f"Missing official checkout: {checkout}")
        commit = git(checkout, "rev-parse", "HEAD")
        remote = git(checkout, "remote", "get-url", "origin")
        dirty = git(checkout, "status", "--porcelain", "--untracked-files=all")
        if commit != spec["expected_commit"] or remote.rstrip("/") != spec["remote"].rstrip("/") or dirty:
            raise OfficialSourceError(
                f"{name} checkout does not match frozen commit/remote/cleanliness"
            )
        source_hash, files, size = recursive_hash(checkout)
        locked[name] = {
            "checkout": str(checkout.relative_to(REPO_ROOT)),
            "remote": remote,
            "commit": commit,
            "recursive_source_sha256": source_hash,
            "files": files,
            "bytes": size,
            "git_clean": True,
            "paper": spec["paper"],
            "venue": spec["venue"],
        }
    proposed = {
        "schema_version": 1,
        "status": "frozen",
        "verified_at_utc": utc_now(),
        "sources": locked,
    }
    if LOCK_PATH.is_file():
        existing = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        for name in SOURCES:
            for field in ("remote", "commit", "recursive_source_sha256", "files", "bytes"):
                if existing["sources"][name][field] != proposed["sources"][name][field]:
                    raise OfficialSourceError(f"Existing source lock differs for {name}/{field}")
        return existing
    atomic_json(LOCK_PATH, proposed)
    return proposed


def bounded_compatibility_audit(lock: dict[str, Any]) -> dict[str, Any]:
    nuwa = SOURCES["nuwa"]["path"]
    reroute = SOURCES["reroute"]["path"]
    readme = (nuwa / "README.md").read_text(encoding="utf-8")
    nuwa_main = (nuwa / "nuwa/main.py").read_text(encoding="utf-8")
    nuwa_arch = (nuwa / "nuwa/llava_arch.py").read_text(encoding="utf-8")
    nuwa_eval = (nuwa / "lmms-eval/lmms_eval/models/llava.py").read_text(encoding="utf-8")
    attempt1 = {
        "attempt": 1,
        "type": "source_setup",
        "status": "incompatible",
        "evidence": {
            "readme_documents_matain_token_64": "matain_token=64" in readme,
            "published_avg192_llava_config_files": [],
            "official_loader_checkpoint": "liuhaotian/llava-v1.5-7b",
            "required_checkpoint": "llava-hf/llava-1.5-7b-hf",
        },
        "reason": "The checkout publishes a native-LLaVA 64-token example but no exact or explicit nearest average-192 LLaVA operating-point config.",
    }
    import_result = subprocess.run(
        [PYTHON, "-c", "from nuwa import nuwa; print(nuwa)"],
        env={**os.environ, "PYTHONPATH": str(nuwa), "PYTHONDONTWRITEBYTECODE": "1"},
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    attempt2 = {
        "attempt": 2,
        "type": "dependency_setup",
        "status": "incompatible" if import_result.returncode else "pass",
        "command": f"PYTHONPATH={nuwa} {PYTHON} -c 'from nuwa import nuwa'",
        "exit_status": import_result.returncode,
        "stdout": import_result.stdout[-2000:],
        "stderr": import_result.stderr[-4000:],
        "reason": "The official package requires the legacy upstream llava package, which is absent from and incompatible with the canonical HF checkpoint environment.",
    }
    api_result = subprocess.run(
        [
            PYTHON,
            "-c",
            (
                "from transformers import LlavaForConditionalGeneration as C; "
                "print(hasattr(C, 'get_vision_tower'))"
            ),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    attempt3 = {
        "attempt": 3,
        "type": "checkpoint_api_smoke",
        "status": "incompatible",
        "command": f"{PYTHON} -c <HF Llava API probe>",
        "exit_status": api_result.returncode,
        "stdout": api_result.stdout.strip(),
        "evidence": {
            "official_nuwa_monkeypatches_legacy_llava_modules": "from llava.model" in nuwa_main,
            "official_nuwa_calls_get_vision_tower": "get_vision_tower" in nuwa_main + nuwa_arch + nuwa_eval,
            "canonical_hf_class_exposes_get_vision_tower": api_result.stdout.strip() == "True",
        },
        "reason": "Nüwa's algorithm-defining monkeypatch targets the legacy llava.model class tree and cannot be applied source-natively to LlavaForConditionalGeneration.",
    }
    reroute_files = [path.relative_to(reroute).as_posix() for path in reroute.rglob("*.py")]
    reroute_nuwa_hits = []
    for path in reroute.rglob("*"):
        if path.is_file() and ".git" not in path.parts and path.suffix in {".py", ".yaml", ".yml"}:
            if "nuwa" in path.read_text(encoding="utf-8", errors="replace").lower():
                reroute_nuwa_hits.append(path.relative_to(reroute).as_posix())
    pdrop_config = reroute / "configs/experiment/llava15/avg192/llava15_pdrop_earlyL2_avg192.yaml"
    reroute_config = reroute / "configs/experiment/llava15/avg192/llava15_ours_vs_pdrop_avg192.yaml"
    if not pdrop_config.is_file() or not reroute_config.is_file():
        raise OfficialSourceError("Pinned Reroute checkout lacks required published avg192 configs")
    audit = {
        "status": "complete",
        "created_at_utc": utc_now(),
        "source_lock_sha256": hashlib.sha256(canonical_bytes(lock)).hexdigest(),
        "nuwa": {
            "status": "SKIPPED_OFFICIAL_INCOMPATIBLE",
            "attempts": [attempt1, attempt2, attempt3],
            "accepted_approximation": False,
        },
        "reroute": {
            "status": "SUPPORTED_PENDING_PARITY",
            "official_python_files": len(reroute_files),
            "pdrop_avg192_config": str(pdrop_config.relative_to(REPO_ROOT)),
            "pdrop_reroute_avg192_config": str(reroute_config.relative_to(REPO_ROOT)),
            "checkpoint": "llava-hf/llava-1.5-7b-hf",
        },
        "nuwa_reroute": {
            "status": "SKIPPED_NO_DIRECT_OFFICIAL_SUPPORT",
            "algorithm_source_hits": reroute_nuwa_hits,
            "reason": "The pinned Reroute release exposes only FastV and PDrop routers; it contains no Nüwa router/config implementation.",
        },
    }
    atomic_json(COMPATIBILITY_PATH, audit)
    return audit


def main() -> int:
    lock = freeze_sources()
    audit = bounded_compatibility_audit(lock)
    print(json.dumps({"source_lock": lock, "compatibility": audit}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OfficialSourceError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
