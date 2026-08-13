#!/usr/bin/env python3
"""Publish one final, evidence-backed eligibility status per official method."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "experiments/k12_b48_expanded_evaluation_20260813"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_pass(name: str) -> dict[str, Any]:
    path = CAMPAIGN / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise RuntimeError(f"Admission evidence is not passing: {path}")
    return payload


def main() -> int:
    fastv = load_pass("fastv_admission_report.json")
    original = load_pass("original_official_admission_report.json")
    entries = [
        {
            "key": "fastv",
            "display_name": "FastV",
            "status": "matched_b48_eligible",
            "official_repo": "https://github.com/pkunlp-icler/FastV",
            "commit": "d1659729b5bf1be225e99ee15783deeea80f63b1",
            "evidence": "fastv_admission_report.json",
            "exact_trace": fastv["exact_b48"]["trace"],
        },
        {
            "key": "visionzip",
            "display_name": "VisionZip",
            "status": original["methods"]["official_visionzip"]["status"],
            "official_repo": "https://github.com/dvlab-research/VisionZip",
            "commit": "8f86b55c6f000eb033e6912538af2dd7dcb30502",
            "evidence": "original_official_admission_report.json",
            "exact_trace": original["methods"]["official_visionzip"]["trace"],
        },
        {
            "key": "feather",
            "display_name": "FEATHER",
            "status": "excluded_checkpoint_or_parity_mismatch",
            "official_repo": "https://github.com/markendo/FEATHER",
            "commit": "c2a09b2765967601054c7b1fd513ac3e94ee4fc9",
            "reason": "The official method targets the Prism/SigLIP stack and uses an adaptive union that cannot guarantee exact final B=48 on canonical LLaVA-1.5-7B; porting it would alter checkpoint and selection semantics.",
            "evidence": "literature_verification.json and pinned official source inspection",
        },
        {
            "key": "nuwa",
            "display_name": "Nüwa",
            "status": "excluded_no_exact_b48_support",
            "official_repo": "https://github.com/Man-PaperRejected/Nuwa",
            "commit": "175b7d57a7890b9818dd26d82f2d91d9868b0d80",
            "reason": "The official LLaVA configurations expose 64, 128 and 192 stage-1 tokens, not 48. Adding B=48 would require a non-official algorithm configuration.",
            "evidence": "literature_verification.json and pinned official configuration dictionaries",
        },
        {
            "key": "transprune",
            "display_name": "TransPrune",
            "status": original["methods"]["official_transprune"]["status"],
            "official_repo": "https://github.com/liaolea/TransPrune",
            "commit": "c9e8284f5808f7692c40590ffc99537e9220a01d",
            "evidence": "original_official_admission_report.json",
            "exact_trace": original["methods"]["official_transprune"]["trace"],
        },
    ]
    allowed = {
        "matched_b48_eligible", "excluded_no_exact_b48_support", "skipped_official_failure",
        "excluded_checkpoint_or_parity_mismatch", "out_of_scope_after_literature_verification",
    }
    if len({entry["key"] for entry in entries}) != 5 or any(entry["status"] not in allowed for entry in entries):
        raise RuntimeError("Eligibility table does not contain exactly one allowed status per method.")
    payload = {
        "status": "pass",
        "created_at_utc": utc_now(),
        "policy": "Only author-official, parity-verified methods guaranteeing exactly 48 final visual tokens are eligible.",
        "methods": entries,
    }
    path = CAMPAIGN / "method_eligibility.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    print(json.dumps({"status": "pass", "path": str(path), "methods": len(entries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
