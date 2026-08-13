#!/usr/bin/env python3
"""Build the K=12/B=48 claim-to-artifact audit and reuse ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "experiments/k12_b48_expanded_evaluation_20260813"
PRIOR_AUDIT = ROOT / "experiments/final_selector_ablation_and_repair_20260811/prelaunch_artefact_inventory.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_pruning.artifacts import read_jsonl_strict, sha256_file, validate_completed_run  # noqa: E402


REFCOCO_MANIFESTS = {
    "val": ROOT / "data/manifests/final_evaluation/refcoco_val_unique.jsonl",
    "testA": ROOT / "data/manifests/final_evaluation/refcoco_testA_unique.jsonl",
    "testB": ROOT / "data/manifests/final_evaluation/refcoco_testB_unique.jsonl",
}


REUSE_CANDIDATES = (
    {
        "claim": "RefCOCO val pure query top-k K=12/B=48",
        "split": "val", "strategy": "query_topk",
        "path": "experiments/final_selector_ablation_and_repair_20260811/runs/mech_val_k12_b48_query_topk",
    },
    {
        "claim": "RefCOCO val query top-k plus NMS K=12/B=48",
        "split": "val", "strategy": "target_nms_only",
        "path": "experiments/final_selector_ablation_and_repair_20260811/runs/mech_val_k12_b48_target_nms_only",
    },
    {
        "claim": "RefCOCO val structured Q-TAC K=12/B=48",
        "split": "val", "strategy": "structured_tca",
        "path": "experiments/final_selector_ablation_and_repair_20260811/runs/mech_val_k12_b48_structured_tca",
    },
    {
        "claim": "RefCOCO testA primary Q-TAC K=12/B=48",
        "split": "testA", "strategy": "structured_tca",
        "path": "experiments/final_selector_ablation_and_repair_20260811/runs/clean_testA_k12_b48_structured_seed0",
    },
    {
        "claim": "RefCOCO testB primary Q-TAC K=12/B=48",
        "split": "testB", "strategy": "structured_tca",
        "path": "experiments/final_selector_ablation_and_repair_20260811/runs/clean_testB_k12_b48_structured_seed0",
    },
    {
        "claim": "RefCOCO testA unpruned reference",
        "split": "testA", "strategy": None,
        "path": "experiments/final_evaluation/runs/official/final_refcoco_testA_full_b576_seed0",
        "budget": 576, "full": True,
    },
    {
        "claim": "RefCOCO testB unpruned reference",
        "split": "testB", "strategy": None,
        "path": "experiments/final_evaluation/runs/official/final_refcoco_testB_full_b576_seed0",
        "budget": 576, "full": True,
    },
)


REQUIRED_NEW_OR_REPAIRED = (
    ("RefCOCO val unpruned reference", "rerun_required_corrupt", "Historical val full run uses 395 duplicate sample IDs; rerun on the unique locked manifest."),
    ("RefCOCO val random K=12/B=48", "rerun_required_incomplete", "No complete exact-primary K=12 mid-layer random val run exists."),
    ("RefCOCO val uniform K=12/B=48", "rerun_required_incomplete", "No complete exact-primary K=12 mid-layer uniform val run exists."),
    ("RefCOCO val compact-position primary control", "rerun_required_wrong_config", "Existing K=12 compact control uses the superseded more-context allocation, not 21/12/15."),
    ("Exact-primary K=12/B=48 serial timing", "rerun_required_wrong_config", "Existing timing uses a different anchor allocation and cannot be attributed to the locked primary."),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fingerprint(config: dict[str, Any]) -> str:
    keys = (
        "model/name", "model/precision", "dataset/name", "dataset/split", "dataset/manifest_checksum",
        "method/name", "selection/method", "selection/strategy", "tokens/target_visual_tokens",
        "ug/prune_after_layer", "ug/map_source", "ug/fusion_mode", "ug/probe_layers",
        "ug/role_preset", "ug/position_mode", "ug/decode_mode", "method/seed",
    )
    projection = {key: config.get(key) for key in keys}
    return hashlib.sha256(json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def finite_quality(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        for key in ("score", "box_iou", "latency_s", "actual_llm_visual_tokens"):
            value = row.get(key)
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                return False
    return True


def audit_candidate(spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = ROOT / spec["path"]
    manifest_path = REFCOCO_MANIFESTS[spec["split"]]
    manifest = read_jsonl_strict(manifest_path)
    budget = int(spec.get("budget", 48))
    report = validate_completed_run(
        run_dir, manifest, budget=budget, original_tokens=576, task="grounding",
        strategy=spec.get("strategy"), require_retained=True,
    )
    issues = list(report.get("issues", []))
    config: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        rows = read_jsonl_strict(run_dir / "per_sample_results.jsonl")
        expected = {
            "model/name": "llava-hf/llava-1.5-7b-hf",
            "model/precision": "bf16", "dataset/name": "refcoco", "dataset/split": spec["split"],
            "dataset/manifest_checksum": sha256_file(manifest_path), "method/seed": 0,
            "tokens/target_visual_tokens": budget,
        }
        if not spec.get("full"):
            expected.update({
                "ug/prune_after_layer": 12, "ug/map_source": "query_max", "ug/fusion_mode": "single_layer",
                "ug/probe_layers": "12", "ug/role_preset": "more_anchors", "ug/position_mode": "gap_preserving",
                "ug/decode_mode": "layer_aware_cache", "selection/strategy": spec["strategy"],
            })
        mismatches = {}
        for key, value in expected.items():
            actual = config.get(key)
            if actual == value:
                continue
            # The completed August held-out full references recorded the first
            # 16 hex characters only. Exact manifest identity is nevertheless
            # independently proved above by full ordered ID/row validation,
            # and by reopening the configured immutable manifest path.
            legacy_full_checksum = (
                spec.get("full") and key == "dataset/manifest_checksum"
                and isinstance(actual, str) and actual == str(value)[:16]
                and Path(str(config.get("dataset/manifest"))).name == manifest_path.name
            )
            if not legacy_full_checksum:
                mismatches[key] = {"expected": value, "actual": actual}
        if mismatches:
            issues.append(f"configuration mismatch: {mismatches}")
        if not finite_quality(rows):
            issues.append("one or more required numeric values are non-finite")
    except (OSError, ValueError) as exc:
        issues.append(str(exc))
    status = "verified_reusable" if report.get("status") == "pass" and not issues else "rerun_required_corrupt"
    return {
        "claim": spec["claim"], "artifact": str(run_dir.relative_to(ROOT)), "status": status,
        "issues": issues, "validation": report, "config_fingerprint": fingerprint(config) if config else None,
        "config_sha256": sha256_file(run_dir / "config.json") if (run_dir / "config.json").is_file() else None,
        "predictions_sha256": sha256_file(run_dir / "per_sample_results.jsonl") if (run_dir / "per_sample_results.jsonl").is_file() else None,
        "manifest": str(manifest_path.relative_to(ROOT)), "manifest_sha256": sha256_file(manifest_path),
    }


def historical_rows() -> list[dict[str, Any]]:
    if not PRIOR_AUDIT.is_file():
        raise RuntimeError(f"Run scripts/audit_dissertation_artifacts.py first: {PRIOR_AUDIT}")
    payload = json.loads(PRIOR_AUDIT.read_text(encoding="utf-8"))
    output = []
    for item in payload.get("entries", []):
        path = str(item.get("run_path") or "")
        lower = path.lower()
        if "k20" in lower or "b72" in lower or "b128" in lower or "b192" in lower:
            status = "superseded_not_reported"
            reason = "Outside the K=12/B=48-only reporting scope."
        elif "latency" in lower:
            status = "rerun_required_wrong_config"
            reason = "Historical timing is not the locked 21/12/15 exact-primary timing protocol."
        elif item.get("status") == "replacement_required" and not item.get("replacement_run_id"):
            status = "rerun_required_incomplete"
            reason = "; ".join(item.get("issues", []))
        else:
            status = "development_only_not_claimed"
            reason = "Historical development evidence retained for provenance but excluded from current claims."
        output.append({
            "claim": item.get("claims"), "artifact": path, "status": status, "reason": reason,
            "prior_audit_status": item.get("status"), "prior_audit_issues": item.get("issues", []),
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CAMPAIGN / "artifact_ledger.json")
    args = parser.parse_args()
    rows = historical_rows()
    rows.extend(audit_candidate(spec) for spec in REUSE_CANDIDATES)
    rows.extend({"claim": claim, "artifact": None, "status": status, "reason": reason} for claim, status, reason in REQUIRED_NEW_OR_REPAIRED)
    allowed = {
        "verified_reusable", "rerun_required_incomplete", "rerun_required_corrupt",
        "rerun_required_wrong_config", "superseded_not_reported", "development_only_not_claimed",
        "blocked_external_scoring",
    }
    invalid = [row for row in rows if row["status"] not in allowed]
    payload = {
        "campaign_id": "k12_b48_expanded_evaluation_20260813", "audited_at_utc": utc_now(),
        "status": "pass" if not invalid else "fail", "allowed_statuses": sorted(allowed),
        "source_prior_audit": str(PRIOR_AUDIT.relative_to(ROOT)),
        "source_prior_audit_sha256": sha256_file(PRIOR_AUDIT), "rows": rows,
        "counts": {status: sum(row["status"] == status for row in rows) for status in sorted(allowed)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": payload["status"], "rows": len(rows), "counts": payload["counts"], "output": str(args.output)}, indent=2))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
