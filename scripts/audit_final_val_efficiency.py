#!/usr/bin/env python
"""Independent fail-closed terminal audit for the final validation campaign."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vlm_pruning.final_statistics import select_internal_candidate  # noqa: E402


CAMPAIGN = REPO / "experiments/final_val_efficiency_baselines_20260818"
ANALYSIS = CAMPAIGN / "analysis"
PROTOCOL_PATH = CAMPAIGN / "FROZEN_PROTOCOL.json"
REGISTRY_PATH = CAMPAIGN / "EXPERIMENT_REGISTRY.json"
CONTROLLER_PATH = CAMPAIGN / "CONTROLLER_STATE.json"
REPORT_PATH = CAMPAIGN / "RESEARCH_REPORT.md"
FINAL_PATH = CAMPAIGN / "FINAL_COMPLETION_REPORT.json"
TEST_LOG = ANALYSIS / "FINAL_TEST_SUITE.log"
TEST_REPORT = ANALYSIS / "FINAL_TEST_SUITE.json"
DISSERTATION = Path("/workspace/Dissertation_Draft")
DERIVED = DISSERTATION / "analysis/derived/final_val_efficiency_baselines"
ASSET_MANIFEST = DERIVED / "ASSET_MANIFEST.json"
PDF_QA = DERIVED / "PDF_QA.json"
SOURCE_PDF = DISSERTATION / "Dissertation_Template.pdf"
ROOT_PDF = Path("/workspace/dissertation_final.pdf")
CHANGE_RECORD = CAMPAIGN / "DISSERTATION_CHANGE_RECORD.json"
TERMINAL_STATE = "GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED"
QUALITY_METHODS = {
    "Full", "K20", "K12", "M448_OPT", "Z_M448_0", "FASTV_MATCHED_199",
    "FEATHER_LLAVA_MAIN", "NUWA_STACK_FULL", "NUWA_NATIVE_192",
}
TIMING_METHODS = QUALITY_METHODS - {"K20"}
NATURAL_COUNTS = {
    "Full": 5, "K12": 5, "M448_OPT": 5, "Z_M448_0": 5,
    "FEATHER_LLAVA_MAIN": 5, "FASTV_STACK_FULL": 5, "FASTV_MATCHED_199": 5,
    "NUWA_STACK_FULL": 5, "NUWA_NATIVE_192": 5,
}
STACK_METHODS = {
    "canonical": {"Full", "K12", "M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN"},
    "fastv": {"FASTV_STACK_FULL", "FASTV_MATCHED_199"},
    "nuwa": {"NUWA_STACK_FULL", "NUWA_NATIVE_192"},
}


class FinalAuditError(RuntimeError):
    """Raised when a terminal requirement lacks exact passing evidence."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalAuditError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    require(path.is_file(), f"Missing JSON artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalAuditError(f"Invalid JSON artifact: {path}: {exc}") from exc


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"Missing JSONL artifact: {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        require(bool(line.strip()), f"Blank JSONL line: {path}:{number}")
        value = json.loads(line)
        require(isinstance(value, dict), f"Non-object JSONL line: {path}:{number}")
        rows.append(value)
    return rows


def manifest_audit(protocol: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for name in ("quality", "timing"):
        frozen = protocol["manifests"][name]
        path = REPO / frozen["path"]
        require(sha256_file(path) == frozen["sha256"], f"{name} manifest hash differs")
        rows = read_jsonl(path)
        ids = [str(row.get("sample_id") or "") for row in rows]
        require(len(rows) == int(frozen["rows"]), f"{name} manifest row count differs")
        require(len(ids) == len(set(ids)) and all(ids), f"{name} manifest IDs differ")
        require(ids[0] == frozen["first_id"] and ids[-1] == frozen["last_id"], f"{name} manifest order differs")
        require(all(str(row.get("dataset", "")).lower() == "refcoco" for row in rows), f"{name} manifest dataset differs")
        require(all(str(row.get("split", "")).lower() == "val" for row in rows), f"{name} manifest split differs")
        require(all((REPO / str(row["image_path"])).is_file() for row in rows), f"{name} manifest image is missing")
        output[name] = {"status": "pass", "path": frozen["path"], "sha256": frozen["sha256"], "rows": len(rows), "unique_ids": len(set(ids))}
    return output


def protected_audit(protocol: dict[str, Any]) -> dict[str, Any]:
    evidence = {}
    for relative, expected in protocol["protected_inputs"].items():
        path = REPO / relative
        require(path.is_file(), f"Protected input is missing: {relative}")
        actual = sha256_file(path)
        require(actual == expected, f"Protected input changed: {relative}")
        evidence[relative] = actual
    return {"status": "pass", "sha256": evidence}


def git_source_inventory(path: Path) -> tuple[str, int]:
    inventory = subprocess.check_output(["git", "-C", str(path), "ls-files", "-s", "-z"])
    return hashlib.sha256(inventory).hexdigest(), inventory.count(b"\0")


def provenance_audit(protocol: dict[str, Any]) -> dict[str, Any]:
    sources = {}
    for name, frozen in protocol["official_sources"].items():
        path = Path(frozen["path"])
        require(path.is_dir(), f"Official source is missing: {name}")
        commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(path), "status", "--short"], text=True).strip()
        digest, files = git_source_inventory(path)
        require(commit == frozen["commit"] and not status, f"Official source revision/cleanliness differs: {name}")
        require(digest == frozen["recursive_sha256"] and files == int(frozen["files"]), f"Official source inventory differs: {name}")
        sources[name] = {"commit": commit, "recursive_sha256": digest, "files": files, "clean": True}
    parity = read_json(ANALYSIS / "PARITY_AUDIT.json")
    require(parity.get("status") == "PASS_WITH_REGISTERED_STACK_SEPARATION", "Parity audit is not terminal")
    return {"status": "pass", "official_sources": sources, "parity_sha256": sha256_file(ANALYSIS / "PARITY_AUDIT.json")}


def historical_audit(protocol: dict[str, Any]) -> dict[str, Any]:
    evidence = {}
    for method, frozen in protocol["historical_evidence"].items():
        root = REPO / frozen["path"]
        files = {
            "config.json": frozen["config_sha256"],
            "metrics.json": frozen["metrics_sha256"],
            str((REPO / frozen["predictions"]).relative_to(root)): frozen["predictions_sha256"],
        }
        for relative, expected in files.items():
            require(sha256_file(root / relative) == expected, f"Historical evidence changed: {method}/{relative}")
        evidence[method] = files
    return {"status": "pass", "methods": evidence, "rerun": False}


def quality_audit() -> dict[str, Any]:
    metrics = read_json(ANALYSIS / "QUALITY_METHOD_METRICS.json")
    require({row["method"] for row in metrics} == QUALITY_METHODS, "Quality method inventory differs")
    inventory = read_json(ANALYSIS / "QUALITY_RUN_INVENTORY.json")
    require(inventory.get("status") == "complete", "Quality run inventory is incomplete")
    require(set(inventory.get("methods", {})) == QUALITY_METHODS, "Quality run hashes are incomplete")
    for method, record in inventory["methods"].items():
        path = REPO / record["path"]
        require(int(record["rows"]) == 8_811, f"Quality row count differs: {method}")
        require(record.get("independent_rescore") == "pass" and record.get("trace_validation") == "pass", f"Quality validation failed: {method}")
        require(sha256_file(path) == record["sha256"], f"Quality artifact hash differs: {method}")
        rows = read_jsonl(path)
        require(len(rows) == 8_811, f"Quality physical rows differ: {method}")
    new_methods = QUALITY_METHODS - {"Full", "K20", "K12", "M448_OPT", "Z_M448_0"}
    require(new_methods == {"FASTV_MATCHED_199", "FEATHER_LLAVA_MAIN", "NUWA_STACK_FULL", "NUWA_NATIVE_192"}, "New quality cell set differs")
    contrasts = read_json(ANALYSIS / "QUALITY_PAIRED_CONTRASTS.json")
    require(len(contrasts) >= 18, "Paired quality contrast inventory is incomplete")
    return {"status": "pass", "methods": len(metrics), "rows_per_method": 8_811, "new_cells": sorted(new_methods), "inventory_sha256": sha256_file(ANALYSIS / "QUALITY_RUN_INVENTORY.json"), "contrasts": len(contrasts)}


def expected_fixed_counts(trigger: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for stack, methods in STACK_METHODS.items():
        if trigger.get("stacks", {}).get(stack, {}).get("required") is True:
            counts.update({method: 3 for method in methods})
    return counts


def timing_audit() -> dict[str, Any]:
    natural = read_json(ANALYSIS / "TIMING_NATURAL_INVENTORY.json")
    require(natural.get("status") == "complete", "Natural timing inventory is incomplete")
    runs = natural.get("runs", [])
    counts = Counter(str(row["method"]) for row in runs)
    require(dict(counts) == NATURAL_COUNTS, f"Natural timing repetition counts differ: {dict(counts)}")
    for row in runs:
        root = REPO / row["run_dir"]
        filename = "predictions.jsonl" if row["stack"] == "nuwa" else "timing.jsonl"
        require(sha256_file(root / filename) == row["timing_sha256"], f"Timing hash differs: {root}")
        require(sha256_file(root / "config.json") == row["config_sha256"], f"Timing config hash differs: {root}")
        require(int(row["rows"]) == 1_000 and row["validation_status"] == "pass", f"Timing validation differs: {root}")
    require(set(natural.get("ledgers", {})) == set(STACK_METHODS), "Natural timing ledgers are incomplete")
    for item in natural["ledgers"].values():
        require(sha256_file(REPO / item["path"]) == item["sha256"], "Natural timing ledger hash differs")
    trigger = read_json(ANALYSIS / "FIXED32_TRIGGER.json")
    require(trigger.get("status") == "complete", "Fixed32 trigger is incomplete")
    fixed_counts = expected_fixed_counts(trigger)
    if fixed_counts:
        fixed = read_json(ANALYSIS / "TIMING_FIXED32_INVENTORY.json")
        require(fixed.get("status") == "complete", "Triggered fixed32 timing is incomplete")
        observed = Counter(str(row["method"]) for row in fixed.get("runs", []))
        require(dict(observed) == fixed_counts, f"Fixed32 repetition counts differ: {dict(observed)}")
        for row in fixed["runs"]:
            root = REPO / row["run_dir"]
            filename = "predictions.jsonl" if row["stack"] == "nuwa" else "timing.jsonl"
            require(sha256_file(root / filename) == row["timing_sha256"], f"Fixed32 timing hash differs: {root}")
    else:
        require(not (ANALYSIS / "TIMING_FIXED32_INVENTORY.json").exists(), "Untriggered fixed32 analysis exists")
    return {"status": "pass", "natural_runs": len(runs), "natural_counts": dict(counts), "fixed32_counts": fixed_counts, "trigger_sha256": sha256_file(ANALYSIS / "FIXED32_TRIGGER.json")}


def analysis_audit(protocol: dict[str, Any]) -> dict[str, Any]:
    accounting = read_json(ANALYSIS / "FLOP_KV_ACCOUNTING.json")
    methods = accounting.get("methods", [])
    require({row["method"] for row in methods} == TIMING_METHODS | {"FASTV_STACK_FULL"}, "FLOP/KV method inventory differs")
    require("ranking" in accounting["scope"]["selector_scope"].lower(), "FLOP scope does not separate ranking")
    detail = read_json(ANALYSIS / "FLOP_KV_PER_SAMPLE_AUDIT.json")
    require(set(detail) == TIMING_METHODS | {"FASTV_STACK_FULL"}, "Per-sample accounting inventory differs")
    require(all(len(rows) == 1_000 for rows in detail.values()), "Per-sample accounting rows differ")
    pareto = read_json(ANALYSIS / "PARETO_FRONTIERS.json")
    required_axes = {
        f"{quality}_vs_{cost}"
        for quality in ("acc_at_0_5", "mean_iou")
        for cost in (
            "visual_token_layer_work", "modeled_total_prefill_flops", "natural_end_to_end_s",
            "prefill_including_overhead_s", "time_to_first_token_s", "prompt_kv_bytes", "final_kv_bytes",
        )
    }
    require(required_axes <= set(pareto.get("axes", {})), "Pareto axes are incomplete")
    decision = read_json(ANALYSIS / "FROZEN_FINAL_CANDIDATE.json")
    quality = {row["method"]: row for row in read_json(ANALYSIS / "QUALITY_METHOD_METRICS.json")}
    timing_rows = {(row["stack"], row["method"]): row for row in read_json(ANALYSIS / "TIMING_NATURAL_SUMMARY.json")}
    contrasts = read_json(ANALYSIS / "TIMING_NATURAL_CONTRASTS.json")
    upper = {
        row["left"]: float(row["upper_95"])
        for row in contrasts
        if row["stack"] == "canonical" and row["right"] == "Full" and row["metric"] == "end_to_end_latency_s"
    }
    q = {
        method: {
            "parse_rate": float(quality[method]["parse_rate"]),
            "acc_at_0_5": float(quality[method]["acc_at_0_5"]),
            "mean_iou": float(quality[method]["mean_iou"]),
        }
        for method in ("Full", "K12", "M448_OPT", "Z_M448_0")
    }
    t = {"Full": {"end_to_end_latency_s": float(timing_rows[("canonical", "Full")]["end_to_end_latency_s"]["mean"])}}
    for method in ("K12", "M448_OPT", "Z_M448_0"):
        t[method] = {
            "end_to_end_latency_s": float(timing_rows[("canonical", method)]["end_to_end_latency_s"]["mean"]),
            "delta_vs_full_upper_95_s": upper[method],
        }
    recomputed = select_internal_candidate(q, t, {"K12": 7872, "M448_OPT": 6848, "Z_M448_0": 6272})
    for key in ("status", "selected_method", "method_decisions"):
        require(decision.get(key) == recomputed.get(key), f"Frozen candidate decision differs: {key}")
    require(decision.get("validation_only") is True and decision.get("testA_testB_authorized") is False, "Candidate scope authorization differs")
    return {"status": "pass", "accounting_methods": len(methods), "per_sample_accounting_rows": 1_000, "pareto_axes": len(pareto["axes"]), "candidate_status": decision["status"], "selected_method": decision["selected_method"], "decision_sha256": sha256_file(ANALYSIS / "FROZEN_FINAL_CANDIDATE.json")}


def verify_asset_group(root: Path, files: dict[str, str], label: str) -> int:
    require(bool(files), f"{label} asset group is empty")
    for name, expected in files.items():
        path = root / name
        require(path.is_file() and sha256_file(path) == expected, f"{label} asset differs: {name}")
    return len(files)


def dissertation_audit() -> dict[str, Any]:
    manifest = read_json(ASSET_MANIFEST)
    require(manifest.get("status") == "complete" and manifest.get("testA_testB_included") is False, "Dissertation asset manifest differs")
    generated = verify_asset_group(DISSERTATION / "chapters/generated_final_val_efficiency", manifest["generated_tex"], "generated TeX")
    figures = verify_asset_group(DISSERTATION / "images/final_evaluation/final_val_efficiency", manifest["figures"], "figure")
    derived = verify_asset_group(DERIVED, manifest["derived_json"], "derived JSON")
    change = read_json(CHANGE_RECORD)
    require(change.get("status") == "complete", "Dissertation change record is incomplete")
    require(Path(change["backup_directory"]).is_dir(), "Dissertation backup directory is missing")
    for name, item in change["baseline_files"].items():
        backup = Path(item["backup_path"])
        require(backup.is_file() and sha256_file(backup) == item["backup_sha256"], f"Dissertation backup differs: {name}")
    for name, item in change["final_files"].items():
        path = Path(item["path"])
        require(path.is_file() and sha256_file(path) == item["sha256_after"], f"Final dissertation file differs: {name}")
    qa = read_json(PDF_QA)
    require(qa.get("status") == "pass", "Final PDF QA did not pass")
    visual = qa.get("visual_inspection", {})
    require(visual.get("status") == "pass", "Final PDF visual inspection did not pass")
    require(sorted(visual.get("inspected_pages", [])) == sorted(visual.get("pages_to_render", [])), "Not every changed PDF page was inspected")
    require(SOURCE_PDF.is_file() and ROOT_PDF.is_file(), "Final PDF is missing")
    require(sha256_file(SOURCE_PDF) == sha256_file(ROOT_PDF), "User-facing PDF differs from validated PDF")
    main_tex = (DISSERTATION / "Dissertation_Template.tex").read_text(encoding="utf-8")
    active = (DISSERTATION / "chapters/final_evaluation.tex").read_text(encoding="utf-8")
    forbidden_inputs = ("generated_final_heldout", "generated_final_pope", "generated_final_textvqa", "generated_final_timing")
    require(not any(name in main_tex or name in active for name in forbidden_inputs), "Obsolete frozen held-out assets remain active")
    pdf_text = subprocess.check_output(["pdftotext", str(ROOT_PDF), "-"], text=True)
    required_text = (
        "Final RefCOCO-validation efficiency",
        "validation-only",
        "Z-M448-0",
        "FEATHER",
        "Nüwa",
        "testA/testB",
    )
    missing = [text for text in required_text if text not in pdf_text]
    require(not missing, f"Final PDF omits required content: {missing}")
    return {"status": "pass", "generated_tex": generated, "figures": figures, "derived_json": derived, "asset_manifest_sha256": sha256_file(ASSET_MANIFEST), "pdf_qa_sha256": sha256_file(PDF_QA), "source_pdf_sha256": sha256_file(SOURCE_PDF), "user_pdf_sha256": sha256_file(ROOT_PDF), "backup_directory": change["backup_directory"]}


def iter_campaign_configs() -> Iterable[Path]:
    for path in CAMPAIGN.rglob("config.json"):
        relative = path.relative_to(CAMPAIGN)
        if relative.parts and relative.parts[0] in {"envs", "official_sources"}:
            continue
        yield path


def scope_and_runtime_audit(protocol: dict[str, Any]) -> dict[str, Any]:
    forbidden = ("testa", "testb", "refcoco+", "refcocog", "transfer", "learned_router", "training")
    configs = []
    for path in iter_campaign_configs():
        raw = path.read_text(encoding="utf-8").lower()
        require(not any(term in raw for term in forbidden), f"Forbidden scope appears in campaign config: {path}")
        value = json.loads(raw)
        split = str(value.get("split") or value.get("dataset/split") or "val").lower()
        require(split in {"", "val"}, f"Non-validation split appears in campaign config: {path}")
        configs.append(str(path.relative_to(CAMPAIGN)))
    path_violations = [
        str(path.relative_to(CAMPAIGN))
        for path in CAMPAIGN.rglob("*")
        if not ({"envs", "official_sources"} & set(path.relative_to(CAMPAIGN).parts))
        and any(term in path.name.lower() for term in forbidden)
    ]
    require(not path_violations, f"Forbidden campaign paths exist: {path_violations[:5]}")
    process_lines = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines()
    runner_names = (
        "run_official_baseline.py", "run_progressive_oracle_refcoco.py",
        "run_native_nuwa_refcoco.py", "run_final_val_canonical_timing.py",
        "run_final_val_fastv_timing.py", "run_final_val_timing_campaign.py",
    )
    active = [line.strip() for line in process_lines if any(name in line for name in runner_names) and "ps -eo" not in line]
    require(not active, f"Campaign worker remains active: {active}")
    gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"], text=True).strip()
    require(not gpu, f"GPU compute process remains active: {gpu}")
    campaign_bytes = sum(path.stat().st_size for path in CAMPAIGN.rglob("*") if path.is_file())
    require(campaign_bytes <= int(protocol["storage"]["campaign_cap_bytes"]), "Campaign artifact cap exceeded")
    free = shutil.disk_usage("/workspace").free
    require(free >= int(protocol["storage"]["minimum_runtime_bytes"]), "Final free storage is below the registered reserve")
    return {"status": "pass", "configs_checked": len(configs), "forbidden_paths": [], "active_workers": [], "gpu_compute": [], "campaign_bytes": campaign_bytes, "workspace_free_bytes": free}


def code_and_git_audit() -> dict[str, Any]:
    compile_targets = sorted(str(path.relative_to(REPO)) for path in (REPO / "scripts").glob("*final_val*.py"))
    compile_process = subprocess.run([sys.executable, "-m", "py_compile", *compile_targets], cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    require(compile_process.returncode == 0, f"Final source compile failed: {compile_process.stdout[-2000:]}")
    test_process = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    match = re.search(r"Ran (\d+) tests", test_process.stdout)
    tests_run = int(match.group(1)) if match else -1
    require(test_process.returncode == 0 and tests_run >= 300, f"Final unit suite failed or was incomplete: {test_process.stdout[-4000:]}")
    atomic_write(TEST_LOG, test_process.stdout.encode())
    diff = subprocess.run(["git", "diff", "--check"], cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    require(diff.returncode == 0, f"git diff --check failed: {diff.stdout}")
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO, text=True).splitlines()
    allowed = {"?? PRO_REASONING_RESEARCH_HANDOFF.md", "?? legacy_goal2.md"}
    require(set(status) <= allowed, f"Unexpected Git status at terminal audit: {status}")
    result = {"status": "pass", "tests": tests_run, "test_log": str(TEST_LOG.relative_to(REPO)), "test_log_sha256": sha256_file(TEST_LOG), "compile_targets": compile_targets, "git_diff_check": "pass", "git_status": status, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}
    atomic_json(TEST_REPORT, result)
    result["test_report_sha256"] = sha256_file(TEST_REPORT)
    return result


def finalize_state(requirements: dict[str, Any]) -> dict[str, Any]:
    completed = utc_now()
    report = REPORT_PATH.read_text(encoding="utf-8")
    require("RefCOCO testA/testB and transfer evaluation were not run" in report, "Campaign report omits validation-only scope")
    report += (
        "\n## Independent terminal audit\n\n"
        f"Campaign terminal state: `{TERMINAL_STATE}`. The independent audit passed all registered "
        "manifest, source, quality, timing, accounting, decision, dissertation, PDF, scope, test and "
        "quiescence checks. Human review is now required; no held-out or transfer work was launched.\n"
    )
    atomic_write(REPORT_PATH, report.encode())
    registry = read_json(REGISTRY_PATH)
    require(registry.get("phase_status", {}).get("dissertation") == "pass", "Dissertation phase is not complete")
    registry["phase_status"]["final_audit"] = "pass"
    registry["status"] = TERMINAL_STATE
    registry["terminal_state"] = TERMINAL_STATE
    registry["final_completion_report"] = str(FINAL_PATH.relative_to(REPO))
    registry["updated_at_utc"] = completed
    atomic_json(REGISTRY_PATH, registry)
    controller = read_json(CONTROLLER_PATH)
    controller.update({"phase": "complete", "status": TERMINAL_STATE, "active_workers": [], "next_action": "human_review_required", "updated_at_utc": completed})
    atomic_json(CONTROLLER_PATH, controller)
    payload = {
        "schema_version": 1,
        "status": TERMINAL_STATE,
        "terminal_state": TERMINAL_STATE,
        "completed_at_utc": completed,
        "campaign": "final_val_efficiency_baselines_20260818",
        "requirements": requirements,
        "issues": [],
        "next_action": "human_review_only",
        "testA_testB_or_transfer_executed": False,
        "artifacts": {
            "registry_sha256": sha256_file(REGISTRY_PATH),
            "controller_sha256": sha256_file(CONTROLLER_PATH),
            "report_sha256": sha256_file(REPORT_PATH),
            "candidate_sha256": sha256_file(ANALYSIS / "FROZEN_FINAL_CANDIDATE.json"),
            "claim_evidence_ledger_sha256": sha256_file(ANALYSIS / "CLAIM_EVIDENCE_LEDGER.json"),
            "dissertation_pdf_sha256": sha256_file(ROOT_PDF),
        },
    }
    atomic_json(FINAL_PATH, payload)
    return payload


def main() -> int:
    require(not FINAL_PATH.exists(), "Final completion report already exists; refusing to overwrite it")
    protocol = read_json(PROTOCOL_PATH)
    registry = read_json(REGISTRY_PATH)
    require(registry.get("phase_status", {}).get("analysis") == "pass", "Analysis phase is not complete")
    require(registry.get("phase_status", {}).get("dissertation") == "pass", "Dissertation phase is not complete")
    requirements = {
        "manifests": manifest_audit(protocol),
        "protected_inputs": protected_audit(protocol),
        "official_sources_and_parity": provenance_audit(protocol),
        "historical_evidence_reused_not_rerun": historical_audit(protocol),
        "quality": quality_audit(),
        "timing": timing_audit(),
        "flops_kv_pareto_and_candidate": analysis_audit(protocol),
        "dissertation_and_pdf": dissertation_audit(),
        "forbidden_scope_storage_and_quiescence": scope_and_runtime_audit(protocol),
        "source_tests_and_git": code_and_git_audit(),
    }
    payload = finalize_state(requirements)
    print(json.dumps({"status": payload["status"], "output": str(FINAL_PATH.relative_to(REPO))}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FinalAuditError, OSError, KeyError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
