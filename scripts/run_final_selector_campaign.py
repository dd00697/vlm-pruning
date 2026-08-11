#!/usr/bin/env python3
"""Register, execute, and strictly validate the final selector/repair campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "experiments/final_selector_ablation_and_repair_20260811"
FINAL_MANIFESTS = {
    "val": ROOT / "data/manifests/final_evaluation/refcoco_val_unique.jsonl",
    "testA": ROOT / "data/manifests/final_evaluation/refcoco_testA_unique.jsonl",
    "testB": ROOT / "data/manifests/final_evaluation/refcoco_testB_unique.jsonl",
}
OFFICIAL_VAL = ROOT / "data/manifests/official_baselines/refcoco_val.jsonl"
EXPECTED = {"val": 8811, "testA": 5657, "testB": 5095}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_pruning.artifacts import read_jsonl_strict, sha256_file, validate_completed_run  # noqa: E402
from vlm_pruning.selectors import UGPruneConfig  # noqa: E402
from vlm_pruning.selectors.ug_prune import budget_split  # noqa: E402


@dataclass(frozen=True)
class Cell:
    run_id: str
    family: str
    split: str
    manifest: Path
    expected_rows: int
    runner: str = "mid_layer"
    budget: int = 48
    layer: int = 20
    selection_method: str = "ug"
    strategy: str = "structured_tca"
    map_source: str = "query_max"
    fusion: str = "single_layer"
    probes: str | None = None
    role: str = "more_anchors"
    position: str = "gap_preserving"

    def registry_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["manifest"] = str(self.manifest.relative_to(ROOT))
        row["manifest_sha256"] = sha256_file(self.manifest)
        if self.family == "incomplete_repair":
            row["historical_source_manifest"] = str(OFFICIAL_VAL.relative_to(ROOT))
            row["historical_source_manifest_sha256"] = sha256_file(OFFICIAL_VAL)
            row["identity_policy"] = "same historical row order with final split/source-row-qualified unique IDs"
        row["config_hash"] = hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return row


def cells() -> list[Cell]:
    result: list[Cell] = []
    for layer in (20, 12):
        for strategy in ("query_topk", "target_nms_only", "structured_tca"):
            result.append(Cell(
                f"mech_val_k{layer}_b48_{strategy}", "mechanism", "val", FINAL_MANIFESTS["val"], EXPECTED["val"],
                layer=layer, strategy=strategy, probes=str(layer),
            ))
    result.extend([
        Cell("repair_val_prellm_random_b48_seed0", "incomplete_repair", "val", FINAL_MANIFESTS["val"], EXPECTED["val"], runner="official", layer=0, selection_method="random", strategy="not_applicable", role="not_applicable", position="compact"),
        Cell("repair_val_k12_b72_fused_more_context", "incomplete_repair", "val", FINAL_MANIFESTS["val"], EXPECTED["val"], budget=72, layer=12, fusion="soft_average", probes="4,8,12", role="more_context"),
        Cell("repair_val_k12_b128_pooled_more_anchors", "incomplete_repair", "val", FINAL_MANIFESTS["val"], EXPECTED["val"], budget=128, layer=12, map_source="query_pooled", fusion="soft_average", probes="4,8,12"),
        Cell("repair_val_k12_b48_single_layer_more_context", "incomplete_repair", "val", FINAL_MANIFESTS["val"], EXPECTED["val"], layer=12, probes="12", role="more_context"),
        Cell("repair_val_k16_b48_single_layer_more_context", "incomplete_repair", "val", FINAL_MANIFESTS["val"], EXPECTED["val"], layer=16, probes="16", role="more_context"),
        Cell("clean_val_k20_b48_structured_compact", "provenance_repair", "val", FINAL_MANIFESTS["val"], EXPECTED["val"], layer=20, probes="20", position="compact"),
    ])
    for split in ("testA", "testB"):
        for method in ("random", "uniform"):
            result.append(Cell(
                f"clean_{split}_k20_b48_{method}_seed0", "provenance_repair", split, FINAL_MANIFESTS[split], EXPECTED[split],
                layer=20, probes="20", selection_method=method,
            ))
        result.append(Cell(
            f"clean_{split}_k12_b48_structured_seed0", "provenance_repair", split, FINAL_MANIFESTS[split], EXPECTED[split],
            layer=12, probes="12",
        ))
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_manifest(cell: Cell, limit: int | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in cell.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows if limit is None else rows[:limit]


def output_root(phase: str) -> Path:
    return CAMPAIGN / ("runs" if phase == "full" else f"smoke/{phase}")


def command(cell: Cell, *, phase: str, run_id: str | None = None, limit: int | None = None) -> list[str]:
    name = run_id or cell.run_id
    common = [
        "--dataset", "refcoco", "--task", "grounding", "--manifest", str(cell.manifest.relative_to(ROOT)),
        "--split", cell.split, "--target-visual-tokens", str(cell.budget), "--seed", "0", "--precision", "bf16",
        "--max-new-tokens", "48", "--max-figures", "0", "--run-name", name,
        "--output-root", str(output_root(phase).relative_to(ROOT)),
        "--figures-root", str((CAMPAIGN / "figures").relative_to(ROOT)),
        "--tables-root", str((CAMPAIGN / "tables").relative_to(ROOT)), "--resume", "--skip-existing",
    ]
    if limit is not None:
        common.extend(["--limit", str(limit)])
    provenance = ["--no-wandb"] if phase != "full" else [
        "--wandb", "--wandb-mode", "offline", "--wandb-group", "final-selector-repair-20260811",
        "--wandb-job-type", cell.family, "--wandb-tags", f"final-repair,{cell.family},{cell.split}",
    ]
    if cell.runner == "official":
        return [sys.executable, "scripts/run_official_baseline.py", *common, "--method", cell.selection_method,
                "--position-mode", cell.position, "--prompt-profile", "registered",
                "--selector-id-mode", "original_sample_id",
                "--reports-root", str((CAMPAIGN / "reports").relative_to(ROOT)), *provenance]
    return [
        sys.executable, "scripts/run_ug_prune_single_pass.py", *common,
        "--prune-after-layer", str(cell.layer), "--position-mode", cell.position,
        "--selection-method", cell.selection_method, "--selector-strategy", cell.strategy,
        "--ug-map-source", cell.map_source, "--ug-fusion-mode", cell.fusion,
        "--ug-probe-layers", cell.probes or str(cell.layer), "--ug-role-preset", cell.role,
        "--decode-mode", "layer_aware_cache", "--diagnostics-mode", "benchmark", *provenance,
    ]


def run_dir(cell: Cell, phase: str, run_id: str | None = None) -> Path:
    return output_root(phase) / (run_id or cell.run_id)


def integrity(cell: Cell, phase: str, *, run_id: str | None = None, limit: int | None = None) -> dict[str, Any]:
    report = validate_completed_run(
        run_dir(cell, phase, run_id), read_manifest(cell, limit), budget=cell.budget, original_tokens=576,
        task="grounding", strategy=cell.strategy if cell.selection_method == "ug" else None,
    )
    if report.get("status") != "pass":
        return report
    directory = run_dir(cell, phase, run_id)
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        environment = json.loads((directory / "run_environment.json").read_text(encoding="utf-8"))
        expected = {
            "model/name": "llava-hf/llava-1.5-7b-hf", "model/precision": "bf16",
            "model/decoding": "deterministic_greedy", "model/max_new_tokens": 48,
            "dataset/name": "refcoco", "dataset/split": cell.split,
            "dataset/manifest_checksum": sha256_file(cell.manifest), "dataset/num_samples": len(read_manifest(cell, limit)),
            "task": "grounding", "tokens/original_visual_tokens": 576,
            "tokens/target_visual_tokens": cell.budget, "tokens/actual_visual_tokens": cell.budget,
            "position_mode": cell.position, "limit": limit,
        }
        if cell.runner == "official":
            expected.update({"method/name": cell.selection_method, "prompt/profile": "registered", "selector/id_mode": "original_sample_id"})
        else:
            expected.update({
                "selection/method": cell.selection_method,
                "selection/strategy": cell.strategy if cell.selection_method == "ug" else "not_applicable",
                "ug/prune_after_layer": cell.layer, "ug/map_source": cell.map_source,
                "ug/fusion_mode": cell.fusion, "ug/probe_layers": cell.probes or str(cell.layer),
                "ug/role_preset": cell.role, "ug/position_mode": cell.position,
                "ug/decode_mode": "layer_aware_cache", "ug/diagnostics_mode": "benchmark",
            })
        mismatches = {key: {"expected": value, "actual": config.get(key)} for key, value in expected.items() if config.get(key) != value}
        if config.get("source/git_dirty") is not False or environment.get("git_dirty") is not False:
            mismatches["source/clean"] = {"expected": True, "actual": {"config_dirty": config.get("source/git_dirty"), "environment_dirty": environment.get("git_dirty")}}
        if not config.get("source/sha256"):
            mismatches["source/sha256"] = {"expected": "non-empty", "actual": config.get("source/sha256")}
        role_failures = []
        if cell.selection_method == "ug":
            requested_target, requested_context, requested_anchor = budget_split(
                cell.budget,
                UGPruneConfig(
                    selector_strategy=cell.strategy,
                    role_mode="target_context_anchor",
                    target_ratio=float(config["ug/target_ratio"]), context_ratio=float(config["ug/context_ratio"]),
                    anchor_ratio=float(config["ug/anchor_ratio"]), position_mode=cell.position,
                ),
            )
            expected_roles = {
                "query_topk": cell.budget if cell.strategy == "query_topk" else 0,
                "target": requested_target, "context": requested_context, "anchor": requested_anchor,
            }
            for prediction in read_jsonl_strict(directory / "per_sample_results.jsonl"):
                requested = prediction.get("ug_budget_split") or {}
                realised = prediction.get("ug_realised_role_counts") or {}
                for role, count in expected_roles.items():
                    if int(requested.get(role, 0)) != count:
                        role_failures.append(f"{prediction.get('sample_id')}: requested {role}={requested.get(role)} expected {count}")
                if sum(int(value) for value in realised.values()) != cell.budget:
                    role_failures.append(f"{prediction.get('sample_id')}: realised roles do not sum to {cell.budget}")
                if int(prediction.get("ug_nms_fallback_count", -1)) < 0:
                    role_failures.append(f"{prediction.get('sample_id')}: invalid NMS fallback count")
                if len(role_failures) >= 20:
                    break
        if role_failures:
            mismatches["selector/roles"] = {"expected": "exact requested and realised roles", "actual": role_failures}
        report["config_contract"] = {"status": "pass" if not mismatches else "fail", "mismatches": mismatches}
        if mismatches:
            report["status"] = "fail"
            report["issues"].append(f"configuration contract mismatch: {mismatches}")
    except Exception as exc:
        report["status"] = "fail"
        report["issues"].append(f"configuration validation failed: {exc}")
    return report


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_event(value: dict[str, Any]) -> None:
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    with (CAMPAIGN / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": utc_now(), **value}, sort_keys=True) + "\n")


def register() -> list[Cell]:
    registered = cells()
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    historical = [json.loads(line) for line in OFFICIAL_VAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    qualified = [json.loads(line) for line in FINAL_MANIFESTS["val"].read_text(encoding="utf-8").splitlines() if line.strip()]
    identity_fields = ("query", "gt_box", "image_path", "image_id", "width", "height")
    mismatches = [index for index, (left, right) in enumerate(zip(historical, qualified)) if any(left.get(key) != right.get(key) for key in identity_fields)]
    unique_ids = len({str(row["sample_id"]) for row in qualified}) == len(qualified)
    equivalence = {
        "status": "pass" if len(historical) == len(qualified) == EXPECTED["val"] and not mismatches and unique_ids else "fail",
        "historical_manifest": str(OFFICIAL_VAL.relative_to(ROOT)), "historical_sha256": sha256_file(OFFICIAL_VAL),
        "qualified_manifest": str(FINAL_MANIFESTS["val"].relative_to(ROOT)), "qualified_sha256": sha256_file(FINAL_MANIFESTS["val"]),
        "rows": len(qualified), "scientific_field_order_mismatches": mismatches[:20], "qualified_ids_unique": unique_ids,
    }
    write_json(CAMPAIGN / "repair_manifest_identity_equivalence.json", equivalence)
    if equivalence["status"] != "pass":
        raise RuntimeError("Historical-to-qualified repair manifest equivalence failed")
    payload = {
        "campaign": "final_selector_ablation_and_repair_20260811", "created_at_utc": utc_now(),
        "model": "llava-hf/llava-1.5-7b-hf", "precision": "bf16", "seed": 0,
        "original_visual_tokens": 576, "full_runs": [cell.registry_dict() for cell in registered],
    }
    write_json(CAMPAIGN / "campaign_registry.json", payload)
    with (CAMPAIGN / "commands.jsonl").open("w", encoding="utf-8") as handle:
        for cell in registered:
            handle.write(json.dumps({"run_id": cell.run_id, "command": command(cell, phase="full"), "shell": shlex.join(command(cell, phase="full"))}, sort_keys=True) + "\n")
    return registered


def semantic_projection(row: dict[str, Any]) -> dict[str, Any]:
    keys = ("sample_id", "retained_indices", "raw_model_output", "raw_parsed_box", "predicted_box", "box_iou", "success_iou_0_5")
    return {key: row.get(key) for key in keys}


def compare_runs(left: Path, right: Path) -> dict[str, Any]:
    def load(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (path / "per_sample_results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    a, b = load(left), load(right)
    mismatches = [index for index, (x, y) in enumerate(zip(a, b)) if semantic_projection(x) != semantic_projection(y)]
    return {"left": str(left), "right": str(right), "rows_left": len(a), "rows_right": len(b), "semantic_mismatch_count": len(mismatches), "first_mismatches": mismatches[:20], "status": "pass" if len(a) == len(b) and not mismatches else "fail"}


ARCHIVED_REPAIRS = {
    "repair_val_prellm_random_b48_seed0": ROOT / "experiments/runs_official_baselines/official_fullmatrix_refcoco_random_tokens48_seed0_bf16",
    "repair_val_k12_b72_fused_more_context": ROOT / "experiments/runs_ug_prune_single_pass/ablation_fullmatrix_refcoco_ug_k12_b72_more_context_bf16",
    "repair_val_k12_b128_pooled_more_anchors": ROOT / "experiments/runs_ug_prune_single_pass/strong_ablation_refcoco_query_pooled_more_anchors_k12_b128_bf16",
    "repair_val_k12_b48_single_layer_more_context": ROOT / "experiments/runs_ug_prune_single_pass/strong_ablation_refcoco_depth_l12_b48_single_layer_bf16",
    "repair_val_k16_b48_single_layer_more_context": ROOT / "experiments/runs_ug_prune_single_pass/strong_ablation_refcoco_depth_l16_b48_single_layer_bf16",
}


def compare_archived_overlap(cell: Cell) -> dict[str, Any]:
    fresh = read_jsonl_strict(run_dir(cell, "serial", f"serial_{cell.run_id}") / "per_sample_results.jsonl")
    archived_path = ARCHIVED_REPAIRS[cell.run_id]
    archived = read_jsonl_strict(archived_path / "per_sample_results.jsonl")
    archived_by_source = {int(row.get("source_row")): row for row in archived if row.get("source_row") is not None}
    archived_by_identity = {(str(row.get("sample_id")), str(row.get("query")), str(row.get("image_path"))): row for row in archived}
    fields = ("retained_indices", "raw_model_output", "raw_parsed_box", "predicted_box", "box_iou", "success_iou_0_5")
    mismatches=[]
    for row in fresh:
        old=archived_by_source.get(int(row.get("source_row")))
        if old is None:
            old=archived_by_identity.get((str(row.get("original_sample_id")),str(row.get("query")),str(row.get("image_path"))))
        if old is None:
            mismatches.append({"source_row":row.get("source_row"),"reason":"missing archived row"});continue
        id_equal=str(row.get("original_sample_id"))==str(old.get("sample_id"))
        changed=[field for field in fields if (field != "retained_indices" or old.get(field) is not None) and row.get(field)!=old.get(field)]
        if not id_equal or changed:
            mismatches.append({"source_row":row.get("source_row"),"sample_id_equal":id_equal,"changed_fields":changed})
    return {"run_id":cell.run_id,"archived_run":str(archived_path),"fresh_overlap_rows":len(fresh),"compared_rows":len(fresh),"mismatch_count":len(mismatches),"first_mismatches":mismatches[:20],"status":"match" if not mismatches else "drift_documented"}


def execute_jobs(jobs: list[tuple[Cell, str, str, int | None]], *, parallel: int, max_attempts: int, poll: float) -> None:
    state_path = CAMPAIGN / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"runs": {}}
    pending = list(jobs)
    active: dict[str, tuple[Cell, str, str, int | None, subprocess.Popen[Any], Any]] = {}
    while pending or active:
        while pending and len(active) < parallel:
            cell, phase, name, limit = pending.pop(0)
            key = f"{phase}:{name}"
            if integrity(cell, phase, run_id=name, limit=limit).get("status") == "pass":
                state["runs"].setdefault(key, {})["status"] = "complete"
                continue
            row = state["runs"].setdefault(key, {"attempts": 0})
            row["attempts"] += 1
            row.update({"status": "running", "started_at": utc_now(), "run_id": cell.run_id})
            log = CAMPAIGN / "logs" / f"{phase}_{name}.attempt{row['attempts']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a", encoding="utf-8")
            env = dict(os.environ)
            env.update({"HF_HOME": "/workspace/cache/huggingface", "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_XET": "1"})
            cmd = command(cell, phase=phase, run_id=name, limit=limit)
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            active[key] = (cell, phase, name, limit, process, handle)
            append_event({"event": "start", "key": key, "attempt": row["attempts"], "command": cmd})
            write_json(state_path, state)
        if not active:
            break
        time.sleep(poll)
        try:
            free_mib = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0])
            current = state.setdefault("gpu", {})
            current["minimum_free_mib"] = min(int(current.get("minimum_free_mib", free_mib)), free_mib)
        except Exception:
            pass
        for key, item in list(active.items()):
            cell, phase, name, limit, process, handle = item
            code = process.poll()
            if code is None:
                continue
            handle.close()
            report = integrity(cell, phase, run_id=name, limit=limit)
            row = state["runs"][key]
            row.update({"returncode": code, "finished_at": utc_now(), "integrity": report})
            if code == 0 and report.get("status") == "pass":
                row["status"] = "complete"
                append_event({"event": "complete", "key": key})
            elif row["attempts"] < max_attempts:
                row["status"] = "retry_pending"
                pending.append((cell, phase, name, limit))
                append_event({"event": "retry", "key": key, "returncode": code, "issues": report.get("issues")})
            else:
                row["status"] = "failed"
                append_event({"event": "failed", "key": key, "returncode": code, "issues": report.get("issues")})
            del active[key]
            write_json(state_path, state)
    failures = [key for key, row in state["runs"].items() if row.get("status") == "failed" and any(key == f"{phase}:{name}" for _, phase, name, _ in jobs)]
    if failures:
        raise SystemExit("Terminal campaign failures: " + ", ".join(failures))


def serial_jobs(registered: list[Cell]) -> list[tuple[Cell, str, str, int | None]]:
    mechanism = [cell for cell in registered if cell.family == "mechanism"]
    repairs = [cell for cell in registered if cell.family == "incomplete_repair"]
    jobs = [(cell, "serial", f"serial_{cell.run_id}", 20) for cell in [*mechanism, *repairs]]
    for cell in mechanism:
        if cell.strategy == "structured_tca":
            jobs.append((cell, "serial", f"serial_repeat_{cell.run_id}", 20))
    return jobs


def evaluate_serial(registered: list[Cell]) -> dict[str, Any]:
    comparisons = []
    for cell in registered:
        if cell.family == "mechanism" and cell.strategy == "structured_tca":
            comparisons.append(compare_runs(run_dir(cell, "serial", f"serial_{cell.run_id}"), run_dir(cell, "serial", f"serial_repeat_{cell.run_id}")))
    overlaps=[compare_archived_overlap(cell) for cell in registered if cell.family=="incomplete_repair"]
    report = {"status": "pass" if all(row["status"] == "pass" for row in comparisons) else "fail", "determinism_comparisons": comparisons, "historical_overlap_comparisons":overlaps}
    write_json(CAMPAIGN / "serial_smoke_report.json", report)
    return report


def soak_jobs(registered: list[Cell], label: str = "soak3") -> list[tuple[Cell, str, str, int | None]]:
    ids = {cell.run_id: cell for cell in registered}
    chosen = [ids["repair_val_k12_b128_pooled_more_anchors"], ids["mech_val_k20_b48_structured_tca"], ids["mech_val_k12_b48_query_topk"]]
    return [(cell, "soak", f"{label}_{cell.run_id}", 20) for cell in chosen]


def evaluate_soak(registered: list[Cell], jobs: list[tuple[Cell, str, str, int | None]], parallel: int) -> dict[str, Any]:
    comparisons = []
    memory = []
    for cell, _, name, _ in jobs:
        comparisons.append(compare_runs(run_dir(cell, "serial", f"serial_{cell.run_id}"), run_dir(cell, "soak", name)))
        metrics = json.loads((run_dir(cell, "soak", name) / "metrics.json").read_text(encoding="utf-8"))
        memory.append({"run_id": name, "peak_allocated_mb": metrics.get("efficiency/peak_gpu_memory_mb"), "peak_reserved_mb": metrics.get("efficiency/peak_gpu_memory_reserved_mb")})
    state = json.loads((CAMPAIGN / "state.json").read_text(encoding="utf-8"))
    minimum = int(state.get("gpu", {}).get("minimum_free_mib", 0))
    status = "pass" if minimum >= 3072 and all(row["status"] == "pass" for row in comparisons) else "fail"
    report = {"status": status, "attempted_parallelism": parallel, "minimum_free_mib": minimum, "required_margin_mib": 3072, "per_process_memory": memory, "semantic_comparisons": comparisons, "chosen_max_parallel": parallel if status == "pass" else None}
    write_json(CAMPAIGN / "gpu_soak_report.json", report)
    return report


def git_clean() -> tuple[str, bool]:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    clean = not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    return revision, clean


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["register", "serial", "soak", "full"], default="register")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()
    registered = register()
    if not args.execute or args.phase == "register":
        print(f"Registered {len(registered)} full runs at {CAMPAIGN}")
        return 0
    if args.phase == "serial":
        execute_jobs(serial_jobs(registered), parallel=1, max_attempts=args.max_attempts, poll=args.poll_seconds)
        report = evaluate_serial(registered)
        if report["status"] != "pass":
            raise SystemExit("Serial semantic smoke failed")
    elif args.phase == "soak":
        serial_report = json.loads((CAMPAIGN / "serial_smoke_report.json").read_text(encoding="utf-8"))
        if serial_report.get("status") != "pass":
            raise SystemExit("Passing serial_smoke_report.json required before soak")
        attempts=[]; report=None
        for parallel in (3, 2, 1):
            state_path=CAMPAIGN/"state.json"
            state=json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"runs":{}}
            state["gpu"]={}
            write_json(state_path,state)
            jobs=soak_jobs(registered, f"soak{parallel}")
            failed=False
            try:
                execute_jobs(jobs, parallel=parallel, max_attempts=args.max_attempts, poll=args.poll_seconds)
            except SystemExit as exc:
                failed=True
                attempts.append({"parallelism":parallel,"status":"failed_jobs","detail":str(exc)})
            if not failed:
                report=evaluate_soak(registered,jobs,parallel)
                attempts.append({"parallelism":parallel,"status":report["status"],"minimum_free_mib":report["minimum_free_mib"]})
                if report["status"]=="pass": break
        if report is None or report["status"] != "pass":
            raise SystemExit("GPU soak failed at parallelism 3, 2, and 1")
        report["attempts"]=attempts
        write_json(CAMPAIGN / "gpu_soak_report.json", report)
    else:
        revision, clean = git_clean()
        if not clean:
            raise SystemExit("Refusing full inference from a dirty Git worktree")
        soak = json.loads((CAMPAIGN / "gpu_soak_report.json").read_text(encoding="utf-8"))
        parallel = min(args.max_parallel, int(soak.get("chosen_max_parallel", 1)))
        write_json(CAMPAIGN / "experiment_epoch.json", {"git_commit": revision, "git_clean_at_launch": clean, "registered_at_utc": utc_now(), "max_parallel": parallel})
        jobs = [(cell, "full", cell.run_id, None) for cell in registered]
        execute_jobs(jobs, parallel=parallel, max_attempts=args.max_attempts, poll=args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
