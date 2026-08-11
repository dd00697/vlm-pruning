#!/usr/bin/env python3
"""Validate and analyse the final selector/repair campaign from raw JSONLs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DISS = Path("/workspace/Dissertation_Draft")
DERIVED = DISS / "analysis/derived/final_selector_ablation_and_repair"
FIGURES = DISS / "images/final_selector_ablation_and_repair"
CAMPAIGN = ROOT / "experiments/final_selector_ablation_and_repair_20260811"
SEED = 20260811
BOOTSTRAPS = 10_000

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_final_selector_campaign import cells, integrity, run_dir  # noqa: E402
from vlm_pruning.artifacts import read_jsonl_strict, sha256_file  # noqa: E402
from vlm_pruning.geometry.grid import patch_bounds, patch_to_rc  # noqa: E402
from vlm_pruning.selectors.ug_prune import selected_spread_stats  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in values for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def load(cell_id: str) -> list[dict[str, Any]]:
    cell = next(item for item in cells() if item.run_id == cell_id)
    report = integrity(cell, "full")
    if report["status"] != "pass":
        raise RuntimeError(f"{cell_id} failed integrity: {report['issues']}")
    return read_jsonl_strict(run_dir(cell, "full") / "per_sample_results.jsonl")


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    requested: defaultdict[str, list[int]] = defaultdict(list)
    realised: defaultdict[str, list[int]] = defaultdict(list)
    for row in rows:
        for role, value in (row.get("ug_budget_split") or {}).items():
            requested[role].append(int(value))
        for role, value in (row.get("ug_realised_role_counts") or {}).items():
            realised[role].append(int(value))
    return {
        "samples": count,
        "acc_at_0_5": sum(bool(row.get("success_iou_0_5")) for row in rows) / count,
        "mean_iou": sum(float(row.get("box_iou") or 0.0) for row in rows) / count,
        "parsed_box_rate": sum(row.get("raw_parsed_box") is not None for row in rows) / count,
        "invalid_output_rate": sum(bool(row.get("invalid")) for row in rows) / count,
        "exact_budget_rate": sum(float(row.get("selection/exact_budget") or 0.0) for row in rows) / count,
        **{f"requested_{role}_mean": statistics.fmean(values) for role, values in requested.items()},
        **{f"realised_{role}_mean": statistics.fmean(values) for role, values in realised.items()},
    }


def bootstrap(delta: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAPS, dtype=np.float64)
    for start in range(0, BOOTSTRAPS, 200):
        size = min(200, BOOTSTRAPS - start)
        indices = rng.integers(0, len(delta), size=(size, len(delta)))
        estimates[start : start + size] = delta[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(delta.mean()), float(low), float(high)


def paired_rows(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> None:
    if [row["sample_id"] for row in left] != [row["sample_id"] for row in right]:
        raise RuntimeError("Paired rows are not identically manifest aligned")


def mechanism_quality() -> tuple[dict[tuple[int, str], list[dict[str, Any]]], list[dict[str, Any]]]:
    loaded: dict[tuple[int, str], list[dict[str, Any]]] = {}
    table = []
    for layer in (12, 20):
        for strategy in ("query_topk", "target_nms_only", "structured_tca"):
            rows = load(f"mech_val_k{layer}_b48_{strategy}")
            loaded[(layer, strategy)] = rows
            table.append({"boundary": layer, "strategy": strategy, **summary(rows)})
    write_csv(DERIVED / "mechanism_quality.csv", table)
    write_json(DERIVED / "mechanism_quality.json", table)
    return loaded, table


def paired_intervals(loaded: dict[tuple[int, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    contrasts = (("structured_tca", "query_topk"), ("target_nms_only", "query_topk"), ("structured_tca", "target_nms_only"))
    output = []
    for layer in (12, 20):
        for number, (left_name, right_name) in enumerate(contrasts):
            left, right = loaded[(layer, left_name)], loaded[(layer, right_name)]
            paired_rows(left, right)
            for field, metric, scale in (("success_iou_0_5", "acc_at_0_5_pp", 100.0), ("box_iou", "mean_iou", 1.0)):
                delta = np.asarray([float(a.get(field) or 0.0) - float(b.get(field) or 0.0) for a, b in zip(left, right)]) * scale
                mean, low, high = bootstrap(delta, SEED + layer * 10 + number)
                output.append({"boundary": layer, "contrast": f"{left_name}-minus-{right_name}", "metric": metric, "mean_delta": mean, "lower_95": low, "upper_95": high, "bootstrap_replicates": BOOTSTRAPS, "seed": SEED + layer * 10 + number})
    write_csv(DERIVED / "mechanism_paired_intervals.csv", output)
    write_json(DERIVED / "mechanism_paired_intervals.json", output)
    return output


def patch_overlap(indices: list[int], gt_box: list[float], width: int, height: int) -> tuple[float, float]:
    inside = intersect = 0
    for index in indices:
        x0, y0, x1, y1 = patch_bounds(index, width, height, 24)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        inside += int(gt_box[0] <= cx <= gt_box[2] and gt_box[1] <= cy <= gt_box[3])
        intersect += int(max(x0, gt_box[0]) < min(x1, gt_box[2]) and max(y0, gt_box[1]) < min(y1, gt_box[3]))
    return inside / len(indices), intersect / len(indices)


def subgroup_labels(manifest: dict[str, Any]) -> list[str]:
    text = str(manifest["query"]).lower()
    words = text.split()
    labels = ["all", "length_short" if len(words) <= 3 else "length_medium" if len(words) <= 6 else "length_long"]
    if any(word in text for word in ("left", "right", "top", "bottom", "above", "below", "behind", "front", "next to", "between")):
        labels.append("spatial_or_relation_keyword")
    box, width, height = manifest["gt_box"], int(manifest["width"]), int(manifest["height"])
    cx, cy = (box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height)
    labels.append(f"centre_{min(2, int(cy * 3))}_{min(2, int(cx * 3))}")
    labels.append("border_near" if min(box[0] / width, box[1] / height, (width - box[2]) / width, (height - box[3]) / height) <= .1 else "border_far")
    return labels


def diagnostics(loaded: dict[tuple[int, str], list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = read_jsonl_strict(ROOT / "data/manifests/final_evaluation/refcoco_val_unique.jsonl")
    by_id = {str(row["sample_id"]): row for row in manifest}
    strategy_rows = []
    subgroup_acc: defaultdict[tuple[int, str, str], list[tuple[float, float]]] = defaultdict(list)
    for (layer, strategy), rows in loaded.items():
        occupied = []; row_span = []; col_span = []; spread = []; centre = []; intersect = []
        role_centre: defaultdict[str, list[float]] = defaultdict(list)
        fallback = []
        for row in rows:
            item = by_id[str(row["sample_id"])]
            retained = [int(value) for value in row["retained_indices"]]
            stats = selected_spread_stats(retained, bins=(3, 3), grid=24)
            occupied.append(stats["selected_occupied_bins_3x3"]); row_span.append(stats["selected_row_span"]); col_span.append(stats["selected_col_span"]); spread.append(stats["selected_mean_pairwise_spread"])
            inside, overlap = patch_overlap(retained, item["gt_box"], int(item["width"]), int(item["height"]))
            centre.append(inside); intersect.append(overlap); fallback.append(int(row.get("ug_nms_fallback_count") or 0))
            for role, values in (row.get("ug_role_labels") or {}).items():
                if values:
                    value, _ = patch_overlap(values, item["gt_box"], int(item["width"]), int(item["height"]))
                    role_centre[role].append(value)
            for label in subgroup_labels(item):
                subgroup_acc[(layer, strategy, label)].append((float(bool(row.get("success_iou_0_5"))), float(row.get("box_iou") or 0.0)))
        strategy_rows.append({
            "boundary": layer, "strategy": strategy, "samples": len(rows),
            "occupied_bins_3x3_mean": statistics.fmean(occupied), "row_span_mean": statistics.fmean(row_span),
            "column_span_mean": statistics.fmean(col_span), "normalised_pairwise_spread_mean": statistics.fmean(spread),
            "selected_centres_inside_gt_rate": statistics.fmean(centre), "selected_patches_intersect_gt_rate": statistics.fmean(intersect),
            "nms_fallback_examples": sum(value > 0 for value in fallback), "nms_fallback_tokens": sum(fallback),
            **{f"{role}_centres_inside_gt_rate": statistics.fmean(values) for role, values in role_centre.items()},
        })
    comparisons = []
    for layer in (12, 20):
        for left_name, right_name in (("target_nms_only", "query_topk"), ("structured_tca", "query_topk"), ("structured_tca", "target_nms_only")):
            left, right = loaded[(layer, left_name)], loaded[(layer, right_name)]; paired_rows(left, right)
            intersections=[]; jaccards=[]; retained_changed=prediction_changed=left_only=right_only=0
            for a,b in zip(left,right):
                aset,bset=set(a["retained_indices"]),set(b["retained_indices"]); inter=len(aset&bset)
                intersections.append(inter);jaccards.append(inter/len(aset|bset));retained_changed += aset!=bset
                prediction_changed += a.get("raw_model_output") != b.get("raw_model_output")
                left_only += bool(a.get("success_iou_0_5")) and not bool(b.get("success_iou_0_5"))
                right_only += bool(b.get("success_iou_0_5")) and not bool(a.get("success_iou_0_5"))
            comparisons.append({"boundary":layer,"comparison":f"{left_name}-vs-{right_name}","samples":len(left),"intersection_mean":statistics.fmean(intersections),"jaccard_mean":statistics.fmean(jaccards),"retained_set_changed_count":retained_changed,"retained_set_changed_rate":retained_changed/len(left),"prediction_changed_count":prediction_changed,"prediction_changed_rate":prediction_changed/len(left),"left_only_correct":left_only,"right_only_correct":right_only})
    subgroups=[]
    for (layer,strategy,label),values in sorted(subgroup_acc.items()):
        subgroups.append({"boundary":layer,"strategy":strategy,"subgroup":label,"samples":len(values),"acc_at_0_5":statistics.fmean(value[0] for value in values),"mean_iou":statistics.fmean(value[1] for value in values)})
    write_csv(DERIVED / "mechanism_selection_summary.csv", strategy_rows)
    write_csv(DERIVED / "mechanism_pairwise_selection.csv", comparisons)
    write_csv(DERIVED / "mechanism_subgroups.csv", subgroups)
    write_json(DERIVED / "mechanism_diagnostics.json", {"strategy_summary":strategy_rows,"pairwise":comparisons,"subgroups":subgroups})
    return strategy_rows, comparisons


OLD_REPLACED = {
    "repair_val_prellm_random_b48_seed0": ROOT / "experiments/runs_official_baselines/official_fullmatrix_refcoco_random_tokens48_seed0_bf16",
    "repair_val_k12_b72_fused_more_context": ROOT / "experiments/runs_ug_prune_single_pass/ablation_fullmatrix_refcoco_ug_k12_b72_more_context_bf16",
    "repair_val_k12_b128_pooled_more_anchors": ROOT / "experiments/runs_ug_prune_single_pass/strong_ablation_refcoco_query_pooled_more_anchors_k12_b128_bf16",
    "repair_val_k12_b48_single_layer_more_context": ROOT / "experiments/runs_ug_prune_single_pass/strong_ablation_refcoco_depth_l12_b48_single_layer_bf16",
    "repair_val_k16_b48_single_layer_more_context": ROOT / "experiments/runs_ug_prune_single_pass/strong_ablation_refcoco_depth_l16_b48_single_layer_bf16",
}


OLD_PROVENANCE = {
    "clean_val_k20_b48_structured_compact": ROOT / "experiments/followup_evaluation/runs/refcoco/compact_ablation/followup_refcoco_val_ug_k20_b48_compact_seed0",
    "clean_testA_k20_b48_random_seed0": ROOT / "experiments/followup_evaluation/runs/refcoco/controls_k20_b48/followup_refcoco_testA_random_k20_b48_seed0",
    "clean_testA_k20_b48_uniform_seed0": ROOT / "experiments/followup_evaluation/runs/refcoco/controls_k20_b48/followup_refcoco_testA_uniform_k20_b48_seed0",
    "clean_testB_k20_b48_random_seed0": ROOT / "experiments/followup_evaluation/runs/refcoco/controls_k20_b48/followup_refcoco_testB_random_k20_b48_seed0",
    "clean_testB_k20_b48_uniform_seed0": ROOT / "experiments/followup_evaluation/runs/refcoco/controls_k20_b48/followup_refcoco_testB_uniform_k20_b48_seed0",
    "clean_testA_k12_b48_structured_seed0": ROOT / "experiments/followup_evaluation/runs/refcoco/k12_depth_only/followup_refcoco_testA_ug_k12_b48_depth_only_seed0",
    "clean_testB_k12_b48_structured_seed0": ROOT / "experiments/followup_evaluation/runs/refcoco/k12_depth_only/followup_refcoco_testB_ug_k12_b48_depth_only_seed0",
}


def repair_and_provenance() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repair_rows=[]; ledger=[]
    by_cell={cell.run_id:cell for cell in cells()}
    for run_id, old in {**OLD_REPLACED, **OLD_PROVENANCE}.items():
        cell=by_cell[run_id]; new=run_dir(cell,"full"); rows=load(run_id); old_rows=read_jsonl_strict(old/"per_sample_results.jsonl")
        old_metrics=json.loads((old/"metrics.json").read_text())
        repair_rows.append({"run_id":run_id,"family":cell.family,**summary(rows),"archived_physical_rows":len(old_rows),"archived_stored_acc_at_0_5":old_metrics.get("quality/acc_at_0_5"),"archived_stored_mean_iou":old_metrics.get("quality/mean_iou")})
        ledger.append({"old_run":str(old),"old_rows":len(old_rows),"old_predictions_sha256":sha256_file(old/"per_sample_results.jsonl"),"replacement_run_id":run_id,"replacement_run":str(new),"replacement_rows":len(rows),"replacement_predictions_sha256":sha256_file(new/"per_sample_results.jsonl"),"reason":"incomplete physical predictions" if run_id in OLD_REPLACED else "dirty parent revision", "old_authority":"superseded","replacement_authority":"verified_current"})
    duplicate=ROOT/"experiments/runs_ug_prune_single_pass/ablation_fullmatrix_refcoco_ug_k12_b48_more_context_single_layer_bf16"
    duplicate_rows=read_jsonl_strict(duplicate/"per_sample_results.jsonl")
    replacement=run_dir(by_cell["repair_val_k12_b48_single_layer_more_context"],"full")
    ledger.append({"old_run":str(duplicate),"old_rows":len(duplicate_rows),"old_predictions_sha256":sha256_file(duplicate/"per_sample_results.jsonl"),"replacement_run_id":"repair_val_k12_b48_single_layer_more_context","replacement_run":str(replacement),"replacement_rows":8811,"replacement_predictions_sha256":sha256_file(replacement/"per_sample_results.jsonl"),"reason":"duplicate scientific cell; no separate GPU rerun","old_authority":"superseded_duplicate","replacement_authority":"verified_current"})
    gap=load("mech_val_k20_b48_structured_tca"); compact=load("clean_val_k20_b48_structured_compact"); paired_rows(gap,compact)
    mismatches=sum(a["retained_indices"]!=b["retained_indices"] for a,b in zip(gap,compact))
    if mismatches: raise RuntimeError(f"K20 compact/gap retained indices differ on {mismatches} rows")
    write_csv(DERIVED/"repair_and_provenance_results.csv",repair_rows)
    write_json(DERIVED/"position_selection_identity.json",{"samples":len(gap),"retained_set_mismatches":mismatches,"status":"pass"})
    write_json(DERIVED/"artefact_supersession_ledger.json",ledger)
    return repair_rows,ledger


def matched_followup_analysis() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    table=[]; intervals=[]
    for split in ("testA","testB"):
        final_frozen=ROOT/f"experiments/final_evaluation/runs/frozen/final_refcoco_{split}_frozen_k20_b48/per_sample_results.jsonl"
        k20=read_jsonl_strict(final_frozen)
        k12=load(f"clean_{split}_k12_b48_structured_seed0")
        methods={"k20_structured":k20,"k12_structured":k12}
        for method in ("random","uniform"):
            methods[f"k20_{method}"]=load(f"clean_{split}_k20_b48_{method}_seed0")
            control=ROOT/f"experiments/followup_evaluation/runs/refcoco/controls_k12_b48/followup_refcoco_{split}_{method}_k12_b48_seed0/per_sample_results.jsonl"
            methods[f"k12_{method}"]=read_jsonl_strict(control)
        for name,rows in methods.items(): table.append({"split":split,"method":name,**summary(rows)})
        for layer in (12,20):
            structured=methods[f"k{layer}_structured"]
            for comparator in ("random","uniform"):
                controls=methods[f"k{layer}_{comparator}"];paired_rows(structured,controls)
                for field,metric,scale in (("success_iou_0_5","acc_at_0_5_pp",100.0),("box_iou","mean_iou",1.0)):
                    delta=np.asarray([float(a.get(field) or 0)-float(b.get(field) or 0) for a,b in zip(structured,controls)])*scale
                    mean,low,high=bootstrap(delta,SEED+layer+(0 if comparator=="random" else 100))
                    intervals.append({"split":split,"boundary":layer,"contrast":f"structured-minus-{comparator}","metric":metric,"mean_delta":mean,"lower_95":low,"upper_95":high,"bootstrap_replicates":BOOTSTRAPS})
    gap=load("mech_val_k20_b48_structured_tca");compact=load("clean_val_k20_b48_structured_compact");paired_rows(gap,compact)
    for name,rows in (("gap_preserving",gap),("compact",compact)):table.append({"split":"val","method":f"k20_structured_{name}",**summary(rows)})
    for field,metric,scale in (("success_iou_0_5","acc_at_0_5_pp",100.0),("box_iou","mean_iou",1.0)):
        delta=np.asarray([float(a.get(field) or 0)-float(b.get(field) or 0) for a,b in zip(compact,gap)])*scale
        mean,low,high=bootstrap(delta,SEED+200);intervals.append({"split":"val","boundary":20,"contrast":"compact-minus-gap_preserving","metric":metric,"mean_delta":mean,"lower_95":low,"upper_95":high,"bootstrap_replicates":BOOTSTRAPS})
    write_csv(DERIVED/"clean_matched_controls_and_position.csv",table);write_csv(DERIVED/"clean_matched_paired_intervals.csv",intervals)
    return table,intervals


def figure(table: list[dict[str, Any]], intervals: list[dict[str, Any]]) -> None:
    FIGURES.mkdir(parents=True,exist_ok=True)
    labels=("query_topk","target_nms_only","structured_tca"); x=np.arange(3); width=.36
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    for offset,layer in ((-.5,12),(.5,20)):
        values=[next(row for row in table if row["boundary"]==layer and row["strategy"]==name)["acc_at_0_5"]*100 for name in labels]
        ax.bar(x+offset*width,values,width,label=f"K={layer}")
    ax.set_xticks(x, ["Query top-k","Target NMS only","Structured T/C/A"]);ax.set_ylabel("RefCOCO val Acc@0.5 (%)");ax.legend();ax.grid(axis="y",alpha=.25)
    fig.tight_layout();fig.savefig(FIGURES/"selector_mechanism_ladder.pdf");fig.savefig(FIGURES/"selector_mechanism_ladder.png",dpi=220);plt.close(fig)


def latex_tables(table: list[dict[str, Any]], intervals: list[dict[str, Any]]) -> None:
    lines=[r"\begin{tabular}{llrrrr}",r"\toprule",r"Boundary & Selector & $n$ & Acc@0.5 & mIoU & Parsed \\",r"\midrule"]
    for row in table: lines.append(f"K={row['boundary']} & {row['strategy'].replace('_',' ')} & {row['samples']} & {100*row['acc_at_0_5']:.2f} & {row['mean_iou']:.4f} & {100*row['parsed_box_rate']:.2f} \\")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    (DERIVED/"generated_mechanism_table.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    loaded,table=mechanism_quality();intervals=paired_intervals(loaded);strategy_rows,comparisons=diagnostics(loaded);repairs,ledger=repair_and_provenance();matched,matched_intervals=matched_followup_analysis();figure(table,intervals);latex_tables(table,intervals)
    claims={"retrospective_selector_ladder":{"artifacts":["mechanism_quality.csv","mechanism_paired_intervals.csv","mechanism_diagnostics.json"],"runs":[cell.run_id for cell in cells() if cell.family=="mechanism"]},"repaired_development":{"artifacts":["repair_and_provenance_results.csv","artefact_supersession_ledger.json"],"runs":[cell.run_id for cell in cells() if cell.family=="incomplete_repair"]},"clean_provenance":{"artifacts":["position_selection_identity.json","repair_and_provenance_results.csv"],"runs":[cell.run_id for cell in cells() if cell.family=="provenance_repair"]}}
    write_json(DERIVED/"claim_to_artefact_ledger.json",claims)
    outputs=[path for path in [*DERIVED.glob("*"),*FIGURES.glob("selector_mechanism_ladder.*")] if path.is_file()]
    write_json(DERIVED/"analysis_manifest.json",{"status":"complete","bootstrap_replicates":BOOTSTRAPS,"seed":SEED,"files":{str(path.relative_to(DISS)):sha256_file(path) for path in outputs}})
    print(json.dumps({"status":"complete","mechanism_rows":len(table),"interval_rows":len(intervals),"repair_rows":len(repairs),"ledger_rows":len(ledger),"matched_rows":len(matched),"matched_interval_rows":len(matched_intervals),"output":str(DERIVED)},indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
