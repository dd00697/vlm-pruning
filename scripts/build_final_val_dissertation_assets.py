#!/usr/bin/env python
"""Generate dissertation tables, figures, macros, and derived evidence from audited results."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO_ROOT / "experiments/final_val_efficiency_baselines_20260818"
ANALYSIS = CAMPAIGN / "analysis"
DISSERTATION = Path("/workspace/Dissertation_Draft")
DERIVED = DISSERTATION / "analysis/derived/final_val_efficiency_baselines"
GENERATED = DISSERTATION / "chapters/generated_final_val_efficiency"
FIGURES = DISSERTATION / "images/final_evaluation/final_val_efficiency"

DISPLAY = {
    "Full": "Full",
    "K20": "K20/B48",
    "K12": "K12/B48",
    "M448_OPT": "M448",
    "Z_M448_0": "Z-M448-0",
    "FEATHER_LLAVA_MAIN": "FEATHER (LLaVA adapter)",
    "FASTV_MATCHED_199": "FastV (official fork)",
    "FASTV_STACK_FULL": "FastV-stack Full",
    "NUWA_STACK_FULL": "N\\\"uwa-stack Full",
    "NUWA_NATIVE_192": "N\\\"uwa NT192/DIST280",
}


class DissertationAssetError(RuntimeError):
    """Raised when audited analysis cannot populate a dissertation asset."""


def read(path: Path) -> Any:
    if not path.is_file():
        raise DissertationAssetError(f"Missing prerequisite: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def tex(value: Any) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def table(
    *,
    caption: str,
    label: str,
    columns: str,
    header: str,
    rows: list[str],
    small: bool = True,
) -> str:
    size = "\\small\n" if small else ""
    return "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            size.rstrip(),
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{columns}}}",
            "\\toprule",
            header + "\\\\",
            "\\midrule",
            *(row + "\\\\" for row in rows),
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def mean(timing: dict[str, Any], field: str) -> float:
    return float(timing[field]["mean"])


def std(timing: dict[str, Any], field: str) -> float:
    return float(timing[field]["std"])


def quality_tables(records: dict[str, dict[str, Any]]) -> str:
    order = ["Full", "K20", "K12", "M448_OPT", "Z_M448_0", "FASTV_MATCHED_199", "FEATHER_LLAVA_MAIN"]
    rows = []
    for method in order:
        row = records[method]
        stack_note = "official fork" if method == "FASTV_MATCHED_199" else "canonical"
        rows.append(
            f"{DISPLAY[method]} & {stack_note} & {100*row['acc_at_0_5']:.2f} & "
            f"{row['mean_iou']:.4f} & {row['average_visual_tokens']:.1f} & "
            f"{row['visual_token_layer_work']:,.0f}"
        )
    main = table(
        caption=(
            "Complete RefCOCO-validation quality and visual work. FastV is labelled by its "
            "official fork because disabled-path parity with the canonical stack failed."
        ),
        label="tab:final-val-quality-work",
        columns="llrrrr",
        header="Method & Stack & Acc.@0.5 (\\%) & mIoU & Avg. visual & Token work",
        rows=rows,
    )
    native_rows = []
    for method in ("NUWA_STACK_FULL", "NUWA_NATIVE_192"):
        row = records[method]
        native_rows.append(
            f"{DISPLAY[method]} & {100*row['acc_at_0_5']:.2f} & {row['mean_iou']:.4f} & "
            f"{row['average_visual_tokens']:.1f} & {row['visual_token_layer_work']:,.0f}"
        )
    native = table(
        caption=(
            "Source-native N\\\"uwa quality. The pruned row is compared only with the native Full "
            "control because checkpoint and preprocessing parity with canonical LLaVA failed."
        ),
        label="tab:final-val-nuwa-quality",
        columns="lrrrr",
        header="Method & Acc.@0.5 (\\%) & mIoU & Avg. visual & Token work",
        rows=native_rows,
    )
    return main + native


def timing_tables(
    timing: dict[tuple[str, str], dict[str, Any]],
    accounting: dict[str, dict[str, Any]],
    fixed_timing: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> str:
    methods = ["Full", "K12", "M448_OPT", "Z_M448_0", "FEATHER_LLAVA_MAIN"]
    rows = []
    for method in methods:
        row = timing[("canonical", method)]
        account = accounting[method]
        rows.append(
            f"{DISPLAY[method]} & {mean(row, 'end_to_end_latency_s'):.4f}$\\pm${std(row, 'end_to_end_latency_s'):.4f} & "
            f"{mean(row, 'prefill_including_pruning_latency_s'):.4f} & "
            f"{mean(row, 'time_to_first_token_s'):.4f} & "
            f"{mean(row, 'selector_latency_s'):.4f} & "
            f"{mean(row, 'gather_latency_s') + mean(row, 'merge_latency_s'):.4f} & "
            f"{mean(row, 'decode_latency_s'):.4f} & "
            f"{mean(row, 'generated_token_count'):.2f}"
        )
    primary = table(
        caption=(
            "Five-repetition canonical-stack timing on the ordered 1,000-example validation subset. "
            "Natural-output end-to-end latency is primary; prefill includes all pruning overhead."
        ),
        label="tab:final-val-canonical-timing",
        columns="lrrrrrrr",
        header="Method & E2E s & Prefill s & TTFT s & Select s & Gather/merge s & Decode s & Output tok.",
        rows=rows,
    )
    resource_rows = []
    for method in methods:
        row = timing[("canonical", method)]
        account = accounting[method]
        resource_rows.append(
            f"{DISPLAY[method]} & {mean(row, 'throughput_examples_per_s'):.3f} & "
            f"{account['mean_analytical_prompt_kv_bytes']/2**20:.1f} & "
            f"{account['mean_analytical_final_kv_bytes']/2**20:.1f} & "
            f"{account['mean_peak_allocated_gpu_bytes']/2**30:.3f} & "
            f"{account['mean_peak_reserved_gpu_bytes']/2**30:.3f}"
        )
    resources = table(
        caption=(
            "Canonical throughput, analytical KV storage and physical peak CUDA memory. Model weights "
            "dominate peak allocation, so KV reduction need not reduce the peak proportionally."
        ),
        label="tab:final-val-resources",
        columns="lrrrrr",
        header="Method & Example/s & Prompt KV MiB & Final KV MiB & Peak alloc. GiB & Peak reserv. GiB",
        rows=resource_rows,
    )
    native_rows = []
    for stack, full, pruned in (
        ("fastv", "FASTV_STACK_FULL", "FASTV_MATCHED_199"),
        ("nuwa", "NUWA_STACK_FULL", "NUWA_NATIVE_192"),
    ):
        for method in (full, pruned):
            row = timing[(stack, method)]
            native_rows.append(
                f"{tex(stack)} & {DISPLAY[method]} & {mean(row, 'end_to_end_latency_s'):.4f} & "
                f"{mean(row, 'prefill_including_pruning_latency_s'):.4f} & "
                f"{mean(row, 'generated_token_count'):.2f} & "
                f"{mean(row, 'average_visual_tokens'):.1f}"
            )
    native = table(
        caption=(
            "Source-native repeated timing. Only Full--pruned changes within each row pair are "
            "comparable; raw cross-stack latency ranking is prohibited."
        ),
        label="tab:final-val-native-timing",
        columns="llrrrr",
        header="Stack & Method & E2E s & Prefill s & Output tok. & Avg. visual",
        rows=native_rows,
    )
    fixed = ""
    if fixed_timing:
        fixed_rows = []
        for (stack, method), row in fixed_timing.items():
            fixed_rows.append(
                f"{tex(stack)} & {DISPLAY[method]} & {mean(row, 'end_to_end_latency_s'):.4f} & "
                f"{mean(row, 'prefill_including_pruning_latency_s'):.4f} & "
                f"{mean(row, 'decode_latency_s'):.4f} & {mean(row, 'generated_token_count'):.1f}"
            )
        fixed = table(
            caption=(
                "Three-repetition fixed-32 diagnostic for software stacks that met the registered "
                "natural-output trigger. It diagnoses output-length effects and does not replace "
                "the natural-output endpoint."
            ),
            label="tab:final-val-fixed32-timing",
            columns="llrrrr",
            header="Stack & Method & E2E s & Prefill s & Decode s & Output tok.",
            rows=fixed_rows,
        )
    return primary + resources + native + fixed


def flop_table(accounting: dict[str, dict[str, Any]]) -> str:
    methods = [
        "Full", "K12", "M448_OPT", "Z_M448_0", "FASTV_MATCHED_199",
        "FEATHER_LLAVA_MAIN", "NUWA_NATIVE_192",
    ]
    rows = []
    for method in methods:
        row = accounting[method]
        rows.append(
            f"{DISPLAY[method]} & {row['mean_llm_prefill_flops']/1e12:.3f} & "
            f"{row['mean_score_flops']/1e9:.3f} & {row['mean_distance_flops']/1e9:.3f} & "
            f"{row['mean_merge_flops']/1e9:.3f} & "
            f"{row['mean_modeled_total_multimodal_prefill_flops']/1e12:.3f}"
        )
    return table(
        caption=(
            "Trace-driven modeled FLOPs (two FLOPs per multiply--accumulate). Total includes the "
            "constant vision encoder, realized projector, decoder and explicit pruning arithmetic; "
            "ranking and gather memory movement are reported separately in the machine-readable audit."
        ),
        label="tab:final-val-flops",
        columns="lrrrrr",
        header="Method & LLM TFLOPs & Score GFLOPs & Distance GFLOPs & Merge GFLOPs & Total TFLOPs",
        rows=rows,
    )


def uncertainty_and_decision(
    quality_contrasts: list[dict[str, Any]],
    timing_contrasts: list[dict[str, Any]],
    decision: dict[str, Any],
) -> str:
    wanted_quality = {
        ("K12-minus-Full", "acc_at_0_5_pp"),
        ("M448_OPT-minus-Full", "acc_at_0_5_pp"),
        ("Z_M448_0-minus-Full", "acc_at_0_5_pp"),
        ("FEATHER_LLAVA_MAIN-minus-Full", "acc_at_0_5_pp"),
        ("Z_M448_0-minus-K12", "mean_iou"),
        ("NUWA_NATIVE_192-minus-NUWA_STACK_FULL", "acc_at_0_5_pp"),
    }
    rows = []
    for row in quality_contrasts:
        if (row["contrast"], row["metric"]) in wanted_quality:
            rows.append(
                f"{tex(row['contrast'])} & {tex(row['metric'])} & {row['point_delta']:+.4f} & "
                f"[{row['lower_95']:+.4f}, {row['upper_95']:+.4f}]"
            )
    for row in timing_contrasts:
        if (
            row["stack"] == "canonical"
            and row["right"] == "Full"
            and row["left"] in {"K12", "M448_OPT", "Z_M448_0"}
            and row["metric"] == "end_to_end_latency_s"
        ):
            rows.append(
                f"{tex(row['contrast'])} & E2E seconds & {row['point_delta']:+.4f} & "
                f"[{row['lower_95']:+.4f}, {row['upper_95']:+.4f}]"
            )
    effects = table(
        caption=(
            "Selected paired 95\\% bootstrap intervals. Quality uses 10,000 paired example "
            "resamples; timing uses 10,000 hierarchical paired repetition/example resamples. "
            "Intervals containing zero are inconclusive, not evidence of equivalence."
        ),
        label="tab:final-val-paired-effects",
        columns="llrr",
        header="Contrast & Metric & Difference & 95\\% interval",
        rows=rows,
    )
    decision_rows = []
    for method in ("K12", "M448_OPT", "Z_M448_0"):
        item = decision["method_decisions"][method]
        decision_rows.append(
            f"{DISPLAY[method]} & {'Pass' if item['quality_gate_pass'] else 'Fail'} & "
            f"{'Pass' if item['acceleration_gate_pass'] else 'Fail'} & "
            f"{item['visual_token_layer_work']:,} & "
            f"{'Selected' if decision['selected_method'] == method else '--'}"
        )
    decision_table = table(
        caption=(
            "Preregistered internal candidate rule. Minimum token-layer work is applied only after "
            "both the practical-quality and measured-acceleration gates pass."
        ),
        label="tab:final-val-decision",
        columns="lrrrr",
        header="Method & Quality gate & Acceleration gate & Token work & Outcome",
        rows=decision_rows,
    )
    return effects + decision_table


def main_tex() -> str:
    return """% Generated from audited final validation artifacts; do not hand-edit.
\\input{chapters/generated_final_val_efficiency/quality_work}
\\input{chapters/generated_final_val_efficiency/timing_resources}
\\input{chapters/generated_final_val_efficiency/flops}
\\input{chapters/generated_final_val_efficiency/uncertainty_decision}

\\begin{figure}[H]
  \\centering
  \\includegraphics[width=0.98\\textwidth]{final_evaluation/final_val_efficiency/quality_vs_token_work.pdf}
  \\caption{Complete validation Acc.@0.5 and mIoU against visual token-layer work. Dashed lines show the canonical-stack non-dominated frontier. The hollow FastV marker denotes its separate official fork.}
  \\label{fig:final-val-quality-work}
\\end{figure}

\\begin{figure}[H]
  \\centering
  \\includegraphics[width=0.98\\textwidth]{final_evaluation/final_val_efficiency/quality_vs_natural_latency.pdf}
  \\caption{Canonical-stack validation quality against five-repetition natural-output end-to-end latency.}
  \\label{fig:final-val-quality-latency}
\\end{figure}

\\begin{figure}[H]
  \\centering
  \\includegraphics[width=0.92\\textwidth]{final_evaluation/final_val_efficiency/canonical_timing_decomposition.pdf}
  \\caption{Canonical timing decomposition. Scoring, ranking, gather, merge and segmented-cache dispatch explain why lower token work does not map one-to-one to end-to-end speed.}
  \\label{fig:final-val-timing-components}
\\end{figure}
"""


def macros(decision: dict[str, Any], records: dict[str, dict[str, Any]]) -> str:
    selected = decision["selected_method"]
    selected_display = DISPLAY[selected] if selected else "No candidate"
    row = records[selected] if selected else records["Full"]
    return "\n".join(
        [
            "% Generated from audited final validation artifacts; do not hand-edit.",
            f"\\newcommand{{\\FinalValSelectedMethod}}{{{selected_display}}}",
            f"\\newcommand{{\\FinalValSelectedAcc}}{{{100*row['acc_at_0_5']:.2f}\\%}}",
            f"\\newcommand{{\\FinalValSelectedIoU}}{{{row['mean_iou']:.4f}}}",
            f"\\newcommand{{\\FinalValSelectedAverageTokens}}{{{row['average_visual_tokens']:.1f}}}",
            f"\\newcommand{{\\FinalValSelectedWork}}{{{row['visual_token_layer_work']:,.0f}}}",
            f"\\newcommand{{\\FinalValDecisionStatus}}{{{tex(decision['status'])}}}",
            "",
        ]
    )


def main() -> int:
    joined_list = read(ANALYSIS / "JOINED_METHOD_RECORDS.json")
    records = {row["method"]: row for row in joined_list}
    timing_list = read(ANALYSIS / "TIMING_NATURAL_SUMMARY.json")
    timing = {(row["stack"], row["method"]): row for row in timing_list}
    fixed_path = ANALYSIS / "TIMING_FIXED32_SUMMARY.json"
    fixed_timing = None
    if fixed_path.is_file():
        fixed_list = read(fixed_path)
        fixed_timing = {(row["stack"], row["method"]): row for row in fixed_list}
    accounting_list = read(ANALYSIS / "FLOP_KV_ACCOUNTING.json")["methods"]
    accounting = {row["method"]: row for row in accounting_list}
    quality_contrasts = read(ANALYSIS / "QUALITY_PAIRED_CONTRASTS.json")
    timing_contrasts = read(ANALYSIS / "TIMING_NATURAL_CONTRASTS.json")
    decision = read(ANALYSIS / "FROZEN_FINAL_CANDIDATE.json")
    GENERATED.mkdir(parents=True, exist_ok=True)
    atomic_write(GENERATED / "quality_work.tex", quality_tables(records))
    atomic_write(
        GENERATED / "timing_resources.tex",
        timing_tables(timing, accounting, fixed_timing),
    )
    atomic_write(GENERATED / "flops.tex", flop_table(accounting))
    atomic_write(
        GENERATED / "uncertainty_decision.tex",
        uncertainty_and_decision(quality_contrasts, timing_contrasts, decision),
    )
    atomic_write(GENERATED / "main.tex", main_tex())
    atomic_write(GENERATED / "macros.tex", macros(decision, records))
    FIGURES.mkdir(parents=True, exist_ok=True)
    for source in sorted((ANALYSIS / "figures").glob("*.pdf")):
        shutil.copy2(source, FIGURES / source.name)
    DERIVED.mkdir(parents=True, exist_ok=True)
    sources = [
        "QUALITY_METHOD_METRICS.json", "QUALITY_PAIRED_CONTRASTS.json",
        "TIMING_NATURAL_SUMMARY.json", "TIMING_NATURAL_CONTRASTS.json",
        "FLOP_KV_ACCOUNTING.json", "PARETO_FRONTIERS.json",
        "FROZEN_FINAL_CANDIDATE.json", "CLAIM_EVIDENCE_LEDGER.json",
        "FIXED32_TRIGGER.json",
    ]
    if fixed_path.is_file():
        sources.extend(
            ["TIMING_FIXED32_SUMMARY.json", "TIMING_FIXED32_CONTRASTS.json", "TIMING_FIXED32_INVENTORY.json"]
        )
    for filename in sources:
        shutil.copy2(ANALYSIS / filename, DERIVED / filename)
    generated_files = sorted(GENERATED.glob("*.tex"))
    figure_files = sorted(FIGURES.glob("*.pdf"))
    derived_files = sorted(
        path for path in DERIVED.glob("*.json") if path.name != "ASSET_MANIFEST.json"
    )
    manifest = {
        "status": "complete",
        "source_campaign": str(CAMPAIGN),
        "generated_tex": {path.name: sha256_file(path) for path in generated_files},
        "figures": {path.name: sha256_file(path) for path in figure_files},
        "derived_json": {path.name: sha256_file(path) for path in derived_files},
        "testA_testB_included": False,
    }
    atomic_write(DERIVED / "ASSET_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
