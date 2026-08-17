#!/usr/bin/env python
"""Build the final analysis and pre-audit state for the hardware-aware campaign."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO_ROOT / "experiments/hardware_aware_zero_tail_refcoco_val_20260816"
REGISTRY_PATH = CAMPAIGN_ROOT / "EXPERIMENT_REGISTRY.json"
CONTROLLER_PATH = CAMPAIGN_ROOT / "CONTROLLER_STATE.json"
REPORT_PATH = CAMPAIGN_ROOT / "RESEARCH_REPORT.md"
ANALYSIS_DIR = CAMPAIGN_ROOT / "analysis"
FRONTIERS_PATH = ANALYSIS_DIR / "FINAL_PARETO_FRONTIERS.json"
DECISION_PATH = ANALYSIS_DIR / "FINAL_DECISION.json"
COMPLETION_PATH = ANALYSIS_DIR / "FINAL_ANALYSIS_COMPLETED.json"
PRIMARY_OUTCOME = "D — Quality/memory improvement only; no real latency win"
PRE_AUDIT_STATE = "FINAL_AUDIT_PENDING"


class FinalizationError(RuntimeError):
    """Raised when prerequisite evidence cannot support finalization."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FinalizationError(f"Missing prerequisite: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalizationError(f"Expected JSON object: {path}")
    return value


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def source_record() -> dict[str, Any]:
    paths = (Path("scripts/finalize_hardware_aware_campaign.py"),)
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *(str(path) for path in paths)],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if status:
        raise FinalizationError("Final analysis requires clean committed source")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    return {
        "git_commit": commit,
        "source_sha256": {str(path): sha256_file(REPO_ROOT / path) for path in paths},
    }


def mean(summary: dict[str, Any], field: str) -> float:
    value = summary[field]
    return float(value["mean"] if isinstance(value, dict) else value)


def pareto_frontier(
    records: list[dict[str, Any]], quality_field: str, cost_field: str
) -> list[str]:
    """Return points not dominated by higher quality and lower cost."""
    result: list[str] = []
    for candidate in records:
        dominated = False
        for other in records:
            if other["method"] == candidate["method"]:
                continue
            quality_no_worse = float(other[quality_field]) >= float(candidate[quality_field])
            cost_no_worse = float(other[cost_field]) <= float(candidate[cost_field])
            one_strict = (
                float(other[quality_field]) > float(candidate[quality_field])
                or float(other[cost_field]) < float(candidate[cost_field])
            )
            if quality_no_worse and cost_no_worse and one_strict:
                dominated = True
                break
        if not dominated:
            result.append(str(candidate["method"]))
    return result


def require_complete(payload: dict[str, Any], name: str) -> None:
    if payload.get("status") != "complete":
        raise FinalizationError(f"{name} is not complete")


def build_method_records(
    natural: dict[str, Any], zero_quality: dict[str, Any], early_quality: dict[str, Any]
) -> list[dict[str, Any]]:
    quality = zero_quality["metrics"]
    final_visual = {"K12_ORIGINAL": 48, "M448_OPT": 48, "Z_K12_0": 0, "Z_M448_0": 0}
    records: list[dict[str, Any]] = []
    for method, summary in natural["summaries"].items():
        quality_key = "M448" if method == "M448_OPT" else method
        record = {
            "method": method,
            "quality_source_method": quality_key,
            "acc_at_0_5": float(quality[quality_key]["acc_at_0_5"]),
            "mean_iou": float(quality[quality_key]["mean_iou"]),
            "parse_rate": float(quality[quality_key]["parse_rate"]),
            "quality_feasible": True,
            "matched_natural_timing": True,
            "natural_end_to_end_s": mean(summary, "end_to_end_latency_s"),
            "prefill_including_overhead_s": mean(summary, "main_prefill_with_overhead_latency_s"),
            "time_to_first_token_s": mean(summary, "time_to_first_token_s"),
            "prompt_kv_bytes": mean(summary, "prompt_kv_total_bytes"),
            "final_kv_bytes": mean(summary, "final_kv_total_bytes"),
            "visual_token_layer_work": mean(summary, "visual_token_layer_work"),
            "average_visual_tokens": mean(summary, "average_visual_tokens"),
            "final_visual_tokens": final_visual[method],
        }
        records.append(record)

    e256_metrics = read_json(
        CAMPAIGN_ROOT / "early_compression/quality/e256_pool_attempt1/metrics.json"
    )
    e256_quality = early_quality["metrics"]["E256_POOL"]
    records.append(
        {
            "method": "E256_POOL",
            "quality_source_method": "E256_POOL",
            "acc_at_0_5": float(e256_quality["acc_at_0_5"]),
            "mean_iou": float(e256_quality["mean_iou"]),
            "parse_rate": float(e256_quality["parse_rate"]),
            "quality_feasible": False,
            "matched_natural_timing": False,
            "natural_end_to_end_s": None,
            "prefill_including_overhead_s": None,
            "time_to_first_token_s": None,
            "prompt_kv_bytes": float(e256_metrics["prompt_kv_total_bytes_mean"]),
            "final_kv_bytes": float(e256_metrics["final_kv_total_bytes_mean"]),
            "visual_token_layer_work": float(e256_metrics["visual_token_layer_work"]),
            "average_visual_tokens": float(e256_metrics["average_visual_tokens"]),
            "final_visual_tokens": 256,
            "unmatched_single_pass_timing_excluded_reason": (
                "Quality gate failed, so the registered matched timing branch was not triggered"
            ),
        }
    )
    return records


def build_frontiers(records: list[dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    timed = [row for row in records if row["matched_natural_timing"]]
    feasible = [row for row in records if row["quality_feasible"]]
    axes = [
        ("acc_vs_natural_end_to_end", "acc_at_0_5", "natural_end_to_end_s", timed),
        ("miou_vs_natural_end_to_end", "mean_iou", "natural_end_to_end_s", timed),
        ("acc_vs_prefill", "acc_at_0_5", "prefill_including_overhead_s", timed),
        ("miou_vs_prefill", "mean_iou", "prefill_including_overhead_s", timed),
        ("acc_vs_ttft", "acc_at_0_5", "time_to_first_token_s", timed),
        ("miou_vs_ttft", "mean_iou", "time_to_first_token_s", timed),
        ("acc_vs_prompt_kv", "acc_at_0_5", "prompt_kv_bytes", records),
        ("miou_vs_prompt_kv", "mean_iou", "prompt_kv_bytes", records),
        ("acc_vs_final_kv", "acc_at_0_5", "final_kv_bytes", records),
        ("miou_vs_final_kv", "mean_iou", "final_kv_bytes", records),
        ("acc_vs_token_layer_work", "acc_at_0_5", "visual_token_layer_work", records),
        ("miou_vs_token_layer_work", "mean_iou", "visual_token_layer_work", records),
        ("acc_vs_final_visual_tokens", "acc_at_0_5", "final_visual_tokens", records),
        ("miou_vs_final_visual_tokens", "mean_iou", "final_visual_tokens", records),
    ]
    frontiers: dict[str, Any] = {}
    for name, quality_field, cost_field, eligible in axes:
        feasible_eligible = [row for row in feasible if row in eligible]
        frontiers[name] = {
            "quality_axis": quality_field,
            "efficiency_axis": cost_field,
            "eligible_methods": [row["method"] for row in eligible],
            "all_evaluated_frontier": pareto_frontier(eligible, quality_field, cost_field),
            "quality_feasible_frontier": pareto_frontier(
                feasible_eligible, quality_field, cost_field
            ),
        }
    return {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": utc_now(),
        "source": source,
        "dominance_rule": "higher quality and lower cost, with at least one strict inequality",
        "latency_eligibility_note": (
            "Only counterbalanced matched natural timing enters latency frontiers; "
            "the E256 quality-run timing is excluded"
        ),
        "methods": records,
        "frontiers": frontiers,
    }


def build_decision(
    natural: dict[str, Any], fixed32: dict[str, Any], strict: dict[str, Any], frontiers: dict[str, Any]
) -> dict[str, Any]:
    k12 = natural["summaries"]["K12_ORIGINAL"]
    zm = natural["summaries"]["Z_M448_0"]
    return {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": utc_now(),
        "primary_outcome": PRIMARY_OUTCOME,
        "strict_winners": strict["strict_winners"],
        "single_recommended_method": "K12_ORIGINAL",
        "recommendation_reason": (
            "It has the lowest matched natural end-to-end mean and the highest Acc@0.5; "
            "no candidate passes the strict actual-efficiency gate"
        ),
        "memory_constrained_alternative": "Z_M448_0",
        "memory_alternative_reason": (
            "It preserves the exploratory quality screen while minimizing prompt/final KV, "
            "token-layer work, average visual tokens, and final visual tokens, but is not an acceleration win"
        ),
        "algorithmic_efficiency": {
            "Z_M448_0_token_layer_work_reduction_vs_k12": 1.0
            - mean(zm, "visual_token_layer_work") / mean(k12, "visual_token_layer_work"),
            "Z_M448_0_average_visual_token_reduction_vs_k12": 1.0
            - mean(zm, "average_visual_tokens") / mean(k12, "average_visual_tokens"),
        },
        "memory_efficiency": {
            "Z_M448_0_prompt_kv_reduction_vs_k12": 1.0
            - mean(zm, "prompt_kv_total_bytes") / mean(k12, "prompt_kv_total_bytes"),
            "Z_M448_0_final_kv_reduction_vs_k12": 1.0
            - mean(zm, "final_kv_total_bytes") / mean(k12, "final_kv_total_bytes"),
        },
        "realized_hardware_efficiency": {
            "Z_M448_0_prefill_change_vs_k12": mean(zm, "main_prefill_with_overhead_latency_s")
            / mean(k12, "main_prefill_with_overhead_latency_s")
            - 1.0,
            "Z_M448_0_natural_e2e_change_vs_k12": mean(zm, "end_to_end_latency_s")
            / mean(k12, "end_to_end_latency_s")
            - 1.0,
            "fixed32_is_secondary_only": True,
            "fixed32_m448_opt_prefill_change_vs_k12": mean(
                fixed32["summaries"]["M448_OPT"], "main_prefill_with_overhead_latency_s"
            )
            / mean(fixed32["summaries"]["K12_ORIGINAL"], "main_prefill_with_overhead_latency_s")
            - 1.0,
        },
        "future_learned_router": (
            "Do not study it as the next efficiency fix. Replay showed scoring was not the dominant "
            "gap, and a router would add prediction and gather cost. Revisit only as a separately "
            "authorized quality-adaptive late-exit study after a native packed runtime exists."
        ),
        "later_8811_confirmation": (
            "Not justified for an acceleration claim because there is no strict latency winner. "
            "A separately authorized held-out confirmation is justified only if Z_M448_0 is advanced "
            "as a memory-constrained method."
        ),
        "frontiers_sha256": sha256_file(FRONTIERS_PATH),
        "frontier_summary": {
            name: value["quality_feasible_frontier"]
            for name, value in frontiers["frontiers"].items()
        },
    }


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def render_report(
    attribution: dict[str, Any], repair: dict[str, Any], zero: dict[str, Any],
    natural: dict[str, Any], fixed: dict[str, Any], early: dict[str, Any],
    decision: dict[str, Any], source: dict[str, Any]
) -> str:
    ns = natural["summaries"]
    nc = natural["candidate_minus_k12_contrasts"]
    fq = fixed["summaries"]
    early_metrics = early["metrics"]["E256_POOL"]
    comp = attribution["m448_minus_k12_component_group_s"]
    zero_metrics = zero["metrics"]
    return f"""# Hardware-aware zero-tail RefCOCO validation campaign

## Scope, status, and frozen protocol

This completed discovery campaign used exactly the frozen 1,000-row RefCOCO-validation manifest, SHA-256 `98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd`. It did not launch an 8,811-row, held-out, transfer, learned-router, or dissertation job. All results use LLaVA-1.5-7B in BF16, batch size 1, on one NVIDIA A40.

Primary outcome: `{PRIMARY_OUTCOME}`.

Single recommendation: **K12_ORIGINAL** for the stated objective of quality-constrained real latency. **Z_M448_0** is the memory-constrained alternative, not an acceleration result.

Campaign state before independent audit: `{PRE_AUDIT_STATE}`.

## Phase 0 and instrumentation

Provenance, storage, checkpoint, historical hashes, manifest order/IDs/images, environment, and source cleanliness passed before GPU work. Instrumentation separately measured synchronized score/rank, gather/merge, cache/dispatch, decoder segments, TTFT, decode, prompt/final KV, and CPU/CUDA clocks. Two-row and twenty-row K12/M448/replay smokes matched their intended historical semantics; replay disabled only score/rank.

## 1–3. M448 attribution, replay, and repairs

The five-repetition 100-row profile measured an M448-minus-K12 prefill gap of {fmt(attribution['m448_minus_k12_prefill_wall_s']*1000, 3)} ms. Synchronized component contrasts were score/rank {fmt(comp['score_rank']*1000, 3)} ms, gather/merge +{fmt(comp['gather_merge']*1000, 3)} ms, cache/dispatch +{fmt(comp['cache_segment_dispatch']*1000, 3)} ms, and decoder-prefill {fmt(comp['decoder_prefill']*1000, 3)} ms. Natural decode wall time was +{fmt(attribution['m448_minus_k12_decode_wall_s']*1000, 3)} ms. Scopes overlap, so these figures are attribution signals rather than an additive decomposition.

Replay removed only {fmt(attribution['live_minus_replay_prefill_wall_s']*1000, 3)} ms of live prefill—below the registered {fmt(attribution['twenty_percent_positive_prefill_gap_s']*1000, 3)} ms trigger—and did not remove segmented execution. This directly rejects selector scoring as the dominant bottleneck.

Fused gather/merge and native-cache handling were semantically exact. Fused gather/merge passed the selection guard; native cache alone failed it. Their exact composition, `R_FUSED_NATIVE`, had the lowest eligible profile E2E mean and became `M448_OPT`. The bounded `torch.compile` attempt was rejected at a full-graph timing break, and CUDA graphs were rejected because the dynamic path synchronized during capture. Neither was broadened into a search.

| Profile/repair | Prefill (s) | Natural E2E (s) | Decision |
|---|---:|---:|---|
| P_K12 | 0.099797 | 0.810815 | profile reference |
| P_M448 | 0.112727 | 0.839612 | reproduced gap |
| P_M448_REPLAY | 0.111690 | 0.844104 | non-deployable attribution only |
| R_FUSED_GATHER_MERGE | 0.104903 | 0.833248 | exact, eligible |
| R_NATIVE_CACHE | 0.116477 | 0.845692 | exact, guard failed |
| R_FUSED_NATIVE | 0.103313 | 0.830558 | selected M448_OPT |

## 4. Zero-tail quality and failure movement

Both mandatory zero-tail cells passed all 1,000-row integrity and K12-like quality checks. Visual tokens are physically absent after K20, text and gap positions survive, generated-token caches grow correctly, and K20 zero-tail performs no scoring/ranking.

| Method | Acc@0.5 | mIoU | Avg visual tokens | Token-layer work | Final visual tokens |
|---|---:|---:|---:|---:|---:|
| K12_ORIGINAL | {zero_metrics['K12_ORIGINAL']['acc_at_0_5']:.3f} | {zero_metrics['K12_ORIGINAL']['mean_iou']:.6f} | 246 | 7,872 | 48 |
| M448 / M448_OPT semantics | {zero_metrics['M448']['acc_at_0_5']:.3f} | {zero_metrics['M448']['mean_iou']:.6f} | 214 | 6,848 | 48 |
| Z_K12_0 | {zero_metrics['Z_K12_0']['acc_at_0_5']:.3f} | {zero_metrics['Z_K12_0']['mean_iou']:.6f} | 228 | 7,296 | 0 |
| Z_M448_0 | {zero_metrics['Z_M448_0']['acc_at_0_5']:.3f} | {zero_metrics['Z_M448_0']['mean_iou']:.6f} | 196 | 6,272 | 0 |

Against K12, Z_K12_0 gained 47 previously incorrect rows and lost 54 previously correct rows; Z_M448_0 gained 80 and lost 88. Z_M448_0 versus M448 gained 23 and lost 22. The paired 95% Acc intervals include zero for both zero-tail/K12 contrasts. Z_K12_0's mIoU increase was +0.00765 (95% interval +0.00054 to +0.01462); Z_M448_0's was +0.00908 (-0.00038 to +0.01891). `Z_M448_16` was correctly skipped because Z_M448_0 already passed.

## 5. Matched natural timing and proxy/latency separation

Twenty serial trials completed: five counterbalanced repetitions per method over all 1,000 rows. All overhead is included. Each candidate reduced prompt KV but was slower than K12 in mean natural E2E; no candidate passed the strict gate.

| Method | Prefill (s) | TTFT (s) | Decode (s) | Natural E2E (s) | Prompt KV (bytes) | Final KV (bytes) |
|---|---:|---:|---:|---:|---:|---:|
| K12_ORIGINAL | {fmt(mean(ns['K12_ORIGINAL'],'main_prefill_with_overhead_latency_s'))} | {fmt(mean(ns['K12_ORIGINAL'],'time_to_first_token_s'))} | {fmt(mean(ns['K12_ORIGINAL'],'decode_latency_s'))} | {fmt(mean(ns['K12_ORIGINAL'],'end_to_end_latency_s'))} | {mean(ns['K12_ORIGINAL'],'prompt_kv_total_bytes'):.0f} | {mean(ns['K12_ORIGINAL'],'final_kv_total_bytes'):.0f} |
| M448_OPT | {fmt(mean(ns['M448_OPT'],'main_prefill_with_overhead_latency_s'))} | {fmt(mean(ns['M448_OPT'],'time_to_first_token_s'))} | {fmt(mean(ns['M448_OPT'],'decode_latency_s'))} | {fmt(mean(ns['M448_OPT'],'end_to_end_latency_s'))} | {mean(ns['M448_OPT'],'prompt_kv_total_bytes'):.0f} | {mean(ns['M448_OPT'],'final_kv_total_bytes'):.0f} |
| Z_K12_0 | {fmt(mean(ns['Z_K12_0'],'main_prefill_with_overhead_latency_s'))} | {fmt(mean(ns['Z_K12_0'],'time_to_first_token_s'))} | {fmt(mean(ns['Z_K12_0'],'decode_latency_s'))} | {fmt(mean(ns['Z_K12_0'],'end_to_end_latency_s'))} | {mean(ns['Z_K12_0'],'prompt_kv_total_bytes'):.0f} | {mean(ns['Z_K12_0'],'final_kv_total_bytes'):.0f} |
| Z_M448_0 | {fmt(mean(ns['Z_M448_0'],'main_prefill_with_overhead_latency_s'))} | {fmt(mean(ns['Z_M448_0'],'time_to_first_token_s'))} | {fmt(mean(ns['Z_M448_0'],'decode_latency_s'))} | {fmt(mean(ns['Z_M448_0'],'end_to_end_latency_s'))} | {mean(ns['Z_M448_0'],'prompt_kv_total_bytes'):.0f} | {mean(ns['Z_M448_0'],'final_kv_total_bytes'):.0f} |

Candidate-minus-K12 hierarchical paired 95% intervals were:

| Method | Prefill delta (s), 95% interval | E2E delta (s), 95% interval | Classification |
|---|---|---|---|
| M448_OPT | {nc['M448_OPT']['main_prefill_with_overhead_latency_s']['point_delta']:+.6f} [{nc['M448_OPT']['main_prefill_with_overhead_latency_s']['lower_95']:+.6f}, {nc['M448_OPT']['main_prefill_with_overhead_latency_s']['upper_95']:+.6f}] | {nc['M448_OPT']['end_to_end_latency_s']['point_delta']:+.6f} [{nc['M448_OPT']['end_to_end_latency_s']['lower_95']:+.6f}, {nc['M448_OPT']['end_to_end_latency_s']['upper_95']:+.6f}] | MEMORY_OR_COMPUTE_PROXY_ONLY |
| Z_K12_0 | {nc['Z_K12_0']['main_prefill_with_overhead_latency_s']['point_delta']:+.6f} [{nc['Z_K12_0']['main_prefill_with_overhead_latency_s']['lower_95']:+.6f}, {nc['Z_K12_0']['main_prefill_with_overhead_latency_s']['upper_95']:+.6f}] | {nc['Z_K12_0']['end_to_end_latency_s']['point_delta']:+.6f} [{nc['Z_K12_0']['end_to_end_latency_s']['lower_95']:+.6f}, {nc['Z_K12_0']['end_to_end_latency_s']['upper_95']:+.6f}] | MEMORY_OR_COMPUTE_PROXY_ONLY |
| Z_M448_0 | {nc['Z_M448_0']['main_prefill_with_overhead_latency_s']['point_delta']:+.6f} [{nc['Z_M448_0']['main_prefill_with_overhead_latency_s']['lower_95']:+.6f}, {nc['Z_M448_0']['main_prefill_with_overhead_latency_s']['upper_95']:+.6f}] | {nc['Z_M448_0']['end_to_end_latency_s']['point_delta']:+.6f} [{nc['Z_M448_0']['end_to_end_latency_s']['lower_95']:+.6f}, {nc['Z_M448_0']['end_to_end_latency_s']['upper_95']:+.6f}] | MEMORY_OR_COMPUTE_PROXY_ONLY |

The secondary identical-continuation fixed32 diagnostic used three repetitions per method. M448_OPT prefill was {fmt(mean(fq['M448_OPT'],'main_prefill_with_overhead_latency_s'))} s versus K12 {fmt(mean(fq['K12_ORIGINAL'],'main_prefill_with_overhead_latency_s'))} s, but its decode was {fmt(mean(fq['M448_OPT'],'decode_latency_s'))} s versus {fmt(mean(fq['K12_ORIGINAL'],'decode_latency_s'))} s. Its fixed32 E2E was 0.00818 s faster (95% interval -0.01259 to -0.00509), while Z_M448_0 was 0.01124 s slower. This diagnostic does not override natural generation.

## 6–7. Contiguous pooling and adaptation branch

Phase 5 triggered because decoder-internal methods had no strict winner. E256_POOL performed one exact area-weighted 24×24-to-16×16 pool before the projector; every patch contributed, mass summed to 576, anchors/positions were deterministic, and one contiguous cache traversed all 32 layers.

E256_POOL scored Acc@0.5 {early_metrics['acc_at_0_5']:.3f}, mIoU {early_metrics['mean_iou']:.6f}, and parse rate 1.0. It failed the K12-like quality screen and its 0.419 LoRA near-quality Acc trigger. Therefore E196_POOL, E256_LORA training, and both early matched-timing branches were correctly skipped. Its single quality-pass timing is not matched or counterbalanced and cannot establish that pooling outperformed decoder compaction in actual latency.

No validation label entered training because no training was launched. The train manifest was verified at preflight but unused.

## 8. Pareto frontiers and recommendation

Separate machine-readable frontiers are in `analysis/FINAL_PARETO_FRONTIERS.json`. The quality-feasible summaries are:

- Acc versus natural E2E, prefill, and TTFT: K12_ORIGINAL only.
- mIoU versus natural E2E, prefill, and TTFT: K12_ORIGINAL, M448_OPT, and Z_M448_0.
- Acc versus prompt/final KV and token-layer work: K12_ORIGINAL, Z_K12_0, and Z_M448_0.
- mIoU versus prompt/final KV and token-layer work: Z_M448_0 only.
- Acc versus final visual-token count: K12_ORIGINAL and Z_K12_0; mIoU versus final visual-token count: Z_M448_0 only.

Algorithmically, Z_M448_0 reduces token-layer work by {decision['algorithmic_efficiency']['Z_M448_0_token_layer_work_reduction_vs_k12']*100:.1f}% and average visual tokens by {decision['algorithmic_efficiency']['Z_M448_0_average_visual_token_reduction_vs_k12']*100:.1f}%. In memory, it reduces prompt KV by {decision['memory_efficiency']['Z_M448_0_prompt_kv_reduction_vs_k12']*100:.1f}% and final KV by {decision['memory_efficiency']['Z_M448_0_final_kv_reduction_vs_k12']*100:.1f}%. On hardware, however, prefill rises {decision['realized_hardware_efficiency']['Z_M448_0_prefill_change_vs_k12']*100:.1f}% and natural E2E rises {decision['realized_hardware_efficiency']['Z_M448_0_natural_e2e_change_vs_k12']*100:.1f}%.

Thus K12_ORIGINAL is the single recommendation for the registered actual-latency objective. Z_M448_0 is scientifically useful when KV capacity or mIoU is prioritized, but it must be labelled a memory/compute-proxy trade-off.

## 9–10. Learned-router and confirmation decisions

A learned late-exit router is not the next efficiency fix supported by these data. Replay showed that eliminating scoring saved only about 1 ms, while gather/cache/decode behavior remained. A learned decision can be useful only if it changes per-example quality allocation enough to repay prediction, synchronization, and gather costs; it should be a later separately authorized study built on native packed execution, not assumed to be faster.

An 8,811-row confirmation is not justified for an acceleration claim because no method achieved one. A later, separately authorized held-out confirmation is scientifically reasonable only if Z_M448_0 is advanced specifically as a memory-constrained method. None was launched here.

## Limitations

The quality screen uses a discovery subset and is not an equivalence or held-out claim. Timing reflects one A40, batch size 1, this software implementation, and natural output-length distributions. Fixed32 results show output-length dependence but are secondary. No transfer dataset, held-out split, multi-GPU/batch scaling, or alternative kernel implementation was tested.

## Failures and preservation

Rejected attempts remain preserved with machine-readable reasons. They include provenance restart artifacts, two initial instrumentation-oracle/schema errors, bounded repair smoke/profile errors, timing-smoke environment/reference/access errors, compiler/capture incompatibilities, and one fixed32 M448_OPT trial interrupted at a persistent-turn boundary and replaced under a new attempt. No rejected rows entered accepted aggregates and no accepted output was overwritten.

## Provenance

- Final-analysis source epoch: `{source['git_commit']}`
- Natural timing source epoch: `{natural['source']['git_commit']}`
- E256 analysis source epoch: `{early['source']['git_commit']}`
- Natural timing repetitions/rows: 5 per method / 1,000 per trial
- Fixed32 repetitions/rows: 3 per method / 1,000 per trial
- Bootstrap resamples: 10,000 paired hierarchical for timing and paired for quality
"""


def run() -> int:
    source = source_record()
    attribution = read_json(CAMPAIGN_ROOT / "profiling/aggregate/ATTRIBUTION.json")
    repair = read_json(CAMPAIGN_ROOT / "runtime_repairs/aggregate/M448_OPT_SELECTION.json")
    zero = read_json(CAMPAIGN_ROOT / "zero_tail/analysis/QUALITY_RESULTS.json")
    natural = read_json(CAMPAIGN_ROOT / "matched_timing/analysis/NATURAL_RESULTS.json")
    fixed = read_json(CAMPAIGN_ROOT / "matched_timing/analysis/FIXED32_RESULTS.json")
    strict = read_json(CAMPAIGN_ROOT / "matched_timing/analysis/STRICT_GATE.json")
    early = read_json(CAMPAIGN_ROOT / "early_compression/analysis/QUALITY_RESULTS.json")
    branches = read_json(CAMPAIGN_ROOT / "early_compression/analysis/BRANCH_DECISIONS.json")
    for name, payload in (
        ("zero-tail quality", zero), ("natural timing", natural), ("fixed32 timing", fixed),
        ("strict gate", strict), ("early quality", early), ("early branches", branches),
    ):
        require_complete(payload, name)
    if strict.get("strict_winners") != [] or strict.get("phase5_decision") != "TRIGGERED_NO_DECODER_INTERNAL_STRICT_WIN":
        raise FinalizationError("Strict gate does not support Phase 5 finalization")
    if branches.get("phase5_result") != "NO_QUALITY_FEASIBLE_EARLY_COMPRESSION_CANDIDATE":
        raise FinalizationError("Early-compression branch is not terminal")
    records = build_method_records(natural, zero, early)
    frontiers = build_frontiers(records, source)
    atomic_write(FRONTIERS_PATH, canonical_bytes(frontiers))
    decision = build_decision(natural, fixed, strict, frontiers)
    decision["source"] = source
    decision["evidence_hashes"] = {
        "attribution": sha256_file(CAMPAIGN_ROOT / "profiling/aggregate/ATTRIBUTION.json"),
        "repair_selection": sha256_file(CAMPAIGN_ROOT / "runtime_repairs/aggregate/M448_OPT_SELECTION.json"),
        "zero_quality": sha256_file(CAMPAIGN_ROOT / "zero_tail/analysis/QUALITY_RESULTS.json"),
        "natural_timing": sha256_file(CAMPAIGN_ROOT / "matched_timing/analysis/NATURAL_RESULTS.json"),
        "fixed32_timing": sha256_file(CAMPAIGN_ROOT / "matched_timing/analysis/FIXED32_RESULTS.json"),
        "strict_gate": sha256_file(CAMPAIGN_ROOT / "matched_timing/analysis/STRICT_GATE.json"),
        "early_quality": sha256_file(CAMPAIGN_ROOT / "early_compression/analysis/QUALITY_RESULTS.json"),
        "early_branches": sha256_file(CAMPAIGN_ROOT / "early_compression/analysis/BRANCH_DECISIONS.json"),
    }
    atomic_write(DECISION_PATH, canonical_bytes(decision))

    registry = read_json(REGISTRY_PATH)
    natural_attempts = [trial["run_dir"] for trial in natural["trials"]]
    fixed_attempts = [trial["run_dir"] for trial in fixed["trials"]]
    registry["cells"]["T_DECODER_NATURAL"].update(
        {"status": "complete", "attempts": natural_attempts, "accepted_trials": 20}
    )
    registry["cells"]["T_DECODER_FIXED32"].update(
        {
            "status": "complete_secondary_diagnostic",
            "attempts": fixed_attempts,
            "accepted_trials": 9,
            "rejected_attempts": [
                "matched_timing/fixed32/rep3_pos3_m448_opt_attempt1"
            ],
        }
    )
    registry["cells"]["E256_POOL"].update(
        {
            "status": "complete_quality_not_feasible",
            "attempts": ["early_compression/quality/e256_pool_attempt1"],
            "smokes": [
                "early_compression/smokes/e256_pool_smoke2_attempt1",
                "early_compression/smokes/e256_pool_smoke20_attempt1",
            ],
        }
    )
    registry["cells"]["E196_POOL"]["status"] = "SKIPPED_CONSERVATIVE_POOL_NOT_QUALITY_FEASIBLE"
    registry["cells"]["E256_LORA"]["status"] = "SKIPPED_LORA_NEAR_QUALITY_GATE_FAILED"
    registry["cells"]["T_EARLY_NATURAL"] = {
        "phase": "early_compression", "kind": "conditional_matched_timing", "attempts": [],
        "status": "SKIPPED_NO_QUALITY_FEASIBLE_EARLY_CANDIDATE",
    }
    registry["cells"]["T_EARLY_FIXED32"] = {
        "phase": "early_compression", "kind": "conditional_secondary_diagnostic", "attempts": [],
        "status": "SKIPPED_NO_QUALITY_FEASIBLE_EARLY_CANDIDATE",
    }
    branch_map = {row["cell_id"]: row for row in registry["branch_decisions"]}
    for cell_id, payload in branches["decisions"].items():
        branch_map[cell_id] = {
            "cell_id": cell_id,
            "decision": payload["decision"],
            "evidence_path": relative(CAMPAIGN_ROOT / "early_compression/analysis/BRANCH_DECISIONS.json"),
            "reason": payload,
        }
    branch_map["PHASE5"] = {
        "cell_id": "PHASE5",
        "decision": "triggered",
        "evidence_path": relative(CAMPAIGN_ROOT / "matched_timing/analysis/STRICT_GATE.json"),
        "reason": "No decoder-internal strict actual-efficiency winner",
    }
    registry["branch_decisions"] = list(branch_map.values())
    registry["phases"]["decoder_timing"] = {
        "status": "complete",
        "evidence_path": relative(CAMPAIGN_ROOT / "matched_timing/analysis/NATURAL_RESULTS.json"),
        "strict_gate": "no_winner",
    }
    registry["phases"]["early_compression"] = {
        "status": "complete",
        "evidence_path": relative(CAMPAIGN_ROOT / "early_compression/analysis/BRANCH_DECISIONS.json"),
        "result": branches["phase5_result"],
    }
    registry["phases"]["analysis"] = {
        "status": "complete", "evidence_path": relative(DECISION_PATH)
    }
    registry["phases"]["final_audit"] = {"status": "pending"}
    registry["source_commit"] = source["git_commit"]
    registry["source_epochs"].update(
        {
            "early_compression": "51426178bc8ecec99443574e5402ea483dc5fa94",
            "early_analysis": early["source"]["git_commit"],
            "final_analysis": source["git_commit"],
        }
    )
    registry["strict_actual_efficiency_winners"] = []
    registry["primary_outcome"] = PRIMARY_OUTCOME
    registry["terminal_outcome"] = decision
    registry["status"] = PRE_AUDIT_STATE
    registry["updated_at_utc"] = utc_now()
    atomic_write(REGISTRY_PATH, canonical_bytes(registry))

    controller = read_json(CONTROLLER_PATH)
    controller["accepted_prefixes"].update(
        {
            "natural_timing": {"accepted_trials": 20, "rows_per_trial": 1000},
            "fixed32_timing": {"accepted_trials": 9, "rows_per_trial": 1000},
            "early_compression_quality": {"E256_POOL": 1000},
        }
    )
    controller["active_workers"] = []
    controller["current_phase"] = "final_audit"
    controller["state"] = PRE_AUDIT_STATE
    controller["next_required_action"] = "run independent final completion audit"
    controller["stop_reason"] = None
    controller["primary_outcome"] = PRIMARY_OUTCOME
    controller["decision_history"].extend(
        [
            {"phase": "decoder_timing", "status": "complete", "evidence_path": relative(CAMPAIGN_ROOT / "matched_timing/analysis/STRICT_GATE.json"), "decision": "no_strict_winner"},
            {"phase": "early_compression", "status": "complete", "evidence_path": relative(CAMPAIGN_ROOT / "early_compression/analysis/BRANCH_DECISIONS.json"), "decision": branches["phase5_result"]},
            {"phase": "analysis", "status": "complete", "evidence_path": relative(DECISION_PATH), "primary_outcome": PRIMARY_OUTCOME},
        ]
    )
    controller["updated_at_utc"] = utc_now()
    atomic_write(CONTROLLER_PATH, canonical_bytes(controller))

    atomic_write(
        REPORT_PATH,
        render_report(attribution, repair, zero, natural, fixed, early, decision, source).encode("utf-8"),
    )
    completion = {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "source": source,
        "frontiers_sha256": sha256_file(FRONTIERS_PATH),
        "decision_sha256": sha256_file(DECISION_PATH),
        "registry_sha256": sha256_file(REGISTRY_PATH),
        "controller_sha256": sha256_file(CONTROLLER_PATH),
        "report_sha256": sha256_file(REPORT_PATH),
        "pre_audit_state": PRE_AUDIT_STATE,
        "primary_outcome": PRIMARY_OUTCOME,
    }
    atomic_write(COMPLETION_PATH, canonical_bytes(completion))
    print(json.dumps({"status": "complete", "state": PRE_AUDIT_STATE, "primary_outcome": PRIMARY_OUTCOME}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (FinalizationError, OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
