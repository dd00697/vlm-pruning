#!/usr/bin/env python
"""Build trace-driven FLOP, work, KV, and physical-memory accounting."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_final_val_timing_campaign import plans  # noqa: E402
from vlm_pruning.final_efficiency import account_trace  # noqa: E402


CAMPAIGN = REPO_ROOT / "experiments/final_val_efficiency_baselines_20260818"
ANALYSIS = CAMPAIGN / "analysis"
ROWS = 1_000
HIDDEN = 4096
VISION_HIDDEN = 1024
VISION_INTERMEDIATE = 4096
HEADS = 32

METHOD_STACK = {
    "Full": "canonical",
    "K12": "canonical",
    "M448_OPT": "canonical",
    "Z_M448_0": "canonical",
    "FEATHER_LLAVA_MAIN": "canonical",
    "FASTV_STACK_FULL": "fastv",
    "FASTV_MATCHED_199": "fastv",
    "NUWA_STACK_FULL": "nuwa",
    "NUWA_NATIVE_192": "nuwa",
}


class AccountingError(RuntimeError):
    """Raised when physical traces cannot support exact registered accounting."""


def atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def clip_vit_l_14_336_flops() -> int:
    """Count multiply-add as two FLOPs for CLIP ViT-L/14 at 336 pixels."""
    patches = 24 * 24
    sequence = patches + 1
    patch_embedding = 2 * patches * VISION_HIDDEN * 3 * 14 * 14
    per_block = (
        8 * sequence * VISION_HIDDEN * VISION_HIDDEN
        + 4 * sequence * sequence * VISION_HIDDEN
        + 4 * sequence * VISION_HIDDEN * VISION_INTERMEDIATE
    )
    return int(patch_embedding + 24 * per_block)


def projector_flops(tokens: int) -> int:
    """Two-layer LLaVA MLP projector, 1024->4096->4096, two FLOPs per MAC."""
    return int(
        2 * tokens * VISION_HIDDEN * HIDDEN
        + 2 * tokens * HIDDEN * HIDDEN
    )


def query_similarity_flops(visual_tokens: int, query_tokens: int, hidden: int = HIDDEN) -> int:
    """L2 normalization, dense similarities, and z-scoring arithmetic."""
    if min(visual_tokens, query_tokens, hidden) <= 0:
        raise ValueError("Query-similarity dimensions must be positive")
    normalization = 4 * (visual_tokens + query_tokens) * hidden
    similarities = 2 * visual_tokens * query_tokens * hidden
    zscore = 4 * visual_tokens
    return int(normalization + similarities + zscore)


def feather_score_flops(sequence_tokens: int, visual_tokens: int) -> int:
    """Extra RoPE-free LN, Q/K projections, QK scores, softmax, and head mean."""
    layer_norm = 5 * sequence_tokens * HIDDEN
    query_and_key_projections = 4 * sequence_tokens * HIDDEN * HIDDEN
    query_key_scores = 2 * visual_tokens * HIDDEN
    softmax = 5 * HEADS * visual_tokens
    head_mean = HEADS * visual_tokens
    return int(layer_norm + query_and_key_projections + query_key_scores + softmax + head_mean)


def m448_merge_flops(incoming: int = 576, retained: int = 448, hidden: int = HIDDEN) -> tuple[int, int]:
    """Nearest-raster distance arithmetic and mass-preserving weighted reduction."""
    raster_distance = 5 * incoming * retained
    weighted_reduction = 2 * incoming * hidden + retained * hidden
    return int(raster_distance), int(weighted_reduction)


def nuwa_metric_flops(text_tokens: int) -> tuple[int, int, int]:
    """Official Nüwa NT192 stage-1 graph/aggregation and stage-2 score arithmetic."""
    patches, retained, vision_dim = 576, 336, 1024
    layer_average_and_normalization = 9 * (patches + 1) * vision_dim + 4 * patches * vision_dim
    similarity = 2 * patches * patches * vision_dim
    score = layer_average_and_normalization + similarity + HEADS * (text_tokens + retained) ** 2
    distance_penalty = patches * patches
    merge = 2 * retained * patches * vision_dim + 3 * retained * patches
    return int(score), int(distance_penalty), int(merge)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != ROWS:
        raise AccountingError(f"Expected {ROWS} timing rows: {path}")
    return rows


def first_repetition_rows(stack: str) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for repetition, _, method, run_dir, _ in plans(stack, "natural"):
        if repetition != 1:
            continue
        filename = "predictions.jsonl" if stack == "nuwa" else "timing.jsonl"
        output[method] = read_jsonl(run_dir / filename)
    return output


def physical_summary() -> dict[tuple[str, str], dict[str, Any]]:
    rows = json.loads((ANALYSIS / "TIMING_NATURAL_SUMMARY.json").read_text(encoding="utf-8"))
    return {(str(row["stack"]), str(row["method"])): row for row in rows}


def stat_mean(summary: dict[str, Any], field: str) -> float | None:
    value = summary[field]
    return float(value["mean"]) if int(value.get("available_rows", 0)) else None


def text_tokens_from_prompt_kv(row: dict[str, Any]) -> int | None:
    prompt_kv = int(row.get("prompt_kv_total_bytes") or 0)
    if prompt_kv <= 0:
        return None
    work = int(row["visual_token_layer_work"])
    units = prompt_kv / (2 * HIDDEN * 2)
    text = (units - work) / 32
    if not float(text).is_integer() or text < 0:
        raise AccountingError("Physical prompt KV does not imply an integral text-token count")
    return int(text)


def query_token_counts() -> dict[str, int]:
    path = REPO_ROOT / (
        "experiments/final_selector_ablation_and_repair_20260811/runs/"
        "repair_val_k12_b48_single_layer_more_context/per_sample_results.jsonl"
    )
    counts = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            counts[str(row["sample_id"])] = len(row["ug_query_token_positions"])
    return counts


def pruning_flops(
    method: str,
    trace: list[int],
    *,
    text_tokens: int,
    query_tokens: int,
) -> dict[str, int]:
    score = distance = merge = gather = 0
    ranking_elements = gathered_elements = 0
    if method == "K12":
        score = query_similarity_flops(576, query_tokens)
        ranking_elements = 576
        gathered_elements = (text_tokens + 48) * HIDDEN
    elif method in {"M448_OPT", "Z_M448_0"}:
        score = query_similarity_flops(576, query_tokens) + query_similarity_flops(448, query_tokens)
        distance, merge = m448_merge_flops()
        ranking_elements = 576 + 448
        gathered_elements = ((text_tokens + 448) + (text_tokens + 48)) * HIDDEN
    elif method == "FEATHER_LLAVA_MAIN":
        score = feather_score_flops(text_tokens + 576, 576)
        score += feather_score_flops(text_tokens + trace[8], trace[8])
        ranking_elements = 576 + trace[8]
        gathered_elements = ((text_tokens + trace[8]) + (text_tokens + trace[16])) * HIDDEN
    elif method == "FASTV_MATCHED_199":
        score = HEADS * 576
        ranking_elements = 576
        gathered_elements = (text_tokens + 160) * HIDDEN
    elif method == "NUWA_NATIVE_192":
        score, distance, merge = nuwa_metric_flops(text_tokens)
        ranking_elements = 576 + 336
        gathered_elements = (text_tokens + 48) * HIDDEN
    return {
        "score_flops": int(score),
        "distance_flops": int(distance),
        "merge_flops": int(merge),
        "gather_flops": int(gather),
        "ranking_elements_not_flops": int(ranking_elements),
        "gathered_elements_not_flops": int(gathered_elements),
    }


def main() -> int:
    stack_rows = {stack: first_repetition_rows(stack) for stack in ("canonical", "fastv", "nuwa")}
    canonical_full_text = {
        str(row["sample_id"]): text_tokens_from_prompt_kv(row)
        for row in stack_rows["canonical"]["Full"]
    }
    if any(value is None for value in canonical_full_text.values()):
        raise AccountingError("Canonical Full did not expose physical prompt KV")
    query_counts = query_token_counts()
    summaries = physical_summary()
    records = []
    detail = {}
    for method, stack in METHOD_STACK.items():
        rows = stack_rows[stack][method]
        sample_accounts = []
        for row in rows:
            trace = [int(value) for value in row["realized_visual_tokens_by_layer"]]
            text_tokens = text_tokens_from_prompt_kv(row)
            if text_tokens is None:
                text_tokens = int(canonical_full_text[str(row["sample_id"])])
            query_tokens = int(query_counts.get(str(row["sample_id"]), 1))
            pruning = pruning_flops(
                method,
                trace,
                text_tokens=text_tokens,
                query_tokens=query_tokens,
            )
            pruning_total = sum(pruning[key] for key in (
                "score_flops", "distance_flops", "merge_flops", "gather_flops"
            ))
            decoder = account_trace(
                trace,
                text_tokens=text_tokens,
                pruning_metric_flops=pruning_total,
            )
            projector_tokens = 336 if method == "NUWA_NATIVE_192" else 576
            vision = clip_vit_l_14_336_flops()
            projector = projector_flops(projector_tokens)
            generated = int(row["generated_token_count"])
            analytical_final_kv = decoder.analytical_prompt_kv_bytes + (
                2 * max(generated - 1, 0) * HIDDEN * 2 * 32
            )
            sample_accounts.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "text_tokens": text_tokens,
                    "query_tokens": query_tokens,
                    "trace": trace,
                    **decoder.as_dict(),
                    **pruning,
                    "vision_encoder_flops": vision,
                    "projector_flops": projector,
                    "modeled_total_multimodal_prefill_flops": (
                        decoder.modeled_total_prefill_flops + vision + projector
                    ),
                    "analytical_final_kv_bytes": analytical_final_kv,
                }
            )
        timing = summaries[(stack, method)]
        mean = lambda key: float(np.mean([float(row[key]) for row in sample_accounts]))
        mean_trace = np.mean(
            np.asarray([row["trace"] for row in sample_accounts], dtype=np.float64), axis=0
        ).tolist()
        record = {
            "method": method,
            "stack": stack,
            "mean_final_visual_tokens": mean_trace[-1],
            "mean_visual_token_layer_work": mean("visual_token_layer_work"),
            "mean_average_visual_tokens": mean("average_visual_tokens"),
            "mean_text_tokens": mean("text_tokens"),
            "mean_total_sequence_length_by_layer": [
                float(value + mean("text_tokens")) for value in mean_trace
            ],
            "mean_attention_flops": mean("attention_flops"),
            "mean_linear_and_mlp_flops": mean("linear_and_mlp_flops"),
            "mean_llm_prefill_flops": mean("llm_prefill_flops"),
            "mean_score_flops": mean("score_flops"),
            "mean_distance_flops": mean("distance_flops"),
            "mean_merge_flops": mean("merge_flops"),
            "mean_gather_flops": mean("gather_flops"),
            "mean_ranking_elements_not_flops": mean("ranking_elements_not_flops"),
            "mean_gathered_elements_not_flops": mean("gathered_elements_not_flops"),
            "vision_encoder_flops": clip_vit_l_14_336_flops(),
            "mean_projector_flops": mean("projector_flops"),
            "mean_modeled_total_multimodal_prefill_flops": mean(
                "modeled_total_multimodal_prefill_flops"
            ),
            "mean_analytical_prompt_kv_bytes": mean("analytical_prompt_kv_bytes"),
            "mean_physical_prompt_kv_bytes": stat_mean(timing, "prompt_kv_total_bytes"),
            "mean_analytical_final_kv_bytes": mean("analytical_final_kv_bytes"),
            "mean_physical_final_kv_bytes": stat_mean(timing, "final_kv_total_bytes"),
            "mean_peak_allocated_gpu_bytes": stat_mean(timing, "peak_allocated_gpu_bytes"),
            "mean_peak_reserved_gpu_bytes": stat_mean(timing, "peak_reserved_gpu_bytes"),
            "physical_kv_scope": (
                "cache_disabled_by_official_source"
                if stack == "fastv"
                else "instrumented_tensor_storage"
            ),
        }
        records.append(record)
        detail[method] = sample_accounts
    scope = {
        "multiply_add_convention": "two FLOPs per multiply-accumulate",
        "llm_formula": "per layer: attention=8*n*d^2+4*n^2*d; gated_MLP=6*n*d*m",
        "vision_formula": "CLIP ViT-L/14@336 patch convolution plus 24 attention/MLP blocks",
        "projector_formula": "two-layer 1024->4096->4096 MLP over realized projector tokens",
        "selector_scope": (
            "Dense arithmetic explicitly executed by query similarity, FEATHER RoPE-free Q/K scoring, "
            "M448 raster merge, and official Nüwa graph/aggregation. Ranking comparisons and gather "
            "memory movement are reported as element counts, not mislabeled as FLOPs."
        ),
        "timing_separation": "FLOPs, token work, KV storage, peak memory, and measured latency are distinct",
    }
    atomic_json(ANALYSIS / "FLOP_KV_ACCOUNTING.json", {"scope": scope, "methods": records})
    atomic_csv(ANALYSIS / "FLOP_KV_ACCOUNTING.csv", records)
    atomic_json(ANALYSIS / "FLOP_KV_PER_SAMPLE_AUDIT.json", detail)
    print(json.dumps({"status": "complete", "methods": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
