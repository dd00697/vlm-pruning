#!/usr/bin/env python
"""Audit the frozen 20-row parity gates for the final validation campaign."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CAMPAIGN_ROOT = REPO_ROOT / "experiments/final_val_efficiency_baselines_20260818"
MANIFEST = (
    REPO_ROOT
    / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
)
CANONICAL_MODEL = Path(
    "/workspace/cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/"
    "snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
)
NATIVE_CLIP = Path(
    "/workspace/cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/"
    "snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
)
LLAVA_ROOT = Path("/workspace/third_party_official/LLaVA")
FEATHER_SOURCE = Path(
    "/workspace/third_party_official/FEATHER/prismatic-vlms/prismatic/models/"
    "backbones/llm/llama2_models.py"
)

CANONICAL_SMOKES = {
    "Full": CAMPAIGN_ROOT / "timing/canonical/smokes/rep1_pos1_full_attempt3/timing.jsonl",
    "K12": CAMPAIGN_ROOT / "timing/canonical/smokes/rep1_pos2_k12_attempt2/timing.jsonl",
    "M448_OPT": CAMPAIGN_ROOT / "timing/canonical/smokes/rep1_pos3_m448_opt_attempt2/timing.jsonl",
    "Z_M448_0": CAMPAIGN_ROOT / "timing/canonical/smokes/rep1_pos4_z_m448_0_attempt2/timing.jsonl",
    "FEATHER_LLAVA_MAIN": (
        CAMPAIGN_ROOT
        / "timing/canonical/smokes/rep1_pos5_feather_llava_main_attempt2/timing.jsonl"
    ),
}
HISTORICAL = {
    "Full": REPO_ROOT / (
        "experiments/runs_official_baselines/"
        "official_fullmatrix_refcoco_full_tokens576_seed0_bf16/per_sample_results.jsonl"
    ),
    "K12": REPO_ROOT / (
        "experiments/final_selector_ablation_and_repair_20260811/runs/"
        "repair_val_k12_b48_single_layer_more_context/per_sample_results.jsonl"
    ),
    "M448_OPT": REPO_ROOT / (
        "experiments/recent_methods_full_refcoco_val_20260817/runs/"
        "v4_m448_attempt1/predictions.jsonl"
    ),
    "Z_M448_0": REPO_ROOT / (
        "experiments/recent_methods_full_refcoco_val_20260817/runs/"
        "v5_z_m448_0_attempt1/predictions.jsonl"
    ),
}
FASTV_FULL = (
    CAMPAIGN_ROOT
    / "timing/fastv_native/smokes/rep1_pos1_fastv_stack_full_attempt3/timing.jsonl"
)
FASTV_ACTIVE = (
    CAMPAIGN_ROOT
    / "timing/fastv_native/smokes/rep1_pos2_fastv_matched_199_attempt2/timing.jsonl"
)
NUWA_FULL = CAMPAIGN_ROOT / "smokes/nuwa_stack_full_smoke20_attempt1/predictions.jsonl"
NUWA_ACTIVE = CAMPAIGN_ROOT / "smokes/nuwa_native_192_smoke20_attempt1/predictions.jsonl"


class ParityAuditError(RuntimeError):
    """Raised when a required smoke artifact or invariant is missing."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ParityAuditError(f"Missing parity input: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ParityAuditError(f"Empty parity input: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_output(text: str, sample: dict[str, Any]) -> dict[str, Any]:
    from scripts.run_baseline import parse_box

    raw, box, failure = parse_box(
        text, int(sample["width"]), int(sample["height"]), "normalized_0_1"
    )
    return {"raw_box": raw, "pixel_box": box, "failure": failure}


def maximum_box_delta(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None:
        return None
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def compare_stack_rows(
    canonical: list[dict[str, Any]],
    other: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(canonical) != 20 or len(other) != 20 or len(manifest) < 20:
        raise ParityAuditError("Stack parity requires exactly 20 measured rows")
    if [row["sample_id"] for row in canonical] != [row["sample_id"] for row in other]:
        raise ParityAuditError("Stack parity IDs/order differ")
    records = []
    for base, candidate, sample in zip(canonical, other, manifest[:20]):
        base_parsed = parse_output(str(base["raw_model_output"]), sample)
        candidate_parsed = parse_output(str(candidate["raw_model_output"]), sample)
        records.append(
            {
                "sample_id": base["sample_id"],
                "raw_output_exact": base["raw_model_output"] == candidate["raw_model_output"],
                "canonical_raw_output": base["raw_model_output"],
                "other_raw_output": candidate["raw_model_output"],
                "canonical_parsed_box": base_parsed["pixel_box"],
                "other_parsed_box": candidate_parsed["pixel_box"],
                "parsed_box_exact": base_parsed["pixel_box"] == candidate_parsed["pixel_box"],
                "parsed_box_max_abs_delta_px": maximum_box_delta(
                    base_parsed["pixel_box"], candidate_parsed["pixel_box"]
                ),
                "generated_token_counts": [
                    int(base["generated_token_count"]), int(candidate["generated_token_count"])
                ],
            }
        )
    exact_output = sum(bool(row["raw_output_exact"]) for row in records)
    exact_box = sum(bool(row["parsed_box_exact"]) for row in records)
    finite_deltas = [
        float(row["parsed_box_max_abs_delta_px"])
        for row in records
        if row["parsed_box_max_abs_delta_px"] is not None
    ]
    summary = {
        "rows": 20,
        "ordered_ids_exact": True,
        "raw_output_exact_rows": exact_output,
        "raw_output_exact_rate": exact_output / 20,
        "parsed_box_exact_rows": exact_box,
        "parsed_box_exact_rate": exact_box / 20,
        "maximum_parsed_box_delta_px": max(finite_deltas) if finite_deltas else None,
        "exact_output_parity": exact_output == 20,
    }
    return summary, records


def historical_parity(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    original_ids = {
        row["sample_id"]: row.get("metadata", {}).get("original_sample_id") for row in manifest[:20]
    }
    expected_traces = {
        "Full": [576] * 32,
        "K12": [576] * 12 + [48] * 20,
        "M448_OPT": [576] * 4 + [448] * 8 + [48] * 20,
        "Z_M448_0": [576] * 4 + [448] * 8 + [48] * 8 + [0] * 12,
    }
    result: dict[str, Any] = {}
    for method in expected_traces:
        current = read_jsonl(CANONICAL_SMOKES[method])
        historical = read_jsonl(HISTORICAL[method])
        historical_by_id = {row["sample_id"]: row for row in historical}
        raw_exact = 0
        retained_exact: list[bool] = []
        aligned = 0
        for row in current:
            historical_id = original_ids[row["sample_id"]] if method == "Full" else row["sample_id"]
            prior = historical_by_id.get(historical_id)
            if prior is None:
                continue
            aligned += 1
            raw_exact += int(row["raw_model_output"] == prior["raw_model_output"])
            old_indices = prior.get("retained_indices", prior.get("final_retained_indices"))
            if old_indices is not None and row.get("retained_indices") is not None:
                retained_exact.append(list(row["retained_indices"]) == list(old_indices))
        trace_ok = all(row["realized_visual_tokens_by_layer"] == expected_traces[method] for row in current)
        if aligned != 20 or not trace_ok:
            raise ParityAuditError(f"Historical parity alignment/trace failed for {method}")
        result[method] = {
            "aligned_rows": aligned,
            "exact_trace_rows": sum(
                row["realized_visual_tokens_by_layer"] == expected_traces[method] for row in current
            ),
            "raw_output_exact_rows": raw_exact,
            "raw_output_exact_rate": raw_exact / aligned,
            "retained_indices_compared_rows": len(retained_exact),
            "retained_indices_exact_rows": sum(retained_exact),
            "semantic_status": (
                "PASS_EXACT_OUTPUT_AND_TRACE"
                if raw_exact == aligned
                else "PASS_FROZEN_CONFIG_AND_TRACE_WITH_OUTPUT_DRIFT_RECORDED"
            ),
            "historical_path": str(HISTORICAL[method].relative_to(REPO_ROOT)),
            "historical_sha256": sha256_file(HISTORICAL[method]),
        }
    return result


def tensor_hash(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def image_preprocessing_parity(manifest: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    from transformers import AutoProcessor, CLIPImageProcessor

    if str(LLAVA_ROOT) not in sys.path:
        sys.path.insert(0, str(LLAVA_ROOT))
    from llava.mm_utils import process_images

    canonical = AutoProcessor.from_pretrained(CANONICAL_MODEL, local_files_only=True)
    native = CLIPImageProcessor.from_pretrained(NATIVE_CLIP, local_files_only=True)
    native_config = SimpleNamespace(image_aspect_ratio="pad")
    rows = []
    for sample in manifest[:20]:
        with Image.open(REPO_ROOT / sample["image_path"]) as opened:
            image = opened.convert("RGB")
        canonical_tensor = canonical.image_processor(images=[image], return_tensors="pt")[
            "pixel_values"
        ][0]
        native_tensor = process_images([image], native, native_config)[0]
        if canonical_tensor.shape != native_tensor.shape:
            raise ParityAuditError("Canonical and native Nüwa image tensor shapes differ")
        delta = (canonical_tensor.float() - native_tensor.float()).abs()
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "original_image_size": list(image.size),
                "tensor_shape": list(canonical_tensor.shape),
                "canonical_sha256": tensor_hash(canonical_tensor),
                "nuwa_native_sha256": tensor_hash(native_tensor),
                "exact": bool(torch.equal(canonical_tensor, native_tensor)),
                "max_abs_delta": float(delta.max().item()),
                "mean_abs_delta": float(delta.mean().item()),
            }
        )
    summary = {
        "rows": len(rows),
        "exact_rows": sum(bool(row["exact"]) for row in rows),
        "maximum_abs_delta": max(float(row["max_abs_delta"]) for row in rows),
        "mean_of_mean_abs_delta": sum(float(row["mean_abs_delta"]) for row in rows) / len(rows),
        "canonical_policy": "HF LLaVA CLIP shortest-edge resize plus center crop",
        "nuwa_native_policy": "official LLaVA image_aspect_ratio=pad then pinned CLIP preprocessing",
        "official_process_images_source": str(LLAVA_ROOT / "llava/mm_utils.py"),
        "official_process_images_sha256": sha256_file(LLAVA_ROOT / "llava/mm_utils.py"),
    }
    return summary, rows


def feather_fixture_audit() -> dict[str, Any]:
    import torch

    from vlm_pruning.llava_hooks.feather_pruning import (
        FEATHER_KEEP_RATIO,
        FEATHER_STRIDE,
        feather_select_original_indices,
    )

    def upstream(scores: Any, incoming: list[int], stage: int) -> list[int]:
        if stage == 0:
            ranked = scores.topk(int(len(incoming) * FEATHER_KEEP_RATIO)).indices
            size = int(math.sqrt(len(incoming)))
            grid = torch.arange(len(incoming)).reshape(size, size)
            uniform = grid[::FEATHER_STRIDE, ::FEATHER_STRIDE].flatten()
            local = torch.unique(torch.cat((ranked, uniform)), sorted=True)
        else:
            local = scores.topk(
                int(len(incoming) * FEATHER_KEEP_RATIO * FEATHER_KEEP_RATIO)
            ).indices.sort().values
        return [incoming[int(index)] for index in local.tolist()]

    generator = torch.Generator().manual_seed(20260818)
    first_scores = torch.rand(576, generator=generator)
    first, first_meta = feather_select_original_indices(
        torch, first_scores, list(range(576)), stage_index=0
    )
    first_upstream = upstream(first_scores, list(range(576)), 0)
    second_scores = torch.rand(len(first), generator=generator)
    second, second_meta = feather_select_original_indices(
        torch, second_scores, first, stage_index=1
    )
    second_upstream = upstream(second_scores, first, 1)
    source = FEATHER_SOURCE.read_text(encoding="utf-8")
    fragments = (
        "self.fastV_ratio * self.fastV_ratio",
        "torch.unique(combined_stride_fast_indices, sorted=True",
        "fastv_predefined_mask.replace('no_rope_s=', '')",
        "text_query_states = query_states[:,:,-1:,:]",
    )
    exact = first == first_upstream and second == second_upstream
    if not exact or not all(fragment in source for fragment in fragments):
        raise ParityAuditError("FEATHER source/fixture parity failed")
    return {
        "status": "PASS_EXACT_BOTH_BOUNDARIES",
        "seed": 20260818,
        "boundary_8_realized_tokens": len(first),
        "boundary_16_realized_tokens": len(second),
        "boundary_8_metadata": first_meta,
        "boundary_16_metadata": second_meta,
        "official_source": str(FEATHER_SOURCE),
        "official_source_sha256": sha256_file(FEATHER_SOURCE),
        "required_source_fragments_present": True,
    }


def main() -> int:
    manifest = read_jsonl(MANIFEST)
    canonical = read_jsonl(CANONICAL_SMOKES["Full"])
    fastv_summary, fastv_rows = compare_stack_rows(canonical, read_jsonl(FASTV_FULL), manifest)
    nuwa_summary, nuwa_rows = compare_stack_rows(canonical, read_jsonl(NUWA_FULL), manifest)
    image_summary, image_rows = image_preprocessing_parity(manifest)
    fastv_summary.update(
        {
            "classification": "WITHIN_FASTV_STACK_ONLY_OUTPUT_PARITY_FAILED",
            "active_trace_pass": all(
                row["realized_visual_tokens_by_layer"] == [576] * 3 + [160] * 29
                for row in read_jsonl(FASTV_ACTIVE)
            ),
        }
    )
    nuwa_summary.update(
        {
            "classification": "WITHIN_NUWA_STACK_ONLY_OUTPUT_AND_PREPROCESSING_PARITY_FAILED",
            "active_trace_pass": all(
                row["realized_visual_tokens_by_layer"] == [336] * 16 + [48] * 16
                for row in read_jsonl(NUWA_ACTIVE)
            ),
            "image_preprocessing": image_summary,
        }
    )
    if not fastv_summary["active_trace_pass"] or not nuwa_summary["active_trace_pass"]:
        raise ParityAuditError("An external active-path physical trace failed")
    result = {
        "schema_version": 1,
        "status": "PASS_WITH_REGISTERED_STACK_SEPARATION",
        "scope": "first 20 ordered rows of frozen RefCOCO validation timing manifest",
        "historical_internal": historical_parity(manifest),
        "fastv_disabled_vs_canonical_full": fastv_summary,
        "nuwa_stack_full_vs_canonical_full": nuwa_summary,
        "feather_upstream_fixture": feather_fixture_audit(),
        "presentation_rule": (
            "FastV and Nüwa raw latency/quality are reported only against their source-native Full "
            "controls; normalized within-stack effects may be compared across stacks."
        ),
        "inputs": {
            "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
            "manifest_sha256": sha256_file(MANIFEST),
            "canonical_smokes": {key: str(value.relative_to(REPO_ROOT)) for key, value in CANONICAL_SMOKES.items()},
            "fastv_full": str(FASTV_FULL.relative_to(REPO_ROOT)),
            "nuwa_full": str(NUWA_FULL.relative_to(REPO_ROOT)),
        },
    }
    analysis = CAMPAIGN_ROOT / "analysis"
    atomic_json(analysis / "PARITY_AUDIT.json", result)
    atomic_json(
        analysis / "PARITY_ROW_DIAGNOSTICS.json",
        {"fastv": fastv_rows, "nuwa": nuwa_rows, "nuwa_image_preprocessing": image_rows},
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
