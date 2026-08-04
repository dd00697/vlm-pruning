#!/usr/bin/env python
"""Validate final manifests and write the pre-registered dissertation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_ROOT = REPO_ROOT / "experiments" / "final_evaluation" / "protocol"
DEFAULT_MANIFEST_ROOT = REPO_ROOT / "data" / "manifests" / "final_evaluation"
SOURCE_MANIFEST_ROOT = REPO_ROOT / "data" / "manifests" / "official_baselines"
HASHED_SOURCES = (
    "scripts/prepare_official_baseline_data.py",
    "scripts/register_final_protocol.py",
    "scripts/run_official_baseline.py",
    "scripts/run_ug_prune_single_pass.py",
    "scripts/run_final_evaluation.py",
    "scripts/finalize_selection_provenance.py",
    "scripts/freeze_final_method.py",
    "scripts/run_final_postfreeze.py",
    "scripts/run_final_timing_trial.py",
    "scripts/run_final_timing_campaign.py",
    "scripts/final_completion_audit.py",
    "scripts/run_final_verification.py",
    "vlm_pruning/llava_hooks/mid_layer_pruning.py",
    "vlm_pruning/prior_methods.py",
    "vlm_pruning/efficiency.py",
    "../Dissertation_Draft/analysis/build_final_evaluation.py",
    "../Dissertation_Draft/analysis/qa_final_pdf.py",
    "vlm_pruning/selectors/ug_prune.py",
    "tests/test_final_protocol.py",
    "third_party/FastV/src/FastV/lmms-eval/fastv_kvcache.py",
    "third_party/VisionZip/visionzip/clip_encoder.py",
)


class ProtocolError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ProtocolError(f"Missing manifest: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ProtocolError(f"Malformed JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_pope_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        category = str(row.get("category") or "").strip().lower()
        if category not in {"random", "popular", "adversarial"}:
            raise ProtocolError(f"Unsupported POPE category: {category!r}")
        original_id = str(row["sample_id"])
        rewritten = dict(row)
        rewritten["sample_id"] = f"pope_{category}_{original_id}"
        rewritten["split"] = "test"
        metadata = dict(row.get("metadata") or {})
        metadata["original_sample_id"] = original_id
        metadata["final_id_qualified_by_category"] = True
        rewritten["metadata"] = metadata
        output.append(rewritten)
    return output


def final_refcoco_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Qualify expression IDs by strict split, upstream row and expression ordinal."""
    ordinals: Counter[int] = Counter()
    output: list[dict[str, Any]] = []
    for fallback_row, row in enumerate(rows):
        metadata = dict(row.get("metadata") or {})
        source_row = int(metadata.get("source_row", fallback_row))
        expression_ordinal = ordinals[source_row]
        ordinals[source_row] += 1
        rewritten = dict(row)
        original_id = str(row["sample_id"])
        rewritten["sample_id"] = f"refcoco_{split}_{source_row:06d}_{expression_ordinal}"
        metadata["original_sample_id"] = original_id
        metadata["final_id_qualified_by_split_source_row_expression"] = True
        rewritten["metadata"] = metadata
        output.append(rewritten)
    return output


def validate_unique_ids(name: str, rows: list[dict[str, Any]]) -> None:
    ids = [str(row.get("sample_id")) for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = [item for item, count in Counter(ids).items() if count > 1][:10]
        raise ProtocolError(f"{name} contains duplicate sample IDs: {duplicates}")


def validate_images(name: str, rows: list[dict[str, Any]]) -> None:
    missing = [str(row.get("image_path")) for row in rows if not Path(str(row.get("image_path"))).exists()]
    if missing:
        raise ProtocolError(f"{name} has {len(missing)} missing images; first: {missing[0]}")


def validate_ref_split(name: str, rows: list[dict[str, Any]], expected_split: str) -> None:
    validate_unique_ids(name, rows)
    validate_images(name, rows)
    wrong = [
        row.get("metadata", {}).get("source_split")
        for row in rows
        if row.get("metadata", {}).get("source_split") != expected_split
    ]
    if wrong:
        raise ProtocolError(f"{name} does not exclusively use strict upstream split {expected_split}")


def manifest_record(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "rows": len(rows),
        "unique_sample_ids": len({str(row["sample_id"]) for row in rows}),
        "sha256": sha256(path),
    }


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def repository_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    ref_val_source_path = SOURCE_MANIFEST_ROOT / "refcoco_val.jsonl"
    ref_a_source_path = args.manifest_root / "refcoco_testA.jsonl"
    ref_b_source_path = args.manifest_root / "refcoco_testB.jsonl"
    ref_val_path = args.manifest_root / "refcoco_val_unique.jsonl"
    ref_a_path = args.manifest_root / "refcoco_testA_unique.jsonl"
    ref_b_path = args.manifest_root / "refcoco_testB_unique.jsonl"
    pope_source_path = SOURCE_MANIFEST_ROOT / "pope_val.jsonl"
    pope_path = args.manifest_root / "pope_all_unique.jsonl"
    textvqa_path = SOURCE_MANIFEST_ROOT / "textvqa_validation.jsonl"

    ref_val = final_refcoco_rows(read_jsonl(ref_val_source_path), "val")
    ref_a = final_refcoco_rows(read_jsonl(ref_a_source_path), "testA")
    ref_b = final_refcoco_rows(read_jsonl(ref_b_source_path), "testB")
    pope = final_pope_rows(read_jsonl(pope_source_path))
    textvqa = read_jsonl(textvqa_path)
    write_jsonl(ref_val_path, ref_val)
    write_jsonl(ref_a_path, ref_a)
    write_jsonl(ref_b_path, ref_b)
    write_jsonl(pope_path, pope)

    validate_ref_split("RefCOCO validation", ref_val, "val")
    validate_ref_split("RefCOCO testA", ref_a, "testA")
    validate_ref_split("RefCOCO testB", ref_b, "testB")
    for left_name, left, right_name, right in (
        ("validation", ref_val, "testA", ref_a),
        ("validation", ref_val, "testB", ref_b),
        ("testA", ref_a, "testB", ref_b),
    ):
        overlap = {str(row["sample_id"]) for row in left} & {str(row["sample_id"]) for row in right}
        if overlap:
            raise ProtocolError(f"RefCOCO {left_name}/{right_name} sample ID overlap: {len(overlap)}")

    validate_unique_ids("POPE", pope)
    validate_images("POPE", pope)
    categories = Counter(str(row.get("category") or "").lower() for row in pope)
    expected_categories = Counter({"random": 3000, "popular": 3000, "adversarial": 3000})
    if categories != expected_categories:
        raise ProtocolError(f"Unexpected POPE category counts: {dict(categories)}")

    validate_unique_ids("TextVQA", textvqa)
    validate_images("TextVQA", textvqa)
    bad_answers = sum(
        not isinstance(row.get("answer"), list) or len(row.get("answer")) != 10 for row in textvqa
    )
    if bad_answers:
        raise ProtocolError(f"TextVQA rows without ten annotator answers: {bad_answers}")

    expected_revisions = {
        REPO_ROOT / "third_party/FastV": "d1659729b5bf1be225e99ee15783deeea80f63b1",
        REPO_ROOT / "third_party/VisionZip": "8f86b55c6f000eb033e6912538af2dd7dcb30502",
    }
    for path, expected_revision in expected_revisions.items():
        actual_revision = repository_revision(path)
        if actual_revision != expected_revision:
            raise ProtocolError(
                f"Pinned source mismatch for {path}: {actual_revision!r} != {expected_revision!r}."
            )

    sources = {
        relative: sha256(REPO_ROOT / relative)
        for relative in HASHED_SOURCES
        if (REPO_ROOT / relative).exists()
    }
    commit = git_output("rev-parse", "HEAD")
    dirty = bool(git_output("status", "--short"))
    protocol = {
        "protocol_id": "msc-final-evaluation-v1",
        "registered_at_utc": utc_now(),
        "repository": {
            "path": str(REPO_ROOT),
            "git_commit": commit,
            "git_dirty": dirty,
            "git_diff_stat": git_output("diff", "--stat"),
            "source_sha256": sources,
            "published_adapter_fidelity_sha256": (
                sha256(args.protocol_root / "published_adapter_fidelity.md")
                if (args.protocol_root / "published_adapter_fidelity.md").exists()
                else None
            ),
            "selection_code_epoch_equivalence_sha256": (
                sha256(args.protocol_root / "selection_code_epoch_equivalence.json")
                if (args.protocol_root / "selection_code_epoch_equivalence.json").exists()
                else None
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "datasets": package_version("datasets"),
            "wandb": package_version("wandb"),
            "hf_home": os.environ.get("HF_HOME"),
        },
        "model": {
            "checkpoint": "llava-hf/llava-1.5-7b-hf",
            "precision": "bf16",
            "image_resolution": "336x336",
            "original_visual_tokens": 576,
            "batch_size": 1,
            "decode": "greedy",
        },
        "prompts": {
            "grounding": (
                'Identify the target described by this referring expression: "{query}".\n'
                "Return exactly one bounding box in normalized 0-1 image coordinates as "
                "[x_min, y_min, x_max, y_max]. Do not include any other text."
            ),
            "pope": "{question}\nAnswer with exactly one word: yes or no.",
            "textvqa": "{question}\nAnswer as briefly as possible.",
        },
        "max_new_tokens": {"grounding": 48, "pope": 16, "textvqa": 16},
        "manifests": {
            "refcoco_val": manifest_record(ref_val_path, ref_val),
            "refcoco_testA": manifest_record(ref_a_path, ref_a),
            "refcoco_testB": manifest_record(ref_b_path, ref_b),
            "pope": manifest_record(pope_path, pope),
            "textvqa": manifest_record(textvqa_path, textvqa),
        },
        "upstream_dataset_revisions": {
            "lmms-lab/RefCOCO": "a5dff0b3194715fda69b2d6a0d2aaafb41eaa407",
            "lmms-lab/POPE": "4db1276663dfa5eb8ad16a52d24c31a09e470896",
            "lmms-lab/textvqa": "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        },
        "selection": {
            "fixed": {
                "prune_after_layer": 20,
                "target_visual_tokens": 48,
                "fusion_mode": "single_layer",
                "position_mode": "gap_preserving",
                "decode_mode": "layer_aware_cache",
            },
            "factorial": {
                "query_scoring": ["query_max", "query_pooled"],
                "role_preset": ["more_context", "more_anchors"],
            },
            "decision_rule": [
                "highest full-manifest Acc@0.5",
                "if exactly tied, highest mIoU",
                "if exactly tied, lower selector overhead on the registered diagnostic subset",
                "if still tied, query_max plus more_context",
            ],
        },
        "held_out_methods": {
            "full": {"tokens": 576, "seeds": [0]},
            "final_method": {"tokens": 48, "seeds": [0]},
            "random": {"tokens": 48, "seeds": [0, 1, 2]},
            "uniform": {"tokens": 48, "seeds": [0]},
            "fastv": {
                "tokens": 48,
                "seeds": [0],
                "source": "pkunlp-icler/FastV",
                "revision": "d1659729b5bf1be225e99ee15783deeea80f63b1",
                "k": 3,
                "rule": "mean-head last-prompt-token attention at layer K-1; prune before K",
                "hf_adaptation": "static selected set during cached decoding with two cache banks",
            },
            "visionzip": {
                "tokens": 48,
                "seeds": [0],
                "source": "JIA-Lab-research/VisionZip",
                "revision": "8f86b55c6f000eb033e6912538af2dd7dcb30502",
                "dominant_tokens": 40,
                "dominant_composition": "1 CLIP CLS token + 39 attention-ranked patch tokens",
                "contextual_tokens": 8,
                "recipe": "official 54+10 composition at 64 tokens scaled proportionally to 48",
                "contextual_merge_metric": (
                    "mean-head raw CLIP key vectors from the selected layer, matching "
                    "the pinned implementation"
                ),
                "metric_capture": "temporary k_proj forward hook without recomputation",
            },
        },
        "metrics": {
            "refcoco": [
                "Acc@0.5",
                "mIoU with unparsable outputs scored as IoU zero",
                "parsed-box rate",
                "invalid-output rate",
            ],
            "pope": ["accuracy", "precision", "recall", "F1", "yes-answer ratio", "invalid-output rate"],
            "textvqa": ["official VQA soft accuracy", "normalized exact match", "invalid-output rate"],
            "textvqa_evaluator_reference": (
                "https://gist.github.com/ronghanghu/ec6efe1e0fb8fee65866912006711de3"
            ),
            "uncertainty": {
                "method": "paired percentile bootstrap on aligned sample IDs",
                "replicates": 10000,
                "seed": 20260729,
            },
        },
        "timing": {
            "split": "refcoco_testA",
            "subset_rows": 1000,
            "subset_seed": 20260729,
            "subset_manifest": "data/manifests/final_evaluation/refcoco_testA_timing1000_seed20260729.jsonl",
            "subset_sha256": (
                "4c2ff7787fc62ceb1c238d9d06950cb618270d4fd5b93819941bf7524b876715"
            ),
            "repetitions": 5,
            "warmup_rows": 20,
            "methods": ["full", "k12_b48", "final_k20_b48", "fastv_b48", "visionzip_b48"],
            "method_configurations": {
                "k12_b48": {
                    "prune_after_layer": 12,
                    "tokens": 48,
                    "query_scoring": "query_max",
                    "role_allocation": "more_context",
                    "fusion": "single_layer",
                    "probe_layer": 12,
                    "positions": "gap_preserving",
                    "decode": "layer_aware_cache",
                },
                "final_k20_b48": "exact configuration in frozen_method.json",
                "fastv_b48": "pinned K=3 adapter",
                "visionzip_b48": "pinned pre-LLM 40-dominant plus 8-contextual adapter",
            },
            "boundaries": {
                "vision_encoder": "synchronised CLIP vision-tower forward only",
                "projector": "synchronised multimodal projector forward only",
                "selector": "selection/map computation; flagged when nested within LLM prefill",
                "llm_prefill": "synchronised first language-model forward through first generated token",
                "decode": "synchronised cached decoding after the first generated token",
                "end_to_end": "after image I/O and prompt creation, through completed decoding",
                "host_remainder": "end-to-end minus non-overlapping measured components",
            },
            "component_reconciliation_gate": (
                "mean and p95 absolute remainder must not exceed both 0.10 seconds "
                "and 15% of end-to-end latency"
            ),
            "orders": [
                ["full", "k12_b48", "final_k20_b48", "fastv_b48", "visionzip_b48"],
                ["k12_b48", "final_k20_b48", "fastv_b48", "visionzip_b48", "full"],
                ["final_k20_b48", "fastv_b48", "visionzip_b48", "full", "k12_b48"],
                ["fastv_b48", "visionzip_b48", "full", "k12_b48", "final_k20_b48"],
                ["visionzip_b48", "full", "k12_b48", "final_k20_b48", "fastv_b48"],
            ],
        },
        "error_analysis": {
            "object_area_pixels": {"small": "<32^2", "medium": "32^2 to <96^2", "large": ">=96^2"},
            "expression_words": {"short": "1-3", "medium": "4-6", "long": "7+"},
            "keyword_groups": {
                "horizontal": ["left", "right"],
                "vertical": ["top", "bottom", "above", "below", "under"],
                "depth_order": ["front", "behind", "back"],
                "relational": ["next to", "beside", "between", "near", "with", "holding", "wearing"],
            },
            "box_region": "normalised centre in a fixed 3x3 grid",
            "border_proximity": "any ground-truth side within 10% of the image edge",
            "qualitative_rule": (
                "recorded largest IoU losses, largest gains, parsing failures and representative "
                "successes balanced across testA/testB and object-size bands"
            ),
        },
    }
    write_json(args.protocol_root / "protocol.json", protocol)
    write_json(
        args.protocol_root / "manifest_audit.json",
        {
            "registered_at_utc": protocol["registered_at_utc"],
            "manifests": protocol["manifests"],
            "pope_category_counts": dict(categories),
            "textvqa_ten_answer_rows": len(textvqa),
            "refcoco_split_id_overlap": 0,
        },
    )
    print(json.dumps(protocol["manifests"], indent=2, sort_keys=True))
    print(f"Wrote protocol to {args.protocol_root / 'protocol.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
