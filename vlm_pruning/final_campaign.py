"""Locked specification for the K=12/B=48 dissertation campaign.

This module contains protocol metadata only.  It deliberately contains no
observed quality values so that registration can happen before new benchmark
outcomes are inspected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CAMPAIGN_ID = "k12_b48_expanded_evaluation_20260813"
CAMPAIGN_ROOT = Path("experiments") / CAMPAIGN_ID
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MODEL_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
LMMS_EVAL_COMMIT = "75571cfe0fa36ca5ec2e6aa2bb268f9d38295af3"


@dataclass(frozen=True)
class DatasetProtocol:
    key: str
    dataset_id: str
    revision: str
    config: str | None
    split: str
    expected_rows: int
    task: str
    evaluator: str
    prompt: str
    image_dataset_config: str | None = None
    image_dataset_rows: int | None = None
    expected_manifest_rows: int | None = None
    notes: str = ""


DATASETS: tuple[DatasetProtocol, ...] = (
    DatasetProtocol(
        "gqa",
        "lmms-lab-encoder/GQA",
        "a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8",
        "testdev_balanced_instructions",
        "testdev",
        12578,
        "vqa",
        "lmms-eval gqa exact_match (ignore case and punctuation)",
        "{question}\nAnswer the question using a single word or phrase.",
        image_dataset_config="testdev_balanced_images",
        image_dataset_rows=398,
    ),
    DatasetProtocol(
        "mmbench_en_dev",
        "lmms-lab-encoder/MMBench",
        "56ba1af8954932c4804bd3f522e05ed96e63b654",
        "en",
        "dev",
        4329,
        "vqa",
        "lmms-eval mmbench_en_dev_static",
        "Official MMBench question, optional hint, labelled choices, then option-letter instruction.",
        notes="Static official extraction is locked; no API-based answer rescue and no circular augmentation.",
    ),
    DatasetProtocol(
        "mmmu_val",
        "lmms-lab-encoder/MMMU",
        "364f2e2eb107b36e07ff4c5a15f5947a759cef47",
        None,
        "validation",
        900,
        "vqa",
        "lmms-eval mmmu_acc",
        "Official MMMU prompt for multiple-choice or open-ended rows.",
        notes=(
            "All subjects and all 900 rows. Multi-image rows use the prospectively locked disjoint-canvas "
            "Q-TAC extension; results retain an image_count tag and are also reported by one/multi-image subset."
        ),
    ),
    DatasetProtocol(
        "mme",
        "lmms-lab-encoder/MME",
        "d6c9023f017b564f7b3ccccf5348166bce8fdbcd",
        None,
        "test",
        2374,
        "vqa",
        "official MME perception/cognition category scoring",
        "{question}\nAnswer with exactly one word: yes or no.",
    ),
    DatasetProtocol(
        "vqav2_val",
        "lmms-lab-encoder/VQAv2",
        "32665d35052eb4a6d4414851c3c829a72754915a",
        None,
        "validation",
        214354,
        "vqa",
        "lmms-eval EvalAI VQA soft accuracy",
        "{question}\nAnswer the question using a single word or phrase.",
    ),
    DatasetProtocol(
        "textvqa_val",
        "lmms-lab-encoder/textvqa",
        "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        None,
        "validation",
        5000,
        "vqa",
        "lmms-eval/EvalAI VQA soft accuracy",
        "{question}\nAnswer the question using a single word or phrase.",
    ),
    DatasetProtocol(
        "pope_all",
        "lmms-lab-encoder/POPE",
        "4db1276663dfa5eb8ad16a52d24c31a09e470896",
        "Full",
        "random+popular+adversarial",
        9000,
        "vqa",
        "official POPE accuracy/precision/recall/F1/yes-ratio by subset",
        "{question}\nAnswer with exactly one word: yes or no.",
    ),
    DatasetProtocol(
        "scienceqa_img_test",
        "lmms-lab-encoder/ScienceQA",
        "69dd4d6b67373d38f96a5badd5d24d0eb5bcdc50",
        "ScienceQA-IMG",
        "test",
        2017,
        "vqa",
        "lmms-eval scienceqa_img exact_match",
        "Official optional context, question, labelled choices, then option-letter instruction.",
    ),
    DatasetProtocol(
        "mmvet_test",
        "lmms-lab-encoder/MMVet",
        "b310d12d0d9e765db953da28a7ef7ff43620dc83",
        None,
        "test",
        218,
        "vqa",
        "lmms-eval MM-Vet judge, pinned model gpt-4o-2024-11-20",
        "First reason step by step, then answer the official MM-Vet question.",
        notes="Predictions are mandatory; judge scores are blocked_external_scoring if credentials are unavailable.",
    ),
    DatasetProtocol(
        "refcoco_plus_val",
        "lmms-lab-encoder/RefCOCOplus",
        "a283082593c437efc2bb9876d43bc3075cd914fe",
        None,
        "val",
        3805,
        "grounding",
        "locked box parser, Acc@0.5 and mean IoU",
        "Locked normalized-0-1 one-box grounding prompt.",
        expected_manifest_rows=10758,
        notes="Evaluate every official referring expression; 3,805 source objects expand to 10,758 scored rows.",
    ),
    DatasetProtocol(
        "refcoco_plus_testA",
        "lmms-lab-encoder/RefCOCOplus",
        "a283082593c437efc2bb9876d43bc3075cd914fe",
        None,
        "testA",
        1975,
        "grounding",
        "locked box parser, Acc@0.5 and mean IoU",
        "Locked normalized-0-1 one-box grounding prompt.",
        expected_manifest_rows=5726,
        notes="Evaluate every official referring expression; 1,975 source objects expand to 5,726 scored rows.",
    ),
    DatasetProtocol(
        "refcoco_plus_testB",
        "lmms-lab-encoder/RefCOCOplus",
        "a283082593c437efc2bb9876d43bc3075cd914fe",
        None,
        "testB",
        1798,
        "grounding",
        "locked box parser, Acc@0.5 and mean IoU",
        "Locked normalized-0-1 one-box grounding prompt.",
        expected_manifest_rows=4889,
        notes="Evaluate every official referring expression; 1,798 source objects expand to 4,889 scored rows.",
    ),
)


PRIMARY_CONFIG: dict[str, Any] = {
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "precision": "bf16",
    "seed": 0,
    "prune_after_layer": 12,
    "probe_layers": [12],
    "final_visual_tokens": 48,
    "query_policy": "query_max",
    "fusion": "single_layer",
    "selector": "structured_tca",
    "role_preset": "more_anchors",
    "role_counts": {"target": 21, "context": 12, "anchor": 15},
    "anchor_pattern": "fps",
    "position_mode": "gap_preserving",
    "decode_mode": "layer_aware_cache",
    "diagnostics_mode": "benchmark",
    "generation": "deterministic greedy",
}


MULTI_IMAGE_EXTENSION: dict[str, Any] = {
    "scope": "MMMU rows with image_count > 1 only",
    "outcome_blind": True,
    "visual_sequence": "official prompt order; 576 projected patches per image before pruning",
    "global_budget": 48,
    "query_scoring": "global query-max ranking over every image patch",
    "anchor_count": 15,
    "prune_layer": 12,
    "anchor_geometry": (
        "FPS on disjoint normalized image canvases: coordinates are (image_index + x, y); "
        "distances and context neighborhoods never wrap across image boundaries"
    ),
    "allocation": {"target": 21, "context": 12, "anchor": 15},
    "ordering": "original image order then raster order, with original gap-preserving position IDs",
    "required_reporting": ["all_900", "single_image_857", "multi_image_43"],
    "published_baseline_rule": (
        "A method unsupported for multi-image exact-total-B48 is evaluated only on the locked 857-row "
        "single-image subset and is never presented as a full MMMU-val score."
    ),
}


METHODS: tuple[dict[str, Any], ...] = (
    {
        "key": "fastv",
        "official_repo": "https://github.com/pkunlp-icler/FastV",
        "commit": "d1659729b5bf1be225e99ee15783deeea80f63b1",
        "status": "gate_pending",
        "exact_b48_plan": "fastv_k=3 and fastv_r=1-48/576 in the official HF implementation",
    },
    {
        "key": "visionzip",
        "official_repo": "https://github.com/dvlab-research/VisionZip",
        "commit": "8f86b55c6f000eb033e6912538af2dd7dcb30502",
        "status": "gate_pending",
        "exact_b48_plan": "official visionzip(model, dominant=40, contextual=8)",
    },
    {
        "key": "feather",
        "official_repo": "https://github.com/markendo/FEATHER",
        "commit": "c2a09b2765967601054c7b1fd513ac3e94ee4fc9",
        "status": "excluded_checkpoint_or_parity_mismatch",
        "reason": (
            "Official FEATHER targets the Prism/SigLIP stack and its adaptive union does not guarantee "
            "exactly 48 final tokens on canonical LLaVA-1.5-7B. Porting it would alter method semantics."
        ),
    },
    {
        "key": "nuwa",
        "display_name": "Nüwa",
        "official_repo": "https://github.com/Man-PaperRejected/Nuwa",
        "commit": "175b7d57a7890b9818dd26d82f2d91d9868b0d80",
        "status": "excluded_no_exact_b48_support",
        "reason": (
            "The official LLaVA selector exposes stage-1 configurations for 64/128/192 tokens, not 48; "
            "adding a 48-token configuration would change the official selection algorithm."
        ),
    },
    {
        "key": "transprune",
        "official_repo": "https://github.com/liaolea/TransPrune",
        "commit": "c9e8284f5808f7692c40590ffc99537e9220a01d",
        "status": "gate_pending",
        "exact_b48_plan": "official ratios ending at 1/12; code computes int(576/12)=48",
    },
)


MECHANISM_RUNS: tuple[str, ...] = (
    "full",
    "random",
    "uniform",
    "query_topk",
    "target_nms_only",
    "structured_tca",
)


MANDATORY_CORE_METHODS: tuple[str, ...] = ("full", "random", "uniform", "structured_tca")


def protocol_payload() -> dict[str, Any]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "scope": "exact final B=48 only; no Pareto or budget-sweep runs",
        "primary_config": PRIMARY_CONFIG,
        "multi_image_extension": MULTI_IMAGE_EXTENSION,
        "datasets": [asdict(item) for item in DATASETS],
        "methods": list(METHODS),
        "mechanism_runs": list(MECHANISM_RUNS),
        "mandatory_core_methods": list(MANDATORY_CORE_METHODS),
        "lmms_eval_commit": LMMS_EVAL_COMMIT,
        "reporting": {
            "primary_section_anchor_count": 12,
            "forbidden_primary_section_terms": [
                "K=20",
                "K20",
                "changed from",
                "previously K",
                "original K",
            ],
            "prospective_scope": "new benchmark suite only",
            "historical_k12_wording": "verified existing K=12 evidence",
        },
    }


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    primary = payload.get("primary_config", {})
    if primary.get("prune_after_layer") != 12:
        issues.append("primary prune layer is not K=12")
    if primary.get("final_visual_tokens") != 48:
        issues.append("primary final visual-token budget is not 48")
    if primary.get("role_counts") != {"target": 21, "context": 12, "anchor": 15}:
        issues.append("primary role allocation is not 21/12/15")
    if sum(primary.get("role_counts", {}).values()) != 48:
        issues.append("primary role allocation does not sum to 48")
    if payload.get("scope") != "exact final B=48 only; no Pareto or budget-sweep runs":
        issues.append("scope unexpectedly permits non-B48 runs")
    keys = [item.get("key") for item in payload.get("datasets", [])]
    if len(keys) != len(set(keys)):
        issues.append("dataset keys are not unique")
    for item in payload.get("datasets", []):
        if not item.get("revision") or len(item["revision"]) != 40:
            issues.append(f"dataset {item.get('key')} has no immutable 40-character revision")
        if int(item.get("expected_rows", 0)) <= 0:
            issues.append(f"dataset {item.get('key')} has no positive expected row count")
        manifest_rows = item.get("expected_manifest_rows") or item.get("expected_rows")
        if int(manifest_rows or 0) <= 0:
            issues.append(f"dataset {item.get('key')} has no positive expected manifest row count")
    if "query_topk" not in payload.get("mechanism_runs", []):
        issues.append("pure query-top-k mechanism cell is missing")
    return issues
