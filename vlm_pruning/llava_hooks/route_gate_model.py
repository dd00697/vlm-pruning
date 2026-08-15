"""Frozen tiny route-gate model and calibration utilities.

This module deliberately contains no LLaVA code.  It defines the one allowed
gate architecture and the train-only threshold rule used by both training and
physical validation inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

from vlm_pruning.llava_hooks.route_gate import FEATURE_DIMENSION, FEATURE_SCHEMA_SHA256


TRAINING_SPEC: dict[str, Any] = {
    "schema_version": 1,
    "feature_dimension": FEATURE_DIMENSION,
    "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
    "architecture": {
        "kind": "two_head_mlp",
        "hidden_width": 64,
        "activation": "GELU",
        "dropout_probability": 0.1,
        "outputs": ["safe_route_logit", "safe_minus_aggressive_iou"],
        "trainable_parameters": 525570,
        "maximum_trainable_parameters": 1000000,
    },
    "normalization": {
        "kind": "per_feature_zscore",
        "fit_partition": "fit",
        "fit_rows": 2160,
        "variance_denominator": "population",
        "minimum_standard_deviation": 1e-6,
        "stored_dtype": "float32",
    },
    "partitions": {"fit": 2160, "early_stop": 240, "calibration": 600},
    "objective": {
        "feature_and_model_dtype": "float32",
        "route_iou_comparison_source": "stored_float64_json_values",
        "regression_target_dtype": "float32",
        "classification_target": "safe_iou_strictly_greater_than_aggressive_iou",
        "classification_loss": "binary_cross_entropy_with_logits_mean_after_example_weight",
        "ordinary_example_weight": 1.0,
        "critical_safe_correct_aggressive_wrong_weight": 5.0,
        "regression_target": "safe_iou_minus_aggressive_iou",
        "regression_loss": "smooth_l1_mean",
        "smooth_l1_beta": 1.0,
        "regression_loss_weight": 0.5,
    },
    "optimizer": {
        "name": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 64,
    },
    "schedule": {"name": "CosineAnnealingLR", "t_max_epochs": 50, "eta_min": 0.0},
    "training": {
        "maximum_epochs": 50,
        "early_stopping_monitor": "early_stop_total_loss",
        "early_stopping_patience_epochs": 8,
        "minimum_improvement": 0.0,
        "weight_initialization": "PyTorch Linear defaults after torch.manual_seed(training_seed)",
        "batch_order": "epochwise torch.randperm from a generator seeded training_seed_plus_1000",
        "primary_seed": 0,
        "required_seeds": [0, 1, 2],
        "deterministic_cpu": True,
    },
    "calibration": {
        "partition": "calibration",
        "rows": 600,
        "safe_choice_rule": "probability_safe_greater_than_or_equal_to_threshold",
        "objective": "minimum_expected_average_visual_tokens",
        "maximum_acc_at_0_5_loss_vs_always_safe": 0.005,
        "maximum_mean_iou_loss_vs_always_safe": 0.005,
        "fallback": "always_safe_if_no_positive_aggressive_rate_is_feasible",
    },
    "route_tokens": {"K12_QUERY_TOPK": 246.0, "M448": 214.0},
}


def canonical_checksum(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


TRAINING_SPEC_SHA256 = canonical_checksum(TRAINING_SPEC)


def normalization_sha256(mean: Any, standard_deviation: Any) -> str:
    """Hash the schema and exact float32 normalization tensors."""

    import numpy as np

    mean_array = np.asarray(mean, dtype=np.float32).reshape(-1)
    std_array = np.asarray(standard_deviation, dtype=np.float32).reshape(-1)
    if mean_array.shape != (FEATURE_DIMENSION,) or std_array.shape != (FEATURE_DIMENSION,):
        raise ValueError("Route-gate normalization has the wrong dimension")
    digest = hashlib.sha256()
    digest.update(b"adaptive-route-gate-normalization-v1\0")
    digest.update(FEATURE_SCHEMA_SHA256.encode("ascii"))
    digest.update(mean_array.tobytes())
    digest.update(std_array.tobytes())
    return digest.hexdigest()


def build_gate_model(torch: Any) -> Any:
    """Construct the only registered route-gate architecture."""

    class TinyRouteGate(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = torch.nn.Linear(FEATURE_DIMENSION, 64)
            self.activation = torch.nn.GELU()
            self.dropout = torch.nn.Dropout(p=0.1)
            self.outputs = torch.nn.Linear(64, 2)

        def forward(self, features: Any) -> Any:
            return self.outputs(self.dropout(self.activation(self.input(features))))

    model = TinyRouteGate()
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameters != TRAINING_SPEC["architecture"]["trainable_parameters"]:
        raise RuntimeError(f"Frozen route-gate parameter count changed: {parameters}")
    return model


def selected_metrics(
    safe_iou: Sequence[float],
    aggressive_iou: Sequence[float],
    safe_valid: Sequence[bool],
    aggressive_valid: Sequence[bool],
    choose_safe: Sequence[bool],
) -> dict[str, float | int]:
    if not (
        len(safe_iou)
        == len(aggressive_iou)
        == len(safe_valid)
        == len(aggressive_valid)
        == len(choose_safe)
    ):
        raise ValueError("Route metric arrays are not aligned")
    count = len(choose_safe)
    if count == 0:
        raise ValueError("Cannot score an empty route mixture")
    chosen_iou = [float(left if safe else right) for left, right, safe in zip(safe_iou, aggressive_iou, choose_safe, strict=True)]
    chosen_valid = [bool(left if safe else right) for left, right, safe in zip(safe_valid, aggressive_valid, choose_safe, strict=True)]
    safe_count = sum(bool(value) for value in choose_safe)
    return {
        "rows": count,
        "acc_at_0_5": sum(value >= 0.5 for value in chosen_iou) / count,
        "mean_iou": sum(chosen_iou) / count,
        "parse_rate": sum(chosen_valid) / count,
        "safe_rows": safe_count,
        "aggressive_rows": count - safe_count,
        "safe_rate": safe_count / count,
        "aggressive_rate": (count - safe_count) / count,
        "average_visual_tokens": (
            safe_count * TRAINING_SPEC["route_tokens"]["K12_QUERY_TOPK"]
            + (count - safe_count) * TRAINING_SPEC["route_tokens"]["M448"]
        )
        / count,
    }


def calibrate_threshold(
    probabilities_safe: Sequence[float],
    safe_iou: Sequence[float],
    aggressive_iou: Sequence[float],
    safe_valid: Sequence[bool],
    aggressive_valid: Sequence[bool],
) -> dict[str, Any]:
    """Apply the frozen constrained threshold search on calibration only."""

    if len(probabilities_safe) != 600:
        raise ValueError("Threshold calibration requires exactly 600 records")
    probabilities = [float(value) for value in probabilities_safe]
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError("Route-gate probabilities must be finite values in [0, 1]")
    always_safe = selected_metrics(
        safe_iou,
        aggressive_iou,
        safe_valid,
        aggressive_valid,
        [True] * len(probabilities),
    )
    candidates = [0.0]
    candidates.extend(math.nextafter(value, math.inf) for value in sorted(set(probabilities)))
    evidence: list[dict[str, Any]] = []
    for threshold in candidates:
        choose_safe = [value >= threshold for value in probabilities]
        item = selected_metrics(safe_iou, aggressive_iou, safe_valid, aggressive_valid, choose_safe)
        item.update(
            {
                "threshold": threshold,
                "acc_loss_vs_always_safe": float(always_safe["acc_at_0_5"]) - float(item["acc_at_0_5"]),
                "mean_iou_loss_vs_always_safe": float(always_safe["mean_iou"]) - float(item["mean_iou"]),
            }
        )
        item["constraint_satisfied"] = (
            item["acc_loss_vs_always_safe"] <= 0.005 + 1e-12
            and item["mean_iou_loss_vs_always_safe"] <= 0.005 + 1e-12
        )
        evidence.append(item)
    feasible = [item for item in evidence if item["constraint_satisfied"] and item["aggressive_rows"] > 0]
    if feasible:
        chosen = min(
            feasible,
            key=lambda item: (
                float(item["average_visual_tokens"]),
                float(item["threshold"]),
            ),
        )
        successful = True
        fallback = False
    else:
        chosen = evidence[0]
        successful = False
        fallback = True
    return {
        "status": "calibrated" if successful else "always_safe_fallback",
        "successful_positive_aggressive_rate": successful,
        "always_safe_fallback": fallback,
        "threshold": float(chosen["threshold"]),
        "chosen_metrics": chosen,
        "always_safe_metrics": always_safe,
        "candidate_thresholds_evaluated": len(evidence),
        "feasible_positive_aggressive_thresholds": len(feasible),
        "threshold_evidence": evidence,
    }
