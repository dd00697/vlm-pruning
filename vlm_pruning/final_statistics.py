"""Frozen statistical and decision utilities for the final validation campaign."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def paired_bootstrap_interval(
    left: Sequence[float],
    right: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 20_260_818,
) -> dict[str, float | int]:
    """Return a percentile interval for the paired mean ``left - right``."""
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.ndim != 1 or left_array.shape != right_array.shape or not left_array.size:
        raise ValueError("Paired inputs must be non-empty one-dimensional arrays of equal length")
    if resamples <= 0 or not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ValueError("Bootstrap inputs and resample count must be valid")
    delta = left_array - right_array
    estimates = np.empty(resamples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, resamples, 100):
        count = min(100, resamples - start)
        indices = rng.integers(0, delta.size, size=(count, delta.size))
        estimates[start : start + count] = delta[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "point_delta": float(delta.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "paired_rows": int(delta.size),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
    }


def hierarchical_paired_bootstrap_interval(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    *,
    resamples: int = 10_000,
    seed: int = 20_260_818,
) -> dict[str, float | int]:
    """Bootstrap paired repetitions and paired requests for a repeated timing contrast.

    Arrays have shape ``repetition x request``. Each draw resamples repetitions and,
    independently within each selected repetition, request indices. Pairing between the
    two methods is retained at both levels.
    """
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if (
        left_array.ndim != 2
        or left_array.shape != right_array.shape
        or not left_array.size
        or min(left_array.shape) <= 0
    ):
        raise ValueError("Hierarchical inputs must be equal non-empty repetition-by-request arrays")
    if resamples <= 0 or not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ValueError("Bootstrap inputs and resample count must be valid")
    delta = left_array - right_array
    repetitions, requests = delta.shape
    estimates = np.empty(resamples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for draw in range(resamples):
        repetition_indices = rng.integers(0, repetitions, size=repetitions)
        values = []
        for repetition_index in repetition_indices:
            request_indices = rng.integers(0, requests, size=requests)
            values.append(delta[repetition_index, request_indices].mean())
        estimates[draw] = float(np.mean(values))
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "point_delta": float(delta.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "paired_repetitions": int(repetitions),
        "paired_rows_per_repetition": int(requests),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
    }


def pareto_frontier(
    records: Sequence[Mapping[str, Any]],
    *,
    quality_field: str,
    cost_field: str,
    method_field: str = "method",
) -> list[str]:
    """Return methods not dominated by higher quality and lower cost."""
    output: list[str] = []
    for candidate in records:
        candidate_quality = float(candidate[quality_field])
        candidate_cost = float(candidate[cost_field])
        dominated = False
        for other in records:
            if other[method_field] == candidate[method_field]:
                continue
            other_quality = float(other[quality_field])
            other_cost = float(other[cost_field])
            if (
                other_quality >= candidate_quality
                and other_cost <= candidate_cost
                and (other_quality > candidate_quality or other_cost < candidate_cost)
            ):
                dominated = True
                break
        if not dominated:
            output.append(str(candidate[method_field]))
    return output


def select_internal_candidate(
    quality: Mapping[str, Mapping[str, float]],
    timing: Mapping[str, Mapping[str, float]],
    work: Mapping[str, int],
    *,
    eligible: Sequence[str] = ("K12", "M448_OPT", "Z_M448_0"),
) -> dict[str, Any]:
    """Apply the preregistered quality, acceleration, objective, and tie-break rules."""
    full_quality = quality["Full"]
    full_timing = timing["Full"]
    decisions: dict[str, dict[str, Any]] = {}
    passing: list[str] = []
    for method in eligible:
        metrics = quality[method]
        timing_metrics = timing[method]
        quality_checks = {
            "parse_rate_at_least_0_995": float(metrics["parse_rate"]) >= 0.995,
            "acc_drop_at_most_3pp": (
                float(full_quality["acc_at_0_5"]) - float(metrics["acc_at_0_5"])
            )
            <= 0.03,
            "miou_drop_at_most_0_015": (
                float(full_quality["mean_iou"]) - float(metrics["mean_iou"])
            )
            <= 0.015,
        }
        acceleration_checks = {
            "natural_e2e_mean_below_full": float(timing_metrics["end_to_end_latency_s"])
            < float(full_timing["end_to_end_latency_s"]),
            "paired_upper_95_below_zero": float(timing_metrics["delta_vs_full_upper_95_s"]) < 0.0,
        }
        quality_pass = all(quality_checks.values())
        acceleration_pass = all(acceleration_checks.values())
        decisions[method] = {
            "quality_checks": quality_checks,
            "quality_gate_pass": quality_pass,
            "acceleration_checks": acceleration_checks,
            "acceleration_gate_pass": acceleration_pass,
            "passes_both_gates": quality_pass and acceleration_pass,
            "visual_token_layer_work": int(work[method]),
        }
        if quality_pass and acceleration_pass:
            passing.append(method)
    if not passing:
        selection = None
        status = "NO_FINAL_CANDIDATE_PASSES_REGISTERED_GATE"
    else:
        selection = min(
            passing,
            key=lambda method: (
                int(work[method]),
                float(timing[method]["end_to_end_latency_s"]),
                -float(quality[method]["mean_iou"]),
                -float(quality[method]["acc_at_0_5"]),
            ),
        )
        status = "FINAL_CANDIDATE_SELECTED_VALIDATION_ONLY"
    return {
        "status": status,
        "selected_method": selection,
        "eligible_methods": list(eligible),
        "passing_methods": passing,
        "method_decisions": decisions,
        "rule": {
            "objective": "minimum_visual_token_layer_work",
            "tie_breaks": ["lower_e2e", "higher_miou", "higher_acc"],
            "held_out_authorized": False,
        },
    }
