"""Audited trace-driven FLOP, token-work, and KV accounting for the final campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class TraceAccounting:
    visual_trace: list[int]
    text_tokens: int
    visual_token_layer_work: int
    average_visual_tokens: float
    final_visual_tokens: int
    sequence_lengths: list[int]
    attention_flops: float
    linear_and_mlp_flops: float
    llm_prefill_flops: float
    pruning_metric_flops: float
    modeled_total_prefill_flops: float
    analytical_prompt_kv_bytes: int
    scope: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def account_trace(
    visual_trace: Sequence[int],
    *,
    text_tokens: int,
    hidden_size: int = 4096,
    intermediate_size: int = 11008,
    kv_hidden_size: int = 4096,
    dtype_bytes: int = 2,
    pruning_metric_flops: float = 0.0,
) -> TraceAccounting:
    """Apply the dissertation's registered decoder-prefill formula layer by layer.

    Per layer, attention FLOPs are ``8*n*d^2 + 4*n^2*d`` and gated-MLP
    FLOPs are ``6*n*d*m``. KV storage is the physically cacheable K and V
    state, ``2*n*kv_hidden*dtype_bytes`` per layer.
    """
    trace = [int(value) for value in visual_trace]
    if not trace or any(value < 0 for value in trace):
        raise ValueError("Visual trace must be non-empty and non-negative")
    if text_tokens < 0 or min(hidden_size, intermediate_size, kv_hidden_size, dtype_bytes) <= 0:
        raise ValueError("Model dimensions, dtype bytes, and text count must be valid")
    if pruning_metric_flops < 0:
        raise ValueError("Pruning metric FLOPs cannot be negative")
    sequence = [text_tokens + value for value in trace]
    attention = sum(
        8.0 * length * hidden_size * hidden_size
        + 4.0 * length * length * hidden_size
        for length in sequence
    )
    mlp = sum(6.0 * length * hidden_size * intermediate_size for length in sequence)
    llm = attention + mlp
    prompt_kv = sum(2 * length * kv_hidden_size * dtype_bytes for length in sequence)
    work = sum(trace)
    return TraceAccounting(
        visual_trace=trace,
        text_tokens=int(text_tokens),
        visual_token_layer_work=work,
        average_visual_tokens=work / len(trace),
        final_visual_tokens=trace[-1],
        sequence_lengths=sequence,
        attention_flops=attention,
        linear_and_mlp_flops=mlp,
        llm_prefill_flops=llm,
        pruning_metric_flops=float(pruning_metric_flops),
        modeled_total_prefill_flops=llm + float(pruning_metric_flops),
        analytical_prompt_kv_bytes=int(prompt_kv),
        scope=(
            "decoder prefill only: attention projections/matmul, gated MLP, supplied pruning-metric "
            "operations; vision encoder and projector are excluded and reported as constants"
        ),
    )


def known_trace(method: str) -> list[int]:
    traces = {
        "Full": [576] * 32,
        "K12": [576] * 12 + [48] * 20,
        "M448_OPT": [576] * 4 + [448] * 8 + [48] * 20,
        "Z_M448_0": [576] * 4 + [448] * 8 + [48] * 8 + [0] * 12,
        "FASTV_MATCHED_199": [576] * 3 + [160] * 29,
        "NUWA_NATIVE_192": [336] * 16 + [48] * 16,
    }
    try:
        return traces[method]
    except KeyError as exc:
        raise ValueError(f"Unregistered fixed trace: {method}") from exc
