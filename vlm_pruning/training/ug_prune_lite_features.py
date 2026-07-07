"""Feature projection helpers for UG-Prune-Lite caches and inference."""

from __future__ import annotations

import math
from typing import Any


def random_projection_spec(*, input_dim: int, output_dim: int, seed: int = 0) -> dict[str, Any]:
    if int(output_dim) <= 0:
        raise ValueError("Projection output_dim must be positive.")
    if int(output_dim) > int(input_dim):
        raise ValueError("Projection output_dim cannot exceed input_dim.")
    return {
        "type": "gaussian_random_projection",
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "seed": int(seed),
        "scale": "1/sqrt(output_dim)",
    }


def projection_matrix(torch: Any, spec: dict[str, Any] | None, *, device: Any, dtype: Any | None = None) -> Any | None:
    if not spec:
        return None
    if spec.get("type") != "gaussian_random_projection":
        raise ValueError(f"Unsupported feature projection type: {spec.get('type')}")
    input_dim = int(spec["input_dim"])
    output_dim = int(spec["output_dim"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(spec.get("seed", 0)))
    matrix = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float32)
    matrix = matrix / math.sqrt(float(output_dim))
    return matrix.to(device=device, dtype=dtype or torch.float32)


def maybe_apply_projection(features: Any, matrix: Any | None, spec: dict[str, Any] | None) -> Any:
    if matrix is None or spec is None:
        return features
    input_dim = int(spec["input_dim"])
    output_dim = int(spec["output_dim"])
    actual_dim = int(features.shape[-1])
    if actual_dim == output_dim:
        return features
    if actual_dim != input_dim:
        raise ValueError(f"Feature dim {actual_dim} is incompatible with projection {input_dim}->{output_dim}.")
    return features.to(device=matrix.device, dtype=matrix.dtype).matmul(matrix)
