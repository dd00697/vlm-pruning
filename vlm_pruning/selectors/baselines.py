"""Baseline full/random/uniform selector wrappers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable

from .base import SelectionResult, SelectorError


RetainedIndexFn = Callable[[str, int, int, int, str, int], list[int]]


def _load_retained_indices() -> RetainedIndexFn:
    try:
        module = importlib.import_module("run_baseline")
    except ModuleNotFoundError:
        scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        module = importlib.import_module("run_baseline")
    return module.retained_indices


class BaselineSelector:
    def __init__(self, method: str) -> None:
        if method not in {"full", "random", "uniform"}:
            raise SelectorError(f"Unsupported Phase 0 baseline selector: {method}")
        self.name = method
        self._retained_indices = _load_retained_indices()

    def select(
        self,
        *,
        sample_id: str,
        budget: int,
        original_tokens: int,
        grid: int,
        seed: int,
    ) -> SelectionResult:
        indices = self._retained_indices(self.name, budget, original_tokens, seed, sample_id, grid)
        result = SelectionResult(
            selected_indices=sorted(indices),
            method=self.name,
            budget=budget,
            original_tokens=original_tokens,
            grid=grid,
            role_labels={},
            metadata={
                "selector_family": "phase0_baseline",
                "source_function": "scripts.run_baseline.retained_indices",
                "seed": seed if self.name == "random" else 0,
                "sample_id": sample_id,
            },
        )
        return result.validate()


class FullSelector(BaselineSelector):
    def __init__(self) -> None:
        super().__init__("full")


class RandomSelector(BaselineSelector):
    def __init__(self) -> None:
        super().__init__("random")


class UniformSelector(BaselineSelector):
    def __init__(self) -> None:
        super().__init__("uniform")


def selector_for_method(method: str) -> BaselineSelector:
    if method == "full":
        return FullSelector()
    if method == "random":
        return RandomSelector()
    if method == "uniform":
        return UniformSelector()
    raise SelectorError(f"Phase 0 official baseline selector does not support method: {method}")

