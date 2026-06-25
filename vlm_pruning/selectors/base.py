"""Common selector interface for visual-token retention."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class SelectorError(ValueError):
    """Raised when a selector returns invalid visual-token indices."""


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: list[int]
    method: str
    budget: int
    original_tokens: int
    grid: int
    role_labels: dict[str, list[int]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "SelectionResult":
        if self.original_tokens <= 0:
            raise SelectorError(f"original_tokens must be positive, got {self.original_tokens}.")
        if self.grid <= 0:
            raise SelectorError(f"grid must be positive, got {self.grid}.")
        if self.budget <= 0:
            raise SelectorError(f"budget must be positive, got {self.budget}.")
        if self.budget > self.original_tokens:
            raise SelectorError(f"budget {self.budget} exceeds original token count {self.original_tokens}.")
        if len(self.selected_indices) != self.budget:
            raise SelectorError(
                f"selected index count {len(self.selected_indices)} does not match budget {self.budget}."
            )
        if any(not isinstance(index, int) for index in self.selected_indices):
            raise SelectorError("selected indices must all be integers.")
        if len(set(self.selected_indices)) != len(self.selected_indices):
            raise SelectorError("selected indices must be unique.")
        if min(self.selected_indices, default=0) < 0 or max(self.selected_indices, default=-1) >= self.original_tokens:
            raise SelectorError(
                f"selected indices must be in [0, {self.original_tokens - 1}], got range "
                f"[{min(self.selected_indices)}, {max(self.selected_indices)}]."
            )
        return self

    def as_metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "budget": self.budget,
            "original_tokens": self.original_tokens,
            "grid": self.grid,
            **self.metadata,
        }


class VisualTokenSelector(Protocol):
    name: str

    def select(
        self,
        *,
        sample_id: str,
        budget: int,
        original_tokens: int,
        grid: int,
        seed: int,
    ) -> SelectionResult:
        """Return retained visual-token indices in original patch coordinates."""

