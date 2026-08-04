"""Measured-efficiency helpers shared by the final timing campaign."""

from __future__ import annotations

from typing import Any


def cache_tensor_bytes(cache: Any) -> int:
    """Return allocated tensor bytes in a Transformers cache without double counting."""
    seen_objects: set[int] = set()
    seen_storage: set[tuple[str, int]] = set()

    def visit(value: Any) -> int:
        if value is None:
            return 0
        # Newer Transformers cache objects retain configuration classes.  A
        # ``torch.Tensor`` class exposes unbound storage methods but is not an
        # allocated tensor instance and must not be counted.
        if isinstance(value, type):
            return 0
        identity = id(value)
        if identity in seen_objects:
            return 0
        seen_objects.add(identity)
        if hasattr(value, "untyped_storage") and hasattr(value, "element_size"):
            storage = value.untyped_storage()
            pointer = int(storage.data_ptr())
            key = (str(value.device), pointer)
            if key in seen_storage:
                return 0
            seen_storage.add(key)
            return int(storage.nbytes())
        if isinstance(value, dict):
            return sum(visit(key) + visit(item) for key, item in value.items())
        if isinstance(value, (list, tuple, set)):
            return sum(visit(item) for item in value)
        if hasattr(value, "__dict__"):
            return sum(visit(item) for item in vars(value).values())
        return 0

    return visit(cache)


def balanced_latin_square(methods: list[str], repetitions: int = 5) -> list[list[str]]:
    """Return the registered cyclic order where each method occupies each slot."""
    if len(methods) != repetitions:
        raise ValueError("The final protocol requires one method per order slot and repetition.")
    return [methods[offset:] + methods[:offset] for offset in range(repetitions)]
