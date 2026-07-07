"""Dataset and cache helpers for UG-Prune-Lite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vlm_pruning.selectors.ug_prune_lite import box_patch_labels
from vlm_pruning.training.ug_prune_lite_features import maybe_apply_projection, projection_matrix


def read_jsonl(path: Path, *, start_index: int = 0, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start_index or not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def sample_ids_from_manifest(path: Path, *, limit: int | None = None) -> set[str]:
    return {str(row["sample_id"]) for row in read_jsonl(path, limit=limit)}


def load_exclude_ids(paths: list[Path] | None = None) -> set[str]:
    out: set[str] = set()
    for path in paths or []:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = line.strip()
                if value:
                    out.add(value)
    return out


def multi_hot(torch: Any, indices: list[int], *, size: int = 576, device: Any | None = None) -> Any:
    labels = torch.zeros(size, dtype=torch.float32, device=device)
    if indices:
        labels[torch.tensor(indices, dtype=torch.long, device=device)] = 1.0
    return labels


class UGPruneLiteCacheDataset:
    def __init__(
        self,
        torch: Any,
        cache_dir: Path,
        *,
        limit: int | None = None,
        device: Any | None = None,
        feature_projection: dict[str, Any] | None = None,
    ) -> None:
        self.torch = torch
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.metadata = json.loads((self.cache_dir / "metadata.json").read_text(encoding="utf-8"))
        self.feature_projection = feature_projection
        self._projection_matrix: Any | None = None
        self.examples = read_jsonl(self.cache_dir / "examples.jsonl", limit=limit)
        if not self.examples:
            raise ValueError(f"UG-Prune-Lite cache is empty: {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.examples[int(index)]
        feature_path = self.cache_dir / row["feature_path"]
        try:
            payload = self.torch.load(feature_path, map_location=self.device or "cpu", weights_only=True)
        except TypeError:
            payload = self.torch.load(feature_path, map_location=self.device or "cpu")
        roles = row.get("role_labels") or {}
        selected = [int(item) for item in row.get("selected_indices") or []]
        target = [int(item) for item in roles.get("target") or []]
        context = [int(item) for item in roles.get("context") or []]
        width = int(row.get("width") or row.get("metadata", {}).get("width") or 1)
        height = int(row.get("height") or row.get("metadata", {}).get("height") or 1)
        device = self.device
        visual_features = payload["visual_features"].to(device=device, dtype=self.torch.float32)
        query_embedding = payload["query_embedding"].to(device=device, dtype=self.torch.float32)
        teacher_score_map = payload.get("teacher_score_map")
        has_teacher_score_map = teacher_score_map is not None
        if teacher_score_map is None:
            teacher_score_map = self.torch.zeros(576, dtype=self.torch.float32, device=device)
        else:
            teacher_score_map = teacher_score_map.to(device=device, dtype=self.torch.float32).reshape(-1)
            if int(teacher_score_map.numel()) != 576:
                raise ValueError(
                    f"Expected teacher_score_map with 576 values for {row['sample_id']}, "
                    f"got {int(teacher_score_map.numel())}."
                )
        if self.feature_projection:
            if self._projection_matrix is None:
                self._projection_matrix = projection_matrix(
                    self.torch,
                    self.feature_projection,
                    device=visual_features.device,
                    dtype=visual_features.dtype,
                )
            visual_features = maybe_apply_projection(visual_features, self._projection_matrix, self.feature_projection)
            query_embedding = maybe_apply_projection(query_embedding, self._projection_matrix, self.feature_projection)
        return {
            "sample_id": str(row["sample_id"]),
            "visual_features": visual_features,
            "query_embedding": query_embedding,
            "target_labels": multi_hot(self.torch, target, device=device),
            "context_labels": multi_hot(self.torch, context, device=device),
            "keep_labels": multi_hot(self.torch, selected, device=device),
            "box_labels": box_patch_labels(self.torch, row["gt_box"], width, height, device=device),
            "teacher_score_map": teacher_score_map,
            "has_teacher_score_map": self.torch.tensor(float(has_teacher_score_map), dtype=self.torch.float32, device=device),
            "selected_indices": selected,
            "role_labels": roles,
        }


def collate_ug_prune_lite(batch: list[dict[str, Any]], torch: Any) -> dict[str, Any]:
    keys = [
        "visual_features",
        "query_embedding",
        "target_labels",
        "context_labels",
        "keep_labels",
        "box_labels",
        "teacher_score_map",
        "has_teacher_score_map",
    ]
    out = {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}
    out["sample_id"] = [item["sample_id"] for item in batch]
    out["selected_indices"] = [item["selected_indices"] for item in batch]
    out["role_labels"] = [item["role_labels"] for item in batch]
    return out
