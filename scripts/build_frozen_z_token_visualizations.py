#!/usr/bin/env python3
"""Build outcome-independent visualizations of the frozen Z-M448-0 token schedule."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.frozen_z_error_analysis_common import atomic_json, now, record_file, require, sha256  # noqa: E402

INPUT = REPO / "experiments/frozen_z_m448_0_refcoco_testab_20260820"
CAMPAIGN = REPO / "experiments/frozen_z_token_visualization_extension_20260820"
DISSERTATION = Path("/workspace/Dissertation_Draft")
FIGURE_DIR = DISSERTATION / "images/final_evaluation"
DERIVED = DISSERTATION / "analysis/derived/frozen_z_token_visualization"
LEDGER = INPUT / "QUALITY_ARTIFACT_LEDGER.json"
EXPECTED_LEDGER_SHA256 = "7d182ae55896dc8b5188f729bdeda16d2d61ac57306d30e0a35145b4074ff2db"
PREDICTIONS = {
    "testA": INPUT / "quality/z_m448_0_testa_attempt1/predictions.scored.jsonl",
    "testB": INPUT / "quality/z_m448_0_testb_attempt1/predictions.scored.jsonl",
}
EXPECTED_PREDICTION_SHA256 = {
    "testA": "ac4f462a102f64918c60cbb1371591c5cbe6efbe42a402adf97d86373a4779ef",
    "testB": "6d44b44599f5df8aae8e5f4a0c8ec43355d5962319379e6139f458615c7cc1bb",
}
EXPECTED_ROWS = {"testA": 5657, "testB": 5095}
GRID = 24
PATCHES = GRID * GRID


def iter_rows(path: Path, split: str) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            require(row.get("manifest_index") == index, f"Manifest order differs: {split}/{index}")
            require(row.get("split") == split and row.get("method") == "Z_M448_0", f"Row identity differs: {split}/{index}")
            yield row


def stage(row: dict[str, Any], index: int, boundary: int, budget: int) -> dict[str, Any]:
    stages = row.get("stage_traces")
    require(isinstance(stages, list) and len(stages) == 3, "Frozen stage trace is missing")
    value = stages[index]
    require(value.get("stage_index") == index and value.get("boundary") == boundary, "Stage boundary differs")
    require(value.get("realized_budget") == budget, "Stage budget differs")
    return value


def validate_row(row: dict[str, Any]) -> None:
    require(row.get("realized_visual_tokens_by_layer") == [576] * 4 + [448] * 8 + [48] * 8 + [0] * 12,
            "Frozen visual trace differs")
    s4 = stage(row, 0, 4, 448)
    s12 = stage(row, 1, 12, 48)
    s20 = stage(row, 2, 20, 0)
    out4 = s4.get("outgoing_original_indices")
    masses = s4.get("merge_output_masses")
    out12 = s12.get("outgoing_original_indices")
    require(len(out4) == len(set(out4)) == len(masses) == 448, "Layer-4 merge trace differs")
    require(all(isinstance(i, int) and 0 <= i < PATCHES for i in out4), "Layer-4 raster index differs")
    require(math.isclose(sum(float(v) for v in masses), 576.0), "Layer-4 mass is not conserved")
    require(len(out12) == len(set(out12)) == 48 and set(out12).issubset(set(out4)), "Layer-12 mask differs")
    require(s20.get("outgoing_original_indices") == [] and s20.get("selector") == "none"
            and s20.get("score_map_recomputed") is False, "Layer-20 zero-tail trace differs")


def processor_crop(image: Image.Image) -> tuple[Image.Image, tuple[float, float, float]]:
    """Reproduce CLIP shortest-edge resize plus 336x336 centre crop geometrically."""
    width, height = image.size
    side = min(width, height)
    left = (width - side) / 2.0
    top = (height - side) / 2.0
    cropped = image.crop((left, top, left + side, top + side)).resize((336, 336), Image.Resampling.BICUBIC)
    return cropped, (left, top, 336.0 / side)


def mapped_box(box: list[float], transform: tuple[float, float, float]) -> tuple[float, float, float, float]:
    left, top, scale = transform
    x1, y1, x2, y2 = box
    values = ((x1 - left) * scale, (y1 - top) * scale, (x2 - left) * scale, (y2 - top) * scale)
    return tuple(max(0.0, min(336.0, float(value))) for value in values)


def grid_extent() -> tuple[int, int, int, int]:
    return (0, 336, 336, 0)


def draw_box(axis: Any, box: tuple[float, float, float, float]) -> None:
    x1, y1, x2, y2 = box
    axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#00e5ff", linewidth=2.0))


def aggregate_mask(split: str) -> tuple[np.ndarray, int, dict[str, Any]]:
    counts = np.zeros(PATCHES, dtype=np.int64)
    rows = 0
    first_id = None
    last_id = None
    for row in iter_rows(PREDICTIONS[split], split):
        validate_row(row)
        indices = stage(row, 1, 12, 48)["outgoing_original_indices"]
        counts[indices] += 1
        first_id = first_id or row["sample_id"]
        last_id = row["sample_id"]
        rows += 1
    require(rows == EXPECTED_ROWS[split], f"Row count differs: {split}")
    require(int(counts.sum()) == rows * 48, f"Aggregate selection count differs: {split}")
    rates = 100.0 * counts.reshape(GRID, GRID) / rows
    audit = {
        "rows": rows,
        "first_sample_id": first_id,
        "last_sample_id": last_id,
        "selected_positions_total": int(counts.sum()),
        "mean_patch_selection_percent": float(rates.mean()),
        "minimum_patch_selection_percent": float(rates.min()),
        "maximum_patch_selection_percent": float(rates.max()),
        "selection_counts_24x24": counts.reshape(GRID, GRID).tolist(),
    }
    return rates, rows, audit


def atomic_figure(figure: Any, target: Path, **kwargs: Any) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.stem}.", suffix=target.suffix, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(temporary, **kwargs)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    require(CAMPAIGN.is_dir(), "Visualization campaign was not initialized")
    require(sha256(LEDGER) == EXPECTED_LEDGER_SHA256, "Held-out quality ledger differs")
    for split, path in PREDICTIONS.items():
        require(sha256(path) == EXPECTED_PREDICTION_SHA256[split], f"Accepted predictions differ: {split}")

    example = next(iter_rows(PREDICTIONS["testA"], "testA"))
    require(example["manifest_index"] == 0 and example["sample_id"] == "refcoco_testA_000000_0",
            "Mechanical example selection differs")
    validate_row(example)
    image_path = REPO / example["image_path"]
    require(image_path.is_file(), "Example source image is missing")
    raw_image = Image.open(image_path).convert("RGB")
    processed, transform = processor_crop(raw_image)
    box = mapped_box(example["gt_box"], transform)

    s4 = stage(example, 0, 4, 448)
    s12 = stage(example, 1, 12, 48)
    mass = np.full(PATCHES, np.nan, dtype=np.float64)
    mass[np.asarray(s4["outgoing_original_indices"], dtype=np.int64)] = np.asarray(s4["merge_output_masses"], dtype=np.float64)
    mass = mass.reshape(GRID, GRID)
    mask = np.zeros(PATCHES, dtype=bool)
    mask[np.asarray(s12["outgoing_original_indices"], dtype=np.int64)] = True
    mask = mask.reshape(GRID, GRID)
    rates_a, rows_a, audit_a = aggregate_mask("testA")
    rates_b, rows_b, audit_b = aggregate_mask("testB")

    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 10, "figure.dpi": 140})
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.15), constrained_layout=True)
    image_array = np.asarray(processed)

    axes[0, 0].imshow(image_array)
    draw_box(axes[0, 0], box)
    axes[0, 0].set_title("(a) CLIP crop before pruning: 576 patches")
    axes[0, 0].text(0.01, 0.02, 'query: “person bottom left”', transform=axes[0, 0].transAxes,
                    color="white", fontsize=8, bbox={"facecolor": "black", "alpha": 0.7, "pad": 2})

    axes[0, 1].imshow(image_array)
    cmap_mass = plt.get_cmap("magma").copy()
    cmap_mass.set_bad(alpha=0.08)
    mass_image = axes[0, 1].imshow(mass, extent=grid_extent(), cmap=cmap_mass,
                                   norm=BoundaryNorm(np.arange(0.5, 7.5, 1.0), cmap_mass.N), alpha=0.72)
    draw_box(axes[0, 1], box)
    axes[0, 1].set_title("(b) After layer 4: 448 merged states")
    fig.colorbar(mass_image, ax=axes[0, 1], fraction=0.046, pad=0.02, label="represented patch mass")

    axes[0, 2].imshow(image_array)
    dark = np.where(mask, np.nan, 1.0)
    cmap_dark = plt.get_cmap("gray").copy()
    cmap_dark.set_bad(alpha=0.0)
    axes[0, 2].imshow(dark, extent=grid_extent(), cmap=cmap_dark, vmin=0, vmax=1, alpha=0.78)
    draw_box(axes[0, 2], box)
    axes[0, 2].set_title("(c) After layer 12: 48-state mask")

    axes[1, 0].imshow(np.asarray(processed.convert("L")), cmap="gray", vmin=0, vmax=255, alpha=0.28)
    axes[1, 0].text(0.5, 0.56, "0 live visual states", ha="center", va="center", transform=axes[1, 0].transAxes,
                    fontsize=15, weight="bold", color="#7b1fa2")
    axes[1, 0].text(0.5, 0.43, "visual information may persist\nin text hidden states", ha="center", va="center",
                    transform=axes[1, 0].transAxes, fontsize=9)
    axes[1, 0].set_title("(d) After layer 20: zero visual tail")

    vmax = max(float(rates_a.max()), float(rates_b.max()))
    heat_a = axes[1, 1].imshow(rates_a, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
    axes[1, 1].set_title(f"(e) Layer-12 selection frequency: testA (n={rows_a:,})")
    axes[1, 2].imshow(rates_b, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
    axes[1, 2].set_title(f"(f) Layer-12 selection frequency: testB (n={rows_b:,})")
    fig.colorbar(heat_a, ax=[axes[1, 1], axes[1, 2]], fraction=0.046, pad=0.02, label="samples selecting patch (%)")

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.suptitle("Frozen Z-M448-0 visual-state progression and aggregate 48-state masks", fontsize=13, weight="bold")

    pdf = FIGURE_DIR / "z_m448_0_token_visualization.pdf"
    png = FIGURE_DIR / "z_m448_0_token_visualization.png"
    atomic_figure(fig, pdf, bbox_inches="tight")
    atomic_figure(fig, png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    analysis = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": now(),
        "scope": "descriptive visualization only; no inference, tuning, subgrouping, or candidate selection",
        "selection_rule": {
            "kind": "mechanical_first_manifest_row",
            "split": "testA",
            "manifest_index": 0,
            "sample_id": example["sample_id"],
            "outcome_fields_used_for_selection": [],
        },
        "example": {
            "sample_id": example["sample_id"],
            "query": example["query"],
            "image": record_file(image_path),
            "source_dimensions": [example["width"], example["height"]],
            "processor_geometry": "CLIP shortest-edge 336 resize followed by 336x336 centre crop",
            "ground_truth_box_source_pixels": example["gt_box"],
            "ground_truth_box_processor_pixels_clipped": list(box),
            "layer4_destination_indices": s4["outgoing_original_indices"],
            "layer4_output_masses": s4["merge_output_masses"],
            "layer12_retained_indices": s12["outgoing_original_indices"],
            "layer20_retained_indices": [],
        },
        "aggregate_layer12": {"testA": audit_a, "testB": audit_b},
        "interpretation_limits": [
            "The mask shows surviving token positions, not a segmentation prediction or attention explanation.",
            "Layer-4 colour encodes represented patch mass at each surviving destination, not importance.",
            "The deterministic example is illustrative and is not evidence of typical quality.",
            "Aggregate maps describe spatial selection frequency only and are not outcome-conditioned.",
        ],
        "source_bindings": {
            "quality_artifact_ledger": record_file(LEDGER),
            "testA_predictions": record_file(PREDICTIONS["testA"]),
            "testB_predictions": record_file(PREDICTIONS["testB"]),
        },
    }
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    atomic_json(CAMPAIGN / "TOKEN_VISUALIZATION_DATA.json", analysis)
    DERIVED.mkdir(parents=True, exist_ok=True)
    atomic_json(DERIVED / "TOKEN_VISUALIZATION_DATA.json", analysis)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": now(),
        "logical_figures": 1,
        "panels": 6,
        "sources": analysis["source_bindings"],
        "generated": {
            "figure_pdf": {"path": str(pdf), **record_file(pdf)},
            "figure_png": {"path": str(png), **record_file(png)},
            "data": {"path": str(DERIVED / "TOKEN_VISUALIZATION_DATA.json"), **record_file(DERIVED / "TOKEN_VISUALIZATION_DATA.json")},
        },
    }
    atomic_json(CAMPAIGN / "ASSET_MANIFEST.json", manifest)
    atomic_json(DERIVED / "ASSET_MANIFEST.json", manifest)
    print(json.dumps({"status": "complete", "pdf": manifest["generated"]["figure_pdf"], "panels": 6}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
