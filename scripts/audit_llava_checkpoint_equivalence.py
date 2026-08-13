#!/usr/bin/env python
"""Prove tensor equality between HF-converted and original LLaVA checkpoints.

The canonical ``llava-hf`` checkpoint stores the language model, projector and
CLIP tower in one safetensors index.  The author-original checkpoint stores the
language model/projector in PyTorch shards and loads the pinned OpenAI CLIP
tower separately.  This utility maps those packaging-only name differences,
hashes tensor bytes, and fails unless every scientific parameter is identical.
Derived rotary ``inv_freq`` buffers in the old stack are intentionally reported
but excluded because they are not parameters and are absent from the converted
checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    canonical = sub.add_parser("snapshot-canonical")
    canonical.add_argument("--checkpoint", required=True, type=Path)
    canonical.add_argument("--output", required=True, type=Path)
    compare = sub.add_parser("compare-original")
    compare.add_argument("--canonical-summary", required=True, type=Path)
    compare.add_argument("--original-checkpoint", required=True, type=Path)
    compare.add_argument("--vision-checkpoint", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def tensor_digest(tensor: Any) -> dict[str, Any]:
    import torch

    value = tensor.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "byte_count": len(raw),
    }


def semantic_canonical_name(name: str) -> str:
    if name.startswith("language_model."):
        return name.removeprefix("language_model.")
    projector = {
        "multi_modal_projector.linear_1.bias": "model.mm_projector.0.bias",
        "multi_modal_projector.linear_1.weight": "model.mm_projector.0.weight",
        "multi_modal_projector.linear_2.bias": "model.mm_projector.2.bias",
        "multi_modal_projector.linear_2.weight": "model.mm_projector.2.weight",
    }
    if name in projector:
        return projector[name]
    if name.startswith("vision_tower."):
        return "clip::" + name.removeprefix("vision_tower.")
    raise ValueError(f"Unmapped canonical tensor: {name}")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def indexed_safetensors(checkpoint: Path) -> Iterator[tuple[str, Any]]:
    from safetensors import safe_open

    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name, shard in index.items():
        by_shard.setdefault(shard, []).append(name)
    for shard, names in sorted(by_shard.items()):
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as handle:
            for name in sorted(names):
                yield name, handle.get_tensor(name)


def snapshot_canonical(checkpoint: Path, output: Path) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    source_names: dict[str, str] = {}
    for source_name, tensor in indexed_safetensors(checkpoint):
        semantic = semantic_canonical_name(source_name)
        if semantic in tensors:
            raise RuntimeError(f"Duplicate semantic tensor {semantic}")
        tensors[semantic] = tensor_digest(tensor)
        source_names[semantic] = source_name
    counts = {
        "language_and_projector": sum(not name.startswith("clip::") for name in tensors),
        "vision": sum(name.startswith("clip::") for name in tensors),
        "total": len(tensors),
    }
    if counts != {"language_and_projector": 295, "vision": 391, "total": 686}:
        raise RuntimeError(f"Unexpected canonical tensor counts: {counts}")
    payload = {
        "status": "snapshotted",
        "checkpoint": str(checkpoint.resolve()),
        "index_sha256": hashlib.sha256((checkpoint / "model.safetensors.index.json").read_bytes()).hexdigest(),
        "counts": counts,
        "tensors": tensors,
        "source_names": source_names,
    }
    atomic_json(output, payload)
    return payload


def pytorch_index_tensors(checkpoint: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    import torch

    index_path = checkpoint / "pytorch_model.bin.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name, shard in index.items():
        by_shard.setdefault(shard, []).append(name)
    result: dict[str, dict[str, Any]] = {}
    derived = []
    for shard, names in sorted(by_shard.items()):
        state = torch.load(checkpoint / shard, map_location="cpu", weights_only=True)
        for name in sorted(names):
            if name.endswith("self_attn.rotary_emb.inv_freq"):
                derived.append(name)
                continue
            result[name] = tensor_digest(state[name])
        del state
    return result, derived


def vision_tensors(checkpoint: Path, wanted: set[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    from safetensors import safe_open

    candidates = [checkpoint / "model.safetensors"]
    if (checkpoint / "model.safetensors.index.json").is_file():
        values: dict[str, dict[str, Any]] = {}
        ignored = []
        for source_name, tensor in indexed_safetensors(checkpoint):
            name = "clip::" + source_name
            if name in wanted:
                values[name] = tensor_digest(tensor)
            else:
                ignored.append(name)
        ignored.sort()
        return values, ignored
    for path in candidates:
        if path.is_file():
            with safe_open(path, framework="pt", device="cpu") as handle:
                names = ["clip::" + name for name in handle.keys()]
                values = {
                    name: tensor_digest(handle.get_tensor(name.removeprefix("clip::")))
                    for name in names if name in wanted
                }
                return values, sorted(name for name in names if name not in wanted)
    # Older CLIP snapshots can use a single PyTorch file.
    import torch

    path = checkpoint / "pytorch_model.bin"
    if not path.is_file():
        raise FileNotFoundError(f"No supported vision model file in {checkpoint}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    names = ["clip::" + name for name in state]
    values = {
        name: tensor_digest(state[name.removeprefix("clip::")])
        for name in names if name in wanted
    }
    return values, sorted(name for name in names if name not in wanted)


def compare_original(canonical_summary: Path, original: Path, vision: Path, output: Path) -> dict[str, Any]:
    canonical = json.loads(canonical_summary.read_text(encoding="utf-8"))
    expected = canonical["tensors"]
    original_tensors, derived = pytorch_index_tensors(original)
    clip_tensors, unused_clip = vision_tensors(
        vision, {name for name in expected if name.startswith("clip::")},
    )
    original_tensors.update(clip_tensors)
    missing = sorted(set(expected) - set(original_tensors))
    extra = sorted(set(original_tensors) - set(expected))
    mismatches = {
        name: {"canonical": expected[name], "original": original_tensors[name]}
        for name in sorted(set(expected) & set(original_tensors))
        if expected[name] != original_tensors[name]
    }
    status = "pass" if not missing and not extra and not mismatches else "fail"
    payload = {
        "status": status,
        "canonical_summary": str(canonical_summary.resolve()),
        "canonical_checkpoint": canonical["checkpoint"],
        "original_checkpoint": str(original.resolve()),
        "vision_checkpoint": str(vision.resolve()),
        "expected_tensor_count": len(expected),
        "observed_tensor_count": len(original_tensors),
        "exact_tensor_matches": len(set(expected) & set(original_tensors)) - len(mismatches),
        "missing": missing,
        "extra": extra,
        "mismatches": mismatches,
        "excluded_derived_rotary_buffers": sorted(derived),
        "exclusion_reason": "Rotary inv_freq values are deterministic non-parameter buffers in the old stack and are not serialized by the converted stack.",
        "unused_clip_checkpoint_tensors": unused_clip,
        "unused_clip_reason": "The author LLaVA stack instantiates CLIPVisionModel; text-encoder/projection tensors packaged in the source CLIPModel checkpoint are not loaded or used.",
    }
    atomic_json(output, payload)
    if status != "pass":
        raise RuntimeError(
            f"Checkpoint tensor equivalence failed: missing={len(missing)} extra={len(extra)} mismatches={len(mismatches)}"
        )
    return payload


def main() -> int:
    args = parse_args()
    if args.action == "snapshot-canonical":
        payload = snapshot_canonical(args.checkpoint, args.output)
    else:
        payload = compare_original(
            args.canonical_summary, args.original_checkpoint, args.vision_checkpoint, args.output,
        )
    print(json.dumps({key: payload[key] for key in ("status", "counts", "exact_tensor_matches") if key in payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
