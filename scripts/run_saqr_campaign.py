#!/usr/bin/env python
"""Run the frozen single-seed SAQR cache, training, and quality cells."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for entry in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from run_baseline import apply_chat_template, box_iou, eos_token_ids, parse_box  # noqa: E402
from run_official_baseline import load_model  # noqa: E402
from vlm_pruning.efficiency import cache_tensor_bytes  # noqa: E402
from vlm_pruning.features.query_tokens import resolve_query_token_positions  # noqa: E402
from vlm_pruning.llava_hooks.early_compression import (  # noqa: E402
    ContiguousPoolForwarder,
    _selected_vision_hidden,
)
from vlm_pruning.llava_hooks.saqr import (  # noqa: E402
    LoRALinear,
    QueryConditioner,
    SAQRError,
    SAQRTrainingBoxHead,
    SpatiallyAnchoredQueryResampler,
    install_projector_lora,
    merge_projector_lora,
    normalized_hidden_alignment,
    prepare_saqr_inputs,
    sparse_top64_tail_kd,
    spatial_regularization,
    total_saqr_loss,
    trainable_parameter_registry,
)


CAMPAIGN_ROOT = REPO_ROOT / "experiments/trained_prellm_resampler_refcoco_20260817"
TRAIN_MANIFEST = REPO_ROOT / "experiments/adaptive_efficiency_refcoco_val_20260815/gate_data/refcoco_train_gate3000.jsonl"
DISCOVERY_MANIFEST = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
TRAIN_SHA256 = "8f2b567f27ccc05d1fef7b8368f1de7c454127e224c8f25f24944444d2b1d18f"
DISCOVERY_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
MODEL_SNAPSHOT = Path("/workspace/cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9")
MODEL_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
MAX_NEW_TOKENS = 48
HIDDEN_LAYERS = (4, 12, 20, 32)
REGISTERED_PROMPT = (
    'Identify the target described by this referring expression: "{query}".\n'
    "Return exactly one bounding box in normalized 0-1 image coordinates as "
    "[x_min, y_min, x_max, y_max]. Do not include any other text."
)
CACHE_ROOT = CAMPAIGN_ROOT / "training_cache"
CHECKPOINT_ROOT = CAMPAIGN_ROOT / "checkpoints"
QUALITY_ROOT = CAMPAIGN_ROOT / "quality"


class CampaignError(RuntimeError):
    """Raised when an executable campaign contract fails closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(torch: Any, path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise CampaignError(f"Blank JSONL line {line_number}: {path}")
            rows.append(json.loads(line))
    return rows


def frozen_manifest(path: Path, expected_hash: str, expected_rows: int) -> list[dict[str, Any]]:
    actual = sha256_file(path)
    if actual != expected_hash:
        raise CampaignError(f"Frozen manifest hash mismatch for {path}: {actual}")
    rows = read_jsonl(path)
    if len(rows) != expected_rows:
        raise CampaignError(f"Frozen manifest row mismatch: {len(rows)} != {expected_rows}")
    ids = [str(row["sample_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise CampaignError("Frozen manifest contains duplicate sample IDs")
    return rows


def set_determinism(torch: Any, seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_frozen_model() -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    os.environ.setdefault("HF_HOME", "/workspace/cache/huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not MODEL_SNAPSHOT.exists():
        raise CampaignError(f"Pinned local model snapshot is missing: {MODEL_SNAPSHOT}")
    return load_model(str(MODEL_SNAPSHOT), "bf16")


def normalized_box(row: dict[str, Any]) -> list[float]:
    width = float(row["width"])
    height = float(row["height"])
    x0, y0, x1, y1 = [float(value) for value in row["gt_box"]]
    box = [x0 / width, y0 / height, x1 / width, y1 / height]
    if not (0.0 <= box[0] < box[2] <= 1.0 and 0.0 <= box[1] < box[3] <= 1.0):
        raise CampaignError(f"Invalid normalized box for {row['sample_id']}: {box}")
    return box


def target_answer(row: dict[str, Any]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in normalized_box(row)) + "]"


def moved_processor_inputs(torch: Any, processor: Any, model: Any, device: Any, image: Image.Image, text_prompt: str) -> dict[str, Any]:
    inputs = processor(images=[image], text=text_prompt, return_tensors="pt")
    dtype = next(model.parameters()).dtype
    return {
        key: (value.to(device=device, dtype=dtype) if key == "pixel_values" else value.to(device=device))
        for key, value in inputs.items()
        if hasattr(value, "to")
    }


def teacher_cache_row(torch: Any, processor: Any, model: Any, device: Any, row: dict[str, Any]) -> dict[str, Any]:
    image = Image.open(row["image_path"]).convert("RGB")
    prompt = REGISTERED_PROMPT.format(query=row["query"])
    text_prompt = apply_chat_template(processor, prompt, 1)
    moved = moved_processor_inputs(torch, processor, model, device, image, text_prompt)
    input_ids = moved["input_ids"]
    attention_mask = moved.get("attention_mask", torch.ones_like(input_ids))
    image_token_id = int(getattr(model.config, "image_token_index", 32000))
    image_positions = torch.nonzero(input_ids[0] == image_token_id, as_tuple=False).flatten()
    if int(image_positions.numel()) != 576:
        raise CampaignError("Teacher prompt does not contain 576 image placeholders")
    resolution = resolve_query_token_positions(
        processor.tokenizer,
        text_prompt,
        str(row["query"]),
        input_ids=input_ids[0],
        image_token_id=image_token_id,
    )
    if not resolution.token_positions:
        raise CampaignError(f"Query mask resolution failed: {resolution.as_dict()}")

    with torch.inference_mode():
        clip_features = _selected_vision_hidden(torch, model, moved["pixel_values"])
        projected = model.model.multi_modal_projector(clip_features)
        prompt_embeddings = model.get_input_embeddings()(input_ids).to(dtype=projected.dtype).clone()
        prompt_embeddings[0, image_positions, :] = projected[0]

        generated = model.generate(
            **moved,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
        generated_ids = generated[0, input_ids.shape[1] :]
        raw_output = processor.decode(generated_ids, skip_special_tokens=True).strip()

        answer_ids_raw = processor.tokenizer(target_answer(row), add_special_tokens=False)["input_ids"]
        if answer_ids_raw and isinstance(answer_ids_raw[0], list):
            answer_ids_raw = answer_ids_raw[0]
        answer_ids = torch.tensor(answer_ids_raw, device=device, dtype=input_ids.dtype).unsqueeze(0)
        if answer_ids.numel() == 0:
            raise CampaignError("Ground-truth answer tokenization is empty")
        answer_embeddings = model.get_input_embeddings()(answer_ids).to(dtype=projected.dtype)
        full_embeddings = torch.cat((prompt_embeddings, answer_embeddings), dim=1)
        full_attention = torch.ones(full_embeddings.shape[:2], device=device, dtype=attention_mask.dtype)
        full_positions = torch.arange(full_embeddings.shape[1], device=device).unsqueeze(0)
        outputs = model(
            inputs_embeds=full_embeddings,
            attention_mask=full_attention,
            position_ids=full_positions,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    prompt_length = int(input_ids.shape[1])
    answer_count = int(answer_ids.shape[1])
    target_logits = outputs.logits[:, prompt_length - 1 : prompt_length - 1 + answer_count, :].float()
    top_logits, top_indices = torch.topk(target_logits, k=64, dim=-1)
    scaled = target_logits / 2.0
    tail_mask = torch.ones_like(scaled, dtype=torch.bool)
    tail_mask.scatter_(-1, top_indices, False)
    tail_logmass = torch.logsumexp(scaled.masked_fill(~tail_mask, -torch.inf), dim=-1)
    prompt_text_indices = torch.nonzero(input_ids[0] != image_token_id, as_tuple=False).flatten()
    answer_indices = torch.arange(prompt_length, prompt_length + answer_count, device=device)
    teacher_text_indices = torch.cat((prompt_text_indices, answer_indices))
    teacher_hidden = [
        outputs.hidden_states[layer][0, teacher_text_indices, :].to(dtype=torch.bfloat16).cpu()
        for layer in HIDDEN_LAYERS
    ]
    raw_box, predicted_box, parse_error = parse_box(
        raw_output,
        int(row["width"]),
        int(row["height"]),
        "normalized_0_1",
    )
    iou = box_iou(predicted_box, [float(value) for value in row["gt_box"]]) if predicted_box else 0.0
    del outputs, target_logits, scaled, projected, prompt_embeddings, full_embeddings
    return {
        "schema_version": 1,
        "sample_id": str(row["sample_id"]),
        "partition": str(row["training_partition"]),
        "query": str(row["query"]),
        "manifest_index": int(row["manifest_index"]),
        "clip_features": clip_features[0].to(dtype=torch.bfloat16).cpu(),
        "prompt_input_ids": input_ids[0].to(dtype=torch.int32).cpu(),
        "query_positions": torch.tensor(resolution.token_positions, dtype=torch.int32),
        "query_resolution_status": resolution.resolver_status,
        "answer_ids": answer_ids[0].to(dtype=torch.int32).cpu(),
        "gt_box_normalized": torch.tensor(normalized_box(row), dtype=torch.float32),
        "teacher_top_indices": top_indices[0].to(dtype=torch.int32).cpu(),
        "teacher_top_logits": top_logits[0].to(dtype=torch.bfloat16).cpu(),
        "teacher_tail_logmass": tail_logmass[0].to(dtype=torch.float32).cpu(),
        "teacher_hidden_layers": list(HIDDEN_LAYERS),
        "teacher_hidden": teacher_hidden,
        "teacher_text_position_ids": torch.cat(
            (prompt_text_indices, answer_indices), dim=0
        ).to(dtype=torch.int32).cpu(),
        "teacher_greedy": {
            "raw_output": raw_output,
            "generated_token_ids": generated_ids.to(dtype=torch.int32).cpu(),
            "raw_box": raw_box,
            "predicted_box": predicted_box,
            "parse_error": parse_error,
            "iou": iou,
            "correct": bool(iou >= 0.5),
        },
        "contracts": {
            "query_input": "registered_prompt_referring_expression_tokens_only",
            "answer_leakage_to_query": False,
            "teacher_leakage_to_query": False,
            "validation_data": False,
            "model_revision": MODEL_REVISION,
        },
    }


def cache_path(index: int, *, root: Path = CACHE_ROOT) -> Path:
    return root / f"shard_{index // 50:03d}" / f"row_{index:04d}.pt"


def validate_cache_payload(torch: Any, payload: dict[str, Any], row: dict[str, Any]) -> None:
    if payload["sample_id"] != str(row["sample_id"]):
        raise CampaignError("Cache row/sample identity mismatch")
    if payload["partition"] != str(row["training_partition"]):
        raise CampaignError("Cache partition mismatch")
    if tuple(payload["clip_features"].shape) != (576, 1024) or payload["clip_features"].dtype != torch.bfloat16:
        raise CampaignError("Cache CLIP feature shape/dtype mismatch")
    answer_count = int(payload["answer_ids"].numel())
    if tuple(payload["teacher_top_indices"].shape) != (answer_count, 64):
        raise CampaignError("Cache top-64 target shape mismatch")
    if len(payload["teacher_hidden"]) != 4 or payload["teacher_hidden_layers"] != list(HIDDEN_LAYERS):
        raise CampaignError("Cache hidden-layer target contract mismatch")
    text_count = int(payload["teacher_text_position_ids"].numel())
    if any(tuple(hidden.shape) != (text_count, 4096) for hidden in payload["teacher_hidden"]):
        raise CampaignError("Cache teacher text-hidden shape mismatch")
    if not payload["query_positions"].numel():
        raise CampaignError("Cache query mask is empty")
    if payload["contracts"] != {
        "query_input": "registered_prompt_referring_expression_tokens_only",
        "answer_leakage_to_query": False,
        "teacher_leakage_to_query": False,
        "validation_data": False,
        "model_revision": MODEL_REVISION,
    }:
        raise CampaignError("Cache leakage/revision contract mismatch")


def run_cache(args: argparse.Namespace) -> int:
    import torch

    rows = frozen_manifest(TRAIN_MANIFEST, TRAIN_SHA256, 3000)
    selected = rows[: args.limit] if args.limit else rows
    output_root = args.output_root or CACHE_ROOT
    index_path = output_root / "INDEX.jsonl"
    if index_path.exists() and not args.resume:
        raise CampaignError(f"Cache index already exists; use --resume: {index_path}")
    existing = read_jsonl(index_path) if index_path.exists() else []
    if [row["sample_id"] for row in existing] != [str(row["sample_id"]) for row in selected[: len(existing)]]:
        raise CampaignError("Cache resume index is not an exact manifest prefix")
    set_determinism(torch)
    torch_module, processor, model, device, environment = load_frozen_model()
    started = time.time()
    try:
        for index, row in enumerate(selected):
            path = cache_path(index, root=output_root)
            if index < len(existing):
                if not path.exists() or sha256_file(path) != existing[index]["sha256"]:
                    raise CampaignError(f"Cached prefix row failed hash validation: {path}")
                payload = torch.load(path, map_location="cpu", weights_only=False)
                validate_cache_payload(torch, payload, row)
                continue
            payload = teacher_cache_row(torch, processor, model, device, row)
            validate_cache_payload(torch, payload, row)
            atomic_torch_save(torch, path, payload)
            record = {
                "index": index,
                "sample_id": str(row["sample_id"]),
                "partition": str(row["training_partition"]),
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            append_jsonl(index_path, record)
            if (index + 1) % 10 == 0 or index + 1 == len(selected):
                elapsed = time.time() - started
                print(f"cache {index + 1}/{len(selected)} elapsed={elapsed:.1f}s", flush=True)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    final_index = read_jsonl(index_path)
    if len(final_index) != len(selected):
        raise CampaignError("Cache did not finish the requested exact prefix")
    total_bytes = sum(int(row["bytes"]) for row in final_index)
    summary = {
        "status": "COMPLETE",
        "created_at": utc_now(),
        "rows": len(final_index),
        "full_cache": len(final_index) == 3000,
        "manifest_sha256": TRAIN_SHA256,
        "partitions": {
            name: sum(row["partition"] == name for row in final_index)
            for name in ("fit", "early_stop", "calibration")
        },
        "bytes": total_bytes,
        "gib": total_bytes / 1024**3,
        "index_sha256": sha256_file(index_path),
        "environment": environment,
    }
    atomic_json(output_root / "SUMMARY.json", summary)
    return 0


def freeze_base_and_install_lora(model: Any) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    installed = install_projector_lora(model.model.multi_modal_projector, rank=8)
    if len(installed) != 2:
        raise CampaignError("Exactly two projector LoRA adapters are required")
    return installed


def frozen_parameter_checksum(torch: Any, model: Any) -> str:
    """Hash every frozen tensor's name, shape, dtype, and exact bytes."""

    digest = hashlib.sha256()
    count = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if not (name.endswith("lora_A") or name.endswith("lora_B")):
                raise CampaignError(f"Unregistered trainable model parameter: {name}")
            continue
        tensor = parameter.detach().cpu().contiguous().view(torch.uint8).numpy()
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(str(parameter.dtype).encode())
        digest.update(memoryview(tensor))
        count += 1
    if count == 0:
        raise CampaignError("Frozen model checksum saw no tensors")
    return digest.hexdigest()


def projector_lora_state(projector: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, module in projector.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    if len(state) != 4:
        raise CampaignError("Projector checkpoint must contain four LoRA tensors")
    return state


def load_projector_lora_state(projector: Any, state: dict[str, Any]) -> None:
    expected: set[str] = set()
    for name, module in projector.named_modules():
        if isinstance(module, LoRALinear):
            for suffix, parameter in (("lora_A", module.lora_A), ("lora_B", module.lora_B)):
                key = f"{name}.{suffix}"
                expected.add(key)
                parameter.data.copy_(state[key].to(device=parameter.device, dtype=parameter.dtype))
    if expected != set(state):
        raise CampaignError("Projector LoRA checkpoint key mismatch")


def save_checkpoint(
    torch: Any,
    path: Path,
    *,
    stage: str,
    epoch: int,
    resampler: Any,
    conditioner: Any,
    box_head: Any,
    projector: Any,
    optimizer: Any,
    scheduler: Any,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "model_revision": MODEL_REVISION,
        "seed": 0,
        "stage": stage,
        "epoch": epoch,
        "resampler": {key: value.detach().cpu() for key, value in resampler.state_dict().items()},
        "conditioner": {key: value.detach().cpu() for key, value in conditioner.state_dict().items()},
        "box_head": {key: value.detach().cpu() for key, value in box_head.state_dict().items()},
        "projector_lora": projector_lora_state(projector),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": metrics,
        "architecture": {"bottleneck_dim": 256, "lora_rank": 8, "deterministic_floor": 0.25},
    }
    atomic_torch_save(torch, path, payload)


def load_checkpoint(torch: Any, path: Path, resampler: Any, conditioner: Any, box_head: Any | None, projector: Any) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["model_revision"] != MODEL_REVISION or int(payload["seed"]) != 0:
        raise CampaignError("Checkpoint revision/seed mismatch")
    resampler.load_state_dict(payload["resampler"], strict=True)
    conditioner.load_state_dict(payload["conditioner"], strict=True)
    if box_head is not None:
        box_head.load_state_dict(payload["box_head"], strict=True)
    load_projector_lora_state(projector, payload["projector_lora"])
    return payload


def build_modules(torch: Any, model: Any, device: Any) -> tuple[Any, Any, Any, dict[str, int]]:
    hidden_dim = int(model.get_input_embeddings().weight.shape[1])
    conditioner = QueryConditioner(hidden_dim, bottleneck_dim=256, dropout=0.20).to(device=device)
    resampler = SpatiallyAnchoredQueryResampler(
        hidden_dim, bottleneck_dim=256, input_grid=24, deterministic_floor=0.25
    ).to(device=device)
    box_head = SAQRTrainingBoxHead(hidden_dim, bottleneck_dim=256).to(device=device)
    registry = trainable_parameter_registry(
        ("conditioner", conditioner), ("resampler", resampler), ("box_head", box_head),
        ("projector", model.model.multi_modal_projector),
    )
    if not registry or not any("lora_" in name for name in registry):
        raise CampaignError("Trainable-parameter registry omitted projector LoRA")
    return conditioner, resampler, box_head, registry


def cache_records() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    summary_path = CACHE_ROOT / "SUMMARY.json"
    index_path = CACHE_ROOT / "INDEX.jsonl"
    if not summary_path.exists() or not index_path.exists():
        raise CampaignError("The validated 3,000-row training cache is missing")
    summary = json.loads(summary_path.read_text())
    records = read_jsonl(index_path)
    if summary.get("status") != "COMPLETE" or len(records) != 3000 or not summary.get("full_cache"):
        raise CampaignError("Training cache is not a complete accepted 3,000-row artifact")
    by_id = {str(record["sample_id"]): record for record in records}
    if len(by_id) != 3000:
        raise CampaignError("Training cache index contains duplicate IDs")
    return records, by_id


def records_from_root(root: Path, *, require_full: bool) -> list[dict[str, Any]]:
    index_path = root / "INDEX.jsonl"
    summary_path = root / "SUMMARY.json"
    if not index_path.exists() or not summary_path.exists():
        raise CampaignError(f"Cache root is incomplete: {root}")
    summary = json.loads(summary_path.read_text())
    records = read_jsonl(index_path)
    if summary.get("status") != "COMPLETE" or int(summary.get("rows", -1)) != len(records):
        raise CampaignError("Cache root summary/index disagree")
    if require_full and (len(records) != 3000 or not summary.get("full_cache")):
        raise CampaignError("This operation requires the full accepted 3,000-row cache")
    return records


def cached_payload(torch: Any, record: dict[str, Any]) -> dict[str, Any]:
    path = REPO_ROOT / record["path"]
    if sha256_file(path) != record["sha256"]:
        raise CampaignError(f"Training cache row hash changed: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def run_train_smoke(args: argparse.Namespace) -> int:
    """Run one-step/resume and 20-step repeated-fit overfit checks."""

    import torch

    records = records_from_root(args.cache_root, require_full=False)
    fit = [record for record in records if record["partition"] == "fit"]
    if not fit:
        raise CampaignError("Training smoke cache contains no fit row")
    repeated = [fit[0]] * 20
    set_determinism(torch)
    torch_module, processor, model, device, environment = load_frozen_model()
    del processor
    freeze_base_and_install_lora(model)
    conditioner, resampler, box_head, registry = build_modules(torch, model, device)
    modules = (
        ("conditioner", conditioner),
        ("resampler", resampler),
        ("box_head", box_head),
        ("projector", model.model.multi_modal_projector),
    )
    optimizer = torch.optim.AdamW(optimizer_groups(modules, main_lr=2e-4, lora_lr=5e-5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_lambda(step, total_steps=20, warmup_steps=1),
    )
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    # One exact optimizer step, checkpoint, reload, and verify state restoration.
    one = run_epoch(
        torch,
        model=model,
        device=device,
        records=[fit[0]],
        conditioner=conditioner,
        resampler=resampler,
        box_head=box_head,
        output_grid=16,
        epoch=0,
        optimizer=optimizer,
        scheduler=scheduler,
        accumulation=1,
    )
    resume_path = output / "one_step_resume.pt"
    save_checkpoint(
        torch,
        resume_path,
        stage="SMOKE",
        epoch=0,
        resampler=resampler,
        conditioner=conditioner,
        box_head=box_head,
        projector=model.model.multi_modal_projector,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=one,
    )
    before = {key: value.detach().cpu().clone() for key, value in resampler.state_dict().items()}
    payload = load_checkpoint(
        torch, resume_path, resampler, conditioner, box_head, model.model.multi_modal_projector
    )
    if any(not torch.equal(before[key], value.detach().cpu()) for key, value in resampler.state_dict().items()):
        raise CampaignError("One-step checkpoint resume did not restore exact SAQR state")
    # Start a deterministic fresh model trajectory for the registered 20-step check.
    del model, conditioner, resampler, box_head, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    set_determinism(torch)
    torch_module, processor, model, device, _ = load_frozen_model()
    del processor
    freeze_base_and_install_lora(model)
    conditioner, resampler, box_head, registry = build_modules(torch, model, device)
    modules = (
        ("conditioner", conditioner),
        ("resampler", resampler),
        ("box_head", box_head),
        ("projector", model.model.multi_modal_projector),
    )
    optimizer = torch.optim.AdamW(optimizer_groups(modules, main_lr=2e-4, lora_lr=5e-5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_lambda(step, total_steps=20, warmup_steps=1),
    )
    overfit = run_epoch(
        torch,
        model=model,
        device=device,
        records=repeated,
        conditioner=conditioner,
        resampler=resampler,
        box_head=box_head,
        output_grid=16,
        epoch=0,
        optimizer=optimizer,
        scheduler=scheduler,
        accumulation=1,
    )
    if not overfit["last_5_mean"] < overfit["first_5_mean"]:
        raise CampaignError(f"20-step fit-only smoke loss did not fall: {overfit}")
    atomic_json(
        output / "SMOKE_RESULT.json",
        {
            "status": "PASS",
            "one_step": one,
            "resume_checkpoint_sha256": sha256_file(resume_path),
            "overfit_20_step": overfit,
            "trainable_parameters": registry,
            "environment": environment,
        },
    )
    return 0


def build_student_forward(
    torch: Any,
    model: Any,
    device: Any,
    payload: dict[str, Any],
    conditioner: Any,
    resampler: Any,
    box_head: Any,
    *,
    output_grid: int,
    epoch: int,
    training: bool,
) -> tuple[Any, dict[str, Any]]:
    dtype = next(model.parameters()).dtype
    input_ids = payload["prompt_input_ids"].to(device=device, dtype=torch.long).unsqueeze(0)
    image_token_id = int(getattr(model.config, "image_token_index", 32000))
    image_positions = torch.nonzero(input_ids[0] == image_token_id, as_tuple=False).flatten()
    if int(image_positions.numel()) != 576:
        raise CampaignError("Cached prompt does not contain exactly 576 image tokens")
    prompt_embeddings = model.get_input_embeddings()(input_ids).to(dtype=dtype)
    query_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    query_positions = payload["query_positions"].to(device=device, dtype=torch.long)
    query_mask[0, query_positions] = True
    query_vector, query_diagnostics = conditioner(
        prompt_embeddings.detach(),
        query_mask,
        dropout_keys=[payload["sample_id"]],
        seed=0,
        epoch=epoch,
    )
    clip_features = payload["clip_features"].to(device=device, dtype=dtype).unsqueeze(0)
    projected = model.model.multi_modal_projector(clip_features)
    visual, evidence, assignment = resampler(
        projected, query_vector, output_grid=output_grid, return_assignment=True
    )
    anchors = torch.tensor(evidence.anchor_indices, device=device, dtype=torch.long)
    anchor_positions = image_positions[anchors]
    assembled = prompt_embeddings.clone()
    assembled[0, anchor_positions, :] = visual[0]
    keep = input_ids[0] != image_token_id
    keep[anchor_positions] = True
    selected_positions = torch.nonzero(keep, as_tuple=False).flatten()
    compact_ids = input_ids[:, selected_positions]
    compact_embeddings = assembled[:, selected_positions, :]
    answer_ids = payload["answer_ids"].to(device=device, dtype=torch.long).unsqueeze(0)
    answer_embeddings = model.get_input_embeddings()(answer_ids).to(dtype=dtype)
    full_embeddings = torch.cat((compact_embeddings, answer_embeddings), dim=1)
    full_attention = torch.ones(full_embeddings.shape[:2], device=device, dtype=torch.long)
    prompt_position_ids = selected_positions.unsqueeze(0)
    answer_position_ids = torch.arange(
        input_ids.shape[1], input_ids.shape[1] + answer_ids.shape[1], device=device
    ).unsqueeze(0)
    position_ids = torch.cat((prompt_position_ids, answer_position_ids), dim=1)
    outputs = model(
        inputs_embeds=full_embeddings,
        attention_mask=full_attention,
        position_ids=position_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    compact_prompt_length = int(compact_ids.shape[1])
    answer_count = int(answer_ids.shape[1])
    student_logits = outputs.logits[:, compact_prompt_length - 1 : compact_prompt_length - 1 + answer_count, :]
    answer_ce = torch.nn.functional.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]).float(), answer_ids.reshape(-1)
    )
    kd = sparse_top64_tail_kd(
        student_logits,
        payload["teacher_top_indices"].to(device=device, dtype=torch.long).unsqueeze(0),
        payload["teacher_top_logits"].to(device=device).unsqueeze(0),
        payload["teacher_tail_logmass"].to(device=device).unsqueeze(0),
        temperature=2.0,
    )
    compact_text_indices = torch.nonzero(compact_ids[0] != image_token_id, as_tuple=False).flatten()
    student_text_indices = torch.cat(
        (
            compact_text_indices,
            torch.arange(compact_prompt_length, compact_prompt_length + answer_count, device=device),
        )
    )
    student_hidden = [outputs.hidden_states[layer][0, student_text_indices, :] for layer in HIDDEN_LAYERS]
    teacher_hidden = [hidden.to(device=device) for hidden in payload["teacher_hidden"]]
    hidden_loss = normalized_hidden_alignment(student_hidden, teacher_hidden)
    predicted_box = box_head(visual)
    target_box = payload["gt_box_normalized"].to(device=device).unsqueeze(0)
    spatial = spatial_regularization(assignment, input_grid=24, output_grid=output_grid)
    total, components = total_saqr_loss(
        answer_ce=answer_ce,
        logit_kd=kd,
        hidden=hidden_loss,
        predicted_box=predicted_box,
        target_box=target_box,
        spatial=spatial,
    )
    diagnostics = {
        "components": {name: float(value.detach().cpu()) for name, value in components.items()},
        "query": query_diagnostics,
        "mass": evidence.as_dict(),
        "position_ids": position_ids.detach().cpu().tolist(),
    }
    return total, diagnostics


def optimizer_groups(modules: Sequence[tuple[str, Any]], *, main_lr: float, lora_lr: float) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float], list[Any]] = {}
    for prefix, module in modules:
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            full_name = f"{prefix}.{name}".lower()
            is_lora = "lora_" in full_name
            lr = lora_lr if is_lora else main_lr
            no_decay = is_lora or full_name.endswith("bias") or "norm" in full_name
            groups.setdefault((lr, 0.0 if no_decay else 0.01), []).append(parameter)
    return [
        {"params": parameters, "lr": lr, "weight_decay": weight_decay}
        for (lr, weight_decay), parameters in groups.items()
    ]


def cosine_lambda(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def partition_rows(rows: list[dict[str, Any]], partition: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["partition"] == partition]
    expected = {"fit": 2160, "early_stop": 240, "calibration": 600}[partition]
    if len(selected) != expected:
        raise CampaignError(f"Cache partition {partition} has {len(selected)} rows, expected {expected}")
    return selected


def run_epoch(
    torch: Any,
    *,
    model: Any,
    device: Any,
    records: Sequence[dict[str, Any]],
    conditioner: Any,
    resampler: Any,
    box_head: Any,
    output_grid: int,
    epoch: int,
    optimizer: Any | None,
    scheduler: Any | None,
    accumulation: int,
    max_rows: int | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    conditioner.train(training)
    resampler.train(training)
    box_head.train(training)
    model.eval()
    order = list(records)
    if training:
        random.Random(1000 * output_grid + epoch).shuffle(order)
    if max_rows is not None:
        order = order[:max_rows]
    losses: list[float] = []
    components: dict[str, list[float]] = {}
    if training:
        optimizer.zero_grad(set_to_none=True)
    for row_index, record in enumerate(order):
        payload = cached_payload(torch, record)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else contextlib.nullcontext()
        )
        with torch.set_grad_enabled(training), context:
            loss, diagnostics = build_student_forward(
                torch,
                model,
                device,
                payload,
                conditioner,
                resampler,
                box_head,
                output_grid=output_grid,
                epoch=epoch,
                training=training,
            )
            if not bool(torch.isfinite(loss)):
                raise CampaignError(f"Non-finite loss at row {row_index}")
            if training:
                (loss / float(accumulation)).backward()
        losses.append(float(loss.detach().cpu()))
        for name, value in diagnostics["components"].items():
            components.setdefault(name, []).append(float(value))
        if training and ((row_index + 1) % accumulation == 0 or row_index + 1 == len(order)):
            trainable = [
                parameter
                for module in (conditioner, resampler, box_head, model.model.multi_modal_projector)
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        if (row_index + 1) % 25 == 0 or row_index + 1 == len(order):
            print(
                f"{'train' if training else 'eval'} grid={output_grid} epoch={epoch} "
                f"{row_index + 1}/{len(order)} mean_loss={statistics.fmean(losses):.6f}",
                flush=True,
            )
        del payload, loss
    return {
        "rows": len(order),
        "mean_loss": statistics.fmean(losses),
        "first_5_mean": statistics.fmean(losses[:5]),
        "last_5_mean": statistics.fmean(losses[-5:]),
        "components": {name: statistics.fmean(values) for name, values in components.items()},
    }


def run_train(args: argparse.Namespace) -> int:
    import torch

    manifest_rows = frozen_manifest(TRAIN_MANIFEST, TRAIN_SHA256, 3000)
    records, by_id = cache_records()
    ordered_records = [by_id[str(row["sample_id"])] for row in manifest_rows]
    fit = partition_rows(ordered_records, "fit")
    early = partition_rows(ordered_records, "early_stop")
    set_determinism(torch)
    torch_module, processor, model, device, environment = load_frozen_model()
    del processor
    installed = freeze_base_and_install_lora(model)
    conditioner, resampler, box_head, registry = build_modules(torch, model, device)
    frozen_checksum_before = frozen_parameter_checksum(torch, model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    stages = (
        ("A", 16, 2, 2e-4, 5e-5),
        ("B", 14, 2, 1e-4, 2.5e-5),
    )
    training_summary: dict[str, Any] = {
        "status": "RUNNING",
        "seed": 0,
        "trainable_parameters": registry,
        "trainable_parameter_count": sum(registry.values()),
        "installed_projector_layers": installed,
        "environment": environment,
        "frozen_parameter_checksum_before": frozen_checksum_before,
        "stages": {},
    }
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    for stage, output_grid, epochs, main_lr, lora_lr in stages:
        if stage == "B":
            selected_a = CHECKPOINT_ROOT / "stage_A_selected.pt"
            if not selected_a.exists():
                raise CampaignError("Stage B requires the selected Stage-A checkpoint")
            load_checkpoint(torch, selected_a, resampler, conditioner, box_head, model.model.multi_modal_projector)
        total_steps = math.ceil((args.max_fit_rows or len(fit)) / 8) * epochs
        warmup_steps = max(1, int(math.ceil(total_steps * 0.05)))
        modules = (
            ("conditioner", conditioner),
            ("resampler", resampler),
            ("box_head", box_head),
            ("projector", model.model.multi_modal_projector),
        )
        optimizer = torch.optim.AdamW(optimizer_groups(modules, main_lr=main_lr, lora_lr=lora_lr))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step, total=total_steps, warmup=warmup_steps: cosine_lambda(
                step, total_steps=total, warmup_steps=warmup
            ),
        )
        stage_metrics: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            started = time.time()
            fit_metrics = run_epoch(
                torch,
                model=model,
                device=device,
                records=fit,
                conditioner=conditioner,
                resampler=resampler,
                box_head=box_head,
                output_grid=output_grid,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                accumulation=8,
                max_rows=args.max_fit_rows,
            )
            early_metrics = run_epoch(
                torch,
                model=model,
                device=device,
                records=early,
                conditioner=conditioner,
                resampler=resampler,
                box_head=box_head,
                output_grid=output_grid,
                epoch=epoch,
                optimizer=None,
                scheduler=None,
                accumulation=8,
                max_rows=args.max_early_rows,
            )
            metrics = {
                "epoch": epoch,
                "fit": fit_metrics,
                "early_stop": early_metrics,
                "wall_time_s": time.time() - started,
            }
            stage_metrics.append(metrics)
            checkpoint = CHECKPOINT_ROOT / f"stage_{stage}_epoch_{epoch}.pt"
            save_checkpoint(
                torch,
                checkpoint,
                stage=stage,
                epoch=epoch,
                resampler=resampler,
                conditioner=conditioner,
                box_head=box_head,
                projector=model.model.multi_modal_projector,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
            )
            metrics["checkpoint"] = str(checkpoint.relative_to(REPO_ROOT))
            metrics["checkpoint_sha256"] = sha256_file(checkpoint)
            training_summary["stages"][stage] = stage_metrics
            atomic_json(CHECKPOINT_ROOT / "TRAINING_SUMMARY.partial.json", training_summary)
        selected_metrics = min(stage_metrics, key=lambda item: (item["early_stop"]["mean_loss"], item["epoch"]))
        selected_path = REPO_ROOT / selected_metrics["checkpoint"]
        selected_payload = torch.load(selected_path, map_location="cpu", weights_only=False)
        destination = CHECKPOINT_ROOT / f"stage_{stage}_selected.pt"
        atomic_torch_save(torch, destination, selected_payload)
        training_summary["stages"][stage] = {
            "epochs": stage_metrics,
            "selected_epoch": selected_metrics["epoch"],
            "selected_checkpoint": str(destination.relative_to(REPO_ROOT)),
            "selected_checkpoint_sha256": sha256_file(destination),
        }
    training_summary["status"] = "COMPLETE"
    training_summary["frozen_parameter_checksum_after"] = frozen_parameter_checksum(torch, model)
    if training_summary["frozen_parameter_checksum_after"] != frozen_checksum_before:
        raise CampaignError("A frozen LLM/vision/projector-base parameter changed during training")
    training_summary["completed_at"] = utc_now()
    atomic_json(CHECKPOINT_ROOT / "TRAINING_SUMMARY.json", training_summary)
    return 0


def inference_modules(torch: Any, model: Any, device: Any, checkpoint: Path) -> tuple[Any, Any]:
    freeze_base_and_install_lora(model)
    conditioner, resampler, box_head, registry = build_modules(torch, model, device)
    load_checkpoint(torch, checkpoint, resampler, conditioner, None, model.model.multi_modal_projector)
    del box_head
    conditioner.eval()
    resampler.eval()
    merge_projector_lora(model.model.multi_modal_projector)
    if any(isinstance(module, SAQRTrainingBoxHead) for module in resampler.modules()):
        raise CampaignError("Training-only box head leaked into inference graph")
    return conditioner, resampler


def run_forward_smoke(args: argparse.Namespace) -> int:
    """Exercise untrained SAQR256/196 generation without creating a result cell."""

    import torch

    rows = frozen_manifest(TRAIN_MANIFEST, TRAIN_SHA256, 3000)[:2]
    set_determinism(torch)
    torch_module, processor, model, device, environment = load_frozen_model()
    freeze_base_and_install_lora(model)
    conditioner, resampler, box_head, registry = build_modules(torch, model, device)
    del box_head
    conditioner.eval()
    resampler.eval()
    merge_projector_lora(model.model.multi_modal_projector)
    forwarder = ContiguousPoolForwarder(torch, model)
    eos_ids = eos_token_ids(model, processor)
    results: list[dict[str, Any]] = []
    for grid in (16, 14):
        for row in rows:
            image = Image.open(row["image_path"]).convert("RGB")
            prompt = REGISTERED_PROMPT.format(query=row["query"])
            text_prompt = apply_chat_template(processor, prompt, 1)
            with torch.inference_mode():
                compacted, diagnostics, assignment = prepare_saqr_inputs(
                    processor,
                    model,
                    device,
                    image,
                    text_prompt,
                    str(row["query"]),
                    conditioner,
                    resampler,
                    output_grid=grid,
                    sample_id=str(row["sample_id"]),
                )
                generated, generation = forwarder.generate_natural(
                    compacted, max_new_tokens=MAX_NEW_TOKENS, eos_ids=eos_ids
                )
            expected_prompt_length = int(compacted["input_ids"].shape[1])
            if generation["cache/prompt_per_layer_lengths"] != [expected_prompt_length] * 32:
                raise CampaignError("Forward smoke prompt cache is not one contiguous dense cache")
            expected_final = expected_prompt_length + max(int(generated.numel()) - 1, 0)
            if generation["cache/final_per_layer_lengths"] != [expected_final] * 32:
                raise CampaignError("Forward smoke cache growth is inconsistent")
            if diagnostics["resampler"]["input_mass"] != [576.0] or diagnostics["resampler"]["output_mass"] != [576.0]:
                raise CampaignError("Forward smoke violated SAQR mass conservation")
            results.append({
                "sample_id": str(row["sample_id"]),
                "grid": grid,
                "visual_tokens": grid * grid,
                "generated_token_count": int(generated.numel()),
                "raw_output": processor.decode(generated, skip_special_tokens=True).strip(),
                "diagnostics": diagnostics,
                "cache": generation,
            })
            image.close()
    atomic_json(
        args.output_root / "FORWARD_SMOKE.json",
        {
            "status": "PASS",
            "scope": "untrained_architecture_only_not_a_quality_result",
            "rows_per_grid": 2,
            "grids": [16, 14],
            "trainable_parameters": registry,
            "environment": environment,
            "results": results,
        },
    )
    return 0


def quality_rows(split: str) -> tuple[list[dict[str, Any]], Path, str]:
    if split == "calibration":
        rows = frozen_manifest(TRAIN_MANIFEST, TRAIN_SHA256, 3000)
        return [row for row in rows if row["training_partition"] == "calibration"], TRAIN_MANIFEST, TRAIN_SHA256
    rows = frozen_manifest(DISCOVERY_MANIFEST, DISCOVERY_SHA256, 1000)
    return rows, DISCOVERY_MANIFEST, DISCOVERY_SHA256


def validate_quality_artifact(path: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    predictions = read_jsonl(path)
    if [str(item["sample_id"]) for item in predictions] != [str(row["sample_id"]) for row in rows]:
        raise CampaignError("Quality predictions are not an exact manifest-order match")
    rescored = []
    for prediction, row in zip(predictions, rows, strict=True):
        _, box, error = parse_box(
            prediction["raw_output"], int(row["width"]), int(row["height"]), "normalized_0_1"
        )
        iou = box_iou(box, [float(value) for value in row["gt_box"]]) if box else 0.0
        if abs(iou - float(prediction["iou"])) > 1e-9 or bool(iou >= 0.5) != bool(prediction["correct"]):
            raise CampaignError("Independent quality rescoring disagrees with stored row")
        rescored.append((error is None, iou))
    return {
        "rows": len(rescored),
        "parse_rate": statistics.fmean(float(parsed) for parsed, _ in rescored),
        "acc_at_0_5": statistics.fmean(float(iou >= 0.5) for _, iou in rescored),
        "miou": statistics.fmean(iou for _, iou in rescored),
        "median_iou": statistics.median(iou for _, iou in rescored),
        "predictions_sha256": sha256_file(path),
    }


def run_quality(args: argparse.Namespace) -> int:
    import torch

    rows, manifest_path, manifest_hash = quality_rows(args.split)
    if args.limit:
        rows = rows[: args.limit]
    checkpoint = args.checkpoint or (CHECKPOINT_ROOT / "stage_B_selected.pt")
    if not checkpoint.exists():
        raise CampaignError(f"Selected SAQR196 checkpoint is missing: {checkpoint}")
    suffix = "_no_query" if args.no_query else ""
    run_name = args.run_name or f"{args.split}_saqr196{suffix}_attempt1"
    run_dir = (args.output_root or QUALITY_ROOT) / run_name
    predictions_path = run_dir / "predictions.jsonl"
    existing = read_jsonl(predictions_path) if predictions_path.exists() and args.resume else []
    if predictions_path.exists() and not args.resume:
        raise CampaignError(f"Quality artifact exists; use --resume: {predictions_path}")
    if [item["sample_id"] for item in existing] != [str(row["sample_id"]) for row in rows[: len(existing)]]:
        raise CampaignError("Quality resume artifact is not an exact manifest prefix")
    set_determinism(torch)
    torch_module, processor, model, device, environment = load_frozen_model()
    conditioner, resampler = inference_modules(torch, model, device, checkpoint)
    forwarder = ContiguousPoolForwarder(torch, model)
    eos_ids = eos_token_ids(model, processor)
    config = {
        "status": "RUNNING",
        "cell": "Q_SAQR196_S0_NO_QUERY" if args.no_query else ("C_SAQR196_S0" if args.split == "calibration" else "Q_SAQR196_S0"),
        "split": args.split,
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_sha256": manifest_hash,
        "rows": len(rows),
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "force_zero_query": bool(args.no_query),
        "output_grid": 14,
        "environment": environment,
    }
    atomic_json(run_dir / "config.json", config)
    for index, row in enumerate(rows):
        if index < len(existing):
            continue
        image = Image.open(row["image_path"]).convert("RGB")
        prompt = REGISTERED_PROMPT.format(query=row["query"])
        text_prompt = apply_chat_template(processor, prompt, 1)
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        started = time.perf_counter()
        with torch.inference_mode():
            compacted, diagnostics, assignment = prepare_saqr_inputs(
                processor,
                model,
                device,
                image,
                text_prompt,
                str(row["query"]),
                conditioner,
                resampler,
                output_grid=14,
                force_zero_query=args.no_query,
                sample_id=str(row["sample_id"]),
            )
            generated_ids, generation = forwarder.generate_natural(
                compacted, max_new_tokens=MAX_NEW_TOKENS, eos_ids=eos_ids
            )
        raw_output = processor.decode(generated_ids, skip_special_tokens=True).strip()
        total_wall = time.perf_counter() - started
        raw_box, predicted_box, parse_error = parse_box(
            raw_output, int(row["width"]), int(row["height"]), "normalized_0_1"
        )
        iou = box_iou(predicted_box, [float(value) for value in row["gt_box"]]) if predicted_box else 0.0
        result = {
            "index": index,
            "sample_id": str(row["sample_id"]),
            "raw_output": raw_output,
            "generated_token_ids": generated_ids.detach().cpu().tolist(),
            "generated_token_count": int(generated_ids.numel()),
            "raw_box": raw_box,
            "predicted_box": predicted_box,
            "parse_error": parse_error,
            "iou": iou,
            "correct": bool(iou >= 0.5),
            "diagnostics": diagnostics,
            "cache": generation,
            "uncontrolled_wall_time_s": total_wall,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
        }
        append_jsonl(predictions_path, result)
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            print(f"quality {run_name} {index + 1}/{len(rows)}", flush=True)
    metrics = validate_quality_artifact(predictions_path, rows)
    config["status"] = "COMPLETE"
    config["completed_at"] = utc_now()
    atomic_json(run_dir / "config.json", config)
    atomic_json(run_dir / "metrics.json", metrics)
    return 0


def validate_cache_command(args: argparse.Namespace) -> int:
    import torch

    rows = frozen_manifest(TRAIN_MANIFEST, TRAIN_SHA256, 3000)
    index = read_jsonl(CACHE_ROOT / "INDEX.jsonl")
    if len(index) != 3000:
        raise CampaignError("Cache validator requires exactly 3,000 indexed rows")
    for position, (record, row) in enumerate(zip(index, rows, strict=True)):
        if int(record["index"]) != position or record["sample_id"] != str(row["sample_id"]):
            raise CampaignError("Cache index ordering/identity mismatch")
        payload = cached_payload(torch, record)
        validate_cache_payload(torch, payload, row)
    evidence = {
        "status": "PASS",
        "rows": 3000,
        "index_sha256": sha256_file(CACHE_ROOT / "INDEX.jsonl"),
        "validated_at": utc_now(),
    }
    atomic_json(CACHE_ROOT / "INDEPENDENT_VALIDATION.json", evidence)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--limit", type=int)
    cache.add_argument("--output-root", type=Path)
    cache.add_argument("--resume", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--max-fit-rows", type=int)
    train.add_argument("--max-early-rows", type=int)
    smoke = sub.add_parser("train-smoke")
    smoke.add_argument("--cache-root", type=Path, required=True)
    smoke.add_argument("--output-root", type=Path, required=True)
    forward_smoke = sub.add_parser("forward-smoke")
    forward_smoke.add_argument("--output-root", type=Path, required=True)
    quality = sub.add_parser("quality")
    quality.add_argument("--split", choices=["calibration", "discovery"], required=True)
    quality.add_argument("--no-query", action="store_true")
    quality.add_argument("--checkpoint", type=Path)
    quality.add_argument("--limit", type=int)
    quality.add_argument("--output-root", type=Path)
    quality.add_argument("--run-name")
    quality.add_argument("--resume", action="store_true")
    sub.add_parser("validate-cache")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "cache":
        return run_cache(args)
    if args.command == "train":
        return run_train(args)
    if args.command == "train-smoke":
        return run_train_smoke(args)
    if args.command == "forward-smoke":
        return run_forward_smoke(args)
    if args.command == "quality":
        return run_quality(args)
    if args.command == "validate-cache":
        return validate_cache_command(args)
    raise CampaignError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CampaignError, SAQRError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
