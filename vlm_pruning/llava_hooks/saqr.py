"""Spatially Anchored Query-Conditioned Residual Resampler (SAQR).

This module contains the fixed-shape, mass-conserving pre-LLM merger used by
the quick RefCOCO feasibility campaign.  It deliberately contains no hard
selection, top-k token routing, or data-dependent tensor shape.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from vlm_pruning.llava_hooks.early_compression import (
    _selected_vision_hidden,
    axis_area_overlap,
    nearest_cell_anchors,
)
from vlm_pruning.features.query_tokens import resolve_query_token_positions


class SAQRError(RuntimeError):
    """Raised when a frozen SAQR architecture or data contract is violated."""


def _sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_area_assignment(
    input_grid: int,
    output_grid: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``[slots, patches]`` area assignment with unit column mass."""

    overlap = axis_area_overlap(torch, input_grid, output_grid, device=device)
    assignment = torch.einsum("ri,cj->rcij", overlap, overlap).reshape(
        output_grid * output_grid, input_grid * input_grid
    )
    if not torch.allclose(
        assignment.sum(dim=0),
        torch.ones(input_grid * input_grid, device=assignment.device, dtype=assignment.dtype),
        atol=1e-12,
        rtol=0.0,
    ):
        raise SAQRError("Deterministic area assignment does not conserve patch mass")
    return assignment.to(dtype=dtype)


def raster_coordinates(grid: int, *, device: Any = None) -> torch.Tensor:
    """Return normalized cell-centre coordinates in raster order."""

    axis = (torch.arange(grid, device=device, dtype=torch.float32) + 0.5) / float(grid)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(grid * grid, 2)


def coordinate_encoding(coords: torch.Tensor) -> torch.Tensor:
    """Fixed coordinate features shared by the 16x16 and 14x14 grids."""

    values = [coords]
    for frequency in (1.0, 2.0):
        angle = coords * (2.0 * math.pi * frequency)
        values.extend((torch.sin(angle), torch.cos(angle)))
    return torch.cat(values, dim=-1)


def deterministic_query_keep(key: str, *, seed: int, epoch: int, probability: float) -> bool:
    """Stateless reproducible per-example query-dropout decision."""

    if not 0.0 <= probability < 1.0:
        raise SAQRError("Query dropout probability must be in [0, 1)")
    digest = hashlib.sha256(f"{seed}\0{epoch}\0{key}".encode("utf-8")).digest()
    uniform = int.from_bytes(digest[:8], "big") / float(2**64)
    return uniform >= probability


class QueryConditioner(nn.Module):
    """Attention-pool frozen LLM token embeddings into a 256-D query vector."""

    def __init__(self, embedding_dim: int, bottleneck_dim: int = 256, dropout: float = 0.20):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.dropout_probability = float(dropout)
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.token_projection = nn.Linear(embedding_dim, bottleneck_dim)
        self.attention_score = nn.Linear(bottleneck_dim, 1, bias=False)
        self.output_norm = nn.LayerNorm(bottleneck_dim)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        query_mask: torch.Tensor,
        *,
        dropout_keys: Sequence[str] | None = None,
        seed: int = 0,
        epoch: int = 0,
        force_zero: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if token_embeddings.ndim != 3 or query_mask.shape != token_embeddings.shape[:2]:
            raise SAQRError("Query embeddings/mask must have shapes [B,T,H] and [B,T]")
        mask = query_mask.to(device=token_embeddings.device, dtype=torch.bool)
        if not bool(torch.all(mask.any(dim=1))):
            raise SAQRError("Every query-conditioning row must contain at least one token")
        projected = torch.tanh(self.token_projection(self.input_norm(token_embeddings.float())))
        scores = self.attention_score(projected).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        pooled = self.output_norm(torch.einsum("bt,btd->bd", weights, projected))
        kept = torch.ones(pooled.shape[0], device=pooled.device, dtype=torch.bool)
        if self.training and self.dropout_probability > 0.0:
            if dropout_keys is None or len(dropout_keys) != pooled.shape[0]:
                raise SAQRError("Training query dropout requires one stable key per batch row")
            kept = torch.tensor(
                [
                    deterministic_query_keep(
                        str(key), seed=seed, epoch=epoch, probability=self.dropout_probability
                    )
                    for key in dropout_keys
                ],
                device=pooled.device,
                dtype=torch.bool,
            )
        if force_zero:
            kept.zero_()
        pooled = pooled * kept[:, None].to(dtype=pooled.dtype)
        return pooled, {
            "query_token_counts": mask.sum(dim=1).detach().cpu().tolist(),
            "query_kept": kept.detach().cpu().tolist(),
            "force_zero": bool(force_zero),
        }


@dataclass
class SAQREvidence:
    input_grid: int
    output_grid: int
    input_mass: list[float]
    output_mass: list[float]
    minimum_slot_mass: list[float]
    mean_entropy: list[float]
    coverage: list[float]
    mean_barycentre_drift: list[float]
    deterministic_floor: float
    gate: list[float]
    anchor_indices: list[int]
    anchor_checksum: str
    assignment_checksum: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class SpatiallyAnchoredQueryResampler(nn.Module):
    """Fixed-shape differentiable merger from 24x24 patches to 16x16/14x14 slots."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        bottleneck_dim: int = 256,
        input_grid: int = 24,
        deterministic_floor: float = 0.25,
        locality_scale: float = 12.0,
        global_scale: float = 0.10,
    ) -> None:
        super().__init__()
        if not 0.0 < deterministic_floor <= 1.0:
            raise SAQRError("The deterministic assignment floor must be strictly positive")
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.input_grid = int(input_grid)
        self.deterministic_floor = float(deterministic_floor)
        self.locality_scale = float(locality_scale)
        self.global_scale = float(global_scale)

        self.patch_norm = nn.LayerNorm(hidden_dim)
        self.patch_projection = nn.Linear(hidden_dim, bottleneck_dim)
        self.slot_projection = nn.Sequential(
            nn.Linear(10, bottleneck_dim), nn.GELU(), nn.Linear(bottleneck_dim, bottleneck_dim)
        )
        self.query_scale = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.gate_projection = nn.Linear(bottleneck_dim, 1)
        self.residual_norm = nn.LayerNorm(hidden_dim)
        self.residual_adapter = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        self.residual_logit = nn.Parameter(torch.tensor(-4.59511985013459))
        self.base_gate_logit = nn.Parameter(torch.tensor(1.0986122886681098))

        patch_coords = raster_coordinates(input_grid)
        self.register_buffer("patch_coordinates", patch_coords, persistent=True)
        for grid in (16, 14):
            self.register_buffer(
                f"area_assignment_{grid}",
                deterministic_area_assignment(input_grid, grid),
                persistent=True,
            )
            self.register_buffer(
                f"slot_coordinates_{grid}", raster_coordinates(grid), persistent=True
            )

    def _registered_grid(self, output_grid: int) -> tuple[torch.Tensor, torch.Tensor]:
        if output_grid not in (16, 14):
            raise SAQRError("SAQR supports only the preregistered 16x16 and 14x14 grids")
        return (
            getattr(self, f"area_assignment_{output_grid}"),
            getattr(self, f"slot_coordinates_{output_grid}"),
        )

    def forward(
        self,
        projected_patches: torch.Tensor,
        query_vector: torch.Tensor,
        *,
        output_grid: int,
        return_assignment: bool = False,
    ) -> tuple[torch.Tensor, SAQREvidence, torch.Tensor | None]:
        if projected_patches.ndim != 3:
            raise SAQRError("Projected patches must have shape [B,576,H]")
        batch, patches, hidden = projected_patches.shape
        if patches != self.input_grid**2 or hidden != self.hidden_dim:
            raise SAQRError(
                f"Expected [B,{self.input_grid**2},{self.hidden_dim}], got {tuple(projected_patches.shape)}"
            )
        if query_vector.shape != (batch, self.bottleneck_dim):
            raise SAQRError("Query vector has the wrong fixed bottleneck shape")

        deterministic, slot_coords = self._registered_grid(output_grid)
        deterministic = deterministic.to(
            device=projected_patches.device, dtype=torch.float32
        ).unsqueeze(0).expand(batch, -1, -1)
        slot_coords = slot_coords.to(device=projected_patches.device)
        patch_coords = self.patch_coordinates.to(device=projected_patches.device)

        patch_keys = F.normalize(
            self.patch_projection(self.patch_norm(projected_patches.float())), dim=-1
        )
        slot_base = self.slot_projection(coordinate_encoding(slot_coords))
        query_scale = 1.0 + 0.25 * torch.tanh(self.query_scale(query_vector.float()))
        conditioned_slots = F.normalize(slot_base[None, :, :] * query_scale[:, None, :], dim=-1)
        semantic = torch.einsum("bsd,bpd->bsp", conditioned_slots, patch_keys)
        distances = torch.sum(
            (slot_coords[:, None, :] - patch_coords[None, :, :]) ** 2, dim=-1
        )
        logits = -self.locality_scale * distances[None, :, :] + self.global_scale * semantic
        learned = torch.softmax(logits, dim=1)

        raw_gate = self.base_gate_logit + self.gate_projection(query_vector.float()).squeeze(-1)
        gate = self.deterministic_floor + (1.0 - self.deterministic_floor) * torch.sigmoid(raw_gate)
        assignment = gate[:, None, None] * deterministic + (1.0 - gate[:, None, None]) * learned
        # Autocast may form the mixture in BF16.  The architecture contract is
        # unit column mass, so restore that invariant explicitly in FP32.
        assignment = assignment.float()
        assignment = assignment / assignment.sum(dim=1, keepdim=True).clamp_min(1e-12)
        column_mass = assignment.sum(dim=1)
        if not torch.allclose(column_mass, torch.ones_like(column_mass), atol=2e-5, rtol=0.0):
            raise SAQRError("Mixed assignment failed the unit patch-mass invariant")

        slot_mass = assignment.sum(dim=2).clamp_min(1e-6)
        patches_fp32 = projected_patches.float()
        base_mass = deterministic.sum(dim=2).clamp_min(1e-6)
        base = torch.einsum("bsp,bph->bsh", deterministic, patches_fp32) / base_mass[:, :, None]
        contextual = torch.einsum("bsp,bph->bsh", assignment, patches_fp32) / slot_mass[:, :, None]
        residual = self.residual_adapter(self.residual_norm(contextual))
        output = base + torch.sigmoid(self.residual_logit) * residual

        entropy = -(assignment.clamp_min(1e-12) * assignment.clamp_min(1e-12).log()).sum(dim=1)
        barycentres = torch.einsum("bsp,pd->bsd", assignment, patch_coords) / slot_mass[:, :, None]
        drift = torch.linalg.vector_norm(barycentres - slot_coords[None, :, :], dim=-1)
        coverage = (assignment > 0.0).any(dim=1).float().mean(dim=1)
        anchors = nearest_cell_anchors(self.input_grid, output_grid)
        rounded = [
            round(float(value), 8)
            for value in deterministic[0].detach().cpu().reshape(-1).tolist()
        ]
        evidence = SAQREvidence(
            input_grid=self.input_grid,
            output_grid=output_grid,
            input_mass=[float(value) for value in column_mass.sum(dim=1).detach().cpu()],
            output_mass=[float(value) for value in slot_mass.sum(dim=1).detach().cpu()],
            minimum_slot_mass=[float(value) for value in slot_mass.min(dim=1).values.detach().cpu()],
            mean_entropy=[float(value) for value in entropy.mean(dim=1).detach().cpu()],
            coverage=[float(value) for value in coverage.detach().cpu()],
            mean_barycentre_drift=[float(value) for value in drift.mean(dim=1).detach().cpu()],
            deterministic_floor=self.deterministic_floor,
            gate=[float(value) for value in gate.detach().cpu()],
            anchor_indices=anchors,
            anchor_checksum=_sha(anchors),
            assignment_checksum=_sha(rounded),
        )
        return output.to(dtype=projected_patches.dtype), evidence, assignment if return_assignment else None


class SAQRTrainingBoxHead(nn.Module):
    """Auxiliary normalized-box predictor instantiated only by the training path."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 4),
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        raw = torch.sigmoid(self.network(visual_tokens.mean(dim=1)))
        lower = torch.minimum(raw[:, :2], raw[:, 2:])
        upper = torch.maximum(raw[:, :2], raw[:, 2:])
        return torch.cat((lower, upper), dim=-1)


class LoRALinear(nn.Module):
    """Rank-r additive adapter with exact merge/unmerge parity."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float | None = None):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise SAQRError("Projector LoRA may wrap only nn.Linear modules")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.merged = False
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def delta(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.base(inputs)
        if not self.merged:
            update = F.linear(F.linear(inputs, self.lora_A), self.lora_B) * self.scaling
            result = result + update
        return result

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight.add_(self.delta().to(self.base.weight.dtype))
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight.sub_(self.delta().to(self.base.weight.dtype))
            self.merged = False


def install_projector_lora(projector: nn.Module, *, rank: int = 8) -> list[str]:
    """Wrap both and only both existing projector linear layers."""

    candidates = [(name, module) for name, module in projector.named_modules() if isinstance(module, nn.Linear)]
    if len(candidates) != 2:
        raise SAQRError(f"Expected exactly two projector linear layers, found {len(candidates)}")
    installed: list[str] = []
    for name, module in candidates:
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = projector.get_submodule(parent_name)
        else:
            parent, child_name = projector, name
        setattr(parent, child_name, LoRALinear(module, rank=rank))
        installed.append(name)
    return installed


def merge_projector_lora(projector: nn.Module) -> None:
    wrappers = [module for module in projector.modules() if isinstance(module, LoRALinear)]
    if len(wrappers) != 2:
        raise SAQRError("Inference merge requires exactly two installed projector LoRA modules")
    for module in wrappers:
        module.merge()


def trainable_parameter_registry(*modules: tuple[str, nn.Module]) -> dict[str, int]:
    registry: dict[str, int] = {}
    for prefix, module in modules:
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                registry[f"{prefix}.{name}"] = int(parameter.numel())
    return registry


def assert_gradient_allowlist(model: nn.Module, allowed_names: Iterable[str]) -> None:
    allowed = set(allowed_names)
    unexpected = [name for name, parameter in model.named_parameters() if parameter.grad is not None and name not in allowed]
    if unexpected:
        raise SAQRError(f"Unexpected gradients outside registered allowlist: {unexpected[:8]}")


def sparse_top64_tail_kd(
    student_logits: torch.Tensor,
    teacher_top_indices: torch.Tensor,
    teacher_top_logits: torch.Tensor,
    teacher_tail_logmass: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL over cached teacher top-k categories plus one exact tail category."""

    if student_logits.shape[:-1] != teacher_top_indices.shape[:-1]:
        raise SAQRError("Student and cached-teacher token dimensions do not match")
    if teacher_top_indices.shape != teacher_top_logits.shape:
        raise SAQRError("Teacher top indices/logits must have identical shape")
    scaled_student = student_logits.float() / float(temperature)
    student_top = torch.gather(scaled_student, -1, teacher_top_indices.long())
    mask = torch.ones_like(scaled_student, dtype=torch.bool)
    mask.scatter_(-1, teacher_top_indices.long(), False)
    student_tail = torch.logsumexp(scaled_student.masked_fill(~mask, -torch.inf), dim=-1, keepdim=True)
    student_categories = torch.cat((student_top, student_tail), dim=-1)
    teacher_categories = torch.cat(
        (teacher_top_logits.float() / float(temperature), teacher_tail_logmass.float().unsqueeze(-1)),
        dim=-1,
    )
    teacher_probability = torch.softmax(teacher_categories, dim=-1)
    teacher_log_probability = torch.log_softmax(teacher_categories, dim=-1)
    student_log_probability = torch.log_softmax(student_categories, dim=-1)
    return (
        torch.sum(teacher_probability * (teacher_log_probability - student_log_probability), dim=-1).mean()
        * float(temperature) ** 2
    )


def normalized_hidden_alignment(student: Sequence[torch.Tensor], teacher: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(student) != 4 or len(teacher) != 4:
        raise SAQRError("Hidden alignment requires exactly layers 4, 12, 20, and 32")
    losses = []
    for student_hidden, teacher_hidden in zip(student, teacher, strict=True):
        if student_hidden.shape != teacher_hidden.shape:
            raise SAQRError("Hidden alignment tensors must contain the same text positions")
        losses.append(F.mse_loss(F.normalize(student_hidden.float(), dim=-1), F.normalize(teacher_hidden.float(), dim=-1)))
    return torch.stack(losses).mean()


def generalized_iou_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean generalized-IoU loss for normalized xyxy boxes."""

    intersection_lower = torch.maximum(predicted[:, :2], target[:, :2])
    intersection_upper = torch.minimum(predicted[:, 2:], target[:, 2:])
    intersection = (intersection_upper - intersection_lower).clamp_min(0).prod(dim=-1)
    predicted_area = (predicted[:, 2:] - predicted[:, :2]).clamp_min(0).prod(dim=-1)
    target_area = (target[:, 2:] - target[:, :2]).clamp_min(0).prod(dim=-1)
    union = predicted_area + target_area - intersection
    iou = intersection / union.clamp_min(1e-8)
    enclosing_lower = torch.minimum(predicted[:, :2], target[:, :2])
    enclosing_upper = torch.maximum(predicted[:, 2:], target[:, 2:])
    enclosing = (enclosing_upper - enclosing_lower).clamp_min(0).prod(dim=-1)
    giou = iou - (enclosing - union) / enclosing.clamp_min(1e-8)
    return (1.0 - giou).mean()


def spatial_regularization(
    assignment: torch.Tensor,
    *,
    input_grid: int,
    output_grid: int,
) -> torch.Tensor:
    """Penalize empty/imbalanced slots and large spatial barycentre drift."""

    slot_mass = assignment.sum(dim=-1)
    expected = float(input_grid * input_grid) / float(output_grid * output_grid)
    imbalance = ((slot_mass / expected) - 1.0).square().mean()
    collapse = F.relu(0.25 * expected - slot_mass).square().mean()
    patch_coords = raster_coordinates(input_grid, device=assignment.device)
    slot_coords = raster_coordinates(output_grid, device=assignment.device)
    barycentres = torch.einsum("bsp,pd->bsd", assignment, patch_coords) / slot_mass.clamp_min(1e-6)[..., None]
    drift = (barycentres - slot_coords[None, :, :]).square().sum(dim=-1).mean()
    return imbalance + collapse + drift


def total_saqr_loss(
    *,
    answer_ce: torch.Tensor,
    logit_kd: torch.Tensor,
    hidden: torch.Tensor,
    predicted_box: torch.Tensor,
    target_box: torch.Tensor,
    spatial: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    box_l1 = F.l1_loss(predicted_box.float(), target_box.float())
    giou = generalized_iou_loss(predicted_box.float(), target_box.float())
    components = {
        "answer_ce": answer_ce,
        "logit_kd": logit_kd,
        "hidden": hidden,
        "box_l1": box_l1,
        "giou": giou,
        "spatial": spatial,
    }
    total = answer_ce + 0.5 * logit_kd + 0.10 * hidden + 2.0 * box_l1 + giou + 0.01 * spatial
    return total, components


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prepare_saqr_inputs(
    processor: Any,
    model: Any,
    device: Any,
    image: Any,
    text_prompt: str,
    query: str,
    conditioner: QueryConditioner,
    resampler: SpatiallyAnchoredQueryResampler,
    *,
    output_grid: int = 14,
    force_zero_query: bool = False,
    cached_clip_features: torch.Tensor | None = None,
    sample_id: str | None = None,
    training_epoch: int = 0,
) -> tuple[dict[str, torch.Tensor | int], dict[str, Any], torch.Tensor | None]:
    """Build one gap-positioned dense SAQR prompt.

    Query conditioning is resolved exclusively from the registered prompt's
    referring-expression token span.  ``cached_clip_features`` is accepted by
    the training path only; inference normally computes the frozen tower.
    """

    inputs = processor(images=[image], text=text_prompt, return_tensors="pt")
    dtype = next(model.parameters()).dtype
    moved: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            value = value.to(device=device, dtype=dtype) if key == "pixel_values" else value.to(device=device)
        moved[key] = value
    input_ids = moved["input_ids"]
    attention_mask = moved.get("attention_mask", torch.ones_like(input_ids))
    image_token_id = int(getattr(model.config, "image_token_index", 32000))
    image_positions = torch.nonzero(input_ids[0] == image_token_id, as_tuple=False).flatten()
    if int(image_positions.numel()) != resampler.input_grid**2:
        raise SAQRError(f"Expected 576 image placeholders, found {image_positions.numel()}")

    resolution = resolve_query_token_positions(
        processor.tokenizer,
        text_prompt,
        query,
        input_ids=input_ids[0],
        image_token_id=image_token_id,
    )
    if not resolution.token_positions or not resolution.resolver_status.startswith(("matched", "resolved")):
        raise SAQRError(f"Could not prove query-token mask: {resolution.as_dict()}")
    query_positions = torch.tensor(resolution.token_positions, device=device, dtype=torch.long)
    if bool(torch.any(input_ids[0, query_positions] == image_token_id)):
        raise SAQRError("Query mask overlaps image placeholders")

    text_embeddings = model.get_input_embeddings()(input_ids).to(dtype=dtype)
    query_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    query_mask[0, query_positions] = True
    _sync()
    query_started = time.perf_counter()
    query_vector, query_evidence = conditioner(
        text_embeddings.detach(),
        query_mask,
        dropout_keys=[sample_id or "inference"],
        seed=0,
        epoch=training_epoch,
        force_zero=force_zero_query,
    )
    _sync()
    query_latency = time.perf_counter() - query_started

    if cached_clip_features is None:
        _sync()
        vision_started = time.perf_counter()
        with torch.no_grad():
            selected = _selected_vision_hidden(torch, model, moved["pixel_values"])
        _sync()
        vision_latency = time.perf_counter() - vision_started
    else:
        selected = cached_clip_features.to(device=device, dtype=dtype)
        if selected.ndim == 2:
            selected = selected.unsqueeze(0)
        vision_latency = 0.0
    if tuple(selected.shape[1:]) != (resampler.input_grid**2, selected.shape[-1]):
        raise SAQRError("Cached/computed CLIP features do not contain exactly 576 patches")

    _sync()
    projector_started = time.perf_counter()
    projected = model.model.multi_modal_projector(selected)
    _sync()
    projector_latency = time.perf_counter() - projector_started
    if int(projected.shape[-1]) != resampler.hidden_dim:
        raise SAQRError("Projector and SAQR hidden dimensions do not agree")

    _sync()
    resampler_started = time.perf_counter()
    resampled, resampler_evidence, assignment = resampler(
        projected, query_vector, output_grid=output_grid, return_assignment=conditioner.training
    )
    _sync()
    resampler_latency = time.perf_counter() - resampler_started

    anchors = torch.tensor(
        resampler_evidence.anchor_indices, device=device, dtype=torch.long
    )
    anchor_positions = image_positions[anchors]
    _sync()
    assembly_started = time.perf_counter()
    assembled = text_embeddings.clone()
    assembled[0, anchor_positions, :] = resampled[0].to(dtype=assembled.dtype)
    keep = input_ids[0] != image_token_id
    keep[anchor_positions] = True
    selected_positions = torch.nonzero(keep, as_tuple=False).flatten()
    compacted = {
        "input_ids": input_ids[:, selected_positions],
        "inputs_embeds": assembled[:, selected_positions, :],
        "attention_mask": attention_mask[:, selected_positions],
        "position_ids": selected_positions.unsqueeze(0),
        "selected_positions": selected_positions,
        "full_prompt_length": int(input_ids.shape[1]),
        "query_positions": query_positions,
    }
    _sync()
    assembly_latency = time.perf_counter() - assembly_started
    compact_image_count = int((compacted["input_ids"] == image_token_id).sum().item())
    if compact_image_count != output_grid**2:
        raise SAQRError("SAQR compact prompt has the wrong visual-token count")
    original_text_positions = torch.nonzero(input_ids[0] != image_token_id, as_tuple=False).flatten()
    compact_text_positions = compacted["position_ids"][0][compacted["input_ids"][0] != image_token_id]
    if not torch.equal(original_text_positions, compact_text_positions):
        raise SAQRError("SAQR changed a text token's registered position")
    diagnostics = {
        "vision_encoder_latency_s": vision_latency,
        "projector_latency_s": projector_latency,
        "query_adapter_latency_s": query_latency,
        "resampler_latency_s": resampler_latency,
        "input_assembly_latency_s": assembly_latency,
        "original_visual_tokens": resampler.input_grid**2,
        "actual_llm_visual_tokens": output_grid**2,
        "original_prompt_length": int(input_ids.shape[1]),
        "compacted_prompt_length": int(compacted["input_ids"].shape[1]),
        "position_ids": compacted["position_ids"][0].detach().cpu().tolist(),
        "query_resolution": resolution.as_dict(),
        "query_evidence": query_evidence,
        "resampler": resampler_evidence.as_dict(),
        "execution": "one_dense_uninterrupted_llm_forward",
        "selector_or_topk": False,
        "decoder_segmentation": False,
    }
    return compacted, diagnostics, assignment
