from __future__ import annotations

import copy
import unittest

import torch
import numpy as np
from torch import nn

from vlm_pruning.llava_hooks.saqr import (
    LoRALinear,
    QueryConditioner,
    SAQRTrainingBoxHead,
    SpatiallyAnchoredQueryResampler,
    deterministic_area_assignment,
    deterministic_query_keep,
    generalized_iou_loss,
    merge_projector_lora,
    normalized_hidden_alignment,
    sparse_top64_tail_kd,
    spatial_regularization,
    total_saqr_loss,
)
from scripts.analyse_saqr_campaign import hierarchical_interval, paired_interval
from scripts.run_saqr_campaign import cosine_lambda, optimizer_groups
from scripts.run_saqr_matched_timing import ORDER


class SAQRArchitectureTests(unittest.TestCase):
    def test_registered_grids_are_exact_and_mass_conserving(self):
        for grid, count in ((16, 256), (14, 196)):
            assignment = deterministic_area_assignment(24, grid)
            self.assertEqual(tuple(assignment.shape), (count, 576))
            self.assertTrue(torch.allclose(assignment.sum(dim=0), torch.ones(576), atol=1e-6))
            self.assertTrue(torch.all(assignment.sum(dim=1) > 0.0))
            self.assertAlmostEqual(float(assignment.sum()), 576.0, places=5)

    def test_mixed_forward_is_fixed_shape_mass_conserving_and_dense_gradient(self):
        module = SpatiallyAnchoredQueryResampler(32, bottleneck_dim=16)
        patches = torch.randn(2, 576, 32, requires_grad=True)
        query = torch.randn(2, 16)
        for grid, count in ((16, 256), (14, 196)):
            output, evidence, assignment = module(
                patches, query, output_grid=grid, return_assignment=True
            )
            self.assertEqual(tuple(output.shape), (2, count, 32))
            self.assertEqual(tuple(assignment.shape), (2, count, 576))
            self.assertTrue(torch.allclose(assignment.sum(dim=1), torch.ones(2, 576), atol=2e-5))
            self.assertEqual(evidence.coverage, [1.0, 1.0])
            self.assertGreaterEqual(min(evidence.gate), module.deterministic_floor)
        output.sum().backward()
        self.assertTrue(torch.all(patches.grad.abs().sum(dim=-1) > 0.0))

    def test_auxiliary_head_is_separate_from_inference_module(self):
        module = SpatiallyAnchoredQueryResampler(32, bottleneck_dim=16)
        self.assertFalse(any(isinstance(child, SAQRTrainingBoxHead) for child in module.modules()))


class SAQRQueryTests(unittest.TestCase):
    def test_query_mask_excludes_answer_tokens_and_zero_query_is_exact(self):
        conditioner = QueryConditioner(12, bottleneck_dim=8)
        conditioner.eval()
        embeddings = torch.randn(1, 7, 12)
        mask = torch.tensor([[False, True, True, True, False, False, False]])
        original, _ = conditioner(embeddings, mask)
        changed = embeddings.clone()
        changed[:, 4:, :] += 10_000.0
        repeated, _ = conditioner(changed, mask)
        self.assertTrue(torch.allclose(original, repeated))
        zero, diagnostics = conditioner(changed, mask, force_zero=True)
        self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))
        self.assertTrue(diagnostics["force_zero"])

    def test_query_dropout_is_stateless_and_reproducible(self):
        first = [deterministic_query_keep(str(i), seed=0, epoch=1, probability=0.2) for i in range(100)]
        second = [deterministic_query_keep(str(i), seed=0, epoch=1, probability=0.2) for i in range(100)]
        self.assertEqual(first, second)
        self.assertGreater(sum(first), 65)
        self.assertLess(sum(first), 95)


class SAQRLoRATests(unittest.TestCase):
    def test_lora_merge_parity_and_frozen_base(self):
        torch.manual_seed(0)
        base = nn.Linear(5, 7)
        wrapped = LoRALinear(base, rank=2)
        wrapped.lora_B.data.normal_()
        inputs = torch.randn(3, 5)
        before = wrapped(inputs)
        wrapped.merge()
        after = wrapped(inputs)
        self.assertTrue(torch.allclose(before, after, atol=2e-6, rtol=2e-6))
        self.assertFalse(wrapped.base.weight.requires_grad)

    def test_exact_two_layer_projector_merge(self):
        projector = nn.Sequential(LoRALinear(nn.Linear(4, 6), rank=2), nn.GELU(), LoRALinear(nn.Linear(6, 8), rank=2))
        for module in projector.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.data.normal_()
        inputs = torch.randn(2, 4)
        before = projector(inputs)
        merge_projector_lora(projector)
        self.assertTrue(torch.allclose(before, projector(inputs), atol=2e-6, rtol=2e-6))


class SAQRLossTests(unittest.TestCase):
    def test_sparse_kd_matches_full_category_kl(self):
        student = torch.tensor([[[0.2, 0.1, -0.3, 0.7, -0.2]]])
        teacher = torch.tensor([[[0.3, -0.4, 0.8, 0.2, -0.1]]])
        top_logits, top_indices = torch.topk(teacher, k=2, dim=-1)
        mask = torch.ones_like(teacher, dtype=torch.bool)
        mask.scatter_(-1, top_indices, False)
        tail = torch.logsumexp((teacher / 2.0).masked_fill(~mask, -torch.inf), dim=-1)
        actual = sparse_top64_tail_kd(student, top_indices, top_logits, tail, temperature=2.0)
        student_scaled = student / 2.0
        student_top = torch.gather(student_scaled, -1, top_indices)
        student_tail = torch.logsumexp(student_scaled.masked_fill(~mask, -torch.inf), dim=-1, keepdim=True)
        teacher_categories = torch.cat((top_logits / 2.0, tail[..., None]), -1)
        student_categories = torch.cat((student_top, student_tail), -1)
        probability = torch.softmax(teacher_categories, -1)
        expected = (probability * (torch.log_softmax(teacher_categories, -1) - torch.log_softmax(student_categories, -1))).sum(-1).mean() * 4.0
        self.assertTrue(torch.allclose(actual, expected))

    def test_hidden_box_spatial_and_total_loss_are_finite(self):
        hidden = normalized_hidden_alignment(
            [torch.randn(1, 3, 4) for _ in range(4)],
            [torch.randn(1, 3, 4) for _ in range(4)],
        )
        predicted = torch.tensor([[0.1, 0.2, 0.7, 0.8]])
        target = torch.tensor([[0.2, 0.1, 0.8, 0.9]])
        self.assertTrue(torch.isfinite(generalized_iou_loss(predicted, target)))
        assignment = deterministic_area_assignment(24, 14)[None]
        spatial = spatial_regularization(assignment, input_grid=24, output_grid=14)
        total, components = total_saqr_loss(
            answer_ce=torch.tensor(1.0),
            logit_kd=torch.tensor(2.0),
            hidden=hidden,
            predicted_box=predicted,
            target_box=target,
            spatial=spatial,
        )
        expected = 1.0 + 0.5 * 2.0 + 0.1 * hidden + 2.0 * components["box_l1"] + components["giou"] + 0.01 * spatial
        self.assertTrue(torch.allclose(total, expected))
        self.assertTrue(torch.isfinite(total))


class SAQRCampaignProtocolTests(unittest.TestCase):
    def test_frozen_timing_order_is_exact_and_counterbalanced(self):
        self.assertEqual(ORDER[1], ("K12_ORIGINAL", "SAQR196"))
        self.assertEqual(ORDER[2], ("SAQR196", "K12_ORIGINAL"))
        self.assertEqual(ORDER[3], ("K12_ORIGINAL", "SAQR196"))

    def test_cosine_schedule_has_warmup_and_reaches_zero(self):
        self.assertAlmostEqual(cosine_lambda(0, total_steps=20, warmup_steps=2), 0.5)
        self.assertAlmostEqual(cosine_lambda(1, total_steps=20, warmup_steps=2), 1.0)
        self.assertAlmostEqual(cosine_lambda(20, total_steps=20, warmup_steps=2), 0.0)

    def test_optimizer_groups_apply_lora_lr_and_exclude_lora_decay(self):
        module = nn.Module()
        module.normal = nn.Linear(3, 4)
        module.adapter = LoRALinear(nn.Linear(4, 5), rank=2)
        groups = optimizer_groups((("projector", module),), main_lr=2e-4, lora_lr=5e-5)
        lora_parameters = {id(module.adapter.lora_A), id(module.adapter.lora_B)}
        matching = [group for group in groups if any(id(parameter) in lora_parameters for parameter in group["params"])]
        self.assertTrue(matching)
        self.assertTrue(all(group["lr"] == 5e-5 and group["weight_decay"] == 0.0 for group in matching))

    def test_paired_and_hierarchical_bootstrap_are_deterministic(self):
        candidate = np.asarray([1.0, 0.0, 1.0, 0.0])
        reference = np.asarray([1.0, 1.0, 0.0, 0.0])
        self.assertEqual(paired_interval(candidate, reference, seed=9), paired_interval(candidate, reference, seed=9))
        matrix_candidate = np.tile(candidate, (3, 1))
        matrix_reference = np.tile(reference, (3, 1))
        first = hierarchical_interval(matrix_candidate, matrix_reference, seed=10)
        second = hierarchical_interval(matrix_candidate, matrix_reference, seed=10)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
