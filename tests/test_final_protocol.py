import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from vlm_pruning.prior_methods import visionzip_budget, visionzip_compress
from vlm_pruning.efficiency import balanced_latin_square, cache_tensor_bytes
from vlm_pruning.llava_hooks.mid_layer_pruning import MidLayerUGPruneForwarder

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_script("prepare_official_baseline_data.py")
official = load_script("run_official_baseline.py")
register = load_script("register_final_protocol.py")
campaign = load_script("run_final_evaluation.py")
selection_provenance = load_script("finalize_selection_provenance.py")
postfreeze = load_script("run_final_postfreeze.py")
timing_campaign = load_script("run_final_timing_campaign.py")
reproduction = load_script("reproduce.py")


class ReproductionRegistryTests(unittest.TestCase):
    def context(self, *, full_audit=False):
        return {
            "python": sys.executable,
            "max_parallel": 3,
            "wandb_mode": "offline",
            "dissertation_root": str(REPO_ROOT.parent / "Dissertation_Draft"),
            "full_audit": full_audit,
        }

    def test_registry_expands_every_stage_without_a_shell(self):
        registry = reproduction.load_registry()
        expected_counts = {
            "selection": 13,
            "postfreeze": 4,
            "timing": 3,
            "verify": 2,
            "ablations": 1,
        }
        for stage, expected in expected_counts.items():
            commands = reproduction.expand_stage(registry, stage, self.context())
            self.assertEqual(len(commands), expected)
            self.assertTrue(all(command[0] == sys.executable for command in commands))

    def test_all_pipeline_and_optional_pdf_audit(self):
        registry = reproduction.load_registry()
        self.assertEqual(
            reproduction.selected_stages(registry, "all"),
            ["selection", "postfreeze", "timing", "verify"],
        )
        verify = reproduction.expand_stage(
            registry,
            "verify",
            self.context(full_audit=True),
        )
        self.assertEqual(len(verify), 3)
        self.assertEqual(verify[-1][1], "scripts/final_completion_audit.py")


class StrictSplitTests(unittest.TestCase):
    def test_strict_split_requires_exact_key(self):
        dataset = {"validation": [1], "testA": [2], "testB": [3]}
        selected, actual = prepare.choose_split(dataset, "testA", strict=True)
        self.assertEqual(selected, [2])
        self.assertEqual(actual, "testA")
        with self.assertRaises(prepare.DataPreparationError):
            prepare.choose_split(dataset, "val", strict=True)

    def test_non_strict_alias_is_retained_for_development(self):
        selected, actual = prepare.choose_split({"validation": [1]}, "val")
        self.assertEqual(selected, [1])
        self.assertEqual(actual, "validation")


class FinalScorerTests(unittest.TestCase):
    def test_grounding_miou_counts_invalid_outputs_as_zero(self):
        args = SimpleNamespace(task="grounding", dataset="refcoco")
        rows = [
            {
                "valid": True,
                "invalid": False,
                "box_iou": 0.8,
                "success_iou_0_5": True,
                "raw_parsed_box": [0, 0, 1, 1],
                "latency_s": 1.0,
                "generated_token_count": 1,
            },
            {
                "valid": False,
                "invalid": True,
                "box_iou": 0.0,
                "success_iou_0_5": False,
                "raw_parsed_box": None,
                "latency_s": 1.0,
                "generated_token_count": 1,
            },
        ]
        metrics = official.aggregate(rows, args, 2.0, 1.0, 1.0)
        self.assertAlmostEqual(metrics["quality/mean_iou"], 0.4)
        self.assertAlmostEqual(metrics["quality/acc_at_0_5"], 0.5)

    def test_vqa_normalisation_and_consensus_scores(self):
        self.assertEqual(official.vqa_normalize_answer("The two, cats!"), "2 cats")
        self.assertEqual(official.vqa_normalize_answer("Don't"), "don't")
        self.assertEqual(official.vqa_normalize_answer("1,000?"), "1000")
        self.assertEqual(official.vqa_normalize_answer("3.14"), "3.14")
        self.assertEqual(official.vqa_normalize_answer("Chris's"), "chris 's")
        for matches, expected in ((0, 0.0), (1, 0.3), (2, 0.6), (3, 0.9), (4, 1.0)):
            answers = ["dakota"] * matches + ["other"] * (10 - matches)
            self.assertAlmostEqual(official.vqa_soft_accuracy("Dakota.", answers), expected)

    def test_textvqa_uses_soft_score_as_headline(self):
        sample = {"answer": ["dakota"] * 3 + ["other"] * 7}
        scored = official.score_vqa(sample, "Dakota", "textvqa")
        self.assertAlmostEqual(scored["vqa_soft_score"], 0.9)
        args = SimpleNamespace(task="vqa", dataset="textvqa")
        metrics = official.aggregate(
            [{**scored, "latency_s": 1.0, "generated_token_count": 1}],
            args,
            wall_time_s=1.0,
            peak_alloc_mb=1.0,
            peak_reserved_mb=1.0,
        )
        self.assertAlmostEqual(metrics["quality/vqa_accuracy"], 0.9)
        self.assertAlmostEqual(metrics["quality/main_score"], 0.9)

    def test_pope_reports_each_category(self):
        rows = []
        for category in ("random", "popular", "adversarial"):
            rows.extend(
                [
                    {
                        "category": category,
                        "answer_mode": "yes_no",
                        "prediction": "yes",
                        "target": "yes",
                        "correct": True,
                        "invalid": False,
                        "latency_s": 1.0,
                        "generated_token_count": 1,
                    },
                    {
                        "category": category,
                        "answer_mode": "yes_no",
                        "prediction": "no",
                        "target": "no",
                        "correct": True,
                        "invalid": False,
                        "latency_s": 1.0,
                        "generated_token_count": 1,
                    },
                ]
            )
        args = SimpleNamespace(task="vqa", dataset="pope")
        metrics = official.aggregate(
            rows,
            args,
            wall_time_s=6.0,
            peak_alloc_mb=1.0,
            peak_reserved_mb=1.0,
        )
        for category in ("random", "popular", "adversarial"):
            self.assertEqual(metrics[f"quality/pope/{category}/samples"], 2)
            self.assertEqual(metrics[f"quality/pope/{category}/accuracy"], 1.0)


class ManifestRegistrationTests(unittest.TestCase):
    def test_published_source_revisions_are_pinned(self):
        self.assertEqual(
            register.repository_revision(REPO_ROOT / "third_party/FastV"),
            "d1659729b5bf1be225e99ee15783deeea80f63b1",
        )
        self.assertEqual(
            register.repository_revision(REPO_ROOT / "third_party/VisionZip"),
            "8f86b55c6f000eb033e6912538af2dd7dcb30502",
        )

    def test_pope_ids_are_qualified_by_category(self):
        rows = [
            {"sample_id": "1", "category": "random", "metadata": {}},
            {"sample_id": "1", "category": "popular", "metadata": {}},
            {"sample_id": "1", "category": "adversarial", "metadata": {}},
        ]
        final = register.final_pope_rows(rows)
        ids = [row["sample_id"] for row in final]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "pope_random_1")
        self.assertEqual(final[0]["metadata"]["original_sample_id"], "1")

    def test_manifest_uniqueness_rejects_duplicates(self):
        with self.assertRaises(register.ProtocolError):
            register.validate_unique_ids("test", [{"sample_id": "x"}, {"sample_id": "x"}])

    def test_refcoco_ids_are_qualified_by_source_row_and_expression(self):
        rows = [
            {"sample_id": "duplicate_0", "metadata": {"source_row": 4}},
            {"sample_id": "duplicate_0", "metadata": {"source_row": 5}},
            {"sample_id": "another", "metadata": {"source_row": 5}},
        ]
        final = register.final_refcoco_rows(rows, "val")
        self.assertEqual(
            [row["sample_id"] for row in final],
            ["refcoco_val_000004_0", "refcoco_val_000005_0", "refcoco_val_000005_1"],
        )
        self.assertEqual(final[0]["metadata"]["original_sample_id"], "duplicate_0")


class FinalCampaignTests(unittest.TestCase):
    def test_factorial_contains_exactly_four_registered_cells(self):
        self.assertEqual(
            campaign.FACTORIAL,
            (
                ("query_max", "more_context"),
                ("query_max", "more_anchors"),
                ("query_pooled", "more_context"),
                ("query_pooled", "more_anchors"),
            ),
        )

    def test_gpu_memory_parser_shape_or_none(self):
        value = campaign.gpu_memory()
        self.assertTrue(
            value is None
            or set(value) == {"total_mib", "used_mib", "free_mib"}
        )

    def test_selection_provenance_command_and_semantic_signature(self):
        command = selection_provenance.smoke_commands(Path("smoke"), 20)[0]
        selection_provenance.verify_command(
            command, "query_max", "more_context"
        )
        row = {
            "sample_id": "x",
            "raw_model_output": "[0,0,1,1]",
            "retained_indices": list(range(48)),
            "box_iou": 1.0,
        }
        self.assertEqual(
            selection_provenance.semantic_signature(row),
            selection_provenance.semantic_signature(dict(row)),
        )
        with self.assertRaises(RuntimeError):
            selection_provenance.verify_command(
                command, "query_pooled", "more_context"
            )

    def test_postfreeze_registers_16_heldout_and_8_generalisation_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            old_final = postfreeze.FINAL
            try:
                postfreeze.FINAL = Path(directory)
                frozen = {
                    "method": {
                        "role_allocation": "more_context",
                        "query_scoring": "query_max",
                    },
                }
                protocol_path = Path(directory) / "protocol/protocol.json"
                protocol_path.parent.mkdir()
                protocol_path.write_text("{}")
                frozen["registered_protocol_sha256"] = hashlib.sha256(
                    protocol_path.read_bytes()
                ).hexdigest()
                (Path(directory) / "frozen_method.json").write_text(json.dumps(frozen))
                common = {
                    "wandb_mode": "disabled",
                    "wandb_project": "test",
                }
                heldout = postfreeze.entries(SimpleNamespace(phase="heldout", **common))
                general = postfreeze.entries(SimpleNamespace(phase="generalisation", **common))
                self.assertEqual(len(heldout), 16)
                self.assertEqual(len(general), 8)
                self.assertEqual(sum(item.id.startswith("final_refcoco_testA_random") for item in heldout), 3)
                self.assertEqual(sum(item.id.startswith("final_refcoco_testB_random") for item in heldout), 3)

                smoke = postfreeze.entries(
                    SimpleNamespace(phase="heldout", limit=5, **common)
                )
                self.assertEqual(len(smoke), 16)
                self.assertTrue(all(item.rows == 5 for item in smoke))
                self.assertTrue(all(item.limit == 5 for item in smoke))
                self.assertTrue(
                    all(item.id.startswith("smoke5_final_") for item in smoke)
                )
                self.assertTrue(
                    all("--limit 5" in " ".join(item.command) for item in smoke)
                )
                self.assertTrue(
                    all("--no-wandb" in item.command for item in smoke)
                )
            finally:
                postfreeze.FINAL = old_final


class PublishedAdapterTests(unittest.TestCase):
    def test_visionzip_resolves_direct_and_wrapped_clip_towers(self):
        direct = SimpleNamespace(encoder=SimpleNamespace(layers=[object()]))
        wrapped = SimpleNamespace(vision_model=direct)
        self.assertIs(official.clip_vision_backbone(direct), direct)
        self.assertIs(official.clip_vision_backbone(wrapped), direct)
        with self.assertRaises(official.BaselineError):
            official.clip_vision_backbone(SimpleNamespace())

    def test_fastv_uses_mean_head_last_token_attention(self):
        forwarder = object.__new__(MidLayerUGPruneForwarder)
        forwarder.torch = torch
        forwarder.image_token_id = 99
        forwarder.target_visual_tokens = 2
        forwarder.grid = 2
        forwarder.prune_after_layer = 3
        input_ids = torch.tensor([[1, 99, 99, 99, 99, 2]])
        attention = torch.zeros(1, 2, 6, 6)
        attention[0, 0, -1, 1:5] = torch.tensor([0.1, 0.8, 0.2, 0.5])
        attention[0, 1, -1, 1:5] = torch.tensor([0.3, 0.4, 0.2, 0.9])
        result = forwarder.fastv_selection_from_attention(
            attention, input_ids=input_ids, original_tokens=4
        )
        self.assertEqual(result.selected_indices, [1, 3])
        self.assertEqual(result.metadata["fastv_attention_layer_index"], 2)

    def test_visionzip_registered_48_token_composition(self):
        budget = visionzip_budget(48)
        self.assertEqual((budget.dominant, budget.contextual, budget.total), (40, 8, 48))

    def test_visionzip_produces_exact_budget_deterministically(self):
        patch_hidden = torch.arange(1 * 64 * 4, dtype=torch.float32).reshape(1, 64, 4)
        cls_hidden = torch.full((1, 1, 4), -7.0)
        attention = torch.arange(2 * 64, dtype=torch.float32).reshape(1, 2, 64)
        first, indices_a, diagnostics = visionzip_compress(
            torch, patch_hidden, attention, output_tokens=48, cls_hidden=cls_hidden
        )
        second, indices_b, _ = visionzip_compress(
            torch, patch_hidden, attention, output_tokens=48, cls_hidden=cls_hidden
        )
        self.assertEqual(tuple(first.shape), (1, 48, 4))
        self.assertTrue(torch.equal(first[:, :1], cls_hidden))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(indices_a, indices_b)
        self.assertEqual(len(indices_a), 48)
        self.assertEqual(len(set(indices_a)), 48)
        self.assertEqual(diagnostics["visionzip_dominant_tokens"], 40)
        self.assertEqual(diagnostics["visionzip_dominant_patch_tokens"], 39)
        self.assertEqual(diagnostics["visionzip_cls_tokens"], 1)
        self.assertEqual(diagnostics["visionzip_contextual_tokens"], 8)

    def test_visionzip_contextual_merge_uses_registered_key_metric(self):
        patch_hidden = torch.arange(12 * 3, dtype=torch.float32).reshape(1, 12, 3)
        cls_hidden = torch.full((1, 1, 3), -1.0)
        attention = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12)
        key_metric = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [-1.0, -1.0],
                    [1.0, 0.1],
                    [0.1, 1.0],
                    [0.9, 0.2],
                    [0.2, 0.9],
                    [-0.9, -0.2],
                    [-0.2, -0.9],
                ]
            ]
        )
        output, _, diagnostics = visionzip_compress(
            torch,
            patch_hidden,
            attention,
            output_tokens=10,
            cls_hidden=cls_hidden,
            merge_metric=key_metric,
        )
        # Seven high-attention patches (5..11) are dominant. The remaining
        # patches are [0..4], targets are 0 and 2, and the key metric assigns
        # patches 1/3/4 to the closest target before hidden states are averaged.
        expected_context_0 = patch_hidden[:, 0, :] + patch_hidden[:, 3:5, :].mean(dim=1)
        expected_context_1 = patch_hidden[:, 2, :] + patch_hidden[:, 1, :]
        self.assertTrue(torch.equal(output[:, -2, :], expected_context_0))
        self.assertTrue(torch.equal(output[:, -1, :], expected_context_1))
        self.assertEqual(
            diagnostics["visionzip_merge_metric"],
            "selected_clip_layer_raw_keys_mean_heads",
        )


class EfficiencyProtocolTests(unittest.TestCase):
    def test_cache_bytes_counts_shared_storage_once(self):
        tensor = torch.zeros(2, 3, dtype=torch.float32)
        cache = SimpleNamespace(layers=[SimpleNamespace(keys=tensor, values=tensor)])
        self.assertEqual(cache_tensor_bytes(cache), tensor.untyped_storage().nbytes())

    def test_cache_bytes_ignores_tensor_classes(self):
        tensor = torch.zeros(2, 3, dtype=torch.float32)
        cache = SimpleNamespace(tensor_type=torch.Tensor, tensor=tensor)
        self.assertEqual(cache_tensor_bytes(cache), tensor.untyped_storage().nbytes())

    def test_registered_latin_square_balances_every_position(self):
        methods = ["full", "k12", "final", "fastv", "visionzip"]
        orders = balanced_latin_square(methods)
        self.assertEqual(len(orders), 5)
        for position in range(5):
            self.assertEqual({order[position] for order in orders}, set(methods))

    def test_timing_smoke_and_full_campaign_registration(self):
        smoke_output, smoke_logs, smoke = timing_campaign.entries_for_mode(True)
        full_output, full_logs, full = timing_campaign.entries_for_mode(False)
        self.assertEqual(len(smoke), 5)
        self.assertEqual(len(full), 25)
        self.assertEqual(smoke_output.name, "timing_smoke")
        self.assertEqual(smoke_logs.name, "timing_smoke")
        self.assertEqual(full_output, timing_campaign.OUTPUT)
        self.assertEqual(full_logs, timing_campaign.LOGS)
        self.assertEqual(
            [(row["position"], row["method"]) for row in smoke],
            [
                (1, "full"),
                (2, "k12"),
                (3, "final"),
                (4, "fastv"),
                (5, "visionzip"),
            ],
        )
        for row in smoke:
            command = row["command"]
            self.assertIn("--limit", command)
            self.assertIn("5", command)
            self.assertIn("--wandb-mode", command)
            self.assertIn("disabled", command)
        self.assertTrue(all("--limit" not in row["command"] for row in full))

    def test_timing_completion_requires_metrics_and_exactly_1000_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            old_output = timing_campaign.OUTPUT
            old_manifest = timing_campaign.MANIFEST
            try:
                timing_campaign.OUTPUT = Path(directory)
                timing_campaign.MANIFEST = Path(directory) / "manifest.jsonl"
                run_dir = Path(directory) / "rep1_pos1_full"
                run_dir.mkdir()
                (run_dir / "metrics.json").write_text(json.dumps({"samples": 1000}))
                valid_rows = [
                    {
                        "sample_id": f"s{index}",
                        "requested_visual_tokens": 576,
                        "actual_visual_tokens": 576,
                        "repetition": 1,
                        "order_position": 1,
                        "peak_allocated_gpu_bytes": 1,
                        "peak_reserved_gpu_bytes": 1,
                        "vision_encoder_latency_s": 0.1,
                        "projector_latency_s": 0.1,
                        "selector_latency_s": 0.0,
                        "llm_prefill_latency_s": 0.2,
                        "decode_latency_s": 0.1,
                        "decode_time_per_token_s": 0.1,
                        "end_to_end_latency_s": 0.5,
                        "prompt_kv_cache_bytes": 1,
                        "final_kv_cache_bytes": 1,
                    }
                    for index in range(1000)
                ]
                timing_campaign.MANIFEST.write_text(
                    "".join(
                        json.dumps({"sample_id": row["sample_id"]}) + "\n"
                        for row in valid_rows
                    )
                )
                (run_dir / "config.json").write_text(
                    json.dumps(
                        {
                            "method": "full",
                            "repetition": 1,
                            "order_position": 1,
                            "warmup": 20,
                            "limit": 1000,
                            "checkpoint": "llava-hf/llava-1.5-7b-hf",
                            "precision": "bf16",
                            "manifest_sha256": hashlib.sha256(
                                timing_campaign.MANIFEST.read_bytes()
                            ).hexdigest(),
                        }
                    )
                )
                (run_dir / "wandb/offline-run-fixture").mkdir(parents=True)
                (run_dir / "wandb_run.json").write_text(
                    json.dumps(
                        {
                            "enabled": True,
                            "mode": "offline",
                            "id": "fixture",
                        }
                    )
                )
                (run_dir / "run_environment.json").write_text("{}")
                (run_dir / "per_sample_timing.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in valid_rows[:999])
                )
                self.assertFalse(timing_campaign.complete(1, 1, "full"))
                (run_dir / "per_sample_timing.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in valid_rows)
                )
                self.assertTrue(timing_campaign.complete(1, 1, "full"))
            finally:
                timing_campaign.OUTPUT = old_output
                timing_campaign.MANIFEST = old_manifest


if __name__ == "__main__":
    unittest.main()
