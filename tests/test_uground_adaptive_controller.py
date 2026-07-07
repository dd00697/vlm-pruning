from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vlm_pruning.controllers import (
    ACCURACY_FIRST_REWARD_MODE,
    AdaptiveAction,
    PRIMARY_BASELINE_ACTION,
    action_budget,
    action_grid,
    action_name,
    allowed_action_indices,
    assert_allowed_nonfinal_rows,
    build_selector_config,
    coerce_role_preset,
    extract_feature_dict,
    is_v6_rescue_allowed,
    is_reserved_refcoco500_row,
    oracle_utility,
    parse_action_config,
    policy_constraint_decision,
    reward_for_mode,
    role_ratios,
    state_feature_names,
    state_feature_values,
    state_feature_values_for_names,
)


class UGroundAdaptiveActionTests(unittest.TestCase):
    def test_action_name_and_role_presets_are_stable(self) -> None:
        action = AdaptiveAction(prune_after_layer=10, target_visual_tokens=58, role_preset="more-context")

        self.assertEqual(coerce_role_preset("more-context"), "more_context")
        self.assertEqual(action_name(action), "k10_b58_more_context_soft_average_query_max")
        self.assertEqual(role_ratios("more_context"), (0.45, 0.40, 0.15))

    def test_build_selector_config_applies_action_knobs(self) -> None:
        config = build_selector_config(
            AdaptiveAction(
                prune_after_layer=12,
                target_visual_tokens=72,
                map_source="query_pooled",
                fusion_mode="single_layer",
                role_preset="more_anchors",
            )
        )

        self.assertEqual(config.probe_layers, (12,))
        self.assertEqual(config.map_source, "query_pooled")
        self.assertEqual(config.fusion_mode, "single_layer")
        self.assertAlmostEqual(config.target_ratio, 0.45)
        self.assertAlmostEqual(config.context_ratio, 0.25)
        self.assertAlmostEqual(config.anchor_ratio, 0.30)

    def test_coverage_zip_action_name_and_config_round_trip(self) -> None:
        action = AdaptiveAction(
            prune_after_layer=12,
            target_visual_tokens=58,
            role_preset="default",
            selection_strategy="coverage_zip",
        )

        self.assertEqual(action.name, "k12_b58_default_soft_average_query_max_coverage_zip")
        parsed = parse_action_config(action.name)
        config = build_selector_config(action)

        self.assertEqual(parsed.selection_strategy, "coverage_zip")
        self.assertEqual(parsed.target_visual_tokens, 58)
        self.assertEqual(config.selection_strategy, "coverage_zip")
        self.assertAlmostEqual(config.zip_alpha, 0.20)

    def test_action_grid_expands_compact_grid(self) -> None:
        actions = action_grid(
            ks=(10, 12),
            budgets=(48, 58),
            role_presets=("default", "more_context"),
            fusion_modes=("soft_average",),
            map_sources=("query_max",),
        )

        self.assertEqual(len(actions), 8)
        self.assertEqual(len({action.name for action in actions}), 8)

    def test_action_grid_can_expand_selection_strategies(self) -> None:
        actions = action_grid(
            ks=(12,),
            budgets=(48,),
            role_presets=("default",),
            selection_strategies=("standard", "coverage_zip"),
        )

        self.assertEqual(
            [action.name for action in actions],
            [
                "k12_b48_default_soft_average_query_max",
                "k12_b48_default_soft_average_query_max_coverage_zip",
            ],
        )


class UGroundAdaptiveProtocolTests(unittest.TestCase):
    def test_reserved_refcoco500_guard_detects_source_rows(self) -> None:
        reserved = {
            "sample_id": "reserved",
            "dataset": "refcoco",
            "split": "val",
            "metadata": {"source_split": "val", "source_row": 42},
        }
        allowed = {
            "sample_id": "allowed",
            "dataset": "refcoco",
            "split": "val",
            "metadata": {"source_split": "val", "source_row": 7736},
        }

        self.assertTrue(is_reserved_refcoco500_row(reserved))
        self.assertFalse(is_reserved_refcoco500_row(allowed))
        assert_allowed_nonfinal_rows([allowed], context="unit-test")
        with self.assertRaises(ValueError):
            assert_allowed_nonfinal_rows([reserved], context="unit-test")

    def test_oracle_utility_penalizes_latency_tokens_and_parse_failures(self) -> None:
        good = oracle_utility(iou=0.7, normalized_latency=0.8, token_fraction=0.1, parse_failed=False)
        slow = oracle_utility(iou=0.7, normalized_latency=1.4, token_fraction=0.1, parse_failed=False)
        failed = oracle_utility(iou=0.7, normalized_latency=0.8, token_fraction=0.1, parse_failed=True)

        self.assertGreater(good, slow)
        self.assertGreater(good, failed)

    def test_feature_extraction_is_numeric_and_zero_filled(self) -> None:
        features = extract_feature_dict(
            {
                "ug/map_entropy_norm": "0.9",
                "ug/map_peakiness": 2.5,
                "latency_s": None,
            }
        )

        self.assertAlmostEqual(features["ug/map_entropy_norm"], 0.9)
        self.assertAlmostEqual(features["ug/map_peakiness"], 2.5)
        self.assertEqual(features["latency_s"], 0.0)


class UGroundPolicyLearningTests(unittest.TestCase):
    def test_parse_primary_action_and_action_features(self) -> None:
        parsed = parse_action_config(PRIMARY_BASELINE_ACTION)

        self.assertEqual(parsed.prune_after_layer, 12)
        self.assertEqual(parsed.target_visual_tokens, 48)
        self.assertEqual(parsed.role_preset, "more_context")
        self.assertEqual(parsed.selection_strategy, "standard")
        self.assertEqual(action_budget(PRIMARY_BASELINE_ACTION), 48)

    def test_v6_rescue_task_mask_rules(self) -> None:
        actions = [
            "k12_b48_default_soft_average_query_max",
            "k12_b48_default_soft_average_query_max_coverage_zip",
            "k12_b144_more_context_soft_average_query_max",
            "k12_b288_more_context_soft_average_query_max",
        ]

        self.assertTrue(is_v6_rescue_allowed(actions[2], {"task_family": "grounding"}))
        self.assertFalse(is_v6_rescue_allowed(actions[3], {"task_family": "grounding"}))
        self.assertTrue(is_v6_rescue_allowed(actions[3], {"task_family": "open_vqa"}))
        self.assertFalse(is_v6_rescue_allowed(actions[2], {"task_family": "yes_no_vqa"}))
        self.assertEqual(
            allowed_action_indices(actions, {"task_family": "yes_no_vqa"}, {"task_action_mask": "v6_rescue"}),
            [0, 1],
        )

    def test_rich_state_feature_vector_has_stable_shape(self) -> None:
        row = {
            "query": "man left in red shirt",
            "ug/map_entropy_norm": 0.8,
            "ug/map_peakiness": 3.0,
            "ug/map_downsample_6x6": [0.1] * 36,
            "ug/map_selected_mask_6x6": [0.0] * 36,
        }
        names = state_feature_names()
        values = state_feature_values(row)

        self.assertEqual(len(values), len(names))
        self.assertGreater(len(values), 180)
        self.assertIn(1.0, values[-5:])

    def test_checkpoint_feature_schema_can_use_old_feature_names(self) -> None:
        row = {
            "question": "what color is the sign?",
            "dataset": "textvqa",
            "ug/map_downsample_6x6": [0.1] * 36,
        }
        old_names = [name for name in state_feature_names() if not name.startswith("task/")]
        values = state_feature_values_for_names(row, old_names)

        self.assertEqual(len(values), len(old_names))

    def test_accuracy_first_reward_prefers_success(self) -> None:
        fail = {
            "box_iou": 0.49,
            "success_iou_0_5": False,
            "normalized_latency": 0.8,
            "token_fraction": 48 / 576,
            "valid": True,
        }
        success = dict(fail)
        success["box_iou"] = 0.51
        success["success_iou_0_5"] = True

        self.assertGreater(
            reward_for_mode(success, ACCURACY_FIRST_REWARD_MODE),
            reward_for_mode(fail, ACCURACY_FIRST_REWARD_MODE),
        )

    def test_policy_script_final_grid_guard(self) -> None:
        from scripts.train_ug_prune_policy import ensure_nonfinal_grid
        from run_official_baseline import write_json

        with TemporaryDirectory() as tmp:
            grid_dir = Path(tmp)
            write_json(grid_dir / "config.json", {"purpose": "final"})
            with self.assertRaises(Exception):
                ensure_nonfinal_grid(grid_dir, context="unit-test")

    def test_policy_constraint_blocks_b96_when_margin_is_small(self) -> None:
        actions = [
            "k12_b72_more_context_soft_average_query_max",
            "k12_b96_default_soft_average_query_max",
        ]

        decision = policy_constraint_decision(
            [0.50, 0.53],
            actions,
            {"ug/map_entropy_norm": 0.99, "ug/map_peakiness": 1.0},
            {"b96_policy": "q_margin", "b96_min_margin": 0.05},
        )

        self.assertEqual(decision.selected_action, actions[0])
        self.assertEqual(decision.raw_best_action, actions[1])
        self.assertFalse(decision.b96_allowed)

    def test_policy_constraint_allows_b96_for_uncertain_high_margin_sample(self) -> None:
        actions = [
            "k12_b72_more_context_soft_average_query_max",
            "k12_b96_default_soft_average_query_max",
        ]

        decision = policy_constraint_decision(
            [0.50, 0.62],
            actions,
            {"ug/map_entropy_norm": 0.97, "ug/map_peakiness": 1.2},
            {
                "b96_policy": "uncertainty",
                "b96_min_margin": 0.05,
                "b96_entropy_min": 0.95,
                "b96_peakiness_max": 2.0,
            },
        )

        self.assertEqual(decision.selected_action, actions[1])
        self.assertTrue(decision.b96_allowed)

    def test_policy_constraint_masks_v6_pope_rescue_actions(self) -> None:
        actions = [
            "k12_b48_default_soft_average_query_max",
            "k12_b288_more_context_soft_average_query_max",
        ]

        decision = policy_constraint_decision(
            [0.1, 10.0],
            actions,
            {"dataset": "pope", "task_family": "yes_no_vqa"},
            {"task_action_mask": "v6_rescue"},
        )

        self.assertEqual(decision.raw_best_action, actions[1])
        self.assertEqual(decision.selected_action, actions[0])
        self.assertIn("task_action_mask", decision.block_reasons)

    def test_policy_script_named_action_sets_and_custom_reward(self) -> None:
        from scripts.train_ug_prune_policy import ACTION_SETS, RewardRun, custom_reward

        self.assertIn("strict6", ACTION_SETS)
        self.assertTrue(all("_b96_" not in action for action in ACTION_SETS["strict6"]))
        self.assertIn("v6_rescue", ACTION_SETS)
        self.assertIn("k12_b288_more_context_soft_average_query_max", ACTION_SETS["v6_rescue"])

        row = {
            "box_iou": 0.7,
            "success_iou_0_5": True,
            "normalized_latency": 1.2,
            "token_fraction": 96 / 576,
            "valid": True,
        }
        loose = custom_reward(row, RewardRun(name="loose", base="iou", latency_weight=0.05, token_weight=0.02, parse_failure_penalty=0.25))
        strict = custom_reward(row, RewardRun(name="strict", base="iou", latency_weight=0.20, token_weight=0.20, parse_failure_penalty=0.25))

        self.assertLess(strict, loose)

    def test_policy_script_task_rewards_use_vqa_scores(self) -> None:
        from scripts.train_ug_prune_policy import TASK_ACCURACY_REWARD_MODE, TASK_QUALITY_REWARD_MODE, RewardRun, custom_reward

        textvqa = {
            "dataset": "textvqa",
            "task_family": "open_vqa",
            "score": 1.0,
            "correct": True,
            "normalized_latency": 1.0,
            "token_fraction": 58 / 576,
            "valid": True,
        }
        pope = {
            "dataset": "pope",
            "task_family": "yes_no_vqa",
            "score": 0.0,
            "correct": False,
            "normalized_latency": 1.0,
            "token_fraction": 58 / 576,
            "valid": True,
        }

        self.assertGreater(
            custom_reward(textvqa, RewardRun(name="textvqa", base=TASK_QUALITY_REWARD_MODE, latency_weight=0.0, token_weight=0.0)),
            custom_reward(pope, RewardRun(name="pope", base=TASK_QUALITY_REWARD_MODE, latency_weight=0.0, token_weight=0.0)),
        )
        self.assertGreater(
            custom_reward(textvqa, RewardRun(name="textvqa", base=TASK_ACCURACY_REWARD_MODE, latency_weight=0.0, token_weight=0.0)),
            custom_reward(pope, RewardRun(name="pope", base=TASK_ACCURACY_REWARD_MODE, latency_weight=0.0, token_weight=0.0)),
        )

    def test_policy_dataset_keeps_pope_without_rescue_rows(self) -> None:
        from scripts.train_ug_prune_policy import RewardRun, build_dataset

        candidate_actions = [
            "k12_b48_default_soft_average_query_max",
            "k12_b48_default_soft_average_query_max_coverage_zip",
            "k12_b144_more_context_soft_average_query_max",
            "k12_b288_more_context_soft_average_query_max",
        ]
        base_row = {
            "dataset": "pope",
            "task": "vqa",
            "task_family": "yes_no_vqa",
            "score": 1.0,
            "correct": True,
            "valid": True,
            "actual_llm_visual_tokens": 48,
            "original_visual_tokens": 576,
        }
        grouped = {
            "pope:sample": {
                candidate_actions[0]: {**base_row, "action": candidate_actions[0]},
                candidate_actions[1]: {**base_row, "action": candidate_actions[1], "score": 0.0, "correct": False},
            }
        }

        dataset = build_dataset(
            grouped,
            candidate_actions=candidate_actions,
            feature_action_name=candidate_actions[0],
            fallback_feature_action=candidate_actions[0],
            reward_run=RewardRun(name="task_accuracy", base="task_accuracy", latency_weight=0.0, token_weight=0.0),
            action_constraints={"task_action_mask": "v6_rescue"},
        )

        self.assertEqual(dataset["sample_ids"], ["pope:sample"])
        self.assertEqual(dataset["labels"], [0])
        self.assertNotIn(candidate_actions[2], dataset["rows"][0])

    def test_grid_runner_parses_exact_coverage_zip_action(self) -> None:
        from scripts.run_uground_adaptive_grid import action_from_name

        action = action_from_name("k12_b58_more_context_soft_average_query_max_coverage_zip")

        self.assertEqual(action.prune_after_layer, 12)
        self.assertEqual(action.target_visual_tokens, 58)
        self.assertEqual(action.role_preset, "more_context")
        self.assertEqual(action.selection_strategy, "coverage_zip")

    def test_policy_v4_manifest_split_is_deterministic_and_disjoint(self) -> None:
        from scripts.build_ug_prune_policy_v4_manifests import split_rows

        rows = [
            {
                "dataset": "pope",
                "sample_id": str(idx),
                "answer": "yes" if idx % 2 == 0 else "no",
                "category": "popular" if idx % 3 == 0 else "random",
            }
            for idx in range(30)
        ]
        first = split_rows(rows, counts=(12, 8, 10), seed="unit")
        second = split_rows(rows, counts=(12, 8, 10), seed="unit")

        self.assertEqual([row["sample_id"] for row in first["train"]], [row["sample_id"] for row in second["train"]])
        train_ids = {row["sample_id"] for row in first["train"]}
        dev_ids = {row["sample_id"] for row in first["dev"]}
        final_ids = {row["sample_id"] for row in first["final"]}
        self.assertFalse(train_ids & dev_ids)
        self.assertFalse(train_ids & final_ids)
        self.assertFalse(dev_ids & final_ids)
        self.assertEqual((len(train_ids), len(dev_ids), len(final_ids)), (12, 8, 10))


if __name__ == "__main__":
    unittest.main()
