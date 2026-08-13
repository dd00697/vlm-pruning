from __future__ import annotations

import unittest

from vlm_pruning.final_campaign import (
    DATASETS,
    MECHANISM_RUNS,
    METHODS,
    MULTI_IMAGE_EXTENSION,
    PRIMARY_CONFIG,
    protocol_payload,
    validate_protocol,
)


class K12B48ProtocolTests(unittest.TestCase):
    def test_locked_primary_configuration(self) -> None:
        self.assertEqual(PRIMARY_CONFIG["prune_after_layer"], 12)
        self.assertEqual(PRIMARY_CONFIG["probe_layers"], [12])
        self.assertEqual(PRIMARY_CONFIG["final_visual_tokens"], 48)
        self.assertEqual(PRIMARY_CONFIG["role_counts"], {"target": 21, "context": 12, "anchor": 15})
        self.assertEqual(sum(PRIMARY_CONFIG["role_counts"].values()), 48)
        self.assertEqual(PRIMARY_CONFIG["decode_mode"], "layer_aware_cache")

    def test_no_pareto_scope(self) -> None:
        payload = protocol_payload()
        self.assertIn("no Pareto", payload["scope"])
        serialised = str(payload).lower()
        self.assertNotIn("b=72", serialised)
        self.assertNotIn("b=128", serialised)

    def test_dataset_revisions_and_counts_are_locked(self) -> None:
        self.assertEqual(len(DATASETS), 12)
        self.assertEqual({item.key for item in DATASETS}, {
            "gqa", "mmbench_en_dev", "mmmu_val", "mme", "vqav2_val", "textvqa_val",
            "pope_all", "scienceqa_img_test", "mmvet_test", "refcoco_plus_val",
            "refcoco_plus_testA", "refcoco_plus_testB",
        })
        for item in DATASETS:
            self.assertEqual(len(item.revision), 40)
            self.assertGreater(item.expected_rows, 0)
        self.assertEqual(next(item for item in DATASETS if item.key == "mmmu_val").expected_rows, 900)
        self.assertEqual(next(item for item in DATASETS if item.key == "scienceqa_img_test").split, "test")
        self.assertEqual(next(item for item in DATASETS if item.key == "refcoco_plus_val").expected_manifest_rows, 10758)

    def test_mechanism_ladder_includes_pure_query_topk(self) -> None:
        self.assertEqual(
            MECHANISM_RUNS,
            ("full", "random", "uniform", "query_topk", "target_nms_only", "structured_tca"),
        )

    def test_official_method_preclassification(self) -> None:
        statuses = {item["key"]: item["status"] for item in METHODS}
        self.assertEqual(statuses["nuwa"], "excluded_no_exact_b48_support")
        self.assertEqual(statuses["feather"], "excluded_checkpoint_or_parity_mismatch")
        self.assertEqual(statuses["fastv"], "gate_pending")
        self.assertEqual(statuses["visionzip"], "gate_pending")
        self.assertEqual(statuses["transprune"], "gate_pending")

    def test_multi_image_extension_is_total_b48(self) -> None:
        self.assertEqual(MULTI_IMAGE_EXTENSION["global_budget"], 48)
        self.assertEqual(MULTI_IMAGE_EXTENSION["anchor_count"], 15)
        self.assertEqual(MULTI_IMAGE_EXTENSION["prune_layer"], 12)
        self.assertEqual(MULTI_IMAGE_EXTENSION["allocation"], {"target": 21, "context": 12, "anchor": 15})
        self.assertTrue(MULTI_IMAGE_EXTENSION["outcome_blind"])

    def test_protocol_self_validation(self) -> None:
        self.assertEqual(validate_protocol(protocol_payload()), [])


if __name__ == "__main__":
    unittest.main()
