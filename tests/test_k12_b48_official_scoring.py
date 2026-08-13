from __future__ import annotations

import unittest

from scripts.score_k12_b48_official import (
    score_gqa,
    score_mmmu,
    score_pope,
    score_scienceqa,
    score_vqa,
)


class OfficialScoringTests(unittest.TestCase):
    def test_gqa_ignores_case_and_punctuation_but_not_articles(self) -> None:
        rows = [{"sample_id": "a", "raw_model_output": "Blue!", "target_answer": "blue"}]
        scored, metrics = score_gqa(rows)
        self.assertTrue(scored[0]["official_correct"])
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_vqa_executes_evalai_normalizer_and_soft_consensus(self) -> None:
        rows = [{"sample_id": "a", "raw_model_output": "two", "target_answer": ["2"] * 10}]
        sources = []
        scored, metrics = score_vqa(rows, sources)
        self.assertEqual(scored[0]["official_score"], 1.0)
        self.assertEqual(len(sources), 1)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_scienceqa_only_accepts_exact_letter_or_uppercase_period_prefix(self) -> None:
        rows = [
            {"sample_id": "a", "raw_model_output": "B.", "target_answer": "B"},
            {"sample_id": "b", "raw_model_output": "b. because", "target_answer": "B"},
        ]
        scored, _ = score_scienceqa(rows)
        self.assertTrue(scored[0]["official_correct"])
        self.assertFalse(scored[1]["official_correct"])

    def test_pope_reports_each_subset_and_prediction_yes_ratio(self) -> None:
        rows = [
            {"sample_id": "a", "category": "random", "raw_model_output": "yes", "target_answer": "yes"},
            {"sample_id": "b", "category": "popular", "raw_model_output": "no", "target_answer": "no"},
            {"sample_id": "c", "category": "adversarial", "raw_model_output": "maybe", "target_answer": "yes"},
        ]
        _, metrics = score_pope(rows)
        self.assertEqual(set(metrics["subsets"]), {"random", "popular", "adversarial"})
        self.assertEqual(metrics["overall"]["yes_ratio"], 1 / 3)
        self.assertEqual(metrics["overall"]["invalid_outputs"], 1)

    def test_mmmu_executes_pinned_mcq_and_open_parsers(self) -> None:
        rows = [
            {
                "sample_id": "mc",
                "raw_model_output": "The answer is (B).",
                "target_answer": "B",
                "choices": ["one", "two", "three"],
                "benchmark_fields": {"question_type": "multiple-choice"},
                "image_count": 1,
            },
            {
                "sample_id": "open",
                "raw_model_output": "Therefore the answer is 42.",
                "target_answer": "42",
                "choices": [],
                "benchmark_fields": {"question_type": "open"},
                "image_count": 2,
            },
        ]
        sources = []
        scored, metrics = score_mmmu(rows, sources)
        self.assertTrue(all(row["official_correct"] for row in scored))
        self.assertEqual(metrics["subsets"]["single_image_857"]["samples"], 1)
        self.assertEqual(metrics["subsets"]["multi_image_43"]["samples"], 1)
        self.assertEqual(len(sources), 2)


if __name__ == "__main__":
    unittest.main()
