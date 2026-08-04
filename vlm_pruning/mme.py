"""Official MME pairing, manifest, and score helpers.

MME reports ``Acc + Acc+`` for each category.  ``Acc`` is ordinary
question-level accuracy and ``Acc+`` requires both complementary questions for
an image/category pair to be correct.  Each category therefore has a maximum
score of 200.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


PERCEPTION_CATEGORIES = (
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
)
COGNITION_CATEGORIES = (
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
)
ALL_CATEGORIES = PERCEPTION_CATEGORIES + COGNITION_CATEGORIES


def canonical_category(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() == "ocr":
        return "OCR"
    return text.lower()


def pair_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    explicit = row.get("mme_pair_id") or metadata.get("mme_pair_id")
    if explicit:
        return str(explicit)
    category = canonical_category(row.get("category") or metadata.get("category"))
    image = row.get("image_id") or row.get("image_path")
    return f"{category}::{image}"


def unique_manifest_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return MME rows with stable unique question IDs and pair metadata."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_row, original in enumerate(rows):
        row = dict(original)
        metadata = dict(row.get("metadata") or {})
        category = canonical_category(row.get("category") or metadata.get("category"))
        if category not in ALL_CATEGORIES:
            raise ValueError(f"Unsupported MME category {category!r} at row {source_row}.")
        row["category"] = category
        pid = f"{category}::{row.get('image_id') or row.get('image_path')}"
        metadata.update(
            {
                "mme_pair_id": pid,
                "original_sample_id": str(row.get("sample_id")),
                "original_source_row": source_row,
            }
        )
        row["metadata"] = metadata
        row["mme_pair_id"] = pid
        grouped[pid].append(row)

    output: list[dict[str, Any]] = []
    for pid, pair in grouped.items():
        if len(pair) != 2:
            raise ValueError(f"MME pair {pid!r} contains {len(pair)} questions, expected 2.")
        for question_index, row in enumerate(pair):
            metadata = dict(row["metadata"])
            metadata["mme_question_index"] = question_index
            row["metadata"] = metadata
            row["mme_question_index"] = question_index
            row["sample_id"] = f"mme::{pid}::q{question_index}"
            output.append(row)

    if len({str(row["sample_id"]) for row in output}) != len(output):
        raise ValueError("Generated MME sample IDs are not unique.")
    return output


def _is_correct(row: dict[str, Any]) -> bool:
    if row.get("correct") is not None:
        return bool(row["correct"])
    prediction = str(row.get("prediction") or "").strip().lower()
    target = str(row.get("target") or row.get("target_answer") or row.get("answer") or "").strip().lower()
    return prediction in {"yes", "no"} and prediction == target


def score_mme(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute official category, perception, cognition, and total MME scores."""

    materialized = list(rows)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        category = canonical_category(row.get("category"))
        if category not in ALL_CATEGORIES:
            raise ValueError(f"Unexpected MME category in results: {category!r}.")
        by_category[category].append(row)

    missing = [category for category in ALL_CATEGORIES if not by_category.get(category)]
    if missing:
        raise ValueError(f"MME results omit categories: {', '.join(missing)}")

    categories: dict[str, dict[str, Any]] = {}
    for category in ALL_CATEGORIES:
        category_rows = by_category[category]
        pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in category_rows:
            pairs[pair_id(row)].append(row)
        malformed = {pid: len(pair) for pid, pair in pairs.items() if len(pair) != 2}
        if malformed:
            preview = list(malformed.items())[:5]
            raise ValueError(f"Malformed MME pairs in {category}: {preview}")
        correct = sum(_is_correct(row) for row in category_rows)
        both_correct = sum(all(_is_correct(row) for row in pair) for pair in pairs.values())
        accuracy = 100.0 * correct / len(category_rows)
        accuracy_plus = 100.0 * both_correct / len(pairs)
        categories[category] = {
            "questions": len(category_rows),
            "pairs": len(pairs),
            "correct": correct,
            "both_correct_pairs": both_correct,
            "accuracy": accuracy,
            "accuracy_plus": accuracy_plus,
            "score": accuracy + accuracy_plus,
        }

    perception = sum(categories[category]["score"] for category in PERCEPTION_CATEGORIES)
    cognition = sum(categories[category]["score"] for category in COGNITION_CATEGORIES)
    question_correct = sum(_is_correct(row) for row in materialized)
    invalid = sum(str(row.get("prediction") or "").strip().lower() not in {"yes", "no"} for row in materialized)
    return {
        "samples": len(materialized),
        "pairs": len(materialized) // 2,
        "categories": categories,
        "perception": perception,
        "cognition": cognition,
        "total": perception + cognition,
        "question_accuracy": question_correct / len(materialized) if materialized else 0.0,
        "invalid_outputs": invalid,
        "valid_output_rate": 1.0 - invalid / len(materialized) if materialized else 0.0,
    }

