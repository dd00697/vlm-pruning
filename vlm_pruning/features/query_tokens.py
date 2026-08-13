"""Resolve referring-expression token positions in LLaVA prompts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class QueryTokenResolution:
    query: str
    prompt: str
    char_start: int | None
    char_end: int | None
    prompt_token_positions: list[int]
    token_positions: list[int]
    token_ids: list[int]
    token_texts: list[str]
    resolver_status: str
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        return list(value[0])
    if isinstance(value, tuple) and value and isinstance(value[0], (list, tuple)):
        return list(value[0])
    return list(value)


def _find_query_span(prompt: str, query: str) -> tuple[int | None, int | None, str]:
    start = prompt.find(query)
    if start >= 0:
        return start, start + len(query), "exact"
    lower_start = prompt.lower().find(query.lower())
    if lower_start >= 0:
        return lower_start, lower_start + len(query), "case_insensitive"
    return None, None, "not_found"


def _find_subsequence_starts(sequence: Sequence[int], pattern: Sequence[int]) -> list[int]:
    if not pattern or len(pattern) > len(sequence):
        return []
    starts: list[int] = []
    last = len(sequence) - len(pattern) + 1
    for start in range(last):
        if list(sequence[start : start + len(pattern)]) == list(pattern):
            starts.append(start)
    return starts


def _best_subsequence_start(
    sequence: Sequence[int],
    pattern: Sequence[int],
    *,
    preferred_start: int | None = None,
) -> int | None:
    starts = _find_subsequence_starts(sequence, pattern)
    if not starts:
        return None
    if preferred_start is None:
        return starts[0]
    return min(starts, key=lambda item: abs(item - preferred_start))


def _expected_actual_start(
    prompt_positions: list[int],
    prompt_ids: list[int],
    actual_ids: list[int],
    image_token_id: int | None,
) -> int | None:
    if not prompt_positions:
        return None
    expected = prompt_positions[0]
    if image_token_id is None:
        return expected
    prompt_image_positions = [idx for idx, token_id in enumerate(prompt_ids) if int(token_id) == int(image_token_id)]
    actual_image_positions = [idx for idx, token_id in enumerate(actual_ids) if int(token_id) == int(image_token_id)]
    if not prompt_image_positions or not actual_image_positions:
        return expected
    if expected > prompt_image_positions[-1]:
        expected += len(actual_image_positions) - len(prompt_image_positions)
    return expected


def _token_texts(tokenizer: Any, token_ids: list[int]) -> list[str]:
    if not token_ids:
        return []
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        try:
            return [str(token) for token in tokenizer.convert_ids_to_tokens(token_ids)]
        except Exception:
            pass
    if hasattr(tokenizer, "decode"):
        out: list[str] = []
        for token_id in token_ids:
            try:
                out.append(str(tokenizer.decode([token_id], skip_special_tokens=False)))
            except Exception:
                out.append(str(token_id))
        return out
    return [str(token_id) for token_id in token_ids]


def _encode_with_offsets(tokenizer: Any, prompt: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    input_ids = _as_list(encoded["input_ids"])
    offsets = _as_list(encoded["offset_mapping"])
    clean_offsets: list[tuple[int, int]] = []
    for item in offsets:
        if item is None:
            clean_offsets.append((0, 0))
            continue
        start, end = item
        clean_offsets.append((int(start), int(end)))
    return [int(token_id) for token_id in input_ids], clean_offsets


def _encode_query_only(tokenizer: Any, query: str) -> list[int]:
    try:
        encoded = tokenizer(query, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer(query)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return [int(token_id) for token_id in _as_list(ids)]


def resolve_query_token_positions(
    tokenizer: Any,
    prompt: str,
    query: str,
    *,
    input_ids: Sequence[int] | Any | None = None,
    image_token_id: int | None = None,
) -> QueryTokenResolution:
    """Map the referring expression to actual model-input token positions.

    The processor path may expand one ``<image>`` placeholder into many image-token
    positions. This resolver first finds the query in tokenizer offsets, then
    aligns the corresponding token ids against the actual ``input_ids`` when
    they are provided.
    """

    char_start, char_end, span_status = _find_query_span(prompt, query)
    if char_start is None or char_end is None:
        return QueryTokenResolution(
            query=query,
            prompt=prompt,
            char_start=None,
            char_end=None,
            prompt_token_positions=[],
            token_positions=[],
            token_ids=[],
            token_texts=[],
            resolver_status="query_not_found",
            message=f"Query string was not found in prompt ({span_status}).",
        )

    try:
        prompt_ids, offsets = _encode_with_offsets(tokenizer, prompt)
    except Exception as exc:
        # The canonical LLaVA SentencePiece tokenizer is intentionally loaded
        # in slow mode and does not expose offset mappings. Its query-only ids
        # are nevertheless an exact subsequence of the processor input. Align
        # those ids directly, preferring the tokenized prefix position when a
        # phrase appears more than once. This is an observation/alignment
        # fallback; it does not alter which query tokens feed query-max.
        if input_ids is not None:
            actual_ids = [int(token_id) for token_id in _as_list(input_ids)]
            token_ids = _encode_query_only(tokenizer, query)
            preferred_start = None
            try:
                preferred_start = len(_encode_query_only(tokenizer, prompt[:char_start]))
            except Exception:
                pass
            start = _best_subsequence_start(actual_ids, token_ids, preferred_start=preferred_start)
            if start is not None and token_ids:
                token_positions = list(range(start, start + len(token_ids)))
                return QueryTokenResolution(
                    query=query,
                    prompt=prompt,
                    char_start=char_start,
                    char_end=char_end,
                    prompt_token_positions=[],
                    token_positions=token_positions,
                    token_ids=token_ids,
                    token_texts=_token_texts(tokenizer, token_ids),
                    resolver_status="matched_slow_tokenizer_subsequence",
                    message=f"Offset mappings unavailable ({type(exc).__name__}); aligned exact token-id subsequence.",
                )
        return QueryTokenResolution(
            query=query,
            prompt=prompt,
            char_start=char_start,
            char_end=char_end,
            prompt_token_positions=[],
            token_positions=[],
            token_ids=[],
            token_texts=[],
            resolver_status="tokenizer_offsets_failed",
            message=f"{type(exc).__name__}: {exc}",
        )

    prompt_positions: list[int] = []
    for idx, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if max(start, char_start) < min(end, char_end):
            if image_token_id is not None and int(prompt_ids[idx]) == int(image_token_id):
                continue
            prompt_positions.append(idx)

    token_ids = [prompt_ids[idx] for idx in prompt_positions]
    if not token_ids:
        token_ids = _encode_query_only(tokenizer, query)
        prompt_positions = []
        if not token_ids:
            return QueryTokenResolution(
                query=query,
                prompt=prompt,
                char_start=char_start,
                char_end=char_end,
                prompt_token_positions=[],
                token_positions=[],
                token_ids=[],
                token_texts=[],
                resolver_status="no_query_tokens",
                message="Tokenizer produced no query tokens.",
            )

    actual_ids = [int(token_id) for token_id in _as_list(input_ids)] if input_ids is not None else None
    if actual_ids is None:
        token_positions = prompt_positions
        status = "matched_prompt_offsets"
        message = None
    else:
        expected = _expected_actual_start(prompt_positions, prompt_ids, actual_ids, image_token_id)
        start = _best_subsequence_start(actual_ids, token_ids, preferred_start=expected)
        if start is None and prompt_positions and len(actual_ids) == len(prompt_ids):
            token_positions = prompt_positions
            status = "matched_equal_length_offsets"
            message = None
        elif start is None:
            fallback_ids = _encode_query_only(tokenizer, query)
            start = _best_subsequence_start(actual_ids, fallback_ids, preferred_start=expected)
            if start is None:
                token_positions = []
                status = "actual_input_alignment_failed"
                message = "Query token ids were not found in model input ids."
            else:
                token_ids = fallback_ids
                token_positions = list(range(start, start + len(token_ids)))
                status = "matched_query_token_fallback"
                message = None
        else:
            token_positions = list(range(start, start + len(token_ids)))
            status = "matched_actual_input_subsequence"
            message = None

    return QueryTokenResolution(
        query=query,
        prompt=prompt,
        char_start=char_start,
        char_end=char_end,
        prompt_token_positions=prompt_positions,
        token_positions=token_positions,
        token_ids=token_ids,
        token_texts=_token_texts(tokenizer, token_ids),
        resolver_status=status,
        message=message,
    )
