from __future__ import annotations

"""Sequence-conditioned boundary-query contract for repeated phones.

The existing neural evidence model is intentionally transcript-independent: it
emits frame tracks such as ``boundary_likelihood``.  That is useful evidence,
but it cannot identify which occurrence of a repeated phone owns a peak.  This
module defines the query side of a separate ranker: every known-sequence phone
boundary gets a stable, occurrence-aware feature vector and the resulting
per-boundary scores are decoded jointly under a strict monotonic constraint.

It is deliberately independent of the current runtime backend.  Until a
trained ranker and a held-out evaluation jointly improve slot shifts,
``onnx_neural_evidence_v1`` remains the production/default evidence path.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .neural_evidence_dataset import phone_roles


OCCURRENCE_QUERY_SCHEMA_VERSION = 1
OCCURRENCE_ROLE_CLASSES = (
    "silence",
    "breath",
    "vowel",
    "stop",
    "fricative",
    "consonant",
)
OCCURRENCE_QUERY_FEATURES = tuple(
    [f"left_{role}" for role in OCCURRENCE_ROLE_CLASSES]
    + [f"right_{role}" for role in OCCURRENCE_ROLE_CLASSES]
    + [
        "boundary_progress",
        "left_occurrence_progress",
        "right_occurrence_progress",
        "repeated_phone_pair",
    ]
)


class OccurrenceRankerError(ValueError):
    pass


@dataclass(frozen=True)
class OccurrenceBoundaryQuery:
    """One ordered phone boundary query in the known transcript."""

    boundary_index: int
    left_phone: str
    right_phone: str
    left_occurrence_index: int
    left_occurrence_count: int
    right_occurrence_index: int
    right_occurrence_count: int
    features: tuple[float, ...]


def build_occurrence_boundary_queries(phones: Sequence[str]) -> tuple[OccurrenceBoundaryQuery, ...]:
    """Return one stable query per interior boundary of ``phones``.

    Occurrence indices are zero-based and count only the exact normalized phone
    token.  They therefore distinguish the first and second ``a`` even when
    their coarse acoustic role is identical.
    """
    tokens = tuple(_normalise_phone(phone) for phone in phones)
    if len(tokens) < 2:
        return ()
    counts = {token: tokens.count(token) for token in set(tokens)}
    seen: dict[str, int] = {}
    occurrence_index: list[int] = []
    for token in tokens:
        occurrence_index.append(seen.get(token, 0))
        seen[token] = occurrence_index[-1] + 1
    denominator = max(1, len(tokens) - 1)
    queries: list[OccurrenceBoundaryQuery] = []
    for boundary_index in range(1, len(tokens)):
        left = tokens[boundary_index - 1]
        right = tokens[boundary_index]
        left_count = int(counts[left])
        right_count = int(counts[right])
        left_index = int(occurrence_index[boundary_index - 1])
        right_index = int(occurrence_index[boundary_index])
        features = (
            *_role_one_hot(left),
            *_role_one_hot(right),
            float(boundary_index) / float(denominator),
            _occurrence_progress(left_index, left_count),
            _occurrence_progress(right_index, right_count),
            float(left_count > 1 or right_count > 1),
        )
        queries.append(
            OccurrenceBoundaryQuery(
                boundary_index=boundary_index,
                left_phone=left,
                right_phone=right,
                left_occurrence_index=left_index,
                left_occurrence_count=left_count,
                right_occurrence_index=right_index,
                right_occurrence_count=right_count,
                features=tuple(float(value) for value in features),
            )
        )
    return tuple(queries)


def query_feature_matrix(queries: Sequence[OccurrenceBoundaryQuery]) -> np.ndarray:
    """Return the validated ``[boundary, feature]`` ranker input matrix."""
    if not queries:
        return np.zeros((0, len(OCCURRENCE_QUERY_FEATURES)), dtype=np.float32)
    values = np.asarray([query.features for query in queries], dtype=np.float32)
    expected = (len(queries), len(OCCURRENCE_QUERY_FEATURES))
    if values.shape != expected or not np.all(np.isfinite(values)):
        raise OccurrenceRankerError("invalid occurrence boundary query features")
    return values


def same_phone_occurrence_competitor_pairs(queries: Sequence[OccurrenceBoundaryQuery]) -> np.ndarray:
    """Return directed query pairs that compete for a repeated-phone boundary.

    A row ``[query, competitor]`` means that the first query's labelled frame
    is positive and the second query's labelled frame is an explicit
    same-phone, wrong-occurrence negative.  We keep left- and right-phone
    ownership separate: sharing an acoustic role is not enough, and a unique
    phone may never create a competitor merely because it has the same role.
    """
    pairs: set[tuple[int, int]] = set()
    for index, query in enumerate(queries):
        for other_index, other in enumerate(queries):
            if index == other_index:
                continue
            same_left = (
                query.left_occurrence_count > 1
                and query.left_phone == other.left_phone
            )
            same_right = (
                query.right_occurrence_count > 1
                and query.right_phone == other.right_phone
            )
            if same_left or same_right:
                pairs.add((index, other_index))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(pairs), dtype=np.int64)


def frame_indices_for_boundary_times(
    frame_times_ms: Sequence[float], boundary_times_ms: Sequence[float]
) -> np.ndarray:
    """Map manual boundary times to nearest valid frame indices monotonically."""
    times = np.asarray(frame_times_ms, dtype=np.float64)
    boundaries = np.asarray(boundary_times_ms, dtype=np.float64)
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise OccurrenceRankerError("frame times must be finite and strictly increasing")
    if boundaries.ndim != 1 or not np.all(np.isfinite(boundaries)) or np.any(np.diff(boundaries) <= 0.0):
        raise OccurrenceRankerError("boundary times must be finite and strictly increasing")
    indices = np.searchsorted(times, boundaries, side="left")
    indices = np.clip(indices, 0, len(times) - 1)
    previous = -1
    for index, value in enumerate(indices):
        current = max(previous + 1, int(value))
        if current >= len(times):
            raise OccurrenceRankerError("not enough frames for strictly ordered boundary targets")
        indices[index] = current
        previous = current
    return indices.astype(np.int64, copy=False)


def monotonic_boundary_argmax(boundary_scores: np.ndarray) -> np.ndarray:
    """Jointly select strictly ordered frames for all boundary queries.

    Independent argmax permits two repeated-phone queries to select the same
    neural peak.  This dynamic program forces every following boundary to use a
    later frame, which turns occurrence identity into an explicit decoding
    invariant rather than a best-effort post-processing rule.
    """
    scores = np.asarray(boundary_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0 or not np.all(np.isfinite(scores)):
        raise OccurrenceRankerError("boundary scores must be a finite non-empty [boundary, frame] matrix")
    boundary_count, frame_count = scores.shape
    if boundary_count > frame_count:
        raise OccurrenceRankerError("strict monotonic assignment needs at least one frame per boundary")
    dynamic = np.full((boundary_count, frame_count), -np.inf, dtype=np.float64)
    backpointer = np.full((boundary_count, frame_count), -1, dtype=np.int64)
    dynamic[0] = scores[0]
    for boundary_index in range(1, boundary_count):
        prefix_best = np.maximum.accumulate(dynamic[boundary_index - 1])
        prefix_argmax = np.empty(frame_count, dtype=np.int64)
        best_index = 0
        for frame_index in range(frame_count):
            if dynamic[boundary_index - 1, frame_index] > dynamic[boundary_index - 1, best_index]:
                best_index = frame_index
            prefix_argmax[frame_index] = best_index
        dynamic[boundary_index, 1:] = scores[boundary_index, 1:] + prefix_best[:-1]
        backpointer[boundary_index, 1:] = prefix_argmax[:-1]
    final_frame = int(np.argmax(dynamic[-1]))
    if not np.isfinite(dynamic[-1, final_frame]):
        raise OccurrenceRankerError("monotonic boundary assignment has no valid path")
    result = np.empty(boundary_count, dtype=np.int64)
    result[-1] = final_frame
    for boundary_index in range(boundary_count - 1, 0, -1):
        previous = int(backpointer[boundary_index, result[boundary_index]])
        if previous < 0:
            raise OccurrenceRankerError("monotonic boundary assignment backtrace failed")
        result[boundary_index - 1] = previous
    return result


def _normalise_phone(phone: str) -> str:
    token = str(phone).strip().casefold()
    if not token:
        raise OccurrenceRankerError("phone tokens must be non-empty")
    return token


def _role_one_hot(phone: str) -> tuple[float, ...]:
    roles = phone_roles(phone)
    if "silence_likelihood" in roles:
        role = "silence"
    elif "breath_likelihood" in roles:
        role = "breath"
    elif "vowel_stable_likelihood" in roles:
        role = "vowel"
    elif "burst_likelihood" in roles:
        role = "stop"
    elif "frication_likelihood" in roles:
        role = "fricative"
    else:
        role = "consonant"
    return tuple(float(name == role) for name in OCCURRENCE_ROLE_CLASSES)


def _occurrence_progress(index: int, count: int) -> float:
    return 0.0 if count <= 1 else float(index) / float(count - 1)


__all__ = [
    "OCCURRENCE_QUERY_FEATURES",
    "OCCURRENCE_QUERY_SCHEMA_VERSION",
    "OCCURRENCE_ROLE_CLASSES",
    "OccurrenceBoundaryQuery",
    "OccurrenceRankerError",
    "build_occurrence_boundary_queries",
    "frame_indices_for_boundary_times",
    "monotonic_boundary_argmax",
    "query_feature_matrix",
    "same_phone_occurrence_competitor_pairs",
]
