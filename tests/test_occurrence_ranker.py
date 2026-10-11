from __future__ import annotations

import csv
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pytest

from universal_auto_oto.occurrence_ranker import (
    OccurrenceRankerError,
    build_occurrence_boundary_queries,
    frame_indices_for_boundary_times,
    monotonic_boundary_argmax,
    query_feature_matrix,
    same_phone_occurrence_competitor_pairs,
)


def test_boundary_queries_preserve_phone_occurrence_identity() -> None:
    queries = build_occurrence_boundary_queries(("a", "k", "a", "k"))

    assert [query.boundary_index for query in queries] == [1, 2, 3]
    assert queries[0].right_phone == "k"
    assert queries[0].right_occurrence_index == 0
    assert queries[2].right_phone == "k"
    assert queries[2].right_occurrence_index == 1
    assert queries[0].features[-1] == 1.0
    assert queries[2].features[-1] == 1.0
    matrix = query_feature_matrix(queries)
    assert matrix.shape == (3, len(queries[0].features))
    assert matrix[0, -2] == 0.0
    assert matrix[2, -2] == 1.0


def test_same_phone_competitors_are_exact_occurrence_negatives() -> None:
    queries = build_occurrence_boundary_queries(("a", "k", "a", "k"))

    pairs = same_phone_occurrence_competitor_pairs(queries)

    assert pairs.tolist() == [[0, 2], [2, 0]]


def test_monotonic_assignment_avoids_shared_peak_for_adjacent_queries() -> None:
    scores = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.99, 0.1, 0.0],
            [0.0, 0.1, 0.2, 0.98, 0.4, 0.95],
        ],
        dtype=np.float64,
    )

    indices = monotonic_boundary_argmax(scores)

    assert indices.tolist() == [3, 5]


def test_frame_targets_require_strictly_ordered_boundaries() -> None:
    assert frame_indices_for_boundary_times([0.0, 5.0, 10.0, 15.0], [2.0, 3.0]).tolist() == [1, 2]
    with pytest.raises(OccurrenceRankerError):
        frame_indices_for_boundary_times([0.0, 5.0, 10.0], [8.0, 7.0])

