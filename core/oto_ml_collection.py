"""Recursive training-data collection helpers for OTO ML."""

from __future__ import annotations

from core.oto_ml_collection_build import build_datasets_from_candidates
from core.oto_ml_collection_discovery import (
    TrainingCandidate,
    discover_training_candidates,
    discover_training_candidates_from_dataset_root,
    load_training_roots,
    write_candidate_csv,
    write_candidate_manifest,
)
