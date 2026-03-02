"""Recursive training-data collection helpers for OTO ML.

Compatibility layer:
- keeps public imports stable for older scripts/tests
- exposes discovery helpers and dataset builder from one module
"""

from __future__ import annotations

import os
from typing import Dict, List

from core.oto_ml_collection_discovery import (
    TrainingCandidate,
    _discover_oto_files,
    discover_training_candidates,
    discover_training_candidates_from_dataset_root,
    load_training_roots,
    write_candidate_csv,
    write_candidate_manifest,
)
from core.oto_ml_dataset import build_and_save_oto_ml_dataset


def build_datasets_from_candidates(
    candidates: List[TrainingCandidate],
    workspace_root: str,
) -> Dict[str, object]:
    """
    Build training CSVs from candidate rows.

    Note:
    Kept in this module so tests and external callers can mock
    `core.oto_ml_collection.build_and_save_oto_ml_dataset` directly.
    """
    results: List[TrainingCandidate] = []
    summary = {
        "total": len(candidates),
        "ready": 0,
        "built": 0,
        "skipped": 0,
        "saved_rows": 0,
    }
    written_outputs = set()
    for candidate in candidates:
        if candidate.status != "ready":
            summary["skipped"] += 1
            results.append(candidate)
            continue
        summary["ready"] += 1
        out_csv = os.path.join(
            workspace_root,
            "datasets",
            candidate.language,
            f"dataset_{candidate.language}_{candidate.format_type}.csv",
        )
        if out_csv not in written_outputs and os.path.exists(out_csv):
            os.remove(out_csv)
        append = out_csv in written_outputs
        stats = build_and_save_oto_ml_dataset(
            language=candidate.language,
            auto_oto_path=candidate.auto_oto,
            manual_oto_path=candidate.manual_oto,
            tg_dir=candidate.tg_dir,
            wav_dir=candidate.wav_dir,
            out_csv=out_csv,
            custom_phonemes_path=candidate.custom_phonemes,
            voicebank_id=candidate.voicebank_id,
            append=append,
            format_type_override=candidate.format_type,
        )
        written_outputs.add(out_csv)
        candidate.status = "built"
        candidate.reason = ""
        candidate.saved_rows = int(stats.get("saved_rows", 0))
        candidate.matched_rows = int(stats.get("matched_rows", 0))
        candidate.skipped_rows = int(stats.get("skipped_rows", 0))
        candidate.out_csv = str(stats.get("out_csv", ""))
        summary["built"] += 1
        summary["saved_rows"] += candidate.saved_rows
        results.append(candidate)
    return {
        "summary": summary,
        "candidates": results,
    }


__all__ = [
    "TrainingCandidate",
    "_discover_oto_files",
    "build_and_save_oto_ml_dataset",
    "build_datasets_from_candidates",
    "discover_training_candidates",
    "discover_training_candidates_from_dataset_root",
    "load_training_roots",
    "write_candidate_csv",
    "write_candidate_manifest",
]
