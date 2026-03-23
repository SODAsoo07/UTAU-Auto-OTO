from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

AUTOFREE_BACKEND = "autofree_v1"
FEATURE_SCHEMA_VERSION = "af_v2"
TARGET_SCHEMA_VERSION = "af_v1_abs"

SOURCE_MODES = ["textgrid", "audio_only"]
TOKEN_SOURCES = ["filename", "external_map", "oto_rows"]
SOURCE_DETAILS = ["textgrid", "boundary_guided", "boundary_guided_pause", "uniform_split"]

FEATURE_NAMES = [
    "language",
    "format_type",
    "voicebank_id",
    "wav_norm",
    "alias_norm",
    "occurrence_index",
    "row_index_in_wav",
    "source_mode",
    "source_detail",
    "token_source",
    "source_confidence",
    "onset_ms",
    "voiced_onset_ms",
    "nucleus_start_ms",
    "nucleus_end_ms",
    "nucleus_span_ms",
    "tail_ms",
    "tail_margin_ms",
    "blank_confidence",
    "voiced_confidence",
    "unvoiced_confidence",
    "boundary_confidence",
    "mel_window_energy_mean",
    "mel_window_silence_ratio",
    "mel_window_voiced_ratio",
    "prev_gap_ms",
    "next_gap_ms",
    "neighbor_boundary_gap_ms",
    "file_row_count",
    "file_mean_energy",
    "file_voiced_ratio",
]

CATEGORICAL_FEATURES = [
    "language",
    "format_type",
    "voicebank_id",
    "wav_norm",
    "alias_norm",
    "source_mode",
    "source_detail",
    "token_source",
]

FEATURE_DEFAULTS: Dict[str, object] = {name: 0.0 for name in FEATURE_NAMES}
for _name in CATEGORICAL_FEATURES:
    FEATURE_DEFAULTS[_name] = ""

TARGET_NAMES = [
    "target_offset_ms",
    "target_cons_ms",
    "target_cutoff_abs_ms",
    "target_pre_ms",
    "target_ovl_ms",
]
CONFIDENCE_TARGET_NAME = "target_confidence"


def get_feature_schema() -> Dict[str, object]:
    return {
        "version": 1,
        "backend": AUTOFREE_BACKEND,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "source_modes": list(SOURCE_MODES),
        "token_sources": list(TOKEN_SOURCES),
        "source_details": list(SOURCE_DETAILS),
    }


def get_target_schema() -> Dict[str, object]:
    return {
        "version": 1,
        "backend": AUTOFREE_BACKEND,
        "target_schema_version": TARGET_SCHEMA_VERSION,
        "coordinate_system": "absolute_ms",
        "target_names": list(TARGET_NAMES),
        "confidence_name": CONFIDENCE_TARGET_NAME,
    }


def write_feature_schema(path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(get_feature_schema(), f, ensure_ascii=False, indent=2)
    return path


def write_target_schema(path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(get_target_schema(), f, ensure_ascii=False, indent=2)
    return path


def canonicalize_feature_row(row: Dict[str, object], feature_names: Optional[List[str]] = None) -> Dict[str, object]:
    names = list(feature_names or FEATURE_NAMES)
    out: Dict[str, object] = {}
    for name in names:
        val = row.get(name, FEATURE_DEFAULTS.get(name, 0.0))
        if name in CATEGORICAL_FEATURES:
            out[name] = "" if val is None else str(val)
        else:
            try:
                out[name] = float(val)
            except Exception:
                out[name] = 0.0
    return out


def dataset_fieldnames() -> List[str]:
    return [
        "row_uid",
        "language",
        "format_type",
        "voicebank_id",
        "wav",
        "alias",
        "wav_norm",
        "alias_norm",
        "occurrence_index",
        "row_index_in_wav",
        "line_index",
        "source_mode",
        "token_source",
        "source_confidence",
        *FEATURE_NAMES,
        *TARGET_NAMES,
        CONFIDENCE_TARGET_NAME,
        "label_source",
        "match_strategy",
        "sample_weight",
        "quality_score",
        "skip_with_reason",
    ]


def write_dataset_csv(path: str, rows: List[Dict[str, object]], append: bool = False) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mode = "a" if append and os.path.exists(path) else "w"
    fieldnames = dataset_fieldnames()
    with open(path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        for row in rows:
            merged = dict(row)
            merged.update(canonicalize_feature_row(row))
            writer.writerow(merged)
    return path


__all__ = [
    "AUTOFREE_BACKEND",
    "CATEGORICAL_FEATURES",
    "CONFIDENCE_TARGET_NAME",
    "FEATURE_DEFAULTS",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "SOURCE_MODES",
    "SOURCE_DETAILS",
    "TARGET_NAMES",
    "TARGET_SCHEMA_VERSION",
    "TOKEN_SOURCES",
    "canonicalize_feature_row",
    "dataset_fieldnames",
    "get_feature_schema",
    "get_target_schema",
    "write_dataset_csv",
    "write_feature_schema",
    "write_target_schema",
]
