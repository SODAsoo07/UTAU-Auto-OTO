"""
OTO ML feature schema definitions.

피처 이름, 타겟 이름, 카테고리컬 피처, 기본값, 스키마 관련 상수와 유틸리티를 정의합니다.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

FEATURE_VERSION = "v10"
TRAIN_ROW_MATCH_VERSION = "v10"
TARGET_NAMES = ["delta_offset", "delta_cons", "delta_cutoff", "delta_pre", "delta_ovl"]
AUX_TARGET_NAMES = ["aux_vowel_start_rel", "aux_vowel_end_rel", "aux_next_onset_rel"]

FEATURE_NAMES = [
    "language", "format_type", "alias_type", "alias_group", "row_index_in_wav", "row_ratio_in_wav",
    "file_row_count", "file_cv_count", "file_vc_count", "file_vv_count", "file_vcv_count",
    "file_br_count", "file_mono_count", "file_cv_ratio", "file_vc_ratio",
    "file_vc_cv_ratio", "file_cv_vc_balance",
    "is_head_row", "is_tail_row", "wav_duration_ms", "base_offset", "base_cons",
    "base_cutoff_abs", "base_pre", "base_ovl", "base_cons_gap", "base_cut_gap",
    "base_ovl_ratio", "curr_phone_start_ms", "curr_phone_end_ms", "curr_phone_len_ms",
    "curr_vowel_start_ms", "curr_vowel_end_ms", "curr_vowel_len_ms", "syllable_start_ms",
    "syllable_end_ms", "syllable_len_ms", "prev_phone_gap_ms", "next_phone_gap_ms",
    "expected_anchor_ms", "base_offset_to_expected_ms", "base_pre_to_expected_ms",
    "base_cutoff_to_next_anchor_ms", "energy_mean", "energy_min", "energy_max",
    "energy_slope_pre", "energy_slope_post", "valley_energy", "valley_dist_from_cutoff_ms",
    "db_mean", "db_min", "db_silence_ratio", "f0_voicing_mean", "f0_voicing_near_pre",
    "zcr_mean", "spectral_flux_mean", "onset_class", "voicing_class", "is_tense",
    "is_diphthong", "coda_type", "vowel_class", "mora_position", "bridge_type",
    "is_nasal_or_sonorant", "prev_alias_type", "next_alias_type", "prev_base_pre",
    "next_base_pre", "prev_base_offset", "next_base_offset", "prev_base_cutoff_abs",
    "next_base_cutoff_abs",
    "mapping_confidence", "used_alias_occurrence_mapping", "used_exact_vowel_fix",
    "used_nuclei_fallback", "used_alias_based_syllables", "words_vs_alias_score_margin",
    "jump_blocked_flag", "mapping_reason_code",
    "local_peak_db", "local_valley_db", "mel_window_energy_mean", "mel_window_silence_ratio",
    "mel_voiced_formant_ratio", "mel_silence_sparse_ratio", "mel_unvoiced_diffuse_ratio",
    "mel_breath_like_ratio", "blank_span_confidence", "mel_offset_candidate_ms",
    "mel_cutoff_candidate_ms", "onset_patch_energy_mean", "onset_patch_voiced_ratio",
    "onset_patch_unvoiced_ratio", "tail_patch_energy_mean", "tail_patch_silence_ratio",
    "syllable_blank_confidence", "syllable_mel_voiced_conf", "syllable_mel_silence_conf",
    "syllable_mel_unvoiced_conf", "syllable_mel_breath_conf",
]

CATEGORICAL_FEATURES = [
    "language", "format_type", "alias_type", "alias_group", "onset_class", "voicing_class",
    "coda_type", "vowel_class", "mora_position", "bridge_type", "prev_alias_type",
    "next_alias_type", "mapping_reason_code",
]

FEATURE_DEFAULTS = {name: 0.0 for name in FEATURE_NAMES}
for _name in CATEGORICAL_FEATURES:
    FEATURE_DEFAULTS[_name] = ""

# Delta 클리핑 상수 -------------------------------------------------------

KR_DELTA_CLIP_LIMITS = {
    "delta_offset": [-220.0, 220.0],
    "delta_cons": [-220.0, 220.0],
    "delta_cutoff": [-260.0, 260.0],
    "delta_pre": [-180.0, 180.0],
    "delta_ovl": [-140.0, 140.0],
}

KR_DELTA_CLIP_LIMITS_BY_TYPE = {
    "cv": {
        "delta_offset": [-160.0, 160.0],
        "delta_cons": [-180.0, 180.0],
        "delta_cutoff": [-220.0, 220.0],
        "delta_pre": [-140.0, 140.0],
        "delta_ovl": [-100.0, 100.0],
    },
    "cv_head": {
        "delta_offset": [-200.0, 200.0],
        "delta_cons": [-180.0, 180.0],
        "delta_cutoff": [-220.0, 220.0],
        "delta_pre": [-140.0, 140.0],
        "delta_ovl": [-100.0, 100.0],
    },
    "vc": {
        "delta_offset": [-200.0, 200.0],
        "delta_cons": [-180.0, 180.0],
        "delta_cutoff": [-240.0, 240.0],
        "delta_pre": [-160.0, 160.0],
        "delta_ovl": [-120.0, 120.0],
    },
    "vv": {
        "delta_offset": [-220.0, 220.0],
        "delta_cons": [-200.0, 200.0],
        "delta_cutoff": [-260.0, 260.0],
        "delta_pre": [-120.0, 120.0],
        "delta_ovl": [-110.0, 110.0],
    },
    "vcv": {
        "delta_offset": [-200.0, 200.0],
        "delta_cons": [-200.0, 200.0],
        "delta_cutoff": [-260.0, 260.0],
        "delta_pre": [-180.0, 180.0],
        "delta_ovl": [-140.0, 140.0],
    },
}

JA_DELTA_CLIP_LIMITS = {
    "delta_offset": [-260.0, 260.0],
    "delta_cons": [-220.0, 220.0],
    "delta_cutoff": [-280.0, 280.0],
    "delta_pre": [-220.0, 220.0],
    "delta_ovl": [-160.0, 160.0],
}


# Public API ---------------------------------------------------------------

def get_delta_clip_limits(language: str, alias_type: str = "") -> Dict[str, List[float]]:
    if str(language).strip().lower().startswith("ja") or str(language).strip().lower() == "japanese":
        return dict(JA_DELTA_CLIP_LIMITS)
    a_type = str(alias_type or "").strip().lower()
    if a_type and a_type in KR_DELTA_CLIP_LIMITS_BY_TYPE:
        return dict(KR_DELTA_CLIP_LIMITS_BY_TYPE[a_type])
    return dict(KR_DELTA_CLIP_LIMITS)


def get_feature_schema() -> Dict[str, object]:
    return {
        "version": 1,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_names": list(TARGET_NAMES),
        "aux_target_names": list(AUX_TARGET_NAMES),
    }


def write_feature_schema(path: str) -> str:
    schema = get_feature_schema()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    return path


def canonicalize_feature_row(row: Dict[str, object], feature_names: Optional[List[str]] = None) -> Dict[str, object]:
    names = feature_names or FEATURE_NAMES
    out = {}
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
        "voicebank_id", "wav", "alias", "wav_norm", "alias_norm", "occurrence_index", "line_index", "source_oto_id", "source_row_id",
        *FEATURE_NAMES,
        "manual_offset", "manual_cons", "manual_cutoff", "manual_pre", "manual_ovl",
        *TARGET_NAMES,
        *AUX_TARGET_NAMES,
        "label_source", "sample_weight",
        "train_keep_default", "train_skip_reason", "train_quality_score", "skipped_reason",
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
