from __future__ import annotations

from typing import Dict

from core.format_type_utils import normalize_format_type


def augment_mapping_quality_features(language: str, format_type: str, alias_type: str, feat: Dict[str, object]) -> Dict[str, object]:
    """
    학습/추론 공통으로 사용하는 매핑 신뢰도 및 보조 품질 feature를 채웁니다.
    """
    feat["local_peak_db"] = float(feat.get("db_mean", 0.0) or 0.0)
    feat["local_valley_db"] = float(feat.get("db_min", 0.0) or 0.0)
    feat["mel_window_energy_mean"] = float(feat.get("energy_mean", 0.0) or 0.0)
    feat["mel_window_silence_ratio"] = float(feat.get("db_silence_ratio", 0.0) or 0.0)

    lang = str(language or "").strip().lower()
    fmt = normalize_format_type(lang, format_type)
    alias_type = str(alias_type or "").strip().lower()

    used_alias_occ = 0.0
    used_exact_vowel_fix = 0.0
    used_nuclei_fallback = 0.0
    used_alias_based_syllables = 0.0

    if lang == "korean":
        if fmt in {"cvvc", "cv", "cvc"} and alias_type in {"cv", "cv_head", "mono", "vv"}:
            used_alias_occ = 1.0
        if fmt in {"cvvc", "cv", "cvc"}:
            used_alias_based_syllables = 1.0
        if alias_type in {"cv", "cv_head", "mono", "vv"} and (
            float(feat.get("is_diphthong", 0.0) or 0.0) >= 0.5
            or "glide" in str(feat.get("alias_group", "") or "")
        ):
            used_exact_vowel_fix = 1.0
    else:
        if fmt == "vcv" and alias_type in {"vv", "vcv", "cv_head"}:
            used_alias_occ = 1.0
        if fmt == "cv" and alias_type in {"cv", "cv_head", "mono"}:
            used_alias_occ = 1.0
            used_alias_based_syllables = 1.0
        if alias_type in {"cv", "cv_head", "vcv", "vv", "mono"}:
            used_exact_vowel_fix = 1.0

    if float(feat.get("curr_vowel_len_ms", 0.0) or 0.0) < 20.0 and alias_type in {"cv", "cv_head", "vv", "mono"}:
        used_nuclei_fallback = 1.0

    off_err = abs(float(feat.get("base_offset_to_expected_ms", 0.0) or 0.0))
    pre_err = abs(float(feat.get("base_pre_to_expected_ms", 0.0) or 0.0))
    vowel_len = float(feat.get("curr_vowel_len_ms", 0.0) or 0.0)
    db_silence = float(feat.get("db_silence_ratio", 0.0) or 0.0)
    energy_mean = float(feat.get("energy_mean", 0.0) or 0.0)

    confidence = 1.0
    confidence -= min(off_err / 260.0, 0.45)
    confidence -= min(pre_err / 260.0, 0.35)
    if vowel_len > 0:
        confidence += min(vowel_len / 300.0, 0.15)
    confidence -= min(db_silence * 0.30, 0.20)
    confidence += min(max(energy_mean, 0.0) * 0.10, 0.10)
    confidence = max(0.0, min(1.0, confidence))

    feat["mapping_confidence"] = confidence
    feat["used_alias_occurrence_mapping"] = used_alias_occ
    feat["used_exact_vowel_fix"] = used_exact_vowel_fix
    feat["used_nuclei_fallback"] = used_nuclei_fallback
    feat["used_alias_based_syllables"] = used_alias_based_syllables
    feat["words_vs_alias_score_margin"] = max(0.0, (float(feat.get("syllable_len_ms", 0.0) or 0.0) - vowel_len)) - (off_err * 0.1)
    return feat


__all__ = ["augment_mapping_quality_features"]
