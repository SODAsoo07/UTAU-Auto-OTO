from __future__ import annotations

import re

from core.kr_oto_rules import IPA_VOWELS, KR_VOWELS, _cv_match_score, _normalize_cv_match_token, _split_kr_syllable_parts, is_glide, normalize_ipa_mark
from core.oto_mapping_candidates import collect_phone_activity_metrics, is_active_candidate
from core.oto_mapping_confidence import evaluate_index_plan
from core.oto_mapping_plan import build_monotonic_index_plan


def _clean_mark(mark):
    return re.sub(r"[0-9]", "", str(mark or "").strip().lower())


def _is_vowel_mark(mark):
    norm = str(normalize_ipa_mark(mark) or "").strip().lower()
    return norm in IPA_VOWELS or mark in KR_VOWELS


def collect_kr_syllable_activity_metrics(syl_info):
    return collect_phone_activity_metrics(
        (syl_info or {}).get("phones") or [],
        clean_mark_fn=_clean_mark,
        is_vowel_fn=_is_vowel_mark,
    )


def is_kr_cv_syllable_active(syl_info, *, require_vowel=True, min_active_ms=16.0, min_vowel_ms=10.0):
    return is_active_candidate(
        (syl_info or {}).get("phones") or [],
        clean_mark_fn=_clean_mark,
        is_vowel_fn=_is_vowel_mark,
        require_vowel=require_vowel,
        min_active_ms=min_active_ms,
        min_vowel_ms=min_vowel_ms,
    )


def build_kr_cv_anchor_plan(expected_tokens, syllables_info, *, use_mel=False):
    if not expected_tokens or not syllables_info:
        return {"indices": None, "score_rows": [], "meta": evaluate_index_plan([], [])}

    token_list = [_normalize_cv_match_token(t) for t in (expected_tokens or []) if t]
    if not token_list or len(syllables_info) < len(token_list):
        return {"indices": None, "score_rows": [], "meta": evaluate_index_plan([], [])}

    target_count = len(token_list)
    cand_count = len(syllables_info)
    def _mel_score(syl):
        if not use_mel:
            return 0.0
        if not isinstance(syl, dict):
            return 0.0
        voiced = float(syl.get("mel_voiced_formant_conf", 0.0) or 0.0)
        unvoiced = float(syl.get("mel_unvoiced_diffuse_conf", 0.0) or 0.0)
        silence = float(syl.get("mel_silence_sparse_conf", 0.0) or 0.0)
        breath = float(syl.get("mel_breath_like_conf", 0.0) or 0.0)
        blank = float(syl.get("blank_confidence", 0.0) or 0.0)
        voiced = max(0.0, min(1.0, voiced))
        unvoiced = max(0.0, min(1.0, unvoiced))
        silence = max(0.0, min(1.0, silence))
        breath = max(0.0, min(1.0, breath))
        blank = max(0.0, min(1.0, blank))
        return (22.0 * voiced) + (8.0 * unvoiced) - (26.0 * silence) - (28.0 * blank) - (10.0 * breath)

    score_rows = []
    for i, target_tok in enumerate(token_list):
        t_onset, t_vowel, t_coda = _split_kr_syllable_parts(target_tok)
        ideal = 0.0 if target_count <= 1 else (float(i) * float(max(cand_count - 1, 0)) / float(target_count - 1))
        row = []
        for j, syl in enumerate(syllables_info):
            cand_tok = _normalize_cv_match_token(
                syl.get("roman_cv") or syl.get("roman") or syl.get("word") or ""
            )
            c_onset, c_vowel, c_coda = _split_kr_syllable_parts(cand_tok)
            active_ms, vowel_ms, _cnt = collect_kr_syllable_activity_metrics(syl)
            active = is_kr_cv_syllable_active(syl, require_vowel=True)
            score = float(_cv_match_score(target_tok, cand_tok)) - (abs(float(j) - ideal) * 9.0)
            if target_tok and cand_tok == target_tok:
                score += 80.0
            if bool(t_coda) != bool(c_coda):
                score -= 12.0
            if t_vowel and c_vowel and (t_vowel != c_vowel):
                score -= 18.0
            if t_vowel and c_vowel and is_glide(t_vowel) != is_glide(c_vowel):
                score -= 22.0
            if t_onset and c_onset and t_onset[:1] == c_onset[:1]:
                score += 6.0
            if active:
                score += min(active_ms, 120.0) * 0.04
                score += min(vowel_ms, 100.0) * 0.07
            else:
                score -= 180.0
            score += _mel_score(syl)
            row.append(float(score))
        score_rows.append(row)

    indices = build_monotonic_index_plan(score_rows)
    meta = evaluate_index_plan(score_rows, indices or [])
    return {"indices": indices, "score_rows": score_rows, "meta": meta}


def resolve_kr_planned_cv_index(planned_indices, expected_seq_idx, target_clean, syllables_info, *, alias_type="cv"):
    if not planned_indices or not syllables_info:
        return None
    if not (0 <= int(expected_seq_idx) < len(planned_indices)):
        return None
    idx = int(planned_indices[int(expected_seq_idx)])
    if not (0 <= idx < len(syllables_info)):
        return None

    require_vowel = str(alias_type or "").strip().lower() in {"cv", "cv_head", "vcv"}
    if require_vowel and not is_kr_cv_syllable_active(syllables_info[idx], require_vowel=True):
        return None

    target_norm = _normalize_cv_match_token(target_clean)
    if not target_norm:
        return idx
    cand_tok = _normalize_cv_match_token(
        syllables_info[idx].get("roman_cv") or syllables_info[idx].get("roman") or syllables_info[idx].get("word") or ""
    )
    match_score = float(_cv_match_score(target_norm, cand_tok))
    _to, tv, tc = _split_kr_syllable_parts(target_norm)
    _co, cv, cc = _split_kr_syllable_parts(cand_tok)
    if match_score < 60.0:
        return None
    if tv and cv and tv != cv:
        return None
    if bool(tc) != bool(cc):
        return None
    return idx


__all__ = [
    "build_kr_cv_anchor_plan",
    "collect_kr_syllable_activity_metrics",
    "is_kr_cv_syllable_active",
    "resolve_kr_planned_cv_index",
]
