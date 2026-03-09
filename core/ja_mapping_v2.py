from __future__ import annotations

from core.ja_oto_mapping import (
    _clean_phone_mark,
    _is_nucleus_phone,
    _ja_soft_cv_match_level,
    _ja_special_mora_class,
    _normalize_ja_syllable_token,
    _syllable_info_token,
)
from core.oto_mapping_candidates import collect_phone_activity_metrics, is_active_candidate
from core.oto_mapping_confidence import evaluate_index_plan
from core.oto_mapping_plan import build_monotonic_index_plan


def collect_ja_syllable_activity_metrics(syl_info):
    return collect_phone_activity_metrics(
        (syl_info or {}).get("phones") or [],
        clean_mark_fn=_clean_phone_mark,
        is_vowel_fn=_is_nucleus_phone,
    )


def is_ja_cv_syllable_active(syl_info, *, require_vowel=True, min_active_ms=16.0, min_vowel_ms=10.0):
    return is_active_candidate(
        (syl_info or {}).get("phones") or [],
        clean_mark_fn=_clean_phone_mark,
        is_vowel_fn=_is_nucleus_phone,
        require_vowel=require_vowel,
        min_active_ms=min_active_ms,
        min_vowel_ms=min_vowel_ms,
    )


def build_ja_cv_anchor_plan(expected_tokens, syllables_info):
    if not expected_tokens or not syllables_info:
        return {"indices": None, "score_rows": [], "meta": evaluate_index_plan([], [])}

    token_list = [_normalize_ja_syllable_token(t) for t in (expected_tokens or []) if t]
    if not token_list or len(syllables_info) < len(token_list):
        return {"indices": None, "score_rows": [], "meta": evaluate_index_plan([], [])}

    target_count = len(token_list)
    cand_count = len(syllables_info)
    score_rows = []
    for i, target_tok in enumerate(token_list):
        target_cls = _ja_special_mora_class(target_tok)
        ideal = 0.0 if target_count <= 1 else (float(i) * float(max(cand_count - 1, 0)) / float(target_count - 1))
        row = []
        for j, syl in enumerate(syllables_info):
            cand_tok = _normalize_ja_syllable_token(_syllable_info_token(syl))
            cand_cls = _ja_special_mora_class(cand_tok)
            soft = int(_ja_soft_cv_match_level(target_tok, cand_tok) or 0) if cand_tok else 0
            active_ms, vowel_ms, _cnt = collect_ja_syllable_activity_metrics(syl)
            active = is_ja_cv_syllable_active(syl, require_vowel=True)
            score = (soft * 42.0) - (abs(float(j) - ideal) * 10.0)
            if cand_tok == target_tok:
                score += 120.0
            if target_cls == cand_cls:
                score += 10.0
            if target_cls in {"youon", "inserted"} and cand_cls not in {"youon", "inserted"}:
                score -= 20.0
            elif target_cls == "plain" and cand_cls in {"youon", "inserted"}:
                score -= 14.0
            if not cand_tok:
                score -= 120.0
            if active:
                score += min(active_ms, 120.0) * 0.05
                score += min(vowel_ms, 100.0) * 0.08
            else:
                score -= 180.0
            row.append(float(score))
        score_rows.append(row)

    indices = build_monotonic_index_plan(score_rows)
    meta = evaluate_index_plan(score_rows, indices or [])
    return {"indices": indices, "score_rows": score_rows, "meta": meta}


def resolve_ja_planned_cv_index(planned_indices, expected_seq_idx, target_tok, syllables_info, *, alias_type="cv"):
    if not planned_indices or not syllables_info:
        return None
    if not (0 <= int(expected_seq_idx) < len(planned_indices)):
        return None
    idx = int(planned_indices[int(expected_seq_idx)])
    if not (0 <= idx < len(syllables_info)):
        return None

    target_norm = _normalize_ja_syllable_token(target_tok)
    cand_tok = _normalize_ja_syllable_token(_syllable_info_token(syllables_info[idx]))
    require_vowel = str(alias_type or "").strip().lower() in {"cv", "cv_head", "vcv"}
    if require_vowel and not is_ja_cv_syllable_active(syllables_info[idx], require_vowel=True):
        return None
    if not target_norm:
        return idx

    soft = int(_ja_soft_cv_match_level(target_norm, cand_tok) or 0) if cand_tok else 0
    target_cls = _ja_special_mora_class(target_norm)
    cand_cls = _ja_special_mora_class(cand_tok)
    min_soft = 2 if target_cls in {"youon", "inserted"} else 1
    if cand_tok != target_norm and soft < min_soft:
        return None
    if target_cls == "plain" and cand_cls in {"youon", "inserted"} and cand_tok != target_norm:
        return None
    return idx


__all__ = [
    "build_ja_cv_anchor_plan",
    "collect_ja_syllable_activity_metrics",
    "is_ja_cv_syllable_active",
    "resolve_ja_planned_cv_index",
]
