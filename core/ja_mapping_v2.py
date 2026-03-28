from __future__ import annotations

from core.ja_oto_mapping import (
    _clean_phone_mark,
    _is_nucleus_phone,
    _ja_soft_cv_match_level,
    _ja_special_mora_class,
    _normalize_ja_syllable_token,
    _syllable_info_token,
)
from core.ja_lab_generator import split_ja_romaji_syllable
from core.mapping_supervised_runtime import rescore_mapping_score_rows
from core.oto_mapping_candidates import collect_phone_activity_metrics, is_active_candidate
from core.oto_mapping_confidence import evaluate_index_plan
from core.oto_mapping_plan import build_monotonic_index_plan


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


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


def _candidate_duration_ms(syl) -> float:
    if not isinstance(syl, dict):
        return 0.0
    start = _safe_float(syl.get("start_time"), 0.0)
    end = _safe_float(syl.get("end_time"), 0.0)
    return max(0.0, (end - start) * 1000.0)


def build_ja_cv_anchor_score_grid(expected_tokens, syllables_info, *, use_mel=False):
    if not expected_tokens or not syllables_info:
        return {"token_list": [], "score_rows": [], "feature_rows": []}

    token_list = [_normalize_ja_syllable_token(t) for t in (expected_tokens or []) if t]
    if not token_list or len(syllables_info) < len(token_list):
        return {"token_list": [], "score_rows": [], "feature_rows": []}

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
        return (20.0 * voiced) + (7.0 * unvoiced) - (24.0 * silence) - (26.0 * blank) - (9.0 * breath)

    score_rows = []
    feature_rows = []
    for i, target_tok in enumerate(token_list):
        target_cls = _ja_special_mora_class(target_tok)
        t_onset, t_vowel = split_ja_romaji_syllable(target_tok)
        t_has_coda = 1.0 if target_tok in {"n", "nn", "xn"} else 0.0
        ideal = 0.0 if target_count <= 1 else (float(i) * float(max(cand_count - 1, 0)) / float(target_count - 1))
        row = []
        feat_row = []
        for j, syl in enumerate(syllables_info):
            cand_tok = _normalize_ja_syllable_token(_syllable_info_token(syl))
            cand_cls = _ja_special_mora_class(cand_tok)
            c_onset, c_vowel = split_ja_romaji_syllable(cand_tok)
            c_has_coda = 1.0 if cand_tok in {"n", "nn", "xn"} else 0.0
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
            score += _mel_score(syl)
            row.append(float(score))
            blank = max(0.0, min(1.0, _safe_float((syl or {}).get("blank_confidence"), 0.0)))
            feat_row.append(
                {
                    "target_pos": (float(i) / float(max(target_count - 1, 1))),
                    "cand_pos": (float(j) / float(max(cand_count - 1, 1))),
                    "pos_delta": (float(j) - ideal),
                    "abs_pos_delta": abs(float(j) - ideal),
                    "target_has_onset": 1.0 if t_onset else 0.0,
                    "target_has_coda": t_has_coda,
                    "cand_has_onset": 1.0 if c_onset else 0.0,
                    "cand_has_coda": c_has_coda,
                    "onset_initial_match": 1.0 if (t_onset and c_onset and t_onset[:1] == c_onset[:1]) else 0.0,
                    "vowel_match": 1.0 if (t_vowel and c_vowel and t_vowel == c_vowel) else 0.0,
                    "coda_match": 1.0 if (t_has_coda == c_has_coda) else 0.0,
                    "target_is_glide": 1.0 if ("y" in str(t_onset or "")) else 0.0,
                    "cand_is_glide": 1.0 if ("y" in str(c_onset or "")) else 0.0,
                    "special_class_match": 1.0 if target_cls == cand_cls else 0.0,
                    "text_match_score": max(0.0, min(1.0, float(soft) / 4.0)),
                    "active_ms": max(0.0, float(active_ms)),
                    "vowel_ms": max(0.0, float(vowel_ms)),
                    "blank_conf": blank,
                    "mel_voiced": max(0.0, min(1.0, _safe_float((syl or {}).get("mel_voiced_formant_conf"), 0.0))),
                    "mel_silence": max(0.0, min(1.0, _safe_float((syl or {}).get("mel_silence_sparse_conf"), 0.0))),
                    "mel_unvoiced": max(0.0, min(1.0, _safe_float((syl or {}).get("mel_unvoiced_diffuse_conf"), 0.0))),
                    "mel_breath": max(0.0, min(1.0, _safe_float((syl or {}).get("mel_breath_like_conf"), 0.0))),
                    "cand_duration_ms": _candidate_duration_ms(syl),
                }
            )
        score_rows.append(row)
        feature_rows.append(feat_row)

    return {"token_list": token_list, "score_rows": score_rows, "feature_rows": feature_rows}


def build_ja_cv_anchor_plan(expected_tokens, syllables_info, *, use_mel=False, format_type=""):
    grid = build_ja_cv_anchor_score_grid(expected_tokens, syllables_info, use_mel=use_mel)
    token_list = list(grid.get("token_list") or [])
    score_rows = list(grid.get("score_rows") or [])
    feature_rows = list(grid.get("feature_rows") or [])
    if not token_list or not score_rows:
        return {"indices": None, "score_rows": [], "meta": evaluate_index_plan([], [])}

    rescored_rows, model_meta = rescore_mapping_score_rows(
        language="japanese",
        format_type=(str(format_type or "").strip().lower() or "cv"),
        score_rows=score_rows,
        feature_rows=feature_rows,
    )
    used_rows = rescored_rows if rescored_rows else score_rows
    indices = build_monotonic_index_plan(used_rows)
    meta = evaluate_index_plan(used_rows, indices or [])
    if isinstance(model_meta, dict):
        meta["mapping_supervised"] = dict(model_meta)
    return {"indices": indices, "score_rows": used_rows, "meta": meta}


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
    "build_ja_cv_anchor_score_grid",
    "collect_ja_syllable_activity_metrics",
    "is_ja_cv_syllable_active",
    "resolve_ja_planned_cv_index",
]
