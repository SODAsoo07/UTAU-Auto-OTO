from __future__ import annotations

import os
import re

from core.kr_auto_alias_setup import expand_kr_filename_alias_tokens
from core.kr_oto_rules import (
    KR_CONSONANTS,
    KR_VOWELS,
    _cv_match_score,
    _extract_kr_cv_alias_token,
    _kr_cv_kernel,
    _normalize_cv_match_token,
    _split_kr_syllable_parts,
    classify_alias,
)


def _alias_to_cv_target(alias, alias_type):
    a = (alias or "").strip()
    if not a:
        return ""

    if alias_type == "cv":
        tok = re.sub(r"[^a-z]", "", a.lower())
        return _kr_cv_kernel(tok)
    if alias_type == "cv_head":
        parts = a.split()
        if len(parts) >= 2 and parts[0] == "-":
            tok = re.sub(r"[^a-z]", "", parts[1].lower())
            return _kr_cv_kernel(tok)
        tok = re.sub(r"[^a-z]", "", a.lower().lstrip("-"))
        return _kr_cv_kernel(tok)
    if alias_type == "vcv":
        parts = a.split()
        if len(parts) >= 2:
            tok = re.sub(r"[^a-z]", "", parts[1].lower())
            return _kr_cv_kernel(tok)
    if alias_type == "mono":
        tok = re.sub(r"[^a-z]", "", a.lower())
        if tok in KR_VOWELS:
            return tok
    return ""


def _extract_cv_targets_from_lines(lines, custom_map=None):
    targets = []
    type_cache = {}
    for line in lines or []:
        if "=" not in line:
            continue
        alias = line.split("=", 1)[1].split(",", 1)[0].strip()
        if not alias:
            continue
        a_type = type_cache.get(alias)
        if a_type is None:
            a_type = classify_alias(alias, custom_map)
            type_cache[alias] = a_type
        tok = _alias_to_cv_target(alias, a_type)
        if tok:
            targets.append(tok)
    if not targets:
        return []

    collapsed = []
    for tok in targets:
        if not collapsed or collapsed[-1] != tok:
            collapsed.append(tok)
    return collapsed


def _extract_kr_cv_targets_from_filename(filename):
    """파일명에서 한국어 CV 계열 음절 토큰을 추출합니다."""
    out = []
    for tok in expand_kr_filename_alias_tokens(
        os.path.basename(filename or ""),
        split_syllable_parts_fn=_split_kr_syllable_parts,
        kr_vowels=KR_VOWELS,
        kr_consonants=KR_CONSONANTS,
    ):
        norm = _normalize_cv_match_token(tok)
        onset, vowel, coda = _split_kr_syllable_parts(norm)
        if vowel:
            out.append(norm)
    return out


def _iter_kr_cvvc_tokens(token_source):
    if not token_source:
        return []
    if isinstance(token_source, list) and token_source and isinstance(token_source[0], dict):
        out = []
        for syl in token_source:
            tok = _normalize_cv_match_token(
                syl.get("roman") or syl.get("roman_cv") or syl.get("word") or ""
            )
            if tok:
                out.append(tok)
        return out
    out = []
    for tok in token_source:
        norm = _normalize_cv_match_token(tok)
        if norm:
            out.append(norm)
    return out


def _iter_kr_cvvc_cv_slots(token_source):
    """
    CVVC token source에서 "CV occurrence 슬롯" 인덱스를 만듭니다.
    - vowel이 있는 모든 음절을 슬롯으로 집계합니다.
    - 받침(coda)이 있어도 CV kernel(onset+vowel)로 환원해 포함합니다.
    """
    out = []
    cv_slot = 0
    for raw_idx, tok in enumerate(_iter_kr_cvvc_tokens(token_source)):
        onset, vowel, coda = _split_kr_syllable_parts(tok)
        if not vowel:
            continue
        kernel = _kr_cv_kernel(tok)
        if not kernel:
            continue
        out.append((raw_idx, cv_slot, tok, kernel, onset, vowel, coda))
        cv_slot += 1
    return out


def _build_kr_cvvc_occurrence_map(token_source):
    """CVVC 파일의 filename 기반 음절 순서에서 CV/CV_HEAD occurrence map을 만듭니다."""
    occ = {}
    for _raw_idx, cv_slot, _tok, kernel, _onset, _vowel, _coda in _iter_kr_cvvc_cv_slots(token_source):
        occ.setdefault(kernel, []).append(cv_slot)
    return occ


def _is_kr_order_locked_cv_format(file_format):
    fmt = str(file_format or "").strip().lower()
    return fmt in {"cvvc", "cvc"}


def _resolve_kr_cvvc_occurrence_index(alias, alias_type, occurrence_map, occurrence_state):
    """한국어 CVVC에서 같은 CV 토큰의 등장 순서대로 CV/CV_HEAD를 직접 매핑합니다."""
    if alias_type not in {"cv", "cv_head"}:
        return None
    tok = _extract_kr_cv_alias_token(alias)
    if not tok:
        return None
    idxs = occurrence_map.get(tok) or []
    if not idxs:
        return None
    # CV와 CV_HEAD는 보통 같은 음절 인덱스를 함께 참조하므로,
    # 카운터를 분리하지 않으면 한쪽이 다른 쪽의 매핑 순서를 밀어버릴 수 있습니다.
    state_key = (str(alias_type), tok)
    used = int(occurrence_state.get(state_key, 0))
    if used >= len(idxs):
        used = len(idxs) - 1
    occurrence_state[state_key] = used + 1
    return idxs[used]


def _extract_kr_vv_pair_key(alias):
    """한국어 VV alias를 filename pair occurrence 매핑용 키로 변환합니다."""
    parts = [p for p in re.split(r"\s+", (alias or "").strip().lower()) if p]
    if len(parts) != 2:
        return ""
    left = _normalize_cv_match_token(parts[0])
    right = _normalize_cv_match_token(parts[1])
    lo, lv, lc = _split_kr_syllable_parts(left)
    ro, rv, rc = _split_kr_syllable_parts(right)
    if not lv or not rv:
        return ""
    if lo or ro or lc or rc:
        return ""
    return f"{left} {right}"


def _build_kr_cvvc_vv_occurrence_map(token_source):
    """CVVC 파일의 순수 모음-모음 연결(VV) 등장 순서를 맵으로 만듭니다."""
    occ = {}
    slot_by_raw = {}
    for raw_idx, cv_slot, tok, _kernel, onset, vowel, coda in _iter_kr_cvvc_cv_slots(token_source):
        slot_by_raw[raw_idx] = (cv_slot, tok, onset, vowel, coda)

    if not slot_by_raw:
        return occ

    max_raw_idx = max(slot_by_raw.keys())
    for raw_idx in range(1, max_raw_idx + 1):
        prev_info = slot_by_raw.get(raw_idx - 1)
        curr_info = slot_by_raw.get(raw_idx)
        if not prev_info or not curr_info:
            continue
        _prev_slot, prev_tok, po, pv, pc = prev_info
        curr_slot, curr_tok, co, cv, cc = curr_info
        if not pv or not cv:
            continue
        if po or co or pc or cc:
            continue
        occ.setdefault(f"{prev_tok} {curr_tok}", []).append(curr_slot)
    return occ


def _resolve_kr_cvvc_vv_index(alias, occurrence_map, occurrence_state):
    """한국어 CVVC의 VV alias를 filename pair occurrence 기준으로 매핑합니다."""
    key = _extract_kr_vv_pair_key(alias)
    if not key:
        return None
    idxs = occurrence_map.get(key) or []
    if not idxs:
        return None
    used = int(occurrence_state.get(key, 0))
    if used >= len(idxs):
        used = len(idxs) - 1
    occurrence_state[key] = used + 1
    return idxs[used]


def _find_kr_cv_vowel_match_index(target_clean, romaji_syllables, expected_idx, search_back=1, search_fwd=2):
    """
    CV 목표 모음과 일치하는 음절을 기대 인덱스 근방에서 재탐색합니다.
    한 음절 밀림(특히 이중모음/반모음) 보정용 안전장치입니다.
    """
    if not target_clean or not romaji_syllables:
        return None
    n = len(romaji_syllables)
    if n <= 0:
        return None
    e = max(0, min(int(expected_idx), n - 1))

    t_onset, t_vowel, _t_coda = _split_kr_syllable_parts(target_clean)
    if not t_vowel:
        return None

    lo = max(0, e - max(0, int(search_back)))
    hi = min(n - 1, e + max(0, int(search_fwd)))
    best_idx = None
    best_score = -10**9
    for i in range(lo, hi + 1):
        cand = romaji_syllables[i]
        c_onset, c_vowel, _c_coda = _split_kr_syllable_parts(cand)
        if not c_vowel or c_vowel != t_vowel:
            continue
        score = 100 - (abs(i - e) * 16)
        if t_onset and c_onset:
            if t_onset == c_onset:
                score += 18
            elif t_onset[:1] == c_onset[:1]:
                score += 8
            else:
                score -= 10
        elif t_onset != c_onset:
            score -= 4
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _should_allow_kr_exact_vowel_fix(
    file_format,
    forced_selected_idx,
    *,
    alias_type=None,
    severe_vowel_mismatch=False,
):
    """
    한국어 CVVC에서 filename occurrence로 이미 강제 매핑된 CV/CV_HEAD는
    뒤의 모음 재탐색이 다시 덮어쓰지 않도록 막습니다.

    단, CV_HEAD에서 강제 매핑 결과가 목표 모음과 크게 어긋난 경우에는
    1칸 내 재탐색 보정을 허용합니다.
    """
    fmt = str(file_format or "").strip().lower()
    if _is_kr_order_locked_cv_format(fmt) and forced_selected_idx is not None:
        a_type = str(alias_type or "").strip().lower()
        if a_type == "cv_head" and bool(severe_vowel_mismatch):
            return True
        return False
    return True


def _score_kr_syllable_mapping(candidate_infos, cv_targets):
    """candidate 음절열이 alias 기반 cv_targets와 얼마나 일치하는지 점수화."""
    if not candidate_infos or not cv_targets:
        return -1.0

    cand = []
    for s in candidate_infos:
        tok = s.get("roman_cv") or s.get("roman") or s.get("word") or ""
        tok = _kr_cv_kernel(tok)
        if tok:
            cand.append(tok)
    if not cand:
        return -1.0

    try:
        blank_penalty_scale = float(os.environ.get("UTOA_KR_BLANK_SCORE_PENALTY", "26.0"))
    except Exception:
        blank_penalty_scale = 26.0
    blank_penalty_scale = max(0.0, min(blank_penalty_scale, 80.0))

    def _avg_with_shift(shift):
        vals = []
        for i, tgt in enumerate(cv_targets):
            j = i + shift
            if 0 <= j < len(cand):
                score = float(_cv_match_score(tgt, cand[j]))
                try:
                    blank_conf = float((candidate_infos[j] or {}).get("blank_confidence", 0.0) or 0.0)
                except Exception:
                    blank_conf = 0.0
                blank_conf = max(0.0, min(blank_conf, 1.0))
                score -= blank_conf * blank_penalty_scale
                if blank_conf >= 0.75:
                    score -= 8.0
                vals.append(score)
        if not vals:
            return -1.0
        coverage = len(vals) / float(max(len(cv_targets), 1))
        length_penalty = abs(len(cand) - len(cv_targets)) * 4.0
        return (sum(vals) / float(len(vals))) * coverage - length_penalty

    return max(_avg_with_shift(0), _avg_with_shift(1), _avg_with_shift(-1))


def _is_kr_glide_vowel(vowel):
    v = str(vowel or "").strip().lower()
    return v in {"ya", "ye", "yeo", "yo", "yu", "wa", "wae", "we", "weo", "wi", "wo"}


def _compute_kr_glide_mismatch_ratio(candidate_infos, cv_targets):
    """candidate 토큰과 target 토큰의 활음(이중모음) 불일치 비율을 반환합니다."""
    if not candidate_infos or not cv_targets:
        return 0.0

    cand = []
    for s in candidate_infos:
        tok = s.get("roman_cv") or s.get("roman") or s.get("word") or ""
        tok = _kr_cv_kernel(tok)
        if tok:
            cand.append(tok)

    if not cand:
        return 0.0

    n = min(len(cand), len(cv_targets))
    if n <= 0:
        return 0.0

    compared = 0
    mismatch = 0
    for i in range(n):
        _co, cv, _cc = _split_kr_syllable_parts(cand[i])
        _to, tv, _tc = _split_kr_syllable_parts(cv_targets[i])
        if not cv or not tv:
            continue
        compared += 1
        if _is_kr_glide_vowel(cv) != _is_kr_glide_vowel(tv):
            mismatch += 1

    if compared <= 0:
        return 0.0
    return float(mismatch) / float(compared)


def _should_prefer_alias_based_syllables(file_format, used_words_based, base_score, alt_score):
    """한국어 음절열 후보 중 alias/filename 기반 결과를 채택할지 결정합니다."""
    fmt = str(file_format or "").strip().lower()
    base_score = float(base_score)
    alt_score = float(alt_score)

    if not used_words_based:
        return alt_score >= base_score

    if fmt == "cvvc":
        return alt_score >= 40.0

    if fmt == "cvc":
        return alt_score >= 60.0 and alt_score >= (base_score - 2.0)

    if fmt == "vcv":
        # VCV는 음절 경계가 한 칸만 밀려도 체감 오류가 커서,
        # alias 기반 점수가 words 기반보다 확실히 우세하면 더 적극적으로 전환한다.
        if alt_score >= max(base_score + 4.0, 62.0):
            return True
        return base_score < 48.0 and alt_score >= 56.0

    allow_alias_override = True
    if used_words_based and base_score >= 58.0:
        allow_alias_override = False
    return allow_alias_override and base_score < 66.0 and alt_score >= 70.0 and alt_score >= (base_score + 8.0)


def _clamp_kr_cv_index_to_order(
    file_format,
    target_clean,
    romaji_syllables,
    expected_idx,
    mapped_idx,
):
    fmt = str(file_format or "").strip().lower()
    if not _is_kr_order_locked_cv_format(fmt):
        return int(mapped_idx)
    if not romaji_syllables:
        return int(expected_idx)

    e = max(0, min(int(expected_idx), len(romaji_syllables) - 1))
    m = max(0, min(int(mapped_idx), len(romaji_syllables) - 1))
    if m <= e:
        return e

    target_norm = _normalize_cv_match_token(target_clean)
    expected_tok = _normalize_cv_match_token(romaji_syllables[e])
    mapped_tok = _normalize_cv_match_token(romaji_syllables[m])
    expected_score = float(_cv_match_score(target_norm, expected_tok))
    mapped_score = float(_cv_match_score(target_norm, mapped_tok))

    if fmt == "cvc":
        return e

    if m > (e + 1):
        return e
    if mapped_tok == target_norm and expected_tok != target_norm:
        return m
    if mapped_score >= max(expected_score + 18.0, 92.0):
        return m
    return e


__all__ = [
    "_alias_to_cv_target",
    "_build_kr_cvvc_occurrence_map",
    "_build_kr_cvvc_vv_occurrence_map",
    "_clamp_kr_cv_index_to_order",
    "_compute_kr_glide_mismatch_ratio",
    "_extract_cv_targets_from_lines",
    "_extract_kr_cv_targets_from_filename",
    "_extract_kr_vv_pair_key",
    "_find_kr_cv_vowel_match_index",
    "_is_kr_order_locked_cv_format",
    "_iter_kr_cvvc_tokens",
    "_resolve_kr_cvvc_occurrence_index",
    "_resolve_kr_cvvc_vv_index",
    "_score_kr_syllable_mapping",
    "_should_allow_kr_exact_vowel_fix",
    "_should_prefer_alias_based_syllables",
]
