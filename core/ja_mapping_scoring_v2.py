from __future__ import annotations

import os
import re

from core.ja_lab_generator import split_ja_romaji_syllable
from core.ja_oto_mapping import (
    _extract_ja_onset_token,
    _ja_soft_cv_match_level,
    _ja_special_mora_class,
    _ja_syllable_tail,
    _normalize_ja_syllable_token,
    _syllable_info_token,
)

JA_VOWELS = {"a", "i", "u", "e", "o"}
_JA_KANA_RE = re.compile(r"[ぁ-ゖァ-ヺー]")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _normalize_ja_syllable_token_strict(token):
    raw = str(token or "").strip()
    if not raw:
        return ""
    return raw.lower().replace("'", "").strip("-_ ")


def _is_kana_token(token):
    return bool(_JA_KANA_RE.search(str(token or "")))


def _syllable_info_raw_token(syl_info):
    if not isinstance(syl_info, dict):
        return ""
    for k in ("roman", "word", "kana"):
        raw = syl_info.get(k, "")
        if raw:
            return str(raw).strip()
    return ""


def should_allow_ja_soft_forward_shift(target_tok, expected_tok, mapped_tok):
    target_norm = _normalize_ja_syllable_token(target_tok)
    expected_norm = _normalize_ja_syllable_token(expected_tok)
    mapped_norm = _normalize_ja_syllable_token(mapped_tok)
    if not target_norm or not mapped_norm:
        return False
    target_strict = _normalize_ja_syllable_token_strict(target_tok)
    mapped_strict = _normalize_ja_syllable_token_strict(mapped_tok)
    target_special = _ja_special_mora_class(target_tok)
    expected_special = _ja_special_mora_class(expected_tok)
    mapped_special = _ja_special_mora_class(mapped_tok)
    if target_special in {"plain", "youon", "inserted"} and mapped_special != target_special:
        return False
    if expected_special in {"plain", "youon", "inserted"} and mapped_special != expected_special:
        return False
    if target_special in {"nasal", "geminate", "long"}:
        return False
    if target_special == "plain" and mapped_special in {"nasal", "geminate", "long"}:
        return False
    mapped_level = int(_ja_soft_cv_match_level(target_norm, mapped_norm) or 0)
    expected_level = int(_ja_soft_cv_match_level(target_norm, expected_norm) or 0)
    if target_special in {"youon", "inserted"}:
        if (
            target_strict
            and mapped_strict
            and target_strict != mapped_strict
            and not (_is_kana_token(target_tok) or _is_kana_token(mapped_tok))
        ):
            return False
        if mapped_special not in {"youon", "inserted"} and mapped_level < 3:
            return False
        if expected_special in {"youon", "inserted"} and mapped_special not in {"youon", "inserted"}:
            return False
    if mapped_level >= 3 and expected_level < 3:
        return True
    if mapped_level >= 2 and expected_level <= 1:
        return True
    return False


def clamp_ja_cv_index_to_order(
    target_tok,
    expected_idx,
    mapped_idx,
    syllables_info,
    *,
    format_type,
    filename_order_locked,
    mapping_tier,
):
    if not syllables_info:
        return int(expected_idx)
    fmt = str(format_type or "").strip().lower()
    if fmt not in {"cvvc", "cv"}:
        return int(mapped_idx)

    e = max(0, min(int(expected_idx), len(syllables_info) - 1))
    m = max(0, min(int(mapped_idx), len(syllables_info) - 1))
    if m <= e:
        return e if m < e else m

    target_norm = _normalize_ja_syllable_token(target_tok)
    expected_raw = _syllable_info_raw_token(syllables_info[e])
    mapped_raw = _syllable_info_raw_token(syllables_info[m])
    expected_norm = _normalize_ja_syllable_token(_syllable_info_token(syllables_info[e]))
    mapped_norm = _normalize_ja_syllable_token(_syllable_info_token(syllables_info[m]))
    expected_level = int(_ja_soft_cv_match_level(target_norm, expected_norm) or 0) if target_norm else 0
    mapped_level = int(_ja_soft_cv_match_level(target_norm, mapped_norm) or 0) if target_norm else 0

    if fmt == "cv":
        # CV는 파일명/순서 정합을 최우선으로 두고 전진 매핑(+1)도 차단한다.
        return e

    # CVVC에서 CV/CV_HEAD 전방 이동은 오매핑을 유발하기 쉬우므로
    # 기본값은 저티어/순서잠금 상황에서 차단한다.
    # 필요 시 `UTOA_JA_CVVC_STRICT_CV_FORWARD=0`으로 이전 동작을 복원 가능.
    strict_cvvc_forward = _env_bool("UTOA_JA_CVVC_STRICT_CV_FORWARD", True)
    tier = str(mapping_tier or "").strip().lower()
    if fmt == "cvvc" and strict_cvvc_forward and (filename_order_locked or tier != "high"):
        return e

    if m > (e + 1):
        return e
    if not target_norm:
        return e

    allow_forward = bool(
        m == (e + 1)
        and should_allow_ja_soft_forward_shift(
            target_tok,
            expected_raw or expected_norm,
            mapped_raw or mapped_norm,
        )
    )
    if allow_forward:
        if mapped_norm == target_norm and expected_norm != target_norm:
            return m
        if mapped_level >= 2 and expected_level <= 1:
            return m
        if not filename_order_locked and str(mapping_tier or "").strip().lower() == "high" and mapped_level >= 3:
            return m
    return e


def find_ja_cv_vowel_match_index(target_tok, expected_idx, syllables_info, search_back=1, search_fwd=2):
    if not target_tok or not syllables_info:
        return None
    n = len(syllables_info)
    if n <= 0:
        return None
    e = max(0, min(int(expected_idx), n - 1))

    t_onset = _extract_ja_onset_token(target_tok)
    _to, t_vowel = split_ja_romaji_syllable(target_tok)
    if t_vowel not in JA_VOWELS:
        return None
    t_tail = _ja_syllable_tail(target_tok)
    target_special = _ja_special_mora_class(target_tok)
    target_strict = _normalize_ja_syllable_token_strict(target_tok)
    expected_soft_match = _ja_soft_cv_match_level(
        target_tok,
        _syllable_info_token(syllables_info[e]),
    )

    lo = max(0, e - max(0, int(search_back)))
    hi = min(n - 1, e + max(0, int(search_fwd)))
    best_idx = None
    best_score = -10**9
    for i in range(lo, hi + 1):
        cand_raw = _syllable_info_raw_token(syllables_info[i])
        cand = _syllable_info_token(syllables_info[i])
        if not cand:
            continue
        cand_strict = _normalize_ja_syllable_token_strict(cand_raw) if cand_raw else ""
        if target_special in {"plain", "youon", "inserted"}:
            cand_special = _ja_special_mora_class(cand)
            if cand_special != target_special:
                continue
            if (
                target_special in {"youon", "inserted"}
                and target_strict
                and cand_strict
                and target_strict != cand_strict
                and not (_is_kana_token(target_tok) or _is_kana_token(cand_raw))
            ):
                continue
        c_onset = _extract_ja_onset_token(cand)
        _co, c_vowel = split_ja_romaji_syllable(cand)
        if c_vowel != t_vowel:
            continue
        soft_match = _ja_soft_cv_match_level(target_tok, cand)

        score = 100 - (abs(i - e) * 18)
        if t_onset and c_onset:
            if c_onset == t_onset:
                score += 20
            elif c_onset[:1] == t_onset[:1]:
                score += 8
            else:
                score -= 12
        if soft_match >= 2:
            score += 24
        elif soft_match == 1:
            score += 8
        if soft_match > expected_soft_match:
            score += 10
        c_tail = _ja_syllable_tail(cand)
        if t_tail == c_tail:
            score += 6
        elif t_tail and c_tail and t_tail != c_tail:
            score -= 4

        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def find_ja_exact_target_index(target_tok, expected_idx, syllables_info, search_back=3, search_fwd=3):
    if not target_tok or not syllables_info:
        return None
    n = len(syllables_info)
    if n <= 0:
        return None
    e = max(0, min(int(expected_idx), n - 1))
    tgt_norm = _normalize_ja_syllable_token(target_tok)
    if not tgt_norm:
        return None
    exp_norm = _normalize_ja_syllable_token(_syllable_info_token(syllables_info[e]))
    if exp_norm == tgt_norm:
        return None

    lo = max(0, e - max(0, int(search_back)))
    hi = min(n - 1, e + max(0, int(search_fwd)))
    best_idx = None
    best_key = None
    for i in range(lo, hi + 1):
        cand_norm = _normalize_ja_syllable_token(_syllable_info_token(syllables_info[i]))
        if cand_norm != tgt_norm:
            continue
        key = (abs(i - e), 0 if i > e else 1, i)
        if best_key is None or key < best_key:
            best_key = key
            best_idx = i
    return best_idx


def prefer_vcv_candidate_index(expected_idx, mapped_idx, target_tok, syllables_info, max_delta=1):
    if not syllables_info:
        return expected_idx
    n = len(syllables_info)
    e = max(0, min(int(expected_idx), n - 1))
    m = max(0, min(int(mapped_idx), n - 1))
    if m == e:
        return e
    if abs(m - e) > max(0, int(max_delta)):
        return e

    tgt = _normalize_ja_syllable_token(target_tok)
    if not tgt:
        return e
    exp_tok = _normalize_ja_syllable_token(_syllable_info_token(syllables_info[e]))
    map_tok = _normalize_ja_syllable_token(_syllable_info_token(syllables_info[m]))
    if map_tok != tgt:
        return e
    if exp_tok == tgt:
        return e
    return m


__all__ = [
    "JA_VOWELS",
    "clamp_ja_cv_index_to_order",
    "find_ja_cv_vowel_match_index",
    "find_ja_exact_target_index",
    "prefer_vcv_candidate_index",
    "should_allow_ja_soft_forward_shift",
]
