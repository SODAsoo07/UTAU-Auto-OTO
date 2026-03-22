from __future__ import annotations

from functools import lru_cache

from core.kr_oto_rules import _cv_match_score, _is_kr_glide_vowel, _split_kr_syllable_parts


@lru_cache(maxsize=8192)
def _split_parts_cached(token):
    return _split_kr_syllable_parts(token)


@lru_cache(maxsize=65536)
def _cv_match_score_cached(target, token):
    return _cv_match_score(target, token)


def resolve_cv_syllable_index(
    target_clean,
    romaji_syllables,
    cv_seq_idx,
    current_w_idx,
    *,
    mapping_confidence=1.0,
    max_jump_default=1,
    max_jump_high_conf=2,
    high_conf_threshold=0.82,
    return_meta=False,
):
    """CV 계열 alias를 words/roman 음절 인덱스에 매핑합니다."""
    meta = {
        "jump_blocked": 0,
        "raw_chosen_idx": int(cv_seq_idx),
        "chosen_idx": int(cv_seq_idx),
        "max_forward_jump": int(max_jump_default),
        "best_score": -1.0,
        "expected_score": -1.0,
        "mapping_confidence": float(mapping_confidence or 0.0),
    }
    if cv_seq_idx >= len(romaji_syllables):
        if return_meta:
            return current_w_idx, cv_seq_idx, meta
        return current_w_idx, cv_seq_idx

    name_match_idx = None
    best_score = -1
    scan_start = max(cv_seq_idx - 1, 0)
    scan_end = min(cv_seq_idx + 3, len(romaji_syllables))
    for i in range(scan_start, scan_end):
        score = _cv_match_score_cached(target_clean, romaji_syllables[i])
        score -= abs(i - cv_seq_idx) * 4
        if score > best_score:
            best_score = score
            name_match_idx = i
        if score >= 98:
            break

    expected_score = -1
    if 0 <= cv_seq_idx < len(romaji_syllables):
        expected_score = _cv_match_score_cached(target_clean, romaji_syllables[cv_seq_idx])
    meta["best_score"] = float(best_score)
    meta["expected_score"] = float(expected_score)

    if name_match_idx is not None and best_score >= 64:
        chosen_idx = name_match_idx
        best_gain = best_score - expected_score
        target_onset, target_vowel, _target_coda = _split_parts_cached(target_clean)
        expected_tok = romaji_syllables[cv_seq_idx] if 0 <= cv_seq_idx < len(romaji_syllables) else ""
        best_tok = romaji_syllables[name_match_idx] if 0 <= name_match_idx < len(romaji_syllables) else ""
        exp_onset, exp_vowel, _exp_coda = _split_parts_cached(expected_tok)
        best_onset, best_vowel, _best_coda = _split_parts_cached(best_tok)
        same_vowel_expected = bool(target_vowel and exp_vowel and target_vowel == exp_vowel)
        best_vowel_match = bool(target_vowel and best_vowel and target_vowel == best_vowel)
        target_glide = _is_kr_glide_vowel(target_vowel)
        expected_glide = _is_kr_glide_vowel(exp_vowel)
        best_glide = _is_kr_glide_vowel(best_vowel)
        same_onset_expected = bool(
            target_onset and exp_onset and (target_onset == exp_onset or target_onset[:1] == exp_onset[:1])
        )
        if name_match_idx > cv_seq_idx and expected_score >= max(50, best_score - 20):
            chosen_idx = cv_seq_idx
        if name_match_idx != cv_seq_idx and (not best_vowel_match):
            if best_gain < 30:
                chosen_idx = cv_seq_idx
            elif name_match_idx > cv_seq_idx and float(mapping_confidence or 0.0) < 0.72:
                chosen_idx = cv_seq_idx
        if name_match_idx > (cv_seq_idx + 1):
            if expected_score >= 46:
                chosen_idx = cv_seq_idx
            elif best_gain < 30:
                chosen_idx = cv_seq_idx
            elif same_vowel_expected and best_gain < 34:
                chosen_idx = cv_seq_idx
            elif best_glide != target_glide and best_gain < 46:
                chosen_idx = cv_seq_idx
        if abs(name_match_idx - cv_seq_idx) == 1:
            min_gain = 22
            if same_vowel_expected:
                min_gain = 18
            elif same_onset_expected:
                min_gain = 20
            if best_glide != target_glide:
                min_gain = max(min_gain, 28)
            if best_gain < min_gain:
                chosen_idx = cv_seq_idx
            if same_vowel_expected and (not best_vowel_match) and name_match_idx > cv_seq_idx:
                chosen_idx = cv_seq_idx
            if same_vowel_expected and (not best_vowel_match) and best_gain < 22:
                chosen_idx = cv_seq_idx
            if same_vowel_expected and best_glide != target_glide:
                chosen_idx = cv_seq_idx
            if (
                same_onset_expected
                and (not (target_onset and best_onset and target_onset[:1] == best_onset[:1]))
                and best_gain < 24
            ):
                chosen_idx = cv_seq_idx
        if name_match_idx < cv_seq_idx:
            if same_vowel_expected or expected_score >= 38 or best_gain < 32:
                chosen_idx = cv_seq_idx
            if same_vowel_expected and expected_glide == target_glide and best_glide != target_glide:
                chosen_idx = cv_seq_idx
        max_forward_jump = int(max(0, max_jump_default))
        conf = float(mapping_confidence or 0.0)
        if conf < 0.64:
            max_forward_jump = 0
        if conf >= float(high_conf_threshold):
            if best_score >= 84 and best_gain >= 24 and best_vowel_match:
                max_forward_jump = int(max(max_forward_jump, max_jump_high_conf))
        meta["max_forward_jump"] = int(max_forward_jump)
        raw_chosen_idx = int(chosen_idx)
        if chosen_idx > (cv_seq_idx + max_forward_jump):
            chosen_idx = cv_seq_idx + max_forward_jump
            meta["jump_blocked"] = 1
        meta["raw_chosen_idx"] = int(raw_chosen_idx)
        meta["chosen_idx"] = int(chosen_idx)
        if chosen_idx < cv_seq_idx:
            chosen_idx = cv_seq_idx
        current_w_idx = chosen_idx
    else:
        current_w_idx = cv_seq_idx

    cv_seq_idx = current_w_idx + 1
    if return_meta:
        return current_w_idx, cv_seq_idx, meta
    return current_w_idx, cv_seq_idx


__all__ = ["resolve_cv_syllable_index"]
