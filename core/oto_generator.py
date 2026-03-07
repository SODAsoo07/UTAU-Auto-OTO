"""
TextGrid를 기반으로 한국어 OTO.ini를 생성합니다.
- phones/words tier를 이용해 OTO 파라미터를 계산합니다.
- CV/VC/VCV/VV/단모음/숨소리(br) 케이스를 처리합니다.
"""

import os
import re
import json
import wave
import datetime
from dataclasses import replace
import logging
from functools import lru_cache
from types import SimpleNamespace
import textgrid
import copy

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None
from core.lab_generator import load_custom_phonemes
from core.kr_oto_rules import (
    GLOTTAL_MARKS,
    IPA_PLOSIVES,
    IPA_VOWELS,
    KR_BATCHIM_MARKERS,
    KR_CONSONANTS,
    KR_PLOSIVE_ONSETS,
    KR_SIBILANT_ONSETS,
    KR_SONORANT_CONSONANTS,
    KR_TENSE_CONSONANTS,
    KR_VOICED_ONSETS,
    KR_VOICELESS_ONSETS,
    KR_VOWELS,
    _canonicalize_kr_coda,
    _cv_match_score,
    _detect_glottal_kind,
    _extract_alias_onset,
    _extract_kr_cv_alias_token,
    _extract_vc_right_token,
    _extract_vowel_consonant,
    _is_kr_closure_token,
    _is_kr_glide_vowel,
    _is_kr_plosive_coda_alias,
    _is_sonorant_consonant,
    _is_tense_consonant,
    _kr_cv_kernel,
    _looks_like_vv_alias,
    _normalize_cv_match_token,
    _split_kr_syllable_parts,
    classify_alias,
    detect_alias_format,
    find_vowel_phone,
    is_breath,
    is_glide,
    is_plosive_ipa,
    is_plosive_roman,
    normalize_ipa_mark,
)
from core.kr_oto_mapping import (
    _alias_to_cv_target,
    _build_kr_cvvc_occurrence_map,
    _build_kr_cvvc_vv_occurrence_map,
    _extract_cv_targets_from_lines,
    _extract_kr_cv_targets_from_filename,
    _find_kr_cv_vowel_match_index,
    _iter_kr_cvvc_tokens,
    _resolve_kr_cvvc_occurrence_index,
    _resolve_kr_cvvc_vv_index,
    _score_kr_syllable_mapping,
    _should_allow_kr_exact_vowel_fix,
    _should_prefer_alias_based_syllables,
)
from core.kr_oto_bridge import (
    _apply_kr_consonant_timing_shaping,
    _compute_kr_cvvc_vc_timing_direct,
    _compute_vc_from_adjacent_cv,
    _refine_kr_bridge_with_adjacent_cv,
    _recenter_kr_params_around_pre,
)
from core.kr_oto_cv import (
    _compute_kr_cv_timing,
    _estimate_cv_anchor_from_syllable,
    _prepare_cv_bounds_from_syllable,
    _prepare_cv_head_syllable_timing,
    _select_kr_cv_onset_slice,
)
from core.kr_oto_vc import (
    _compute_kr_vc_timing,
    _prepare_vc_bounds_from_context,
    _uses_kr_vc_context,
)
from core.kr_oto_vv import _compute_kr_cvvc_vv_timing_direct
from core.kr_oto_postprocess import (
    KrPostprocessContext,
    guard_kr_vc_cutoff_to_next_segment as _guard_kr_vc_cutoff_to_next_segment_core,
    log_post_timing_events as _log_post_timing_events_core,
)
from core.timing_anchor_profiles import (
    get_anchor_profile,
    is_anchor_lock_enabled,
)
from core.timing_anchor_runtime import (
    AnchorTimingContext,
    apply_anchor_lock,
    append_timing_anchor_log,
)
from core.textio_utils import load_template_oto_lines
from core.oto_profile_presets import get_kr_profile_preset

logger = logging.getLogger(__name__)

# ==============================================================================
# 접미사 처리 유틸리티
# ==============================================================================

def _normalize_alias_suffix(suffix):
    s = (suffix or "").strip()
    if not s:
        return ""
    return s[1:] if s.startswith("_") else s


def apply_alias_suffix(alias, suffix):
    """에일리어스 끝에 `_<suffix>` 접미사를 붙입니다."""
    suf = _normalize_alias_suffix(suffix)
    if not suf:
        return alias
    a = (alias or "").strip()
    if not a:
        return alias
    return f"{a}_{suf}"


def apply_suffix_to_oto_line(line, suffix):
    """`wav=alias,params...` 라인에서 alias에만 접미사를 적용합니다."""
    suf = _normalize_alias_suffix(suffix)
    if not suf:
        return line
    if not line or '=' not in line:
        return line
    left, right = line.split('=', 1)
    if ',' in right:
        alias_part, rest = right.split(',', 1)
        alias_part = apply_alias_suffix(alias_part.strip(), suf)
        return f"{left}={alias_part},{rest}"
    alias_part = apply_alias_suffix(right.strip(), suf)
    return f"{left}={alias_part}"

# ==============================================================================
# 기본 튜닝 파라미터
# ==============================================================================

DEFAULT_PARAMS = {
    'VC_CONSONANT_RATIO': 0.5,
    'VC_VOWEL_START': 0.3,
    'VC_PRE_OFFSET': 25,
    'VC_OVL_RATIO': 0.3,
    'CV_PRE_RATIO': 1.0,
    'CV_OVL_RATIO': 0.4,
    'CV_VOWEL_USE': 0.9,
    'DIPHTHONG_CV_PRE_RATIO': 0.35,
    'DIPHTHONG_CV_CONSONANT_RATIO': 0.6,
    'DIPHTHONG_CV_VOWEL_USE': 0.8,
    'DIPHTHONG_VC_VOWEL_START': 0.3,
    'DIPHTHONG_VC_CONSONANT': 0.5,
    'DIPHTHONG_VC_PRE_EXTEND': 1.2,
}

# 한국어 매핑 신뢰도 임계치 기본값(포맷별)
# - CVVC: 현재 안정성 기준값 유지
# - VCV: 점프 허용 전 신뢰도를 조금 더 엄격하게 본다
# - CVC/CV_SIMPLE: 정보량이 상대적으로 단순해 과도한 저신뢰 판정을 완화
# - VC_ONLY/VV_ONLY: CV 정렬 점프 로직 영향이 거의 없어 완화값 사용
KR_MAPPING_CONF_THRESHOLD_BY_FORMAT = {
    "cvvc": 0.60,
    "vcv": 0.62,
    "cvc": 0.58,
    "cv_simple": 0.58,
    "mono": 0.58,
    "vc_only": 0.56,
    "vv_only": 0.56,
    "default": 0.60,
}


def _resolve_kr_mapping_conf_threshold(file_format, override_threshold=None):
    if override_threshold is not None:
        try:
            return float(override_threshold)
        except Exception:
            pass
    fmt = str(file_format or "").strip().lower()
    base = KR_MAPPING_CONF_THRESHOLD_BY_FORMAT.get(
        fmt,
        KR_MAPPING_CONF_THRESHOLD_BY_FORMAT["default"],
    )
    return float(base)


def normalize_key(name):
    base = os.path.splitext(name)[0]
    clean = re.sub(r"[^a-zA-Z0-9가-힣]", "", base)
    return clean.lower()


def is_diphthong_file(filename):
    """파일명에 이중모음 표식이 포함되는지 확인합니다."""
    clean = filename.lower().replace("'", "").replace(".wav", "")
    diphthong_markers = [
        'ya','yeo','yo','yu','ye','wa','wo','wi','we','weo','eui',
        'gya','gye','gyeo','nya','nye','nyeo','dya','dye','dyeo',
        'rya','rye','ryeo','mya','mye','myeo','bya','bye','byeo',
        'sya','sye','syeo','jya','jye','jyeo','chya','chye','chyeo',
        'kya','kye','kyeo','tya','tye','tyeo','pya','pye','pyeo',
        'hya','hye','hyeo',
        'gwa','gwe','gwi','gweo','nwa','nwe','nwi','nweo',
        'dwa','dwe','dwi','dweo','rwa','rwe','rwi','rweo',
        'mwa','mwe','mwi','mweo','bwa','bwe','bwi','bweo',
        'swa','swe','swi','sweo','jwa','jwe','jwi','jweo',
        'chwa','chwe','chwi','chweo','kwa','kwe','kwi','kweo',
        'twa','twe','twi','tweo','pwa','pwe','pwi','pweo',
        'hwa','hwe','hwi','hweo',
    ]
    for diph in diphthong_markers:
        if diph in clean:
            return True
    return False


@lru_cache(maxsize=16384)
def is_diphthong(alias):
    """에일리어스 문자열에 이중모음 패턴이 있는지 확인합니다."""
    clean = alias.replace(' ', '').lower()
    diphthongs = ['ya','yeo','yo','yu','ye','wa','wo','wi','we','weo','eui','ui']
    for diph in diphthongs:
        if diph in clean:
            return True
    return False


def adaptive_overlap(pre, consonant_hint="", mode="cv"):
    """
    자음 성질/에일리어스 타입에 따라 overlap을 동적으로 조정합니다.
    """
    p = max(float(pre), 0.0)
    if p <= 0:
        return 0.0

    hint = normalize_ipa_mark(consonant_hint or "")
    hard = {'k', 'g', 't', 'd', 'p', 'b', 'c', 'ch', 'j', 'kk', 'tt', 'pp', 'gg', 'dd', 'bb'}
    fric = {'s', 'ss', 'sh', 'h', 'f', 'z'}
    sonorant = {'m', 'n', 'ng', 'r', 'l', 'y', 'w'}

    base_by_mode = {
        'cv': 0.46,
        'cv_head': 0.40,
        'vcv': 0.50,
        'vc': 0.44,
        'vv': 0.54,
    }
    ratio = base_by_mode.get(mode, 0.46)

    if hint in hard:
        ratio -= 0.16
    elif hint in fric:
        ratio += 0.05
    elif hint in sonorant:
        ratio += 0.08

    ratio = max(0.20, min(0.72, ratio))
    ovl = p * ratio
    if mode in ('vc', 'vv', 'vcv'):
        ovl = min(ovl, max(p - 14.0, 0.0))
    return max(0.0, min(ovl, p))


def _guard_cv_cutoff_to_next_onset(offset, consonant, cutoff, pre, syll_idx, syllables_info):
    """CV/CV_HEAD cutoff이 다음 음절 onset을 침범하지 않도록 상한을 건다."""
    if syll_idx is None or syll_idx < 0:
        return offset, consonant, cutoff, pre, 0.0
    if (syll_idx + 1) >= len(syllables_info):
        return offset, consonant, cutoff, pre, 0.0

    next_syl = syllables_info[syll_idx + 1]
    next_phones = next_syl.get("phones") or []
    if not next_phones:
        return offset, consonant, cutoff, pre, 0.0

    next_mark = (next_phones[0].mark or "").strip().lower()
    hard_next = is_plosive_ipa(next_mark) or next_mark in {
        "s", "ss", "sh", "ch", "j", "jj", "c", "ts", "h"
    }
    safety = 12.0 if hard_next else 7.0
    next_onset_rel = (next_phones[0].minTime * 1000.0) - offset
    max_cutoff_abs = next_onset_rel - safety
    if max_cutoff_abs <= (pre + 26.0):
        return offset, consonant, cutoff, pre, 0.0

    original_cutoff_abs = abs(cutoff)
    consonant = min(consonant, max_cutoff_abs - 14.0)
    consonant = max(consonant, pre + 16.0)

    cutoff_abs = min(original_cutoff_abs, max_cutoff_abs)
    if cutoff_abs <= (consonant + 8.0):
        cutoff_abs = min(max_cutoff_abs, consonant + 14.0)
        if cutoff_abs <= (consonant + 6.0):
            consonant = max(pre + 12.0, cutoff_abs - 12.0)
    cutoff = -cutoff_abs

    offset, consonant, cutoff, pre, _ovl = validate_oto_params(
        offset, consonant, cutoff, pre, 0.0
    )
    reduction = max(0.0, original_cutoff_abs - abs(cutoff))
    return offset, consonant, cutoff, pre, reduction


def _guard_kr_vc_cutoff_to_next_segment(offset, consonant, cutoff, pre, ovl, syll_idx, syllables_info, alias_text=""):
    """한국어 VC cutoff 가드(호환 래퍼)."""
    return _guard_kr_vc_cutoff_to_next_segment_core(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        syll_idx,
        syllables_info,
        validate_oto_params,
        alias_text=alias_text,
    )


def _guard_cv_head_offset_to_current_onset(offset, consonant, cutoff, pre, syll_idx, syllables_info):
    """
    CV_HEAD(- CV) offset이 현재 음절 onset보다 과도하게 앞서지 않도록 제한합니다.
    공백 영역 과포함을 줄이기 위한 가드입니다.
    """
    if syll_idx is None or syll_idx < 0:
        return offset, consonant, cutoff, pre, 0.0
    if not syllables_info or syll_idx >= len(syllables_info):
        return offset, consonant, cutoff, pre, 0.0

    curr_syl = syllables_info[syll_idx]
    curr_phones = curr_syl.get("phones") or []
    if not curr_phones:
        return offset, consonant, cutoff, pre, 0.0

    c_start = float(curr_phones[0].minTime) * 1000.0
    c_hint = (curr_phones[0].mark or "").strip().lower()
    try:
        _v_idx, v_phone = find_vowel_phone(curr_phones)
        c_end = float(v_phone.minTime) * 1000.0
    except Exception:
        c_end = float(curr_phones[0].maxTime) * 1000.0
    c_len = max(0.0, c_end - c_start)

    # onset 특성별 허용 리드(ms): 파열/치찰은 조금 넓게, 공명음은 더 타이트하게.
    if is_plosive_ipa(c_hint) or c_hint in {"s", "ss", "sh", "ch", "j", "jj", "c", "ts", "h"}:
        base_lead = 44.0
    elif c_hint in {"m", "n", "ng", "l", "r", "y", "w", "ny"}:
        base_lead = 30.0
    else:
        base_lead = 36.0
    lead_cap = min(base_lead, max(22.0, c_len + 16.0))
    offset_floor = max(0.0, c_start - lead_cap)
    if offset >= offset_floor:
        return offset, consonant, cutoff, pre, 0.0

    new_offset = offset_floor
    # 오프셋만 당길 때 상대 길이(pre/cons/cutoff)가 급격히 줄어들지 않도록
    # 기존 상대 파라미터를 우선 보존한다.
    new_pre = max(float(pre), 8.0)
    new_consonant = max(float(consonant), new_pre + 8.0)
    new_cut_abs = max(abs(float(cutoff)), new_consonant + 12.0)
    new_cutoff = -new_cut_abs

    new_offset, new_consonant, new_cutoff, new_pre, _ovl = validate_oto_params(
        new_offset, new_consonant, new_cutoff, new_pre, 0.0
    )
    reduced_ms = max(0.0, new_offset - offset)
    return new_offset, new_consonant, new_cutoff, new_pre, reduced_ms


def _ensure_cv_head_min_vowel_coverage(offset, consonant, cutoff, pre, vowel_start_ms, vowel_end_ms):
    """
    CV_HEAD(-CV)에서 컷오프가 너무 이르게 닫혀 모음이 거의 포함되지 않는 경우를 방지합니다.
    """
    v_start = float(vowel_start_ms)
    v_end = float(vowel_end_ms)
    v_len = max(0.0, v_end - v_start)
    if v_len < 40.0:
        return offset, consonant, cutoff, pre, 0.0

    cut_abs = abs(float(cutoff))
    # 모음 구간을 최소 일부(비율+하한) 포함하도록 컷오프 하한을 설정.
    keep_v_ms = min(max(v_len * 0.30, 70.0), 190.0)
    vowel_start_rel = max(v_start - float(offset), float(pre) + 8.0)
    # pre 이후 너무 빨리 닫히는 케이스(자음만 남는 길이)를 방지.
    min_from_pre = min(max(v_len * 0.24, 90.0), 180.0)
    min_cut_abs = max(float(consonant) + 12.0, vowel_start_rel + keep_v_ms, float(pre) + min_from_pre)
    if cut_abs >= min_cut_abs:
        return offset, consonant, cutoff, pre, 0.0

    new_cutoff = -min_cut_abs
    offset, consonant, new_cutoff, pre, _ovl = validate_oto_params(
        offset, consonant, new_cutoff, pre, 0.0
    )
    extended_ms = max(0.0, abs(new_cutoff) - cut_abs)
    return offset, consonant, new_cutoff, pre, extended_ms


def _prepare_vcv_syllable_timing(syllables_info, current_w_idx, cv_seq_idx, diphthong_cv_consonant_ratio):
    """VCV 계산에 필요한 음절 인덱스 갱신과 기본 타이밍을 산출합니다."""
    if cv_seq_idx < len(syllables_info):
        current_w_idx = cv_seq_idx
        cv_seq_idx = current_w_idx + 1
    if current_w_idx >= len(syllables_info):
        current_w_idx = len(syllables_info) - 1

    curr_syl = syllables_info[current_w_idx]
    curr_phones = curr_syl['phones']

    if current_w_idx > 0:
        prev_syl = syllables_info[current_w_idx - 1]
        prev_phones = prev_syl['phones']
        prev_v_start = prev_phones[-1].minTime * 1000
        prev_v_end = prev_phones[-1].maxTime * 1000
    else:
        prev_v_start = curr_phones[0].minTime * 1000 - 100
        prev_v_end = curr_phones[0].minTime * 1000

    if len(curr_phones) >= 2:
        c_boundary = curr_phones[-1].minTime * 1000
        n_end = curr_phones[-1].maxTime * 1000
    else:
        c_boundary = curr_phones[0].minTime * 1000
        n_end = curr_phones[0].maxTime * 1000

    prev_v_len = prev_v_end - prev_v_start
    offset_padding = min(prev_v_len * 0.6, 200)
    if offset_padding < 80:
        offset_padding = max(prev_v_len * 0.5, 50)

    offset = prev_v_end - offset_padding
    if offset < 0:
        offset = 0

    pre = c_boundary - offset
    c_hint = curr_phones[0].mark if curr_phones else ""
    ovl = adaptive_overlap(pre, c_hint, mode='vcv')

    vowel_len = n_end - c_boundary
    added_cons = min(vowel_len * diphthong_cv_consonant_ratio, 150)
    if added_cons < 50:
        added_cons = 50
    consonant = pre + added_cons
    cutoff = -(consonant + vowel_len * 0.25)
    return current_w_idx, cv_seq_idx, offset, consonant, cutoff, pre, ovl


def _append_alias_rows(
    final_lines,
    real_wav_name,
    alias,
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    generate_openutau=False,
    alias_suffix="",
):
    """에일리어스(및 OpenUtau 변형)를 OTO 라인으로 누적합니다."""
    aliases_to_write = generate_openutau_aliases(alias) if generate_openutau else [alias]
    for a in aliases_to_write:
        a2 = apply_alias_suffix(a, alias_suffix)
        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
        final_lines.append(new_line)


def _resolve_cv_syllable_index(
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
    # diphthong/r-l 혼용 케이스를 위해 탐색 범위를 넓혀 안정적으로 음절 정렬
    scan_start = max(cv_seq_idx - 1, 0)
    scan_end = min(cv_seq_idx + 4, len(romaji_syllables))
    for i in range(scan_start, scan_end):
        score = _cv_match_score(target_clean, romaji_syllables[i])
        score -= abs(i - cv_seq_idx) * 4
        if score > best_score:
            best_score = score
            name_match_idx = i
        if score >= 98:
            break

    expected_score = -1
    if 0 <= cv_seq_idx < len(romaji_syllables):
        expected_score = _cv_match_score(target_clean, romaji_syllables[cv_seq_idx])
    meta["best_score"] = float(best_score)
    meta["expected_score"] = float(expected_score)

    if name_match_idx is not None and best_score >= 62:
        chosen_idx = name_match_idx
        best_gain = best_score - expected_score
        target_onset, target_vowel, _target_coda = _split_kr_syllable_parts(target_clean)
        expected_tok = romaji_syllables[cv_seq_idx] if 0 <= cv_seq_idx < len(romaji_syllables) else ""
        best_tok = romaji_syllables[name_match_idx] if 0 <= name_match_idx < len(romaji_syllables) else ""
        exp_onset, exp_vowel, _exp_coda = _split_kr_syllable_parts(expected_tok)
        best_onset, best_vowel, _best_coda = _split_kr_syllable_parts(best_tok)
        same_vowel_expected = bool(target_vowel and exp_vowel and target_vowel == exp_vowel)
        best_vowel_match = bool(target_vowel and best_vowel and target_vowel == best_vowel)
        same_onset_expected = bool(
            target_onset and exp_onset and (target_onset == exp_onset or target_onset[:1] == exp_onset[:1])
        )
        # 이중모음/종성 포함 토큰에서 발생하는 과도한 앞 점프를 억제한다.
        if (
            name_match_idx > cv_seq_idx
            and expected_score >= max(50, best_score - 20)
        ):
            chosen_idx = cv_seq_idx
        if name_match_idx > (cv_seq_idx + 1):
            if expected_score >= 46:
                chosen_idx = cv_seq_idx
            elif best_gain < 30:
                chosen_idx = cv_seq_idx
            elif same_vowel_expected and best_gain < 34:
                chosen_idx = cv_seq_idx
        # 한 음절 점프는 충분히 큰 이득이 없으면 보수적으로 유지합니다.
        if abs(name_match_idx - cv_seq_idx) == 1:
            min_gain = 22
            if same_vowel_expected:
                min_gain = 18
            elif same_onset_expected:
                min_gain = 20
            if best_gain < min_gain:
                chosen_idx = cv_seq_idx
            # 기대 음절이 이미 target 모음을 만족하면, 모음 불일치 전진 점프를 차단한다.
            if same_vowel_expected and (not best_vowel_match) and name_match_idx > cv_seq_idx:
                chosen_idx = cv_seq_idx
            if same_vowel_expected and (not best_vowel_match) and best_gain < 22:
                chosen_idx = cv_seq_idx
            if same_onset_expected and (not (target_onset and best_onset and target_onset[:1] == best_onset[:1])) and best_gain < 24:
                chosen_idx = cv_seq_idx
        # 뒤로 가는 선택도 점수 이득이 충분하지 않으면 방지합니다.
        if name_match_idx < cv_seq_idx:
            if same_vowel_expected or expected_score >= 38 or best_gain < 32:
                chosen_idx = cv_seq_idx
        # 신뢰도 기반 전진 점프 제한.
        max_forward_jump = int(max(0, max_jump_default))
        conf = float(mapping_confidence or 0.0)
        if conf >= float(high_conf_threshold):
            if best_score >= 84 and best_gain >= 24:
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


def _extract_kr_anchor_target_token(alias_text, alias_type):
    a_type = str(alias_type or "").strip().lower()
    a = str(alias_text or "").strip().lower()
    if not a:
        return ""
    if a_type == "cv":
        return _normalize_cv_match_token(a)
    if a_type == "cv_head":
        parts = [p for p in re.split(r"\s+", a) if p]
        if len(parts) >= 2 and parts[0] == "-":
            return _normalize_cv_match_token(parts[1])
        return _normalize_cv_match_token(a.lstrip("-"))
    if a_type == "vcv":
        parts = [p for p in re.split(r"\s+", a) if p]
        if len(parts) >= 2:
            return _normalize_cv_match_token(parts[1])
    return ""


def _kr_alias_contains_coda(alias_text, alias_type):
    tok = _extract_kr_anchor_target_token(alias_text, alias_type)
    if not tok:
        return False
    _onset, vowel, coda = _split_kr_syllable_parts(tok)
    return bool(vowel and coda)


def _retune_kr_vcv_anchor_profile(profile, alias_text, alias_type):
    if profile is None:
        return profile
    if str(alias_type or "").strip().lower() not in {"vcv", "cv", "cv_head"}:
        return profile
    if not _kr_alias_contains_coda(alias_text, alias_type):
        return profile
    return replace(
        profile,
        pre_window_before_ms=max(1.0, float(profile.pre_window_before_ms) - 2.0),
        pre_window_after_ms=float(profile.pre_window_after_ms) + 4.0,
        pre_floor_ms=float(profile.pre_floor_ms) + 4.0,
        cons_gap_target_ms=min(float(profile.cons_gap_max_ms), float(profile.cons_gap_target_ms) + 8.0),
        cut_gap_target_ms=min(float(profile.cut_gap_max_ms), float(profile.cut_gap_target_ms) + 10.0),
        blend_weight=min(0.72, float(profile.blend_weight) + 0.06),
    )


def _apply_post_timing_pipeline(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    alias_type,
    alias_text,
    file_format,
    mel_ctx_for_file,
    base_shape,
    ph_intervals,
    current_w_idx,
    syllables_info,
    is_vc_plosive_coda=False,
    enable_stabilize=True,
    enable_cutoff_guard=True,
    post_ctx=None,
):
    """후처리 가드(soft mel/base shape/stabilize/cutoff)를 일관 적용합니다."""
    ctx = post_ctx or KrPostprocessContext(
        file_format=file_format,
        mel_ctx_for_file=mel_ctx_for_file,
        ph_intervals=ph_intervals,
        syllables_info=syllables_info,
        validate_fn=validate_oto_params,
        soft_mel_guard_fn=_apply_soft_mel_offset_cutoff_guard,
        base_shape_blend_fn=_apply_base_shape_blend,
        stabilize_fn=_stabilize_params_to_phone_activity,
        recenter_fn=_recenter_kr_params_around_pre,
        cv_cutoff_guard_fn=_guard_cv_cutoff_to_next_onset,
    )
    return ctx.apply(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        alias_type=alias_type,
        alias_text=alias_text,
        base_shape=base_shape,
        current_w_idx=current_w_idx,
        is_vc_plosive_coda=is_vc_plosive_coda,
        enable_stabilize=enable_stabilize,
        enable_cutoff_guard=enable_cutoff_guard,
    )


def _log_post_timing_events(log_fn, fname, alias, soft_off_shift, soft_cut_shift, cutoff_reduced):
    """후처리 가드 로그(호환 래퍼)."""
    _log_post_timing_events_core(log_fn, fname, alias, soft_off_shift, soft_cut_shift, cutoff_reduced)


def _is_kr_nucleus_phone_mark(mark):
    m = normalize_ipa_mark(mark)
    if not m:
        return False
    if m in IPA_VOWELS:
        return True
    return m in {"ɯ", "ʌ", "ɛ", "ə", "æ", "ɑ", "ɐ", "ɔ", "ɪ", "ʊ", "ø", "œ"}


def _map_kr_phone_vowel_to_roman(mark):
    m = normalize_ipa_mark(mark)
    return {
        "ʌ": "eo",
        "ɯ": "eu",
        "ɛ": "ae",
        "æ": "ae",
        "ə": "eo",
        "ø": "oe",
        "œ": "oe",
        "ɪ": "i",
        "ʊ": "u",
        "ɔ": "o",
        "ɑ": "a",
        "ɐ": "a",
    }.get(m, m)


def _estimate_kr_nucleus_token(ph_intervals, nuc_idx):
    """phone nuclei 근방에서 한국어 CV 핵심 토큰을 추정합니다."""
    if nuc_idx < 0 or nuc_idx >= len(ph_intervals):
        return ""
    vowel = _map_kr_phone_vowel_to_roman(ph_intervals[nuc_idx].mark)
    if not vowel:
        return ""

    prev_idx = nuc_idx - 1
    glide = ""
    if prev_idx >= 0 and is_glide(ph_intervals[prev_idx].mark):
        gmark = normalize_ipa_mark(ph_intervals[prev_idx].mark)
        if gmark in {"j", "y"}:
            glide = "y"
        elif gmark in {"w", "ʋ", "ɥ"}:
            glide = "w"
        prev_idx -= 1

    onset = ""
    if prev_idx >= 0:
        prev_mark = re.sub(r"[^a-z]", "", normalize_ipa_mark(ph_intervals[prev_idx].mark))
        if prev_mark and not _is_kr_nucleus_phone_mark(prev_mark) and not is_glide(prev_mark):
            onset = prev_mark

    glide_vowel = {
        ("y", "a"): "ya",
        ("y", "e"): "ye",
        ("y", "eo"): "yeo",
        ("y", "o"): "yo",
        ("y", "u"): "yu",
        ("w", "a"): "wa",
        ("w", "e"): "we",
        ("w", "eo"): "weo",
        ("w", "i"): "wi",
        ("w", "o"): "wo",
        ("w", "ae"): "wae",
        ("y", "ae"): "yae",
    }.get((glide, vowel), "")
    if glide_vowel:
        vowel = glide_vowel

    return _normalize_cv_match_token(f"{onset}{vowel}")


def _select_kr_nuclei_for_targets(ph_intervals, nuclei, cv_targets):
    """target 순서와 nuclei token score를 이용해 monotonic alignment를 수행합니다."""
    if not nuclei or not cv_targets:
        return None
    n_count = len(nuclei)
    t_count = len(cv_targets)
    if n_count < t_count:
        return None

    estimated = [_estimate_kr_nucleus_token(ph_intervals, idx) for idx in nuclei]
    dp = [[-10**9] * n_count for _ in range(t_count)]
    prev = [[-1] * n_count for _ in range(t_count)]

    for j in range(n_count):
        if j > (n_count - t_count):
            break
        base = _cv_match_score(cv_targets[0], estimated[j])
        base -= abs(j - 0) * 3
        dp[0][j] = base

    for i in range(1, t_count):
        lo = i
        hi = n_count - (t_count - i)
        for j in range(lo, hi):
            score_here = _cv_match_score(cv_targets[i], estimated[j])
            best_val = -10**9
            best_k = -1
            prev_lo = i - 1
            prev_hi = j
            for k in range(prev_lo, prev_hi):
                val = dp[i - 1][k]
                if val <= -10**8:
                    continue
                gap_penalty = max(0, (j - k - 1)) * 2
                val = val + score_here - gap_penalty
                if val > best_val:
                    best_val = val
                    best_k = k
            dp[i][j] = best_val
            prev[i][j] = best_k

    end_j = max(range(t_count - 1, n_count), key=lambda j: dp[t_count - 1][j], default=-1)
    if end_j < 0 or dp[t_count - 1][end_j] <= -10**8:
        return None

    picked = [0] * t_count
    cur = end_j
    for i in range(t_count - 1, -1, -1):
        picked[i] = nuclei[cur]
        cur = prev[i][cur]
        if i > 0 and cur < 0:
            return None
    return picked


def _build_kr_syllables_from_phone_nuclei(ph_intervals, cv_targets):
    if not ph_intervals or not cv_targets:
        return None

    target_n = len(cv_targets)
    nuclei = [
        i for i, p in enumerate(ph_intervals)
        if _is_kr_nucleus_phone_mark(p.mark) and not is_glide(p.mark)
    ]
    if len(nuclei) < target_n:
        # 저볼륨 녹음에서는 nucleus가 target 개수보다 적게 검출되는 경우가 있다.
        # 이때 매핑 전체를 포기하면 CVVC 파일이 통째로 mapping_failed로 떨어지므로,
        # phone 구간을 target 개수에 맞춰 contiguous chunk로 강제 분할한다.
        total = len(ph_intervals)
        if total <= 0:
            return None
        selected = []
        for i in range(target_n):
            pos = int(round(i * (total - 1) / float(max(target_n - 1, 1))))
            selected.append(max(0, min(total - 1, pos)))
    else:
        selected = _select_kr_nuclei_for_targets(ph_intervals, nuclei, cv_targets)
        if not selected:
            if target_n == 1:
                selected = [nuclei[0]]
            elif len(nuclei) == target_n:
                selected = list(nuclei)
            else:
                n_count = len(nuclei)
                selected_pos = []
                for i in range(target_n):
                    ideal = int(round(i * (n_count - 1) / float(target_n - 1)))
                    lo = i
                    hi = n_count - (target_n - i)
                    pos = max(lo, min(hi, ideal))
                    selected_pos.append(pos)
                selected = [nuclei[pos] for pos in selected_pos]

    if not selected:
        return None

    out = []
    global_start = float(ph_intervals[0].minTime)
    global_end = float(ph_intervals[-1].maxTime)
    for i, nuc_idx in enumerate(selected):
        if i == 0:
            s_t = global_start
        else:
            prev_n = selected[i - 1]
            s_t = (float(ph_intervals[prev_n].maxTime) + float(ph_intervals[nuc_idx].minTime)) * 0.5

        if i + 1 < len(selected):
            next_n = selected[i + 1]
            e_t = (float(ph_intervals[nuc_idx].maxTime) + float(ph_intervals[next_n].minTime)) * 0.5
        else:
            e_t = global_end

        phones = [
            p for p in ph_intervals
            if float(p.minTime) >= s_t - 1e-6 and float(p.maxTime) <= e_t + 1e-6
        ]
        if not phones:
            phones = [ph_intervals[nuc_idx]]

        tok = cv_targets[i] if i < len(cv_targets) else ""
        out.append({
            "word": tok,
            "roman": tok,
            "roman_cv": tok,
            "start_time": s_t,
            "end_time": e_t,
            "phones": phones,
        })

    return out


def _collect_kr_phone_tier_quality(phone_tier, expected_syllables, min_vowel_phone_ratio=0.5):
    """
    phones tier 품질을 계산합니다.
    - 비침묵 phone 수
    - spn 비율
    - 핵 모음 phone 수
    - 기대 음절 대비 phone 수 비율
    """
    silence_marks = {"", "sil", "pau", "sp"}
    phone_count_non_sil = 0
    spn_count = 0
    known_vowel_phone_count = 0
    for p in phone_tier or []:
        mark = str(getattr(p, "mark", "") or "").strip().lower()
        if mark in silence_marks:
            continue
        phone_count_non_sil += 1
        if mark == "spn":
            spn_count += 1
            continue
        if _is_kr_nucleus_phone_mark(mark):
            known_vowel_phone_count += 1

    expected = max(0, int(expected_syllables or 0))
    spn_ratio = float(spn_count) / float(max(1, phone_count_non_sil))
    phones_vs_expected = (
        float(phone_count_non_sil) / float(max(1, expected))
        if expected > 0 else 0.0
    )
    min_vowel_needed = max(2, int(round(expected * max(0.1, float(min_vowel_phone_ratio or 0.5)))))
    reasons = []
    if expected > 0 and phone_count_non_sil < expected:
        reasons.append("insufficient_phones")
    if expected > 0 and known_vowel_phone_count < min_vowel_needed:
        reasons.append("insufficient_vowel_phones")

    return {
        "phone_count_non_sil": int(phone_count_non_sil),
        "spn_count": int(spn_count),
        "spn_ratio_in_phone_tier": float(spn_ratio),
        "known_vowel_phone_count": int(known_vowel_phone_count),
        "phones_vs_expected_syllables_ratio": float(phones_vs_expected),
        "expected_syllables": int(expected),
        "low_confidence_reasons": reasons,
    }


def _estimate_kr_mapping_confidence(
    phone_quality,
    words_score=0.0,
    alias_score=0.0,
    used_words_based=True,
    used_alias_based=False,
):
    """
    한국어 음절 매핑 신뢰도를 0~1로 추정합니다.
    - phones/words 품질
    - words vs alias 점수 마진
    - 적용된 매핑 경로
    """
    pq = phone_quality or {}
    spn_ratio = float(pq.get("spn_ratio_in_phone_tier", 0.0) or 0.0)
    ratio_vs_expected = float(pq.get("phones_vs_expected_syllables_ratio", 0.0) or 0.0)
    known_vowel_count = float(pq.get("known_vowel_phone_count", 0.0) or 0.0)
    expected = float(max(1, int(pq.get("expected_syllables", 0) or 0)))
    low_reasons = set(pq.get("low_confidence_reasons", []) or [])

    score_words = float(words_score or 0.0)
    score_alias = float(alias_score or 0.0)
    margin = score_words - score_alias

    conf = 1.0
    conf -= min(spn_ratio * 0.55, 0.45)
    conf -= min(abs(1.0 - ratio_vs_expected) * 0.28, 0.28)
    conf += min((known_vowel_count / expected) * 0.18, 0.18)
    conf += min(max(score_words, score_alias) / 100.0 * 0.14, 0.14)
    if used_words_based and not used_alias_based:
        conf += 0.05
    if used_alias_based and not used_words_based:
        conf -= 0.04
    conf += max(-0.12, min(0.12, margin / 100.0))
    if "insufficient_phones" in low_reasons:
        conf -= 0.14
    if "insufficient_vowel_phones" in low_reasons:
        conf -= 0.10
    if "spn_heavy" in low_reasons:
        conf -= 0.20
    conf = max(0.0, min(1.0, conf))
    return float(conf), float(margin)


def _cleanup_timing_anchor_jsonl_files(log_dir, prefix):
    if not log_dir or not os.path.isdir(log_dir):
        return 0, 0
    removed = 0
    failed = 0
    try:
        names = os.listdir(log_dir)
    except Exception:
        return 0, 1
    for name in names:
        if not name.startswith(prefix) or not name.endswith(".jsonl"):
            continue
        path = os.path.join(log_dir, name)
        try:
            os.remove(path)
            removed += 1
        except Exception:
            failed += 1
    return removed, failed


def _synthesize_kr_word_phones(word, w_start, w_end, decompose_hangul_to_roman):
    """
    words 구간에 대응하는 phones가 비어 있을 때 최소 합성 phone을 생성합니다.
    """
    start = float(w_start)
    end = float(w_end)
    if end <= start:
        end = start + 0.03
    duration = max(0.03, end - start)

    roman_parts = []
    for ch in str(word or ""):
        roman_parts.extend(decompose_hangul_to_roman(ch))
    roman_raw = "".join(roman_parts).lower()
    token = _kr_cv_kernel(roman_raw) if roman_raw else ""
    onset, vowel, coda = _split_kr_syllable_parts(token)

    if not vowel and token in KR_VOWELS:
        vowel = token
    if not vowel:
        vowel = "a"
    vowel_ipa = {
        "a": "a",
        "i": "i",
        "u": "u",
        "e": "e",
        "o": "o",
        "eo": "ʌ",
        "eu": "ɯ",
        "ae": "ɛ",
        "oe": "ø",
        "wi": "wi",
        "wo": "wo",
        "wa": "wa",
        "we": "we",
        "weo": "wʌ",
    }.get(vowel, vowel)

    phones = []
    onset_end = start + duration * 0.35
    vowel_end = end - (duration * 0.2 if coda else 0.0)
    if onset:
        phones.append(SimpleNamespace(minTime=start, maxTime=max(start + 0.01, onset_end), mark=onset))
    phones.append(SimpleNamespace(minTime=max(start + 0.005, onset_end if onset else start), maxTime=max(start + 0.02, vowel_end), mark=vowel_ipa))
    if coda:
        phones.append(SimpleNamespace(minTime=max(start + 0.02, vowel_end), maxTime=end, mark=coda))
    return phones


def validate_oto_params(offset, consonant, cutoff, pre, ovl):
    """UTAU OTO 파라미터를 유효 범위로 보정합니다."""
    if offset < 0:
        offset = 0
    if pre < 0:
        pre = 0
    if ovl < 0:
        ovl = 0
    if consonant < 0:
        consonant = 0
    

    if ovl > pre:
        ovl = pre * 0.75
    

    if consonant < pre:
        consonant = pre + 30
    

    cutoff_abs = abs(cutoff)
    if cutoff_abs <= consonant:
        cutoff_abs = consonant + 50
    cutoff = -cutoff_abs
    
    return offset, consonant, cutoff, pre, ovl


def _nearest_phone_edge_ms(anchor_ms, ph_intervals):
    """anchor_ms에 가장 가까운 phone 경계(ms)를 반환합니다."""
    nearest = None
    nearest_dist = float("inf")
    for ph in ph_intervals or []:
        s = float(ph.minTime) * 1000.0
        e = float(ph.maxTime) * 1000.0
        if s <= anchor_ms <= e:
            return anchor_ms, 0.0
        ds = abs(anchor_ms - s)
        de = abs(anchor_ms - e)
        if ds < nearest_dist:
            nearest_dist = ds
            nearest = s
        if de < nearest_dist:
            nearest_dist = de
            nearest = e
    if nearest is None:
        return anchor_ms, 0.0
    return nearest, nearest_dist


def _surrounding_phone_gap(anchor_ms, ph_intervals):
    """
    anchor_ms가 phone 사이 gap에 있을 때 (prev_end, next_start, gap_len_ms) 반환.
    gap 내부가 아니면 (None, None, 0.0).
    """
    prev_end = None
    next_start = None
    for ph in ph_intervals or []:
        s = float(ph.minTime) * 1000.0
        e = float(ph.maxTime) * 1000.0
        if s <= anchor_ms <= e:
            return None, None, 0.0
        if e < anchor_ms:
            prev_end = e
        elif s > anchor_ms and next_start is None:
            next_start = s
            break
    if prev_end is None or next_start is None or next_start <= prev_end:
        return None, None, 0.0
    return prev_end, next_start, (next_start - prev_end)


def _stabilize_params_to_phone_activity(offset, consonant, cutoff, pre, ovl, ph_intervals, alias_type="cv"):
    """
    선행발성 기준점(pre 절대위치)이 무음 gap에 걸리면
    가장 가까운 유효 phone 경계로 스냅해 빈 공간 정렬을 완화합니다.
    """
    if not ph_intervals:
        return validate_oto_params(offset, consonant, cutoff, pre, ovl)

    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
    pre_abs = float(offset) + float(pre)
    nearest_edge, nearest_dist = _nearest_phone_edge_ms(pre_abs, ph_intervals)
    prev_end, next_start, gap_len = _surrounding_phone_gap(pre_abs, ph_intervals)

    # 충분히 넓은 gap 내부이거나 phone과 멀리 떨어진 경우만 보정
    if gap_len < 55.0 and nearest_dist <= 34.0:
        return offset, consonant, cutoff, pre, ovl

    target = nearest_edge
    if prev_end is not None and next_start is not None:
        if alias_type in {"vc", "vv"}:
            # VC/VV는 다음 음절 입구를 겨냥하되, 경계보다 약간 앞에서 잡는다.
            target = next_start - 6.0
            target = max(target, prev_end + 4.0)
        elif alias_type in {"cv", "cv_head"}:
            # CV 계열은 이전 음절 끝으로 끌어당기지 않고,
            # 현재 자음 onset 직전으로 스냅해 앞 발음 유입을 줄인다.
            target = next_start - 4.0
            target = max(target, prev_end + 3.0)
        else:
            # VCV 등은 이전 음절 끝 경계 근처로 당겨서 공백 정렬을 방지한다.
            target = prev_end

    delta = target - pre_abs
    if abs(delta) < 2.0:
        return offset, consonant, cutoff, pre, ovl

    offset = max(float(offset) + delta, 0.0)
    return validate_oto_params(offset, consonant, cutoff, pre, ovl)


def generate_openutau_aliases(base_alias):
    """OpenUtau 호환을 위해 동일 발음의 대체 에일리어스를 추가 생성합니다."""
    aliases = {base_alias}
    parts = base_alias.split()

    if len(parts) == 2:
        v, c = parts[0], parts[1]
        


        batchim_map = {
            'g': 'k', 'gg': 'k', 'k': 'k',
            'd': 't', 'dd': 't', 's': 't', 'ss': 't', 'j': 't', 'jj': 't', 'ch': 't', 't': 't', 'h': 't',
            'b': 'p', 'bb': 'p', 'p': 'p',
            'r': 'l', 'l': 'l',
            'n': 'n', 'm': 'm', 'ng': 'ng'
        }

        stop_variant_map = {
            'g': ['g', 'k', 'gg', 'kk'],
            'gg': ['gg', 'k', 'g', 'kk'],
            'k': ['k', 'g', 'gg', 'kk'],
            'kk': ['kk', 'k', 'g', 'gg'],
            'd': ['d', 't', 'dd', 'tt'],
            'dd': ['dd', 't', 'd', 'tt'],
            't': ['t', 'd', 'dd', 'tt'],
            'tt': ['tt', 't', 'd', 'dd'],
            'b': ['b', 'p', 'bb', 'pp'],
            'bb': ['bb', 'p', 'b', 'pp'],
            'p': ['p', 'b', 'bb', 'pp'],
            'pp': ['pp', 'p', 'b', 'bb'],
            'j': ['j', 'ch', 'jj'],
            'jj': ['jj', 'j', 'ch'],
            'ch': ['ch', 'j', 'jj'],
        }
        

        aliases.add(f"{v} {c}")
        aliases.add(f"{v}{c}")
        

        if c in ['a','e','i','o','u','eo','eu','ae','oe','wi','ya','yeo','yo','yu','ye','wa','wo','we','weo','eui','ui']:
            aliases.add(f"{v} {c}")
            aliases.add(f"{v}{c}")

            aliases.add(f"{v} -")
            aliases.add(f"{v}-")
        else:

            if c in batchim_map:
                mapped_c = batchim_map[c]

                aliases.add(f"{v} {mapped_c}")
                aliases.add(f"{v}{mapped_c}")
                aliases.add(f"{v} {mapped_c.upper()}")
                aliases.add(f"{v}{mapped_c.upper()}")
                

                aliases.add(f"{v} {mapped_c}-")
                aliases.add(f"{v}{mapped_c}-")
                aliases.add(f"{mapped_c}-")
                if c != mapped_c:
                    aliases.add(f"{v} {c}-")
                    aliases.add(f"{v}{c}-")
                    aliases.add(f"{c}-")

            c_lower = c.lower()
            if c_lower in stop_variant_map:
                for vc in stop_variant_map[c_lower]:
                    aliases.add(f"{v} {vc}")
                    aliases.add(f"{v}{vc}")
                    aliases.add(f"{v} {vc.upper()}")
                    aliases.add(f"{v}{vc.upper()}")

    elif len(parts) == 1:
        cv = parts[0]

        aliases.add(f"- {cv}")
        aliases.add(f"-{cv}")
        

        aliases.add(f"{cv}-")
        aliases.add(f"{cv} -")
        

        if cv == 'eui':
            aliases.add('eu i')
            aliases.update(['ui', '- ui', '-ui', 'ui -', 'ui-'])
        elif cv.startswith('-') and 'eui' in cv:
            aliases.add(cv.replace('eui', 'eu i'))

    return list(aliases)


def _parse_oto_line_profile(line):
    s = (line or "").strip()
    if not s or "=" not in s:
        return None
    wav_name, rest = s.split("=", 1)
    parts = rest.split(",")
    if len(parts) < 6:
        return None
    try:
        offset = float(parts[1].strip())
        cons = float(parts[2].strip())
        cutoff = float(parts[3].strip())
        pre = float(parts[4].strip())
        ovl = float(parts[5].strip())
    except ValueError:
        return None
    return {
        "wav": wav_name.strip(),
        "alias": parts[0].strip(),
        "offset": offset,
        "cons": cons,
        "cutoff": cutoff,
        "pre": pre,
        "ovl": ovl,
    }


def _extract_base_timing_shape(line):
    """
    base oto 한 줄에서 상대 타이밍 shape를 추출합니다.
    절대 시각 복사보다 pre/cons_gap/cut_gap/ovl_ratio 블렌딩에 사용.
    """
    p = _parse_oto_line_profile(line)
    if not p:
        return None
    pre = max(float(p["pre"]), 0.0)
    cons = max(float(p["cons"]), 0.0)
    cut_abs = abs(float(p["cutoff"]))
    ovl = max(float(p["ovl"]), 0.0)
    off = max(float(p["offset"]), 0.0)

    # 템플릿 없는 자동 생성(0,0,0,0,0) 라인은 shape로 사용하지 않음.
    if pre < 1.0 and cons < 1.0 and cut_abs < 1.0 and ovl < 1.0:
        return None

    cons_gap = max(cons - pre, 8.0)
    cut_gap = max(cut_abs - cons, 16.0)
    ovl_ratio = (ovl / pre) if pre > 1e-6 else 0.30
    return {
        "offset": off,
        "pre": pre,
        "cons_gap": cons_gap,
        "cut_gap": cut_gap,
        "ovl_ratio": _clamp(ovl_ratio, 0.04, 0.86),
    }


def _apply_base_shape_blend(offset, consonant, cutoff, pre, ovl, base_shape, alias_type="cv"):
    if os.environ.get("UTOA_DISABLE_BASE_SHAPE_BLEND", "").strip().lower() in {"1", "true", "yes", "on"}:
        return validate_oto_params(offset, consonant, cutoff, pre, ovl)
    if not base_shape:
        return validate_oto_params(offset, consonant, cutoff, pre, ovl)
    if alias_type == "cv_head":
        # 템플릿의 head 라인이 비정상적으로 짧은 경우(cut_gap 과소),
        # 그대로 블렌딩하면 -CV 길이가 급격히 줄어들 수 있어 보수적으로 생략한다.
        src_cut_gap = float(base_shape.get("cut_gap", max(abs(cutoff) - consonant, 20.0)))
        src_pre = float(base_shape.get("pre", pre))
        if src_cut_gap < 90.0 or src_pre > 280.0:
            return validate_oto_params(offset, consonant, cutoff, pre, ovl)

    if alias_type == "vc":
        w = 0.16
    elif alias_type == "vv":
        w = 0.16
    elif alias_type == "vcv":
        w = 0.34
    elif alias_type == "cv_head":
        w = 0.22
    else:
        w = 0.24

    pre_t = _clamp(base_shape.get("pre", pre), 12.0, 420.0)
    cons_gap_t = _clamp(base_shape.get("cons_gap", max(consonant - pre, 10.0)), 8.0, 260.0)
    cut_gap_t = _clamp(base_shape.get("cut_gap", max(abs(cutoff) - consonant, 20.0)), 16.0, 300.0)
    ovl_ratio_t = _clamp(base_shape.get("ovl_ratio", (ovl / pre) if pre > 0 else 0.30), 0.04, 0.86)

    pre_new = _blend(pre, pre_t, w)
    cons_gap_now = max(consonant - pre, 10.0)
    cons_gap_new = _blend(cons_gap_now, cons_gap_t, min(0.42, w + 0.07))
    cons_new = pre_new + cons_gap_new

    cut_gap_now = max(abs(cutoff) - consonant, 20.0)
    cut_gap_new = _blend(cut_gap_now, cut_gap_t, min(0.38, w + 0.03))
    if alias_type == "vc":
        cut_gap_new = min(cut_gap_new, cut_gap_now)
    cutoff_new = -(cons_new + cut_gap_new)

    ovl_ratio_now = (ovl / pre) if pre > 1e-6 else 0.30
    ovl_ratio_new = _blend(ovl_ratio_now, ovl_ratio_t, min(0.38, w + 0.04))
    ovl_new = max(0.0, pre_new * _clamp(ovl_ratio_new, 0.04, 0.86))

    return validate_oto_params(offset, cons_new, cutoff_new, pre_new, ovl_new)


def _wav_duration_ms(wav_path):
    try:
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
        if sr <= 0:
            return 0.0
        return (n / float(sr)) * 1000.0
    except Exception:
        return 0.0


def _read_wav_mono_np(wav_path):
    if np is None:
        return None, None
    try:
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception:
        return None, None

    if sw == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sw == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return None, None

    if n_ch > 1:
        audio = audio.reshape(-1, n_ch).mean(axis=1)
    return audio, sr


def _estimate_f0_voicing_strength(frame, sr):
    """
    간단한 자기상관 기반 유성도(0~1) 추정.
    정확한 F0 추적이 아니라 보정 보조 지표로만 사용합니다.
    """
    if np is None or frame is None or len(frame) < 96 or sr <= 0:
        return 0.0
    x = frame.astype(np.float64)
    x = x - np.mean(x)
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    if rms < 1e-4:
        return 0.0

    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    if len(ac) < 8:
        return 0.0
    ac0 = float(ac[0])
    if ac0 <= 1e-9:
        return 0.0

    min_lag = max(2, int(sr / 500.0))
    max_lag = min(len(ac) - 1, int(sr / 70.0))
    if max_lag <= min_lag + 1:
        return 0.0

    search = ac[min_lag:max_lag + 1]
    peak_rel = int(np.argmax(search))
    peak = float(search[peak_rel])
    lag = min_lag + peak_rel
    if lag <= 0:
        return 0.0
    f0 = float(sr) / float(lag)
    if f0 < 70.0 or f0 > 500.0:
        return 0.0

    clarity = peak / (ac0 + 1e-9)
    if clarity < 0.25:
        return 0.0
    return float(np.clip((clarity - 0.25) / 0.45, 0.0, 1.0))


def _mel_envelope(audio, sr):
    if np is None or audio is None or sr is None or len(audio) == 0:
        return None
    n_fft = 1024
    hop = max(1, int(sr * 0.005))
    win = min(n_fft, max(256, int(sr * 0.025)))
    window = np.hanning(win).astype(np.float64)

    f_min = 0.0
    f_max = sr / 2.0
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_points = np.linspace(mel_min, mel_max, 42)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((40, n_fft // 2 + 1), dtype=np.float64)
    for i in range(1, 41):
        left = bins[i - 1]
        center = bins[i]
        right = bins[i + 1]
        if right <= left:
            continue
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for j in range(left, center):
            fb[i - 1, j] = (j - left) / float(center - left)
        for j in range(center, right):
            fb[i - 1, j] = (right - j) / float(right - center)

    frames = []
    db_vals = []
    f0_voicing = []
    times = []
    last_voicing = 0.0
    frame_idx = 0
    for st in range(0, max(len(audio) - win + 1, 1), hop):
        fr_raw = audio[st:st + win]
        if len(fr_raw) < win:
            fr_raw = np.concatenate([fr_raw, np.zeros(win - len(fr_raw), dtype=np.float32)], axis=0)

        rms = float(np.sqrt(np.mean(fr_raw.astype(np.float64) ** 2) + 1e-12))
        db_vals.append(20.0 * np.log10(max(rms, 1e-7)))

        # F0는 저가중치 보조 신호만 제공하도록 저밀도 계산.
        if (frame_idx % 3) == 0:
            last_voicing = _estimate_f0_voicing_strength(fr_raw, sr)
        f0_voicing.append(last_voicing)

        fr = fr_raw.astype(np.float64) * window
        if n_fft > win:
            fr = np.pad(fr, (0, n_fft - win))
        spec = np.fft.rfft(fr)
        power = (spec.real ** 2 + spec.imag ** 2)
        mel = fb @ power
        frames.append(np.log1p(np.maximum(mel, 0.0)).mean())
        times.append(st * 1000.0 / sr)
        frame_idx += 1

    if not frames:
        return None

    e = np.array(frames, dtype=np.float64)
    p10 = float(np.percentile(e, 10))
    p90 = float(np.percentile(e, 90))
    span = max(p90 - p10, 1e-6)
    en = (e - p10) / span
    en = np.clip(en, 0.0, 1.0)
    db_arr = np.array(db_vals, dtype=np.float64) if db_vals else np.zeros_like(en)
    f0v_arr = np.array(f0_voicing, dtype=np.float64) if f0_voicing else np.zeros_like(en)
    if len(f0v_arr) >= 3:
        f0v_arr = np.convolve(f0v_arr, np.array([0.2, 0.6, 0.2], dtype=np.float64), mode="same")
    f0v_arr = np.clip(f0v_arr, 0.0, 1.0)

    db_p20 = float(np.percentile(db_arr, 20)) if len(db_arr) else -60.0
    db_sil_th = max(-58.0, min(-28.0, db_p20 + 6.0))
    return {
        "times_ms": np.array(times, dtype=np.float64),
        "energy": en,
        "span": span,
        "db_db": db_arr,
        "db_silence_th": float(db_sil_th),
        "f0_voicing": f0v_arr,
    }


def _find_wav_path_for_name(wav_name, wav_dir, wav_index):
    cands = [
        os.path.join(wav_dir, wav_name),
        os.path.join(wav_dir, os.path.basename(wav_name)),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    key = normalize_key(wav_name)
    return wav_index.get(key, "")


def _nearest_time_index(times_ms, t_ms):
    if np is None or times_ms is None or len(times_ms) == 0:
        return -1
    idx = int(np.searchsorted(times_ms, t_ms))
    if idx <= 0:
        return 0
    if idx >= len(times_ms):
        return len(times_ms) - 1
    prev_i = idx - 1
    if abs(float(times_ms[idx]) - t_ms) < abs(float(times_ms[prev_i]) - t_ms):
        return idx
    return prev_i


def _apply_soft_mel_offset_cutoff_guard(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    alias_type,
    mel_ctx=None,
    onset_hint="",
    alias_text="",
    file_format="",
):
    """
    이른 단계 soft guard:
    - dB 최소 임계값 + mel 에너지로 무음/유음 전이를 감지
    - F0 유성도는 낮은 가중치(보조)로만 반영
    """
    if np is None or not mel_ctx:
        return offset, consonant, cutoff, pre, ovl, 0.0, 0.0
    if alias_type not in {"cv", "cv_head", "vcv"}:
        return offset, consonant, cutoff, pre, ovl, 0.0, 0.0

    t_ms = mel_ctx.get("times_ms")
    en = mel_ctx.get("energy")
    db_arr = mel_ctx.get("db_db")
    f0v_arr = mel_ctx.get("f0_voicing")
    db_sil_th = float(mel_ctx.get("db_silence_th", -42.0))
    if t_ms is None or en is None or len(t_ms) < 8 or len(en) != len(t_ms):
        return offset, consonant, cutoff, pre, ovl, 0.0, 0.0
    if db_arr is None or len(db_arr) != len(en):
        db_arr = np.zeros_like(en, dtype=np.float64)
    if f0v_arr is None or len(f0v_arr) != len(en):
        f0v_arr = np.zeros_like(en, dtype=np.float64)

    pre_abs = float(offset) + float(pre)
    cons_abs = float(offset) + float(consonant)
    cut_abs = float(offset) + abs(float(cutoff))
    if cut_abs <= pre_abs + 18.0:
        return offset, consonant, cutoff, pre, ovl, 0.0, 0.0

    sound_mask = (db_arr > (db_sil_th + 1.5)) & (en > 0.14)
    silence_mask = (db_arr <= db_sil_th) | (en <= 0.10)

    off_idx = _nearest_time_index(t_ms, offset)
    pre_idx = _nearest_time_index(t_ms, pre_abs)
    cut_idx = _nearest_time_index(t_ms, cut_abs)
    if min(off_idx, pre_idx, cut_idx) < 0:
        return offset, consonant, cutoff, pre, ovl, 0.0, 0.0

    offset_shift_ms = 0.0
    cutoff_shift_ms = 0.0
    hint = (onset_hint or "").strip().lower()
    if hint in {"ɯ", "a", "i", "u", "e", "o"}:
        hint = ""
    if not hint and alias_text:
        parts = [p.strip().lower() for p in str(alias_text).split() if p.strip()]
        token = ""
        if parts:
            if alias_type in {"cv_head", "vcv"} and len(parts) >= 2:
                token = parts[1]
            else:
                token = parts[-1]
        token = re.sub(r"[^a-z]", "", token)
        if token:
            m = re.match(r"^([bcdfghjklmnpqrstvwxyz]+)", token)
            if m:
                hint = m.group(1)
    # 유성/비음 계열(m,n,r,l,w,y...)은 멜 저역 에너지가 약해
    # offset guard가 모음 시작으로 과도 이동할 수 있어 보수적으로 처리한다.
    low_energy_voiced = hint in {
        "m", "n", "ny", "ng", "r", "l", "ry", "w", "y", "j",
        "g", "d", "b", "z", "dz", "v", "gy", "dy", "by",
        "ɴ", "ŋ", "ɲ", "ɾ", "ɹ",
    } or hint.startswith("m")

    # ---- soft offset guard ----
    # 한국어 CVVC의 CV/CV_HEAD는 onset anchor와 후단 guard만으로도 충분한 경우가 많다.
    # 멜 offset soft guard가 추가로 들어가면 단모음/활음 구분이 약한 파일에서
    # offset이 공백 쪽으로 과하게 끌리는 경향이 커진다.
    skip_offset_soft_guard = (alias_type == "cv_head") or (
        str(file_format).strip().lower() == "cvvc" and alias_type in {"cv", "cv_head"}
    )
    if not skip_offset_soft_guard and not low_energy_voiced:
        off_silent = bool(silence_mask[off_idx])
        pre_sound = bool(sound_mask[pre_idx] or (en[pre_idx] > 0.20))
        if off_silent and pre_sound:
            lo = max(0, pre_idx - 120)
            seg = sound_mask[lo:pre_idx + 1]
            if np.any(seg):
                rel = int(np.where(seg)[0][0])
                sound_start_idx = lo + rel
                target_offset = float(t_ms[sound_start_idx]) - 12.0
                target_offset = max(0.0, min(pre_abs - 18.0, target_offset))
                new_offset = _blend(offset, target_offset, 0.36)
                offset_shift_ms = float(new_offset - offset)
                offset = new_offset
                pre = max(pre_abs - offset, 0.0)
                consonant = max(cons_abs - offset, pre + 8.0)

    # ---- soft cutoff guard ----
    cut_idx = _nearest_time_index(t_ms, cut_abs)
    cut_sound = bool(sound_mask[cut_idx] or (en[cut_idx] > 0.22))
    if cut_sound:
        start_idx = _nearest_time_index(t_ms, pre_abs + 16.0)
        if start_idx < 0:
            start_idx = 0
        if start_idx < cut_idx:
            seg = np.where(silence_mask[start_idx:cut_idx + 1])[0]
            if len(seg) > 0:
                last_sil_idx = int(start_idx + seg[-1])
                target_cut_abs = float(t_ms[last_sil_idx]) + 4.0
                target_cut_abs = max(pre_abs + 20.0, min(target_cut_abs, cut_abs))
                if target_cut_abs < cut_abs - 8.0:
                    # F0 유성도는 보조(저가중치)로만 반영
                    f0v = float(f0v_arr[last_sil_idx])
                    blend_w = 0.42 - (0.08 * f0v)
                    blend_w = max(0.26, min(0.44, blend_w))
                    new_cut_abs = _blend(cut_abs, target_cut_abs, blend_w)
                    cutoff_shift_ms = float(cut_abs - new_cut_abs)
                    cut_abs = new_cut_abs
                    cutoff = -(cut_abs - offset)
                    consonant = min(consonant, (cut_abs - offset) - 10.0)
                    consonant = max(consonant, pre + 8.0)

    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
    return offset, consonant, cutoff, pre, ovl, offset_shift_ms, cutoff_shift_ms


def _apply_kr_mel_refine_to_oto_file(oto_path, wav_dir, custom_map=None):
    """
    멜 스펙트로그램 에너지 골짜기를 이용해 CV 계열 cutoff 과연장을 후처리 보정합니다.
    """
    if os.environ.get("UTOA_DISABLE_MEL_REFINER", "").strip().lower() in {"1", "true", "yes", "on"}:
        return 0
    if np is None or not oto_path or not os.path.exists(oto_path) or not os.path.isdir(wav_dir):
        return 0

    raw_lines = []
    parsed = []
    with open(oto_path, "r", encoding="utf-8", errors="replace") as f:
        for idx, raw in enumerate(f):
            line = raw.rstrip("\n")
            raw_lines.append(line)
            row = _parse_oto_line_profile(line)
            if row:
                parsed.append((idx, row))
    if not parsed:
        return 0

    wav_index = {}
    try:
        for fn in os.listdir(wav_dir):
            if fn.lower().endswith(".wav"):
                wav_index[normalize_key(fn)] = os.path.join(wav_dir, fn)
    except Exception:
        pass

    by_wav = {}
    for line_idx, row in parsed:
        by_wav.setdefault(row["wav"], []).append((line_idx, row))
    alias_type_cache = {}

    def _classify_cached(alias_text):
        t = alias_type_cache.get(alias_text)
        if t is None:
            t = classify_alias(alias_text, custom_map)
            alias_type_cache[alias_text] = t
        return t

    mel_cache = {}
    changed = 0
    for wav_name, rows in by_wav.items():
        wav_path = _find_wav_path_for_name(wav_name, wav_dir, wav_index)
        if not wav_path:
            continue
        mel_ctx = mel_cache.get(wav_path)
        if mel_ctx is None:
            audio, sr = _read_wav_mono_np(wav_path)
            mel_ctx = _mel_envelope(audio, sr)
            mel_cache[wav_path] = mel_ctx
        if not mel_ctx:
            continue

        t_ms = mel_ctx["times_ms"]
        en = mel_ctx["energy"]
        db_arr = mel_ctx.get("db_db")
        f0v_arr = mel_ctx.get("f0_voicing")
        db_sil_th = float(mel_ctx.get("db_silence_th", -42.0))
        if db_arr is None or len(db_arr) != len(en):
            db_arr = np.zeros_like(en, dtype=np.float64)
        if f0v_arr is None or len(f0v_arr) != len(en):
            f0v_arr = np.zeros_like(en, dtype=np.float64)
        if len(t_ms) < 8:
            continue

        for i, (_line_idx, row) in enumerate(rows):
            alias = row["alias"]
            alias_type = _classify_cached(alias)
            if alias_type not in {"cv", "cv_head"}:
                continue

            off = float(row["offset"])
            pre_abs = off + float(row["pre"])
            cons_abs = off + float(row["cons"])
            cut_abs = off + abs(float(row["cutoff"]))
            if cut_abs <= pre_abs + 24.0:
                continue

            next_anchor = None
            for j in range(i + 1, len(rows)):
                a2 = rows[j][1]["alias"]
                t2 = _classify_cached(a2)
                if t2 in {"cv", "cv_head", "vcv", "mono"}:
                    next_anchor = float(rows[j][1]["offset"]) + float(rows[j][1]["pre"])
                    break

            search_start = pre_abs + 14.0
            search_end = cut_abs - 8.0
            if next_anchor is not None:
                search_end = min(search_end, next_anchor - 8.0)
            if search_end <= search_start + 25.0:
                continue

            mask = np.where((t_ms >= search_start) & (t_ms <= search_end))[0]
            if len(mask) < 5:
                continue

            local_e = en[mask]
            local_db = db_arr[mask]
            local_f0v = f0v_arr[mask]
            silence_flags = local_db <= db_sil_th
            candidate_mask = mask[silence_flags] if np.any(silence_flags) else mask

            # dB+mel을 주축으로 골짜기 선택, F0는 낮은 가중치(보조)로만 반영.
            best_idx = int(candidate_mask[0])
            best_score = -1e9
            for ci in candidate_mask:
                e_v = float(en[ci])
                db_v = float(db_arr[ci])
                f0_v = float(f0v_arr[ci])
                silence_bonus = 0.28 if db_v <= db_sil_th else 0.0
                score = (1.0 - e_v) + silence_bonus - (0.08 * f0_v)
                if score > best_score:
                    best_score = score
                    best_idx = int(ci)

            valley_t = float(t_ms[best_idx])
            valley_e = float(en[best_idx])
            valley_db = float(db_arr[best_idx])
            valley_f0v = float(f0v_arr[best_idx])
            cut_idx = int(np.argmin(np.abs(t_ms - cut_abs)))
            cut_e = float(en[cut_idx])
            cut_db = float(db_arr[cut_idx])
            contrast = cut_e - valley_e
            db_drop = cut_db - valley_db

            if contrast < 0.12 and db_drop < 2.5:
                continue
            if valley_e > 0.38 and valley_db > (db_sil_th + 6.0):
                continue
            if valley_f0v > 0.70 and contrast < 0.18:
                continue
            if valley_t >= cut_abs - 12.0:
                continue

            target_cut_abs = valley_t + 2.0
            min_cut_abs = pre_abs + 20.0
            if target_cut_abs <= min_cut_abs:
                continue

            new_cons_abs = min(cons_abs, target_cut_abs - 12.0)
            new_cons_abs = max(new_cons_abs, pre_abs + 8.0)
            row["cons"] = max(new_cons_abs - off, 0.0)
            row["cutoff"] = -(target_cut_abs - off)
            changed += 1

    if changed <= 0:
        return 0

    replace_map = {line_idx: row for line_idx, row in parsed}
    out_lines = []
    for i, line in enumerate(raw_lines):
        row = replace_map.get(i)
        if not row:
            out_lines.append(line)
            continue
        o, c, ct, p, ov = validate_oto_params(
            row["offset"], row["cons"], row["cutoff"], row["pre"], row["ovl"]
        )
        out_lines.append(f"{row['wav']}={row['alias']},{o:.2f},{c:.2f},{ct:.2f},{p:.2f},{ov:.2f}")

    with open(oto_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")
    return changed


def _median(vals):
    if not vals:
        return 0.0
    s = sorted(float(v) for v in vals)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _safe_ratio(num, den, fallback=0.0):
    if den is None or den == 0:
        return fallback
    return float(num) / float(den)


def _blend(a, b, w):
    w2 = max(0.0, min(1.0, float(w)))
    return (1.0 - w2) * float(a) + w2 * float(b)


def _clamp(v, lo, hi):
    return max(float(lo), min(float(hi), float(v)))


def _default_kr_profile_cache_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    profile_dir = os.path.join(base_dir, "assets", "profiles")
    os.makedirs(profile_dir, exist_ok=True)
    return os.path.join(profile_dir, "kr_oto_reference_profile.json")


def _build_kr_reference_profile_from_dirs(ref_dirs, custom_map=None):
    bucket_values = {}
    total_rows = 0
    total_wavs = 0
    alias_type_cache = {}

    def _classify_cached(alias_text):
        t = alias_type_cache.get(alias_text)
        if t is None:
            t = classify_alias(alias_text, custom_map)
            alias_type_cache[alias_text] = t
        return t

    for ref_dir in ref_dirs:
        if not ref_dir or not os.path.isdir(ref_dir):
            continue
        oto_path = os.path.join(ref_dir, "oto.ini")
        if not os.path.exists(oto_path):
            continue

        rows_by_wav = {}
        with open(oto_path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                row = _parse_oto_line_profile(raw)
                if not row:
                    continue
                rows_by_wav.setdefault(row["wav"], []).append(row)

        for wav_name, rows in rows_by_wav.items():
            wav_path = os.path.join(ref_dir, wav_name)
            dur_ms = _wav_duration_ms(wav_path)
            if dur_ms <= 0:
                continue
            total_wavs += 1
            for idx, row in enumerate(rows):
                alias_type = _classify_cached(row["alias"])
                b = bucket_values.setdefault(alias_type, {
                    "pre": [],
                    "cons_gap": [],
                    "cut_gap": [],
                    "ovl_ratio": [],
                    "head_off_ratio": [],
                })
                pre = max(row["pre"], 0.0)
                cons = max(row["cons"], pre)
                cut_abs = abs(row["cutoff"])
                b["pre"].append(_clamp(pre, 0.0, 360.0))
                b["cons_gap"].append(_clamp(max(cons - pre, 10.0), 10.0, 220.0))
                b["cut_gap"].append(_clamp(max(cut_abs - cons, 20.0), 20.0, 180.0))
                b["ovl_ratio"].append(_clamp(_safe_ratio(row["ovl"], pre, fallback=0.0), 0.05, 0.78))
                if idx == 0:
                    b["head_off_ratio"].append(_clamp(_safe_ratio(row["offset"], dur_ms, fallback=0.0), 0.0, 0.35))
                total_rows += 1

    if total_rows < 100:
        return None

    buckets = {}
    for alias_type, vals in bucket_values.items():
        n = len(vals["pre"])
        if n < 8:
            continue
        buckets[alias_type] = {
            "n": n,
            "pre": _median(vals["pre"]),
            "cons_gap": _median(vals["cons_gap"]),
            "cut_gap": _median(vals["cut_gap"]),
            "ovl_ratio": _median(vals["ovl_ratio"]),
            "head_off_ratio": _median(vals["head_off_ratio"]) if vals["head_off_ratio"] else None,
        }

    if not buckets:
        return None

    return {
        "version": 1,
        "language": "korean",
        "source_count": len([d for d in ref_dirs if d and os.path.isdir(d)]),
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "rows": total_rows,
        "wavs": total_wavs,
        "buckets": buckets,
    }


def _load_kr_reference_profile(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict) or "buckets" not in obj:
            return None
        return obj
    except Exception:
        return None


def _save_kr_reference_profile(path, profile):
    if not path or not profile:
        return False
    try:
        data = dict(profile)
        # Ensure profile file stores abstract stats only.
        data.pop("source_dirs", None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def _normalize_alias_for_profile(alias):
    a = re.sub(r"\s+", " ", (alias or "").strip().lower())
    # Treat trailing suffix like _C4 as metadata.
    a = re.sub(r"_[a-z0-9]{1,8}$", "", a)
    return a


def _read_text_with_fallback(path):
    try:
        raw = open(path, "rb").read()
    except Exception:
        return ""
    for enc in ("utf-8-sig", "cp932", "utf-8", "euc-kr", "latin-1"):
        try:
            text = raw.decode(enc)
            if "=" in text:
                return text
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def _read_kr_oto_rows_for_profile(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    text = _read_text_with_fallback(path)
    for raw in text.splitlines():
        row = _parse_oto_line_profile(raw)
        if not row:
            continue
        row["wav_norm"] = row["wav"].strip().lower()
        row["alias_norm"] = _normalize_alias_for_profile(row["alias"])
        rows.append(row)
    return rows


def _occurrence_map(rows):
    m = {}
    counters = {}
    for row in rows:
        base_key = (row["wav_norm"], row["alias_norm"])
        idx = counters.get(base_key, 0)
        counters[base_key] = idx + 1
        m[(base_key[0], base_key[1], idx)] = row
    return m


def train_kr_autotune_profile(auto_oto_path, manual_oto_path, custom_phonemes_path=""):
    """부분 수동 OTO와 자동 OTO를 매칭해 한국어 델타 기반 프로파일을 학습합니다."""
    auto_rows = _read_kr_oto_rows_for_profile(auto_oto_path)
    manual_rows = _read_kr_oto_rows_for_profile(manual_oto_path)
    if not auto_rows or not manual_rows:
        return None

    custom_map = load_custom_phonemes(custom_phonemes_path)
    alias_type_cache = {}

    def _classify_alias_cached(alias_text):
        t = alias_type_cache.get(alias_text)
        if t is None:
            t = classify_alias(alias_text, custom_map)
            alias_type_cache[alias_text] = t
        return t
    auto_map = _occurrence_map(auto_rows)
    manual_map = _occurrence_map(manual_rows)

    fields = ["offset", "cons", "cutoff", "pre", "ovl"]
    buckets = {}
    matched = 0
    for key, a_row in auto_map.items():
        m_row = manual_map.get(key)
        if not m_row:
            continue
        alias_for_cls = re.sub(r"_[A-Za-z0-9]{1,8}$", "", a_row["alias"])
        alias_type = _classify_alias_cached(alias_for_cls)
        b = buckets.setdefault(alias_type, {f: [] for f in fields})
        for f in fields:
            b[f].append(float(m_row[f]) - float(a_row[f]))
        matched += 1

    if matched < 8:
        return None

    profile = {
        "version": 1,
        "language": "korean",
        "mode": "delta",
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "matched_pairs": matched,
        "source_auto_name": os.path.basename(auto_oto_path),
        "source_manual_name": os.path.basename(manual_oto_path),
        "buckets": {},
    }
    for alias_type, vals in buckets.items():
        n = len(vals["offset"])
        if n < 3:
            continue
        profile["buckets"][alias_type] = {
            "n": n,
            "offset": _median(vals["offset"]),
            "cons": _median(vals["cons"]),
            "cutoff": _median(vals["cutoff"]),
            "pre": _median(vals["pre"]),
            "ovl": _median(vals["ovl"]),
        }
    if not profile["buckets"]:
        return None
    return profile


def save_kr_autotune_profile(path, profile):
    if not path or not profile:
        return False
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def load_kr_autotune_profile(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict) or "buckets" not in obj:
            return None
        return obj
    except Exception:
        return None


def apply_kr_autotune_profile_to_oto(oto_path, profile, custom_phonemes_path=""):
    """학습된 델타 프로파일을 OTO에 적용합니다. 반환값은 변경 라인 수."""
    if not oto_path or not os.path.exists(oto_path):
        return 0
    if isinstance(profile, str):
        profile = load_kr_autotune_profile(profile)
    if not profile:
        return 0
    buckets = profile.get("buckets") or {}
    if not isinstance(buckets, dict) or not buckets:
        return 0

    custom_map = load_custom_phonemes(custom_phonemes_path)
    changed = 0
    out_lines = []
    alias_type_cache = {}

    def _classify_cached(alias_text):
        t = alias_type_cache.get(alias_text)
        if t is None:
            t = classify_alias(alias_text, custom_map)
            alias_type_cache[alias_text] = t
        return t

    with open(oto_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            row = _parse_oto_line_profile(raw)
            if not row:
                out_lines.append(raw.rstrip("\n"))
                continue

            alias_for_cls = re.sub(r"_[A-Za-z0-9]{1,8}$", "", row["alias"])
            alias_type = _classify_cached(alias_for_cls)
            stat = buckets.get(alias_type)
            if not stat:
                out_lines.append(raw.rstrip("\n"))
                continue

            n = float(stat.get("n", 0))
            if n < 3:
                out_lines.append(raw.rstrip("\n"))
                continue

            w = min(1.0, max(0.25, n / 24.0))
            offset = row["offset"] + _clamp(stat.get("offset", 0.0), -140.0, 140.0) * w
            cons = row["cons"] + _clamp(stat.get("cons", 0.0), -160.0, 160.0) * w
            cutoff = row["cutoff"] + _clamp(stat.get("cutoff", 0.0), -180.0, 180.0) * w
            pre = row["pre"] + _clamp(stat.get("pre", 0.0), -140.0, 140.0) * w
            ovl = row["ovl"] + _clamp(stat.get("ovl", 0.0), -120.0, 120.0) * w
            offset, cons, cutoff, pre, ovl = validate_oto_params(offset, cons, cutoff, pre, ovl)

            if (
                abs(offset - row["offset"]) > 1e-6
                or abs(cons - row["cons"]) > 1e-6
                or abs(cutoff - row["cutoff"]) > 1e-6
                or abs(pre - row["pre"]) > 1e-6
                or abs(ovl - row["ovl"]) > 1e-6
            ):
                changed += 1
            out_lines.append(
                f"{row['wav']}={row['alias']},{offset:.2f},{cons:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
            )

    with open(oto_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")
    return changed


def _resolve_kr_reference_dirs():
    raw = os.environ.get("UTOA_KR_PROFILE_DIRS", "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(";") if x.strip()]


def _profile_weight(n):
    # Keep this conservative; profile should nudge, not overwrite.
    return max(0.12, min(0.38, float(n) / 220.0))


def _apply_kr_profile_to_oto_file(oto_path, wav_dir, profile, custom_map=None):
    if not profile or not oto_path or not os.path.exists(oto_path):
        return 0
    buckets = profile.get("buckets") or {}
    if not isinstance(buckets, dict) or not buckets:
        return 0

    rows_by_wav = {}
    with open(oto_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            row = _parse_oto_line_profile(raw)
            if not row:
                continue
            rows_by_wav.setdefault(row["wav"], []).append(row)

    out_lines = []
    changed = 0
    alias_type_cache = {}

    def _classify_cached(alias_text):
        t = alias_type_cache.get(alias_text)
        if t is None:
            t = classify_alias(alias_text, custom_map)
            alias_type_cache[alias_text] = t
        return t

    for wav_name, rows in rows_by_wav.items():
        dur_ms = _wav_duration_ms(os.path.join(wav_dir, wav_name))
        for idx, row in enumerate(rows):
            alias_type = _classify_cached(row["alias"])
            stat = buckets.get(alias_type)
            if not stat:
                out_lines.append(
                    f"{row['wav']}={row['alias']},{row['offset']:.2f},{row['cons']:.2f},{row['cutoff']:.2f},{row['pre']:.2f},{row['ovl']:.2f}"
                )
                continue

            n = float(stat.get("n", 0))
            w = _profile_weight(n)
            offset = row["offset"]
            pre_t = _clamp(stat.get("pre", row["pre"]), 0.0, 360.0)
            pre = _blend(row["pre"], pre_t, w)
            cons_gap_now = max(row["cons"] - row["pre"], 10.0)
            cons_gap_t = _clamp(stat.get("cons_gap", cons_gap_now), 10.0, 220.0)
            cons_gap = _blend(cons_gap_now, cons_gap_t, w)
            cons = pre + cons_gap
            cut_gap_now = max(abs(row["cutoff"]) - row["cons"], 20.0)
            cut_gap_t = _clamp(stat.get("cut_gap", cut_gap_now), 20.0, 180.0)
            cut_w = min(0.12, w * 0.35)
            cut_gap = _blend(cut_gap_now, cut_gap_t, cut_w)
            cutoff = -(cons + cut_gap)
            ovl_ratio_now = _safe_ratio(row["ovl"], row["pre"], fallback=0.0)
            ovl_ratio_t = _clamp(stat.get("ovl_ratio", ovl_ratio_now), 0.05, 0.78)
            ovl_ratio = _blend(ovl_ratio_now, ovl_ratio_t, w)
            ovl = max(0.0, pre * max(0.10, min(0.80, ovl_ratio)))
            pre, cons, cutoff, ovl = _apply_kr_consonant_timing_shaping(
                row.get("alias", ""), pre, cons, cutoff, ovl
            )

            if idx == 0 and dur_ms > 0 and stat.get("head_off_ratio") is not None:
                target_ratio = _clamp(stat["head_off_ratio"], 0.0, 0.35)
                target_off = max(0.0, min(dur_ms * 0.7, dur_ms * target_ratio))
                offset = _blend(offset, target_off, min(0.30, w))

            offset, cons, cutoff, pre, ovl = validate_oto_params(offset, cons, cutoff, pre, ovl)
            if (
                abs(offset - row["offset"]) > 1e-6
                or abs(cons - row["cons"]) > 1e-6
                or abs(cutoff - row["cutoff"]) > 1e-6
                or abs(pre - row["pre"]) > 1e-6
                or abs(ovl - row["ovl"]) > 1e-6
            ):
                changed += 1
            out_lines.append(
                f"{row['wav']}={row['alias']},{offset:.2f},{cons:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
            )

    with open(oto_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")
    return changed


def generate_oto(
    tg_folder,
    tpl_path,
    out_path,
    params=None,
    generate_openutau=False,
    gen_missing_vowels=False,
    enable_ml_correction=True,
    fallback_format='cvvc',
    custom_phonemes_path='',
    alias_suffix='',
    kr_mapping_words_fallback_enabled=True,
    kr_mapping_spn_ratio_threshold=0.35,
    kr_mapping_min_vowel_phone_ratio=0.5,
    kr_mapping_debug_reason_logging=True,
    kr_anchor_profile_path="",
    kr_mapping_confidence_threshold=None,
    kr_mapping_max_index_jump_default=1,
    kr_mapping_max_index_jump_high_conf=2,
    cleanup_timing_jsonl=True,
    auto_format=None,
    callback=None
):
    """TextGrid와 템플릿/자동 포맷 정보를 사용해 최종 OTO를 생성합니다."""

    try:
        import textgrid
    except ImportError:
        err = "textgrid 모듈이 설치되어 있지 않습니다. `pip install textgrid`를 실행해 주세요."
        logger.error(err)
        if callback:
            callback(err)
        return 0, 0, [err]

    if params is None:
        params = DEFAULT_PARAMS.copy()


    if auto_format:
        af = str(auto_format).strip().lower()
        if af.startswith('cvc'):
            fallback_format = 'cvc'
        elif af.startswith('cvvc'):
            fallback_format = 'cvvc'
        elif af.startswith('vcv'):
            fallback_format = 'vcv'

    auto_gen_format = (fallback_format or "cvvc").strip().lower()
    if auto_gen_format not in {"cvvc", "vcv"}:
        msg = (
            f"⚠ 자동 에일리어스 생성은 현재 CVVC/VCV만 지원합니다. "
            f"{auto_gen_format.upper()} -> CVVC로 전환합니다."
        )
        if callback:
            callback(msg)
        auto_gen_format = "cvvc"

    env_words_fallback = str(os.environ.get("UTOA_KR_MAPPING_WORDS_FALLBACK", "")).strip().lower()
    if env_words_fallback in {"0", "false", "off", "no"}:
        kr_mapping_words_fallback_enabled = False
    elif env_words_fallback in {"1", "true", "on", "yes"}:
        kr_mapping_words_fallback_enabled = True
    env_spn_th = str(os.environ.get("UTOA_KR_MAPPING_SPN_RATIO_THRESHOLD", "")).strip()
    if env_spn_th:
        try:
            kr_mapping_spn_ratio_threshold = float(env_spn_th)
        except Exception:
            pass
    env_vowel_ratio = str(os.environ.get("UTOA_KR_MAPPING_MIN_VOWEL_PHONE_RATIO", "")).strip()
    if env_vowel_ratio:
        try:
            kr_mapping_min_vowel_phone_ratio = float(env_vowel_ratio)
        except Exception:
            pass
    env_debug_reason = str(os.environ.get("UTOA_KR_MAPPING_DEBUG_REASON", "")).strip().lower()
    if env_debug_reason in {"0", "false", "off", "no"}:
        kr_mapping_debug_reason_logging = False
    elif env_debug_reason in {"1", "true", "on", "yes"}:
        kr_mapping_debug_reason_logging = True
    env_anchor_profile = str(os.environ.get("UTOA_KR_ANCHOR_PROFILE_PATH", "")).strip()
    if env_anchor_profile:
        kr_anchor_profile_path = env_anchor_profile
    env_conf_th = str(os.environ.get("UTOA_KR_MAPPING_CONF_THRESHOLD", "")).strip()
    if env_conf_th:
        try:
            kr_mapping_confidence_threshold = float(env_conf_th)
        except Exception:
            pass
    env_jump_default = str(os.environ.get("UTOA_KR_MAPPING_MAX_INDEX_JUMP_DEFAULT", "")).strip()
    if env_jump_default:
        try:
            kr_mapping_max_index_jump_default = int(float(env_jump_default))
        except Exception:
            pass
    env_jump_hi = str(os.environ.get("UTOA_KR_MAPPING_MAX_INDEX_JUMP_HIGH_CONF", "")).strip()
    if env_jump_hi:
        try:
            kr_mapping_max_index_jump_high_conf = int(float(env_jump_hi))
        except Exception:
            pass
    env_cleanup_jsonl = str(os.environ.get("UTOA_CLEANUP_TIMING_JSONL", "")).strip().lower()
    if env_cleanup_jsonl in {"0", "false", "off", "no"}:
        cleanup_timing_jsonl = False
    elif env_cleanup_jsonl in {"1", "true", "on", "yes"}:
        cleanup_timing_jsonl = True

    def log(msg):
        if callback:
            callback(msg)
        else:
            logger.info(msg)

    errors = []
    skipped_entries = []
    anchor_stats = {
        "anchor_locked_count": 0,
        "cutoff_clamped_count": 0,
        "vc_cutoff_leak_guard_count": 0,
    }
    _core_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(_core_dir)
    _anchor_log_dir = os.path.join(_project_dir, "logs")
    _anchor_log_name = f"timing_anchor_kr_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    anchor_log_path = os.path.join(_anchor_log_dir, _anchor_log_name)

    def _record_unset(reason, fname, line, meta=None):
        raw = (line or "").rstrip("\n")
        alias = ""
        if "=" in raw:
            rhs = raw.split("=", 1)[1]
            alias = rhs.split(",", 1)[0].strip()
        skipped_entries.append({
            "reason": reason,
            "file": fname or "",
            "alias": alias,
            "line": raw,
            "meta": meta or {},
        })

    def _record_unset_lines(reason, fname, src_lines, meta=None):
        for raw in (src_lines or []):
            _record_unset(reason, fname, raw, meta=meta)

    def _log_unset_summary():
        total_unset = len(skipped_entries)
        if total_unset <= 0:
            log("[Auto-OTO] 자동 설정 제외 항목: 0")
            return
        by_reason = {}
        for it in skipped_entries:
            r = it["reason"]
            by_reason[r] = by_reason.get(r, 0) + 1
        log(f"[Auto-OTO] 자동 설정 제외 항목: {total_unset}건")
        for reason, count in sorted(by_reason.items(), key=lambda x: (-x[1], x[0])):
            log(f"  - {reason}: {count}")
            shown = 0
            for it in skipped_entries:
                if it["reason"] != reason:
                    continue
                alias_txt = it["alias"] if it["alias"] else "<empty>"
                meta = it.get("meta") or {}
                extra = f" | {meta.get('diag_hint')}" if meta.get("diag_hint") else ""
                log(f"    예시: {it['file']} | alias={alias_txt}{extra}")
                shown += 1
                if shown >= 5:
                    break

    def _apply_kr_anchor_lock(
        *,
        fname: str,
        alias_text: str,
        format_type: str,
        alias_type: str,
        offset: float,
        consonant: float,
        cutoff: float,
        pre: float,
        ovl: float,
        timeline_start_ms: float,
        timeline_end_ms: float,
        file_duration_ms: float,
        anchor_abs_ms: float,
        next_onset_abs_ms: float | None = None,
        next_vowel_abs_ms: float | None = None,
        mapping_confidence: float = 1.0,
        lite: bool = False,
    ):
        fmt = str(format_type or "").strip().lower()
        if not is_anchor_lock_enabled("korean", fmt):
            return offset, consonant, cutoff, pre, ovl

        profile = get_anchor_profile("korean", fmt, alias_type, mode="rhythm_stable")
        if profile is None:
            return offset, consonant, cutoff, pre, ovl
        if fmt == "vcv":
            profile = _retune_kr_vcv_anchor_profile(profile, alias_text, alias_type)

        before = (float(offset), float(consonant), float(cutoff), float(pre), float(ovl))
        ctx = AnchorTimingContext(
            file_duration_ms=float(file_duration_ms or 0.0),
            timeline_start_ms=float(timeline_start_ms or 0.0),
            timeline_end_ms=float(timeline_end_ms or 0.0),
            anchor_abs_ms=float(anchor_abs_ms) if anchor_abs_ms is not None else None,
            next_onset_abs_ms=float(next_onset_abs_ms) if next_onset_abs_ms is not None else None,
            next_vowel_abs_ms=float(next_vowel_abs_ms) if next_vowel_abs_ms is not None else None,
            alias_type=str(alias_type or ""),
            language="korean",
            format_type=fmt,
            mapping_confidence=float(mapping_confidence or 1.0),
        )
        result = apply_anchor_lock(
            before,
            ctx,
            profile,
            validate_fn=validate_oto_params,
            lite=bool(lite),
        )
        rules = set(result.applied_rules or [])
        if rules:
            anchor_stats["anchor_locked_count"] += 1
            if "cutoff_next_onset_clamp" in rules or "cutoff_next_vowel_clamp" in rules:
                anchor_stats["cutoff_clamped_count"] += 1
                if str(alias_type or "").strip().lower() == "vc":
                    anchor_stats["vc_cutoff_leak_guard_count"] += 1
            append_timing_anchor_log(
                anchor_log_path,
                {
                    "event": "anchor_lock",
                    "language": "korean",
                    "format_type": fmt,
                    "alias_type": alias_type,
                    "file": fname,
                    "alias": alias_text,
                    "lite": bool(lite),
                    "before": {
                        "offset": before[0],
                        "consonant": before[1],
                        "cutoff": before[2],
                        "pre": before[3],
                        "ovl": before[4],
                    },
                    "after": {
                        "offset": float(result.offset),
                        "consonant": float(result.consonant),
                        "cutoff": float(result.cutoff),
                        "pre": float(result.pre),
                        "ovl": float(result.ovl),
                    },
                    "anchor_shift_ms": float(result.anchor_shift_ms),
                    "rules": sorted(rules),
                },
            )
        return result.offset, result.consonant, result.cutoff, result.pre, result.ovl

    kr_profile = None
    profile_path = _default_kr_profile_cache_path()
    try:
        env_profile = os.environ.get("UTOA_KR_PROFILE_JSON", "").strip()
        if env_profile:
            kr_profile = _load_kr_reference_profile(env_profile)
            if kr_profile:
                log(f"[KR-Profile] 외부 프로파일 로드: {env_profile}")

        if kr_anchor_profile_path:
            os.environ["UTOA_KR_ANCHOR_PROFILE_PATH"] = str(kr_anchor_profile_path)
            log(f"[KR-Anchor] 외부 앵커 프로파일 사용: {kr_anchor_profile_path}")

        if not kr_profile:
            kr_profile = _load_kr_reference_profile(profile_path)

        # Optional rebuild path for local experiments only (env-driven).
        if not kr_profile:
            ref_dirs = _resolve_kr_reference_dirs()
            if ref_dirs:
                trained = _build_kr_reference_profile_from_dirs(ref_dirs, custom_map=None)
                if trained and _save_kr_reference_profile(profile_path, trained):
                    kr_profile = trained
                    log(f"[KR-Profile] 로컬 샘플 기반 프로파일 생성: {profile_path} (rows={trained.get('rows', 0)})")

        # Default path: use abstract preset from python module.
        if not kr_profile:
            kr_profile = get_kr_profile_preset()
            _save_kr_reference_profile(profile_path, kr_profile)
            log(f"[KR-Profile] 추상화 프리셋 로드: {profile_path}")

        if kr_profile:
            b_count = len((kr_profile.get("buckets") or {}))
            log(f"[KR-Profile] 기준 프로파일 적용 준비: buckets={b_count}")
    except Exception as e:
        log(f"[KR-Profile] 프로파일 로드 실패: {e}")

    if tpl_path and not os.path.exists(tpl_path):
        log(f"⚠ 템플릿 파일을 찾을 수 없습니다: {tpl_path}")
        log(f"⚡ OpenUtau 호환 {auto_gen_format.upper()} 자동 에일리어스 생성으로 전환합니다.")
        tpl_path = ""


    VC_CONSONANT_RATIO = params.get('VC_CONSONANT_RATIO', 0.5) if params else 0.5
    VC_VOWEL_START = params.get('VC_VOWEL_START', 0.3) if params else 0.3
    VC_PRE_OFFSET = params.get('VC_PRE_OFFSET', 25) if params else 25
    VC_OVL_RATIO = params.get('VC_OVL_RATIO', 0.3) if params else 0.3
    CV_PRE_RATIO = params.get('CV_PRE_RATIO', 1.0) if params else 1.0
    CV_OVL_RATIO = params.get('CV_OVL_RATIO', 0.4) if params else 0.4
    DIPHTHONG_CV_PRE_RATIO = params.get('DIPHTHONG_CV_PRE_RATIO', 0.35) if params else 0.35
    DIPHTHONG_CV_CONSONANT_RATIO = params.get('DIPHTHONG_CV_CONSONANT_RATIO', 0.6) if params else 0.6
    DIPHTHONG_VC_VOWEL_START = params.get('DIPHTHONG_VC_VOWEL_START', 0.3) if params else 0.3
    DIPHTHONG_VC_CONSONANT = params.get('DIPHTHONG_VC_CONSONANT', 0.5) if params else 0.5


    template_lines = []
    if tpl_path:
        lines, detected_enc, warning, err = load_template_oto_lines(
            tpl_path,
            require_utf8=True,
            mode_label="한국어 OTO",
        )
        if err:
            log(err)
            log(f"⚡ 템플릿 로드 실패로 OpenUtau 호환 {auto_gen_format.upper()} 자동 에일리어스 생성으로 전환합니다.")
            lines = []
        if warning:
            log(warning)
        template_lines = lines or []


    tg_entries = []
    tg_exact_map = {}
    tg_norm_map = {}
    if os.path.exists(tg_folder):
        for f_name in os.listdir(tg_folder):
            if not f_name.lower().endswith('.textgrid'):
                continue
            base = os.path.splitext(f_name)[0]
            info = {
                'path': os.path.join(tg_folder, f_name),
                'real_name': base + '.wav',
                'base_lower': base.lower(),
                'norm_key': normalize_key(f_name),
            }
            tg_entries.append(info)
            if info['base_lower'] not in tg_exact_map:
                tg_exact_map[info['base_lower']] = info
            tg_norm_map.setdefault(info['norm_key'], []).append(info)

    def _resolve_tg_info(fname):
        wav_name = os.path.basename((fname or '').strip())
        base_lower = os.path.splitext(wav_name)[0].lower()
        if base_lower in tg_exact_map:
            return tg_exact_map[base_lower]

        norm_name = normalize_key(wav_name)
        candidates = tg_norm_map.get(norm_name, [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            exact_name = wav_name.lower()
            same_name = [c for c in candidates if c['real_name'].lower() == exact_name]
            if len(same_name) == 1:
                return same_name[0]
            log(f"파일명 매핑 충돌: {wav_name} (정규화 키 {norm_name}, 후보 {len(candidates)}개) -> 원본 파일명을 유지합니다.")
            return None
        return None

    def _template_match_stats(lines):
        file_names = set()
        for line in (lines or []):
            if "=" not in line:
                continue
            file_names.add(line.split("=", 1)[0].strip())
        total = len(file_names)
        if total == 0:
            return 0, 0, 0.0
        matched = 0
        for fname in file_names:
            if _resolve_tg_info(fname):
                matched += 1
        return matched, total, (matched / float(total))

    final_lines = []


    custom_map = load_custom_phonemes(custom_phonemes_path)
    alias_type_cache = {}

    def _classify_alias_cached(alias_text):
        t = alias_type_cache.get(alias_text)
        if t is None:
            t = classify_alias(alias_text, custom_map)
            alias_type_cache[alias_text] = t
        return t


    use_template = bool(template_lines)
    if use_template:
        t_match, t_total, t_ratio = _template_match_stats(template_lines)
        if t_total == 0 or t_match == 0 or t_ratio < 0.25:
            log(
                f"⚠ 템플릿-TextGrid 매칭률 낮음 ({t_match}/{t_total}, {t_ratio:.1%}) "
                f"-> OpenUtau 호환 {auto_gen_format.upper()} 자동 에일리어스 생성으로 전환"
            )
            use_template = False
        else:
            log(f"📌 템플릿 베이스 OTO 사용 ({t_match}/{t_total}, {t_ratio:.1%})")

    if use_template:
        file_groups = {}
        for line in template_lines:
            fname = line.split('=')[0]
            if fname not in file_groups:
                file_groups[fname] = []
            file_groups[fname].append(line)
    else:

        log(f"⚡ 템플릿 없음/미적합 -> OpenUtau 호환 {auto_gen_format.upper()} 형식으로 에일리어스를 자동 생성합니다.")
        file_groups = {}
        try:
            from core.lab_generator import decompose_hangul_to_roman
        except ImportError:
            pass # fallback
        
        for tg_info in tg_entries:
            tg_path = tg_info['path']
            real_name = tg_info['real_name']
            

            try:
                tg = textgrid.TextGrid.fromFile(tg_path)
                w_tier = None
                for t in tg:
                    if hasattr(t, 'name') and t.name == 'words':
                        w_tier = t
                        break
                if not w_tier:
                    continue
                    
                wd_intervals = [i for i in w_tier if i.mark.strip() not in ['', 'sil', 'spn', 'pau']]
                if not wd_intervals:
                    continue
                    
                lines = []

                base_name = os.path.splitext(real_name)[0].lower()
                is_long = base_name.endswith('long') or len(wd_intervals) == 1
                
                if is_long:

                    for w in wd_intervals:
                        roman_parts = []
                        for ch in w.mark:
                            roman_parts.extend(decompose_hangul_to_roman(ch))
                        roman = "".join(roman_parts).lower()
                        # CV / Mono
                        lines.append(f"{real_name}={roman.capitalize() if roman not in KR_VOWELS else roman},0,0,0,0,0")
                else:
                    coda_alias_map = {
                        "ng": "NG",
                        "n": "N",
                        "l": "L",
                        "r": "L",
                        "m": "M",
                        "k": "K",
                        "g": "K",
                        "t": "T",
                        "d": "T",
                        "s": "T",
                        "ss": "T",
                        "j": "T",
                        "jj": "T",
                        "ch": "T",
                        "h": "H",
                        "p": "P",
                        "b": "P",
                    }

                    for idx, w in enumerate(wd_intervals):
                        roman_parts = []
                        for ch in w.mark:
                            roman_parts.extend(decompose_hangul_to_roman(ch))
                        roman = "".join(roman_parts).lower()
                        
                        if auto_gen_format == 'vcv':
                            if idx == 0:
                                lines.append(f"{real_name}=- {roman},0,0,0,0,0")
                            else:
                                prev_w = wd_intervals[idx-1]
                                prev_parts = []
                                for ch in prev_w.mark:
                                    prev_parts.extend(decompose_hangul_to_roman(ch))
                                prev_roman = "".join(prev_parts).lower()
                                _, prev_vowel, prev_coda = _split_kr_syllable_parts(prev_roman)
                                prev_coda_alias = coda_alias_map.get((prev_coda or "").lower(), "")
                                
                                if prev_coda_alias:
                                    lines.append(f"{real_name}={prev_coda_alias} {roman},0,0,0,0,0")
                                else:
                                    lines.append(f"{real_name}={(prev_vowel or 'a')} {roman},0,0,0,0,0")
                                
                        elif auto_gen_format == 'cvvc':

                            lines.append(f"{real_name}={roman},0,0,0,0,0")
                            
                            if idx > 0:

                                prev_w = wd_intervals[idx-1]
                                prev_parts = []
                                for ch in prev_w.mark:
                                    prev_parts.extend(decompose_hangul_to_roman(ch))
                                prev_roman = "".join(prev_parts).lower()
                                _, prev_vowel, _ = _split_kr_syllable_parts(prev_roman)
                                prev_vowel = prev_vowel or "a"
                                

                                cur_start_cons = ""
                                for char in roman:
                                    if char in KR_CONSONANTS:
                                        cur_start_cons += char
                                    else:
                                        break
                                
                                if cur_start_cons:
                                    lines.append(f"{real_name}={prev_vowel} {cur_start_cons},0,0,0,0,0")
                                else:

                                    _, cur_vowel, _ = _split_kr_syllable_parts(roman)
                                    lines.append(f"{real_name}={prev_vowel} {(cur_vowel or 'a')},0,0,0,0,0")
                
                if lines:
                    file_groups[real_name] = lines

            except Exception as e:
                log(f"경고: {real_name} 자동 템플릿 생성 실패 ({e})")
                
    processed = 0
    total = len(file_groups)
    wav_root_for_signal = os.path.dirname(os.path.abspath(tg_folder.rstrip("\\/")))
    wav_index_for_signal = {}
    try:
        if os.path.isdir(wav_root_for_signal):
            for fn in os.listdir(wav_root_for_signal):
                if fn.lower().endswith(".wav"):
                    wav_index_for_signal[normalize_key(fn)] = os.path.join(wav_root_for_signal, fn)
    except Exception:
        pass
    mel_cache_for_signal = {}

    for fname, lines in file_groups.items():
        tg_info = _resolve_tg_info(fname)

        if not tg_info:
            log(f"경고: {fname}: TextGrid를 찾을 수 없어 원본 라인을 유지합니다.")
            _record_unset_lines("textgrid_missing", fname, lines)
            final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
            processed += 1
            continue
            
        tg_path = tg_info['path']
        real_wav_name = tg_info['real_name']
        wav_path_for_signal = _find_wav_path_for_name(real_wav_name, wav_root_for_signal, wav_index_for_signal)
        mel_ctx_for_file = None
        wav_duration_ms = 0.0
        if wav_path_for_signal:
            mel_ctx_for_file = mel_cache_for_signal.get(wav_path_for_signal)
            if mel_ctx_for_file is None:
                audio_sig, sr_sig = _read_wav_mono_np(wav_path_for_signal)
                mel_ctx_for_file = _mel_envelope(audio_sig, sr_sig)
                mel_cache_for_signal[wav_path_for_signal] = mel_ctx_for_file
                if sr_sig:
                    wav_duration_ms = (len(audio_sig) / float(sr_sig)) * 1000.0
            else:
                wav_duration_ms = _wav_duration_ms(wav_path_for_signal)

        try:
            tg = textgrid.TextGrid.fromFile(tg_path)
            phone_tier = None
            word_tier = None
            for t in tg:
                if isinstance(t, textgrid.IntervalTier):
                    if t.name == 'phones':
                        phone_tier = t
                    elif t.name == 'words':
                        word_tier = t

            if not phone_tier:
                log(f"경고: {fname}: phones tier가 없어 원본 라인을 유지합니다.")
                _record_unset_lines("tier_missing", fname, lines)
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue

            try:
                from core.lab_generator import decompose_hangul_to_roman
            except ImportError:
                def decompose_hangul_to_roman(ch):
                    return [ch]


            phone_silence_marks = {'', 'sil', 'pau', 'sp'}
            ph_intervals_raw = [i for i in phone_tier if i.mark.strip().lower() not in phone_silence_marks]
            ph_intervals_all = [i for i in ph_intervals_raw if i.mark.strip().lower() not in {'spn'}]
            ph_intervals = [i for i in ph_intervals_all if i.mark.strip().lower() not in {'sil'}]
            wd_intervals = [i for i in word_tier if i.mark.strip() not in ['', 'sil', 'spn', 'pau']] if word_tier else []

            if len(ph_intervals) == 0:
                if kr_mapping_words_fallback_enabled and wd_intervals:
                    fallback_phones = []
                    for w in wd_intervals:
                        fallback_phones.extend(
                            _synthesize_kr_word_phones(
                                w.mark,
                                float(w.minTime),
                                float(w.maxTime),
                                decompose_hangul_to_roman,
                            )
                        )
                    ph_intervals = fallback_phones
                    ph_intervals_all = fallback_phones
                if len(ph_intervals) == 0:
                    log(f"경고: {fname}: 유효한 음소 구간이 없어 원본 라인을 유지합니다.")
                    _record_unset_lines("mapping_failed_empty_intervals", fname, lines)
                    final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                    processed += 1
                    continue

            timeline_start_ms = float(ph_intervals[0].minTime * 1000.0)
            timeline_end_ms = float(ph_intervals[-1].maxTime * 1000.0)
            if wd_intervals:
                timeline_start_ms = min(timeline_start_ms, float(wd_intervals[0].minTime * 1000.0))
                timeline_end_ms = max(timeline_end_ms, float(wd_intervals[-1].maxTime * 1000.0))
            if wav_duration_ms <= 0.0:
                wav_duration_ms = timeline_end_ms


            if len(ph_intervals) == 1 and len(wd_intervals) == 1:
                log(f"처리: {fname}: 단모음 파일")
                vowel = ph_intervals[0]
                v_start = vowel.minTime * 1000
                v_end = vowel.maxTime * 1000
                v_len = v_end - v_start
                for line in lines:
                    parts = line.split('=', 1)
                    if len(parts) < 2:
                        _record_unset("malformed_line", fname, line)
                        final_lines.append(apply_suffix_to_oto_line(line, alias_suffix))
                        continue
                    alias = parts[1].split(',')[0].strip()
                    if not alias:
                        _record_unset("empty_alias", fname, line)
                        preserved = f"{real_wav_name}={parts[1]}"
                        final_lines.append(apply_suffix_to_oto_line(preserved, alias_suffix))
                        continue


                    offset = v_start
                    pre = 0
                    ovl = 0
                    consonant = min(v_len * 0.25, 120)
                    cutoff = -(v_len * 0.8)
                    
                    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
                    
                    _append_alias_rows(
                        final_lines,
                        real_wav_name,
                        alias,
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        generate_openutau=generate_openutau,
                        alias_suffix=alias_suffix,
                    )
                processed += 1
                continue


            alias_names = []
            for l in lines:
                if "=" not in l:
                    continue
                a = l.split("=", 1)[1].split(",", 1)[0].strip()
                if a:
                    alias_names.append(a)
            if not alias_names:
                log(f"경고: {fname}: 유효한 에일리어스가 없어 원본 라인을 유지합니다.")
                _record_unset_lines("no_valid_alias", fname, lines)
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue
            file_format = detect_alias_format(alias_names, custom_map=custom_map)
            log(f"처리: {fname}: 형식 감지 -> {file_format.upper()}")
            file_mapping_conf_th = _resolve_kr_mapping_conf_threshold(
                file_format,
                override_threshold=kr_mapping_confidence_threshold,
            )
            
            is_vc_only = (file_format == 'vc_only')
            is_vcv_file = (file_format == 'vcv')

            cv_targets = _extract_cv_targets_from_lines(lines, custom_map)
            filename_cv_targets = _extract_kr_cv_targets_from_filename(real_wav_name)
            targets_for_build = cv_targets
            if file_format == "cvvc" and filename_cv_targets:
                targets_for_build = filename_cv_targets
            elif (not targets_for_build) and filename_cv_targets:
                targets_for_build = filename_cv_targets
            expected_syllables = max(len(wd_intervals), len(cv_targets), len(filename_cv_targets))
            phone_quality = _collect_kr_phone_tier_quality(
                phone_tier,
                expected_syllables=expected_syllables,
                min_vowel_phone_ratio=kr_mapping_min_vowel_phone_ratio,
            )
            low_quality_reasons = list(phone_quality.get("low_confidence_reasons", []))
            spn_ratio = float(phone_quality.get("spn_ratio_in_phone_tier", 0.0))
            if spn_ratio >= float(kr_mapping_spn_ratio_threshold):
                low_quality_reasons.append("spn_heavy")
            low_quality_reasons = sorted(set(low_quality_reasons))
            low_phone_quality = bool(low_quality_reasons)
            force_words_phone_fill = bool(
                kr_mapping_words_fallback_enabled and low_phone_quality and wd_intervals
            )
            if force_words_phone_fill and kr_mapping_debug_reason_logging:
                log(
                    f"🧭 {fname}: phones 신뢰도 낮음({','.join(low_quality_reasons)}) "
                    f"→ words 구간 보강 매핑 사용"
                )
            syllables_info = []
            used_words_based = False
            used_alias_based = False
            base_score = 0.0
            alt_score = 0.0
            mapping_reason_code = "filename_token"
            if wd_intervals:
                for w in wd_intervals:
                    w_start = w.minTime
                    w_end = w.maxTime

                    s_phones = [p for p in ph_intervals if p.minTime >= w_start - 0.01 and p.maxTime <= w_end + 0.01]
                    if force_words_phone_fill and not s_phones:
                        s_phones = _synthesize_kr_word_phones(
                            w.mark,
                            float(w_start),
                            float(w_end),
                            decompose_hangul_to_roman,
                        )
                    roman_parts = []
                    for ch in w.mark:
                        roman_parts.extend(decompose_hangul_to_roman(ch))
                    roman_raw = "".join(roman_parts).lower()
                    roman_cv = _kr_cv_kernel(roman_raw)
                    
                    syllables_info.append({
                        'word': w.mark,
                        'roman': roman_raw,
                        'roman_cv': roman_cv,
                        'start_time': w_start,
                        'end_time': w_end,
                        'phones': s_phones
                    })
                used_words_based = len(syllables_info) > 0

            alias_based = _build_kr_syllables_from_phone_nuclei(ph_intervals, targets_for_build) if targets_for_build else None
            if syllables_info and any(len(s.get('phones') or []) == 0 for s in syllables_info) and alias_based:
                syllables_info = alias_based
                used_alias_based = True
                used_words_based = False
                mapping_reason_code = "alias_based_empty_words"
                log(f"🧭 {fname}: words 매핑에 빈 phone 구간 존재 → alias/phone 기반 음절 매핑 사용")
            elif not syllables_info and alias_based:
                syllables_info = alias_based
                used_alias_based = True
                used_words_based = False
                mapping_reason_code = "alias_phone_minimal"
                log(f"🧭 {fname}: words 티어 없음/실패 → phones 핵 기반 음절 매핑 사용")
            elif syllables_info and alias_based and targets_for_build:
                base_score = float(_score_kr_syllable_mapping(syllables_info, targets_for_build))
                alt_score = float(_score_kr_syllable_mapping(alias_based, targets_for_build))
                if low_phone_quality and used_words_based:
                    mapping_reason_code = "words_low_phone_quality"
                    log(
                        f"🧭 {fname}: phones 저신뢰({','.join(low_quality_reasons)}) → words 매핑 고정 "
                        f"(words={base_score:.1f}, alias={alt_score:.1f})"
                    )
                elif _should_prefer_alias_based_syllables(file_format, used_words_based, base_score, alt_score):
                    syllables_info = alias_based
                    used_alias_based = True
                    if file_format == "cvvc":
                        mapping_reason_code = "alias_based_cvvc"
                        log(
                            f"🧭 {fname}: CVVC는 alias 기반 음절 매핑 우선 "
                            f"(words={base_score:.1f}, alias={alt_score:.1f})"
                        )
                    else:
                        mapping_reason_code = "alias_based_recover"
                        log(
                            f"🧭 {fname}: 매핑 이탈 보정 적용 "
                            f"(base={base_score:.1f}, corrected={alt_score:.1f})"
                        )
                else:
                    if file_format != "cvvc" and used_words_based and base_score >= 58.0 and alt_score >= (base_score + 8.0):
                        mapping_reason_code = "words_keep_high_conf"
                        log(
                            f"🧭 {fname}: words 매핑 신뢰도 높음 → alias 보정 생략 "
                            f"(base={base_score:.1f}, corrected={alt_score:.1f})"
                        )
                    else:
                        mapping_reason_code = "words_keep"
                    log(
                        f"🧭 {fname}: TextGrid(words) 매핑 유지 "
                        f"(base={base_score:.1f}, corrected={alt_score:.1f})"
                    )

            mapping_confidence_base, mapping_margin = _estimate_kr_mapping_confidence(
                phone_quality,
                words_score=base_score,
                alias_score=alt_score,
                used_words_based=used_words_based,
                used_alias_based=used_alias_based,
            )
            if kr_mapping_debug_reason_logging and mapping_confidence_base < float(file_mapping_conf_th):
                log(
                    f"🧭 {fname}: KR 매핑 신뢰도 낮음(conf={mapping_confidence_base:.2f}, "
                    f"margin={mapping_margin:+.1f}, reason={mapping_reason_code})"
                )

            file_mapping_low_conf = bool(
                float(mapping_confidence_base) < max(float(file_mapping_conf_th), 0.58)
                or (float(base_score) < 48.0 and float(alt_score) < 48.0)
            )

            if (not syllables_info) or any(len(s['phones']) == 0 for s in syllables_info):
                log(f"경고: {fname}: 음절-음소 매핑 실패로 원본 라인을 유지합니다.")
                fail_reason = "mapping_failed"
                if "spn_heavy" in low_quality_reasons:
                    fail_reason = "mapping_failed_spn_heavy"
                elif "insufficient_phones" in low_quality_reasons or "insufficient_vowel_phones" in low_quality_reasons:
                    fail_reason = "mapping_failed_insufficient_phones"
                elif low_phone_quality and not wd_intervals:
                    fail_reason = "mapping_failed_no_words_support"
                _record_unset_lines(
                    fail_reason,
                    fname,
                    lines,
                    meta={
                        "diag_hint": f"spn_ratio={spn_ratio:.2f}; conf={mapping_confidence_base:.2f}",
                        "phone_quality": phone_quality,
                        "force_words_phone_fill": force_words_phone_fill,
                        "mapping_confidence": mapping_confidence_base,
                        "mapping_reason_code": mapping_reason_code,
                    },
                )
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue
                
            romaji_syllables = [s.get('roman_cv') or s.get('roman', '') for s in syllables_info]
            current_w_idx = 0
            cv_seq_idx = 0
            kr_cvvc_occurrence_source = filename_cv_targets if (file_format == "cvvc" and filename_cv_targets) else syllables_info
            kr_cvvc_occurrence_map = _build_kr_cvvc_occurrence_map(kr_cvvc_occurrence_source) if file_format == "cvvc" else None
            kr_cvvc_occurrence_state = {}
            kr_cvvc_vv_occurrence_map = _build_kr_cvvc_vv_occurrence_map(kr_cvvc_occurrence_source) if file_format == "cvvc" else None
            kr_cvvc_vv_occurrence_state = {}

            cv_anchor_by_idx = {
                i: _estimate_cv_anchor_from_syllable(syllables_info[i], ph_intervals)
                for i in range(len(syllables_info))
            }
            realized_cv_anchor_by_idx = {}
            kr_post_ctx = KrPostprocessContext(
                file_format=file_format,
                mel_ctx_for_file=mel_ctx_for_file,
                ph_intervals=ph_intervals,
                syllables_info=syllables_info,
                validate_fn=validate_oto_params,
                soft_mel_guard_fn=_apply_soft_mel_offset_cutoff_guard,
                base_shape_blend_fn=_apply_base_shape_blend,
                stabilize_fn=_stabilize_params_to_phone_activity,
                recenter_fn=_recenter_kr_params_around_pre,
                cv_cutoff_guard_fn=_guard_cv_cutoff_to_next_onset,
            )
            
            for line_num, line in enumerate(lines):
                parts = line.split('=', 1)
                if len(parts) < 2:
                    _record_unset("malformed_line", fname, line)
                    final_lines.append(apply_suffix_to_oto_line(line, alias_suffix))
                    continue
                alias = parts[1].split(',', 1)[0].strip()
                if not alias:
                    _record_unset("empty_alias", fname, line)
                    preserved = f"{real_wav_name}={parts[1]}"
                    final_lines.append(apply_suffix_to_oto_line(preserved, alias_suffix))
                    continue
                base_shape = _extract_base_timing_shape(line)
                row_mapping_confidence = float(mapping_confidence_base)
                row_mapping_reason_code = str(mapping_reason_code or "")
                row_jump_blocked = 0
                

                alias_type = _classify_alias_cached(alias)
                row_jump_default = int(max(0, kr_mapping_max_index_jump_default))
                row_jump_high_conf = int(max(row_jump_default, kr_mapping_max_index_jump_high_conf))
                if alias_type == "cv_head":
                    row_jump_default = int(max(0, row_jump_default - 1))
                    row_jump_high_conf = int(max(row_jump_default, row_jump_high_conf - 1))
                elif alias_type == "vv":
                    row_jump_default = int(max(row_jump_default, kr_mapping_max_index_jump_default + 1))
                    row_jump_high_conf = int(max(row_jump_high_conf, row_jump_default + 1))

                if alias_type == 'br':

                    first_ph = ph_intervals[0] if ph_intervals else None
                    last_ph = ph_intervals[-1] if ph_intervals else None
                    if first_ph and last_ph:
                        br_start = first_ph.minTime * 1000
                        br_end = last_ph.maxTime * 1000
                        br_len = br_end - br_start
                    else:
                        br_start = 0
                        br_len = 500
                    offset = max(br_start - 30, 0)
                    pre = 0
                    ovl = 0
                    consonant = min(br_len * 0.3, 100)
                    cutoff = -(br_len * 0.85)
                    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
                    
                    _append_alias_rows(
                        final_lines,
                        real_wav_name,
                        alias,
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        generate_openutau=generate_openutau,
                        alias_suffix=alias_suffix,
                    )
                    continue
                

                is_vc = _uses_kr_vc_context(alias_type)
                is_vcv = alias_type == 'vcv'
                is_cv_head = alias_type == 'cv_head'
                is_diph = is_diphthong(alias)
                
                target_clean = _extract_kr_cv_alias_token(alias)




                glottal_kind = _detect_glottal_kind(alias)
                if glottal_kind in ('head', 'tail'):

                    if glottal_kind == 'head':
                        if cv_seq_idx < len(syllables_info):
                            current_w_idx = cv_seq_idx
                            cv_seq_idx = current_w_idx + 1

                    if current_w_idx >= len(syllables_info):
                        current_w_idx = len(syllables_info) - 1

                    curr_syl = syllables_info[current_w_idx]
                    curr_phones = curr_syl['phones']
                    _, v_phone = find_vowel_phone(curr_phones)
                    vowel_start = v_phone.minTime * 1000
                    vowel_end = v_phone.maxTime * 1000


                    g_idx = None
                    for i, p in enumerate(ph_intervals):
                        if abs(p.minTime - v_phone.minTime) < 1e-6 and abs(p.maxTime - v_phone.maxTime) < 1e-6:
                            g_idx = i
                            break

                    if glottal_kind == 'tail':
                        next_ph = ph_intervals[g_idx + 1] if g_idx is not None and g_idx + 1 < len(ph_intervals) else None
                        boundary_end = (next_ph.maxTime * 1000) if next_ph else (curr_syl['end_time'] * 1000)
                        glottal_len = max(boundary_end - vowel_end, 60)

                        offset_padding = min(max(glottal_len, 60), 220)
                        offset = max(boundary_end - offset_padding, 0)
                        pre = boundary_end - offset
                        ovl = pre * 0.2
                        consonant = pre + min(glottal_len * 0.3, 30)
                        cutoff = -(consonant + 30)

                    else:
                        prev_ph = ph_intervals[g_idx - 1] if g_idx is not None and g_idx - 1 >= 0 else None
                        glottal_start = (prev_ph.minTime * 1000) if prev_ph else max(vowel_start - 80, 0)
                        boundary = (prev_ph.maxTime * 1000) if prev_ph else vowel_start

                        offset = max(glottal_start - 30, 0)
                        pre = boundary - offset
                        ovl = pre * 0.3
                        vowel_len = max(vowel_end - vowel_start, 80)
                        added_cons = min(max(vowel_len * 0.5, 80), 150)
                        consonant = pre + added_cons
                        cutoff = -(consonant + vowel_len * 0.25)

                    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)

                    a2 = apply_alias_suffix(alias, alias_suffix)
                    new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                    final_lines.append(new_line)
                    continue




                breath_tail = None
                a_parts = alias.split()
                if len(a_parts) >= 2 and a_parts[0].lower() in KR_VOWELS and a_parts[1].upper() in {'R', 'H'}:
                    breath_tail = a_parts[1].upper()
                elif alias and alias[-1].upper() in {'R', 'H'} and alias[:-1].lower() in KR_VOWELS:
                    breath_tail = alias[-1].upper()

                if breath_tail:
                    if current_w_idx >= len(syllables_info):
                        current_w_idx = len(syllables_info) - 1
                    curr_syl = syllables_info[current_w_idx]
                    curr_phones = curr_syl['phones']
                    _, v_phone = find_vowel_phone(curr_phones)
                    vowel_start = v_phone.minTime * 1000
                    vowel_end = v_phone.maxTime * 1000
                    vowel_len = max(vowel_end - vowel_start, 80)

                    last_end = (ph_intervals_all[-1].maxTime * 1000) if ph_intervals_all else (curr_syl['end_time'] * 1000)


                    offset_padding = min(max(vowel_len * 0.7, 180), 320)
                    offset = max(vowel_end - offset_padding, 0)


                    pre_abs = max(vowel_end - 20, offset)
                    pre = pre_abs - offset
                    ovl = pre * 0.85


                    consonant = max(vowel_end - offset, pre + 10)


                    cutoff = -(max(last_end - offset, consonant + 80))

                    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)

                    a2 = apply_alias_suffix(alias, alias_suffix)
                    new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                    final_lines.append(new_line)
                    continue
                


                if is_vcv:
                    (
                        current_w_idx,
                        cv_seq_idx,
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                    ) = _prepare_vcv_syllable_timing(
                        syllables_info,
                        current_w_idx,
                        cv_seq_idx,
                        DIPHTHONG_CV_CONSONANT_RATIO,
                    )
                    (
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        soft_off_shift,
                        soft_cut_shift,
                        cutoff_reduced,
                    ) = _apply_post_timing_pipeline(
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        alias_type="vcv",
                        alias_text=alias,
                        file_format=file_format,
                        mel_ctx_for_file=mel_ctx_for_file,
                        base_shape=base_shape,
                        ph_intervals=ph_intervals,
                        current_w_idx=current_w_idx,
                        syllables_info=syllables_info,
                        is_vc_plosive_coda=False,
                        enable_stabilize=False,
                        enable_cutoff_guard=False,
                        post_ctx=kr_post_ctx,
                    )
                    vcv_anchor = None
                    vcv_next_onset = None
                    vcv_next_vowel = None
                    if 0 <= current_w_idx < len(syllables_info):
                        vcv_curr = syllables_info[current_w_idx]
                        vcv_phones = vcv_curr.get("phones") or []
                        if vcv_phones:
                            if len(vcv_phones) >= 2:
                                vcv_anchor = float(vcv_phones[-1].minTime) * 1000.0
                                vcv_next_onset = vcv_anchor
                                vcv_next_vowel = float(vcv_phones[-1].maxTime) * 1000.0
                            else:
                                vcv_anchor = float(vcv_phones[0].maxTime) * 1000.0
                    if vcv_anchor is not None:
                        offset, consonant, cutoff, pre, ovl = _apply_kr_anchor_lock(
                            fname=fname,
                            alias_text=alias,
                            format_type=file_format,
                            alias_type="vcv",
                            offset=offset,
                            consonant=consonant,
                            cutoff=cutoff,
                            pre=pre,
                            ovl=ovl,
                            timeline_start_ms=timeline_start_ms,
                            timeline_end_ms=timeline_end_ms,
                            file_duration_ms=wav_duration_ms,
                            anchor_abs_ms=vcv_anchor,
                            next_onset_abs_ms=vcv_next_onset,
                            next_vowel_abs_ms=vcv_next_vowel,
                            mapping_confidence=row_mapping_confidence,
                        )
                    _log_post_timing_events(log, fname, alias, soft_off_shift, soft_cut_shift, cutoff_reduced)
                    
                    _append_alias_rows(
                        final_lines,
                        real_wav_name,
                        alias,
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        generate_openutau=generate_openutau,
                        alias_suffix=alias_suffix,
                    )
                    continue
                

                if is_cv_head:
                    forced_cvvc_idx = _resolve_kr_cvvc_occurrence_index(
                        alias,
                        alias_type,
                        kr_cvvc_occurrence_map or {},
                        kr_cvvc_occurrence_state,
                    )
                    (
                        selected_w_idx,
                        cv_seq_idx,
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        n_start,
                        n_end,
                    ) = _prepare_cv_head_syllable_timing(
                        syllables_info,
                        current_w_idx,
                        cv_seq_idx,
                        alias,
                        forced_w_idx=forced_cvvc_idx,
                    )
                    current_w_idx = max(current_w_idx, selected_w_idx)
                    (
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        soft_off_shift,
                        soft_cut_shift,
                        cutoff_reduced,
                    ) = _apply_post_timing_pipeline(
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        alias_type="cv_head",
                        alias_text=alias,
                        file_format=file_format,
                        mel_ctx_for_file=mel_ctx_for_file,
                        base_shape=base_shape,
                        ph_intervals=ph_intervals,
                        current_w_idx=selected_w_idx,
                        syllables_info=syllables_info,
                        is_vc_plosive_coda=False,
                        enable_stabilize=False,
                        enable_cutoff_guard=False,
                    )
                    (
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        offset_reduced,
                    ) = _guard_cv_head_offset_to_current_onset(
                        offset, consonant, cutoff, pre, selected_w_idx, syllables_info
                    )
                    (
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        cutoff_extended,
                    ) = _ensure_cv_head_min_vowel_coverage(
                        offset, consonant, cutoff, pre, n_start, n_end
                    )
                    (
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        cutoff_reduced_after_offset,
                    ) = _guard_cv_cutoff_to_next_onset(
                        offset, consonant, cutoff, pre, selected_w_idx, syllables_info
                    )
                    cutoff_reduced += cutoff_reduced_after_offset
                    cvh_anchor = None
                    _sw_c_start = None
                    _sw_c_end = None
                    _sw_n_start = None
                    _sw_n_end = None
                    try:
                        _sw_idx, _sw_phones, _sw_c_start, _sw_c_end, _sw_n_start, _sw_n_end = _prepare_cv_bounds_from_syllable(
                            syllables_info, selected_w_idx
                        )
                        cvh_anchor = _sw_c_end
                    except Exception:
                        cvh_anchor = n_start if n_start is not None else cvh_anchor
                    if cvh_anchor is not None:
                        offset, consonant, cutoff, pre, ovl = _apply_kr_anchor_lock(
                            fname=fname,
                            alias_text=alias,
                            format_type=file_format,
                            alias_type="cv_head",
                            offset=offset,
                            consonant=consonant,
                            cutoff=cutoff,
                            pre=pre,
                            ovl=ovl,
                            timeline_start_ms=timeline_start_ms,
                            timeline_end_ms=timeline_end_ms,
                            file_duration_ms=wav_duration_ms,
                            anchor_abs_ms=cvh_anchor,
                            next_onset_abs_ms=n_start,
                            next_vowel_abs_ms=n_end,
                            mapping_confidence=row_mapping_confidence,
                        )
                    if (
                        selected_w_idx is not None
                        and _sw_c_start is not None
                        and _sw_c_end is not None
                    ):
                        realized_cv_anchor_by_idx[selected_w_idx] = {
                            "offset": float(offset),
                            "pre": float(pre),
                            "ovl": float(ovl),
                            "cons": float(consonant),
                            "cutoff": float(cutoff),
                            "pre_abs": float(offset + pre),
                            "cons_abs": float(offset + consonant),
                            "onset_abs": float(_sw_c_start),
                            "vowel_start_abs": float(_sw_n_start if _sw_n_start is not None else n_start),
                            "vowel_end_abs": float(_sw_n_end if _sw_n_end is not None else n_end),
                            "vowel_len": max(
                                12.0,
                                float((_sw_n_end if _sw_n_end is not None else n_end) - (_sw_n_start if _sw_n_start is not None else n_start)),
                            ),
                            "cons_gap": max(float(consonant - pre), 10.0),
                            "cut_gap": max(float(abs(cutoff) - consonant), 16.0),
                        }
                    _log_post_timing_events(log, fname, alias, soft_off_shift, soft_cut_shift, cutoff_reduced)
                    if offset_reduced > 1.0:
                        log(f"🛡️ {fname}: CV_HEAD 오프셋 과선행 보정(+{offset_reduced:.1f}ms) [{alias}]")
                    if cutoff_extended > 1.0:
                        log(f"🛡️ {fname}: CV_HEAD 모음 길이 보정(+{cutoff_extended:.1f}ms) [{alias}]")
                    
                    _append_alias_rows(
                        final_lines,
                        real_wav_name,
                        alias,
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        generate_openutau=generate_openutau,
                        alias_suffix=alias_suffix,
                    )
                    continue
                

                if not is_vc:
                    forced_vv_idx = None
                    if file_format == "cvvc" and alias_type == "vv":
                        forced_vv_idx = _resolve_kr_cvvc_vv_index(
                            alias,
                            kr_cvvc_vv_occurrence_map or {},
                            kr_cvvc_vv_occurrence_state,
                        )
                    forced_cvvc_idx = _resolve_kr_cvvc_occurrence_index(
                        alias,
                        alias_type,
                        kr_cvvc_occurrence_map or {},
                        kr_cvvc_occurrence_state,
                    )
                    expected_cv_idx = cv_seq_idx
                    selected_w_idx = None
                    resolve_meta = {}
                    forced_selected_idx = forced_vv_idx if forced_vv_idx is not None else forced_cvvc_idx
                    forced_gate_rejected = False
                    if forced_selected_idx is not None:
                        forced_score = -1.0
                        expected_score_forced = -1.0
                        forced_vowel_mismatch = False
                        if target_clean and 0 <= forced_selected_idx < len(romaji_syllables):
                            forced_score = float(_cv_match_score(target_clean, romaji_syllables[forced_selected_idx]))
                            if 0 <= expected_cv_idx < len(romaji_syllables):
                                expected_score_forced = float(_cv_match_score(target_clean, romaji_syllables[expected_cv_idx]))
                            _t_on, _t_v, _t_c = _split_kr_syllable_parts(target_clean)
                            _f_on, _f_v, _f_c = _split_kr_syllable_parts(romaji_syllables[forced_selected_idx])
                            forced_vowel_mismatch = bool(_t_v and _f_v and _t_v != _f_v)
                        min_forced_score = 64.0 if alias_type == "cv_head" else 60.0
                        if file_mapping_low_conf:
                            min_forced_score += 6.0
                        keep_forced = True
                        if forced_score >= 0.0 and forced_score < min_forced_score:
                            keep_forced = False
                        if forced_vowel_mismatch and file_mapping_low_conf:
                            keep_forced = False
                        if expected_score_forced >= 0.0 and forced_score >= 0.0 and (forced_score + 14.0) < expected_score_forced:
                            keep_forced = False
                        if keep_forced:
                            selected_w_idx = forced_selected_idx
                            cv_seq_idx = max(cv_seq_idx, selected_w_idx + 1)
                            row_mapping_reason_code = "forced_occurrence_index"
                        else:
                            forced_gate_rejected = True
                            row_mapping_confidence = max(0.0, row_mapping_confidence - 0.08)
                            row_mapping_reason_code = "forced_occurrence_low_score"
                    if forced_selected_idx is None or forced_gate_rejected:
                        selected_w_idx, cv_seq_idx, resolve_meta = _resolve_cv_syllable_index(
                            target_clean,
                            romaji_syllables,
                            cv_seq_idx,
                            current_w_idx,
                            mapping_confidence=row_mapping_confidence,
                            max_jump_default=row_jump_default,
                            max_jump_high_conf=row_jump_high_conf,
                            high_conf_threshold=max(float(file_mapping_conf_th), 0.50),
                            return_meta=True,
                        )
                        row_jump_blocked = int(resolve_meta.get("jump_blocked", 0) or 0)
                        if row_jump_blocked and kr_mapping_debug_reason_logging:
                            log(
                                f"🛡️ {fname}: KR 매핑 전진 점프 차단 "
                                f"({int(resolve_meta.get('raw_chosen_idx', selected_w_idx)) + 1}"
                                f"->{int(resolve_meta.get('chosen_idx', selected_w_idx)) + 1}, {alias})"
                            )
                        if row_jump_blocked:
                            row_mapping_confidence = max(0.0, row_mapping_confidence - 0.18)
                            row_mapping_reason_code = "jump_blocked"
                    target_onset, target_vowel, _target_coda = _split_kr_syllable_parts(target_clean)
                    forced_cv_head_severe_mismatch = False
                    if (
                        forced_selected_idx is not None
                        and alias_type == "cv_head"
                        and target_vowel
                        and 0 <= selected_w_idx < len(romaji_syllables)
                    ):
                        _fc_on, forced_curr_vowel, _fc_coda = _split_kr_syllable_parts(romaji_syllables[selected_w_idx])
                        forced_cv_head_severe_mismatch = bool(forced_curr_vowel and forced_curr_vowel != target_vowel)
                    allow_exact_vowel_fix = (
                        _should_allow_kr_exact_vowel_fix(
                            file_format,
                            forced_selected_idx,
                            alias_type=alias_type,
                            severe_vowel_mismatch=forced_cv_head_severe_mismatch,
                        )
                        and float(row_mapping_confidence) >= float(file_mapping_conf_th)
                    )
                    if target_vowel and 0 <= selected_w_idx < len(romaji_syllables) and allow_exact_vowel_fix:
                        _curr_onset, curr_vowel, _curr_coda = _split_kr_syllable_parts(romaji_syllables[selected_w_idx])
                        need_exact_vowel_fix = bool(curr_vowel and curr_vowel != target_vowel)
                        if file_format == "cvvc" and forced_selected_idx is None and not need_exact_vowel_fix:
                            fixed_probe = _find_kr_cv_vowel_match_index(
                                target_clean, romaji_syllables, expected_cv_idx, search_back=1, search_fwd=4
                            )
                            if fixed_probe is not None and fixed_probe != selected_w_idx:
                                _probe_onset, probe_vowel, _probe_coda = _split_kr_syllable_parts(romaji_syllables[fixed_probe])
                                if probe_vowel == target_vowel:
                                    need_exact_vowel_fix = True
                        if need_exact_vowel_fix:
                            fixed_search_fwd = 4 if file_format == "cvvc" else 2
                            if file_format == "cvvc" and forced_selected_idx is not None and alias_type == "cv_head":
                                fixed_search_fwd = 1
                            fixed_idx = _find_kr_cv_vowel_match_index(
                                target_clean,
                                romaji_syllables,
                                expected_cv_idx,
                                search_back=1,
                                search_fwd=fixed_search_fwd,
                            )
                            if fixed_idx is not None and fixed_idx >= expected_cv_idx:
                                max_forward = int(max(0, row_jump_default))
                                if (
                                    float(row_mapping_confidence) >= max(float(file_mapping_conf_th), 0.50)
                                    and float(resolve_meta.get("best_score", 0.0) or 0.0) >= 84.0
                                ):
                                    max_forward = int(max(max_forward, row_jump_high_conf))
                                raw_fixed_idx = int(fixed_idx)
                                max_allowed_idx = int(expected_cv_idx + max_forward)
                                if fixed_idx > max_allowed_idx:
                                    fixed_idx = max_allowed_idx
                                    row_jump_blocked = 1
                                    row_mapping_confidence = max(0.0, row_mapping_confidence - 0.14)
                                    row_mapping_reason_code = "exact_vowel_fix_jump_blocked"
                                    if kr_mapping_debug_reason_logging:
                                        log(
                                            f"🛡️ {fname}: KR 모음 보정 점프 차단 "
                                            f"({raw_fixed_idx + 1}->{fixed_idx + 1}, {alias})"
                                        )
                                if fixed_idx != selected_w_idx and abs(fixed_idx - expected_cv_idx) <= 2:
                                    log(
                                        f"🧭 {fname}: CV 모음 불일치 보정 "
                                        f"{expected_cv_idx + 1}->{fixed_idx + 1} ({alias})"
                                    )
                                selected_w_idx = fixed_idx
                                cv_seq_idx = max(cv_seq_idx, selected_w_idx + 1)
                    current_w_idx = max(current_w_idx, selected_w_idx)
                    selected_w_idx, curr_phones, c_start, c_end, n_start, n_end = _prepare_cv_bounds_from_syllable(
                        syllables_info, selected_w_idx
                    )
                else:
                    current_w_idx, curr_phones, c_start, c_end, n_start, n_end = _prepare_vc_bounds_from_context(
                        syllables_info, current_w_idx
                    )
                cv_vowel_len = n_end - n_start
                is_vc_plosive_coda = False
                if is_vc:
                    (
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        is_vc_plosive_coda,
                    ) = _compute_kr_vc_timing(
                        alias,
                        alias_type,
                        file_format,
                        curr_phones,
                        current_w_idx,
                        syllables_info,
                        c_start,
                        c_end,
                        n_start,
                        n_end,
                        cv_anchor_by_idx,
                    )

                else:
                    cv_vowel_len = n_end - n_start
                    
                    if is_diph:
                        c_hint = curr_phones[0].mark if curr_phones else ""
                        alias_onset = _extract_alias_onset(alias)
                        offset, consonant, cutoff, pre, ovl = _compute_kr_cv_timing(
                            c_start,
                            c_end,
                            cv_vowel_len,
                            c_hint,
                            alias_onset,
                            True,
                            False,
                        )
                        
                    elif _looks_like_vv_alias(alias):
                        vv_direct = None
                        if file_format == "cvvc":
                            vv_direct = _compute_kr_cvvc_vv_timing_direct(
                                selected_w_idx, syllables_info, n_start, n_end
                            )
                        if vv_direct is not None:
                            offset, consonant, cutoff, pre, ovl = vv_direct
                        else:
                            boundary = c_end
                            v1_len = c_end - c_start
                            
                            offset_padding = 100
                            if v1_len < offset_padding:
                                offset_padding = max(v1_len * 0.8, 50)
                                
                            offset = boundary - offset_padding
                            pre = boundary - offset
                            ovl = adaptive_overlap(pre, "", mode='vv')
                            
                            added_cons = cv_vowel_len * 0.4
                            if added_cons < 60: added_cons = 60
                            consonant = pre + added_cons
                            cutoff = -(consonant + cv_vowel_len * 0.4)
                        
                    else:


                        first_phone_plosive = len(curr_phones) >= 2 and is_plosive_ipa(curr_phones[0].mark)
                        alias_consonant = re.match(r'^([^aeiouyw]+)', alias.lower())
                        roman_plosive = alias_consonant and is_plosive_roman(alias_consonant.group(1)) if alias_consonant else False
                        alias_onset = alias_consonant.group(1) if alias_consonant else ""
                        c_hint = curr_phones[0].mark if curr_phones else ""
                        is_plosive = first_phone_plosive or roman_plosive
                        offset, consonant, cutoff, pre, ovl = _compute_kr_cv_timing(
                            c_start,
                            c_end,
                            cv_vowel_len,
                            c_hint,
                            alias_onset,
                            False,
                            is_plosive,
                        )

                (
                    offset,
                    consonant,
                    cutoff,
                    pre,
                    ovl,
                    soft_off_shift,
                    soft_cut_shift,
                    cutoff_reduced,
                ) = _apply_post_timing_pipeline(
                    offset,
                    consonant,
                    cutoff,
                    pre,
                    ovl,
                    alias_type=alias_type,
                    alias_text=alias,
                    file_format=file_format,
                    mel_ctx_for_file=mel_ctx_for_file,
                    base_shape=base_shape,
                    ph_intervals=ph_intervals,
                    current_w_idx=selected_w_idx if not is_vc else current_w_idx,
                    syllables_info=syllables_info,
                    is_vc_plosive_coda=is_vc_plosive_coda,
                    enable_stabilize=True,
                    enable_cutoff_guard=True,
                    post_ctx=kr_post_ctx,
                )
                bridge_shift = 0.0
                if alias_type in {"vc", "vv"}:
                    prev_idx = None
                    next_idx = None
                    if alias_type == "vc":
                        prev_idx = current_w_idx
                        next_idx = current_w_idx + 1
                    elif selected_w_idx is not None and selected_w_idx >= 1:
                        prev_idx = selected_w_idx - 1
                        next_idx = selected_w_idx
                    prev_anchor = None
                    next_anchor = None
                    if prev_idx is not None:
                        prev_anchor = realized_cv_anchor_by_idx.get(prev_idx) or cv_anchor_by_idx.get(prev_idx)
                    if next_idx is not None:
                        next_anchor = realized_cv_anchor_by_idx.get(next_idx) or cv_anchor_by_idx.get(next_idx)
                    if prev_anchor is not None and next_anchor is not None:
                        pre_abs_before = float(offset + pre)
                        offset, consonant, cutoff, pre, ovl = _refine_kr_bridge_with_adjacent_cv(
                            offset,
                            consonant,
                            cutoff,
                            pre,
                            ovl,
                            alias_type=alias_type,
                            alias_text=alias,
                            prev_cv=prev_anchor,
                            next_cv=next_anchor,
                        )
                        bridge_shift = float((offset + pre) - pre_abs_before)
                anchor_abs = c_end
                next_onset_abs = n_start
                next_vowel_abs = n_end
                if alias_type == "vc":
                    anchor_abs = n_start
                    next_onset_abs = n_start
                    next_vowel_abs = n_end
                elif alias_type == "vv":
                    anchor_abs = c_end
                    next_onset_abs = n_start
                    next_vowel_abs = n_end
                offset, consonant, cutoff, pre, ovl = _apply_kr_anchor_lock(
                    fname=fname,
                    alias_text=alias,
                    format_type=file_format,
                    alias_type=alias_type,
                    offset=offset,
                    consonant=consonant,
                    cutoff=cutoff,
                    pre=pre,
                    ovl=ovl,
                    timeline_start_ms=timeline_start_ms,
                    timeline_end_ms=timeline_end_ms,
                    file_duration_ms=wav_duration_ms,
                    anchor_abs_ms=anchor_abs,
                    next_onset_abs_ms=next_onset_abs,
                    next_vowel_abs_ms=next_vowel_abs,
                    mapping_confidence=row_mapping_confidence,
                )
                if alias_type == "cv" and selected_w_idx is not None:
                    realized_cv_anchor_by_idx[selected_w_idx] = {
                        "offset": float(offset),
                        "pre": float(pre),
                        "ovl": float(ovl),
                        "cons": float(consonant),
                        "cutoff": float(cutoff),
                        "pre_abs": float(offset + pre),
                        "cons_abs": float(offset + consonant),
                        "onset_abs": float(c_start),
                        "vowel_start_abs": float(n_start),
                        "vowel_end_abs": float(n_end),
                        "vowel_len": max(12.0, float(n_end - n_start)),
                        "cons_gap": max(float(consonant - pre), 10.0),
                        "cut_gap": max(float(abs(cutoff) - consonant), 16.0),
                    }
                _log_post_timing_events(log, fname, alias, soft_off_shift, soft_cut_shift, cutoff_reduced)
                if alias_type in {"vc", "vv"} and abs(bridge_shift) > 6.0:
                    log(f"🧭 {fname}: KR 브리지 앵커 미세보정 ({bridge_shift:+.1f}ms) [{alias}]")

                _append_alias_rows(
                    final_lines,
                    real_wav_name,
                    alias,
                    offset,
                    consonant,
                    cutoff,
                    pre,
                    ovl,
                    generate_openutau=generate_openutau,
                    alias_suffix=alias_suffix,
                )

            processed += 1

        except Exception as e:
            err_msg = f"처리 실패 ({fname}): {e}"
            logger.error(err_msg)
            errors.append(err_msg)
            _record_unset_lines("file_exception", fname, lines)
            final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
            processed += 1

        if callback and total > 0 and (processed % 5 == 0 or processed == total):
            callback(f"OTO 생성 중... ({processed}/{total})")


    if gen_missing_vowels:
        log("누락된 단모음 에일리어스 자동 생성을 시작합니다...")
        vowels_list = ['a', 'e', 'i', 'o', 'u', 'eo', 'eu', 'ae', 'oe', 'wi', 'wa', 'we', 'weo', 'ya', 'ye', 'yo', 'yeo', 'yu', 'ui', 'eui']
        template_aliases = set()
        for g_lines in file_groups.values():
            for line in g_lines:
                parts = line.split('=')
                if len(parts) > 1:
                    alias = parts[1].split(',')[0].strip().lower()
                    template_aliases.add(alias)

        for tg_info in tg_entries:

            base_filename = os.path.splitext(tg_info['real_name'])[0].lower()
            

            base_filename = re.sub(r'long$', '', base_filename)
            norm_key_clean = re.sub(r'long$', '', tg_info['norm_key'])
            
            detected_vowel = None
            

            tokens = re.split(r'[-_ ]+', base_filename)
            for token in reversed(tokens):
                clean_token = re.sub(r'[^a-z]', '', token)
                if clean_token in vowels_list:
                    detected_vowel = clean_token
                    break
            

            if not detected_vowel and norm_key_clean in vowels_list:
                detected_vowel = norm_key_clean
                
            if detected_vowel:
                try:
                    tg = textgrid.TextGrid.fromFile(tg_info['path'])
                    w_tier = None
                    for t in tg:
                        if hasattr(t, 'name') and t.name == 'words':
                            w_tier = t
                            break

                    phone_tier = next((t for t in tg if isinstance(t, textgrid.IntervalTier) and t.name == 'phones'), None)
                    if not phone_tier: continue
                    intervals = [i for i in phone_tier if i.mark.strip() not in ['', 'sil', 'spn', 'pau']]
                    
                    if len(intervals) == 1:
                        vowel = intervals[0]
                        v_start = vowel.minTime * 1000
                        v_end = vowel.maxTime * 1000
                        v_len = v_end - v_start
                        
                        alias = detected_vowel
                        if alias not in template_aliases:
                            log(f"추가: 단모음 에일리어스 생성 -> {tg_info['real_name']} [{alias}]")
                            offset = v_start
                            pre = 0
                            ovl = 0
                            consonant = min(v_len * 0.25, 120)
                            cutoff = -(v_len * 0.8)
                            
                            offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
                            
                            _append_alias_rows(
                                final_lines,
                                tg_info['real_name'],
                                alias,
                                offset,
                                consonant,
                                cutoff,
                                pre,
                                ovl,
                                generate_openutau=generate_openutau,
                                alias_suffix=alias_suffix,
                            )
                except:
                    continue


    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            for line in final_lines:
                f.write(line + "\n")
        log(f"1차 생성 완료: OTO 파일 저장 -> {out_path}")
        wav_dir_for_profile = os.path.dirname(os.path.abspath(tg_folder.rstrip("\\/")))
        if kr_profile:
            changed = _apply_kr_profile_to_oto_file(
                out_path, wav_dir_for_profile, kr_profile, custom_map=custom_map
            )
            if changed > 0:
                log(f"[KR-Profile] 기준 프로파일 보정 적용: {changed} lines")
        try:
            from core.oto_ml_refiner import apply_oto_ml_to_oto_file

            ml_changed = apply_oto_ml_to_oto_file(
                "korean",
                out_path,
                tg_dir=tg_folder,
                wav_dir=wav_dir_for_profile,
                custom_phonemes_path=custom_phonemes_path,
                callback=log,
                enabled=enable_ml_correction,
                format_override=auto_gen_format,
            )
            if ml_changed > 0:
                log(f"[OTO-ML] 한국어 수치 보정 적용: {ml_changed} lines")
        except Exception as ml_e:
            log(f"[OTO-ML] 한국어 보정 스킵: {ml_e}")
        mel_changed = _apply_kr_mel_refine_to_oto_file(
            out_path, wav_dir_for_profile, custom_map=custom_map
        )
        if mel_changed > 0:
            log(f"[KR-Mel] 멜 에너지 기반 cutoff 보정 적용: {mel_changed} lines")
    except Exception as e:
        err = f"OTO 파일 저장 실패: {e}"
        logger.error(err)
        errors.append(err)

    if anchor_stats["anchor_locked_count"] > 0:
        log(
            "[AnchorLock] 요약: "
            f"anchor_locked_count={anchor_stats['anchor_locked_count']}, "
            f"cutoff_clamped_count={anchor_stats['cutoff_clamped_count']}, "
            f"vc_cutoff_leak_guard_count={anchor_stats['vc_cutoff_leak_guard_count']}"
        )
        log(f"[AnchorLock] 상세 로그: {anchor_log_path}")

    if cleanup_timing_jsonl:
        removed_count, failed_count = _cleanup_timing_anchor_jsonl_files(
            _anchor_log_dir,
            "timing_anchor_kr_",
        )
        if removed_count > 0:
            log(f"[AnchorLock] timing jsonl 자동 정리: {removed_count} files")
        if failed_count > 0:
            log(f"[AnchorLock] timing jsonl 정리 실패: {failed_count} files")

    _log_unset_summary()
    return processed, total, errors
