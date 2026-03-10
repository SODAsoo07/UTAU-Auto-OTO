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

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None
from core.lab_generator import load_custom_phonemes
from core.kr_oto_rules import (
    IPA_VOWELS,
    KR_CONSONANTS,
    KR_PLOSIVE_ONSETS,
    KR_SIBILANT_ONSETS,
    KR_SONORANT_CONSONANTS,
    KR_TENSE_CONSONANTS,
    KR_VOICED_ONSETS,
    KR_VOICELESS_ONSETS,
    KR_VOWELS,
    _cv_match_score,
    _detect_glottal_kind,
    _extract_alias_onset,
    _extract_vc_right_token,
    _extract_kr_cv_alias_token,
    _canonicalize_kr_coda,
    _kr_cv_kernel,
    _looks_like_vv_alias,
    _normalize_cv_match_token,
    _split_kr_syllable_parts,
    classify_alias,
    detect_alias_format,
    find_vowel_phone,
    is_glide,
    is_plosive_ipa,
    is_plosive_roman,
    normalize_ipa_mark,
)
from core.kr_oto_mapping import (
    _build_kr_cvvc_occurrence_map,
    _build_kr_cvvc_vv_occurrence_map,
    _clamp_kr_cv_index_to_order,
    _compute_kr_glide_mismatch_ratio,
    _extract_cv_targets_from_lines,
    _extract_kr_cv_targets_from_filename,
    _find_kr_cv_vowel_match_index,
    _is_kr_order_locked_cv_format,
    _resolve_kr_cvvc_occurrence_index,
    _resolve_kr_cvvc_vv_index,
    _score_kr_syllable_mapping,
    _should_allow_kr_exact_vowel_fix,
    _should_prefer_alias_based_syllables,
)
from core.kr_oto_bridge import (
    _compute_kr_cvvc_vc_timing_direct,
    _compute_vc_from_adjacent_cv,
    _refine_kr_bridge_with_adjacent_cv,
    _recenter_kr_params_around_pre,
)
from core.kr_auto_alias_setup import build_kr_auto_file_groups
from core.kr_alias_family_stage import (
    build_kr_alias_family_state,
    try_handle_kr_breath_tail_alias,
    try_handle_kr_glottal_alias,
)
from core.kr_file_stage import build_kr_postprocess_context, try_handle_kr_single_vowel_file
from core.kr_loop_prep import prepare_kr_loop_state
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
from core.kr_oto_vv import (
    _compute_kr_cvvc_vv_timing_direct,
    _compute_kr_noninitial_vowel_timing,
    _compute_kr_vv_timing_from_vowel_bounds,
)
from core.kr_oto_postprocess import (
    KrPostprocessContext,
    guard_kr_vc_cutoff_to_next_segment as _guard_kr_vc_cutoff_to_next_segment_core,
    log_post_timing_events as _log_post_timing_events_core,
)
from core.kr_oto_file_ops import (
    _apply_kr_bridge_coherence_to_oto_file,
    _apply_kr_profile_to_oto_file,
    _extract_base_timing_shape,
    _parse_oto_line_profile,
    _read_text_with_fallback,
    _resolve_kr_reference_dirs,
    _retarget_kr_bridge_to_next_cv,
    apply_kr_autotune_profile_to_oto,
    load_kr_autotune_profile,
    save_kr_autotune_profile,
    train_kr_autotune_profile,
)
from core.kr_oto_file_finalize import KrPostFilePipelineContext, run_kr_post_file_pipeline
from core.kr_generator_setup import prepare_kr_profile_setup
from core.generator_finish import (
    GeneratorFinishContext,
    finalize_generator_finish,
    write_oto_lines,
)
from core.alignment_ingest import build_kr_alignment_ingest
from core.kr_candidate_selection_v2 import select_kr_syllable_source
from core.oto_diagnostics import SkippedEntryCollector
from core.oto_diagnostics_adapter_v2 import GeneratorDiagnosticsAdapter
from core.file_prepare import load_named_tiers, prepare_file_context
from core.timing_anchor_profiles import is_anchor_lock_enabled
from core.timing_anchor_runtime import append_timing_anchor_log
from core.anchor_lock_adapter_v2 import apply_language_anchor_lock
from core.textio_utils import load_template_oto_lines
from core.oto_profile_presets import get_kr_profile_preset
from core.oto_normalization import canonicalize_alias_for_matching, normalize_wav_key
from core.format_type_utils import normalize_format_type
from core.kr_mapping_v2 import (
    build_kr_cv_anchor_plan as _build_kr_cv_anchor_plan_v2,
    collect_kr_syllable_activity_metrics as _collect_kr_syllable_activity_metrics_v2,
    is_kr_cv_syllable_active as _is_kr_cv_syllable_active_v2,
    resolve_kr_planned_cv_index as _resolve_kr_planned_cv_index_v2,
)
from core.kr_mapping_scoring_v2 import resolve_cv_syllable_index as _resolve_cv_syllable_index_v2
from core.sinsy_label_ingest import build_sinsy_guided_anchor_plan, load_sinsy_label_entries
from core.kr_timing_v2 import (
    build_realized_cv_anchor as _build_realized_kr_cv_anchor_v2,
    extract_vcv_anchor_points as _extract_vcv_anchor_points_v2,
    prepare_vcv_syllable_timing as _prepare_vcv_syllable_timing_v2,
)
from core.kr_anchor_lock_v2 import (
    build_kr_anchor_lock_log_record as _build_kr_anchor_lock_log_record_v2,
    build_kr_anchor_lock_stats_delta as _build_kr_anchor_lock_stats_delta_v2,
    resolve_kr_anchor_targets as _resolve_kr_anchor_targets_v2,
)
from core.kr_runtime_feedback_v2 import (
    build_kr_bridge_adjust_message as _build_kr_bridge_adjust_message_v2,
    build_kr_cv_head_guard_messages as _build_kr_cv_head_guard_messages_v2,
)
from core.kr_row_runtime_v2 import (
    maybe_build_kr_realized_cv_anchor_record as _maybe_build_kr_realized_cv_anchor_record_v2,
    prepare_kr_cv_head_anchor_context as _prepare_kr_cv_head_anchor_context_v2,
)
from core.kr_row_finalize_v2 import finalize_kr_row as _finalize_kr_row_v2
from core.kr_cv_head_row_v2 import run_kr_cv_head_row as _run_kr_cv_head_row_v2
from core.kr_vcv_row_v2 import run_kr_vcv_row as _run_kr_vcv_row_v2
from core.kr_general_row_v2 import run_kr_general_row as _run_kr_general_row_v2
from core.kr_mapping_select_v2 import (
    resolve_kr_cv_head_forced_index as _resolve_kr_cv_head_forced_index_v2,
    select_kr_general_cv_index as _select_kr_general_cv_index_v2,
    select_kr_vcv_index as _select_kr_vcv_index_v2,
)
from core.oto_row_output_v2 import prepare_oto_alias_rows as _prepare_oto_alias_rows_v2
from core.oto_anchor_graph import build_adjacent_anchor_graph, resolve_bridge_anchor_pair
from core.oto_mapping_policy import resolve_plan_policy
from core.oto_row_abstain import decide_cv_row_abstain
from core.oto_row_policy import apply_row_confidence_penalty
from core.oto_runtime_policy import resolve_runtime_mapping_policy

logger = logging.getLogger(__name__)

__all__ = [
    "generate_oto",
    "validate_oto_params",
    "KR_PLOSIVE_ONSETS",
    "KR_SIBILANT_ONSETS",
    "KR_SONORANT_CONSONANTS",
    "KR_TENSE_CONSONANTS",
    "KR_VOICED_ONSETS",
    "KR_VOICELESS_ONSETS",
    "_compute_kr_cvvc_vc_timing_direct",
    "_compute_vc_from_adjacent_cv",
    "_select_kr_cv_onset_slice",
    "_apply_kr_bridge_coherence_to_oto_file",
    "_apply_kr_profile_to_oto_file",
    "_retarget_kr_bridge_to_next_cv",
    "apply_kr_autotune_profile_to_oto",
    "load_kr_autotune_profile",
    "save_kr_autotune_profile",
    "train_kr_autotune_profile",
]

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
# - CV/CVC: 정보량이 상대적으로 단순해 과도한 저신뢰 판정을 완화
# - VC_ONLY/VV_ONLY: CV 정렬 점프 로직 영향이 거의 없어 완화값 사용
KR_MAPPING_CONF_THRESHOLD_BY_FORMAT = {
    "cv": 0.58,
    "cvvc": 0.60,
    "vcv": 0.62,
    "cvc": 0.58,
    "cv_simple": 0.58,
    "mono": 0.58,
    "vc_only": 0.56,
    "vv_only": 0.56,
    "default": 0.60,
}


def _resolve_kr_mapping_conf_threshold(file_format, override_threshold=None, phone_quality_score=None):
    """포맷별 기본 매핑 신뢰도 임계값을 반환합니다.

    phone_quality_score가 제공되면 phone tier 품질에 따라 동적으로 조정합니다:
    - 품질 < 0.4 -> 임계값을 0.05 낮춤 (과도한 저신뢰 판정 완화)
    - 품질 > 0.8 -> 임계값을 0.03 올림 (고품질 데이터에서 더 엄격)
    """
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
    threshold = float(base)

    # phone tier 품질에 따른 동적 조정
    if phone_quality_score is not None:
        try:
            pq = float(phone_quality_score)
            if pq < 0.4:
                threshold = max(0.40, threshold - 0.05)
            elif pq > 0.8:
                threshold = min(0.85, threshold + 0.03)
        except (TypeError, ValueError):
            pass

    return threshold


def normalize_key(name):
    return normalize_wav_key(name)


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
    hard = {'g', 'd', 'b', 'c', 'j'}
    tense = {'kk', 'tt', 'pp', 'gg', 'dd', 'bb', 'ss', 'jj'}  # 경음: VOT 짧음
    aspirate = {'k', 't', 'p', 'ch'}  # 격음: 기식 길음
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

    if hint in tense:
        ratio -= 0.20  # 경음: VOT가 짧아 overlap을 더 줄임
    elif hint in hard:
        ratio -= 0.14
    elif hint in aspirate:
        ratio += 0.02  # 격음: 기식이 길어 overlap을 약간 넓힘
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


def _prepare_vcv_syllable_timing(
    syllables_info,
    current_w_idx,
    cv_seq_idx,
    diphthong_cv_consonant_ratio,
    *,
    forced_w_idx=None,
):
    """VCV 계산에 필요한 음절 인덱스 갱신과 기본 타이밍을 산출합니다."""
    return _prepare_vcv_syllable_timing_v2(
        syllables_info,
        current_w_idx,
        cv_seq_idx,
        diphthong_cv_consonant_ratio,
        forced_w_idx=forced_w_idx,
        prepare_cv_bounds_fn=_prepare_cv_bounds_from_syllable,
        adaptive_overlap_fn=adaptive_overlap,
        validate_fn=validate_oto_params,
    )


def _fit_oto_to_wav_duration(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    wav_duration_ms,
    *,
    alias_type="",
    validate_fn=None,
):
    """
    생성 파라미터를 WAV 길이 경계 안으로 강제 보정합니다.
    - offset >= 0
    - offset + |cutoff| <= wav_duration
    - ovl <= pre <= consonant < |cutoff| (가능한 범위에서 유지)
    """
    if validate_fn is None:
        validate_fn = validate_oto_params

    offset, consonant, cutoff, pre, ovl = validate_fn(
        offset, consonant, cutoff, pre, ovl, alias_type=alias_type
    )

    try:
        dur = float(wav_duration_ms or 0.0)
    except Exception:
        dur = 0.0
    if dur != dur or dur == float("inf") or dur == float("-inf"):
        dur = 0.0
    if dur <= 0.0:
        return offset, consonant, cutoff, pre, ovl, False

    changed = False
    margin = 1.0
    min_tail = 2.0
    max_offset = max(0.0, dur - min_tail)
    if offset < 0.0:
        offset = 0.0
        changed = True
    if offset > max_offset:
        offset = max_offset
        changed = True

    available = max(min_tail, dur - offset - margin)
    pre = max(0.0, min(float(pre), max(0.0, available - 1.2)))
    cons_hi = max(0.8, available - 0.6)
    consonant = max(pre + 0.4, min(float(consonant), cons_hi))
    cut_abs = abs(float(cutoff))
    cut_abs = max(consonant + 0.3, min(cut_abs, available))
    cutoff = -cut_abs
    ovl = max(0.0, min(float(ovl), max(pre - 0.1, 0.0)))

    end_abs = offset + abs(cutoff)
    if end_abs > dur:
        cut_abs = max(0.6, dur - offset - 0.1)
        if consonant >= cut_abs:
            consonant = max(0.2, cut_abs - 0.2)
        if pre > consonant:
            pre = max(0.0, consonant - 0.1)
        if ovl > pre:
            ovl = max(0.0, pre - 0.05)
        cutoff = -cut_abs
        changed = True

    end_abs2 = offset + abs(cutoff)
    if end_abs2 > dur + 1e-6 or offset > dur + 1e-6:
        # 극단 케이스 최종 안전망
        offset = max(0.0, min(offset, max(0.0, dur - 1.5)))
        cut_abs = max(0.8, min(abs(cutoff), max(0.8, dur - offset - 0.1)))
        consonant = max(0.2, min(consonant, max(0.2, cut_abs - 0.2)))
        pre = max(0.0, min(pre, max(0.0, consonant - 0.1)))
        ovl = max(0.0, min(ovl, max(0.0, pre - 0.05)))
        cutoff = -cut_abs
        changed = True

    if changed:
        offset = float(offset)
        consonant = float(consonant)
        cutoff = float(cutoff)
        pre = float(pre)
        ovl = float(ovl)
    return offset, consonant, cutoff, pre, ovl, changed


def _build_alias_rows(
    real_wav_name,
    alias,
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    generate_openutau=False,
    alias_suffix="",
    alias_type="",
    wav_duration_ms=0.0,
    validate_fn=None,
):
    """에일리어스(및 OpenUtau 변형)를 OTO 라인 문자열로 변환합니다."""
    _params, rows = _prepare_oto_alias_rows_v2(
        real_wav_name,
        alias,
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        generate_openutau=generate_openutau,
        generate_aliases_fn=generate_openutau_aliases,
        alias_transform_fn=lambda alias_item: apply_alias_suffix(alias_item, alias_suffix),
        pre_write_adjust_fn=lambda off, cons, cut, preu, ov: _fit_oto_to_wav_duration(
            off,
            cons,
            cut,
            preu,
            ov,
            wav_duration_ms,
            alias_type=alias_type,
            validate_fn=validate_fn,
        )[:5],
    )
    return rows


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
    alias_type="",
    wav_duration_ms=0.0,
    validate_fn=None,
):
    """에일리어스(및 OpenUtau 변형)를 OTO 라인으로 누적합니다."""
    final_lines.extend(
        _build_alias_rows(
            real_wav_name,
            alias,
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            generate_openutau=generate_openutau,
            alias_suffix=alias_suffix,
            alias_type=alias_type,
            wav_duration_ms=wav_duration_ms,
            validate_fn=validate_fn,
        )
    )


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
    return _resolve_cv_syllable_index_v2(
        target_clean,
        romaji_syllables,
        cv_seq_idx,
        current_w_idx,
        mapping_confidence=mapping_confidence,
        max_jump_default=max_jump_default,
        max_jump_high_conf=max_jump_high_conf,
        high_conf_threshold=high_conf_threshold,
        return_meta=return_meta,
    )


def _remap_kr_forced_cv_index(target_clean, romaji_syllables, expected_idx):
    """강제 occurrence 인덱스가 유효하지 않을 때 사용 가능한 인덱스로 재매핑합니다."""
    if not target_clean or not romaji_syllables:
        return None

    n = len(romaji_syllables)
    if n <= 0:
        return None
    e = max(0, min(int(expected_idx), n - 1))
    target_norm = _normalize_cv_match_token(target_clean)

    if target_norm:
        for i in range(e, n):
            if _normalize_cv_match_token(romaji_syllables[i]) == target_norm:
                return i
        for i in range(e - 1, -1, -1):
            if _normalize_cv_match_token(romaji_syllables[i]) == target_norm:
                return i

    return _find_kr_cv_vowel_match_index(
        target_clean,
        romaji_syllables,
        e,
        search_back=2,
        search_fwd=6,
    )


def _kr_syllable_activity_metrics(syl_info):
    return _collect_kr_syllable_activity_metrics_v2(syl_info)


def _is_kr_cv_syllable_active(syl_info, *, require_vowel=True, min_active_ms=16.0, min_vowel_ms=10.0):
    return _is_kr_cv_syllable_active_v2(
        syl_info,
        require_vowel=require_vowel,
        min_active_ms=min_active_ms,
        min_vowel_ms=min_vowel_ms,
    )


def _build_kr_planned_cv_indices(expected_tokens, syllables_info):
    plan = _build_kr_cv_anchor_plan_v2(expected_tokens, syllables_info)
    return plan.get("indices")


def _resolve_kr_planned_cv_index(planned_indices, expected_seq_idx, target_clean, syllables_info, *, alias_type="cv"):
    return _resolve_kr_planned_cv_index_v2(
        planned_indices,
        expected_seq_idx,
        target_clean,
        syllables_info,
        alias_type=alias_type,
    )


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


def validate_oto_params(offset, consonant, cutoff, pre, ovl, alias_type=""):
    """UTAU OTO 파라미터를 유효 범위로 보정합니다.

    UTAU 필수 순서 제약: ovl < pre <= consonant < |cutoff|
    alias_type에 따라 파라미터 간 최소 간격을 차등 적용합니다.

    Args:
        offset: 파일 앞부분부터의 시작 위치 (ms, >= 0)
        consonant: 고정자음부 (오프셋 기준 상대, 스트레치 불가 구간)
        cutoff: 컷오프 (음수, 오프셋 기준 상대, 이후 소리 잘림)
        pre: 선행발음 (오프셋 기준 상대, 자음->모음 전이점)
        ovl: 오버랩 (오프셋 기준 상대, 앞 노트와 블렌딩)
        alias_type: 에일리어스 타입 (cv, cv_head, vc, vv, vcv 등, 선택적)
    """
    a_type = str(alias_type or "").strip().lower()

    # --- alias_type별 최소 간격 테이블 ---
    _MIN_CONS_GAP = {
        "cv": 20.0, "cv_head": 20.0, "vc": 10.0,
        "vv": 16.0, "vcv": 18.0, "mono": 12.0, "br": 8.0,
    }
    _MIN_CUT_GAP = {
        "cv": 14.0, "cv_head": 14.0, "vc": 8.0,
        "vv": 12.0, "vcv": 12.0, "mono": 10.0, "br": 6.0,
    }
    min_cons_gap = _MIN_CONS_GAP.get(a_type, 14.0) if a_type else 30.0
    min_cut_gap = _MIN_CUT_GAP.get(a_type, 12.0) if a_type else 50.0

    # --- 기본 하한 ---
    if offset < 0:
        offset = 0
    if pre < 0:
        pre = 0
    if ovl < 0:
        ovl = 0
    if consonant < 0:
        consonant = 0

    # --- ovl < pre 강제 ---
    if ovl > pre:
        ovl = pre * 0.75

    # --- pre <= consonant 강제 (alias_type별 최소 간격) ---
    if consonant < pre + min_cons_gap:
        consonant = pre + min_cons_gap

    # --- consonant < |cutoff| 강제 (alias_type별 최소 간격) ---
    cutoff_abs = abs(cutoff)
    if cutoff_abs <= consonant + min_cut_gap:
        cutoff_abs = consonant + min_cut_gap
    cutoff = -cutoff_abs

    # --- 최종 순서 검증 (안전망) ---
    if ovl >= pre:
        ovl = max(0.0, pre - 2.0)
    if consonant < pre:
        consonant = pre + min_cons_gap
    cutoff_abs = abs(cutoff)
    if cutoff_abs <= consonant:
        cutoff_abs = consonant + min_cut_gap
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

    target_offset = max(float(offset) + delta, 0.0)
    if abs(delta) > 30.0:
        # 큰 거리 snap은 과교정을 유발할 수 있어 블렌딩으로 완화한다.
        offset = _blend(float(offset), target_offset, 0.45)
    else:
        # 짧은 거리 보정은 기존처럼 즉시 snap한다.
        offset = target_offset
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
    # onset 보조 탐지: 에너지 이동평균 + 1차 기울기로 유효 onset 후보를 만든다.
    if len(en) >= 5:
        kernel = np.ones(5, dtype=np.float64) / 5.0
        en_ma = np.convolve(en, kernel, mode="same")
    else:
        en_ma = np.asarray(en, dtype=np.float64)
    en_slope = np.diff(en_ma, prepend=float(en_ma[0]) if len(en_ma) else 0.0)
    onset_mask = (en_slope > 0.015) & (en_ma > 0.12) & (db_arr > (db_sil_th - 2.0))

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
        _is_kr_order_locked_cv_format(file_format) and alias_type in {"cv", "cv_head"}
    )
    if not skip_offset_soft_guard and not low_energy_voiced:
        off_silent = bool(silence_mask[off_idx])
        pre_sound = bool(sound_mask[pre_idx] or (en[pre_idx] > 0.20))
        if off_silent and pre_sound:
            lo = max(0, pre_idx - 120)
            sound_start_idx = None
            onset_seg = onset_mask[lo:pre_idx + 1]
            if np.any(onset_seg):
                # onset 후보가 있으면 기존 sound 시작점보다 우선 사용한다.
                rel = int(np.where(onset_seg)[0][0])
                sound_start_idx = lo + rel
            else:
                seg = sound_mask[lo:pre_idx + 1]
                if np.any(seg):
                    rel = int(np.where(seg)[0][0])
                    sound_start_idx = lo + rel
            if sound_start_idx is not None:
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


def _default_kr_profile_cache_path(format_type="general"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    profile_dir = os.path.join(base_dir, "assets", "profiles")
    os.makedirs(profile_dir, exist_ok=True)
    fmt = normalize_format_type("korean", format_type) or "general"
    return os.path.join(profile_dir, f"kr_oto_reference_profile_{fmt}.json")


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

        wav_index = {}
        try:
            for dirpath, _dirnames, filenames in os.walk(ref_dir):
                for fn in filenames:
                    if fn.lower().endswith(".wav"):
                        wav_index[normalize_key(fn)] = os.path.join(dirpath, fn)
        except Exception:
            wav_index = {}

        rows_by_wav = {}
        for raw in _read_text_with_fallback(oto_path).splitlines():
            row = _parse_oto_line_profile(raw)
            if not row:
                continue
            rows_by_wav.setdefault(row["wav"], []).append(row)

        for wav_name, rows in rows_by_wav.items():
            wav_path = _find_wav_path_for_name(wav_name, ref_dir, wav_index)
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
    return canonicalize_alias_for_matching("korean", alias)


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
    callback=None,
    ml_policy="",
    runtime_report=None,
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
    use_sinsy_labels = bool(params.get("USE_SINSY_LABELS", False)) if params else False
    sinsy_label_path = str(params.get("SINSY_LABEL_PATH", "") or "").strip() if params else ""


    if auto_format:
        from core.format_type_utils import normalize_auto_format_value

        normalized_auto_format = normalize_auto_format_value("korean", auto_format)
        if normalized_auto_format:
            fallback_format = normalized_auto_format

    try:
        from core.format_type_utils import normalize_format_type

        auto_gen_format = normalize_format_type("korean", fallback_format or "cvvc") or "cvvc"
    except Exception:
        auto_gen_format = (fallback_format or "cvvc").strip().lower()

    if auto_gen_format not in {"cv", "cvc", "cvvc", "vcv"}:
        msg = (
            f"⚠ 자동 에일리어스 생성은 현재 CV/연단음, CVC, CVVC, VCV만 지원합니다. "
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
    env_use_sinsy = str(os.environ.get("UTOA_USE_SINSY_LABELS", "")).strip().lower()
    if env_use_sinsy in {"0", "false", "off", "no"}:
        use_sinsy_labels = False
    elif env_use_sinsy in {"1", "true", "on", "yes"}:
        use_sinsy_labels = True
    env_sinsy_path = str(os.environ.get("UTOA_SINSY_LABEL_PATH", "")).strip()
    if env_sinsy_path:
        sinsy_label_path = env_sinsy_path
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
    skipped_entries = SkippedEntryCollector()
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
    diagnostics = GeneratorDiagnosticsAdapter(
        skipped_collector=skipped_entries,
        log_fn=log,
    )
    _record_unset = diagnostics.record_unset
    _record_unset_lines = diagnostics.record_unset_lines
    _log_unset_summary = diagnostics.log_unset_summary

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
        def _get_profile(lang, fmt, alias_kind):
            from core.timing_anchor_profiles import get_anchor_profile

            return get_anchor_profile(lang, fmt, alias_kind, mode="rhythm_stable")

        def _apply_stats_delta(alias_kind, applied_rules):
            delta = _build_kr_anchor_lock_stats_delta_v2(alias_kind, applied_rules)
            for key, value in delta.items():
                if value:
                    anchor_stats[key] += int(value)

        def _build_log_record(fmt, alias_kind, before, result):
            return _build_kr_anchor_lock_log_record_v2(
                fname=fname,
                alias_text=alias_text,
                format_type=fmt,
                alias_type=alias_kind,
                lite=bool(lite),
                before=before,
                result=result,
            )

        out = apply_language_anchor_lock(
            language="korean",
            format_type=format_type,
            alias_type=alias_type,
            before=(offset, consonant, cutoff, pre, ovl),
            file_duration_ms=file_duration_ms,
            timeline_start_ms=timeline_start_ms,
            timeline_end_ms=timeline_end_ms,
            anchor_abs_ms=anchor_abs_ms,
            next_onset_abs_ms=next_onset_abs_ms,
            next_vowel_abs_ms=next_vowel_abs_ms,
            mapping_confidence=mapping_confidence,
            validate_fn=lambda o, c, cut, p, v, _atype=alias_type: validate_oto_params(
                o,
                c,
                cut,
                p,
                v,
                alias_type=_atype,
            ),
            lite=bool(lite),
            is_enabled_fn=is_anchor_lock_enabled,
            get_profile_fn=_get_profile,
            retune_profile_fn=lambda profile: (
                _retune_kr_vcv_anchor_profile(profile, alias_text, alias_type)
                if str(format_type or "").strip().lower() == "vcv"
                else profile
            ),
            apply_stats_delta_fn=_apply_stats_delta,
            build_log_record_fn=_build_log_record,
            append_log_fn=append_timing_anchor_log,
            log_path=anchor_log_path,
        )
        if str(alias_type or "").strip().lower() == "vc" and next_onset_abs_ms is not None:
            coda = _canonicalize_kr_coda(_extract_vc_right_token(alias_text))
            is_stoplike = coda in {"k", "t", "p", "h"}
            if is_stoplike:
                o, c, cut, p, v = [float(x) for x in out]
                next_onset_rel = max(float(next_onset_abs_ms) - o, p + 10.0)
                c_cap = next_onset_rel - 7.0
                c = min(c, c_cap)
                c = max(c, p + 8.0)
                cut_abs = abs(cut)
                cut_cap = next_onset_rel - 1.0
                cut_floor = c + 6.0
                if cut_floor > cut_cap:
                    c = max(p + 8.0, cut_cap - 6.0)
                    cut_floor = c + 6.0
                if cut_floor > cut_cap:
                    cut_cap = cut_floor + 0.8
                cut_abs = min(cut_abs, cut_cap)
                cut_abs = max(cut_abs, cut_floor)
                out = validate_oto_params(o, c, -cut_abs, p, v, alias_type="vc")
        return out

    kr_profile_setup = prepare_kr_profile_setup(
        fallback_format=fallback_format,
        auto_gen_format=auto_gen_format,
        kr_anchor_profile_path=kr_anchor_profile_path,
        log_fn=log,
        default_cache_path_fn=_default_kr_profile_cache_path,
        load_profile_fn=_load_kr_reference_profile,
        resolve_reference_dirs_fn=_resolve_kr_reference_dirs,
        build_profile_fn=lambda ref_dirs: _build_kr_reference_profile_from_dirs(ref_dirs, custom_map=None),
        save_profile_fn=_save_kr_reference_profile,
        get_preset_fn=get_kr_profile_preset,
    )
    kr_profile = kr_profile_setup.profile

    if tpl_path and not os.path.exists(tpl_path):
        log(f"⚠ 템플릿 파일을 찾을 수 없습니다: {tpl_path}")
        log(f"⚡ OpenUtau 호환 {auto_gen_format.upper()} 자동 에일리어스 생성으로 전환합니다.")
        tpl_path = ""

    DIPHTHONG_CV_CONSONANT_RATIO = params.get('DIPHTHONG_CV_CONSONANT_RATIO', 0.6) if params else 0.6


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
        try:
            from core.lab_generator import decompose_hangul_to_roman
        except ImportError:
            decompose_hangul_to_roman = lambda ch: [ch]

        file_groups = build_kr_auto_file_groups(
            tg_entries=tg_entries,
            auto_gen_format=auto_gen_format,
            log_fn=log,
            load_textgrid_fn=textgrid.TextGrid.fromFile,
            decompose_hangul_to_roman_fn=decompose_hangul_to_roman,
            split_syllable_parts_fn=_split_kr_syllable_parts,
            kr_vowels=set(KR_VOWELS),
            kr_consonants=set(KR_CONSONANTS),
        )

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
        file_ctx = prepare_file_context(
            fname=fname,
            lines=lines,
            resolve_tg_info_fn=_resolve_tg_info,
            wav_root_for_signal=wav_root_for_signal,
            wav_index_for_signal=wav_index_for_signal,
            mel_cache_for_signal=mel_cache_for_signal,
            find_wav_path_fn=_find_wav_path_for_name,
            read_wav_fn=_read_wav_mono_np,
            mel_envelope_fn=_mel_envelope,
            wav_duration_fn=_wav_duration_ms,
        )
        file_ctx.sinsy_label_entries = []
        file_ctx.sinsy_label_path = ""
        if use_sinsy_labels:
            try:
                file_ctx.sinsy_label_entries = load_sinsy_label_entries(
                    tg_path=file_ctx.tg_path,
                    real_wav_name=file_ctx.real_wav_name,
                    explicit_path=sinsy_label_path,
                )
                if file_ctx.sinsy_label_entries:
                    file_ctx.sinsy_label_path = sinsy_label_path or ""
            except Exception:
                file_ctx.sinsy_label_entries = []
                file_ctx.sinsy_label_path = ""

        if file_ctx.status == "textgrid_missing":
            log(f"경고: {fname}: TextGrid를 찾을 수 없어 원본 라인을 유지합니다.")
            _record_unset_lines("textgrid_missing", fname, lines)
            final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
            processed += 1
            continue

        file_ctx = load_named_tiers(
            file_ctx,
            load_textgrid_fn=textgrid.TextGrid.fromFile,
            tier_predicate=lambda tier: isinstance(tier, textgrid.IntervalTier),
        )
        if file_ctx.status == "textgrid_load_failed":
            log(f"경고: {fname}: TextGrid 로드 실패로 원본 라인을 유지합니다. ({file_ctx.error_message})")
            _record_unset_lines("textgrid_load_failed", fname, lines)
            final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
            processed += 1
            continue
        if file_ctx.status == "tier_missing":
            log(f"경고: {fname}: phones tier가 없어 원본 라인을 유지합니다.")
            _record_unset_lines("tier_missing", fname, lines)
            final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
            processed += 1
            continue

        real_wav_name = file_ctx.real_wav_name
        mel_ctx_for_file = file_ctx.mel_ctx_for_file
        wav_duration_ms = float(file_ctx.wav_duration_ms or 0.0)
        tg = file_ctx.tg
        phone_tier = file_ctx.phone_tier
        word_tier = file_ctx.word_tier

        try:
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
            loop_prep = prepare_kr_loop_state(
                lines=lines,
                real_wav_name=real_wav_name,
                phone_tier=phone_tier,
                word_tier=word_tier,
                wav_duration_ms=wav_duration_ms,
                custom_map=custom_map,
                kr_mapping_words_fallback_enabled=kr_mapping_words_fallback_enabled,
                kr_mapping_spn_ratio_threshold=kr_mapping_spn_ratio_threshold,
                kr_mapping_min_vowel_phone_ratio=kr_mapping_min_vowel_phone_ratio,
                kr_mapping_confidence_threshold=kr_mapping_confidence_threshold,
                debug_log_fn=log,
                debug_reason_logging=kr_mapping_debug_reason_logging,
                decompose_hangul_to_roman_fn=decompose_hangul_to_roman,
                synthesize_word_phones_fn=_synthesize_kr_word_phones,
                detect_alias_format_fn=detect_alias_format,
                extract_cv_targets_from_lines_fn=_extract_cv_targets_from_lines,
                extract_cv_targets_from_filename_fn=_extract_kr_cv_targets_from_filename,
                collect_phone_quality_fn=_collect_kr_phone_tier_quality,
                resolve_mapping_conf_threshold_fn=_resolve_kr_mapping_conf_threshold,
                preferred_format=auto_gen_format,
            )
            if loop_prep.status == "empty_intervals":
                log(f"경고: {fname}: 유효한 음소 구간이 없어 원본 라인을 유지합니다.")
                _record_unset_lines("mapping_failed_empty_intervals", fname, lines)
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue
            if loop_prep.status == "no_valid_alias":
                log(f"경고: {fname}: 유효한 에일리어스가 없어 원본 라인을 유지합니다.")
                _record_unset_lines("no_valid_alias", fname, lines)
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue

            alignment_ingest = build_kr_alignment_ingest(file_ctx, loop_prep)
            ph_intervals_all = alignment_ingest.phones_all
            ph_intervals = alignment_ingest.phones
            wd_intervals = alignment_ingest.words
            wav_duration_ms = float(alignment_ingest.wav_duration_ms or 0.0)
            timeline_start_ms = float(alignment_ingest.timeline_meta.get("timeline_start_ms", 0.0) or 0.0)
            timeline_end_ms = float(alignment_ingest.timeline_meta.get("timeline_end_ms", 0.0) or 0.0)
            file_format = str(alignment_ingest.extra.get("file_format") or "")
            file_mapping_conf_th = float(alignment_ingest.extra.get("file_mapping_conf_th", 0.0) or 0.0)
            filename_cv_targets = list(alignment_ingest.extra.get("filename_cv_targets") or [])
            targets_for_build = list(alignment_ingest.extra.get("targets_for_build") or [])
            sinsy_label_entries = list(alignment_ingest.extra.get("sinsy_label_entries") or [])
            phone_quality = alignment_ingest.phone_quality
            low_quality_reasons = alignment_ingest.low_quality_reasons
            low_phone_quality = alignment_ingest.low_phone_quality
            force_words_phone_fill = bool(alignment_ingest.extra.get("force_words_phone_fill"))
            textgrid_trust_score = float(alignment_ingest.textgrid_trust_score or 0.0)
            textgrid_trust_tier = str(alignment_ingest.textgrid_trust_tier or "low")
            prefer_filename_sequence = bool(alignment_ingest.prefer_filename_sequence)
            spn_ratio = float(phone_quality.get("spn_ratio_in_phone_tier", 0.0))

            if try_handle_kr_single_vowel_file(
                fname=fname,
                lines=lines,
                real_wav_name=real_wav_name,
                ph_intervals=ph_intervals,
                wd_intervals=wd_intervals,
                final_lines=final_lines,
                log_fn=log,
                record_unset_fn=_record_unset,
                validate_fn=validate_oto_params,
                append_alias_rows_fn=_append_alias_rows,
                apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line,
                generate_openutau=generate_openutau,
                alias_suffix=alias_suffix,
            ):
                processed += 1
                continue
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
            kr_source_pick = select_kr_syllable_source(
                file_format=file_format,
                prefer_filename_sequence=prefer_filename_sequence,
                low_phone_quality=low_phone_quality,
                used_words_based=used_words_based,
                words_syllables=syllables_info,
                alias_syllables=alias_based,
                targets_for_build=targets_for_build,
                score_mapping_fn=_score_kr_syllable_mapping,
                should_prefer_alias_fn=_should_prefer_alias_based_syllables,
                compute_glide_mismatch_fn=_compute_kr_glide_mismatch_ratio,
                is_order_locked_format_fn=_is_kr_order_locked_cv_format,
            )
            syllables_info = list(kr_source_pick.get("syllables_info") or [])
            used_words_based = bool(kr_source_pick.get("used_words_based"))
            used_alias_based = bool(kr_source_pick.get("used_alias_based"))
            mapping_reason_code = str(kr_source_pick.get("mapping_reason_code") or mapping_reason_code)
            base_score = float(kr_source_pick.get("base_score", 0.0) or 0.0)
            alt_score = float(kr_source_pick.get("alt_score", 0.0) or 0.0)
            words_glide_mismatch_ratio = float(kr_source_pick.get("words_glide_mismatch_ratio", 0.0) or 0.0)

            if mapping_reason_code == "filename_sequence_lock":
                log(
                    f"🧭 {fname}: TextGrid 신뢰도 {textgrid_trust_tier.upper()} "
                    f"(trust={textgrid_trust_score:.2f}) → 파일명 순서 기반 매핑 고정"
                )
            elif mapping_reason_code == "alias_based_empty_words":
                log(f"🧭 {fname}: words 매핑에 빈 phone 구간 존재 → alias/phone 기반 음절 매핑 사용")
            elif mapping_reason_code == "alias_phone_minimal":
                log(f"🧭 {fname}: words 티어 없음/실패 → phones 핵 기반 음절 매핑 사용")
            elif mapping_reason_code in {"order_locked_length_mismatch", "order_locked_glide_mismatch", "order_locked_low_phone_quality"}:
                log(
                    f"🧭 {fname}: CV 계열 순서 고정 포맷 보정 "
                    f"(reason={mapping_reason_code}, words={base_score:.1f}, alias={alt_score:.1f}, "
                    f"glide_mismatch={words_glide_mismatch_ratio:.2f})"
                )
            elif mapping_reason_code == "words_low_phone_quality":
                log(
                    f"🧭 {fname}: phones 저신뢰({','.join(low_quality_reasons)}) → words 매핑 고정 "
                    f"(words={base_score:.1f}, alias={alt_score:.1f})"
                )
            elif mapping_reason_code == "alias_based_cvvc":
                log(
                    f"🧭 {fname}: CVVC는 alias 기반 음절 매핑 우선 "
                    f"(words={base_score:.1f}, alias={alt_score:.1f})"
                )
            elif mapping_reason_code == "alias_based_recover":
                log(
                    f"🧭 {fname}: 매핑 이탈 보정 적용 "
                    f"(base={base_score:.1f}, corrected={alt_score:.1f})"
                )
            elif mapping_reason_code == "words_keep_high_conf":
                log(
                    f"🧭 {fname}: words 매핑 신뢰도 높음 → alias 보정 생략 "
                    f"(base={base_score:.1f}, corrected={alt_score:.1f})"
                )
            elif alias_based and targets_for_build:
                log(
                    f"🧭 {fname}: TextGrid(words) 매핑 유지 "
                    f"(base={base_score:.1f}, corrected={alt_score:.1f})"
                )

            plan_candidate_source = list(
                filename_cv_targets
                or [s.get('roman_cv') or s.get('roman', '') for s in (syllables_info or [])]
                or []
            )
            kr_cv_plan = {"indices": None, "meta": {}, "source": ""}
            if plan_candidate_source and sinsy_label_entries:
                kr_cv_plan = build_sinsy_guided_anchor_plan(
                    expected_tokens=plan_candidate_source,
                    syllables_info=syllables_info,
                    label_entries=sinsy_label_entries,
                    normalize_expected_fn=_normalize_cv_match_token,
                    normalize_label_fn=_normalize_cv_match_token,
                    label_match_score_fn=lambda target, label: float(_cv_match_score(target, label)),
                )
            if not kr_cv_plan.get("indices"):
                kr_cv_plan = _build_kr_cv_anchor_plan_v2(
                    plan_candidate_source,
                    syllables_info,
                ) if plan_candidate_source else {"indices": None, "meta": {}}
            kr_planned_cv_indices = kr_cv_plan.get("indices")
            kr_anchor_graph = build_adjacent_anchor_graph(kr_planned_cv_indices)
            kr_plan_policy = resolve_plan_policy(
                alignment_trust=textgrid_trust_score,
                plan_meta=kr_cv_plan.get("meta"),
                expected_count=len(plan_candidate_source),
                planned_count=len(kr_planned_cv_indices or []),
                format_type=file_format,
                prefer_sequence=prefer_filename_sequence,
            )

            mapping_confidence_base, mapping_margin = _estimate_kr_mapping_confidence(
                phone_quality,
                words_score=base_score,
                alias_score=alt_score,
                used_words_based=used_words_based,
                used_alias_based=used_alias_based,
            )
            runtime_policy = resolve_runtime_mapping_policy(
                ingest_snapshot=alignment_ingest,
                plan_policy=kr_plan_policy,
                mapping_confidence=mapping_confidence_base,
                mapping_margin=mapping_margin,
                conf_threshold=file_mapping_conf_th,
                format_type=file_format,
                score_a=base_score,
                score_b=alt_score,
                sequence_lock_formats={"cvvc", "cvc"},
                abstain_formats={"cvvc", "vcv", "cvc", "cv"},
            )
            if sinsy_label_entries:
                plan_source = str(kr_cv_plan.get("source") or "")
                if plan_source != "sinsy_labels":
                    log(
                        f"🛡️ {fname}: sinsy 라벨이 있지만 planner에 적용되지 않음 "
                        f"(source={plan_source or 'fallback'})"
                    )
                else:
                    plan_margin = float((kr_cv_plan.get("meta") or {}).get("margin", 0.0) or 0.0)
                    row_margin_floor = float(runtime_policy.get("row_margin_floor", 6.0))
                    if plan_margin < row_margin_floor:
                        log(
                            f"🛡️ {fname}: sinsy planner margin 낮음 "
                            f"(margin={plan_margin:.1f} < {row_margin_floor:.1f})"
                        )
            mapping_confidence_base = float(runtime_policy.get("mapping_confidence", mapping_confidence_base))
            if kr_mapping_debug_reason_logging and mapping_confidence_base < float(file_mapping_conf_th):
                log(
                    f"🧭 {fname}: KR 매핑 신뢰도 낮음(conf={mapping_confidence_base:.2f}, "
                    f"margin={mapping_margin:+.1f}, reason={mapping_reason_code})"
                )

            file_mapping_low_conf = bool(runtime_policy.get("is_low_conf"))

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

            if bool(runtime_policy.get("should_abstain")):
                log(
                    f"경고: {fname}: KR v2 planner abstain "
                    f"(trust={textgrid_trust_score:.2f}, coverage={float(kr_plan_policy.get('coverage', 0.0)):.2f}, "
                    f"margin={float(kr_plan_policy.get('margin', 0.0)):.1f}) → 원본 유지"
                )
                _record_unset_lines(
                    "mapping_v2_abstain",
                    fname,
                    lines,
                    meta={
                        "mapping_confidence": mapping_confidence_base,
                        "mapping_reason_code": mapping_reason_code,
                        "plan_policy": dict(kr_plan_policy or {}),
                    },
                )
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue

            romaji_syllables = [s.get('roman_cv') or s.get('roman', '') for s in syllables_info]
            current_w_idx = 0
            cv_seq_idx = 0
            bridge_seq_idx = 0
            kr_order_locked_format = _is_kr_order_locked_cv_format(file_format)
            kr_cvvc_occurrence_source = filename_cv_targets if (kr_order_locked_format and filename_cv_targets) else syllables_info
            kr_cvvc_occurrence_map = _build_kr_cvvc_occurrence_map(kr_cvvc_occurrence_source) if kr_order_locked_format else None
            kr_cvvc_occurrence_state = {}
            kr_cvvc_vv_occurrence_map = _build_kr_cvvc_vv_occurrence_map(kr_cvvc_occurrence_source) if file_format == "cvvc" else None
            kr_cvvc_vv_occurrence_state = {}
            cv_anchor_by_idx = {
                i: _estimate_cv_anchor_from_syllable(syllables_info[i], ph_intervals)
                for i in range(len(syllables_info))
            }
            realized_cv_anchor_by_idx = {}
            kr_post_ctx = build_kr_postprocess_context(
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
                row_jump_blocked = 0


                alias_type = _classify_alias_cached(alias)
                row_jump_default = int(max(0, kr_mapping_max_index_jump_default))
                row_jump_high_conf = int(max(row_jump_default, kr_mapping_max_index_jump_high_conf))
                if kr_order_locked_format and textgrid_trust_tier == "low":
                    row_jump_default = 0
                    row_jump_high_conf = max(0, min(row_jump_high_conf, 1))
                elif kr_order_locked_format and textgrid_trust_tier == "mid":
                    row_jump_high_conf = max(row_jump_default, min(row_jump_high_conf, 1))
                if file_mapping_low_conf:
                    row_jump_high_conf = int(max(0, min(row_jump_high_conf, row_jump_default)))
                    if kr_order_locked_format:
                        row_jump_default = 0
                    else:
                        row_jump_default = int(max(0, row_jump_default - 1))
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
                        alias_type=alias_type,
                        wav_duration_ms=wav_duration_ms,
                        validate_fn=validate_oto_params,
                    )
                    continue

                family_state = build_kr_alias_family_state(
                    alias=alias,
                    alias_type=alias_type,
                    uses_vc_context_fn=_uses_kr_vc_context,
                    is_diphthong_fn=is_diphthong,
                    extract_cv_alias_token_fn=_extract_kr_cv_alias_token,
                    detect_glottal_kind_fn=_detect_glottal_kind,
                    kr_vowels=KR_VOWELS,
                )
                is_vc = family_state.is_vc
                is_vcv = family_state.is_vcv
                is_cv_head = family_state.is_cv_head
                is_diph = family_state.is_diph
                target_clean = family_state.target_clean
                bridge_pair = {}
                bridge_seq_slot = None
                if alias_type in {"vc", "vv"}:
                    bridge_seq_slot = bridge_seq_idx
                    bridge_pair = resolve_bridge_anchor_pair(
                        kr_anchor_graph,
                        bridge_seq_slot,
                        realized_anchor_by_idx=realized_cv_anchor_by_idx,
                        estimated_anchor_by_idx=cv_anchor_by_idx,
                        local_prev_idx=current_w_idx,
                        local_next_idx=current_w_idx + 1,
                    )
                    if bridge_seq_idx < len(syllables_info) - 1:
                        bridge_seq_idx += 1

                handled_glottal, current_w_idx, cv_seq_idx = try_handle_kr_glottal_alias(
                    alias=alias,
                    state=family_state,
                    current_w_idx=current_w_idx,
                    cv_seq_idx=cv_seq_idx,
                    syllables_info=syllables_info,
                    ph_intervals=ph_intervals,
                    real_wav_name=real_wav_name,
                    alias_suffix=alias_suffix,
                    final_lines=final_lines,
                    validate_fn=validate_oto_params,
                    apply_alias_suffix_fn=apply_alias_suffix,
                    find_vowel_phone_fn=find_vowel_phone,
                    fit_to_wav_fn=_fit_oto_to_wav_duration,
                    wav_duration_ms=wav_duration_ms,
                )
                if handled_glottal:
                    continue

                handled_breath_tail, current_w_idx = try_handle_kr_breath_tail_alias(
                    alias=alias,
                    state=family_state,
                    current_w_idx=current_w_idx,
                    syllables_info=syllables_info,
                    ph_intervals_all=ph_intervals_all,
                    real_wav_name=real_wav_name,
                    alias_suffix=alias_suffix,
                    final_lines=final_lines,
                    validate_fn=validate_oto_params,
                    apply_alias_suffix_fn=apply_alias_suffix,
                    find_vowel_phone_fn=find_vowel_phone,
                    fit_to_wav_fn=_fit_oto_to_wav_duration,
                    wav_duration_ms=wav_duration_ms,
                )
                if handled_breath_tail:
                    continue



                if is_vcv:
                    vcv_selected_w_idx, cv_seq_idx, row_mapping_confidence = _select_kr_vcv_index_v2(
                        target_clean=target_clean,
                        cv_seq_idx=cv_seq_idx,
                        current_w_idx=current_w_idx,
                        romaji_syllables=romaji_syllables,
                        syllables_info=syllables_info,
                        file_format=file_format,
                        row_mapping_confidence=row_mapping_confidence,
                        row_jump_default=row_jump_default,
                        row_jump_high_conf=row_jump_high_conf,
                        file_mapping_conf_th=file_mapping_conf_th,
                        kr_planned_cv_indices=kr_planned_cv_indices,
                        resolve_planned_cv_index_fn=_resolve_kr_planned_cv_index,
                        resolve_cv_syllable_index_fn=_resolve_cv_syllable_index,
                        clamp_cv_index_to_order_fn=_clamp_kr_cv_index_to_order,
                        split_syllable_parts_fn=_split_kr_syllable_parts,
                        find_cv_vowel_match_index_fn=_find_kr_cv_vowel_match_index,
                        cv_match_score_fn=_cv_match_score,
                        apply_row_confidence_penalty_fn=apply_row_confidence_penalty,
                        log_fn=log,
                        debug_logging=kr_mapping_debug_reason_logging,
                        fname=fname,
                        alias=alias,
                    )
                    current_w_idx = max(current_w_idx, vcv_selected_w_idx)
                    current_w_idx, cv_seq_idx = _run_kr_vcv_row_v2(
                        syllables_info=syllables_info,
                        current_w_idx=current_w_idx,
                        cv_seq_idx=cv_seq_idx,
                        forced_w_idx=vcv_selected_w_idx,
                        diphthong_cv_consonant_ratio=DIPHTHONG_CV_CONSONANT_RATIO,
                        alias=alias,
                        file_format=file_format,
                        real_wav_name=real_wav_name,
                        final_lines=final_lines,
                        generate_openutau=generate_openutau,
                        alias_suffix=alias_suffix,
                        wav_duration_ms=wav_duration_ms,
                        timeline_start_ms=timeline_start_ms,
                        timeline_end_ms=timeline_end_ms,
                        row_mapping_confidence=row_mapping_confidence,
                        mel_ctx_for_file=mel_ctx_for_file,
                        base_shape=base_shape,
                        ph_intervals=ph_intervals,
                        kr_post_ctx=kr_post_ctx,
                        fname=fname,
                        log_fn=log,
                        validate_fn=validate_oto_params,
                        prepare_vcv_syllable_timing_fn=_prepare_vcv_syllable_timing,
                        apply_post_timing_pipeline_fn=_apply_post_timing_pipeline,
                        extract_vcv_anchor_points_fn=_extract_vcv_anchor_points_v2,
                        apply_anchor_lock_fn=_apply_kr_anchor_lock,
                        finalize_row_fn=_finalize_kr_row_v2,
                        row_builder_fn=_build_alias_rows,
                        log_post_timing_events_fn=_log_post_timing_events,
                    )
                    continue


                if is_cv_head:
                    forced_cvvc_idx = _resolve_kr_cv_head_forced_index_v2(
                        alias=alias,
                        alias_type=alias_type,
                        cv_seq_idx=cv_seq_idx,
                        target_clean=target_clean,
                        romaji_syllables=romaji_syllables,
                        syllables_info=syllables_info,
                        kr_planned_cv_indices=kr_planned_cv_indices,
                        kr_cvvc_occurrence_map=kr_cvvc_occurrence_map or {},
                        kr_cvvc_occurrence_state=kr_cvvc_occurrence_state,
                        resolve_planned_cv_index_fn=_resolve_kr_planned_cv_index,
                        resolve_cvvc_occurrence_index_fn=_resolve_kr_cvvc_occurrence_index,
                        remap_forced_cv_index_fn=_remap_kr_forced_cv_index,
                        log_fn=log,
                        debug_logging=kr_mapping_debug_reason_logging,
                        fname=fname,
                    )
                    current_w_idx, cv_seq_idx = _run_kr_cv_head_row_v2(
                        syllables_info=syllables_info,
                        current_w_idx=current_w_idx,
                        cv_seq_idx=cv_seq_idx,
                        alias=alias,
                        forced_w_idx=forced_cvvc_idx,
                        file_format=file_format,
                        real_wav_name=real_wav_name,
                        final_lines=final_lines,
                        generate_openutau=generate_openutau,
                        alias_suffix=alias_suffix,
                        wav_duration_ms=wav_duration_ms,
                        timeline_start_ms=timeline_start_ms,
                        timeline_end_ms=timeline_end_ms,
                        row_mapping_confidence=row_mapping_confidence,
                        mel_ctx_for_file=mel_ctx_for_file,
                        base_shape=base_shape,
                        ph_intervals=ph_intervals,
                        fname=fname,
                        log_fn=log,
                        validate_fn=validate_oto_params,
                        prepare_cv_head_syllable_timing_fn=_prepare_cv_head_syllable_timing,
                        apply_post_timing_pipeline_fn=_apply_post_timing_pipeline,
                        guard_cv_head_offset_to_current_onset_fn=_guard_cv_head_offset_to_current_onset,
                        ensure_cv_head_min_vowel_coverage_fn=_ensure_cv_head_min_vowel_coverage,
                        guard_cv_cutoff_to_next_onset_fn=_guard_cv_cutoff_to_next_onset,
                        prepare_cv_head_anchor_context_fn=_prepare_kr_cv_head_anchor_context_v2,
                        prepare_cv_bounds_fn=_prepare_cv_bounds_from_syllable,
                        apply_anchor_lock_fn=_apply_kr_anchor_lock,
                        build_anchor_record_fn=lambda selected_w_idx, **kwargs: _maybe_build_kr_realized_cv_anchor_record_v2(
                            selected_w_idx,
                            build_anchor_fn=_build_realized_kr_cv_anchor_v2,
                            **kwargs,
                        ),
                        finalize_row_fn=_finalize_kr_row_v2,
                        row_builder_fn=_build_alias_rows,
                        build_guard_messages_fn=_build_kr_cv_head_guard_messages_v2,
                        log_post_timing_events_fn=_log_post_timing_events,
                        anchor_store=realized_cv_anchor_by_idx,
                    )
                    continue


                if not is_vc:
                    forced_vv_idx = None
                    planned_vv_idx = None
                    if file_format == "cvvc" and alias_type == "vv":
                        forced_vv_idx = _resolve_kr_cvvc_vv_index(
                            alias,
                            kr_cvvc_vv_occurrence_map or {},
                            kr_cvvc_vv_occurrence_state,
                        )
                    if alias_type == "vv" and bridge_pair.get("next_idx") is not None:
                        planned_vv_idx = int(bridge_pair["next_idx"])
                    forced_cvvc_idx = _resolve_kr_cvvc_occurrence_index(
                        alias,
                        alias_type,
                        kr_cvvc_occurrence_map or {},
                        kr_cvvc_occurrence_state,
                    )
                    expected_cv_idx = cv_seq_idx
                    planned_cv_idx = None
                    if alias_type == "cv":
                        planned_cv_idx = _resolve_kr_planned_cv_index(
                            kr_planned_cv_indices,
                            expected_cv_idx,
                            target_clean,
                            syllables_info,
                            alias_type="cv",
                        )
                        if planned_cv_idx is not None and kr_mapping_debug_reason_logging and planned_cv_idx != expected_cv_idx:
                            log(
                                f"🧭 {fname}: KR CV 전역 anchor plan 적용 "
                                f"({expected_cv_idx + 1}->{planned_cv_idx + 1}, {alias})"
                            )
                    general_cv_selection = _select_kr_general_cv_index_v2(
                        alias=alias,
                        alias_type=alias_type,
                        fname=fname,
                        file_format=file_format,
                        target_clean=target_clean,
                        current_w_idx=current_w_idx,
                        cv_seq_idx=cv_seq_idx,
                        row_mapping_confidence=row_mapping_confidence,
                        row_jump_default=row_jump_default,
                        row_jump_high_conf=row_jump_high_conf,
                        file_mapping_conf_th=file_mapping_conf_th,
                        file_mapping_low_conf=file_mapping_low_conf,
                        romaji_syllables=romaji_syllables,
                        forced_vv_idx=forced_vv_idx,
                        planned_vv_idx=planned_vv_idx,
                        planned_cv_idx=planned_cv_idx,
                        forced_cvvc_idx=forced_cvvc_idx,
                        remap_forced_cv_index_fn=_remap_kr_forced_cv_index,
                        cv_match_score_fn=_cv_match_score,
                        split_syllable_parts_fn=_split_kr_syllable_parts,
                        apply_row_confidence_penalty_fn=apply_row_confidence_penalty,
                        resolve_cv_syllable_index_fn=_resolve_cv_syllable_index,
                        should_allow_exact_vowel_fix_fn=_should_allow_kr_exact_vowel_fix,
                        find_cv_vowel_match_index_fn=_find_kr_cv_vowel_match_index,
                        clamp_cv_index_to_order_fn=_clamp_kr_cv_index_to_order,
                        log_fn=log,
                        debug_logging=kr_mapping_debug_reason_logging,
                    )
                    expected_cv_idx = int(general_cv_selection["expected_cv_idx"])
                    selected_w_idx = int(general_cv_selection["selected_w_idx"])
                    cv_seq_idx = int(general_cv_selection["cv_seq_idx"])
                    row_mapping_confidence = float(general_cv_selection["row_mapping_confidence"])
                    resolve_meta = dict(general_cv_selection["resolve_meta"])
                    row_jump_blocked = int(general_cv_selection["row_jump_blocked"])
                    forced_selected_idx = general_cv_selection["forced_selected_idx"]
                    row_abstain = decide_cv_row_abstain(
                        alias_type=alias_type,
                        format_type=file_format,
                        candidate_idx=selected_w_idx,
                        candidate_count=len(syllables_info),
                        candidate_active=(
                            _is_kr_cv_syllable_active(syllables_info[selected_w_idx], require_vowel=True)
                            if selected_w_idx is not None and 0 <= selected_w_idx < len(syllables_info)
                            else False
                        ),
                        confidence_margin=mapping_margin,
                        min_confidence_margin=runtime_policy.get("row_margin_floor"),
                        active_only_formats={"cvvc", "cvc", "cv"},
                        margin_formats={"cvvc", "cvc", "cv"},
                    )
                    if row_abstain.get("should_skip"):
                        if kr_mapping_debug_reason_logging:
                            log(
                                f"🛡️ {fname}: KR 행 생성 스킵 "
                                f"({row_abstain.get('reason')}, {alias})"
                            )
                        _record_unset(
                            str(row_abstain.get("reason") or "row_abstain"),
                            fname,
                            line,
                            meta={"diag_hint": row_abstain.get("diag_hint", "")},
                        )
                        continue
                    current_w_idx = max(current_w_idx, selected_w_idx)
                    selected_w_idx, curr_phones, c_start, c_end, n_start, n_end = _prepare_cv_bounds_from_syllable(
                        syllables_info, selected_w_idx
                    )
                else:
                    bridge_next_idx = (
                        int(bridge_pair["next_idx"])
                        if bridge_pair.get("next_idx") is not None
                        else None
                    )
                    if bridge_pair.get("prev_idx") is not None:
                        current_w_idx = int(bridge_pair["prev_idx"])
                    current_w_idx, curr_phones, c_start, c_end, n_start, n_end = _prepare_vc_bounds_from_context(
                        syllables_info,
                        current_w_idx,
                        next_w_idx=bridge_next_idx,
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
                        next_w_idx=bridge_next_idx,
                        prev_cv_anchor=bridge_pair.get("prev_anchor"),
                        next_cv_anchor=bridge_pair.get("next_anchor"),
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
                        elif selected_w_idx is not None and selected_w_idx >= 1 and selected_w_idx < len(syllables_info):
                            prev_syl = syllables_info[selected_w_idx - 1]
                            prev_phones = prev_syl.get("phones") or []
                            if prev_phones:
                                _pv_idx, prev_v_phone = find_vowel_phone(prev_phones)
                                prev_v_start = float(prev_v_phone.minTime) * 1000.0
                                prev_v_end = float(prev_v_phone.maxTime) * 1000.0
                                vv_direct = _compute_kr_vv_timing_from_vowel_bounds(
                                    prev_v_start, prev_v_end, n_start, n_end
                                )
                        if vv_direct is not None:
                            offset, consonant, cutoff, pre, ovl = vv_direct
                        else:
                            offset, consonant, cutoff, pre, ovl = _compute_kr_noninitial_vowel_timing(
                                n_start, n_end
                            )

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
                _run_kr_general_row_v2(
                    final_lines=final_lines,
                    real_wav_name=real_wav_name,
                    alias=alias,
                    alias_type=alias_type,
                    file_format=file_format,
                    offset=offset,
                    consonant=consonant,
                    cutoff=cutoff,
                    pre=pre,
                    ovl=ovl,
                    soft_off_shift=soft_off_shift,
                    soft_cut_shift=soft_cut_shift,
                    cutoff_reduced=cutoff_reduced,
                    selected_w_idx=selected_w_idx,
                    current_w_idx=current_w_idx,
                    c_end=c_end,
                    n_start=n_start,
                    n_end=n_end,
                    c_start=c_start,
                    row_mapping_confidence=row_mapping_confidence,
                    timeline_start_ms=timeline_start_ms,
                    timeline_end_ms=timeline_end_ms,
                    wav_duration_ms=wav_duration_ms,
                    generate_openutau=generate_openutau,
                    alias_suffix=alias_suffix,
                    fname=fname,
                    log_fn=log,
                    validate_fn=validate_oto_params,
                    bridge_pair=bridge_pair,
                    realized_cv_anchor_by_idx=realized_cv_anchor_by_idx,
                    cv_anchor_by_idx=cv_anchor_by_idx,
                    refine_bridge_fn=_refine_kr_bridge_with_adjacent_cv,
                    resolve_anchor_targets_fn=_resolve_kr_anchor_targets_v2,
                    apply_anchor_lock_fn=_apply_kr_anchor_lock,
                    build_anchor_record_fn=lambda selected_idx, **kwargs: _maybe_build_kr_realized_cv_anchor_record_v2(
                        selected_idx,
                        build_anchor_fn=_build_realized_kr_cv_anchor_v2,
                        **kwargs,
                    ),
                    build_bridge_message_fn=_build_kr_bridge_adjust_message_v2,
                    finalize_row_fn=_finalize_kr_row_v2,
                    row_builder_fn=_build_alias_rows,
                    log_post_timing_events_fn=_log_post_timing_events,
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
                            offset, consonant, cutoff, pre, ovl = _compute_kr_noninitial_vowel_timing(
                                v_start, v_end
                            )

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
                                alias_type="mono",
                                validate_fn=validate_oto_params,
                            )
                except:
                    continue

    finish_context = GeneratorFinishContext(
        log_fn=log,
        anchor_stats=anchor_stats,
        anchor_log_path=anchor_log_path,
        anchor_log_dir=_anchor_log_dir,
        cleanup_timing_jsonl=cleanup_timing_jsonl,
        timing_jsonl_prefix="timing_anchor_kr_",
    )

    try:
        write_oto_lines(out_path, final_lines)
        log(f"1차 생성 완료: OTO 파일 저장 -> {out_path}")
        run_kr_post_file_pipeline(
            KrPostFilePipelineContext(
                out_path=out_path,
                tg_folder=tg_folder,
                kr_profile=kr_profile,
                custom_map=custom_map,
                custom_phonemes_path=custom_phonemes_path,
                enable_ml_correction=enable_ml_correction,
                auto_gen_format=auto_gen_format,
                ml_policy=ml_policy,
                runtime_report=runtime_report,
                log_fn=log,
                validate_fn=validate_oto_params,
                normalize_key_fn=normalize_key,
            )
        )
    except Exception as e:
        err = f"OTO 파일 저장 실패: {e}"
        logger.error(err)
        errors.append(err)

    finalize_generator_finish(finish_context)

    _log_unset_summary()
    return processed, total, errors
