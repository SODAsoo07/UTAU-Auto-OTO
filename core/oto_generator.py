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
import traceback
from dataclasses import replace
import logging
from functools import lru_cache
from types import SimpleNamespace
from typing import Optional, Dict

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
    _is_kr_sequence_locked_mapping_format,
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
    _build_kr_profile_bucket_keys,
    _extract_base_timing_shape,
    _parse_oto_line_profile,
    _read_text_with_fallback,
    _resolve_kr_reference_dirs,
    _retarget_kr_bridge_to_next_cv,
)
from core.distribution_guard import is_training_paths_enabled
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
from core.mapping_format_policy import (
    KR_ROW_GUARD_ACTIVE_FORMATS,
    KR_ROW_GUARD_BLANK_FORMATS,
    KR_ROW_GUARD_MARGIN_FORMATS,
    KR_RUNTIME_ABSTAIN_FORMATS,
    KR_RUNTIME_STRICT_FORMATS,
    KR_SEQUENCE_LOCK_FORMATS,
    is_kr_sequence_locked_format,
)
from core.cv_anchor_adaptation import derive_cv_bridge_tuning
from core.generation.contracts import (
    GenerationRequest,
    attach_request_metadata,
    finalize_runtime_report,
    initialize_runtime_report,
)
from core.generation.normalizer import normalize_korean_generation_options
from core.generation.output_finalize import persist_korean_generation_output
from core.generation.file_stages import (
    append_preserved_lines,
    handle_mapping_abstain_fallback,
    handle_mapping_failure_fallback,
    handle_kr_file_context_status,
    handle_kr_loop_prep_status,
    resolve_mapping_line_alias,
    resolve_mapping_failure_reason,
)
from core.generation.mapping_runtime import (
    compute_runtime_low_conf_state,
    format_alignment_guard_summary,
    update_mapping_vc_bridge_runtime_report,
    update_kr_mapping_runtime_report,
)
from core.generation.mapping_reason_logs import build_kr_mapping_reason_log
from core.generation.mapping_reason_codes import (
    COMMON_REASON_FILENAME_TOKEN,
    normalize_mapping_reason_code,
)
from core.generation.runtime_diagnostics import (
    build_mapping_abstain_meta,
    build_mapping_failure_meta,
)
from core.generation.plan_runtime import (
    build_language_runtime_state,
    build_common_plan_context,
    extract_language_runtime_state,
    log_sinsy_plan_guard,
    recompute_common_plan_runtime_state,
)

logger = logging.getLogger(__name__)


def _env_float(name, default):
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_bool(name, default=False):
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _resolve_syllable_strict_mode(*, file_format="") -> str:
    fmt = str(file_format or "").strip().lower()
    if fmt not in {"cvvc", "cvc"}:
        return "off"
    raw = str(os.environ.get("UTOA_SYLLABLE_STRICT_MODE", "") or "").strip().lower()
    if raw in {"strict", "full", "full_strict", "hard"}:
        return "strict"
    if raw in {"off", "none", "disable", "disabled", "0", "false"}:
        return "off"
    return "soft"


def _debug_offset_trace(log_fn, stage, fname, alias, before, after, extra=""):
    if not _env_bool("UTOA_DEBUG_OFFSET_TRACE", False):
        return
    try:
        o0, p0, c0, cut0 = before
        o1, p1, c1, cut1 = after
        do = float(o1) - float(o0)
        dp = float(p1) - float(p0)
        dc = float(c1) - float(c0)
        dcut = float(abs(float(cut1)) - abs(float(cut0)))
        msg = (
            f"[OffsetTrace] {stage} {fname} [{alias}] "
            f"o:{o0:.1f}->{o1:.1f} ({do:+.1f}), "
            f"p:{p0:.1f}->{p1:.1f} ({dp:+.1f}), "
            f"c:{c0:.1f}->{c1:.1f} ({dc:+.1f}), "
            f"cut:{abs(float(cut0)):.1f}->{abs(float(cut1)):.1f} ({dcut:+.1f})"
        )
        if extra:
            msg = f"{msg} | {extra}"
        log_fn(msg)
    except Exception:
        return


def _new_vc_bridge_ab_stats():
    return {
        "rows": 0,
        "pre_abs_shift_sum": 0.0,
        "pre_abs_shift_abs_sum": 0.0,
        "cons_shift_sum": 0.0,
        "cutoff_abs_shift_sum": 0.0,
        "ovl_shift_sum": 0.0,
    }


def _accumulate_vc_bridge_ab_stats(stats, base_params, tuned_params):
    if not isinstance(stats, dict):
        return
    if not base_params or not tuned_params:
        return
    try:
        b_off, b_cons, b_cut, b_pre, b_ovl = base_params
        t_off, t_cons, t_cut, t_pre, t_ovl = tuned_params
        pre_abs_delta = (float(t_off) + float(t_pre)) - (float(b_off) + float(b_pre))
        cons_delta = float(t_cons) - float(b_cons)
        cut_delta = abs(float(t_cut)) - abs(float(b_cut))
        ovl_delta = float(t_ovl) - float(b_ovl)
        stats["rows"] = int(stats.get("rows", 0) or 0) + 1
        stats["pre_abs_shift_sum"] = float(stats.get("pre_abs_shift_sum", 0.0) or 0.0) + pre_abs_delta
        stats["pre_abs_shift_abs_sum"] = float(stats.get("pre_abs_shift_abs_sum", 0.0) or 0.0) + abs(pre_abs_delta)
        stats["cons_shift_sum"] = float(stats.get("cons_shift_sum", 0.0) or 0.0) + cons_delta
        stats["cutoff_abs_shift_sum"] = float(stats.get("cutoff_abs_shift_sum", 0.0) or 0.0) + cut_delta
        stats["ovl_shift_sum"] = float(stats.get("ovl_shift_sum", 0.0) or 0.0) + ovl_delta
    except Exception:
        return


def _summarize_vc_bridge_ab_stats(stats):
    if not isinstance(stats, dict):
        return {}
    rows = int(stats.get("rows", 0) or 0)
    if rows <= 0:
        return {}
    denom = float(rows)
    return {
        "rows": rows,
        "pre_abs_shift_mean_ms": float(stats.get("pre_abs_shift_sum", 0.0) or 0.0) / denom,
        "pre_abs_shift_abs_mean_ms": float(stats.get("pre_abs_shift_abs_sum", 0.0) or 0.0) / denom,
        "cons_shift_mean_ms": float(stats.get("cons_shift_sum", 0.0) or 0.0) / denom,
        "cutoff_abs_shift_mean_ms": float(stats.get("cutoff_abs_shift_sum", 0.0) or 0.0) / denom,
        "ovl_shift_mean_ms": float(stats.get("ovl_shift_sum", 0.0) or 0.0) / denom,
    }


def train_kr_autotune_profile(auto_oto_path, manual_oto_path, custom_phonemes_path=""):
    if not is_training_paths_enabled():
        raise RuntimeError("Distribution build does not support autotune profile training.")
    from core.kr_oto_file_ops import train_kr_autotune_profile as _train_kr_profile
    return _train_kr_profile(auto_oto_path, manual_oto_path, custom_phonemes_path=custom_phonemes_path)


def save_kr_autotune_profile(path, profile):
    if not is_training_paths_enabled():
        raise RuntimeError("Distribution build does not support autotune profile training.")
    from core.kr_oto_file_ops import save_kr_autotune_profile as _save_kr_profile
    return _save_kr_profile(path, profile)


def load_kr_autotune_profile(path):
    from core.kr_oto_file_ops import load_kr_autotune_profile as _load_kr_profile
    return _load_kr_profile(path)


def apply_kr_autotune_profile_to_oto(oto_path, profile, custom_phonemes_path=""):
    if not is_training_paths_enabled():
        raise RuntimeError("Distribution build does not support autotune profile training.")
    from core.kr_oto_file_ops import apply_kr_autotune_profile_to_oto as _apply_kr_profile
    return _apply_kr_profile(oto_path, profile, custom_phonemes_path=custom_phonemes_path)

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


def replace_oto_line_wav_name(line, wav_name):
    target = str(wav_name or "").strip()
    if not target or not line or "=" not in line:
        return line
    _left, right = line.split("=", 1)
    return f"{target}={right}"


def apply_output_wav_name_map(oto_path, wav_name_map):
    if not oto_path or not os.path.exists(oto_path):
        return 0
    exact_map = {}
    norm_map = {}
    norm_conflicts = set()
    for src, dst in (wav_name_map or {}).items():
        s = str(src or "").strip()
        d = str(dst or "").strip()
        if not s or not d:
            continue
        exact_map[s] = d
        nk = normalize_key(s)
        if not nk:
            continue
        prev = norm_map.get(nk)
        if prev is None:
            norm_map[nk] = d
        elif str(prev).lower() != d.lower():
            norm_conflicts.add(nk)

    wav_dir = os.path.dirname(os.path.abspath(oto_path))
    wav_norm_map = {}
    wav_norm_conflicts = set()
    if os.path.isdir(wav_dir):
        try:
            for fn in os.listdir(wav_dir):
                if not str(fn).lower().endswith(".wav"):
                    continue
                nk = normalize_key(fn)
                if not nk:
                    continue
                prev = wav_norm_map.get(nk)
                if prev is None:
                    wav_norm_map[nk] = fn
                elif str(prev).lower() != str(fn).lower():
                    wav_norm_conflicts.add(nk)
        except Exception:
            pass

    out_lines = []
    changed = 0
    for raw in _read_text_with_fallback(oto_path).splitlines():
        line = raw.rstrip("\n")
        if "=" not in line:
            out_lines.append(line)
            continue
        wav, rest = line.split("=", 1)
        wav_raw = wav.strip()
        mapped = exact_map.get(wav_raw)
        if not mapped:
            nk = normalize_key(wav_raw)
            if nk:
                if nk not in norm_conflicts:
                    mapped = norm_map.get(nk)
                if not mapped and nk not in wav_norm_conflicts:
                    mapped = wav_norm_map.get(nk)
        if mapped and mapped != wav_raw:
            out_lines.append(f"{mapped}={rest}")
            changed += 1
        else:
            out_lines.append(line)
    if changed:
        write_oto_lines(oto_path, out_lines)
    return changed

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
    "cv": 0.62,
    "cvvc": 0.66,
    "vcv": 0.68,
    "cvc": 0.63,
    "cv_simple": 0.61,
    "mono": 0.61,
    "vc_only": 0.58,
    "vv_only": 0.58,
    "default": 0.64,
}


def _resolve_kr_mapping_conf_threshold(file_format, override_threshold=None, phone_quality_score=None):
    """포맷별 기본 매핑 신뢰도 임계값을 반환합니다.

    phone_quality_score가 제공되면 phone tier 품질에 따라 동적으로 조정합니다:
    - 품질 < 0.4 -> 임계값을 0.03 낮춤 (완화 폭을 줄여 과도한 점프를 억제)
    - 품질 > 0.8 -> 임계값을 0.02 올림
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
                threshold = max(0.42, threshold - 0.03)
            elif pq > 0.8:
                threshold = min(0.85, threshold + 0.02)
        except (TypeError, ValueError):
            pass

    return threshold


def normalize_key(name):
    return normalize_wav_key(name)


def _iter_textgrid_files(tg_folder):
    if not tg_folder or not os.path.exists(tg_folder):
        return
    for dirpath, _dirnames, filenames in os.walk(tg_folder):
        for f_name in filenames:
            if f_name.lower().endswith(".textgrid"):
                yield dirpath, f_name


def _build_wav_index(wav_root):
    wav_index = {}
    if not wav_root or not os.path.isdir(wav_root):
        return wav_index
    try:
        for dirpath, _dirnames, filenames in os.walk(wav_root):
            for fn in filenames:
                if fn.lower().endswith(".wav"):
                    wav_index.setdefault(normalize_key(fn), os.path.join(dirpath, fn))
    except Exception:
        return {}
    return wav_index


def _resolve_real_wav_name_for_textgrid(textgrid_name, wav_root, wav_index):
    base = os.path.splitext(os.path.basename(str(textgrid_name or "")))[0]
    if not base:
        return ""
    inferred_name = base + ".wav"
    wav_path = _find_wav_path_for_name(inferred_name, wav_root, wav_index)
    if wav_path:
        return os.path.basename(wav_path)
    return inferred_name


def _should_keep_template_alias_set_exact(*, use_template, generate_openutau, gen_missing_vowels):
    return bool(use_template) and not bool(generate_openutau) and not bool(gen_missing_vowels)


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


def _guard_cv_cutoff_to_next_onset(
    offset,
    consonant,
    cutoff,
    pre,
    syll_idx,
    syllables_info,
    *,
    alias_type="",
    min_keep_from_pre_ms=None,
):
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
    a_type = str(alias_type or "").strip().lower()
    safety = 12.0 if hard_next else 7.0
    if a_type == "cv_head":
        safety_scale = _env_float("UTOA_KR_CV_HEAD_NEXT_ONSET_SAFETY_SCALE", 0.72)
        safety = max(4.0, safety * max(0.4, min(1.2, float(safety_scale))))
    next_onset_rel = (next_phones[0].minTime * 1000.0) - offset
    max_cutoff_abs = next_onset_rel - safety
    if max_cutoff_abs <= (pre + 26.0):
        return offset, consonant, cutoff, pre, 0.0

    original_cutoff_abs = abs(cutoff)
    consonant = min(consonant, max_cutoff_abs - 14.0)
    consonant = max(consonant, pre + 16.0)

    cutoff_abs = min(original_cutoff_abs, max_cutoff_abs)
    if min_keep_from_pre_ms is not None:
        try:
            keep_ms = max(24.0, float(min_keep_from_pre_ms))
        except Exception:
            keep_ms = 24.0
        cutoff_abs = max(cutoff_abs, min(max_cutoff_abs, pre + keep_ms))
        consonant = min(consonant, cutoff_abs - 10.0)
        consonant = max(consonant, pre + 10.0)
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


def _guard_cv_cutoff_to_next_onset_strict(offset, consonant, cutoff, pre, syll_idx, syllables_info):
    """order-locked CV에서 다음 음절 onset 침범을 더 강하게 막는 가드."""
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
    safety_hard = _env_float("UTOA_KR_CV_ORDER_LOCK_ONSET_GUARD_HARD", 18.0)
    safety_soft = _env_float("UTOA_KR_CV_ORDER_LOCK_ONSET_GUARD_SOFT", 12.0)
    safety = float(safety_hard) if hard_next else float(safety_soft)
    next_onset_rel = (next_phones[0].minTime * 1000.0) - offset
    max_cutoff_abs = next_onset_rel - safety
    if max_cutoff_abs <= (pre + 32.0):
        return offset, consonant, cutoff, pre, 0.0

    original_cutoff_abs = abs(cutoff)
    consonant = min(consonant, max_cutoff_abs - 16.0)
    consonant = max(consonant, pre + 18.0)

    cutoff_abs = min(original_cutoff_abs, max_cutoff_abs)
    if cutoff_abs <= (consonant + 10.0):
        cutoff_abs = min(max_cutoff_abs, consonant + 18.0)
        if cutoff_abs <= (consonant + 8.0):
            consonant = max(pre + 14.0, cutoff_abs - 14.0)
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


def _guard_cv_head_offset_to_current_onset(
    offset,
    consonant,
    cutoff,
    pre,
    syll_idx,
    syllables_info,
    file_format="",
):
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

    fmt = str(file_format or "").strip().lower()
    order_locked = _is_kr_order_locked_cv_format(fmt)

    # onset 특성별 허용 리드(ms): 파열/치찰은 조금 넓게, 공명음은 더 타이트하게.
    if is_plosive_ipa(c_hint) or c_hint in {"s", "ss", "sh", "ch", "j", "jj", "c", "ts", "h"}:
        base_lead = 36.0
    elif c_hint in {"m", "n", "ng", "l", "r", "y", "w", "ny"}:
        base_lead = 24.0
    else:
        base_lead = 30.0
    if order_locked:
        base_lead += 2.0
    lead_cap = min(base_lead, max(18.0, c_len + 12.0))
    offset_floor = max(0.0, c_start - lead_cap)
    late_guard_ms = _env_float(
        "UTOA_KR_CV_HEAD_OFFSET_LATE_GUARD_MS",
        8.0 if order_locked else 10.0,
    )
    offset_ceil = c_start + late_guard_ms

    new_offset = float(offset)
    if new_offset < offset_floor:
        new_offset = float(offset_floor)
    elif new_offset > offset_ceil:
        target_lead = _env_float(
            "UTOA_KR_CV_HEAD_OFFSET_TARGET_LEAD_MS",
            19.0 if order_locked else 15.0,
        )
        target_cap = min(lead_cap, max(12.0, target_lead))
        new_offset = max(0.0, c_start - target_cap)

    if abs(new_offset - float(offset)) < 0.5:
        return offset, consonant, cutoff, pre, 0.0

    # 오프셋만 재정렬할 때 상대 길이(pre/cons/cutoff)가 급격히 줄어들지 않도록
    # 기존 상대 파라미터를 우선 보존한다.
    new_pre = max(float(pre), 8.0)
    new_consonant = max(float(consonant), new_pre + 8.0)
    new_cut_abs = max(abs(float(cutoff)), new_consonant + 12.0)
    new_cutoff = -new_cut_abs

    new_offset, new_consonant, new_cutoff, new_pre, _ovl = validate_oto_params(
        new_offset, new_consonant, new_cutoff, new_pre, 0.0
    )
    adjusted_ms = abs(float(new_offset) - float(offset))
    return new_offset, new_consonant, new_cutoff, new_pre, adjusted_ms


def _ensure_cv_head_min_vowel_coverage(
    offset,
    consonant,
    cutoff,
    pre,
    vowel_start_ms,
    vowel_end_ms,
    file_format="",
):
    """
    CV_HEAD(-CV)에서 컷오프가 너무 이르게 닫혀 모음이 거의 포함되지 않는 경우를 방지합니다.
    """
    fmt = str(file_format or "").strip().lower()
    order_locked = _is_kr_order_locked_cv_format(fmt)
    v_start = float(vowel_start_ms)
    v_end = float(vowel_end_ms)
    v_len = max(0.0, v_end - v_start)

    cut_abs = abs(float(cutoff))
    if v_len < 40.0:
        # vowel 구간 추정이 불안정해도 -CV가 -C로 붕괴하지 않도록 pre 기준 최소 길이를 강제한다.
        min_from_pre_fallback = _env_float(
            "UTOA_KR_CV_HEAD_MIN_FROM_PRE_FALLBACK_MS",
            112.0 if order_locked else 98.0,
        )
        min_cut_abs = max(float(consonant) + 12.0, float(pre) + float(min_from_pre_fallback))
        if cut_abs >= min_cut_abs:
            return offset, consonant, cutoff, pre, 0.0
        new_cutoff = -min_cut_abs
        offset, consonant, new_cutoff, pre, _ovl = validate_oto_params(
            offset, consonant, new_cutoff, pre, 0.0
        )
        extended_ms = max(0.0, abs(new_cutoff) - cut_abs)
        return offset, consonant, new_cutoff, pre, extended_ms

    # 모음 구간을 최소 일부(비율+하한) 포함하도록 컷오프 하한을 설정.
    keep_v_ratio = _env_float(
        "UTOA_KR_CV_HEAD_KEEP_VOWEL_RATIO",
        0.34 if order_locked else 0.30,
    )
    keep_v_floor = _env_float(
        "UTOA_KR_CV_HEAD_KEEP_VOWEL_FLOOR_MS",
        78.0 if order_locked else 70.0,
    )
    keep_v_ceil = _env_float(
        "UTOA_KR_CV_HEAD_KEEP_VOWEL_CEIL_MS",
        205.0 if order_locked else 190.0,
    )
    keep_v_ms = min(max(v_len * keep_v_ratio, keep_v_floor), keep_v_ceil)
    vowel_start_rel = max(v_start - float(offset), float(pre) + 8.0)
    # pre 이후 너무 빨리 닫히는 케이스(자음만 남는 길이)를 방지.
    min_from_pre_ratio = _env_float(
        "UTOA_KR_CV_HEAD_MIN_FROM_PRE_RATIO",
        0.28 if order_locked else 0.24,
    )
    min_from_pre_floor = _env_float(
        "UTOA_KR_CV_HEAD_MIN_FROM_PRE_FLOOR_MS",
        96.0 if order_locked else 90.0,
    )
    min_from_pre_ceil = _env_float(
        "UTOA_KR_CV_HEAD_MIN_FROM_PRE_CEIL_MS",
        186.0 if order_locked else 180.0,
    )
    min_from_pre = min(max(v_len * min_from_pre_ratio, min_from_pre_floor), min_from_pre_ceil)
    min_cut_abs = max(float(consonant) + 12.0, vowel_start_rel + keep_v_ms, float(pre) + min_from_pre)
    if cut_abs >= min_cut_abs:
        return offset, consonant, cutoff, pre, 0.0

    new_cutoff = -min_cut_abs
    offset, consonant, new_cutoff, pre, _ovl = validate_oto_params(
        offset, consonant, new_cutoff, pre, 0.0
    )
    extended_ms = max(0.0, abs(new_cutoff) - cut_abs)
    return offset, consonant, new_cutoff, pre, extended_ms


def _is_sonorant_like_onset(onset, ipa_hint=""):
    o = str(onset or "").strip().lower()
    h = str(ipa_hint or "").strip().lower()
    if o.startswith(("m", "n", "ng", "l", "r")):
        return True
    if h.startswith(("m", "n", "ŋ", "l", "r")):
        return True
    return False


def _ensure_cv_min_vowel_coverage(
    offset,
    consonant,
    cutoff,
    pre,
    vowel_start_ms,
    vowel_end_ms,
    *,
    alias_onset="",
    ipa_onset="",
):
    """
    CV에서 컷오프가 너무 짧아 모음 시작이 거의 포함되지 않는 경우를 보정합니다.
    특히 유성 자음(m/n/l/r)에서 보수적으로 적용합니다.
    """
    if not _is_sonorant_like_onset(alias_onset, ipa_onset):
        return offset, consonant, cutoff, pre, 0.0
    v_start = float(vowel_start_ms)
    v_end = float(vowel_end_ms)
    v_len = max(0.0, v_end - v_start)
    if v_len < 50.0:
        return offset, consonant, cutoff, pre, 0.0

    cut_abs = abs(float(cutoff))
    keep_v_ms = min(max(v_len * 0.22, 50.0), 150.0)
    vowel_start_rel = max(v_start - float(offset), float(pre) + 8.0)
    min_from_pre = min(max(v_len * 0.20, 70.0), 150.0)
    min_cut_abs = max(float(consonant) + 10.0, vowel_start_rel + keep_v_ms, float(pre) + min_from_pre)
    if cut_abs >= min_cut_abs:
        return offset, consonant, cutoff, pre, 0.0

    new_cutoff = -min_cut_abs
    offset, consonant, new_cutoff, pre, _ovl = validate_oto_params(
        offset, consonant, new_cutoff, pre, 0.0
    )
    extended_ms = max(0.0, abs(new_cutoff) - cut_abs)
    return offset, consonant, new_cutoff, pre, extended_ms


def _guard_cv_focus_window(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    c_start_ms,
    c_end_ms,
    vowel_start_ms,
    vowel_end_ms,
    *,
    alias_onset="",
    ipa_onset="",
):
    """
    CV의 핵심 구간을 보정합니다.
    - offset이 onset 뒤로 밀려 자음이 잘리는 것을 방지
    - cutoff가 모음 tail(흐려지는 구간)을 과도하게 포함하지 않도록 제한
    """
    try:
        c_start = float(c_start_ms)
        c_end = float(c_end_ms)
        v_start = float(vowel_start_ms)
        v_end = float(vowel_end_ms)
    except Exception:
        return offset, consonant, cutoff, pre, ovl, 0.0, 0.0

    if v_end <= v_start or c_end < c_start:
        return offset, consonant, cutoff, pre, ovl, 0.0, 0.0

    onset = str(alias_onset or "").strip().lower()
    ipa_hint = normalize_ipa_mark(ipa_onset)
    hard_onset = is_plosive_ipa(ipa_hint) or ipa_hint in {"s", "ss", "sh", "ch", "j", "jj", "c", "ts", "h"}
    sonorant_onset = _is_sonorant_like_onset(onset, ipa_hint)
    c_len = max(0.0, c_end - c_start)
    if hard_onset:
        base_lead = _env_float("UTOA_KR_CV_OFFSET_LEAD_HARD_MS", 22.0)
    elif sonorant_onset:
        base_lead = _env_float("UTOA_KR_CV_OFFSET_LEAD_SONORANT_MS", 16.0)
    else:
        base_lead = _env_float("UTOA_KR_CV_OFFSET_LEAD_DEFAULT_MS", 20.0)
    lead_cap = min(float(base_lead), max(10.0, c_len + 8.0))
    min_allowed_offset = max(0.0, c_start - lead_cap)

    offset_pulled_ms = 0.0
    cutoff_trimmed_ms = 0.0

    # 1) 자음 onset 이후로 offset이 밀리면 offset 자체를 앞당겨
    # pre/cons/cut 절대축도 같이 앞쪽으로 이동시킨다.
    onset_late_slack = max(0.0, _env_float("UTOA_KR_CV_ONSET_LATE_SLACK_MS", 2.0))
    max_allowed_offset = max(0.0, c_start + onset_late_slack)
    if float(offset) > max_allowed_offset:
        shift = float(offset) - max_allowed_offset
        offset = max_allowed_offset
        offset_pulled_ms = shift
    elif float(offset) < min_allowed_offset:
        # 앞 모음 말미가 offset 구간으로 유입되지 않도록 offset 하한을 보장.
        advance = min_allowed_offset - float(offset)
        offset = min_allowed_offset
        pre = max(6.0, float(pre) - advance)
        consonant = max(float(pre) + 8.0, float(consonant) - advance)
        cutoff = -max(float(consonant) + 12.0, abs(float(cutoff)) - advance)

    # 1.5) pre(anchor)가 모음 중반으로 늦게 들어가는 경우를 강하게 제한.
    # 우선 offset을 더 앞당기되 onset 하한을 넘지 않게 하고,
    # 하한에 걸리면 pre를 직접 줄인다.
    pre_abs = float(offset) + float(pre)
    nucleus_late_slack = max(0.0, _env_float("UTOA_KR_CV_NUCLEUS_LATE_SLACK_MS", 9.0))
    max_pre_abs = float(c_end) + nucleus_late_slack
    if pre_abs > max_pre_abs:
        need = pre_abs - max_pre_abs
        movable = max(0.0, float(offset) - min_allowed_offset)
        moved = min(need, movable)
        offset = float(offset) - moved
        offset_pulled_ms += moved
        residual = max(0.0, need - moved)
        if residual > 0.0 or (float(offset) + float(pre)) > max_pre_abs:
            pre = max(6.0, max_pre_abs - float(offset))

    # 2) 모음 안정 구간 이후 tail은 cutoff 상한으로 제한.
    vowel_len = max(0.0, v_end - v_start)
    if vowel_len >= 38.0:
        if hard_onset:
            tail_trim = _env_float(
                "UTOA_KR_CV_HARD_VOWEL_TAIL_TRIM_MS",
                min(max(vowel_len * 0.24, 26.0), 125.0),
            )
            stable_ratio = _env_float("UTOA_KR_CV_HARD_VOWEL_STABLE_RATIO", 0.68)
        elif sonorant_onset:
            tail_trim = _env_float(
                "UTOA_KR_CV_SONORANT_VOWEL_TAIL_TRIM_MS",
                min(max(vowel_len * 0.12, 14.0), 65.0),
            )
            stable_ratio = _env_float("UTOA_KR_CV_SONORANT_VOWEL_STABLE_RATIO", 0.78)
        else:
            tail_trim = _env_float(
                "UTOA_KR_CV_VOWEL_TAIL_TRIM_MS",
                min(max(vowel_len * 0.18, 20.0), 95.0),
            )
            stable_ratio = _env_float("UTOA_KR_CV_VOWEL_STABLE_RATIO", 0.72)

        stable_floor = v_start + max(48.0, min(vowel_len * stable_ratio, vowel_len - 8.0))
        max_cut_abs = (v_end - max(8.0, float(tail_trim)))
        max_cut_abs = max(max_cut_abs, stable_floor)

        min_cut_abs = float(offset) + float(consonant) + 12.0
        curr_cut_abs = float(offset) + abs(float(cutoff))
        if max_cut_abs > (min_cut_abs + 2.0) and curr_cut_abs > max_cut_abs:
            new_cut_rel = max(float(consonant) + 12.0, max_cut_abs - float(offset))
            cutoff_trimmed_ms = max(0.0, abs(float(cutoff)) - new_cut_rel)
            cutoff = -new_cut_rel

    offset, consonant, cutoff, pre, ovl = validate_oto_params(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        alias_type="cv",
    )
    return offset, consonant, cutoff, pre, ovl, offset_pulled_ms, cutoff_trimmed_ms


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
    file_format="",
    wav_duration_ms=0.0,
    validate_fn=None,
):
    """에일리어스(및 OpenUtau 변형)를 OTO 라인 문자열로 변환합니다."""
    generate_aliases_fn = generate_openutau_aliases
    # Korean phonemizer alias set can be opted-in explicitly.
    use_phonemizer_alias = str(os.environ.get("UTOA_KR_USE_PHONEMIZER_ALIAS", "")).strip().lower()
    if use_phonemizer_alias in {"1", "true", "yes", "on"}:
        if str(alias_type or "").strip() or str(file_format or "").strip():
            generate_aliases_fn = (
                lambda alias_item: generate_korean_phonemizer_aliases(
                    alias_item,
                    alias_type=alias_type,
                    format_type=file_format,
                )
            )
    _params, rows = _prepare_oto_alias_rows_v2(
        real_wav_name,
        alias,
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        generate_openutau=generate_openutau,
        generate_aliases_fn=generate_aliases_fn,
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
    file_format="",
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
            file_format=file_format,
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


def _apply_post_timing_strict_cv_guard(
    offset,
    consonant,
    cutoff,
    pre,
    cutoff_reduced,
    *,
    enable_cutoff_guard,
    alias_type,
    file_format,
    current_w_idx,
    syllables_info,
):
    if (
        enable_cutoff_guard
        and alias_type == "cv"
        and _is_kr_order_locked_cv_format(file_format)
        and _env_bool("UTOA_KR_CV_ORDER_LOCK_STRICT_ONSET_GUARD", True)
    ):
        offset, consonant, cutoff, pre, strict_reduced = _guard_cv_cutoff_to_next_onset_strict(
            offset, consonant, cutoff, pre, current_w_idx, syllables_info
        )
        cutoff_reduced += float(strict_reduced or 0.0)
    return offset, consonant, cutoff, pre, cutoff_reduced


def _apply_post_timing_blank_span_guard(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    *,
    base_offset,
    base_consonant,
    base_cutoff,
    base_pre,
    base_ovl,
    alias_type,
    mel_ctx_for_file,
):
    (offset, consonant, cutoff, pre, ovl), _blank_guard_reverted = _guard_kr_blank_region_span(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        base_offset=base_offset,
        base_consonant=base_consonant,
        base_cutoff=base_cutoff,
        base_pre=base_pre,
        base_ovl=base_ovl,
        alias_type=alias_type,
        mel_ctx=mel_ctx_for_file,
    )
    return offset, consonant, cutoff, pre, ovl


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
    base_offset = float(offset)
    base_consonant = float(consonant)
    base_cutoff = float(cutoff)
    base_pre = float(pre)
    base_ovl = float(ovl)
    if _env_bool("UTOA_DISABLE_SOFT_MEL_GUARD", False):
        def _noop_soft_mel_guard(_o, _c, _cut, _p, _v, *_args, **_kwargs):
            return _o, _c, _cut, _p, _v, 0.0, 0.0
        soft_guard_fn = _noop_soft_mel_guard
    else:
        soft_guard_fn = _apply_soft_mel_offset_cutoff_guard
    ctx = post_ctx or KrPostprocessContext(
        file_format=file_format,
        mel_ctx_for_file=mel_ctx_for_file,
        ph_intervals=ph_intervals,
        syllables_info=syllables_info,
        validate_fn=validate_oto_params,
        soft_mel_guard_fn=soft_guard_fn,
        base_shape_blend_fn=_apply_base_shape_blend,
        stabilize_fn=_stabilize_params_to_phone_activity,
        recenter_fn=_recenter_kr_params_around_pre,
        cv_cutoff_guard_fn=_guard_cv_cutoff_to_next_onset,
    )
    result = ctx.apply_result(
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
    offset = float(result.offset)
    consonant = float(result.consonant)
    cutoff = float(result.cutoff)
    pre = float(result.pre)
    ovl = float(result.ovl)
    soft_off_shift = float(result.soft_off_shift)
    soft_cut_shift = float(result.soft_cut_shift)
    cutoff_reduced = float(result.cutoff_reduced)
    offset, consonant, cutoff, pre, cutoff_reduced = _apply_post_timing_strict_cv_guard(
        offset,
        consonant,
        cutoff,
        pre,
        cutoff_reduced,
        enable_cutoff_guard=enable_cutoff_guard,
        alias_type=alias_type,
        file_format=file_format,
        current_w_idx=current_w_idx,
        syllables_info=syllables_info,
    )
    offset, consonant, cutoff, pre, ovl = _apply_post_timing_blank_span_guard(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        base_offset=base_offset,
        base_consonant=base_consonant,
        base_cutoff=base_cutoff,
        base_pre=base_pre,
        base_ovl=base_ovl,
        alias_type=alias_type,
        mel_ctx_for_file=mel_ctx_for_file,
    )
    return (
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        soft_off_shift,
        soft_cut_shift,
        cutoff_reduced,
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


def _build_kr_words_syllables(
    wd_intervals,
    ph_intervals,
    *,
    force_words_phone_fill,
    decompose_hangul_to_roman_fn,
):
    syllables_info = []
    for w in (wd_intervals or []):
        w_start = w.minTime
        w_end = w.maxTime
        s_phones = [p for p in ph_intervals if p.minTime >= w_start - 0.01 and p.maxTime <= w_end + 0.01]
        if force_words_phone_fill and not s_phones:
            s_phones = _synthesize_kr_word_phones(
                w.mark,
                float(w_start),
                float(w_end),
                decompose_hangul_to_roman_fn,
            )
        roman_parts = []
        for ch in w.mark:
            roman_parts.extend(decompose_hangul_to_roman_fn(ch))
        roman_raw = "".join(roman_parts).lower()
        roman_cv = _kr_cv_kernel(roman_raw)
        syllables_info.append(
            {
                "word": w.mark,
                "roman": roman_raw,
                "roman_cv": roman_cv,
                "start_time": w_start,
                "end_time": w_end,
                "phones": s_phones,
            }
        )
    return syllables_info


def _select_kr_syllable_source_state(
    *,
    file_format,
    prefer_filename_sequence,
    low_phone_quality,
    wd_intervals,
    ph_intervals,
    targets_for_build,
    force_words_phone_fill,
    decompose_hangul_to_roman_fn,
    mel_ctx_for_file,
    textgrid_trust_score,
):
    words_syllables = _build_kr_words_syllables(
        wd_intervals,
        ph_intervals,
        force_words_phone_fill=force_words_phone_fill,
        decompose_hangul_to_roman_fn=decompose_hangul_to_roman_fn,
    )
    used_words_based = len(words_syllables) > 0
    alias_syllables = _build_kr_syllables_from_phone_nuclei(ph_intervals, targets_for_build) if targets_for_build else None
    if mel_ctx_for_file:
        _annotate_kr_syllable_blank_confidence(words_syllables, mel_ctx_for_file)
        _annotate_kr_syllable_blank_confidence(alias_syllables, mel_ctx_for_file)

    source_pick = select_kr_syllable_source(
        file_format=file_format,
        prefer_filename_sequence=prefer_filename_sequence,
        low_phone_quality=low_phone_quality,
        used_words_based=used_words_based,
        words_syllables=words_syllables,
        alias_syllables=alias_syllables,
        targets_for_build=targets_for_build,
        score_mapping_fn=_score_kr_syllable_mapping,
        should_prefer_alias_fn=_should_prefer_alias_based_syllables,
        compute_glide_mismatch_fn=_compute_kr_glide_mismatch_ratio,
        is_order_locked_format_fn=_is_kr_sequence_locked_mapping_format,
    )

    syllables_info = list(source_pick.get("syllables_info") or [])
    if mel_ctx_for_file:
        _annotate_kr_syllable_blank_confidence(syllables_info, mel_ctx_for_file)
    syllable_blank_confidences = [
        float((s or {}).get("blank_confidence", 0.0) or 0.0)
        for s in (syllables_info or [])
    ]
    blank_conf_mean = 0.0
    if syllable_blank_confidences:
        blank_conf_mean = float(sum(syllable_blank_confidences) / float(len(syllable_blank_confidences)))
    alignment_weight = max(
        0.0,
        min(
            1.0,
            (textgrid_trust_score * 0.85) - max(0.0, blank_conf_mean - 0.45) * 0.35,
        ),
    )
    anchor_lock_lite = bool(alignment_weight < 0.58 or blank_conf_mean >= 0.55)
    return {
        "syllables_info": syllables_info,
        "used_words_based": bool(source_pick.get("used_words_based")),
        "used_alias_based": bool(source_pick.get("used_alias_based")),
        "base_score": float(source_pick.get("base_score", 0.0) or 0.0),
        "alt_score": float(source_pick.get("alt_score", 0.0) or 0.0),
        "mapping_reason_code": normalize_mapping_reason_code(
            source_pick.get("mapping_reason_code"),
            language="kr",
            fallback_code=COMMON_REASON_FILENAME_TOKEN,
        ),
        "words_glide_mismatch_ratio": float(source_pick.get("words_glide_mismatch_ratio", 0.0) or 0.0),
        "blank_conf_mean": float(blank_conf_mean),
        "alignment_weight": float(alignment_weight),
        "anchor_lock_lite": bool(anchor_lock_lite),
        "syllable_blank_confidences": syllable_blank_confidences,
        "alias_syllables_present": bool(alias_syllables),
    }


def _log_kr_mapping_selection(
    *,
    log_fn,
    fname,
    mapping_reason_code,
    base_score,
    alt_score,
    words_glide_mismatch_ratio,
    low_quality_reasons,
    alias_syllables_present,
    targets_for_build,
    textgrid_trust_tier,
    textgrid_trust_score,
    alignment_weight,
    blank_conf_mean,
    anchor_lock_lite,
):
    reason_line = build_kr_mapping_reason_log(
        mapping_reason_code=mapping_reason_code,
        base_score=base_score,
        alt_score=alt_score,
        words_glide_mismatch_ratio=words_glide_mismatch_ratio,
        low_quality_reasons=low_quality_reasons,
        alias_syllables_present=alias_syllables_present,
        targets_for_build=targets_for_build,
        textgrid_trust_tier=textgrid_trust_tier,
        textgrid_trust_score=textgrid_trust_score,
    )
    if reason_line:
        log_fn(f"🧭 {fname}: {reason_line}")
    log_fn(
        f"🧭 {fname}: "
        f"{format_alignment_guard_summary(alignment_weight=alignment_weight, blank_confidence_mean=blank_conf_mean, anchor_lock_lite=anchor_lock_lite)}"
    )


def _build_kr_plan_context(
    *,
    plan_candidate_source,
    syllables_info,
    sinsy_label_entries,
    file_format,
    textgrid_trust_tier,
    low_phone_quality,
    blank_conf_mean,
    alignment_weight,
    mel_ctx_for_file,
    prefer_filename_sequence,
):
    fmt = str(file_format or "").strip().lower()
    mel_plan_env = {
        "cvvc": "UTOA_KR_CVVC_MEL_PLAN",
        "cvc": "UTOA_KR_CVC_MEL_PLAN",
        "vcv": "UTOA_KR_VCV_MEL_PLAN",
        "cv": "UTOA_KR_CV_MEL_PLAN",
    }
    return build_common_plan_context(
        planned_cv_source=list(plan_candidate_source or []),
        syllables_info=syllables_info,
        sinsy_label_entries=list(sinsy_label_entries or []),
        format_type=file_format,
        alignment_weight=alignment_weight,
        prefer_sequence=bool(prefer_filename_sequence),
        normalize_expected_fn=_normalize_cv_match_token,
        normalize_label_fn=_normalize_cv_match_token,
        label_match_score_fn=lambda target, label: float(_cv_match_score(target, label)),
        should_enable_mel_plan_fn=lambda: bool(
            fmt in mel_plan_env
            and mel_ctx_for_file
            and syllables_info
            and (
                textgrid_trust_tier != "high"
                or low_phone_quality
                or blank_conf_mean >= 0.45
                or alignment_weight < 0.65
            )
            and _env_bool(mel_plan_env[fmt], True)
        ),
        build_cv_anchor_plan_fn=_build_kr_cv_anchor_plan_v2,
        build_sinsy_guided_anchor_plan_fn=build_sinsy_guided_anchor_plan,
        build_adjacent_anchor_graph_fn=build_adjacent_anchor_graph,
        resolve_plan_policy_fn=resolve_plan_policy,
    )


def _compute_kr_runtime_mapping_state(
    *,
    runtime_policy,
    mapping_confidence_base,
    mapping_margin,
    blank_conf_mean,
    syllable_blank_confidences,
    file_mapping_conf_th,
    file_format,
    kr_mapping_debug_reason_logging,
    fname,
    mapping_reason_code,
    log_fn,
):
    runtime_policy = dict(runtime_policy or {})
    mapping_confidence = float(runtime_policy.get("mapping_confidence", mapping_confidence_base))
    if syllable_blank_confidences and blank_conf_mean >= 0.55:
        mapping_confidence = max(
            0.0,
            mapping_confidence - min(0.22, (blank_conf_mean - 0.55) * 0.45 + 0.04),
        )
    if kr_mapping_debug_reason_logging and mapping_confidence < float(file_mapping_conf_th):
        log_fn(
            f"🧭 {fname}: KR 매핑 신뢰도 낮음(conf={mapping_confidence:.2f}, "
            f"margin={float(mapping_margin):+.1f}, reason={mapping_reason_code})"
        )

    file_conf_floor = float(runtime_policy.get("file_conf_floor", file_mapping_conf_th))
    low_conf_state = compute_runtime_low_conf_state(
        runtime_policy=runtime_policy,
        mapping_confidence=mapping_confidence,
        conf_floor=file_conf_floor,
        blank_confidence_mean=blank_conf_mean,
        conf_below_reason="conf_below_floor",
        row_conf_floor_default=file_mapping_conf_th,
    )
    row_conf_floor = float(low_conf_state.get("row_conf_floor", file_mapping_conf_th))
    row_margin_floor = float(low_conf_state.get("row_margin_floor", 6.0))
    file_mapping_low_conf = bool(low_conf_state.get("file_low_conf"))

    row_blank_floor = None
    fmt_norm = str(file_format or "").strip().lower()
    if is_kr_sequence_locked_format(fmt_norm):
        # CV 계열은 blank 구간 오매핑이 발생하면 이후 ML 보정이 과보정될 수 있어
        # 저신뢰 파일에서 row-level blank gate를 더 엄격하게 건다.
        apply_blank_gate = bool(runtime_policy.get("strict_mode")) or bool(file_mapping_low_conf)
        if blank_conf_mean >= 0.45 or file_mapping_low_conf:
            apply_blank_gate = True
        if apply_blank_gate:
            env_key_by_fmt = {
                "cvvc": "UTOA_KR_CVVC_ROW_BLANK_FLOOR",
                "cvc": "UTOA_KR_CVC_ROW_BLANK_FLOOR",
                "cv": "UTOA_KR_CV_ROW_BLANK_FLOOR",
            }
            default_floor_by_fmt = {
                "cvvc": 0.64,
                "cvc": 0.62,
                "cv": 0.60,
            }
            row_blank_floor = _env_float(
                env_key_by_fmt.get(fmt_norm, "UTOA_KR_CVVC_ROW_BLANK_FLOOR"),
                default_floor_by_fmt.get(fmt_norm, 0.64),
            )

    low_conf_reasons = list(low_conf_state.get("low_conf_reasons") or [])

    return {
        "runtime_policy": runtime_policy,
        "mapping_confidence_base": float(mapping_confidence),
        "mapping_margin": float(mapping_margin),
        "mapping_tier": str(runtime_policy.get("mapping_tier") or ""),
        "file_mapping_low_conf": bool(file_mapping_low_conf),
        "file_conf_floor": float(file_conf_floor),
        "row_conf_floor": float(row_conf_floor),
        "row_margin_floor": float(row_margin_floor),
        "row_blank_floor": (float(row_blank_floor) if row_blank_floor is not None else None),
        "low_conf_reasons": low_conf_reasons,
    }


def _recompute_kr_runtime_state(
    *,
    alignment_ingest,
    file_mapping_conf_th,
    file_format,
    phone_quality,
    base_score,
    alt_score,
    used_words_based,
    used_alias_based,
    mapping_reason_code,
    kr_mapping_debug_reason_logging,
    fname,
    blank_conf_mean,
    syllable_blank_confidences,
    plan_candidate_source,
    sinsy_label_entries,
    syllables_info,
    textgrid_trust_tier,
    low_phone_quality,
    alignment_weight,
    mel_ctx_for_file,
    prefer_filename_sequence,
    log_fn,
):
    mapping_confidence_base, mapping_margin = _estimate_kr_mapping_confidence(
        phone_quality,
        words_score=base_score,
        alias_score=alt_score,
        used_words_based=used_words_based,
        used_alias_based=used_alias_based,
    )
    plan_runtime_state = recompute_common_plan_runtime_state(
        build_plan_context_fn=_build_kr_plan_context,
        plan_context_kwargs={
            "plan_candidate_source": plan_candidate_source,
            "syllables_info": syllables_info,
            "sinsy_label_entries": sinsy_label_entries,
            "file_format": file_format,
            "textgrid_trust_tier": textgrid_trust_tier,
            "low_phone_quality": low_phone_quality,
            "blank_conf_mean": blank_conf_mean,
            "alignment_weight": alignment_weight,
            "mel_ctx_for_file": mel_ctx_for_file,
            "prefer_filename_sequence": prefer_filename_sequence,
        },
        ingest_snapshot=alignment_ingest,
        mapping_confidence=mapping_confidence_base,
        mapping_margin=mapping_margin,
        conf_threshold=file_mapping_conf_th,
        format_type=file_format,
        score_a=base_score,
        score_b=alt_score,
        sequence_lock_formats=KR_SEQUENCE_LOCK_FORMATS,
        abstain_formats=KR_RUNTIME_ABSTAIN_FORMATS,
        strict_formats=KR_RUNTIME_STRICT_FORMATS,
        prefer_sequence=prefer_filename_sequence,
        alignment_trust=alignment_weight,
        resolve_runtime_mapping_policy_fn=resolve_runtime_mapping_policy,
    )
    kr_cv_plan = dict(plan_runtime_state.get("cv_plan") or {})
    kr_planned_cv_indices = plan_runtime_state.get("planned_indices")
    kr_anchor_graph = plan_runtime_state.get("anchor_graph")
    kr_plan_policy = dict(plan_runtime_state.get("plan_policy") or {})
    runtime_policy = dict(plan_runtime_state.get("runtime_policy") or {})
    mapping_confidence_base = float(plan_runtime_state.get("mapping_confidence", mapping_confidence_base))
    mapping_margin = float(plan_runtime_state.get("mapping_margin", mapping_margin))
    log_sinsy_plan_guard(
        sinsy_label_entries=sinsy_label_entries,
        cv_plan=kr_cv_plan,
        runtime_policy=runtime_policy,
        fname=fname,
        log_fn=log_fn,
    )
    runtime_state = _compute_kr_runtime_mapping_state(
        runtime_policy=runtime_policy,
        mapping_confidence_base=mapping_confidence_base,
        mapping_margin=mapping_margin,
        blank_conf_mean=blank_conf_mean,
        syllable_blank_confidences=syllable_blank_confidences,
        file_mapping_conf_th=file_mapping_conf_th,
        file_format=file_format,
        kr_mapping_debug_reason_logging=kr_mapping_debug_reason_logging,
        fname=fname,
        mapping_reason_code=mapping_reason_code,
        log_fn=log_fn,
    )
    runtime_policy = dict(runtime_state.get("runtime_policy") or runtime_policy)
    mapping_confidence_base = float(runtime_state.get("mapping_confidence_base", mapping_confidence_base))
    mapping_margin = float(runtime_state.get("mapping_margin", mapping_margin))
    mapping_tier = str(runtime_policy.get("mapping_tier") or "")
    return build_language_runtime_state(
        language_prefix="kr",
        mapping_confidence_base=float(mapping_confidence_base),
        mapping_margin=float(mapping_margin),
        mapping_tier=str(mapping_tier),
        cv_plan=kr_cv_plan,
        planned_indices=kr_planned_cv_indices,
        anchor_graph=kr_anchor_graph,
        plan_policy=kr_plan_policy,
        runtime_policy=runtime_policy,
        postprocess={
            "file_mapping_low_conf": bool(runtime_state.get("file_mapping_low_conf")),
            "file_conf_floor": float(runtime_state.get("file_conf_floor", file_mapping_conf_th)),
            "row_conf_floor": float(runtime_state.get("row_conf_floor", file_mapping_conf_th)),
            "row_margin_floor": float(runtime_state.get("row_margin_floor", 6.0)),
            "row_blank_floor": runtime_state.get("row_blank_floor"),
            "low_conf_reasons": list(runtime_state.get("low_conf_reasons") or []),
            "sequence_lock_applied": False,
            "sequence_lock_reason": "",
        },
    )


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
    force_snap_dist = _env_float("UTOA_KR_PRE_FORCE_SNAP_DIST", 140.0)
    if nearest_dist >= force_snap_dist:
        offset = target_offset
    elif abs(delta) > 30.0:
        # 큰 거리 snap은 과교정을 유발할 수 있어 블렌딩으로 완화한다.
        offset = _blend(float(offset), target_offset, 0.45)
    else:
        # 짧은 거리 보정은 기존처럼 즉시 snap한다.
        offset = target_offset
    return validate_oto_params(offset, consonant, cutoff, pre, ovl)


def generate_openutau_aliases(base_alias):
    """OpenUtau ??? ?? ?? ??? ?? ?????? ?? ?????."""
    aliases = []
    seen = set()

    def _add(alias_text):
        text = str(alias_text or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        aliases.append(text)

    _add(base_alias)
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


        _add(f"{v} {c}")
        _add(f"{v}{c}")


        if c in ['a','e','i','o','u','eo','eu','ae','oe','wi','ya','yeo','yo','yu','ye','wa','wo','we','weo','eui','ui']:
            _add(f"{v} {c}")
            _add(f"{v}{c}")

            _add(f"{v} -")
            _add(f"{v}-")
        else:

            if c in batchim_map:
                mapped_c = batchim_map[c]

                _add(f"{v} {mapped_c}")
                _add(f"{v}{mapped_c}")
                _add(f"{v} {mapped_c.upper()}")
                _add(f"{v}{mapped_c.upper()}")


                _add(f"{v} {mapped_c}-")
                _add(f"{v}{mapped_c}-")
                _add(f"{mapped_c}-")
                if c != mapped_c:
                    _add(f"{v} {c}-")
                    _add(f"{v}{c}-")
                    _add(f"{c}-")

            c_lower = c.lower()
            if c_lower in stop_variant_map:
                for vc in stop_variant_map[c_lower]:
                    _add(f"{v} {vc}")
                    _add(f"{v}{vc}")
                    _add(f"{v} {vc.upper()}")
                    _add(f"{v}{vc.upper()}")

    elif len(parts) == 1:
        cv = parts[0]

        _add(f"- {cv}")
        _add(f"-{cv}")


        _add(f"{cv}-")
        _add(f"{cv} -")


        if cv == 'eui':
            _add('eu i')
            for item in ['ui', '- ui', '-ui', 'ui -', 'ui-']:
                _add(item)
        elif cv.startswith('-') and 'eui' in cv:
            _add(cv.replace('eui', 'eu i'))

    return aliases


def _dedupe_aliases_preserve_order(items):
    out = []
    seen = set()
    for item in items or []:
        alias = str(item or "").strip()
        if not alias or alias in seen:
            continue
        seen.add(alias)
        out.append(alias)
    return out


def _kr_batchim_upper_variant(token):
    tok = str(token or "").strip()
    if not tok:
        return ""
    return {
        "g": "K",
        "k": "K",
        "d": "T",
        "t": "T",
        "s": "T",
        "ss": "T",
        "j": "T",
        "jj": "T",
        "ch": "T",
        "h": "H",
        "r": "L",
        "l": "L",
        "m": "M",
        "b": "P",
        "p": "P",
        "ng": "NG",
        "n": "N",
    }.get(tok.lower(), "")


def generate_korean_phonemizer_aliases(base_alias, *, alias_type="", format_type=""):
    alias = str(base_alias or "").strip()
    if not alias:
        return []

    a_type = str(alias_type or "").strip().lower()
    fmt = str(format_type or "").strip().lower()
    out = [alias]
    dash_enabled = str(os.environ.get("UTOA_ENABLE_HEAD_TAIL_DASH_ALIAS", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

    if a_type in {"cv", "mono"}:
        if dash_enabled and fmt in {"cv", "cvc", "cvvc", "vcv"}:
            out.extend([f"- {alias}", f"-{alias}"])

    elif a_type == "cv_head":
        if dash_enabled and alias.startswith("- "):
            core = alias[2:].strip()
            if core:
                out.append(f"-{core}")

    elif a_type == "vc":
        left = ""
        right = ""
        parts = [p for p in alias.split() if p]
        if len(parts) >= 2:
            left = parts[0]
            right = parts[-1]
        else:
            onset, vowel, coda = _split_kr_syllable_parts(alias.lower().rstrip("-"))
            if vowel and coda:
                left = vowel
                right = coda
        if left and right:
            out.extend([f"{left} {right}", f"{left}{right}"])
            upper = _kr_batchim_upper_variant(right)
            if upper:
                out.extend([f"{left} {upper}", f"{left}{upper}"])
                if fmt == "cv":
                    out.append(upper)
            if fmt == "cv":
                out.append(right)

    return _dedupe_aliases_preserve_order(out)


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
    raw = None
    sr = None
    n_ch = 0
    sw = 0
    try:
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception:
        raw = None

    if raw is not None:
        if sw == 1:
            audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128.0) / 128.0
        elif sw == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 3:
            # 24-bit PCM (little-endian) -> signed int32 -> float32 [-1, 1]
            b = np.frombuffer(raw, dtype=np.uint8)
            if b.size % 3 != 0:
                return None, None
            b = b.reshape(-1, 3)
            v = (
                b[:, 0].astype(np.int32)
                | (b[:, 1].astype(np.int32) << 8)
                | (b[:, 2].astype(np.int32) << 16)
            )
            v = np.where(v & 0x800000, v - 0x1000000, v)
            audio = v.astype(np.float32) / 8388608.0
        elif sw == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            audio = None

        if audio is not None:
            if n_ch > 1:
                try:
                    audio = audio.reshape(-1, n_ch).mean(axis=1)
                except Exception:
                    return None, None
            return audio, sr

    # Fallback decoders for WAV variants wave module can't parse.
    try:
        import soundfile as sf  # type: ignore

        audio, sr2 = sf.read(wav_path, always_2d=False, dtype="float32")
        if audio is None:
            return None, None
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        return np.asarray(audio, dtype=np.float32), int(sr2)
    except Exception:
        pass

    try:
        from scipy.io import wavfile  # type: ignore

        sr3, audio = wavfile.read(wav_path)
        if audio is None:
            return None, None
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        if np.issubdtype(audio.dtype, np.integer):
            max_abs = float(np.iinfo(audio.dtype).max) if np.iinfo(audio.dtype).max > 0 else 1.0
            audio = audio.astype(np.float32) / max_abs
        else:
            audio = audio.astype(np.float32)
        return audio, int(sr3)
    except Exception:
        return None, None


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

    # 40-band filterbank center frequency (Hz), used for coarse spectral-region cues.
    band_centers_hz = np.asarray(hz_points[1:-1], dtype=np.float64)
    f2_idx = np.where((band_centers_hz >= 900.0) & (band_centers_hz <= 2500.0))[0]
    f3_idx = np.where((band_centers_hz >= 2000.0) & (band_centers_hz <= 3800.0))[0]
    high_idx = np.where((band_centers_hz >= 3500.0) & (band_centers_hz <= min(9000.0, f_max)))[0]
    low_idx = np.where((band_centers_hz >= 80.0) & (band_centers_hz <= 700.0))[0]

    frames = []
    rms_vals = []
    db_vals = []
    f0_voicing = []
    f2_ratio_vals = []
    f3_ratio_vals = []
    high_ratio_vals = []
    low_ratio_vals = []
    spec_presence_vals = []
    times = []
    last_voicing = 0.0
    frame_idx = 0
    for st in range(0, max(len(audio) - win + 1, 1), hop):
        fr_raw = audio[st:st + win]
        if len(fr_raw) < win:
            fr_raw = np.concatenate([fr_raw, np.zeros(win - len(fr_raw), dtype=np.float32)], axis=0)

        rms = float(np.sqrt(np.mean(fr_raw.astype(np.float64) ** 2) + 1e-12))
        rms_vals.append(rms)
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

        mel_nonneg = np.maximum(mel, 0.0)
        mel_total = float(np.sum(mel_nonneg) + 1e-12)
        if len(f2_idx):
            f2_ratio_vals.append(float(np.sum(mel_nonneg[f2_idx]) / mel_total))
        else:
            f2_ratio_vals.append(0.0)
        if len(f3_idx):
            f3_ratio_vals.append(float(np.sum(mel_nonneg[f3_idx]) / mel_total))
        else:
            f3_ratio_vals.append(0.0)
        if len(high_idx):
            high_ratio_vals.append(float(np.sum(mel_nonneg[high_idx]) / mel_total))
        else:
            high_ratio_vals.append(0.0)
        if len(low_idx):
            low_ratio_vals.append(float(np.sum(mel_nonneg[low_idx]) / mel_total))
        else:
            low_ratio_vals.append(0.0)
        spec_presence_vals.append(float(np.log1p(mel_total)))
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
    rms_arr = np.asarray(rms_vals, dtype=np.float64) if rms_vals else np.zeros_like(en)
    if len(rms_arr) != len(en):
        rms_arr = np.resize(rms_arr, len(en))
    rms_arr = np.clip(rms_arr, 1e-8, 1.0)
    db_arr = np.array(db_vals, dtype=np.float64) if db_vals else np.zeros_like(en)
    f0v_arr = np.array(f0_voicing, dtype=np.float64) if f0_voicing else np.zeros_like(en)
    if len(f0v_arr) >= 3:
        f0v_arr = np.convolve(f0v_arr, np.array([0.2, 0.6, 0.2], dtype=np.float64), mode="same")
    f0v_arr = np.clip(f0v_arr, 0.0, 1.0)
    if len(en) >= 5:
        en_ma = np.convolve(en, np.ones(5, dtype=np.float64) / 5.0, mode="same")
    else:
        en_ma = np.asarray(en, dtype=np.float64)

    def _normalize01(vals):
        arr = np.asarray(vals, dtype=np.float64) if vals else np.zeros_like(en)
        if len(arr) != len(en):
            arr = np.resize(arr, len(en))
        lo = float(np.percentile(arr, 10)) if len(arr) else 0.0
        hi = float(np.percentile(arr, 90)) if len(arr) else 1.0
        span_local = max(hi - lo, 1e-9)
        out = (arr - lo) / span_local
        return np.clip(out, 0.0, 1.0)

    f2_arr = np.clip(np.asarray(f2_ratio_vals, dtype=np.float64), 0.0, 1.0) if f2_ratio_vals else np.zeros_like(en)
    f3_arr = np.clip(np.asarray(f3_ratio_vals, dtype=np.float64), 0.0, 1.0) if f3_ratio_vals else np.zeros_like(en)
    high_arr = np.clip(np.asarray(high_ratio_vals, dtype=np.float64), 0.0, 1.0) if high_ratio_vals else np.zeros_like(en)
    low_arr = np.clip(np.asarray(low_ratio_vals, dtype=np.float64), 0.0, 1.0) if low_ratio_vals else np.zeros_like(en)
    spec_presence_arr = _normalize01(spec_presence_vals)
    rms_norm_arr = _normalize01(rms_arr.tolist())

    rms_p10 = float(np.percentile(rms_arr, 10)) if len(rms_arr) else 1e-8
    rms_p20 = float(np.percentile(rms_arr, 20)) if len(rms_arr) else rms_p10
    rms_p60 = float(np.percentile(rms_arr, 60)) if len(rms_arr) else rms_p20
    rms_p85 = float(np.percentile(rms_arr, 85)) if len(rms_arr) else rms_p60
    rms_sil_th = max(1e-8, min(0.30, (0.78 * rms_p20) + (0.22 * rms_p10)))
    rms_sound_th = max(rms_sil_th + 1e-8, (0.74 * rms_p60) + (0.26 * rms_p85))
    if rms_sound_th < (rms_sil_th * 1.35):
        rms_sound_th = rms_sil_th * 1.35

    rms_norm_sil_th = max(0.05, min(0.42, float(np.percentile(rms_norm_arr, 22)) + 0.03))
    rms_norm_sound_th = max(
        rms_norm_sil_th + 0.08,
        min(0.95, float(np.percentile(rms_norm_arr, 64)) - 0.02),
    )
    if rms_norm_sound_th <= rms_norm_sil_th:
        rms_norm_sound_th = min(0.95, rms_norm_sil_th + 0.08)

    f2_p60 = float(np.percentile(f2_arr, 60)) if len(f2_arr) else 0.0
    f3_p60 = float(np.percentile(f3_arr, 60)) if len(f3_arr) else 0.0
    high_p65 = float(np.percentile(high_arr, 65)) if len(high_arr) else 0.0
    spec_p40 = float(np.percentile(spec_presence_arr, 40)) if len(spec_presence_arr) else 0.0

    db_p20 = float(np.percentile(db_arr, 20)) if len(db_arr) else -60.0
    db_sil_th = max(-58.0, min(-28.0, db_p20 + 6.0))

    # Frame-level coarse class signals used for blank/span confidence and
    # mapping guards.
    f2_strong_th = max(0.05, min(0.25, (0.7 * 0.115) + (0.3 * f2_p60)))
    f3_strong_th = max(0.04, min(0.22, (0.7 * 0.082) + (0.3 * f3_p60)))
    high_noise_th = max(0.10, min(0.35, (0.7 * 0.20) + (0.3 * high_p65)))
    spec_noise_th = max(0.06, min(0.32, (0.7 * 0.14) + (0.3 * spec_p40)))
    rms_silence_mask = (rms_arr <= rms_sil_th) | (rms_norm_arr <= rms_norm_sil_th)
    rms_sound_mask = (rms_arr >= rms_sound_th) | (rms_norm_arr >= rms_norm_sound_th)
    rms_mid_sound_mask = (rms_arr >= max(rms_sil_th * 1.05, 1e-8)) | (rms_norm_arr >= max(0.08, rms_norm_sil_th * 1.05))

    voiced_formant = (
        (
            (f2_arr > f2_strong_th)
            & (f3_arr > f3_strong_th)
            & (db_arr > (db_sil_th + 1.2))
            & (rms_mid_sound_mask | (en >= 0.13))
        )
        | (
            (f2_arr > max(0.08, f2_p60 * 0.85))
            & (f0v_arr > 0.38)
            & (db_arr > (db_sil_th - 1.0))
            & (rms_mid_sound_mask | (en >= 0.11))
        )
    )
    silence_sparse = (
        (
            (db_arr <= (db_sil_th - 1.5))
            & (en <= 0.12)
            & (spec_presence_arr <= max(0.10, spec_p40 * 0.92))
            & (rms_norm_arr <= max(rms_norm_sil_th, 0.30))
        )
        | (
            rms_silence_mask
            & (db_arr <= (db_sil_th + 0.8))
            & (en <= 0.20)
            & ~voiced_formant
        )
    )
    unvoiced_diffuse = (
        (high_arr >= max(0.12, high_noise_th * 0.88))
        & (spec_presence_arr >= max(0.08, spec_noise_th * 0.86))
        & (db_arr > (db_sil_th - 2.0))
        & ~rms_silence_mask
        & (f2_arr < max(0.05, f2_strong_th * 0.92))
        & (f3_arr < max(0.04, f3_strong_th * 0.92))
        & ~voiced_formant
    )
    breath_like = (
        (en <= 0.24)
        & (db_arr <= (db_sil_th + 5.5))
        & (high_arr >= max(0.08, high_p65 * 0.72))
        & (spec_presence_arr >= max(0.05, spec_p40 * 0.62))
        & (rms_arr >= max(rms_sil_th * 0.85, 1e-8))
        & ~voiced_formant
    )
    return {
        "times_ms": np.array(times, dtype=np.float64),
        "energy": en,
        "energy_ma": en_ma,
        "span": span,
        "rms": rms_arr,
        "rms_norm": rms_norm_arr,
        "db_db": db_arr,
        "db_silence_th": float(db_sil_th),
        "rms_silence_th": float(rms_sil_th),
        "rms_sound_th": float(rms_sound_th),
        "rms_norm_silence_th": float(rms_norm_sil_th),
        "rms_norm_sound_th": float(rms_norm_sound_th),
        "f0_voicing": f0v_arr,
        "f2_ratio": f2_arr,
        "f3_ratio": f3_arr,
        "high_ratio": high_arr,
        "low_ratio": low_arr,
        "spec_presence": spec_presence_arr,
        "f2_p60": f2_p60,
        "f3_p60": f3_p60,
        "high_p65": high_p65,
        "spec_p40": spec_p40,
        "cls_voiced_formant": np.asarray(voiced_formant, dtype=np.float64),
        "cls_silence_sparse": np.asarray(silence_sparse, dtype=np.float64),
        "cls_unvoiced_diffuse": np.asarray(unvoiced_diffuse, dtype=np.float64),
        "cls_breath_like": np.asarray(breath_like, dtype=np.float64),
        "rms_silence_mask": np.asarray(rms_silence_mask, dtype=np.float64),
        "rms_sound_mask": np.asarray(rms_sound_mask, dtype=np.float64),
        "voiced_mask": np.asarray(f0v_arr >= 0.5, dtype=np.float64),
        "formant_mask": np.asarray(voiced_formant, dtype=np.float64),
    }


def _resolve_mel_rms_views(mel_ctx, n_frames: int):
    if np is None or n_frames <= 0:
        return (
            np.zeros(max(int(n_frames), 0), dtype=np.float64),
            np.zeros(max(int(n_frames), 0), dtype=np.float64),
            1e-8,
            1e-8,
            0.10,
            0.20,
        )

    rms_arr = mel_ctx.get("rms") if isinstance(mel_ctx, dict) else None
    if rms_arr is None or len(rms_arr) != n_frames:
        db_arr = mel_ctx.get("db_db") if isinstance(mel_ctx, dict) else None
        if db_arr is not None and len(db_arr) == n_frames:
            rms_arr = np.power(10.0, np.asarray(db_arr, dtype=np.float64) / 20.0)
        else:
            rms_arr = np.zeros(n_frames, dtype=np.float64)
    else:
        rms_arr = np.asarray(rms_arr, dtype=np.float64)
    rms_arr = np.clip(rms_arr, 1e-8, 1.0)

    rms_norm_arr = mel_ctx.get("rms_norm") if isinstance(mel_ctx, dict) else None
    if rms_norm_arr is None or len(rms_norm_arr) != n_frames:
        p10 = float(np.percentile(rms_arr, 10)) if len(rms_arr) else 0.0
        p90 = float(np.percentile(rms_arr, 90)) if len(rms_arr) else 1.0
        span = max(p90 - p10, 1e-9)
        rms_norm_arr = np.clip((rms_arr - p10) / span, 0.0, 1.0)
    else:
        rms_norm_arr = np.clip(np.asarray(rms_norm_arr, dtype=np.float64), 0.0, 1.0)

    rms_sil_th = float(mel_ctx.get("rms_silence_th", 0.0) or 0.0) if isinstance(mel_ctx, dict) else 0.0
    rms_sound_th = float(mel_ctx.get("rms_sound_th", 0.0) or 0.0) if isinstance(mel_ctx, dict) else 0.0
    rms_norm_sil_th = float(mel_ctx.get("rms_norm_silence_th", 0.0) or 0.0) if isinstance(mel_ctx, dict) else 0.0
    rms_norm_sound_th = float(mel_ctx.get("rms_norm_sound_th", 0.0) or 0.0) if isinstance(mel_ctx, dict) else 0.0

    if rms_sil_th <= 0.0 or rms_sound_th <= 0.0:
        rms_p10 = float(np.percentile(rms_arr, 10)) if len(rms_arr) else 1e-8
        rms_p20 = float(np.percentile(rms_arr, 20)) if len(rms_arr) else rms_p10
        rms_p60 = float(np.percentile(rms_arr, 60)) if len(rms_arr) else rms_p20
        rms_p85 = float(np.percentile(rms_arr, 85)) if len(rms_arr) else rms_p60
        rms_sil_th = max(1e-8, min(0.30, (0.78 * rms_p20) + (0.22 * rms_p10)))
        rms_sound_th = max(rms_sil_th + 1e-8, (0.74 * rms_p60) + (0.26 * rms_p85))
        if rms_sound_th < (rms_sil_th * 1.35):
            rms_sound_th = rms_sil_th * 1.35
    else:
        rms_sil_th = max(1e-8, min(1.0, rms_sil_th))
        rms_sound_th = max(rms_sil_th + 1e-8, min(1.0, rms_sound_th))

    if rms_norm_sil_th <= 0.0 or rms_norm_sound_th <= 0.0:
        rms_norm_sil_th = max(0.05, min(0.42, float(np.percentile(rms_norm_arr, 22)) + 0.03))
        rms_norm_sound_th = max(
            rms_norm_sil_th + 0.08,
            min(0.95, float(np.percentile(rms_norm_arr, 64)) - 0.02),
        )
        if rms_norm_sound_th <= rms_norm_sil_th:
            rms_norm_sound_th = min(0.95, rms_norm_sil_th + 0.08)
    else:
        rms_norm_sil_th = max(0.0, min(1.0, rms_norm_sil_th))
        rms_norm_sound_th = max(rms_norm_sil_th + 1e-6, min(1.0, rms_norm_sound_th))

    return (
        rms_arr,
        rms_norm_arr,
        float(rms_sil_th),
        float(rms_sound_th),
        float(rms_norm_sil_th),
        float(rms_norm_sound_th),
    )


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


def _select_time_mask(times_ms, start_ms: float, end_ms: float):
    if np is None or times_ms is None or len(times_ms) == 0:
        return np.array([], dtype=np.int64) if np is not None else []
    return np.where((times_ms >= float(start_ms)) & (times_ms <= float(end_ms)))[0]


def _estimate_mel_voiced_onset(mel_ctx, anchor_ms: float, window_ms: float = 80.0) -> Optional[float]:
    if np is None or not mel_ctx:
        return None
    times_ms = mel_ctx.get("times_ms")
    en = mel_ctx.get("energy")
    f0v = mel_ctx.get("f0_voicing")
    cls_voiced = mel_ctx.get("cls_voiced_formant")
    if times_ms is None or en is None or len(times_ms) == 0:
        return None

    mask = _select_time_mask(times_ms, float(anchor_ms) - float(window_ms), float(anchor_ms) + float(window_ms))
    if len(mask) == 0:
        return None

    if f0v is None or len(f0v) != len(en):
        f0v = np.zeros_like(en, dtype=np.float64)
    if cls_voiced is None or len(cls_voiced) != len(en):
        f2_arr = mel_ctx.get("f2_ratio")
        f3_arr = mel_ctx.get("f3_ratio")
        db_arr = mel_ctx.get("db_db")
        db_sil_th = float(mel_ctx.get("db_silence_th", -42.0))
        if f2_arr is None or len(f2_arr) != len(en):
            f2_arr = np.zeros_like(en, dtype=np.float64)
        if f3_arr is None or len(f3_arr) != len(en):
            f3_arr = np.zeros_like(en, dtype=np.float64)
        if db_arr is None or len(db_arr) != len(en):
            db_arr = np.zeros_like(en, dtype=np.float64)
        formant_mask = (f2_arr >= 0.08) & (f3_arr >= 0.06) & (db_arr > (db_sil_th + 1.0))
    else:
        formant_mask = np.asarray(cls_voiced, dtype=np.float64) >= 0.5

    en_ma = mel_ctx.get("energy_ma")
    if en_ma is None or len(en_ma) != len(en):
        en_ma = np.asarray(en, dtype=np.float64)
    (
        rms_arr,
        rms_norm_arr,
        rms_sil_th,
        rms_sound_th,
        rms_norm_sil_th,
        rms_norm_sound_th,
    ) = _resolve_mel_rms_views(mel_ctx, len(en))
    voiced_mask = np.asarray(f0v, dtype=np.float64) >= 0.5
    rms_support_mask = (
        (rms_arr >= max(rms_sound_th * 0.82, rms_sil_th * 1.08))
        | (rms_norm_arr >= max(0.42, rms_norm_sound_th * 0.72))
    )
    energy_mask = (en_ma >= 0.12) | (np.asarray(en, dtype=np.float64) >= 0.15) | rms_support_mask

    cand_mask = voiced_mask & formant_mask & energy_mask
    if not np.any(cand_mask[mask]):
        cand_mask = voiced_mask & energy_mask
    if not np.any(cand_mask[mask]):
        return None

    for idx in mask:
        if cand_mask[idx]:
            return float(times_ms[idx])
    return None


def _estimate_mel_vowel_nucleus(mel_ctx, onset_ms: Optional[float], search_after_ms: float = 220.0):
    if np is None or not mel_ctx or onset_ms is None:
        return None, None
    times_ms = mel_ctx.get("times_ms")
    en = mel_ctx.get("energy")
    f0v = mel_ctx.get("f0_voicing")
    cls_voiced = mel_ctx.get("cls_voiced_formant")
    if times_ms is None or en is None or len(times_ms) == 0:
        return None, None

    if f0v is None or len(f0v) != len(en):
        f0v = np.zeros_like(en, dtype=np.float64)
    if cls_voiced is None or len(cls_voiced) != len(en):
        cls_voiced = np.zeros_like(en, dtype=np.float64)

    mask = _select_time_mask(times_ms, float(onset_ms), float(onset_ms) + float(search_after_ms))
    if len(mask) == 0:
        return None, None

    voiced_formant = (np.asarray(cls_voiced, dtype=np.float64) >= 0.5) & (np.asarray(f0v, dtype=np.float64) >= 0.35)
    if not np.any(voiced_formant[mask]):
        voiced_formant = np.asarray(f0v, dtype=np.float64) >= 0.45
    if not np.any(voiced_formant[mask]):
        return None, None

    peak_idx = None
    peak_val = -1e9
    for idx in mask:
        if voiced_formant[idx]:
            e_v = float(en[idx])
            if e_v > peak_val:
                peak_val = e_v
                peak_idx = int(idx)
    if peak_idx is None:
        return None, None

    peak_energy = max(float(en[peak_idx]), 0.0)
    energy_floor = max(0.08, peak_energy * 0.70)
    start_idx = peak_idx
    end_idx = peak_idx
    for idx in range(peak_idx, mask[0] - 1, -1):
        if not voiced_formant[idx] or float(en[idx]) < energy_floor:
            break
        start_idx = idx
    for idx in range(peak_idx, mask[-1] + 1):
        if not voiced_formant[idx] or float(en[idx]) < energy_floor:
            break
        end_idx = idx

    return float(times_ms[start_idx]), float(times_ms[end_idx])


def _resolve_mel_onset_weight(alignment_weight: float, textgrid_trust_tier: str) -> float:
    tier = str(textgrid_trust_tier or "").strip().lower()
    w = float(alignment_weight or 0.0)
    mode = str(os.environ.get("UTOA_MEL_WEIGHT_MODE", "mel_first") or "mel_first").strip().lower()

    if mode in {"auto", "textgrid", "textgrid_trust", "legacy"}:
        if tier == "high" and w >= 0.75:
            base = 0.0
        elif w < 0.45:
            base = 0.72
        elif w < 0.65:
            base = 0.58
        else:
            base = 0.36
    else:
        # mel_first: keep MEL influence even when TextGrid trust is high.
        if w < 0.45:
            base = 0.82
        elif w < 0.65:
            base = 0.70
        elif tier == "high" and w >= 0.75:
            base = 0.48
        else:
            base = 0.58

    gamma = max(0.5, min(2.5, _env_float("UTOA_ML_ANCHOR_MEL_GAMMA", 1.2)))
    reliability = max(0.0, min(1.0, w))
    if gamma > 1.0 and reliability < 1.0:
        shrink = max(0.55, reliability ** (gamma - 1.0))
        base *= shrink

    if mode in {"auto", "textgrid", "textgrid_trust", "legacy"}:
        return max(0.0, min(0.92, base))
    return max(0.22, min(0.92, base))


def _apply_mel_voiced_onset_pre_shift(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    mel_onset_abs: Optional[float],
    *,
    weight: float,
):
    if mel_onset_abs is None or weight <= 0.0:
        return offset, consonant, cutoff, pre, ovl, 0.0
    pre_abs = float(offset) + float(pre)
    delta = (float(mel_onset_abs) - pre_abs) * float(weight)
    if abs(delta) < 0.8:
        return offset, consonant, cutoff, pre, ovl, 0.0

    offset_new = float(offset) + float(delta)
    pre_new = float(pre)
    if offset_new < 0.0:
        offset_new = 0.0
        pre_new = max(pre_abs + float(delta), 0.0)
    delta_pre = pre_new - float(pre)
    if abs(delta_pre) > 1e-6:
        consonant = float(consonant) + delta_pre
        ovl = float(ovl) + delta_pre
    offset_new, consonant, cutoff, pre_new, ovl = validate_oto_params(
        offset_new, consonant, cutoff, pre_new, ovl
    )
    return offset_new, consonant, cutoff, pre_new, ovl, float(delta)


def _estimate_mel_cutoff_candidate(mel_ctx, pre_abs: float, cut_abs: float) -> Optional[float]:
    if np is None or not mel_ctx:
        return None
    times_ms = mel_ctx.get("times_ms")
    en = mel_ctx.get("energy")
    cls_silence = mel_ctx.get("cls_silence_sparse")
    if times_ms is None or en is None or len(times_ms) == 0:
        return None
    if cls_silence is None or len(cls_silence) != len(en):
        return None

    mask = _select_time_mask(times_ms, float(pre_abs) + 20.0, float(cut_abs) + 120.0)
    if len(mask) == 0:
        return None
    last_sil_idx = None
    for idx in mask:
        if float(cls_silence[idx]) >= 0.5:
            last_sil_idx = int(idx)
    if last_sil_idx is None:
        return None
    cand_cut = float(times_ms[last_sil_idx]) + 4.0
    return max(float(pre_abs) + 12.0, min(float(cut_abs), cand_cut))


def _compute_file_anchor_adapt_stats(mel_ctx, syllables_info, ph_intervals):
    if np is None and not syllables_info:
        return None
    stats = {}
    if mel_ctx:
        en = mel_ctx.get("energy")
        f0v = mel_ctx.get("f0_voicing")
        if en is not None and len(en):
            stats["file_mean_energy"] = float(np.mean(np.asarray(en, dtype=np.float64)))
        if f0v is not None and len(f0v):
            stats["file_voiced_ratio"] = float(np.mean(np.asarray(f0v, dtype=np.float64) >= 0.5))

    lengths = []
    for syl in syllables_info or []:
        try:
            start_s = float(syl.get("start_time", 0.0) or 0.0)
            end_s = float(syl.get("end_time", 0.0) or 0.0)
            if end_s > start_s:
                lengths.append((end_s - start_s) * 1000.0)
                continue
        except Exception:
            pass
        phones = syl.get("phones") or []
        if phones:
            try:
                start_s = float(phones[0].minTime)
                end_s = float(phones[-1].maxTime)
                if end_s > start_s:
                    lengths.append((end_s - start_s) * 1000.0)
            except Exception:
                pass

    if not lengths and ph_intervals:
        for ph in ph_intervals:
            try:
                if _is_kr_nucleus_phone_mark(getattr(ph, "mark", "")):
                    lengths.append((float(ph.maxTime) - float(ph.minTime)) * 1000.0)
            except Exception:
                continue

    if lengths:
        stats["file_mean_syllable_dur_ms"] = float(sum(lengths) / max(len(lengths), 1))
    if not stats:
        return None
    return stats


def _estimate_kr_blank_confidence_at_time(mel_ctx, t_ms):
    if np is None or not mel_ctx:
        return 0.0
    times_ms = mel_ctx.get("times_ms")
    if times_ms is None or len(times_ms) == 0:
        return 0.0
    idx = _nearest_time_index(times_ms, float(t_ms))
    if idx < 0:
        return 0.0

    cls_sil = mel_ctx.get("cls_silence_sparse")
    cls_voiced = mel_ctx.get("cls_voiced_formant")
    cls_unvoiced = mel_ctx.get("cls_unvoiced_diffuse")
    cls_breath = mel_ctx.get("cls_breath_like")
    db_arr = mel_ctx.get("db_db")
    db_sil_th = float(mel_ctx.get("db_silence_th", -42.0))
    (
        rms_arr,
        rms_norm_arr,
        rms_sil_th,
        rms_sound_th,
        rms_norm_sil_th,
        rms_norm_sound_th,
    ) = _resolve_mel_rms_views(mel_ctx, len(times_ms))

    sil = float(cls_sil[idx]) if cls_sil is not None and len(cls_sil) == len(times_ms) else 0.0
    voiced = float(cls_voiced[idx]) if cls_voiced is not None and len(cls_voiced) == len(times_ms) else 0.0
    unvoiced = float(cls_unvoiced[idx]) if cls_unvoiced is not None and len(cls_unvoiced) == len(times_ms) else 0.0
    breath = float(cls_breath[idx]) if cls_breath is not None and len(cls_breath) == len(times_ms) else 0.0
    db_sparse = 0.0
    if db_arr is not None and len(db_arr) == len(times_ms):
        db_sparse = 1.0 if float(db_arr[idx]) <= (db_sil_th - 1.0) else 0.0
    rms_v = float(rms_arr[idx]) if len(rms_arr) == len(times_ms) else 0.0
    rms_norm_v = float(rms_norm_arr[idx]) if len(rms_norm_arr) == len(times_ms) else 0.0
    rms_blank = 0.0
    if rms_sil_th > 1e-8:
        rms_blank = max(0.0, min(1.0, ((rms_sil_th * 1.15) - rms_v) / max(rms_sil_th * 1.15, 1e-8)))
    rms_blank_norm = max(
        0.0,
        min(1.0, ((rms_norm_sil_th + 0.04) - rms_norm_v) / max(rms_norm_sil_th + 0.04, 1e-8)),
    )
    rms_sound = max(
        0.0,
        min(
            1.0,
            (rms_v - (rms_sound_th * 0.85)) / max(rms_sound_th - rms_sil_th, rms_sound_th * 0.50, 1e-8),
        ),
    )
    rms_sound_norm = max(
        0.0,
        min(1.0, (rms_norm_v - (rms_norm_sound_th - 0.05)) / max(1.0 - (rms_norm_sound_th - 0.05), 1e-8)),
    )
    rms_blank_mix = (0.55 * rms_blank) + (0.45 * rms_blank_norm)
    rms_sound_mix = (0.55 * rms_sound) + (0.45 * rms_sound_norm)

    blank = (
        (0.48 * sil)
        + (0.16 * breath)
        + (0.18 * db_sparse)
        + (0.24 * rms_blank_mix)
        - (0.42 * voiced)
        - (0.14 * unvoiced)
        - (0.18 * rms_sound_mix)
    )
    return max(0.0, min(1.0, float(blank)))


def _blank_conf_at(values, idx):
    try:
        if values is None:
            return 0.0
        i = int(idx)
    except Exception:
        return 0.0
    if i < 0 or i >= len(values):
        return 0.0
    try:
        return float(values[i])
    except Exception:
        return 0.0


def _estimate_kr_mel_class_scores_at_time(mel_ctx, t_ms):
    out = {
        "mel_voiced_formant_conf": 0.0,
        "mel_silence_sparse_conf": 0.0,
        "mel_unvoiced_diffuse_conf": 0.0,
        "mel_breath_like_conf": 0.0,
    }
    if np is None or not mel_ctx:
        return out
    times_ms = mel_ctx.get("times_ms")
    if times_ms is None or len(times_ms) == 0:
        return out
    idx = _nearest_time_index(times_ms, float(t_ms))
    if idx < 0:
        return out
    key_map = {
        "mel_voiced_formant_conf": "cls_voiced_formant",
        "mel_silence_sparse_conf": "cls_silence_sparse",
        "mel_unvoiced_diffuse_conf": "cls_unvoiced_diffuse",
        "mel_breath_like_conf": "cls_breath_like",
    }
    for out_key, src_key in key_map.items():
        arr = mel_ctx.get(src_key)
        if arr is not None and len(arr) == len(times_ms):
            try:
                out[out_key] = max(0.0, min(1.0, float(arr[idx])))
            except Exception:
                out[out_key] = 0.0
    return out


def _collect_kr_blank_region_stats(mel_ctx, start_ms, end_ms):
    stats = {
        "frame_count": 0,
        "blank_ratio": 0.0,
        "voiced_ratio": 0.0,
        "sound_ratio": 0.0,
        "onset_blank_ratio": 0.0,
        "onset_sound_ratio": 0.0,
    }
    if np is None or not mel_ctx:
        return stats

    times_ms = mel_ctx.get("times_ms")
    en = mel_ctx.get("energy")
    db_arr = mel_ctx.get("db_db")
    cls_sil = mel_ctx.get("cls_silence_sparse")
    cls_voiced = mel_ctx.get("cls_voiced_formant")
    db_sil_th = float(mel_ctx.get("db_silence_th", -42.0))
    if (
        times_ms is None
        or en is None
        or db_arr is None
        or len(times_ms) == 0
        or len(en) != len(times_ms)
        or len(db_arr) != len(times_ms)
    ):
        return stats

    start = float(min(start_ms, end_ms))
    end = float(max(start_ms, end_ms))
    if end <= start:
        return stats

    mask = (times_ms >= start) & (times_ms <= end)
    if not np.any(mask):
        left = _nearest_time_index(times_ms, start)
        right = _nearest_time_index(times_ms, end)
        if min(left, right) < 0:
            return stats
        if right < left:
            left, right = right, left
        mask = np.zeros(len(times_ms), dtype=bool)
        mask[left:right + 1] = True

    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return stats

    en_sel = np.asarray(en[idxs], dtype=np.float64)
    db_sel = np.asarray(db_arr[idxs], dtype=np.float64)
    if cls_sil is not None and len(cls_sil) == len(times_ms):
        sil_sel = np.asarray(cls_sil[idxs], dtype=np.float64)
    else:
        sil_sel = np.zeros_like(en_sel, dtype=np.float64)
    if cls_voiced is not None and len(cls_voiced) == len(times_ms):
        voiced_sel = np.asarray(cls_voiced[idxs], dtype=np.float64)
    else:
        voiced_sel = np.zeros_like(en_sel, dtype=np.float64)
    (
        rms_arr,
        rms_norm_arr,
        rms_sil_th,
        rms_sound_th,
        rms_norm_sil_th,
        rms_norm_sound_th,
    ) = _resolve_mel_rms_views(mel_ctx, len(times_ms))
    rms_sel = np.asarray(rms_arr[idxs], dtype=np.float64) if len(rms_arr) == len(times_ms) else np.zeros_like(en_sel)
    rms_norm_sel = (
        np.asarray(rms_norm_arr[idxs], dtype=np.float64)
        if len(rms_norm_arr) == len(times_ms)
        else np.zeros_like(en_sel)
    )

    rms_sound_mask = (
        (rms_sel >= max(rms_sound_th * 0.82, rms_sil_th * 1.06))
        | (rms_norm_sel >= max(0.42, rms_norm_sound_th * 0.75))
    )
    rms_blank_mask = (
        (rms_sel <= (rms_sil_th * 1.08))
        | (rms_norm_sel <= max(0.05, rms_norm_sil_th * 1.05))
    )

    sound_mask = (
        (((db_sel > (db_sil_th + 1.4)) & (en_sel > 0.10)) | (en_sel > 0.16) | (voiced_sel >= 0.50))
        & (rms_sound_mask | (voiced_sel >= 0.50) | (en_sel > 0.20))
        & ~((rms_blank_mask) & (db_sel <= (db_sil_th + 0.4)) & (en_sel <= 0.12))
    )
    blank_mask = (
        (sil_sel >= 0.50)
        | ((db_sel <= (db_sil_th - 1.2)) & (en_sel <= 0.10) & rms_blank_mask)
        | (rms_blank_mask & (db_sel <= (db_sil_th + 0.8)) & (en_sel <= 0.13))
    )
    voiced_mask = voiced_sel >= 0.50

    onset_end = min(end, start + 56.0)
    onset_mask = (times_ms[idxs] >= start) & (times_ms[idxs] <= onset_end)
    if not np.any(onset_mask):
        onset_mask = np.ones(len(idxs), dtype=bool)

    onset_blank_mask = blank_mask[onset_mask]
    onset_sound_mask = sound_mask[onset_mask]

    stats["frame_count"] = int(idxs.size)
    stats["blank_ratio"] = float(np.mean(blank_mask))
    stats["voiced_ratio"] = float(np.mean(voiced_mask))
    stats["sound_ratio"] = float(np.mean(sound_mask))
    stats["onset_blank_ratio"] = float(np.mean(onset_blank_mask)) if onset_blank_mask.size else 0.0
    stats["onset_sound_ratio"] = float(np.mean(onset_sound_mask)) if onset_sound_mask.size else 0.0
    return stats


def _should_veto_kr_blank_region(alias_type, candidate_stats):
    alias_key = str(alias_type or "").strip().lower()
    blank_ratio = float(candidate_stats.get("blank_ratio", 0.0))
    voiced_ratio = float(candidate_stats.get("voiced_ratio", 0.0))
    sound_ratio = float(candidate_stats.get("sound_ratio", 0.0))
    onset_blank_ratio = float(candidate_stats.get("onset_blank_ratio", 0.0))
    onset_sound_ratio = float(candidate_stats.get("onset_sound_ratio", 0.0))
    frame_count = int(candidate_stats.get("frame_count", 0) or 0)
    if frame_count <= 0:
        return False

    full_blank = blank_ratio >= 0.94 and sound_ratio <= 0.08 and voiced_ratio <= 0.06
    onset_blank = onset_blank_ratio >= 0.90 and onset_sound_ratio <= 0.08

    if alias_key in {"cv", "cv_head"}:
        return full_blank or onset_blank
    if alias_key == "vv":
        return full_blank or onset_blank or (blank_ratio >= 0.88 and voiced_ratio <= 0.10)
    if alias_key == "vc":
        return full_blank and blank_ratio >= 0.98
    return full_blank


def _guard_kr_blank_region_span(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    *,
    base_offset,
    base_consonant,
    base_cutoff,
    base_pre,
    base_ovl,
    alias_type,
    mel_ctx=None,
):
    if np is None or not mel_ctx:
        return validate_oto_params(offset, consonant, cutoff, pre, ovl), False

    alias_key = str(alias_type or "").strip().lower()
    if alias_key not in {"cv", "cv_head", "vv", "vc"}:
        return validate_oto_params(offset, consonant, cutoff, pre, ovl), False

    cand = validate_oto_params(offset, consonant, cutoff, pre, ovl, alias_type=alias_key)
    base = validate_oto_params(base_offset, base_consonant, base_cutoff, base_pre, base_ovl, alias_type=alias_key)

    cand_start = float(cand[0])
    cand_end = float(cand[0]) + abs(float(cand[2]))
    base_start = float(base[0])
    base_end = float(base[0]) + abs(float(base[2]))

    cand_stats = _collect_kr_blank_region_stats(mel_ctx, cand_start, cand_end)
    if not _should_veto_kr_blank_region(alias_key, cand_stats):
        return cand, False

    base_stats = _collect_kr_blank_region_stats(mel_ctx, base_start, base_end)
    base_bad = _should_veto_kr_blank_region(alias_key, base_stats)
    if not base_bad:
        return base, True

    # If both spans are poor, prefer the one with more audible support.
    cand_score = (
        float(cand_stats.get("sound_ratio", 0.0))
        + float(cand_stats.get("voiced_ratio", 0.0))
        - float(cand_stats.get("blank_ratio", 0.0))
    )
    base_score = (
        float(base_stats.get("sound_ratio", 0.0))
        + float(base_stats.get("voiced_ratio", 0.0))
        - float(base_stats.get("blank_ratio", 0.0))
    )
    return (base, True) if base_score >= cand_score else (cand, False)


def _annotate_kr_syllable_blank_confidence(syllables_info, mel_ctx):
    if not syllables_info:
        return syllables_info
    for syl in syllables_info:
        start_s = float(syl.get("start_time", 0.0) or 0.0)
        phones = syl.get("phones") or []
        if start_s <= 0.0 and phones:
            try:
                start_s = float(phones[0].minTime)
            except Exception:
                start_s = 0.0
        if start_s < 0.0:
            start_s = 0.0
        t_ms = start_s * 1000.0
        syl["blank_confidence"] = _estimate_kr_blank_confidence_at_time(mel_ctx, t_ms)
        mel_scores = _estimate_kr_mel_class_scores_at_time(mel_ctx, t_ms)
        syl.update(mel_scores)
    return syllables_info


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
    f2_arr = mel_ctx.get("f2_ratio")
    f3_arr = mel_ctx.get("f3_ratio")
    high_arr = mel_ctx.get("high_ratio")
    spec_presence_arr = mel_ctx.get("spec_presence")
    db_sil_th = float(mel_ctx.get("db_silence_th", -42.0))
    if t_ms is None or en is None or len(t_ms) < 8 or len(en) != len(t_ms):
        return offset, consonant, cutoff, pre, ovl, 0.0, 0.0
    if db_arr is None or len(db_arr) != len(en):
        db_arr = np.zeros_like(en, dtype=np.float64)
    if f0v_arr is None or len(f0v_arr) != len(en):
        f0v_arr = np.zeros_like(en, dtype=np.float64)
    if f2_arr is None or len(f2_arr) != len(en):
        f2_arr = np.zeros_like(en, dtype=np.float64)
    if f3_arr is None or len(f3_arr) != len(en):
        f3_arr = np.zeros_like(en, dtype=np.float64)
    if high_arr is None or len(high_arr) != len(en):
        high_arr = np.zeros_like(en, dtype=np.float64)
    if spec_presence_arr is None or len(spec_presence_arr) != len(en):
        spec_presence_arr = np.asarray(en, dtype=np.float64)
    (
        rms_arr,
        rms_norm_arr,
        rms_sil_th,
        rms_sound_th,
        rms_norm_sil_th,
        rms_norm_sound_th,
    ) = _resolve_mel_rms_views(mel_ctx, len(en))

    f2_p60 = float(mel_ctx.get("f2_p60", np.percentile(f2_arr, 60) if len(f2_arr) else 0.0))
    f3_p60 = float(mel_ctx.get("f3_p60", np.percentile(f3_arr, 60) if len(f3_arr) else 0.0))
    high_p65 = float(mel_ctx.get("high_p65", np.percentile(high_arr, 65) if len(high_arr) else 0.0))
    spec_p40 = float(mel_ctx.get("spec_p40", np.percentile(spec_presence_arr, 40) if len(spec_presence_arr) else 0.0))

    base_f2_strong = _env_float("UTOA_MEL_F2_STRONG_MIN", 0.115)
    base_f3_strong = _env_float("UTOA_MEL_F3_STRONG_MIN", 0.082)
    base_high_noise = _env_float("UTOA_MEL_HIGH_NOISE_MIN", 0.20)
    base_spec_noise = _env_float("UTOA_MEL_SPEC_NOISE_MIN", 0.14)
    base_sound_db_margin = _env_float("UTOA_MEL_SOUND_DB_MARGIN", 1.7)
    base_sound_energy = _env_float("UTOA_MEL_SOUND_ENERGY_MIN", 0.15)
    base_weak_f2 = _env_float("UTOA_MEL_WEAK_F2_MAX", 0.05)
    base_weak_f3 = _env_float("UTOA_MEL_WEAK_F3_MAX", 0.05)
    base_weak_high = _env_float("UTOA_MEL_WEAK_HIGH_MAX", 0.10)
    hard_sil_db_margin = _env_float("UTOA_MEL_HARD_SILENCE_DB_MARGIN", -3.0)
    hard_sil_energy = _env_float("UTOA_MEL_HARD_SILENCE_ENERGY_MAX", 0.09)
    soft_sil_energy = _env_float("UTOA_MEL_SOFT_SILENCE_ENERGY_MAX", 0.11)
    noise_db_margin = _env_float("UTOA_MEL_NOISE_DB_MARGIN", 1.0)

    # Adapt thresholds slightly to per-file spectral distribution.
    f2_strong_th = max(0.05, min(0.25, (0.7 * base_f2_strong) + (0.3 * f2_p60)))
    f3_strong_th = max(0.04, min(0.22, (0.7 * base_f3_strong) + (0.3 * f3_p60)))
    high_noise_th = max(0.10, min(0.35, (0.7 * base_high_noise) + (0.3 * high_p65)))
    spec_noise_th = max(0.06, min(0.32, (0.7 * base_spec_noise) + (0.3 * spec_p40)))

    pre_abs = float(offset) + float(pre)
    cons_abs = float(offset) + float(consonant)
    cut_abs = float(offset) + abs(float(cutoff))
    if cut_abs <= pre_abs + 18.0:
        return offset, consonant, cutoff, pre, ovl, 0.0, 0.0

    # Spectral cues:
    # - voiced_formant_mask: strong F2/F3 (+ optional F0) -> likely vowel/voiced region
    # - noisy_unvoiced_mask: high-frequency energy with weak formant -> fricative/plosive noise
    cls_voiced = mel_ctx.get("cls_voiced_formant")
    cls_unvoiced = mel_ctx.get("cls_unvoiced_diffuse")
    cls_silence = mel_ctx.get("cls_silence_sparse")
    if cls_voiced is not None and len(cls_voiced) == len(en):
        voiced_formant_mask = np.asarray(cls_voiced, dtype=np.float64) >= 0.5
    else:
        voiced_f2_f0_th = _env_float("UTOA_MEL_VOICED_F2_MIN", 0.09)
        voiced_f0_th = _env_float("UTOA_MEL_VOICED_F0_MIN", 0.42)
        voiced_formant_mask = (
            ((f2_arr > f2_strong_th) & (f3_arr > f3_strong_th))
            | ((f2_arr > voiced_f2_f0_th) & (f0v_arr > voiced_f0_th))
        )
    if cls_unvoiced is not None and len(cls_unvoiced) == len(en):
        noisy_unvoiced_mask = np.asarray(cls_unvoiced, dtype=np.float64) >= 0.5
    else:
        noisy_unvoiced_mask = (
            (high_arr > high_noise_th)
            & (spec_presence_arr > spec_noise_th)
            & (db_arr > (db_sil_th + noise_db_margin))
            & (en > 0.08)
            & ~voiced_formant_mask
        )

    rms_sound_mask = (
        (rms_arr >= max(rms_sound_th * 0.82, rms_sil_th * 1.08))
        | (rms_norm_arr >= max(0.40, rms_norm_sound_th * 0.74))
    )
    rms_blank_mask = (
        (rms_arr <= (rms_sil_th * 1.08))
        | (rms_norm_arr <= max(0.05, rms_norm_sil_th * 1.05))
    )
    sound_mask = (
        (((db_arr > (db_sil_th + base_sound_db_margin)) & (en > base_sound_energy)) | voiced_formant_mask | noisy_unvoiced_mask)
        & (~rms_blank_mask | voiced_formant_mask | noisy_unvoiced_mask)
    ) | rms_sound_mask
    weak_spec_mask = (f2_arr < base_weak_f2) & (f3_arr < base_weak_f3) & (high_arr < base_weak_high)
    hard_silence_mask = (db_arr <= (db_sil_th + hard_sil_db_margin)) | (en <= hard_sil_energy)
    silence_mask = hard_silence_mask | (((db_arr <= db_sil_th) | (en <= soft_sil_energy)) & weak_spec_mask)
    silence_mask = silence_mask | (rms_blank_mask & (db_arr <= (db_sil_th + 0.8)) & (en <= max(soft_sil_energy + 0.02, 0.13)))
    if cls_silence is not None and len(cls_silence) == len(en):
        silence_mask = silence_mask | (np.asarray(cls_silence, dtype=np.float64) >= 0.5)
    # onset 보조 탐지: 에너지 이동평균 + 1차 기울기로 유효 onset 후보를 만든다.
    if len(en) >= 5:
        kernel = np.ones(5, dtype=np.float64) / 5.0
        en_ma = np.convolve(en, kernel, mode="same")
    else:
        en_ma = np.asarray(en, dtype=np.float64)
    en_slope = np.diff(en_ma, prepend=float(en_ma[0]) if len(en_ma) else 0.0)

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
    hint = re.sub(r"[^a-z]", "", hint or "")
    if len(hint) >= 2 and hint[:2] in (KR_PLOSIVE_ONSETS | KR_SIBILANT_ONSETS):
        hint = hint[:2]
    elif len(hint) >= 1:
        hint = hint[:1] if hint[:1] in (KR_PLOSIVE_ONSETS | KR_SIBILANT_ONSETS) else hint

    is_plosive_like = bool(hint) and (is_plosive_roman(hint) or is_plosive_ipa(hint) or hint in KR_PLOSIVE_ONSETS)
    is_sibilant_like = hint in KR_SIBILANT_ONSETS or hint in {"sh", "ch", "ts", "z", "dz", "j", "jj", "c"}
    is_fricative_like = is_sibilant_like or hint in {"h", "f", "v", "x"}
    # 유성/비음 계열(m,n,r,l,w,y...)은 멜 저역 에너지가 약해
    # offset guard가 모음 시작으로 과도 이동할 수 있어 보수적으로 처리한다.
    low_energy_voiced = hint in {
        "m", "n", "ny", "ng", "r", "l", "ry", "w", "y", "j",
        "g", "d", "b", "z", "dz", "v", "gy", "dy", "by",
        "ɴ", "ŋ", "ɲ", "ɾ", "ɹ",
    } or hint.startswith("m")

    # Onset-class aware onset mask tuning:
    # - sibilant/fricative: flatter slope before vowel, so lower slope threshold
    # - plosive: sharper attack, so keep stricter slope threshold
    onset_slope_th = _env_float("UTOA_MEL_ONSET_SLOPE_BASE", 0.016)
    onset_energy_th = _env_float("UTOA_MEL_ONSET_ENERGY_BASE", 0.125)
    onset_db_margin = _env_float("UTOA_MEL_ONSET_DB_MARGIN_BASE", -1.5)
    if is_sibilant_like:
        onset_slope_th = _env_float("UTOA_MEL_ONSET_SLOPE_SIBILANT", 0.011)
        onset_energy_th = _env_float("UTOA_MEL_ONSET_ENERGY_SIBILANT", 0.11)
        onset_db_margin = _env_float("UTOA_MEL_ONSET_DB_MARGIN_SIBILANT", -3.5)
    elif is_plosive_like:
        onset_slope_th = _env_float("UTOA_MEL_ONSET_SLOPE_PLOSIVE", 0.019)
        onset_energy_th = _env_float("UTOA_MEL_ONSET_ENERGY_PLOSIVE", 0.14)
        onset_db_margin = _env_float("UTOA_MEL_ONSET_DB_MARGIN_PLOSIVE", -1.2)
    onset_mask = (en_slope > onset_slope_th) & (en_ma > onset_energy_th) & (db_arr > (db_sil_th + onset_db_margin))
    if is_sibilant_like or is_plosive_like:
        f0_slope_th = _env_float("UTOA_MEL_ONSET_F0_SLOPE_MIN", 0.03)
        onset_energy_relax = _env_float("UTOA_MEL_ONSET_ENERGY_RELAX", 0.02)
        onset_f0_db_margin = _env_float("UTOA_MEL_ONSET_F0_DB_MARGIN", -2.5)
        f0_slope = np.diff(f0v_arr, prepend=float(f0v_arr[0]) if len(f0v_arr) else 0.0)
        onset_mask = onset_mask | (
            (f0_slope > f0_slope_th)
            & (en_ma > max(0.09, onset_energy_th - onset_energy_relax))
            & (db_arr > (db_sil_th + onset_f0_db_margin))
        )

    # ---- soft offset guard ----
    # 한국어 CVVC의 CV/CV_HEAD는 onset anchor와 후단 guard만으로도 충분한 경우가 많다.
    # 멜 offset soft guard가 추가로 들어가면 단모음/활음 구분이 약한 파일에서
    # offset이 공백 쪽으로 과하게 끌리는 경향이 커진다.
    order_locked = _is_kr_order_locked_cv_format(file_format)
    allow_order_locked_cv = _env_bool("UTOA_KR_ALLOW_ORDER_LOCKED_CV_MEL_OFFSET", True)
    enable_cv_head_soft_guard = _env_bool(
        "UTOA_KR_ENABLE_CV_HEAD_SOFT_MEL_OFFSET_GUARD",
        not order_locked,
    )
    skip_offset_soft_guard = ((alias_type == "cv_head") and (not enable_cv_head_soft_guard)) or (
        order_locked
        and alias_type in {"cv", "cv_head"}
        and not allow_order_locked_cv
    )
    if not skip_offset_soft_guard and not low_energy_voiced:
        off_silent = bool(silence_mask[off_idx])
        pre_sound = bool(sound_mask[pre_idx] or (en[pre_idx] > 0.20))
        if off_silent and pre_sound:
            lo = max(0, pre_idx - 120)
            sound_start_idx = None
            if is_sibilant_like or is_plosive_like:
                # For fricative/plosive-leading syllables, prioritize the first strong formant region
                # (vowel body) to separate C-noise from V more reliably.
                vf_seg = voiced_formant_mask[lo:pre_idx + 1]
                if np.any(vf_seg):
                    rel = int(np.where(vf_seg)[0][0])
                    sound_start_idx = lo + rel
            if sound_start_idx is None:
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
                offset_lead_ms = _env_float("UTOA_MEL_OFFSET_LEAD_BASE_MS", 9.0)
                pre_guard_ms = _env_float("UTOA_MEL_OFFSET_PRE_GUARD_BASE_MS", 18.0)
                if is_sibilant_like or is_fricative_like:
                    offset_lead_ms = _env_float("UTOA_MEL_OFFSET_LEAD_SIBILANT_MS", 12.0)
                    pre_guard_ms = _env_float("UTOA_MEL_OFFSET_PRE_GUARD_SIBILANT_MS", 20.0)
                elif is_plosive_like:
                    offset_lead_ms = _env_float("UTOA_MEL_OFFSET_LEAD_PLOSIVE_MS", 14.0)
                    pre_guard_ms = _env_float("UTOA_MEL_OFFSET_PRE_GUARD_PLOSIVE_MS", 22.0)
                offset_blend = _env_float("UTOA_MEL_OFFSET_BLEND_BASE", 0.36)
                if order_locked and alias_type == "cv_head":
                    offset_lead_ms = max(6.0, offset_lead_ms - 2.0)
                    pre_guard_ms += _env_float("UTOA_MEL_OFFSET_PRE_GUARD_CV_HEAD_ORDER_LOCKED_MS", 2.0)
                    offset_blend = _env_float("UTOA_MEL_OFFSET_BLEND_CV_HEAD_ORDER_LOCKED", 0.26)
                elif order_locked and alias_type == "cv":
                    offset_blend = _env_float("UTOA_MEL_OFFSET_BLEND_CV_ORDER_LOCKED", 0.31)

                target_offset = float(t_ms[sound_start_idx]) - offset_lead_ms
                target_offset = max(0.0, min(pre_abs - pre_guard_ms, target_offset))
                new_offset = _blend(offset, target_offset, offset_blend)
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
                cut_shift_base = _env_float("UTOA_MEL_CUTOFF_TARGET_SHIFT_MS", 4.0)
                min_cut_abs_margin = _env_float("UTOA_MEL_CUTOFF_MIN_FROM_PRE_MS", 20.0)
                target_cut_abs = float(t_ms[last_sil_idx]) + cut_shift_base
                target_cut_abs = max(pre_abs + min_cut_abs_margin, min(target_cut_abs, cut_abs))
                min_cut_reduction = _env_float("UTOA_MEL_MIN_CUT_REDUCTION_BASE_MS", 8.0)
                if is_sibilant_like:
                    min_cut_reduction = _env_float("UTOA_MEL_MIN_CUT_REDUCTION_SIBILANT_MS", 12.0)
                elif is_plosive_like:
                    min_cut_reduction = _env_float("UTOA_MEL_MIN_CUT_REDUCTION_PLOSIVE_MS", 10.0)
                if target_cut_abs < cut_abs - min_cut_reduction:
                    # F0 유성도는 보조(저가중치)로만 반영
                    f0v = float(f0v_arr[last_sil_idx])
                    blend_base = _env_float("UTOA_MEL_CUTOFF_BLEND_BASE", 0.42)
                    blend_f0_scale = _env_float("UTOA_MEL_CUTOFF_BLEND_F0_SCALE", 0.08)
                    blend_w = blend_base - (blend_f0_scale * f0v)
                    if is_sibilant_like:
                        blend_w -= _env_float("UTOA_MEL_CUTOFF_BLEND_SIBILANT_DELTA", 0.08)
                    elif is_plosive_like:
                        blend_w -= _env_float("UTOA_MEL_CUTOFF_BLEND_PLOSIVE_DELTA", 0.04)
                    blend_min = _env_float("UTOA_MEL_CUTOFF_BLEND_MIN", 0.26)
                    blend_max = _env_float("UTOA_MEL_CUTOFF_BLEND_MAX", 0.44)
                    blend_w = max(blend_min, min(blend_max, blend_w))
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
                bucket_keys = _build_kr_profile_bucket_keys(row["alias"], alias_type)
                if not bucket_keys:
                    continue
                pre = max(row["pre"], 0.0)
                cons = max(row["cons"], pre)
                cut_abs = abs(row["cutoff"])
                for bucket_key in bucket_keys:
                    b = bucket_values.setdefault(bucket_key, {
                        "pre": [],
                        "cons_gap": [],
                        "cut_gap": [],
                        "ovl_ratio": [],
                        "head_off_ratio": [],
                    })
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
    for alias_key, vals in bucket_values.items():
        n = len(vals["pre"])
        if n < 8:
            continue
        buckets[alias_key] = {
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
        "version": 2,
        "language": "korean",
        "bucket_key_mode": "alias+phonetic",
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
    kr_mapping_spn_ratio_threshold=0.30,
    kr_mapping_min_vowel_phone_ratio=0.56,
    kr_mapping_debug_reason_logging=True,
    kr_anchor_profile_path="",
    kr_mapping_confidence_threshold=None,
    kr_mapping_max_index_jump_default=0,
    kr_mapping_max_index_jump_high_conf=1,
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
        initialize_runtime_report(runtime_report, language="korean", stage="generate")
        finalize_runtime_report(runtime_report, processed=0, total=0, errors=[err])
        return 0, 0, [err]

    normalized_options = normalize_korean_generation_options(
        params=params,
        default_params=DEFAULT_PARAMS,
        fallback_format=fallback_format,
        auto_format=auto_format,
        kr_mapping_words_fallback_enabled=kr_mapping_words_fallback_enabled,
        kr_mapping_spn_ratio_threshold=kr_mapping_spn_ratio_threshold,
        kr_mapping_min_vowel_phone_ratio=kr_mapping_min_vowel_phone_ratio,
        kr_mapping_debug_reason_logging=kr_mapping_debug_reason_logging,
        kr_anchor_profile_path=kr_anchor_profile_path,
        kr_mapping_confidence_threshold=kr_mapping_confidence_threshold,
        kr_mapping_max_index_jump_default=kr_mapping_max_index_jump_default,
        kr_mapping_max_index_jump_high_conf=kr_mapping_max_index_jump_high_conf,
        cleanup_timing_jsonl=cleanup_timing_jsonl,
        callback=callback,
    )
    params = normalized_options.params
    fallback_format = normalized_options.fallback_format
    auto_gen_format = normalized_options.auto_gen_format
    use_sinsy_labels = normalized_options.use_sinsy_labels
    sinsy_label_path = normalized_options.sinsy_label_path
    kr_mapping_words_fallback_enabled = normalized_options.words_fallback_enabled
    kr_mapping_spn_ratio_threshold = normalized_options.spn_ratio_threshold
    kr_mapping_min_vowel_phone_ratio = normalized_options.min_vowel_phone_ratio
    kr_mapping_debug_reason_logging = normalized_options.debug_reason_logging
    kr_anchor_profile_path = normalized_options.anchor_profile_path
    kr_mapping_confidence_threshold = normalized_options.confidence_threshold
    kr_mapping_max_index_jump_default = normalized_options.max_index_jump_default
    kr_mapping_max_index_jump_high_conf = normalized_options.max_index_jump_high_conf
    cleanup_timing_jsonl = normalized_options.cleanup_timing_jsonl
    kr_disable_cvvc_order_lock = normalized_options.disable_cvvc_order_lock
    initialize_runtime_report(runtime_report, language="korean", stage="generate")
    generation_request = GenerationRequest(
        language="korean",
        tg_folder=str(tg_folder or "").strip(),
        tpl_path=str(tpl_path or "").strip(),
        out_path=str(out_path or "").strip(),
        fallback_format=str(fallback_format or "").strip().lower(),
        custom_phonemes_path=str(custom_phonemes_path or "").strip(),
        alias_suffix=str(alias_suffix or "").strip(),
        auto_format=str(auto_format or "").strip(),
        params=dict(params or {}),
        options={
            "auto_gen_format": auto_gen_format,
            "ml_policy": str(ml_policy or "").strip(),
            "enable_ml_correction": bool(enable_ml_correction),
        },
    )
    attach_request_metadata(runtime_report, generation_request)

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
    file_anchor_adapt_stats = None
    file_anchor_profile_cache: Dict[tuple, object] = {}
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
        voiced_onset_ms: float | None = None,
        mapping_confidence: float = 1.0,
        lite: bool = False,
    ):
        def _get_profile(lang, fmt, alias_kind):
            from core.timing_anchor_profiles import get_anchor_profile

            base = get_anchor_profile(lang, fmt, alias_kind, mode="rhythm_stable")
            if base is None or not file_anchor_adapt_stats:
                return base
            key = (fmt, alias_kind)
            cached = file_anchor_profile_cache.get(key)
            if cached is None:
                from core.timing_anchor_runtime import adapt_profile_to_file

                cached = adapt_profile_to_file(base, **file_anchor_adapt_stats)
                file_anchor_profile_cache[key] = cached
            return cached

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
            voiced_onset_ms=voiced_onset_ms,
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
                bridge_end_rel = next_onset_rel
                if next_vowel_abs_ms is not None:
                    try:
                        bridge_end_rel = max(bridge_end_rel, float(next_vowel_abs_ms) - o)
                    except Exception:
                        pass
                c_cap = bridge_end_rel - 8.0
                c = min(c, c_cap)
                c = max(c, p + 8.0)
                cut_abs = abs(cut)
                cut_cap = bridge_end_rel - 2.0
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
    template_encoding = ""
    if tpl_path:
        lines, detected_enc, warning, err = load_template_oto_lines(
            tpl_path,
            require_utf8=False,
            mode_label="한국어 OTO",
        )
        if err:
            log(err)
            log(f"⚡ 템플릿 로드 실패로 OpenUtau 호환 {auto_gen_format.upper()} 자동 에일리어스 생성으로 전환합니다.")
            lines = []
            detected_enc = ""
        if warning:
            log(warning)
        template_lines = lines or []
        template_encoding = str(detected_enc or "").strip()


    wav_root_for_signal = os.path.dirname(os.path.abspath(tg_folder.rstrip("\\/")))
    wav_index_for_signal = _build_wav_index(wav_root_for_signal)

    tg_entries = []
    tg_exact_map = {}
    tg_norm_map = {}
    if os.path.exists(tg_folder):
        for dirpath, f_name in _iter_textgrid_files(tg_folder):
            base = os.path.splitext(f_name)[0]
            output_wav_name = _resolve_real_wav_name_for_textgrid(
                f_name,
                wav_root_for_signal,
                wav_index_for_signal,
            )
            info = {
                'path': os.path.join(dirpath, f_name),
                'real_name': base + '.wav',
                'output_name': output_wav_name,
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
    wav_name_map = {}


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
    if _should_keep_template_alias_set_exact(
        use_template=use_template,
        generate_openutau=generate_openutau,
        gen_missing_vowels=gen_missing_vowels,
    ):
        log("📌 템플릿 모드: 추가 alias 생성 없이 베이스 OTO alias 집합을 그대로 유지")

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
        output_wav_name = str(getattr(file_ctx, "output_wav_name", "") or "")
        if output_wav_name:
            wav_name_map.setdefault(fname, output_wav_name)
            if file_ctx.real_wav_name and file_ctx.real_wav_name != fname:
                wav_name_map.setdefault(file_ctx.real_wav_name, output_wav_name)
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

        if handle_kr_file_context_status(
            file_ctx=file_ctx,
            fname=fname,
            lines=lines,
            final_lines=final_lines,
            alias_suffix=alias_suffix,
            log_fn=log,
            record_unset_lines_fn=_record_unset_lines,
            apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line,
        ):
            processed += 1
            continue

        file_ctx = load_named_tiers(
            file_ctx,
            load_textgrid_fn=textgrid.TextGrid.fromFile,
            tier_predicate=lambda tier: isinstance(tier, textgrid.IntervalTier),
        )
        if handle_kr_file_context_status(
            file_ctx=file_ctx,
            fname=fname,
            lines=lines,
            final_lines=final_lines,
            alias_suffix=alias_suffix,
            log_fn=log,
            record_unset_lines_fn=_record_unset_lines,
            apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line,
        ):
            processed += 1
            continue

        real_wav_name = file_ctx.real_wav_name
        output_wav_name = output_wav_name or real_wav_name
        mel_ctx_for_file = file_ctx.mel_ctx_for_file
        wav_duration_ms = float(file_ctx.wav_duration_ms or 0.0)
        tg = file_ctx.tg
        phone_tier = file_ctx.phone_tier
        word_tier = file_ctx.word_tier

        try:
            if not phone_tier:
                append_preserved_lines(
                    final_lines,
                    lines,
                    alias_suffix=alias_suffix,
                    apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line,
                )
                log(f"경고: {fname}: phones tier가 없어 원본 라인을 유지합니다.")
                _record_unset_lines("tier_missing", fname, lines)
                processed += 1
                continue

            try:
                strict_skip_aliases = []
                strict_skip_alias_seen = set()

                def _append_strict_skip_alias(alias_name):
                    name = str(alias_name or "").strip()
                    if not name or name in strict_skip_alias_seen:
                        return
                    strict_skip_alias_seen.add(name)
                    strict_skip_aliases.append(name)

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
            if handle_kr_loop_prep_status(
                loop_prep=loop_prep,
                fname=fname,
                lines=lines,
                final_lines=final_lines,
                alias_suffix=alias_suffix,
                log_fn=log,
                record_unset_lines_fn=_record_unset_lines,
                apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line,
            ):
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
            alignment_weight = max(0.0, min(1.0, textgrid_trust_score * 0.85))
            if filename_cv_targets and alignment_weight < 0.55:
                prefer_filename_sequence = True
                targets_for_build = list(filename_cv_targets)

            if try_handle_kr_single_vowel_file(
                fname=fname,
                lines=lines,
                real_wav_name=output_wav_name,
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
            kr_source_state = _select_kr_syllable_source_state(
                file_format=file_format,
                prefer_filename_sequence=prefer_filename_sequence,
                low_phone_quality=low_phone_quality,
                wd_intervals=wd_intervals,
                ph_intervals=ph_intervals,
                targets_for_build=targets_for_build,
                force_words_phone_fill=force_words_phone_fill,
                decompose_hangul_to_roman_fn=decompose_hangul_to_roman,
                mel_ctx_for_file=mel_ctx_for_file,
                textgrid_trust_score=textgrid_trust_score,
            )
            syllables_info = list(kr_source_state.get("syllables_info") or [])
            used_words_based = bool(kr_source_state.get("used_words_based"))
            used_alias_based = bool(kr_source_state.get("used_alias_based"))
            mapping_reason_code = str(kr_source_state.get("mapping_reason_code") or COMMON_REASON_FILENAME_TOKEN)
            mapping_reason_code = normalize_mapping_reason_code(
                mapping_reason_code,
                language="kr",
                fallback_code=COMMON_REASON_FILENAME_TOKEN,
            )
            base_score = float(kr_source_state.get("base_score", 0.0) or 0.0)
            alt_score = float(kr_source_state.get("alt_score", 0.0) or 0.0)
            words_glide_mismatch_ratio = float(kr_source_state.get("words_glide_mismatch_ratio", 0.0) or 0.0)
            blank_conf_mean = float(kr_source_state.get("blank_conf_mean", 0.0) or 0.0)
            alignment_weight = float(kr_source_state.get("alignment_weight", 0.0) or 0.0)
            anchor_lock_lite = bool(kr_source_state.get("anchor_lock_lite"))
            syllable_blank_confidences = list(kr_source_state.get("syllable_blank_confidences") or [])

            _log_kr_mapping_selection(
                log_fn=log,
                fname=fname,
                mapping_reason_code=mapping_reason_code,
                base_score=base_score,
                alt_score=alt_score,
                words_glide_mismatch_ratio=words_glide_mismatch_ratio,
                low_quality_reasons=low_quality_reasons,
                alias_syllables_present=bool(kr_source_state.get("alias_syllables_present")),
                targets_for_build=targets_for_build,
                textgrid_trust_tier=textgrid_trust_tier,
                textgrid_trust_score=textgrid_trust_score,
                alignment_weight=alignment_weight,
                blank_conf_mean=blank_conf_mean,
                anchor_lock_lite=anchor_lock_lite,
            )

            file_anchor_adapt_stats = _compute_file_anchor_adapt_stats(
                mel_ctx_for_file,
                syllables_info,
                ph_intervals,
            )
            file_anchor_profile_cache = {}

            plan_candidate_source = list(
                filename_cv_targets
                or [s.get('roman_cv') or s.get('roman', '') for s in (syllables_info or [])]
                or []
            )
            runtime_state = _recompute_kr_runtime_state(
                alignment_ingest=alignment_ingest,
                file_mapping_conf_th=file_mapping_conf_th,
                file_format=file_format,
                phone_quality=phone_quality,
                base_score=base_score,
                alt_score=alt_score,
                used_words_based=used_words_based,
                used_alias_based=used_alias_based,
                mapping_reason_code=mapping_reason_code,
                kr_mapping_debug_reason_logging=kr_mapping_debug_reason_logging,
                fname=fname,
                blank_conf_mean=blank_conf_mean,
                syllable_blank_confidences=syllable_blank_confidences,
                plan_candidate_source=plan_candidate_source,
                sinsy_label_entries=sinsy_label_entries,
                syllables_info=syllables_info,
                textgrid_trust_tier=textgrid_trust_tier,
                low_phone_quality=low_phone_quality,
                alignment_weight=alignment_weight,
                mel_ctx_for_file=mel_ctx_for_file,
                prefer_filename_sequence=prefer_filename_sequence,
                log_fn=log,
            )
            runtime_view = extract_language_runtime_state(
                runtime_state,
                language_prefix="kr",
                conf_floor_default=file_mapping_conf_th,
                row_margin_floor_default=6.0,
            )
            runtime_policy = dict(runtime_view["runtime_policy"] or {})
            mapping_confidence_base = float(runtime_view["mapping_confidence_base"])
            mapping_margin = float(runtime_view["mapping_margin"])
            kr_cv_plan = dict(runtime_view["kr_cv_plan"] or {})
            kr_planned_cv_indices = runtime_view["kr_planned_cv_indices"]
            kr_anchor_graph = runtime_view["kr_anchor_graph"]
            kr_plan_policy = dict(runtime_view["kr_plan_policy"] or {})
            file_mapping_low_conf = bool(runtime_view["file_mapping_low_conf"])
            file_conf_floor = float(runtime_view["file_conf_floor"])
            row_conf_floor = float(runtime_view["row_conf_floor"])
            row_margin_floor = float(runtime_view["row_margin_floor"])
            row_blank_floor = runtime_view.get("row_blank_floor")
            low_conf_reasons = list(runtime_view.get("low_conf_reasons") or [])
            mapping_tier = str(runtime_view.get("mapping_tier") or runtime_policy.get("mapping_tier") or "")
            update_kr_mapping_runtime_report(
                runtime_report,
                file_format=str(file_format or ""),
                mapping_confidence=float(mapping_confidence_base),
                mapping_margin=float(mapping_margin),
                mapping_tier=str(runtime_policy.get("mapping_tier") or ""),
                trust_score=float(textgrid_trust_score),
                trust_tier=str(textgrid_trust_tier or ""),
                file_conf_floor=float(file_conf_floor),
                row_conf_floor=float(row_conf_floor),
                row_margin_floor=float(row_margin_floor),
                file_low_conf=bool(file_mapping_low_conf),
                low_conf_reasons=list(low_conf_reasons),
                blank_confidence_mean=float(blank_conf_mean),
                plan_source=str(kr_cv_plan.get("source") or ""),
                plan_margin=float((kr_cv_plan.get("meta") or {}).get("margin", 0.0) or 0.0),
                plan_coverage=float(kr_plan_policy.get("coverage", 0.0) or 0.0),
                mapping_reason_code=str(mapping_reason_code or ""),
                row_blank_floor=(float(row_blank_floor) if row_blank_floor is not None else None),
                vc_bridge_tuning={},
            )

            if (not syllables_info) or any(len(s['phones']) == 0 for s in syllables_info):
                fail_reason = resolve_mapping_failure_reason(
                    low_quality_reasons=low_quality_reasons,
                    low_phone_quality=bool(low_phone_quality),
                    has_word_intervals=bool(wd_intervals),
                    has_phone_intervals=True,
                )
                handle_mapping_failure_fallback(
                    fail_reason=fail_reason,
                    fname=fname,
                    lines=lines,
                    final_lines=final_lines,
                    alias_suffix=alias_suffix,
                    log_fn=log,
                    failure_log_message=f"경고: {fname}: 음절-음소 매핑 실패로 원본 라인을 유지합니다.",
                    record_unset_lines_fn=_record_unset_lines,
                    meta=build_mapping_failure_meta(
                        mapping_confidence=mapping_confidence_base,
                        mapping_reason_code=mapping_reason_code,
                        spn_ratio=spn_ratio,
                        include_reason_in_diag=False,
                        extra={
                            "phone_quality": phone_quality,
                            "force_words_phone_fill": force_words_phone_fill,
                        },
                    ),
                    apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line,
                )
                processed += 1
                continue

            if bool(runtime_policy.get("should_abstain")):
                handle_mapping_abstain_fallback(
                    fname=fname,
                    lines=lines,
                    final_lines=final_lines,
                    alias_suffix=alias_suffix,
                    log_fn=log,
                    abstain_log_message=(
                        f"경고: {fname}: KR v2 planner abstain "
                        f"(trust={textgrid_trust_score:.2f}, weight={alignment_weight:.2f}, "
                        f"coverage={float(kr_plan_policy.get('coverage', 0.0)):.2f}, "
                        f"margin={float(kr_plan_policy.get('margin', 0.0)):.1f}) → 원본 유지"
                    ),
                    record_unset_lines_fn=_record_unset_lines,
                    meta=build_mapping_abstain_meta(
                        mapping_confidence=mapping_confidence_base,
                        mapping_reason_code=mapping_reason_code,
                        plan_policy=kr_plan_policy,
                    ),
                    apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line,
                )
                processed += 1
                continue

            romaji_syllables = [s.get('roman_cv') or s.get('roman', '') for s in syllables_info]
            current_w_idx = 0
            cv_seq_idx = 0
            bridge_seq_idx = 0
            kr_order_locked_format = _is_kr_sequence_locked_mapping_format(file_format)
            if kr_order_locked_format and kr_disable_cvvc_order_lock:
                kr_order_locked_format = False
                if kr_mapping_debug_reason_logging:
                    log(f"🧭 {fname}: KR CV-family filename order lock 비활성화(UTOA_KR_DISABLE_CVVC_ORDER_LOCK=1)")
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
            kr_vc_bridge_tuning = derive_cv_bridge_tuning(
                realized_anchor_by_idx=realized_cv_anchor_by_idx,
                estimated_anchor_by_idx=cv_anchor_by_idx,
                trust_tier=textgrid_trust_tier,
                mapping_tier=mapping_tier,
                min_samples=8,
            )
            kr_vc_ab_stats = _new_vc_bridge_ab_stats()
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
            kr_token_invariant_mode = _resolve_syllable_strict_mode(file_format=file_format)
            kr_token_invariant_enabled = kr_token_invariant_mode in {"soft", "strict"}
            kr_token_invariant_hard = kr_token_invariant_mode == "strict"
            kr_token_invariant_soft = kr_token_invariant_mode == "soft"
            expected_cv_count = len([t for t in (filename_cv_targets or []) if str(t or "").strip()])
            if expected_cv_count <= 0:
                expected_cv_count = len([t for t in (romaji_syllables or []) if str(t or "").strip()])
            observed_cv_count = len([s for s in (syllables_info or []) if (s.get("phones") or [])])
            mismatch_delta = abs(int(expected_cv_count) - int(observed_cv_count))
            mismatch_soft_threshold = max(2, int(round(float(expected_cv_count) * 0.30))) if expected_cv_count > 0 else 2
            mismatch_hard_threshold = max(3, int(round(float(expected_cv_count) * 0.45))) if expected_cv_count > 0 else 3
            kr_syllable_count_guard_active = bool(
                kr_token_invariant_enabled
                and expected_cv_count > 0
                and observed_cv_count > 0
                and mismatch_delta >= mismatch_soft_threshold
            )
            kr_syllable_count_guard_hard = bool(
                kr_syllable_count_guard_active
                and (
                    kr_token_invariant_hard
                    or mismatch_delta >= mismatch_hard_threshold
                )
            )
            if kr_syllable_count_guard_active and kr_mapping_debug_reason_logging:
                mode_tag = "hard" if kr_syllable_count_guard_hard else "soft"
                log(
                    f"🧭 {fname}: 음절 수 불변 가드({mode_tag}) "
                    f"(expected={expected_cv_count}, observed={observed_cv_count}, delta={mismatch_delta})"
                )
            kr_confirmed_cv_indices = []
            kr_vc_pair_seq_idx = 0

            def _append_kr_confirmed_cv_idx(idx):
                if idx is None:
                    return
                try:
                    mapped = int(idx)
                except Exception:
                    return
                if not (0 <= mapped < len(syllables_info)):
                    return
                if kr_confirmed_cv_indices and mapped <= kr_confirmed_cv_indices[-1]:
                    return
                kr_confirmed_cv_indices.append(mapped)

            for line_num, line in enumerate(lines):
                alias = resolve_mapping_line_alias(
                    line=line,
                    fname=fname,
                    real_wav_name=real_wav_name,
                    final_lines=final_lines,
                    alias_suffix=alias_suffix,
                    record_unset_fn=_record_unset,
                    apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line,
                )
                if not alias:
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
                if kr_syllable_count_guard_active:
                    row_jump_default = 0
                    row_jump_high_conf = 0 if kr_syllable_count_guard_hard else min(row_jump_high_conf, 1)
                if kr_token_invariant_soft and kr_order_locked_format:
                    if alias_type in {"cv", "cv_head", "vcv"}:
                        row_jump_default = 0
                        if file_mapping_low_conf or textgrid_trust_tier != "high":
                            row_jump_high_conf = 0
                        else:
                            row_jump_high_conf = min(row_jump_high_conf, 1)
                    elif alias_type in {"vc", "vv"}:
                        row_jump_default = 0
                        row_jump_high_conf = min(row_jump_high_conf, 1)

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
                    token_pair_missing = False
                    if (
                        kr_token_invariant_enabled
                        and len(syllables_info) >= 2
                        and len(kr_confirmed_cv_indices) >= 2
                    ):
                        pair_slot = max(0, min(kr_vc_pair_seq_idx, len(kr_confirmed_cv_indices) - 2))
                        local_prev_idx = int(kr_confirmed_cv_indices[pair_slot])
                        local_next_idx = int(kr_confirmed_cv_indices[pair_slot + 1])
                        if kr_vc_pair_seq_idx < (len(kr_confirmed_cv_indices) - 2):
                            kr_vc_pair_seq_idx += 1
                        bridge_pair = {
                            "prev_idx": local_prev_idx,
                            "next_idx": local_next_idx,
                            "prev_anchor": (
                                realized_cv_anchor_by_idx.get(local_prev_idx)
                                or cv_anchor_by_idx.get(local_prev_idx)
                            ),
                            "next_anchor": (
                                realized_cv_anchor_by_idx.get(local_next_idx)
                                or cv_anchor_by_idx.get(local_next_idx)
                            ),
                            "source": "token_invariant",
                        }
                        if alias_type == "vc" and kr_mapping_debug_reason_logging:
                            log(
                                f"🧪 {fname}: KR VC bridge A/B token invariant pair "
                                f"({local_prev_idx + 1}->{local_next_idx + 1}, "
                                f"confirmed={len(kr_confirmed_cv_indices)}, {alias})"
                            )
                    elif kr_token_invariant_enabled and (
                        kr_token_invariant_hard
                        or kr_syllable_count_guard_active
                        or (
                            kr_token_invariant_soft
                            and kr_order_locked_format
                            and (file_mapping_low_conf or textgrid_trust_tier == "low")
                        )
                    ):
                        token_pair_missing = True

                    if token_pair_missing:
                        _append_strict_skip_alias(alias)
                        _record_unset(
                            "token_invariant_pair_missing",
                            fname,
                            line,
                            meta={
                                "mode": str(kr_token_invariant_mode),
                                "alias_type": str(alias_type),
                                "diag_hint": "strict token-invariant pair unavailable",
                                "syllable_count_guard": bool(kr_syllable_count_guard_active),
                                "syllable_count_guard_hard": bool(kr_syllable_count_guard_hard),
                                "expected_cv_count": int(expected_cv_count),
                                "observed_cv_count": int(observed_cv_count),
                            },
                        )
                        final_lines.append(apply_suffix_to_oto_line(line, alias_suffix))
                        continue

                    if not bridge_pair:
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
                    real_wav_name=output_wav_name,
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
                    real_wav_name=output_wav_name,
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
                        syllable_blank_confidences=syllable_blank_confidences,
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
                        real_wav_name=output_wav_name,
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
                        anchor_lock_lite=anchor_lock_lite,
                        alignment_weight=alignment_weight,
                        textgrid_trust_tier=textgrid_trust_tier,
                    )
                    _append_kr_confirmed_cv_idx(current_w_idx)
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
                        real_wav_name=output_wav_name,
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
                        anchor_lock_lite=anchor_lock_lite,
                        alignment_weight=alignment_weight,
                        textgrid_trust_tier=textgrid_trust_tier,
                    )
                    _append_kr_confirmed_cv_idx(current_w_idx)
                    continue


                if not is_vc:
                    selected_w_idx = current_w_idx
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
                        syllable_blank_confidences=syllable_blank_confidences,
                        syllables_info=syllables_info,
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
                    row_blank_floor_safe = (
                        float(row_blank_floor)
                        if row_blank_floor is not None
                        else 0.62
                    )
                    row_abstain = decide_cv_row_abstain(
                        alias_type=alias_type,
                        alias_text=alias,
                        format_type=file_format,
                        candidate_idx=selected_w_idx,
                        candidate_count=len(syllables_info),
                        candidate_active=(
                            _is_kr_cv_syllable_active(syllables_info[selected_w_idx], require_vowel=True)
                            if selected_w_idx is not None and 0 <= selected_w_idx < len(syllables_info)
                            else False
                        ),
                        confidence_margin=mapping_margin,
                        min_confidence_margin=row_margin_floor,
                        row_confidence=row_mapping_confidence,
                        min_row_confidence=row_conf_floor,
                        blank_confidence=_blank_conf_at(syllable_blank_confidences, selected_w_idx),
                        max_blank_confidence=row_blank_floor_safe,
                        # CVC는 파일/음절 특성상 margin 변동이 커 CV 계열이 과도 스킵될 수 있어
                        # row-level abstain 게이트를 CV/CVVC에만 적용한다.
                        active_only_formats=KR_ROW_GUARD_ACTIVE_FORMATS,
                        margin_formats=KR_ROW_GUARD_MARGIN_FORMATS,
                        blank_formats=KR_ROW_GUARD_BLANK_FORMATS,
                        min_confidence_margin_by_alias_type={"cv_head": row_margin_floor + 1.5, "vcv": row_margin_floor + 1.0},
                        min_row_confidence_by_alias_type={"cv_head": row_conf_floor + 0.03, "vcv": row_conf_floor + 0.02},
                        max_blank_confidence_by_alias_type={
                            "cv_head": max(0.0, row_blank_floor_safe - 0.03),
                            "vcv": max(0.0, row_blank_floor_safe - 0.02),
                        },
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
                        if use_template:
                            # 템플릿 모드에서는 매핑이 불확실한 행이라도 기존 alias를 보존한다.
                            final_lines.append(
                                apply_suffix_to_oto_line(line, alias_suffix)
                            )
                        continue
                    current_w_idx = max(current_w_idx, selected_w_idx)
                    _append_kr_confirmed_cv_idx(selected_w_idx)
                    selected_w_idx, curr_phones, c_start, c_end, n_start, n_end = _prepare_cv_bounds_from_syllable(
                        syllables_info, selected_w_idx
                    )
                else:
                    selected_w_idx = current_w_idx
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
                    next_anchor_idx_for_vc = (
                        int(bridge_next_idx)
                        if bridge_next_idx is not None
                        else int(current_w_idx + 1)
                    )
                    vc_prev_anchor = (
                        bridge_pair.get("prev_anchor")
                        or realized_cv_anchor_by_idx.get(current_w_idx)
                        or cv_anchor_by_idx.get(current_w_idx)
                    )
                    vc_next_anchor = (
                        bridge_pair.get("next_anchor")
                        or realized_cv_anchor_by_idx.get(next_anchor_idx_for_vc)
                        or cv_anchor_by_idx.get(next_anchor_idx_for_vc)
                    )
                    kr_vc_bridge_tuning = derive_cv_bridge_tuning(
                        realized_anchor_by_idx=realized_cv_anchor_by_idx,
                        estimated_anchor_by_idx=cv_anchor_by_idx,
                        trust_tier=textgrid_trust_tier,
                        mapping_tier=mapping_tier,
                        min_samples=6,
                    )
                    base_vc_params = None
                    try:
                        base_pack = _compute_kr_vc_timing(
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
                            prev_cv_anchor=vc_prev_anchor,
                            next_cv_anchor=vc_next_anchor,
                            bridge_tuning=None,
                        )
                        base_vc_params = tuple(base_pack[:5]) if base_pack else None
                    except Exception:
                        base_vc_params = None
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
                        prev_cv_anchor=vc_prev_anchor,
                        next_cv_anchor=vc_next_anchor,
                        bridge_tuning=kr_vc_bridge_tuning,
                    )
                    _accumulate_vc_bridge_ab_stats(
                        kr_vc_ab_stats,
                        base_vc_params,
                        (offset, consonant, cutoff, pre, ovl),
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

                row_mel_voiced_onset_ms = None
                if mel_ctx_for_file and alias_type == "cv":
                    pre_abs = float(offset) + float(pre)
                    mel_weight = _resolve_mel_onset_weight(alignment_weight, textgrid_trust_tier)
                    if mel_weight > 0.0 and not _env_bool("UTOA_DISABLE_MEL_ONSET_SHIFT", False):
                        mel_onset = _estimate_mel_voiced_onset(mel_ctx_for_file, pre_abs)
                        if mel_onset is not None and abs(float(mel_onset) - pre_abs) <= 120.0:
                            before = (offset, pre, consonant, cutoff)
                            (
                                offset,
                                consonant,
                                cutoff,
                                pre,
                                ovl,
                                _mel_shift,
                            ) = _apply_mel_voiced_onset_pre_shift(
                                offset,
                                consonant,
                                cutoff,
                                pre,
                                ovl,
                                mel_onset,
                                weight=mel_weight,
                            )
                            row_mel_voiced_onset_ms = float(mel_onset)
                            after = (offset, pre, consonant, cutoff)
                            _debug_offset_trace(
                                log,
                                "mel_onset_shift",
                                fname,
                                alias,
                                before,
                                after,
                                extra=f"mel={row_mel_voiced_onset_ms:.1f}",
                            )

                before_post = (offset, pre, consonant, cutoff)
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
                    # debug trace inside caller (before/after)
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
                _debug_offset_trace(
                    log,
                    "post_timing",
                    fname,
                    alias,
                    before_post,
                    (offset, pre, consonant, cutoff),
                )
                if (
                    alias_type == "cv"
                    and str(file_format or "").strip().lower() in {"cvvc", "cvc"}
                    and _env_bool("UTOA_KR_CV_MIN_VOWEL_GUARD", True)
                ):
                    alias_onset = _extract_alias_onset(alias)
                    ipa_onset = curr_phones[0].mark if curr_phones else ""
                    offset, consonant, cutoff, pre, _cv_vowel_extended = _ensure_cv_min_vowel_coverage(
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        n_start,
                        n_end,
                        alias_onset=alias_onset,
                        ipa_onset=ipa_onset,
                    )
                if alias_type == "cv" and not _env_bool("UTOA_DISABLE_CV_FOCUS_GUARD", False):
                    alias_onset = _extract_alias_onset(alias)
                    ipa_onset = curr_phones[0].mark if curr_phones else ""
                    before_focus = (offset, pre, consonant, cutoff)
                    (
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        cv_offset_pulled,
                        cv_cutoff_trimmed,
                    ) = _guard_cv_focus_window(
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        c_start,
                        c_end,
                        n_start,
                        n_end,
                        alias_onset=alias_onset,
                        ipa_onset=ipa_onset,
                    )
                    _debug_offset_trace(
                        log,
                        "cv_focus_guard",
                        fname,
                        alias,
                        before_focus,
                        (offset, pre, consonant, cutoff),
                    )
                    if (cv_offset_pulled >= 0.8) or (cv_cutoff_trimmed >= 0.8):
                        log(
                            f"🛡️ {fname}: CV 핵심구간 보정 "
                            f"(offset -{cv_offset_pulled:.1f}ms, cutoff -{cv_cutoff_trimmed:.1f}ms) [{alias}]"
                        )
                _run_kr_general_row_v2(
                    final_lines=final_lines,
                    real_wav_name=output_wav_name,
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
                    anchor_lock_lite=anchor_lock_lite,
                    voiced_onset_ms=row_mel_voiced_onset_ms,
                    mel_ctx_for_file=mel_ctx_for_file,
                    textgrid_trust_tier=textgrid_trust_tier,
                    alignment_weight=alignment_weight,
                    bridge_tuning=kr_vc_bridge_tuning,
                )

            if strict_skip_aliases:
                for i in range(0, len(strict_skip_aliases), 24):
                    chunk = strict_skip_aliases[i:i + 24]
                    log(" ".join(chunk))

            kr_vc_ab_summary = _summarize_vc_bridge_ab_stats(kr_vc_ab_stats)
            if kr_vc_ab_summary:
                log(
                    f"🧪 {fname}: KR VC bridge A/B "
                    f"(rows={int(kr_vc_ab_summary.get('rows', 0))}, "
                    f"pre_abs_mean={float(kr_vc_ab_summary.get('pre_abs_shift_mean_ms', 0.0)):+.1f}ms, "
                    f"pre_abs_abs_mean={float(kr_vc_ab_summary.get('pre_abs_shift_abs_mean_ms', 0.0)):.1f}ms)"
                )
            update_mapping_vc_bridge_runtime_report(
                runtime_report,
                vc_bridge_tuning=dict(kr_vc_bridge_tuning or {}),
                vc_bridge_ab_summary=kr_vc_ab_summary,
            )

            processed += 1

        except Exception as e:
            loc = ""
            try:
                tb_last = traceback.extract_tb(e.__traceback__)[-1] if e.__traceback__ else None
                if tb_last is not None:
                    loc = f" [{os.path.basename(tb_last.filename)}:{int(tb_last.lineno)}]"
            except Exception:
                loc = ""
            err_msg = f"처리 실패 ({fname}): {e}{loc}"
            logger.error(err_msg)
            errors.append(err_msg)
            _record_unset_lines("file_exception", fname, lines)
            final_lines.extend([
                apply_suffix_to_oto_line(l, alias_suffix)
                for l in lines
            ])
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
            output_name = str(tg_info.get("output_name", tg_info.get("real_name", "")) or "")


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
                                output_name or tg_info['real_name'],
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

    persist_korean_generation_output(
        out_path=out_path,
        final_lines=final_lines,
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
        wav_name_map=wav_name_map,
        apply_output_wav_name_map_fn=apply_output_wav_name_map,
        oto_encoding=template_encoding,
        errors=errors,
    )

    finalize_generator_finish(finish_context)

    _log_unset_summary()
    finalize_runtime_report(
        runtime_report,
        processed=processed,
        total=total,
        errors=errors,
    )
    return processed, total, errors
