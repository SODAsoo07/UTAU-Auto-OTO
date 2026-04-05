"""
TextGrid・ｼ ・ｰ・們愍・・﨑懋ｵｭ・ｴ OTO.ini・ｼ ・晧┳﨑ｩ・壱共.
- phones/words tier・ｼ ・ｴ・ｩ﨑ｴ OTO 甯誤攵・ｸ奓ｰ・ｼ ・・げ﨑ｩ・壱共.
- CV/VC/VCV/VV/・ｨ・ｨ・・・ｨ・誤ｦｬ(br) ・・ｴ・､・ｼ ・俯ｦｬ﨑ｩ・壱共.
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
from typing import Optional, Dict, Tuple

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
    write_jsonl_records,
    write_oto_lines,
)
from core.alignment_ingest import build_kr_alignment_ingest
from core.kr_candidate_selection_v2 import select_kr_syllable_source
from core.oto_diagnostics import SkippedEntryCollector
from core.oto_diagnostics_adapter_v2 import GeneratorDiagnosticsAdapter
from core.file_prepare import load_named_tiers, prepare_file_context
from core.interval_lookup import build_interval_lookup, intervals_within_bounds
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
from core.generation.file_stages import (
    handle_kr_file_context_status,
    handle_kr_loop_prep_status,
    load_named_tiers_for_generation,
    prepare_file_context_with_sinsy,
)
from core.generation.plan_runtime import (
    build_common_plan_context,
    extract_kr_alignment_ingest_state,
    log_sinsy_plan_guard,
    recompute_common_plan_runtime_state,
    update_single_vowel_span_by_first_phone,
)
from core.generation.mapping_runtime import (
    compute_runtime_low_conf_state,
    update_kr_mapping_runtime_report,
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


def _env_int(name, default):
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _env_bool(name, default=False):
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)

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
# ・瀧ｯｸ・ｬ ・俯ｦｬ ・寀ｸ・ｬ寀ｰ
# ==============================================================================

def _normalize_alias_suffix(suffix):
    s = (suffix or "").strip()
    if not s:
        return ""
    return s[1:] if s.startswith("_") else s


def apply_alias_suffix(alias, suffix):
    """・川攵・ｬ・ｴ・､ ・晧乱 `_<suffix>` ・瀧ｯｸ・ｬ・ｼ ・呷桿・壱共."""
    suf = _normalize_alias_suffix(suffix)
    if not suf:
        return alias
    a = (alias or "").strip()
    if not a:
        return alias
    return f"{a}_{suf}"


def apply_suffix_to_oto_line(line, suffix):
    """`wav=alias,params...` ・ｼ・ｸ・川・ alias・尖ｧ・・瀧ｯｸ・ｬ・ｼ ・・圸﨑ｩ・壱共."""
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
# ・ｰ・ｸ 孖罹享 甯誤攵・ｸ奓ｰ
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


DEFAULT_PARAMS_PRESET_BY_CONTEXT = {
    "korean": {
        "default": {
            "VC_CONSONANT_RATIO": 0.47,
            "VC_VOWEL_START": 0.36,
            "VC_PRE_OFFSET": 20.0,
            "VC_OVL_RATIO": 0.34,
            "CV_PRE_RATIO": 0.92,
            "CV_OVL_RATIO": 0.38,
            "DIPHTHONG_CV_PRE_RATIO": 0.33,
            "DIPHTHONG_CV_CONSONANT_RATIO": 0.58,
            "DIPHTHONG_VC_VOWEL_START": 0.34,
            "DIPHTHONG_VC_CONSONANT": 0.48,
        },
        "cv": {
            "CV_PRE_RATIO": 0.88,
            "CV_OVL_RATIO": 0.36,
            "VC_PRE_OFFSET": 16.0,
        },
        "cvc": {
            "VC_CONSONANT_RATIO": 0.42,
            "VC_VOWEL_START": 0.34,
            "VC_PRE_OFFSET": 18.0,
            "VC_OVL_RATIO": 0.36,
            "CV_PRE_RATIO": 0.90,
        },
        "cvvc": {
            "VC_CONSONANT_RATIO": 0.44,
            "VC_VOWEL_START": 0.35,
            "VC_PRE_OFFSET": 19.0,
            "VC_OVL_RATIO": 0.35,
            "CV_PRE_RATIO": 0.90,
        },
        "vcv": {
            "VC_CONSONANT_RATIO": 0.48,
            "VC_VOWEL_START": 0.40,
            "VC_PRE_OFFSET": 22.0,
            "VC_OVL_RATIO": 0.33,
            "CV_PRE_RATIO": 0.95,
            "CV_OVL_RATIO": 0.36,
        },
    },
    "japanese": {
        "default": {
            "VC_CONSONANT_RATIO": 0.50,
            "VC_VOWEL_START": 0.32,
            "VC_PRE_OFFSET": 20.0,
            "VC_OVL_RATIO": 0.32,
            "CV_PRE_RATIO": 0.86,
            "CV_OVL_RATIO": 0.34,
            "DIPHTHONG_CV_PRE_RATIO": 0.34,
            "DIPHTHONG_CV_CONSONANT_RATIO": 0.58,
            "DIPHTHONG_VC_VOWEL_START": 0.31,
            "DIPHTHONG_VC_CONSONANT": 0.50,
        },
        "cv": {
            "CV_PRE_RATIO": 0.82,
            "CV_OVL_RATIO": 0.33,
        },
        "cvvc": {
            "VC_CONSONANT_RATIO": 0.52,
            "VC_PRE_OFFSET": 18.0,
            "CV_PRE_RATIO": 0.80,
        },
        "vcv": {
            "VC_CONSONANT_RATIO": 0.46,
            "VC_VOWEL_START": 0.36,
            "VC_PRE_OFFSET": 22.0,
            "CV_PRE_RATIO": 0.98,
            "CV_OVL_RATIO": 0.40,
        },
    },
}


def get_default_params_for_context(language="korean", format_type=""):
    lang = str(language or "korean").strip().lower()
    if lang not in DEFAULT_PARAMS_PRESET_BY_CONTEXT:
        lang = "korean"
    fmt = normalize_format_type(lang, format_type) or "general"
    lang_presets = DEFAULT_PARAMS_PRESET_BY_CONTEXT.get(lang, {})
    resolved = dict(DEFAULT_PARAMS)
    resolved.update(lang_presets.get("default", {}))
    if fmt in {"general", "default"}:
        fallback_fmt = "cvvc" if lang == "korean" else "cvvc"
        resolved.update(lang_presets.get(fallback_fmt, {}))
    else:
        resolved.update(lang_presets.get(fmt, {}))
    return resolved

# 﨑懋ｵｭ・ｴ ・､﨑・・・ｰ・・・・ｳ・ｹ・・ｰ・ｸ・・尞ｬ・ｷ・・
# - CVVC/VCV: ・､・､﨑・・ｵ・罹･ｼ ・・紛 ・ｴ・們・愍・・・・箕
# - CV/CVC: CVVC・ｴ・､ ・・剩﨑俯据 ・ｰ・ｴ・・・・・・醐少 ・・箕
# - VC_ONLY/VV_ONLY: CV ・簿ｬ ・戦売 ・・箕・ｴ ・卓符 ・滝ｰ・ｰ・・・
KR_MAPPING_CONF_THRESHOLD_BY_FORMAT = {
    "cv": 0.62,
    "cvvc": 0.67,
    "vcv": 0.66,
    "cvc": 0.63,
    "cv_simple": 0.61,
    "mono": 0.61,
    "vc_only": 0.60,
    "vv_only": 0.60,
    "default": 0.64,
}


def _resolve_kr_threshold_env_override(file_format):
    fmt = str(file_format or "").strip().lower()
    if not fmt:
        return None
    env_key = "UTOA_KR_MAPPING_CONF_THRESHOLD_" + re.sub(r"[^a-z0-9]+", "_", fmt).strip("_").upper()
    raw = str(os.environ.get(env_key, "") or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _resolve_kr_mapping_conf_threshold(file_format, override_threshold=None, phone_quality_score=None):
    """尞ｬ・ｷ・・・ｰ・ｸ ・､﨑・・・ｰ・・・・ｳ・ｰ廷揆 ・倆劍﨑ｩ・壱共.

    phone_quality_score・ ・懋ｳｵ・俯ｩｴ phone tier 峵溢ｧ溢乱 ・ｰ・ｼ ・呷・愍・・・ｰ・倣鮒・壱共:
    - 峵溢ｧ・< 0.4 -> ・・ｳ・ｰ廷揆 0.05 ・ｮ・､ (・ｼ・・復 ・・・ｰ 甯川・・・剩)
    - 峵溢ｧ・> 0.8 -> ・・ｳ・ｰ廷揆 0.03 ・ｬ・ｼ (・峵溢ｧ・・ｰ・ｴ奓ｰ・川・ ・・・・ｲｩ)
    """
    if override_threshold is not None:
        try:
            return float(override_threshold)
        except Exception:
            pass
    fmt = str(file_format or "").strip().lower()
    env_override = _resolve_kr_threshold_env_override(fmt)
    if env_override is not None:
        return float(env_override)
    base = KR_MAPPING_CONF_THRESHOLD_BY_FORMAT.get(
        fmt,
        KR_MAPPING_CONF_THRESHOLD_BY_FORMAT["default"],
    )
    threshold = float(base)

    # phone tier 峵溢ｧ溢乱 ・ｰ・ｸ ・呷・・ｰ・・
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
    """甯護攵・・乱 ・ｴ・瀧ｪｨ・・岺懍享・ｴ 尞ｬ﨑ｨ・俯株・ 嶹菩攤﨑ｩ・壱共."""
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
    """・川攵・ｬ・ｴ・､ ・ｸ・川龍・・・ｴ・瀧ｪｨ・・甯ｨ奓ｴ・ｴ ・壱株・ 嶹菩攤﨑ｩ・壱共."""
    clean = alias.replace(' ', '').lower()
    diphthongs = ['ya','yeo','yo','yu','ye','wa','wo','wi','we','weo','eui','ui']
    for diph in diphthongs:
        if diph in clean:
            return True
    return False


def adaptive_overlap(pre, consonant_hint="", mode="cv"):
    """
    ・川搆 ・ｱ・・・川攵・ｬ・ｴ・､ 夋・・乱 ・ｰ・ｼ overlap・・・呷・愍・・・ｰ・倣鮒・壱共.
    """
    p = max(float(pre), 0.0)
    if p <= 0:
        return 0.0

    hint = normalize_ipa_mark(consonant_hint or "")
    hard = {'g', 'd', 'b', 'c', 'j'}
    tense = {'kk', 'tt', 'pp', 'gg', 'dd', 'bb', 'ss', 'jj'}  # ・ｽ・・ VOT ・ｧ・・
    aspirate = {'k', 't', 'p', 'ch'}  # ・ｩ・・ ・ｰ・・・ｸ・・
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
        ratio -= 0.20  # ・ｽ・・ VOT・ ・ｧ・・overlap・・・・・・桷
    elif hint in hard:
        ratio -= 0.14
    elif hint in aspirate:
        ratio += 0.02  # ・ｩ・・ ・ｰ・晧擽 ・ｸ・ｴ overlap・・・ｽ・・・楠椈
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
    """CV/CV_HEAD cutoff・ｴ ・､・・・護・onset・・・ｨ・被葺・ ・危巡・・・・復・・・ｴ・､."""
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


def _guard_cv_cutoff_to_next_onset_strict(offset, consonant, cutoff, pre, syll_idx, syllables_info):
    """order-locked CV・川・ ・､・・・護・onset ・ｨ・肥揆 ・・・倣葺・・・雅株 ・・・"""
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
    """﨑懋ｵｭ・ｴ VC cutoff ・・・嶸ｸ嶹・・倆詐)."""
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
    CV_HEAD(- CV) offset・ｴ 嶸・椪 ・護・onset・ｴ・､ ・ｼ・・葺・・・樌・・ ・危巡・・・懦復﨑ｩ・壱共.
    ・ｵ・ｱ ・・溜 ・ｼ尞ｬ﨑ｨ・・・・擽・ｰ ・・復 ・・懍桿・壱共.
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

    # onset 孖ｹ・ｱ・・嵭溢圸 ・ｬ・・ms): 甯護龍/・們ｰｰ・ ・ｰ・・・縄ｲ・ ・ｵ・・搆・ ・・夋・ｴ孖ｸ﨑俾ｲ・
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
    # ・､嵓・・・・・ｹ・ｸ ・・・・劇 ・ｸ・ｴ(pre/cons/cutoff)・ ・賀ｲｩ德・・・牟・､・ ・危巡・・
    # ・ｰ・ｴ ・・劇 甯誤攵・ｸ奓ｰ・ｼ ・ｰ・ ・ｴ・ｴ﨑罹共.
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
    CV_HEAD(-CV)・川・ ・ｷ・､嵓・ｰ ・壱ｬｴ ・ｴ・ｴ・・・ｫ嶸 ・ｨ・護擽 ・ｰ・・尞ｬ﨑ｨ・們ｧ ・危株 ・ｽ・ｰ・ｼ ・ｩ・﨑ｩ・壱共.
    """
    v_start = float(vowel_start_ms)
    v_end = float(vowel_end_ms)
    v_len = max(0.0, v_end - v_start)
    if v_len < 40.0:
        return offset, consonant, cutoff, pre, 0.0

    cut_abs = abs(float(cutoff))
    # ・ｨ・・・ｬ・・揆 ・懍・ ・ｼ・(・・惠+﨑倆復) 尞ｬ﨑ｨ﨑俯巡・・・ｷ・､嵓・﨑倆復・・・､・・
    keep_v_ms = min(max(v_len * 0.30, 70.0), 190.0)
    vowel_start_rel = max(v_start - float(offset), float(pre) + 8.0)
    # pre ・ｴ弡・・壱ｬｴ ・ｨ・ｬ ・ｫ德壱株 ・・ｴ・､(・川搆・・・ｨ・・・ｸ・ｴ)・ｼ ・ｩ・.
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


def _is_sonorant_like_onset(onset, ipa_hint=""):
    o = str(onset or "").strip().lower()
    h = str(ipa_hint or "").strip().lower()
    if o.startswith(("m", "n", "ng", "l", "r")):
        return True
    if h.startswith(("m", "n", "ng", "l", "r")):
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
    CV・川・ ・ｷ・､嵓・ｰ ・壱ｬｴ ・ｧ・・・ｨ・・・懍梠・ｴ ・ｰ・・尞ｬ﨑ｨ・們ｧ ・危株 ・ｽ・ｰ・ｼ ・ｴ・倣鮒・壱共.
    孖ｹ德・・・ｱ ・川搆(m/n/l/r)・川・ ・ｴ・們・愍・・・・圸﨑ｩ・壱共.
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
    CV・・﨑ｵ・ｬ ・ｬ・・揆 ・ｴ・倣鮒・壱共.
    - offset・ｴ onset ・､・・・・､ ・川搆・ｴ ・俯ｦｬ・・・・揆 ・ｩ・
    - cutoff・ ・ｨ・・tail(彧尖､・・・・ｬ・・・・・ｼ・・葺・・尞ｬ﨑ｨ﨑們ｧ ・危巡・・・懦復
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

    # 1) ・川搆 onset ・ｴ弡・｡・offset・ｴ ・・ｬ・ｴ offset ・川ｲｴ・ｼ ・橄胸・ｨ
    # pre/cons/cut ・壱劇・簿巡 ・呷擽 ・樌ｪｽ・ｼ・・・ｴ・呷亨墲ｨ・､.
    onset_late_slack = max(0.0, _env_float("UTOA_KR_CV_ONSET_LATE_SLACK_MS", 2.0))
    max_allowed_offset = max(0.0, c_start + onset_late_slack)
    if float(offset) > max_allowed_offset:
        shift = float(offset) - max_allowed_offset
        offset = max_allowed_offset
        offset_pulled_ms = shift
    elif float(offset) < min_allowed_offset:
        # ・・・ｨ・・・尖ｯｸ・ offset ・ｬ・・愍・・・・・据・ ・危巡・・offset 﨑倆復・・・ｴ・･.
        advance = min_allowed_offset - float(offset)
        offset = min_allowed_offset
        pre = max(6.0, float(pre) - advance)
        consonant = max(float(pre) + 8.0, float(consonant) - advance)
        cutoff = -max(float(consonant) + 12.0, abs(float(cutoff)) - advance)

    # 1.5) pre(anchor)・ ・ｨ・・・瀧ｰ們愍・・・ｦ・・・､・ｴ・・・・ｽ・ｰ・ｼ ・倣葺・・・懦復.
    # ・ｰ・ offset・・・・・橄胸・ｰ・・onset 﨑倆復・・・們ｧ ・伎ｲ・﨑俾ｳ,
    # 﨑倆復・・・ｸ・ｬ・ｴ pre・ｼ ・・・・・攤・､.
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

    # 2) ・ｨ・・・溢・・ｬ・・・ｴ弡・tail・ cutoff ・・復・ｼ・・・懦復.
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
    """VCV ・・げ・・﨑・囈﨑・・護・・ｸ・ｱ・､ ・ｱ・・ｼ ・ｰ・ｸ 夋・ｴ・作揆 ・ｰ・懦鮒・壱共."""
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
    ・晧┳ 甯誤攵・ｸ奓ｰ・ｼ WAV ・ｸ・ｴ ・ｽ・・・溢愍・・・菩・・ｴ・倣鮒・壱共.
    - offset >= 0
    - offset + |cutoff| <= wav_duration
    - ovl <= pre <= consonant < |cutoff| (・・･﨑・・肥怱・川・ ・・)
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
    alias_key = str(alias_type or "").strip().lower()
    min_tail = 2.0
    if alias_key in {"cv", "cv_head"}:
        min_tail = _env_float(
            "UTOA_KR_CV_MIN_CUTOFF_SPAN_MS",
            _env_float("UTOA_CV_MIN_CUTOFF_SPAN_MS", 96.0),
        )
        if alias_key == "cv_head":
            min_tail = _env_float(
                "UTOA_KR_CV_HEAD_MIN_CUTOFF_SPAN_MS",
                max(72.0, float(min_tail) - 12.0),
            )
    min_tail = max(2.0, float(min_tail))
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
    cut_floor = consonant + 0.3
    if alias_key in {"cv", "cv_head"}:
        cut_floor = max(cut_floor, min(min_tail, float(available)))
    cut_abs = max(cut_floor, min(cut_abs, available))
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
        # ・ｹ・ｨ ・・ｴ・､ ・懍｢・・溢・ｧ・
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
    """・川攵・ｬ・ｴ・､(・・OpenUtau ・嶸・・ｼ OTO ・ｼ・ｸ ・ｸ・川龍・・・嶹倆鮒・壱共."""
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
    """・川攵・ｬ・ｴ・､(・・OpenUtau ・嶸・・ｼ OTO ・ｼ・ｸ・ｼ・・・・・鮒・壱共."""
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
    """・菩・occurrence ・ｸ・ｱ・､・ ・巐ｨ﨑們ｧ ・喜揆 ・・・ｬ・ｩ ・・･﨑・・ｸ・ｱ・､・・・ｬ・､﨑啄鮒・壱共."""
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


def _build_kr_planned_cv_indices(expected_tokens, syllables_info, *, format_type=""):
    plan = _build_kr_cv_anchor_plan_v2(expected_tokens, syllables_info, format_type=format_type)
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
    """弡・ｲ俯ｦｬ ・・・soft mel/base shape/stabilize/cutoff)・ｼ ・ｼ・ ・・圸﨑ｩ・壱共."""
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
    """弡・ｲ俯ｦｬ ・・・・懋ｷｸ(嶸ｸ嶹・・倆詐)."""
    _log_post_timing_events_core(log_fn, fname, alias, soft_off_shift, soft_cut_shift, cutoff_reduced)


def _is_kr_nucleus_phone_mark(mark):
    m = normalize_ipa_mark(mark)
    if not m:
        return False
    if m in IPA_VOWELS:
        return True
    if m in KR_VOWELS:
        return True
    mapped = _map_kr_phone_vowel_to_roman(m)
    return bool(mapped and mapped in KR_VOWELS)


def _map_kr_phone_vowel_to_roman(mark):
    m = str(normalize_ipa_mark(mark) or '').strip().lower()
    if not m:
        return ''

    direct = {
        'a': 'a',
        'e': 'e',
        'i': 'i',
        'o': 'o',
        'u': 'u',
        'eo': 'eo',
        'eu': 'eu',
        'ae': 'ae',
        'oe': 'oe',
        'ui': 'ui',
        'wa': 'wa',
        'we': 'we',
        'weo': 'weo',
        'wi': 'wi',
        'wo': 'wo',
        'wae': 'wae',
        'ya': 'ya',
        'ye': 'ye',
        'yeo': 'yeo',
        'yo': 'yo',
        'yu': 'yu',
    }
    if m in direct:
        return direct[m]

    ipa_map = {
        '\u0259': 'eo',  # schwa
        '\u028c': 'eo',  # open-mid back unrounded
        '\u026f': 'eu',  # close back unrounded
        '\u025b': 'ae',  # open-mid front unrounded
        '\u00f8': 'oe',  # close-mid front rounded
        '\u0153': 'oe',  # open-mid front rounded
        '\u026a': 'i',
        '\u028a': 'u',
        '\u0252': 'o',
        '\u0250': 'a',
    }
    if m in ipa_map:
        return ipa_map[m]

    if m.startswith('j') and len(m) > 1:
        tail = _map_kr_phone_vowel_to_roman(m[1:])
        return {
            'a': 'ya',
            'e': 'ye',
            'eo': 'yeo',
            'o': 'yo',
            'u': 'yu',
            'ae': 'yae',
        }.get(tail, tail)

    if m.startswith('w') and len(m) > 1:
        tail = _map_kr_phone_vowel_to_roman(m[1:])
        return {
            'a': 'wa',
            'e': 'we',
            'eo': 'weo',
            'i': 'wi',
            'o': 'wo',
            'ae': 'wae',
        }.get(tail, tail)

    return m


def _estimate_kr_nucleus_token(ph_intervals, nuc_idx):
    """phone nuclei ・ｼ・ｩ・川・ 﨑懋ｵｭ・ｴ CV 﨑ｵ・ｬ 奝增ｰ・・・肥倣鮒・壱共."""
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
        elif gmark in {"w", "\u02b7", "\u0265"}:
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
    """target ・懍・・ nuclei token score・ｼ ・ｴ・ｩ﨑ｴ monotonic alignment・ｼ ・倆哩﨑ｩ・壱共."""
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
        # ・・ｼ・ｨ ・ｹ・護乱・罹株 nucleus・ target ・懍・・ｴ・､ ・・ｲ・・・罹据・・・ｽ・ｰ・ ・壱共.
        # ・ｴ・・・､﨑・・・ｲｴ・ｼ 尞ｬ・ｰ﨑俯ｩｴ CVVC 甯護攵・ｴ 奝ｵ・ｸ・・mapping_failed・・・ｨ・ｴ・・・・
        # phone ・ｬ・・揆 target ・懍・・・・樌ｶｰ contiguous chunk・・・菩・・・腹﨑罹共.
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
    phone_lookup = build_interval_lookup(ph_intervals)
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

        phones = intervals_within_bounds(phone_lookup, s_t, e_t)
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
    phones tier 峵溢ｧ溢揆 ・・げ﨑ｩ・壱共.
    - ・・ｹｨ・ｵ phone ・・
    - spn ・・惠
    - 﨑ｵ ・ｨ・・phone ・・
    - ・ｰ・ ・護・・・・phone ・・・・惠
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
    﨑懋ｵｭ・ｴ ・護・・､﨑・・・ｰ・・･ｼ 0~1・・・肥倣鮒・壱共.
    - phones/words 峵溢ｧ・
    - words vs alias ・川・ ・溢ｧ・
    - ・・圸・・・､﨑・・ｽ・・
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
    words ・ｬ・・乱 ・・啄葺・・phones・ ・・牟 ・溢揆 ・・・懍・ 﨑ｩ・ｱ phone・・・晧┳﨑ｩ・壱共.
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
        "eo": "\u0259",
        "eu": "ﾉｯ",
        "ae": "\u025b",
        "oe": "ﾃｸ",
        "wi": "wi",
        "wo": "wo",
        "wa": "wa",
        "we": "we",
        "weo": "w\u0259",
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
    """UTAU OTO 甯誤攵・ｸ奓ｰ・ｼ ・巐ｨ ・肥怱・・・ｴ・倣鮒・壱共.

    UTAU 﨑・・ ・懍・ ・懍平: ovl < pre <= consonant < |cutoff|
    alias_type・・・ｰ・ｼ 甯誤攵・ｸ奓ｰ ・・・懍・ ・・ｲｩ・・・ｨ・ｱ ・・圸﨑ｩ・壱共.

    Args:
        offset: 甯護攵 ・橄ｶ・・ｶ奓ｰ・・・懍梠 ・・ｹ・(ms, >= 0)
        consonant: ・・菩梵・誤ｶ (・､嵓・・ ・ｰ・ ・・劇, ・､孖ｸ・溢ｹ・・一ｰ ・ｬ・・
        cutoff: ・ｷ・､嵓・(・護・, ・､嵓・・ ・ｰ・ ・・劇, ・ｴ弡・・誤ｦｬ ・俯ｦｼ)
        pre: ・嵂雅ｰ懍搆 (・､嵓・・ ・ｰ・ ・・劇, ・川搆->・ｨ・・・・擽・・
        ovl: ・､・・棠 (・､嵓・・ ・ｰ・ ・・劇, ・・・ｸ孖ｸ・ ・罷誤畠)
        alias_type: ・川攵・ｬ・ｴ・､ 夋・・(cv, cv_head, vc, vv, vcv ・ｱ, ・夋晧・
    """
    a_type = str(alias_type or "").strip().lower()

    # --- alias_type・・・懍・ ・・ｲｩ 奛護擽・・---
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
    min_cutoff_span = 0.0
    if a_type in {"cv", "cv_head"}:
        cv_span_default = _env_float(
            "UTOA_KR_CV_MIN_CUTOFF_SPAN_MS",
            _env_float("UTOA_CV_MIN_CUTOFF_SPAN_MS", 96.0),
        )
        if a_type == "cv_head":
            cv_span_default = _env_float(
                "UTOA_KR_CV_HEAD_MIN_CUTOFF_SPAN_MS",
                max(72.0, float(cv_span_default) - 12.0),
            )
        min_cutoff_span = max(float(min_cut_gap) + 8.0, float(cv_span_default))

    # --- ・ｰ・ｸ 﨑倆復 ---
    if offset < 0:
        offset = 0
    if pre < 0:
        pre = 0
    if ovl < 0:
        ovl = 0
    if consonant < 0:
        consonant = 0

    # --- ovl < pre ・菩・---
    if ovl > pre:
        ovl = pre * 0.75

    # --- pre <= consonant ・菩・(alias_type・・・懍・ ・・ｲｩ) ---
    if consonant < pre + min_cons_gap:
        consonant = pre + min_cons_gap

    # --- consonant < |cutoff| ・菩・(alias_type・・・懍・ ・・ｲｩ) ---
    cutoff_abs = abs(cutoff)
    if cutoff_abs <= consonant + min_cut_gap:
        cutoff_abs = consonant + min_cut_gap
    if min_cutoff_span > 0.0 and cutoff_abs < min_cutoff_span:
        cutoff_abs = min_cutoff_span
    cutoff = -cutoff_abs

    # --- ・懍｢・・懍・ ・・・(・溢・ｧ・ ---
    if ovl >= pre:
        ovl = max(0.0, pre - 2.0)
    if consonant < pre:
        consonant = pre + min_cons_gap
    cutoff_abs = abs(cutoff)
    if cutoff_abs <= consonant:
        cutoff_abs = consonant + min_cut_gap
    if min_cutoff_span > 0.0 and cutoff_abs < min_cutoff_span:
        cutoff_abs = min_cutoff_span
    cutoff = -cutoff_abs

    return offset, consonant, cutoff, pre, ovl


def _nearest_phone_edge_ms(anchor_ms, ph_intervals):
    """anchor_ms・・・・･ ・・護垓 phone ・ｽ・・ms)・ｼ ・倆劍﨑ｩ・壱共."""
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
    anchor_ms・ phone ・ｬ・ｴ gap・・・溢揆 ・・(prev_end, next_start, gap_len_ms) ・倆劍.
    gap ・ｴ・・ ・・笈・ｴ (None, None, 0.0).
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
    ・嵂雅ｰ懍┳ ・ｰ・・・pre ・壱劇・・ｹ・・ｴ ・ｴ・・gap・・・ｸ・ｬ・ｴ
    ・・･ ・・護垓 ・巐ｨ phone ・ｽ・・｡・・､・・紛 ・・・ｵ・・・簿ｬ・・・・剩﨑ｩ・壱共.
    """
    if not ph_intervals:
        return validate_oto_params(offset, consonant, cutoff, pre, ovl)

    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
    pre_abs = float(offset) + float(pre)
    nearest_edge, nearest_dist = _nearest_phone_edge_ms(pre_abs, ph_intervals)
    prev_end, next_start, gap_len = _surrounding_phone_gap(pre_abs, ph_intervals)

    # ・ｩ・・梭 ・南捩 gap ・ｴ・・ｴ・ｰ・・phone・ｼ ・・ｬ ・ｨ・ｴ・・・ｽ・ｰ・・・ｴ・・
    if gap_len < 55.0 and nearest_dist <= 34.0:
        return offset, consonant, cutoff, pre, ovl

    target = nearest_edge
    if prev_end is not None and next_start is not None:
        if alias_type in {"vc", "vv"}:
            # VC/VV・・・､・・・護・・・ｵｬ・ｼ ・ｨ・･﨑俯据, ・ｽ・・ｳｴ・､ ・ｽ・・・樌乱・・・｡・罷共.
            target = next_start - 6.0
            target = max(target, prev_end + 4.0)
        elif alias_type in {"cv", "cv_head"}:
            # CV ・・龍・ ・ｴ・・・護・・晧愍・・・護牟・ｹ・ｰ・ ・伎ｳ,
            # 嶸・椪 ・川搆 onset ・・・愍・・・､・・紛 ・・・懍搆 ・・・揆 ・・攤・､.
            target = next_start - 4.0
            target = max(target, prev_end + 3.0)
        else:
            # VCV ・ｱ・ ・ｴ・・・護・・・・ｽ・・・ｼ・俯｡・・ｹ・ｨ・・・ｵ・ｱ ・簿ｬ・・・ｩ・﨑罹共.
            target = prev_end

    delta = target - pre_abs
    if abs(delta) < 2.0:
        return offset, consonant, cutoff, pre, ovl

    target_offset = max(float(offset) + delta, 0.0)
    force_snap_dist = _env_float("UTOA_KR_PRE_FORCE_SNAP_DIST", 140.0)
    if nearest_dist >= force_snap_dist:
        offset = target_offset
    elif abs(delta) > 30.0:
        # 增ｰ ・ｰ・ｬ snap・ ・ｼ・川菩揆 ・・懦腹 ・・・溢牟 ・罷誤畠・ｼ・・・・剩﨑罹共.
        offset = _blend(float(offset), target_offset, 0.45)
    else:
        # ・ｧ・ ・ｰ・ｬ ・ｴ・菩捩 ・ｰ・ｴ・俯涵 ・餓亨 snap﨑罹共.
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
        # 奛懦伯・ｿ・・head ・ｼ・ｸ・ｴ ・・菩メ・・愍・・・ｧ・ ・ｽ・ｰ(cut_gap ・ｼ・・,
        # ・ｸ・・・・罷誤畠﨑俯ｩｴ -CV ・ｸ・ｴ・ ・賀ｲｩ德・・・牟・､ ・・・溢牟 ・ｴ・們・愍・・・晤楫﨑罹共.
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


_PITCH_NOTE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])([A-Ga-g])([#b]?)(-?[0-8])(?![A-Za-z0-9])")


def _note_to_hz(note: str, accidental: str, octave: str) -> Optional[float]:
    pitch_class = {
        "c": 0,
        "d": 2,
        "e": 4,
        "f": 5,
        "g": 7,
        "a": 9,
        "b": 11,
    }
    key = str(note or "").strip().lower()
    if key not in pitch_class:
        return None
    try:
        octave_val = int(str(octave or "").strip())
    except Exception:
        return None
    semitone = int(pitch_class[key])
    acc = str(accidental or "").strip()
    if acc == "#":
        semitone += 1
    elif acc == "b":
        semitone -= 1
    midi = (int(octave_val) + 1) * 12 + int(semitone)
    return float(440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0)))


def _extract_pitch_hint_hz(*tokens: str) -> Optional[float]:
    for raw in tokens:
        text = str(raw or "").strip()
        if not text:
            continue
        candidates = []
        stem = os.path.splitext(os.path.basename(text))[0]
        if stem:
            candidates.append(stem)
        base = os.path.basename(text)
        if base and base not in candidates:
            candidates.append(base)
        try:
            norm_path = os.path.normpath(text)
            for part in str(norm_path).split(os.sep):
                p = str(part or "").strip()
                if p and p not in candidates:
                    candidates.append(p)
        except Exception:
            pass

        matched = []
        for cand in candidates:
            for m in _PITCH_NOTE_TOKEN_RE.finditer(cand):
                hz = _note_to_hz(m.group(1), m.group(2), m.group(3))
                if hz is not None:
                    matched.append((m.start(), hz))
        if matched:
            # Prefer the latest token in text fragments (usually nearest to file/folder suffix).
            matched.sort(key=lambda x: x[0], reverse=True)
            return float(matched[0][1])
    return None


def _resolve_adaptive_f0_profile(*, sr: int, wav_name_hint: str = "", wav_path_hint: str = "") -> Dict[str, float]:
    base_min = max(40.0, float(_env_float("UTOA_F0_MIN_HZ_BASE", 70.0)))
    base_max = max(base_min + 60.0, float(_env_float("UTOA_F0_MAX_HZ_BASE", 500.0)))
    hard_min = max(35.0, float(_env_float("UTOA_F0_MIN_HZ_HARD", 45.0)))
    hard_max = max(base_max + 120.0, float(_env_float("UTOA_F0_MAX_HZ_HARD", 1200.0)))

    note_hz = _extract_pitch_hint_hz(wav_name_hint, wav_path_hint)
    pitch_zone = "unknown"
    f0_min_hz = float(base_min)
    f0_max_hz = float(base_max)

    if note_hz is not None and note_hz > 0.0:
        pitch_zone = "mid"
        if note_hz >= 500.0:
            pitch_zone = "high"
        elif note_hz <= 260.0:
            pitch_zone = "low"

        low_ratio = max(0.20, float(_env_float("UTOA_F0_ADAPT_LOW_RATIO", 0.55)))
        high_ratio = max(1.20, float(_env_float("UTOA_F0_ADAPT_HIGH_RATIO", 1.85)))
        f0_min_hz = min(base_min, note_hz * low_ratio)
        f0_max_hz = max(base_max, note_hz * high_ratio)
    else:
        # No note token in filename: still relax max range to reduce high-register misses.
        f0_min_hz = min(base_min, float(_env_float("UTOA_F0_MIN_HZ_NO_HINT", base_min)))
        f0_max_hz = max(base_max, float(_env_float("UTOA_F0_MAX_HZ_NO_HINT", 720.0)))

    f0_min_hz = max(hard_min, min(f0_min_hz, hard_max - 90.0))
    f0_max_hz = min(hard_max, max(f0_max_hz, f0_min_hz + 90.0))

    clarity_floor = float(_env_float("UTOA_F0_CLARITY_MIN", 0.25))
    if pitch_zone == "high":
        clarity_floor = max(0.16, clarity_floor - float(_env_float("UTOA_F0_CLARITY_RELAX_HIGH", 0.04)))
    elif pitch_zone == "low":
        clarity_floor = max(0.18, clarity_floor - float(_env_float("UTOA_F0_CLARITY_RELAX_LOW", 0.02)))

    return {
        "f0_min_hz": float(f0_min_hz),
        "f0_max_hz": float(f0_max_hz),
        "clarity_floor": float(clarity_floor),
        "note_hint_hz": float(note_hz) if note_hz is not None else 0.0,
        "pitch_zone": str(pitch_zone),
        "sample_rate": float(max(1, int(sr or 1))),
    }


def _estimate_f0_voicing_strength(
    frame,
    sr,
    *,
    f0_min_hz: float = 70.0,
    f0_max_hz: float = 500.0,
    clarity_floor: float = 0.25,
):
    """
    ・・卿﨑・・専ｸｰ・・ｴ ・ｰ・・・・ｱ・・0~1) ・肥・
    ・倣剳﨑・F0 ・肥・擽 ・・笈・ｼ ・ｴ・・・ｴ・ｰ ・岺罹｡罹ｧ・・ｬ・ｩ﨑ｩ・壱共.
    """
    if np is None or frame is None or len(frame) < 96 or sr <= 0:
        return 0.0
    x = frame.astype(np.float64)
    x = x - np.mean(x)
    if len(x) >= 5:
        x = np.convolve(x, np.array([0.2, 0.6, 0.2], dtype=np.float64), mode="same")
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    if rms < 1e-4:
        return 0.0

    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    if len(ac) < 8:
        return 0.0
    ac0 = float(ac[0])
    if ac0 <= 1e-9:
        return 0.0

    f0_lo = max(35.0, float(f0_min_hz))
    f0_hi = max(f0_lo + 10.0, float(f0_max_hz))
    min_lag = max(2, int(sr / f0_hi))
    max_lag = min(len(ac) - 1, int(sr / f0_lo))
    if max_lag <= min_lag + 1:
        return 0.0

    search = ac[min_lag:max_lag + 1]
    peak_rel = int(np.argmax(search))
    peak = float(search[peak_rel])
    lag = min_lag + peak_rel
    if lag <= 0:
        return 0.0
    f0 = float(sr) / float(lag)
    if f0 < f0_lo or f0 > f0_hi:
        return 0.0

    clarity = peak / (ac0 + 1e-9)
    if clarity < float(max(0.05, clarity_floor)):
        return 0.0
    zcr = 0.0
    if len(x) >= 2:
        zcr = float(np.mean((x[:-1] * x[1:]) < 0.0))
    zcr_penalty = float(np.clip((zcr - 0.14) / 0.30, 0.0, 1.0))
    harmonic = float(np.clip((clarity - float(max(0.10, clarity_floor))) / 0.45, 0.0, 1.0))
    voiced_conf = harmonic * (1.0 - (0.42 * zcr_penalty))
    return float(np.clip(voiced_conf, 0.0, 1.0))


def _run_length_true(mask):
    if np is None:
        return []
    arr = np.asarray(mask, dtype=bool)
    out = np.zeros(len(arr), dtype=np.float64)
    i = 0
    n = len(arr)
    while i < n:
        if not arr[i]:
            i += 1
            continue
        j = i + 1
        while j < n and arr[j]:
            j += 1
        run_len = float(j - i)
        out[i:j] = run_len
        i = j
    return out


def _build_f0_quality_metrics(
    f0v_arr,
    en_arr,
    en_ma_arr,
    cls_voiced_arr,
    cls_unvoiced_arr,
    cls_breath_arr,
):
    if np is None or f0v_arr is None:
        return {
            "valid_ratio": 0.0,
            "gap_ratio": 0.0,
            "continuity": 0.0,
            "valid_local": np.array([], dtype=np.float64) if np is not None else [],
            "gap_local": np.array([], dtype=np.float64) if np is not None else [],
            "sparse_local": np.array([], dtype=np.float64) if np is not None else [],
            "continuity_local": np.array([], dtype=np.float64) if np is not None else [],
        }

    n = len(f0v_arr)
    if n <= 0:
        return {
            "valid_ratio": 0.0,
            "gap_ratio": 0.0,
            "continuity": 0.0,
            "valid_local": np.zeros(0, dtype=np.float64),
            "gap_local": np.zeros(0, dtype=np.float64),
            "sparse_local": np.zeros(0, dtype=np.float64),
            "continuity_local": np.zeros(0, dtype=np.float64),
        }

    f0 = np.asarray(f0v_arr, dtype=np.float64)
    en = np.asarray(en_arr, dtype=np.float64) if en_arr is not None and len(en_arr) == n else np.zeros(n, dtype=np.float64)
    en_ma = np.asarray(en_ma_arr, dtype=np.float64) if en_ma_arr is not None and len(en_ma_arr) == n else en
    cls_voiced = (
        np.asarray(cls_voiced_arr, dtype=np.float64)
        if cls_voiced_arr is not None and len(cls_voiced_arr) == n
        else np.zeros(n, dtype=np.float64)
    )
    cls_unvoiced = (
        np.asarray(cls_unvoiced_arr, dtype=np.float64)
        if cls_unvoiced_arr is not None and len(cls_unvoiced_arr) == n
        else np.zeros(n, dtype=np.float64)
    )
    cls_breath = (
        np.asarray(cls_breath_arr, dtype=np.float64)
        if cls_breath_arr is not None and len(cls_breath_arr) == n
        else np.zeros(n, dtype=np.float64)
    )

    valid_th = float(_env_float("UTOA_F0_VALID_TH", 0.46))
    cand_energy_th = float(_env_float("UTOA_F0_CAND_ENERGY_TH", 0.12))
    cand_voiced_th = float(_env_float("UTOA_F0_CAND_VOICED_TH", 0.34))
    exclude_mask = (cls_unvoiced >= 0.5) | (cls_breath >= 0.5)
    candidate_mask = (~exclude_mask) & (
        (en_ma >= cand_energy_th)
        | (cls_voiced >= cand_voiced_th)
        | (f0 >= max(0.24, valid_th * 0.62))
    )
    if not np.any(candidate_mask):
        candidate_mask = (~exclude_mask) & (en_ma >= max(0.08, cand_energy_th * 0.8))
    if not np.any(candidate_mask):
        candidate_mask = (~exclude_mask)

    valid_mask = candidate_mask & (f0 >= valid_th)
    gap_mask = candidate_mask & (~valid_mask)

    run_len = _run_length_true(valid_mask)
    sparse_run_max = int(max(1, _env_int("UTOA_F0_SPARSE_RUN_MAX_FRAMES", 3)))
    sparse_mask = valid_mask & (run_len <= float(sparse_run_max))

    cont_ref = max(2.0, float(_env_float("UTOA_F0_CONTINUITY_REF_FRAMES", 7.0)))
    cont_signal = np.zeros(n, dtype=np.float64)
    if np.any(valid_mask):
        cont_signal[valid_mask] = np.clip(run_len[valid_mask] / cont_ref, 0.0, 1.0)

    local_window = int(max(5, _env_int("UTOA_F0_LOCAL_WINDOW_FRAMES", 9)))
    if local_window % 2 == 0:
        local_window += 1
    kernel = np.ones(local_window, dtype=np.float64) / float(local_window)
    eps = 1.0e-6

    cand_den = np.convolve(candidate_mask.astype(np.float64), kernel, mode="same")
    valid_den = np.convolve(valid_mask.astype(np.float64), kernel, mode="same")
    valid_local = np.convolve(valid_mask.astype(np.float64), kernel, mode="same") / np.maximum(cand_den, eps)
    gap_local = np.convolve(gap_mask.astype(np.float64), kernel, mode="same") / np.maximum(cand_den, eps)
    sparse_local = np.convolve(sparse_mask.astype(np.float64), kernel, mode="same") / np.maximum(cand_den, eps)
    continuity_num = np.convolve(cont_signal, kernel, mode="same")
    continuity_local = np.where(valid_den > eps, continuity_num / np.maximum(valid_den, eps), 0.0)

    valid_local = np.clip(valid_local, 0.0, 1.0)
    gap_local = np.clip(gap_local, 0.0, 1.0)
    sparse_local = np.clip(sparse_local, 0.0, 1.0)
    continuity_local = np.clip(continuity_local, 0.0, 1.0)

    cand_cnt = int(np.sum(candidate_mask))
    if cand_cnt <= 0:
        valid_ratio = 0.0
        gap_ratio = 0.0
    else:
        valid_ratio = float(np.sum(valid_mask)) / float(cand_cnt)
        gap_ratio = float(np.sum(gap_mask)) / float(cand_cnt)

    if np.any(valid_mask):
        continuity = float(np.mean(cont_signal[valid_mask]))
    else:
        continuity = 0.0

    return {
        "valid_ratio": float(np.clip(valid_ratio, 0.0, 1.0)),
        "gap_ratio": float(np.clip(gap_ratio, 0.0, 1.0)),
        "continuity": float(np.clip(continuity, 0.0, 1.0)),
        "valid_local": valid_local,
        "gap_local": gap_local,
        "sparse_local": sparse_local,
        "continuity_local": continuity_local,
    }


def _mel_envelope(audio, sr, wav_name_hint: str = "", wav_path_hint: str = ""):
    if np is None or audio is None or sr is None or len(audio) == 0:
        return None
    n_fft = 1024
    hop = max(1, int(sr * 0.005))
    win = min(n_fft, max(256, int(sr * 0.025)))
    window = np.hanning(win).astype(np.float64)
    f0_profile = _resolve_adaptive_f0_profile(
        sr=int(sr),
        wav_name_hint=str(wav_name_hint or ""),
        wav_path_hint=str(wav_path_hint or ""),
    )
    f0_min_hz = float(f0_profile.get("f0_min_hz", 70.0) or 70.0)
    f0_max_hz = float(f0_profile.get("f0_max_hz", 500.0) or 500.0)
    f0_clarity_floor = float(f0_profile.get("clarity_floor", 0.25) or 0.25)

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
    db_vals = []
    f0_voicing = []
    f2_ratio_vals = []
    f3_ratio_vals = []
    high_ratio_vals = []
    low_ratio_vals = []
    spec_presence_vals = []
    times = []
    last_voicing = 0.0
    f0_stride = int(max(1, _env_int("UTOA_F0_SAMPLE_STRIDE", 2)))
    frame_idx = 0
    for st in range(0, max(len(audio) - win + 1, 1), hop):
        fr_raw = audio[st:st + win]
        if len(fr_raw) < win:
            fr_raw = np.concatenate([fr_raw, np.zeros(win - len(fr_raw), dtype=np.float32)], axis=0)

        rms = float(np.sqrt(np.mean(fr_raw.astype(np.float64) ** 2) + 1e-12))
        db_vals.append(20.0 * np.log10(max(rms, 1e-7)))

        # F0・・・・・卓ｹ・・ｴ・ｰ ・嶸ｸ・・・懋ｳｵ﨑俯巡・・・・・・・・げ.
        if (frame_idx % f0_stride) == 0 or last_voicing <= 0.05:
            curr_voicing = _estimate_f0_voicing_strength(
                fr_raw,
                sr,
                f0_min_hz=f0_min_hz,
                f0_max_hz=f0_max_hz,
                clarity_floor=f0_clarity_floor,
            )
            if frame_idx > 0:
                blend = float(_env_float("UTOA_F0_BLEND", 0.72))
                blend = max(0.0, min(1.0, blend))
                last_voicing = ((1.0 - blend) * float(last_voicing)) + (blend * float(curr_voicing))
            else:
                last_voicing = float(curr_voicing)
        else:
            decay = float(_env_float("UTOA_F0_HOLD_DECAY", 0.02))
            last_voicing = max(0.0, float(last_voicing) - max(0.0, decay))
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

    voiced_formant = (
        ((f2_arr > f2_strong_th) & (f3_arr > f3_strong_th) & (db_arr > (db_sil_th + 1.2)))
        | ((f2_arr > max(0.08, f2_p60 * 0.85)) & (f0v_arr > 0.38) & (db_arr > (db_sil_th - 1.0)))
    )
    silence_sparse = (
        (db_arr <= (db_sil_th - 1.5))
        & (en <= 0.12)
        & (spec_presence_arr <= max(0.10, spec_p40 * 0.92))
    )
    unvoiced_diffuse = (
        (high_arr >= max(0.12, high_noise_th * 0.88))
        & (spec_presence_arr >= max(0.08, spec_noise_th * 0.86))
        & (db_arr > (db_sil_th - 2.0))
        & (f2_arr < max(0.05, f2_strong_th * 0.92))
        & (f3_arr < max(0.04, f3_strong_th * 0.92))
        & ~voiced_formant
    )
    breath_like = (
        (en <= 0.24)
        & (db_arr <= (db_sil_th + 5.5))
        & (high_arr >= max(0.08, high_p65 * 0.72))
        & (spec_presence_arr >= max(0.05, spec_p40 * 0.62))
        & ~voiced_formant
    )
    f0_metrics = _build_f0_quality_metrics(
        f0v_arr=f0v_arr,
        en_arr=en,
        en_ma_arr=en_ma,
        cls_voiced_arr=np.asarray(voiced_formant, dtype=np.float64),
        cls_unvoiced_arr=np.asarray(unvoiced_diffuse, dtype=np.float64),
        cls_breath_arr=np.asarray(breath_like, dtype=np.float64),
    )
    return {
        "times_ms": np.array(times, dtype=np.float64),
        "energy": en,
        "energy_ma": en_ma,
        "span": span,
        "db_db": db_arr,
        "db_silence_th": float(db_sil_th),
        "f0_voicing": f0v_arr,
        "f0_min_hz": float(f0_min_hz),
        "f0_max_hz": float(f0_max_hz),
        "f0_clarity_floor": float(f0_clarity_floor),
        "f0_note_hint_hz": float(f0_profile.get("note_hint_hz", 0.0) or 0.0),
        "f0_pitch_zone": str(f0_profile.get("pitch_zone", "unknown") or "unknown"),
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
        "f0_valid_ratio": float(f0_metrics.get("valid_ratio", 0.0) or 0.0),
        "f0_gap_ratio": float(f0_metrics.get("gap_ratio", 0.0) or 0.0),
        "f0_continuity": float(f0_metrics.get("continuity", 0.0) or 0.0),
        "f0_valid_local": np.asarray(f0_metrics.get("valid_local"), dtype=np.float64),
        "f0_gap_local": np.asarray(f0_metrics.get("gap_local"), dtype=np.float64),
        "f0_sparse_local": np.asarray(f0_metrics.get("sparse_local"), dtype=np.float64),
        "f0_continuity_local": np.asarray(f0_metrics.get("continuity_local"), dtype=np.float64),
        "voiced_mask": np.asarray(f0v_arr >= 0.5, dtype=np.float64),
        "formant_mask": np.asarray(voiced_formant, dtype=np.float64),
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
    voiced_mask = np.asarray(f0v, dtype=np.float64) >= 0.5
    energy_mask = (en_ma >= 0.12) | (np.asarray(en, dtype=np.float64) >= 0.15)

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


def _compute_energy_centroid(mel_ctx, start_ms: float, end_ms: float) -> Optional[float]:
    if np is None or not mel_ctx:
        return None
    try:
        s_ms = float(start_ms)
        e_ms = float(end_ms)
    except Exception:
        return None
    if e_ms <= s_ms:
        return None

    times_ms = mel_ctx.get("times_ms")
    en = mel_ctx.get("energy")
    if times_ms is None or en is None or len(times_ms) == 0 or len(en) != len(times_ms):
        return None

    times_arr = np.asarray(times_ms, dtype=np.float64)
    en_arr = np.asarray(en, dtype=np.float64)
    mask = np.where((times_arr >= s_ms) & (times_arr <= e_ms))[0]
    if len(mask) < 2:
        return None

    weights = np.clip(en_arr[mask], 0.0, None)
    w_sum = float(np.sum(weights))
    if w_sum <= 1e-8:
        return None
    centroid = float(np.sum(times_arr[mask] * weights) / w_sum)
    if centroid < s_ms or centroid > e_ms:
        return None
    return centroid


def _estimate_onset_rise_time(
    mel_ctx,
    c_start_ms: float,
    vowel_start_ms: float,
    vowel_end_ms: float,
) -> Optional[float]:
    if np is None or not mel_ctx:
        return None
    try:
        c_start = float(c_start_ms)
        v_start = float(vowel_start_ms)
        v_end = float(vowel_end_ms)
    except Exception:
        return None
    if v_end <= v_start:
        return None

    times_ms = mel_ctx.get("times_ms")
    en = mel_ctx.get("energy")
    if times_ms is None or en is None or len(times_ms) == 0 or len(en) != len(times_ms):
        return None

    times_arr = np.asarray(times_ms, dtype=np.float64)
    en_arr = np.asarray(en, dtype=np.float64)

    peak_end = min(v_end, v_start + 260.0)
    peak_mask = np.where((times_arr >= v_start) & (times_arr <= peak_end))[0]
    if len(peak_mask) == 0:
        return None

    peak_rel = int(np.argmax(en_arr[peak_mask]))
    peak_idx = int(peak_mask[peak_rel])
    peak_ms = float(times_arr[peak_idx])
    rise_ms = max(0.0, peak_ms - c_start)
    if rise_ms <= 0.0:
        return None
    return rise_ms


def _estimate_context_avg_f0_hz(mel_ctx) -> float:
    if not mel_ctx:
        return 0.0
    note_hz = float(mel_ctx.get("f0_note_hint_hz", 0.0) or 0.0)
    f0_max_hz = float(mel_ctx.get("f0_max_hz", 0.0) or 0.0)
    if note_hz > 0.0 and f0_max_hz > 0.0:
        f0_cap = min(f0_max_hz, max(note_hz * 1.8, note_hz))
        return float((note_hz * 0.72) + (f0_cap * 0.28))
    if note_hz > 0.0:
        return float(note_hz)
    if f0_max_hz > 0.0:
        return float(f0_max_hz * 0.72)
    return 0.0


def _apply_kr_mel_naturalness_adjustments(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    *,
    alias_type: str,
    c_start_ms: float,
    c_end_ms: float,
    vowel_start_ms: float,
    vowel_end_ms: float,
    mel_ctx=None,
):
    if np is None or not mel_ctx:
        return offset, consonant, cutoff, pre, ovl, {}

    alias_key = str(alias_type or "").strip().lower()
    if alias_key not in {"cv", "cv_head", "vcv"}:
        return offset, consonant, cutoff, pre, ovl, {}

    info = {}
    offset, consonant, cutoff, pre, ovl = validate_oto_params(
        offset, consonant, cutoff, pre, ovl, alias_type=alias_key
    )

    # (7) VCV: align preutterance toward consonant energy centroid.
    if alias_key == "vcv" and not _env_bool("UTOA_DISABLE_VCV_CENTROID_ALIGN", False):
        centroid_ms = _compute_energy_centroid(mel_ctx, c_start_ms, c_end_ms)
        if centroid_ms is not None:
            midpoint = (float(c_start_ms) + float(c_end_ms)) * 0.5
            pre_shift = _clamp(float(centroid_ms) - midpoint, -15.0, 15.0)
            if abs(pre_shift) >= 0.5:
                prev_pre = float(pre)
                ovl_gap = max(float(prev_pre) - float(ovl), 4.0)
                cons_gap = max(float(consonant) - float(prev_pre), 10.0)
                cut_gap = max(abs(float(cutoff)) - float(consonant), 12.0)
                pre = max(6.0, float(prev_pre) + float(pre_shift))
                ovl = max(0.0, float(pre) - ovl_gap)
                consonant = float(pre) + cons_gap
                cutoff = -(float(consonant) + cut_gap)
                offset, consonant, cutoff, pre, ovl = validate_oto_params(
                    offset, consonant, cutoff, pre, ovl, alias_type=alias_key
                )
                info["vcv_centroid_ms"] = float(centroid_ms)
                info["vcv_pre_shift_ms"] = float(pre_shift)

    # (8) Consonant velocity correction by onset rise time.
    if not _env_bool("UTOA_DISABLE_KR_RISE_TIME_CORRECTION", False):
        rise_ms = _estimate_onset_rise_time(mel_ctx, c_start_ms, vowel_start_ms, vowel_end_ms)
        if rise_ms is not None:
            rise_ref_ms = _env_float("UTOA_KR_RISE_REF_MS", 70.0)
            rise_mul = _env_float("UTOA_KR_RISE_CORR_PER_MS", 0.15)
            rise_corr = _clamp((float(rise_ms) - float(rise_ref_ms)) * float(rise_mul), -12.0, 12.0)
            if abs(rise_corr) >= 0.25:
                cons_gap = max(float(consonant) - float(pre), 10.0)
                cut_gap = max(abs(float(cutoff)) - float(consonant), 10.0)
                gap_min = 14.0 if alias_key == "vcv" else 16.0
                gap_max = 220.0 if alias_key == "vcv" else 200.0
                new_cons_gap = _clamp(cons_gap + float(rise_corr), gap_min, gap_max)
                consonant = float(pre) + float(new_cons_gap)
                cutoff = -(float(consonant) + cut_gap)
                offset, consonant, cutoff, pre, ovl = validate_oto_params(
                    offset, consonant, cutoff, pre, ovl, alias_type=alias_key
                )
                info["rise_time_ms"] = float(rise_ms)
                info["rise_corr_ms"] = float(rise_corr)

    # (9) High-F0: reduce cutoff span to avoid over-tail in high register.
    if alias_key in {"cv", "cv_head"} and not _env_bool("UTOA_DISABLE_KR_HIGH_F0_CUTOFF_COMPRESS", False):
        pitch_zone = str(mel_ctx.get("f0_pitch_zone", "") or "").strip().lower()
        note_hz = float(mel_ctx.get("f0_note_hint_hz", 0.0) or 0.0)
        f0_max_hz = float(mel_ctx.get("f0_max_hz", 0.0) or 0.0)
        avg_f0_hz = _estimate_context_avg_f0_hz(mel_ctx)
        high_ctx = _is_high_pitch_context(
            pitch_zone=pitch_zone,
            note_hint_hz=note_hz,
            f0_max_hz=f0_max_hz,
        )
        if high_ctx and avg_f0_hz >= 500.0:
            cut_abs = abs(float(cutoff))
            cut_gap = max(cut_abs - float(consonant), 10.0)
            scaled_gap = max(8.0, cut_gap * 0.85)
            if scaled_gap <= (cut_gap - 0.8):
                reduction = min(cut_gap - scaled_gap, 24.0)
                new_cut_abs = float(consonant) + max(8.0, cut_gap - reduction)
                cutoff = -new_cut_abs
                offset, consonant, cutoff, pre, ovl = validate_oto_params(
                    offset, consonant, cutoff, pre, ovl, alias_type=alias_key
                )
                info["high_f0_avg_hz"] = float(avg_f0_hz)
                info["high_f0_cut_reduced_ms"] = float(cut_abs - abs(float(cutoff)))

    return offset, consonant, cutoff, pre, ovl, info


def _resolve_mel_weight_mode(raw_value: str) -> str:
    mode = str(raw_value or "").strip().lower()
    if mode in {"mel_boost", "mel-boost", "melboost", "boost", "mel"}:
        return "mel_boost"
    return "auto"


def _estimate_file_mel_reliability(mel_ctx, *, blank_conf_mean: float = 0.0) -> float:
    if not mel_ctx:
        return 0.0
    if np is None:
        return float(_clamp(1.0 - (float(blank_conf_mean) * 0.85), 0.0, 1.0))

    en = mel_ctx.get("energy")
    f0v = mel_ctx.get("f0_voicing")
    cls_sil = mel_ctx.get("cls_silence_sparse")
    if en is None or len(en) == 0:
        return float(_clamp(1.0 - (float(blank_conf_mean) * 0.85), 0.0, 1.0))
    en_arr = np.asarray(en, dtype=np.float64)
    if f0v is None or len(f0v) != len(en_arr):
        f0_arr = np.zeros_like(en_arr, dtype=np.float64)
    else:
        f0_arr = np.asarray(f0v, dtype=np.float64)
    if cls_sil is None or len(cls_sil) != len(en_arr):
        sil_arr = np.zeros_like(en_arr, dtype=np.float64)
    else:
        sil_arr = np.asarray(cls_sil, dtype=np.float64)

    energy_mean = float(np.mean(en_arr))
    voiced_ratio = float(np.mean(f0_arr >= 0.5))
    silence_ratio = float(np.mean(sil_arr >= 0.5))
    blank_penalty = max(0.0, float(blank_conf_mean) - 0.35)
    score = (
        (0.46 * _clamp(voiced_ratio, 0.0, 1.0))
        + (0.34 * _clamp(1.0 - silence_ratio, 0.0, 1.0))
        + (0.20 * _clamp(energy_mean, 0.0, 1.0))
        - (0.38 * _clamp(blank_penalty, 0.0, 1.0))
    )
    return float(_clamp(score, 0.0, 1.0))


def _apply_mel_boost_alignment_scale(alignment_weight: float, textgrid_trust_tier: str) -> float:
    weight = float(alignment_weight or 0.0)
    tier = str(textgrid_trust_tier or "").strip().lower()
    if tier == "low":
        weight *= 0.72
    elif tier == "mid":
        weight *= 0.84
    else:
        weight *= 0.94
    return float(_clamp(weight, 0.0, 1.0))


def _resolve_mel_reliability_floor(mel_ctx, *, default_floor: Optional[float] = None) -> float:
    base_floor = float(default_floor) if default_floor is not None else float(_env_float("UTOA_MEL_FORCE_MEL_RELIABILITY_MIN", 0.44))
    base_floor = float(_clamp(base_floor, 0.20, 0.95))
    if not mel_ctx:
        return base_floor

    note_hz = float(mel_ctx.get("f0_note_hint_hz", 0.0) or 0.0)
    f0_max_hz = float(mel_ctx.get("f0_max_hz", 0.0) or 0.0)
    pitch_zone = str(mel_ctx.get("f0_pitch_zone", "") or "").strip().lower()
    relax = 0.0

    if pitch_zone == "high" or note_hz >= 500.0 or f0_max_hz >= 860.0:
        relax += float(_env_float("UTOA_MEL_FORCE_RELAX_HIGH", 0.10))
    elif note_hz >= 430.0 or f0_max_hz >= 700.0:
        relax += float(_env_float("UTOA_MEL_FORCE_RELAX_UPPER_MID", 0.05))

    if pitch_zone == "low" or (0.0 < note_hz <= 250.0):
        relax += float(_env_float("UTOA_MEL_FORCE_RELAX_LOW", 0.06))
    elif 0.0 < note_hz <= 300.0:
        relax += float(_env_float("UTOA_MEL_FORCE_RELAX_LOWER_MID", 0.03))

    floor = float(base_floor - max(0.0, relax))
    return float(_clamp(floor, 0.20, 0.95))


def _should_force_mel_branch(
    textgrid_trust_tier: str,
    *,
    trust_score: Optional[float] = None,
    alignment_weight: Optional[float] = None,
    mapping_confidence: Optional[float] = None,
    mel_reliability: Optional[float] = None,
    mel_reliability_floor: Optional[float] = None,
) -> bool:
    tier = str(textgrid_trust_tier or "").strip().lower()
    if tier != "high":
        return True
    trust_min = _env_float("UTOA_MEL_FORCE_TRUST_MIN", 0.82)
    align_min = _env_float("UTOA_MEL_FORCE_ALIGN_WEIGHT_MIN", 0.70)
    map_min = _env_float("UTOA_MEL_FORCE_MAPPING_CONF_MIN", 0.58)
    if mel_reliability_floor is None:
        mel_rel_min = _env_float("UTOA_MEL_FORCE_MEL_RELIABILITY_MIN", 0.44)
    else:
        mel_rel_min = float(mel_reliability_floor)
    if trust_score is not None and float(trust_score) < float(trust_min):
        return True
    if alignment_weight is not None and float(alignment_weight) < float(align_min):
        return True
    if mapping_confidence is not None and float(mapping_confidence) < float(map_min):
        return True
    if mel_reliability is not None and float(mel_reliability) < float(mel_rel_min):
        return True
    return False


def _resolve_mel_onset_weight(
    alignment_weight: float,
    textgrid_trust_tier: str,
    *,
    trust_score: Optional[float] = None,
    mapping_confidence: Optional[float] = None,
    mel_reliability: Optional[float] = None,
    mel_reliability_floor: Optional[float] = None,
) -> float:
    tier = str(textgrid_trust_tier or "").strip().lower()
    w = float(alignment_weight or 0.0)
    mode = _resolve_mel_weight_mode(os.environ.get("UTOA_MEL_WEIGHT_MODE", "auto"))
    if _should_force_mel_branch(
        tier,
        trust_score=trust_score,
        alignment_weight=w,
        mapping_confidence=mapping_confidence,
        mel_reliability=mel_reliability,
        mel_reliability_floor=mel_reliability_floor,
    ):
        mode = "mel_boost"
    if tier == "high" and w >= 0.75:
        return 0.0
    if w < 0.45:
        base = 0.72
    elif w < 0.65:
        base = 0.58
    else:
        base = 0.36
    if mode == "mel_boost":
        if tier == "low":
            base = min(0.90, base + 0.16)
        elif tier == "mid":
            base = min(0.82, base + 0.10)
        else:
            base = min(0.68, base + 0.06)
        if w < 0.35:
            base = min(0.92, base + 0.04)
    return max(0.0, min(1.0, base))


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
    f0_gap_local_arr = mel_ctx.get("f0_gap_local")
    f0_sparse_local_arr = mel_ctx.get("f0_sparse_local")
    f0_cont_local_arr = mel_ctx.get("f0_continuity_local")
    f0_valid_ratio = float(mel_ctx.get("f0_valid_ratio", 0.0) or 0.0)
    f0_gap_ratio = float(mel_ctx.get("f0_gap_ratio", 0.0) or 0.0)
    f0_cont_global = float(mel_ctx.get("f0_continuity", 0.0) or 0.0)
    f0_note_hint_hz = float(mel_ctx.get("f0_note_hint_hz", 0.0) or 0.0)
    f0_max_hz = float(mel_ctx.get("f0_max_hz", 0.0) or 0.0)
    f0_pitch_zone = str(mel_ctx.get("f0_pitch_zone", "") or "").strip().lower()

    sil = float(cls_sil[idx]) if cls_sil is not None and len(cls_sil) == len(times_ms) else 0.0
    voiced = float(cls_voiced[idx]) if cls_voiced is not None and len(cls_voiced) == len(times_ms) else 0.0
    unvoiced = float(cls_unvoiced[idx]) if cls_unvoiced is not None and len(cls_unvoiced) == len(times_ms) else 0.0
    breath = float(cls_breath[idx]) if cls_breath is not None and len(cls_breath) == len(times_ms) else 0.0
    db_sparse = 0.0
    if db_arr is not None and len(db_arr) == len(times_ms):
        db_sparse = 1.0 if float(db_arr[idx]) <= (db_sil_th - 1.0) else 0.0

    gap_penalty, sparse_penalty, continuity_bonus = _compute_f0_blank_adjustment(
        idx=idx,
        n=len(times_ms),
        unvoiced=unvoiced,
        breath=breath,
        f0_gap_local_arr=f0_gap_local_arr,
        f0_sparse_local_arr=f0_sparse_local_arr,
        f0_cont_local_arr=f0_cont_local_arr,
        f0_valid_ratio=f0_valid_ratio,
        f0_gap_ratio=f0_gap_ratio,
        f0_cont_global=f0_cont_global,
        f0_note_hint_hz=f0_note_hint_hz,
        f0_max_hz=f0_max_hz,
        f0_pitch_zone=f0_pitch_zone,
    )
    blank = (
        (0.58 * sil)
        + (0.20 * breath)
        + (0.22 * db_sparse)
        - (0.46 * voiced)
        - (0.12 * unvoiced)
        + gap_penalty
        + sparse_penalty
        - continuity_bonus
    )
    return max(0.0, min(1.0, float(blank)))


def _mel_local_prob(arr, idx, n, fallback=0.0):
    try:
        if arr is None or len(arr) != int(n):
            return max(0.0, min(1.0, float(fallback)))
        return max(0.0, min(1.0, float(arr[int(idx)])))
    except Exception:
        return max(0.0, min(1.0, float(fallback)))


def _compute_f0_blank_adjustment(
    *,
    idx,
    n,
    unvoiced,
    breath,
    f0_gap_local_arr,
    f0_sparse_local_arr,
    f0_cont_local_arr,
    f0_valid_ratio,
    f0_gap_ratio,
    f0_cont_global,
    f0_note_hint_hz=0.0,
    f0_max_hz=0.0,
    f0_pitch_zone="",
):
    f0_gap_local = _mel_local_prob(f0_gap_local_arr, idx, n, fallback=f0_gap_ratio)
    f0_sparse_local = _mel_local_prob(f0_sparse_local_arr, idx, n, fallback=0.0)
    f0_cont_local = _mel_local_prob(f0_cont_local_arr, idx, n, fallback=f0_cont_global)

    gap_w = float(_env_float("UTOA_F0_GAP_PENALTY_WEIGHT", 0.24))
    sparse_w = float(_env_float("UTOA_F0_SPARSE_PENALTY_WEIGHT", 0.16))
    cont_w = float(_env_float("UTOA_F0_CONTINUITY_BONUS_WEIGHT", 0.18))
    pitch_zone = str(f0_pitch_zone or "").strip().lower()
    note_hz = float(f0_note_hint_hz or 0.0)
    max_hz = float(f0_max_hz or 0.0)
    if pitch_zone == "high" or note_hz >= 500.0 or max_hz >= 860.0:
        gap_w *= float(_env_float("UTOA_F0_GAP_PENALTY_SCALE_HIGH", 0.68))
        sparse_w *= float(_env_float("UTOA_F0_SPARSE_PENALTY_SCALE_HIGH", 0.72))
        cont_w *= float(_env_float("UTOA_F0_CONTINUITY_BONUS_SCALE_HIGH", 0.92))
    elif note_hz >= 430.0 or max_hz >= 700.0:
        gap_w *= float(_env_float("UTOA_F0_GAP_PENALTY_SCALE_UPPER_MID", 0.84))
        sparse_w *= float(_env_float("UTOA_F0_SPARSE_PENALTY_SCALE_UPPER_MID", 0.88))
        cont_w *= float(_env_float("UTOA_F0_CONTINUITY_BONUS_SCALE_UPPER_MID", 0.96))
    elif pitch_zone == "low" or (0.0 < note_hz <= 250.0):
        gap_w *= float(_env_float("UTOA_F0_GAP_PENALTY_SCALE_LOW", 0.92))
        sparse_w *= float(_env_float("UTOA_F0_SPARSE_PENALTY_SCALE_LOW", 0.95))
        cont_w *= float(_env_float("UTOA_F0_CONTINUITY_BONUS_SCALE_LOW", 1.00))
    valid_scale = max(0.35, min(1.0, float(f0_valid_ratio) + 0.22))

    if float(unvoiced) >= 0.5 or float(breath) >= 0.5:
        continuity_bonus = 0.0
    else:
        continuity_bonus = cont_w * f0_cont_local * valid_scale
    gap_penalty = gap_w * max(f0_gap_local, min(1.0, float(f0_gap_ratio) + 0.12))
    sparse_penalty = sparse_w * f0_sparse_local
    return float(gap_penalty), float(sparse_penalty), float(continuity_bonus)


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

    sound_mask = (db_sel > (db_sil_th + 1.4)) & (en_sel > 0.10)
    blank_mask = (sil_sel >= 0.50) | ((db_sel <= (db_sil_th - 1.2)) & (en_sel <= 0.10))
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


def _resolve_kr_mel_flux_percentile(syllables_info, mel_ctx):
    if np is None or not mel_ctx:
        return 85.0

    cached = mel_ctx.get("_kr_flux_percentile")
    if cached is not None:
        try:
            return float(cached)
        except Exception:
            pass

    times_ms = mel_ctx.get("times_ms")
    en = mel_ctx.get("energy")
    if times_ms is None or en is None:
        mel_ctx["_kr_flux_percentile"] = 85.0
        return 85.0

    try:
        en_arr = np.asarray(en, dtype=np.float64)
        times_arr = np.asarray(times_ms, dtype=np.float64)
    except Exception:
        mel_ctx["_kr_flux_percentile"] = 85.0
        return 85.0

    if len(en_arr) < 8 or len(times_arr) != len(en_arr):
        mel_ctx["_kr_flux_percentile"] = 85.0
        return 85.0

    wav_rms = float(np.sqrt(np.mean(np.square(en_arr))))
    if wav_rms <= 1e-9:
        mel_ctx["_kr_flux_percentile"] = 85.0
        return 85.0

    feature_rows = []
    for syl in (syllables_info or []):
        start_s = float((syl or {}).get("start_time", 0.0) or 0.0)
        phones = (syl or {}).get("phones") or []
        if start_s <= 0.0 and phones:
            try:
                start_s = float(phones[0].minTime)
            except Exception:
                start_s = 0.0
        t_ms = max(0.0, start_s * 1000.0)
        idx = np.where((times_arr >= t_ms) & (times_arr <= t_ms + 40.0))[0]
        if len(idx) == 0:
            continue
        seg = en_arr[idx]
        seg_rms = float(np.sqrt(np.mean(np.square(seg))))
        feature_rows.append({"rms_norm_wav": float(seg_rms / max(wav_rms, 1e-8))})

    if len(feature_rows) < 4:
        mel_ctx["_kr_flux_percentile"] = 85.0
        return 85.0

    try:
        from core.oto_ml_features import compute_session_rms_percentile

        rms_p25 = float(compute_session_rms_percentile(feature_rows))
    except Exception:
        rms_p25 = -1.0

    flux_percentile = 85.0
    if rms_p25 >= 0.0:
        flux_percentile = 85.0 if rms_p25 > 0.20 else 90.0

    mel_ctx["_kr_flux_percentile"] = float(flux_percentile)
    mel_ctx["_kr_rms_p25"] = float(rms_p25)
    return float(flux_percentile)


def _annotate_kr_syllable_blank_confidence(syllables_info, mel_ctx):
    if not syllables_info:
        return syllables_info
    times_arr = None
    en_arr = None
    flux = None
    flux_threshold = None
    if mel_ctx and np is not None:
        try:
            times_ms = mel_ctx.get("times_ms")
            en = mel_ctx.get("energy")
            if times_ms is not None and en is not None and len(times_ms) > 0 and len(en) == len(times_ms):
                times_arr = np.asarray(times_ms, dtype=np.float64)
                en_arr = np.asarray(en, dtype=np.float64)
                flux = np.diff(en_arr, prepend=en_arr[0])
                flux_percentile = _resolve_kr_mel_flux_percentile(syllables_info, mel_ctx)
                flux_threshold = float(np.percentile(flux, flux_percentile))
        except Exception:
            times_arr = None
            en_arr = None
            flux = None
            flux_threshold = None

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
        # TICKET-008: Annotate onset energy and onset-distance for candidate scoring.
        syl["mel_onset_energy"] = 0.0
        syl["mel_onset_distance_ms"] = 999.0
        if times_arr is not None and en_arr is not None:
            try:
                _window = 25.0
                _mask = np.where((times_arr >= t_ms) & (times_arr <= t_ms + _window))[0]
                if len(_mask) > 0:
                    syl["mel_onset_energy"] = float(np.max(en_arr[_mask]))
                # Distance to nearest flux-based onset within ±50ms
                if flux is not None and flux_threshold is not None:
                    onset_frames = np.where(
                        (flux >= flux_threshold)
                        & (times_arr >= t_ms - 50.0)
                        & (times_arr <= t_ms + 50.0)
                    )[0]
                    if len(onset_frames) > 0:
                        closest = float(np.min(np.abs(times_arr[onset_frames] - t_ms)))
                        syl["mel_onset_distance_ms"] = float(closest)
            except Exception:
                pass
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
    ・ｴ・ｸ ・ｨ・・soft guard:
    - dB ・懍・ ・・ｳ・ｰ・+ mel ・尖ц・・・・ｴ・・・・・・・擽・ｼ ・川ｧ
    - F0 ・・ｱ・・株 ・ｮ・ ・・卓ｹ・・ｴ・ｰ)・罹ｧ・・們・
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

    f2_p60 = float(mel_ctx.get("f2_p60", np.percentile(f2_arr, 60) if len(f2_arr) else 0.0))
    f3_p60 = float(mel_ctx.get("f3_p60", np.percentile(f3_arr, 60) if len(f3_arr) else 0.0))
    high_p65 = float(mel_ctx.get("high_p65", np.percentile(high_arr, 65) if len(high_arr) else 0.0))
    spec_p40 = float(mel_ctx.get("spec_p40", np.percentile(spec_presence_arr, 40) if len(spec_presence_arr) else 0.0))

    base_f2_strong = _env_float("UTOA_MEL_F2_STRONG_MIN", 0.115)
    base_f3_strong = _env_float("UTOA_MEL_F3_STRONG_MIN", 0.082)
    base_high_noise = _env_float("UTOA_MEL_HIGH_NOISE_MIN", 0.20)
    base_spec_noise = _env_float("UTOA_MEL_SPEC_NOISE_MIN", 0.14)
    base_sound_db_margin = _env_float("UTOA_MEL_SOUND_DB_MARGIN", 1.4)
    base_sound_energy = _env_float("UTOA_MEL_SOUND_ENERGY_MIN", 0.13)
    base_weak_f2 = _env_float("UTOA_MEL_WEAK_F2_MAX", 0.05)
    base_weak_f3 = _env_float("UTOA_MEL_WEAK_F3_MAX", 0.05)
    base_weak_high = _env_float("UTOA_MEL_WEAK_HIGH_MAX", 0.10)
    hard_sil_db_margin = _env_float("UTOA_MEL_HARD_SILENCE_DB_MARGIN", -3.0)
    hard_sil_energy = _env_float("UTOA_MEL_HARD_SILENCE_ENERGY_MAX", 0.08)
    soft_sil_energy = _env_float("UTOA_MEL_SOFT_SILENCE_ENERGY_MAX", 0.10)
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

    sound_mask = ((db_arr > (db_sil_th + base_sound_db_margin)) & (en > base_sound_energy)) | voiced_formant_mask | noisy_unvoiced_mask
    weak_spec_mask = (f2_arr < base_weak_f2) & (f3_arr < base_weak_f3) & (high_arr < base_weak_high)
    hard_silence_mask = (db_arr <= (db_sil_th + hard_sil_db_margin)) | (en <= hard_sil_energy)
    silence_mask = hard_silence_mask | (((db_arr <= db_sil_th) | (en <= soft_sil_energy)) & weak_spec_mask)
    if cls_silence is not None and len(cls_silence) == len(en):
        silence_mask = silence_mask | (np.asarray(cls_silence, dtype=np.float64) >= 0.5)
    # onset ・ｴ・ｰ 夋川ｧ: ・尖ц・ ・ｴ・呰初・ + 1・ｨ ・ｰ・ｸ・ｰ・・・巐ｨ onset 弡・ｳｴ・ｼ ・誤蕩・､.
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
    if hint in {"ﾉｯ", "a", "i", "u", "e", "o"}:
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
    # ・・ｱ/・・搆 ・・龍(m,n,r,l,w,y...)・ ・・・・ｭ ・尖ц・・ ・ｽ﨑ｴ
    # offset guard・ ・ｨ・・・懍梠・ｼ・・・ｼ・・・ｴ・呰腹 ・・・溢牟 ・ｴ・們・愍・・・俯ｦｬ﨑罹共.
    low_energy_voiced = hint in {
        "m", "n", "ny", "ng", "r", "l", "ry", "w", "y", "j",
        "g", "d", "b", "z", "dz", "v", "gy", "dy", "by",
        "\u014b", "\u0272", "\u027e", "\u0279",
    } or hint.startswith("m")

    # Onset-class aware onset mask tuning:
    # - sibilant/fricative: flatter slope before vowel, so lower slope threshold
    # - plosive: sharper attack, so keep stricter slope threshold
    onset_slope_th = _env_float("UTOA_MEL_ONSET_SLOPE_BASE", 0.015)
    onset_energy_th = _env_float("UTOA_MEL_ONSET_ENERGY_BASE", 0.12)
    onset_db_margin = _env_float("UTOA_MEL_ONSET_DB_MARGIN_BASE", -2.0)
    if is_sibilant_like:
        onset_slope_th = _env_float("UTOA_MEL_ONSET_SLOPE_SIBILANT", 0.010)
        onset_energy_th = _env_float("UTOA_MEL_ONSET_ENERGY_SIBILANT", 0.10)
        onset_db_margin = _env_float("UTOA_MEL_ONSET_DB_MARGIN_SIBILANT", -4.0)
    elif is_plosive_like:
        onset_slope_th = _env_float("UTOA_MEL_ONSET_SLOPE_PLOSIVE", 0.018)
        onset_energy_th = _env_float("UTOA_MEL_ONSET_ENERGY_PLOSIVE", 0.13)
        onset_db_margin = _env_float("UTOA_MEL_ONSET_DB_MARGIN_PLOSIVE", -1.5)
    onset_mask = (en_slope > onset_slope_th) & (en_ma > onset_energy_th) & (db_arr > (db_sil_th + onset_db_margin))
    if is_sibilant_like or is_plosive_like:
        f0_slope_th = _env_float("UTOA_MEL_ONSET_F0_SLOPE_MIN", 0.03)
        onset_energy_relax = _env_float("UTOA_MEL_ONSET_ENERGY_RELAX", 0.02)
        onset_f0_db_margin = _env_float("UTOA_MEL_ONSET_F0_DB_MARGIN", -3.0)
        f0_slope = np.diff(f0v_arr, prepend=float(f0v_arr[0]) if len(f0v_arr) else 0.0)
        onset_mask = onset_mask | (
            (f0_slope > f0_slope_th)
            & (en_ma > max(0.09, onset_energy_th - onset_energy_relax))
            & (db_arr > (db_sil_th + onset_f0_db_margin))
        )

    # ---- soft offset guard ----
    # 﨑懋ｵｭ・ｴ CVVC・・CV/CV_HEAD・・onset anchor・ 弡・卿 guard・護愍・罹巡 ・ｩ・・復 ・ｽ・ｰ・ ・狩共.
    # ・・offset soft guard・ ・緋ｰ・・・､・ｴ・・ｴ ・ｨ・ｨ・・嶹懍搆 ・ｬ・・擽 ・ｽ﨑・甯護攵・川・
    # offset・ｴ ・ｵ・ｱ ・ｽ・ｼ・・・ｼ﨑俾ｲ・・誤ｦｬ・・・ｽ嵂･・ｴ ・､・・共.
    allow_order_locked_cv = _env_bool("UTOA_KR_ALLOW_ORDER_LOCKED_CV_MEL_OFFSET", True)
    skip_offset_soft_guard = (alias_type == "cv_head") or (
        _is_kr_order_locked_cv_format(file_format)
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
                    # onset 弡・ｳｴ・ ・溢愍・ｴ ・ｰ・ｴ sound ・懍梠・尖ｳｴ・､ ・ｰ・ ・ｬ・ｩ﨑罹共.
                    rel = int(np.where(onset_seg)[0][0])
                    sound_start_idx = lo + rel
                else:
                    seg = sound_mask[lo:pre_idx + 1]
                    if np.any(seg):
                        rel = int(np.where(seg)[0][0])
                        sound_start_idx = lo + rel
            if sound_start_idx is not None:
                offset_lead_ms = _env_float("UTOA_MEL_OFFSET_LEAD_BASE_MS", 12.0)
                pre_guard_ms = _env_float("UTOA_MEL_OFFSET_PRE_GUARD_BASE_MS", 18.0)
                if is_sibilant_like or is_fricative_like:
                    offset_lead_ms = _env_float("UTOA_MEL_OFFSET_LEAD_SIBILANT_MS", 15.0)
                    pre_guard_ms = _env_float("UTOA_MEL_OFFSET_PRE_GUARD_SIBILANT_MS", 20.0)
                elif is_plosive_like:
                    offset_lead_ms = _env_float("UTOA_MEL_OFFSET_LEAD_PLOSIVE_MS", 18.0)
                    pre_guard_ms = _env_float("UTOA_MEL_OFFSET_PRE_GUARD_PLOSIVE_MS", 22.0)
                target_offset = float(t_ms[sound_start_idx]) - offset_lead_ms
                target_offset = max(0.0, min(pre_abs - pre_guard_ms, target_offset))
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
                    # F0 ・・ｱ・・株 ・ｴ・ｰ(・・・卓ｹ・・罹ｧ・・們・
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


def _compute_kr_session_vowel_baseline_ms(syllables_info):
    vals = []
    for syl in (syllables_info or []):
        v_len = (syl or {}).get("cv_vowel_len")
        if v_len is None:
            phones = (syl or {}).get("phones") or []
            if phones:
                try:
                    _v_idx, v_phone = find_vowel_phone(phones)
                    v_len = (float(v_phone.maxTime) - float(v_phone.minTime)) * 1000.0
                except Exception:
                    v_len = None
        try:
            fv = float(v_len)
            if fv > 0.0:
                vals.append(fv)
                syl["cv_vowel_len"] = fv
        except Exception:
            continue

    if not vals:
        return 130.0

    if np is not None:
        med = float(np.median(np.asarray(vals, dtype=np.float64)))
    else:
        med = _median(vals)
    return _clamp(med * 0.80, 80.0, 160.0)


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


def _is_high_pitch_context(
    *,
    pitch_zone: str = "",
    note_hint_hz: float = 0.0,
    f0_max_hz: float = 0.0,
) -> bool:
    zone = str(pitch_zone or "").strip().lower()
    note_hz = float(note_hint_hz or 0.0)
    f0_hi = float(f0_max_hz or 0.0)
    high_note_hz = float(_env_float("UTOA_HIGH_PITCH_NOTE_HZ", 523.25))
    return bool(
        zone in {"high", "ultra_high"}
        or note_hz >= high_note_hz
        or f0_hi >= 860.0
    )


def resolve_file_routing_profile(
    *,
    file_format: str,
    pitch_zone: str,
    note_hint_hz: float,
    f0_max_hz: float,
    textgrid_trust_score: float,
    textgrid_trust_tier: str,
    mapping_confidence_base: float,
    blank_conf_mean: float,
    mel_reliability_score: float,
    mel_reliability_floor: float,
    file_mapping_low_conf: bool,
    alignment_weight_base: float,
    mel_weight_mode: str,
    file_mapping_conf_th: float,
) -> Dict[str, object]:
    high_pitch_mode = _is_high_pitch_context(
        pitch_zone=pitch_zone,
        note_hint_hz=note_hint_hz,
        f0_max_hz=f0_max_hz,
    )
    force_mel_branch = _should_force_mel_branch(
        textgrid_trust_tier,
        trust_score=textgrid_trust_score,
        alignment_weight=alignment_weight_base,
        mapping_confidence=mapping_confidence_base,
        mel_reliability=mel_reliability_score,
        mel_reliability_floor=mel_reliability_floor,
    )
    mode = str(mel_weight_mode or "")
    if force_mel_branch:
        mode = "mel_boost"
    align_final = float(alignment_weight_base or 0.0)
    if mode == "mel_boost":
        align_final = _apply_mel_boost_alignment_scale(align_final, textgrid_trust_tier)
    align_final = float(_clamp(align_final, 0.0, 1.0))

    if high_pitch_mode:
        profile_code = "high_pitch_safe"
    elif str(textgrid_trust_tier or "").strip().lower() == "high" and mapping_confidence_base >= 0.72 and blank_conf_mean < 0.45:
        profile_code = "alignment_strong"
    elif str(textgrid_trust_tier or "").strip().lower() == "low" or file_mapping_low_conf or blank_conf_mean >= 0.58:
        profile_code = "acoustic_safe"
    else:
        profile_code = "hybrid_soft"

    delta_clamp_scale = 1.0
    if profile_code == "acoustic_safe":
        delta_clamp_scale = min(delta_clamp_scale, 0.85)
    if high_pitch_mode:
        delta_clamp_scale = min(
            delta_clamp_scale,
            float(_env_float("UTOA_HIGH_PITCH_DELTA_CLAMP_SCALE", 0.55)),
        )
    delta_clamp_scale = float(_clamp(delta_clamp_scale, 0.20, 1.00))

    anchor_lock_lite_default = bool(
        align_final < 0.58
        or blank_conf_mean >= 0.55
        or mapping_confidence_base < float(file_mapping_conf_th)
        or mel_reliability_score < float(mel_reliability_floor)
        or high_pitch_mode
    )

    row_blank_floor = None
    fmt_norm = str(file_format or "").strip().lower()
    if fmt_norm in {"cvvc", "cvc", "cv"}:
        need_blank_gate = bool(file_mapping_low_conf or blank_conf_mean >= 0.45)
        if profile_code in {"acoustic_safe", "high_pitch_safe"}:
            need_blank_gate = True
        if need_blank_gate:
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
            if high_pitch_mode:
                row_blank_floor += float(_env_float("UTOA_HIGH_PITCH_ROW_BLANK_FLOOR_BONUS", 0.04))
            row_blank_floor = float(_clamp(row_blank_floor, 0.30, 0.95))

    return {
        "profile_code": str(profile_code),
        "alignment_weight_base": float(alignment_weight_base),
        "alignment_weight_final": float(align_final),
        "force_mel_branch": bool(force_mel_branch),
        "anchor_lock_lite_default": bool(anchor_lock_lite_default),
        "row_blank_floor": row_blank_floor,
        "delta_clamp_scale": float(delta_clamp_scale),
        "high_pitch_mode": bool(high_pitch_mode),
        "mel_weight_mode": str(mode),
    }


def _apply_mode_rank(mode: str) -> int:
    key = str(mode or "").strip().lower()
    order = {
        "full_apply": 0,
        "conservative_apply": 1,
        "template_preserve": 2,
        "review_required": 3,
    }
    return int(order.get(key, 0))


def decide_row_application(
    *,
    routing_profile: str,
    pitch_zone: str,
    row_mapping_confidence: float,
    row_conf_floor: float,
    row_blank_confidence: Optional[float],
    row_blank_floor: Optional[float],
    row_jump_blocked: int,
    forced_selected: bool,
    file_mapping_low_conf: bool,
    row_abstain_skip: bool,
    row_abstain_reason: str = "",
) -> Dict[str, object]:
    reasons = []
    mode = "full_apply"

    if row_abstain_skip:
        reason = str(row_abstain_reason or "row_abstain").strip().lower() or "row_abstain"
        return {
            "mode": "review_required",
            "reason_code": reason,
            "reasons": [reason],
        }

    if file_mapping_low_conf or float(row_mapping_confidence) < float(row_conf_floor):
        mode = "conservative_apply"
        reasons.append("low_model_conf")
    if (
        row_blank_floor is not None
        and row_blank_confidence is not None
        and float(row_blank_confidence) >= float(row_blank_floor)
    ):
        mode = "conservative_apply"
        reasons.append("blank_conf_high")
    if int(row_jump_blocked or 0) > 0:
        mode = "conservative_apply"
        reasons.append("jump_blocked")
    if bool(forced_selected) and mode in {"full_apply", "conservative_apply"}:
        mode = "template_preserve"
        reasons.append("plan_mismatch")

    high_pitch_mode = _is_high_pitch_context(
        pitch_zone=pitch_zone,
        note_hint_hz=0.0,
        f0_max_hz=0.0,
    )
    if high_pitch_mode and mode != "review_required":
        if float(row_mapping_confidence) < max(0.0, float(row_conf_floor) - 0.06):
            mode = "review_required"
            reasons.append("high_pitch_unstable")

    if str(routing_profile or "").strip().lower() == "high_pitch_safe" and mode == "full_apply":
        mode = "conservative_apply"
        reasons.append("high_pitch_unstable")

    reason_code = str(reasons[0]) if reasons else ""
    return {
        "mode": str(mode),
        "reason_code": reason_code,
        "reasons": list(reasons),
    }


def _apply_conservative_delta_clamp(
    *,
    mode: str,
    delta_clamp_scale: float,
    pitch_zone: str,
    base_shape: Optional[Dict[str, float]],
    offset: float,
    consonant: float,
    cutoff: float,
    pre: float,
    ovl: float,
) -> Tuple[float, float, float, float, float]:
    m = str(mode or "").strip().lower()
    if m not in {"conservative_apply", "template_preserve"}:
        return float(offset), float(consonant), float(cutoff), float(pre), float(ovl)
    if not isinstance(base_shape, dict):
        return float(offset), float(consonant), float(cutoff), float(pre), float(ovl)

    base_offset = float(base_shape.get("offset", offset) or offset)
    base_cons = float(base_shape.get("cons", consonant) or consonant)
    base_cutoff = float(base_shape.get("cutoff", cutoff) or cutoff)
    base_pre = float(base_shape.get("pre", pre) or pre)
    base_ovl = float(base_shape.get("ovl", ovl) or ovl)

    scale = float(_clamp(delta_clamp_scale if delta_clamp_scale is not None else 0.75, 0.20, 1.00))
    off = base_offset + ((float(offset) - base_offset) * scale)
    cons = base_cons + ((float(consonant) - base_cons) * scale)
    cut = base_cutoff + ((float(cutoff) - base_cutoff) * scale)
    p = base_pre + ((float(pre) - base_pre) * scale)
    o = base_ovl + ((float(ovl) - base_ovl) * scale)

    if _is_high_pitch_context(pitch_zone=pitch_zone, note_hint_hz=0.0, f0_max_hz=0.0):
        off_cap = float(_env_float("UTOA_HIGH_PITCH_OFFSET_CLAMP_MS", 12.0))
        cut_cap = float(_env_float("UTOA_HIGH_PITCH_CUTOFF_CLAMP_MS", 10.0))
        off = base_offset + float(_clamp(off - base_offset, -off_cap, off_cap))
        cut = base_cutoff + float(_clamp(cut - base_cutoff, -cut_cap, cut_cap))

    return validate_oto_params(off, cons, cut, p, o)


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
    kr_mapping_max_index_jump_high_conf=1,
    cleanup_timing_jsonl=True,
    auto_format=None,
    callback=None,
    ml_policy="",
    runtime_report=None,
):
    """TextGrid・ 奛懦伯・ｿ/・尖徐 尞ｬ・ｷ ・簿ｳｴ・ｼ ・ｬ・ｩ﨑ｴ ・懍｢・OTO・ｼ ・晧┳﨑ｩ・壱共."""

    try:
        import textgrid
    except ImportError:
        err = "textgrid ・ｨ・溢擽 ・､・俯据・ｴ ・溢ｧ ・喜慣・壱共. `pip install textgrid`・ｼ ・､嵂駕紛 ・ｼ・ｸ・・"
        logger.error(err)
        if callback:
            callback(err)
        return 0, 0, [err]

    use_context_default_params = params is None
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
            f"笞 ・尖徐 ・川攵・ｬ・ｴ・､ ・晧┳・ 嶸・椪 CV/・ｰ・ｨ・・ CVC, CVVC, VCV・・・・戦鮒・壱共. "
            f"{auto_gen_format.upper()} -> CVVC・・・・劍﨑ｩ・壱共."
        )
        if callback:
            callback(msg)
        auto_gen_format = "cvvc"

    if use_context_default_params:
        params = get_default_params_for_context("korean", auto_gen_format)
        use_sinsy_labels = bool(params.get("USE_SINSY_LABELS", False)) if params else False
        sinsy_label_path = str(params.get("SINSY_LABEL_PATH", "") or "").strip() if params else ""

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
    kr_disable_cvvc_order_lock = _env_bool("UTOA_KR_DISABLE_CVVC_ORDER_LOCK", False)
    kr_mapping_only_enable = _env_bool("UTOA_KR_MAPPING_ONLY_ENABLE", False)

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
    row_apply_decisions = []
    row_apply_mode_counts = {
        "full_apply": 0,
        "conservative_apply": 0,
        "template_preserve": 0,
        "review_required": 0,
    }
    review_queue_export = _env_bool("UTOA_REVIEW_QUEUE_EXPORT", True)
    review_queue_path = os.path.join(
        _anchor_log_dir,
        f"review_queue_kr_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
    )

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
        log(f"笞 奛懦伯・ｿ 甯護攵・・・ｾ・・・・・・慣・壱共: {tpl_path}")
        log(f"笞｡ OpenUtau 嶸ｸ嶹・{auto_gen_format.upper()} ・尖徐 ・川攵・ｬ・ｴ・､ ・晧┳・ｼ・・・・劍﨑ｩ・壱共.")
        tpl_path = ""

    DIPHTHONG_CV_CONSONANT_RATIO = params.get('DIPHTHONG_CV_CONSONANT_RATIO', 0.6) if params else 0.6


    template_lines = []
    if tpl_path:
        lines, detected_enc, warning, err = load_template_oto_lines(
            tpl_path,
            require_utf8=True,
            mode_label="﨑懋ｵｭ・ｴ OTO",
        )
        if err:
            log(err)
            log(f"笞｡ 奛懦伯・ｿ ・罹糖 ・､甯ｨ・・OpenUtau 嶸ｸ嶹・{auto_gen_format.upper()} ・尖徐 ・川攵・ｬ・ｴ・､ ・晧┳・ｼ・・・・劍﨑ｩ・壱共.")
            lines = []
        if warning:
            log(warning)
        template_lines = lines or []


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
            log(f"甯護攵・・・､﨑・・ｩ・・ {wav_name} (・母ｷ懦剩 墲､ {norm_name}, 弡・ｳｴ {len(candidates)}・・ -> ・尖ｳｸ 甯護攵・・揆 ・・﨑ｩ・壱共.")
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
                f"[WARN] 템플릿-TextGrid 매칭률 낮음 ({t_match}/{t_total}, {t_ratio:.1%}) "
                f"-> OpenUtau {auto_gen_format.upper()} 자동 생성으로 전환"
            )
            use_template = False
        else:
            log(f"[INFO] 템플릿 매칭 OTO 사용 ({t_match}/{t_total}, {t_ratio:.1%})")
    if _should_keep_template_alias_set_exact(
        use_template=use_template,
        generate_openutau=generate_openutau,
        gen_missing_vowels=gen_missing_vowels,
    ):
        log("[INFO] 템플릿 alias 유지 모드: 누락 alias를 추가해도 기존 alias는 보존합니다.")

    preloaded_tg_by_path = {}
    if use_template:
        file_groups = {}
        for line in template_lines:
            fname = line.split('=', 1)[0]
            if fname not in file_groups:
                file_groups[fname] = []
            file_groups[fname].append(line)
    else:
        log(f"[INFO] 템플릿 미사용/불가 -> OpenUtau {auto_gen_format.upper()} 자동 생성으로 진행합니다.")
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
            textgrid_cache_by_path=preloaded_tg_by_path,
        )

    processed = 0
    total = len(file_groups)
    mel_cache_for_signal = {}
    single_vowel_span_by_tg_path = {}

    def _norm_tg_path_key(path):
        try:
            return os.path.normcase(os.path.abspath(str(path or "")))
        except Exception:
            return str(path or "")

    for fname, lines in file_groups.items():
        file_ctx = prepare_file_context_with_sinsy(
            fname=fname,
            lines=lines,
            prepare_file_context_fn=prepare_file_context,
            prepare_kwargs={
                "resolve_tg_info_fn": _resolve_tg_info,
                "wav_root_for_signal": wav_root_for_signal,
                "wav_index_for_signal": wav_index_for_signal,
                "mel_cache_for_signal": mel_cache_for_signal,
                "find_wav_path_fn": _find_wav_path_for_name,
                "read_wav_fn": _read_wav_mono_np,
                "mel_envelope_fn": _mel_envelope,
                "wav_duration_fn": _wav_duration_ms,
            },
            use_sinsy_labels=bool(use_sinsy_labels),
            load_sinsy_label_entries_fn=load_sinsy_label_entries,
            sinsy_label_path=sinsy_label_path,
        )
        output_wav_name = str(getattr(file_ctx, "output_wav_name", "") or "")
        if output_wav_name:
            wav_name_map.setdefault(fname, output_wav_name)
            if file_ctx.real_wav_name and file_ctx.real_wav_name != fname:
                wav_name_map.setdefault(file_ctx.real_wav_name, output_wav_name)
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

        file_ctx = load_named_tiers_for_generation(
            file_ctx=file_ctx,
            load_named_tiers_fn=load_named_tiers,
            load_textgrid_fn=textgrid.TextGrid.fromFile,
            preloaded_tg_by_path=preloaded_tg_by_path if not use_template else None,
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
                log(f"[WARN] {fname}: phones tier가 없어 원본 행을 유지합니다.")
                _record_unset_lines("tier_missing", fname, lines)
                final_lines.extend([
                    apply_suffix_to_oto_line(l, alias_suffix)
                    for l in lines
                ])
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
            ingest_state = extract_kr_alignment_ingest_state(alignment_ingest=alignment_ingest)
            ph_intervals_all = ingest_state["ph_intervals_all"]
            ph_intervals = ingest_state["ph_intervals"]
            wd_intervals = ingest_state["wd_intervals"]
            update_single_vowel_span_by_first_phone(
                phone_intervals=ph_intervals,
                tg_path=file_ctx.tg_path,
                single_vowel_span_by_tg_path=single_vowel_span_by_tg_path,
                norm_tg_path_key_fn=_norm_tg_path_key,
            )
            wav_duration_ms = float(ingest_state["wav_duration_ms"])
            timeline_start_ms = float(ingest_state["timeline_start_ms"])
            timeline_end_ms = float(ingest_state["timeline_end_ms"])
            file_format = str(ingest_state["file_format"])
            file_mapping_conf_th = float(ingest_state["file_mapping_conf_th"])
            filename_cv_targets = list(ingest_state["filename_cv_targets"])
            targets_for_build = list(ingest_state["targets_for_build"])
            sinsy_label_entries = list(ingest_state["sinsy_label_entries"])
            phone_quality = ingest_state["phone_quality"]
            low_quality_reasons = ingest_state["low_quality_reasons"]
            low_phone_quality = ingest_state["low_phone_quality"]
            force_words_phone_fill = bool(ingest_state["force_words_phone_fill"])
            textgrid_trust_score = float(ingest_state["textgrid_trust_score"])
            textgrid_trust_tier = str(ingest_state["textgrid_trust_tier"])
            prefer_filename_sequence = bool(ingest_state["prefer_filename_sequence"])
            spn_ratio = float(ingest_state["spn_ratio"])
            alignment_weight = float(ingest_state["alignment_weight"])

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
            syllables_info = []
            used_words_based = False
            used_alias_based = False
            base_score = 0.0
            alt_score = 0.0
            mapping_reason_code = "filename_token"
            if wd_intervals:
                words_phone_lookup = build_interval_lookup(ph_intervals)
                for w in wd_intervals:
                    w_start = w.minTime
                    w_end = w.maxTime

                    s_phones = intervals_within_bounds(
                        words_phone_lookup,
                        float(w_start) - 0.01,
                        float(w_end) + 0.01,
                    )
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
            if mel_ctx_for_file:
                _annotate_kr_syllable_blank_confidence(syllables_info, mel_ctx_for_file)
                _annotate_kr_syllable_blank_confidence(alias_based, mel_ctx_for_file)
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
            if mel_ctx_for_file:
                _annotate_kr_syllable_blank_confidence(syllables_info, mel_ctx_for_file)
            syllable_blank_confidences = [
                float((s or {}).get("blank_confidence", 0.0) or 0.0)
                for s in (syllables_info or [])
            ]
            blank_conf_mean = 0.0
            if syllable_blank_confidences:
                blank_conf_mean = float(sum(syllable_blank_confidences) / float(len(syllable_blank_confidences)))
            alignment_weight_base = max(
                0.0,
                min(
                    1.0,
                    (textgrid_trust_score * 0.85) - max(0.0, blank_conf_mean - 0.45) * 0.35,
                ),
            )
            alignment_weight = float(alignment_weight_base)
            mel_reliability_score = _estimate_file_mel_reliability(
                mel_ctx_for_file,
                blank_conf_mean=blank_conf_mean,
            )
            mel_reliability_floor = _resolve_mel_reliability_floor(
                mel_ctx_for_file,
                default_floor=_env_float("UTOA_MEL_FORCE_MEL_RELIABILITY_MIN", 0.44),
            )
            mel_weight_mode = _resolve_mel_weight_mode(os.environ.get("UTOA_MEL_WEIGHT_MODE", "auto"))
            force_mel_branch = False
            anchor_lock_lite = False
            used_words_based = bool(kr_source_pick.get("used_words_based"))
            used_alias_based = bool(kr_source_pick.get("used_alias_based"))
            mapping_reason_code = str(kr_source_pick.get("mapping_reason_code") or mapping_reason_code)
            base_score = float(kr_source_pick.get("base_score", 0.0) or 0.0)
            alt_score = float(kr_source_pick.get("alt_score", 0.0) or 0.0)
            words_glide_mismatch_ratio = float(kr_source_pick.get("words_glide_mismatch_ratio", 0.0) or 0.0)

            if mapping_reason_code == "filename_sequence_lock":
                log(
                    f"[MAP] {fname}: TextGrid 신뢰도 {textgrid_trust_tier.upper()} "
                    f"(trust={textgrid_trust_score:.2f}) sequence lock kept"
                )
            elif mapping_reason_code == "alias_based_empty_words":
                log(f"[MAP] {fname}: words 기반 phone 매핑 실패 -> alias/phone 기반으로 대체")
            elif mapping_reason_code == "alias_phone_minimal":
                log(f"[MAP] {fname}: words 정보가 부족해 phones 기반으로 보강 매핑")
            elif mapping_reason_code in {"order_locked_length_mismatch", "order_locked_glide_mismatch", "order_locked_low_phone_quality"}:
                log(
                    f"[MAP] {fname}: CV 순서 잠금 유지 "
                    f"(reason={mapping_reason_code}, words={base_score:.1f}, alias={alt_score:.1f}, "
                    f"glide_mismatch={words_glide_mismatch_ratio:.2f})"
                )
            elif mapping_reason_code == "words_low_phone_quality":
                log(
                    f"[MAP] {fname}: phones 신뢰도 낮음({','.join(low_quality_reasons)}) -> words 보강 매핑 사용 "
                    f"(words={base_score:.1f}, alias={alt_score:.1f})"
                )
            elif mapping_reason_code == "alias_based_cvvc":
                log(
                    f"[MAP] {fname}: CVVC에서 alias 기반 매핑 우선 "
                    f"(words={base_score:.1f}, alias={alt_score:.1f})"
                )
            elif mapping_reason_code == "alias_based_recover":
                log(
                    f"[MAP] {fname}: 매핑 점수 보정 적용 "
                    f"(base={base_score:.1f}, corrected={alt_score:.1f})"
                )
            elif mapping_reason_code == "words_keep_high_conf":
                log(
                    f"[MAP] {fname}: words 매핑 신뢰도 높음 -> words 결과 유지 "
                    f"(base={base_score:.1f}, corrected={alt_score:.1f})"
                )
            elif alias_based and targets_for_build:
                log(
                    f"[MAP] {fname}: TextGrid(words) 기반 재매핑 "
                    f"(base={base_score:.1f}, corrected={alt_score:.1f})"
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
            def _should_enable_kr_mel_plan() -> bool:
                fmt = str(file_format or "").strip().lower()
                mel_plan_env = {
                    "cvvc": "UTOA_KR_CVVC_MEL_PLAN",
                    "cvc": "UTOA_KR_CVC_MEL_PLAN",
                    "vcv": "UTOA_KR_VCV_MEL_PLAN",
                    "cv": "UTOA_KR_CV_MEL_PLAN",
                }
                if fmt in mel_plan_env and mel_ctx_for_file and syllables_info:
                    if (
                        textgrid_trust_tier != "high"
                        or low_phone_quality
                        or blank_conf_mean >= 0.45
                        or alignment_weight < 0.65
                    ):
                        return bool(_env_bool(mel_plan_env[fmt], True))
                return False

            kr_plan_context_kwargs = {
                "planned_cv_source": plan_candidate_source,
                "syllables_info": syllables_info,
                "sinsy_label_entries": sinsy_label_entries,
                "format_type": file_format,
                "alignment_weight": alignment_weight,
                "prefer_sequence": prefer_filename_sequence,
                "normalize_expected_fn": _normalize_cv_match_token,
                "normalize_label_fn": _normalize_cv_match_token,
                "label_match_score_fn": (lambda target, label: float(_cv_match_score(target, label))),
                "should_enable_mel_plan_fn": _should_enable_kr_mel_plan,
                "build_cv_anchor_plan_fn": _build_kr_cv_anchor_plan_v2,
                "build_sinsy_guided_anchor_plan_fn": build_sinsy_guided_anchor_plan,
                "build_adjacent_anchor_graph_fn": build_adjacent_anchor_graph,
                "resolve_plan_policy_fn": resolve_plan_policy,
            }

            mapping_confidence_base, mapping_margin = _estimate_kr_mapping_confidence(
                phone_quality,
                words_score=base_score,
                alias_score=alt_score,
                used_words_based=used_words_based,
                used_alias_based=used_alias_based,
            )
            kr_runtime_state = recompute_common_plan_runtime_state(
                build_plan_context_fn=build_common_plan_context,
                plan_context_kwargs=kr_plan_context_kwargs,
                ingest_snapshot=alignment_ingest,
                mapping_confidence=mapping_confidence_base,
                mapping_margin=mapping_margin,
                conf_threshold=file_mapping_conf_th,
                format_type=file_format,
                score_a=base_score,
                score_b=alt_score,
                sequence_lock_formats={"cvvc", "cvc"},
                abstain_formats={"cvvc", "vcv", "cvc", "cv"},
                strict_formats={"cvvc"},
                prefer_sequence=prefer_filename_sequence,
                alignment_trust=alignment_weight,
                resolve_runtime_mapping_policy_fn=resolve_runtime_mapping_policy,
            )
            kr_cv_plan = dict(kr_runtime_state.get("cv_plan") or {})
            kr_planned_cv_indices = kr_runtime_state.get("planned_indices")
            kr_anchor_graph = kr_runtime_state.get("anchor_graph")
            kr_plan_policy = dict(kr_runtime_state.get("plan_policy") or {})
            runtime_policy = dict(kr_runtime_state.get("runtime_policy") or {})
            mapping_confidence_base = float(kr_runtime_state.get("mapping_confidence", mapping_confidence_base) or 0.0)
            mapping_margin = float(kr_runtime_state.get("mapping_margin", mapping_margin) or 0.0)
            log_sinsy_plan_guard(
                sinsy_label_entries=sinsy_label_entries,
                cv_plan=kr_cv_plan,
                runtime_policy=runtime_policy,
                fname=fname,
                log_fn=log,
                log_prefix="[MAP]",
            )
            mapping_confidence_base = float(runtime_policy.get("mapping_confidence", mapping_confidence_base))
            if syllable_blank_confidences:
                if blank_conf_mean >= 0.55:
                    mapping_confidence_base = max(
                        0.0,
                        mapping_confidence_base - min(0.22, (blank_conf_mean - 0.55) * 0.45 + 0.04),
                    )
            if kr_mapping_debug_reason_logging and mapping_confidence_base < float(file_mapping_conf_th):
                log(
                    f"[MAP] {fname}: KR 매핑 신뢰도 낮음(conf={mapping_confidence_base:.2f}, "
                    f"margin={mapping_margin:+.1f}, reason={mapping_reason_code})"
                )

            file_conf_floor = float(runtime_policy.get("file_conf_floor", file_mapping_conf_th))
            low_conf_state = compute_runtime_low_conf_state(
                runtime_policy=runtime_policy,
                mapping_confidence=mapping_confidence_base,
                conf_floor=file_conf_floor,
                blank_confidence_mean=blank_conf_mean,
                conf_below_reason="conf_below_floor",
                row_conf_floor_default=float(file_mapping_conf_th),
            )
            file_mapping_low_conf = bool(low_conf_state.get("file_low_conf"))
            row_conf_floor = float(low_conf_state.get("row_conf_floor", file_mapping_conf_th) or file_mapping_conf_th)
            row_margin_floor = float(low_conf_state.get("row_margin_floor", 6.0) or 6.0)
            low_conf_reasons = list(low_conf_state.get("low_conf_reasons") or [])
            pitch_zone_for_file = str(mel_ctx_for_file.get("f0_pitch_zone", "") if mel_ctx_for_file else "").strip().lower()
            note_hint_hz_for_file = float(mel_ctx_for_file.get("f0_note_hint_hz", 0.0) or 0.0) if mel_ctx_for_file else 0.0
            f0_max_hz_for_file = float(mel_ctx_for_file.get("f0_max_hz", 0.0) or 0.0) if mel_ctx_for_file else 0.0
            routing_profile = resolve_file_routing_profile(
                file_format=str(file_format or ""),
                pitch_zone=pitch_zone_for_file,
                note_hint_hz=note_hint_hz_for_file,
                f0_max_hz=f0_max_hz_for_file,
                textgrid_trust_score=float(textgrid_trust_score),
                textgrid_trust_tier=str(textgrid_trust_tier or ""),
                mapping_confidence_base=float(mapping_confidence_base),
                blank_conf_mean=float(blank_conf_mean),
                mel_reliability_score=float(mel_reliability_score),
                mel_reliability_floor=float(mel_reliability_floor),
                file_mapping_low_conf=bool(file_mapping_low_conf),
                alignment_weight_base=float(alignment_weight_base),
                mel_weight_mode=str(mel_weight_mode or ""),
                file_mapping_conf_th=float(file_mapping_conf_th),
            )
            force_mel_branch = bool(routing_profile.get("force_mel_branch"))
            mel_weight_mode = str(routing_profile.get("mel_weight_mode") or mel_weight_mode)
            alignment_weight = float(routing_profile.get("alignment_weight_final", alignment_weight_base))
            anchor_lock_lite = bool(routing_profile.get("anchor_lock_lite_default", False))
            row_blank_floor = routing_profile.get("row_blank_floor")
            delta_clamp_scale = float(routing_profile.get("delta_clamp_scale", 1.0))
            routing_profile_code = str(routing_profile.get("profile_code") or "hybrid_soft")
            log(
                f"[ROUTING] file={fname} profile={routing_profile_code} "
                f"trust={textgrid_trust_score:.2f} blank={blank_conf_mean:.2f} "
                f"mel_rel={mel_reliability_score:.2f} align={alignment_weight:.2f}"
            )
            log(
                f"[MAP] {fname}: align_base={alignment_weight_base:.2f}, align_final={alignment_weight:.2f}, "
                f"blank_mean={blank_conf_mean:.2f}, map_conf={mapping_confidence_base:.2f}, "
                f"mel_rel={mel_reliability_score:.2f}/{mel_reliability_floor:.2f}, "
                f"anchor_lock_lite={anchor_lock_lite}, "
                f"mel_weight_mode={mel_weight_mode}, force_mel_branch={force_mel_branch}, "
                f"delta_clamp_scale={delta_clamp_scale:.2f}"
            )
            if isinstance(runtime_report, dict):
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
                    row_blank_floor=(
                        float(row_blank_floor)
                        if row_blank_floor is not None
                        else None
                    ),
                    extra_fields={
                        "routing_profile": str(routing_profile_code),
                        "alignment_weight_base": float(alignment_weight_base),
                        "alignment_weight": float(alignment_weight),
                        "mel_reliability_score": float(mel_reliability_score),
                        "mel_reliability_floor": float(mel_reliability_floor),
                        "mel_weight_mode": str(mel_weight_mode),
                        "force_mel_branch": bool(force_mel_branch),
                        "anchor_lock_lite": bool(anchor_lock_lite),
                        "delta_clamp_scale": float(delta_clamp_scale),
                        "high_pitch_mode": bool(routing_profile.get("high_pitch_mode", False)),
                    },
                )

            if (not syllables_info) or any(len(s['phones']) == 0 for s in syllables_info):
                log(f"[WARN] {fname}: 음절 경계 해석 실패로 원본 행을 유지합니다.")
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
                final_lines.extend([
                    apply_suffix_to_oto_line(l, alias_suffix)
                    for l in lines
                ])
                processed += 1
                continue

            if bool(runtime_policy.get("should_abstain")):
                log(
                    f"[WARN] {fname}: KR v2 planner abstain "
                    f"(trust={textgrid_trust_score:.2f}, weight={alignment_weight:.2f}, "
                    f"coverage={float(kr_plan_policy.get('coverage', 0.0)):.2f}, "
                    f"margin={float(kr_plan_policy.get('margin', 0.0)):.1f}) -> 원본행 유지"
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
                final_lines.extend([
                    apply_suffix_to_oto_line(l, alias_suffix)
                    for l in lines
                ])
                processed += 1
                continue

            romaji_syllables = [s.get('roman_cv') or s.get('roman', '') for s in syllables_info]
            current_w_idx = 0
            cv_seq_idx = 0
            bridge_seq_idx = 0
            kr_order_locked_format = _is_kr_order_locked_cv_format(file_format)
            if kr_order_locked_format and kr_disable_cvvc_order_lock:
                kr_order_locked_format = False
                if kr_mapping_debug_reason_logging:
                    log(f"[MAP] {fname}: KR CVVC/CVC filename order lock 비활성(UTOA_KR_DISABLE_CVVC_ORDER_LOCK=1)")
            kr_cvvc_occurrence_source = filename_cv_targets if (kr_order_locked_format and filename_cv_targets) else syllables_info
            kr_cvvc_occurrence_map = _build_kr_cvvc_occurrence_map(kr_cvvc_occurrence_source) if kr_order_locked_format else None
            kr_cvvc_occurrence_state = {}
            kr_cvvc_vv_occurrence_map = _build_kr_cvvc_vv_occurrence_map(kr_cvvc_occurrence_source) if file_format == "cvvc" else None
            kr_cvvc_vv_occurrence_state = {}
            file_has_explicit_vc_alias = False
            file_has_cv_family_alias = False
            for _line in lines:
                _parts = _line.split("=", 1)
                if len(_parts) < 2:
                    continue
                _alias = _parts[1].split(",", 1)[0].strip()
                if not _alias:
                    continue
                _alias_type = _classify_alias_cached(_alias)
                if _alias_type == "vc":
                    file_has_explicit_vc_alias = True
                elif _alias_type in {"cv", "cv_head", "vcv", "mono"}:
                    file_has_cv_family_alias = True

            _format_norm = str(file_format or "").strip().lower()
            file_has_mixed_cv_vc = file_has_explicit_vc_alias and (
                file_has_cv_family_alias or _format_norm in {"cvvc", "cvc", "vcv"}
            )
            if file_has_mixed_cv_vc:
                kr_cv_timing_mode = "vcv_context" if _format_norm == "vcv" else "vc_context"
            else:
                kr_cv_timing_mode = "standalone"
            kr_cv_v_ref_baseline = _compute_kr_session_vowel_baseline_ms(syllables_info)
            for _syl in (syllables_info or []):
                _syl["v_ref_baseline_ms"] = float(kr_cv_v_ref_baseline)
            if kr_mapping_debug_reason_logging:
                log(
                    f"[MAP] {fname}: KR CV timing mode={kr_cv_timing_mode} "
                    f"(format={_format_norm or 'unknown'}, "
                    f"vc_alias={'yes' if file_has_explicit_vc_alias else 'no'}, "
                    f"cv_family={'yes' if file_has_cv_family_alias else 'no'}, "
                    f"v_ref={kr_cv_v_ref_baseline:.1f}ms)"
                )
            cv_anchor_by_idx = {
                i: _estimate_cv_anchor_from_syllable(
                    syllables_info[i],
                    ph_intervals,
                    cv_mode=kr_cv_timing_mode,
                )
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
                row_apply_mode = "full_apply"
                row_apply_reason_code = ""


                alias_type = _classify_alias_cached(alias)
                row_jump_default = int(max(0, kr_mapping_max_index_jump_default))
                row_jump_high_conf = int(max(row_jump_default, kr_mapping_max_index_jump_high_conf))
                if kr_mapping_only_enable and alias_type in {"cv", "cv_head", "vcv", "vv", "mono"}:
                    row_jump_default = 0
                    row_jump_high_conf = 0
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
                    continue


                if is_cv_head:
                    forced_cvvc_idx = _resolve_kr_cv_head_forced_index_v2(
                        alias=alias,
                        alias_type=alias_type,
                        file_format=file_format,
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
                        expected_idx=cv_seq_idx,
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
                                f"[MAP] {fname}: KR CV 앵커 계획 보정 "
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
                    row_blank_conf = _blank_conf_at(syllable_blank_confidences, selected_w_idx)
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
                        blank_confidence=row_blank_conf,
                        max_blank_confidence=row_blank_floor_safe,
                        # CVC・・甯護攵/・護・孖ｹ・ｱ・・margin ・・呷擽 ・､ CV ・・龍・ｴ ・ｼ・・・､墲ｵ・ ・・・溢牟
                        # row-level abstain ・護擽孖ｸ・ｼ CV/CVVC・尖ｧ・・・圸﨑罹共.
                        active_only_formats={"cvvc", "cv"},
                        margin_formats={"cvvc", "cv"},
                        blank_formats={"cvvc", "cvc", "cv"},
                        min_confidence_margin_by_alias_type={"cv_head": row_margin_floor + 1.5, "vcv": row_margin_floor + 1.0},
                        min_row_confidence_by_alias_type={"cv_head": row_conf_floor + 0.03, "vcv": row_conf_floor + 0.02},
                        max_blank_confidence_by_alias_type={
                            "cv_head": max(0.0, row_blank_floor_safe - 0.03),
                            "vcv": max(0.0, row_blank_floor_safe - 0.02),
                        },
                    )
                    row_apply = decide_row_application(
                        routing_profile=routing_profile_code,
                        pitch_zone=pitch_zone_for_file,
                        row_mapping_confidence=row_mapping_confidence,
                        row_conf_floor=row_conf_floor,
                        row_blank_confidence=row_blank_conf,
                        row_blank_floor=row_blank_floor,
                        row_jump_blocked=row_jump_blocked,
                        forced_selected=forced_selected_idx is not None,
                        file_mapping_low_conf=file_mapping_low_conf,
                        row_abstain_skip=bool(row_abstain.get("should_skip")),
                        row_abstain_reason=str(row_abstain.get("reason") or ""),
                    )
                    row_apply_mode = str(row_apply.get("mode") or "full_apply")
                    row_apply_reason_code = str(row_apply.get("reason_code") or "")
                    if row_apply_mode in row_apply_mode_counts:
                        row_apply_mode_counts[row_apply_mode] = int(row_apply_mode_counts[row_apply_mode]) + 1
                    else:
                        row_apply_mode_counts[row_apply_mode] = 1
                    row_apply_decisions.append(
                        {
                            "file": str(fname),
                            "alias": str(alias),
                            "alias_type": str(alias_type),
                            "format_type": str(file_format or ""),
                            "selected_w_idx": int(selected_w_idx) if selected_w_idx is not None else -1,
                            "mapping_confidence": float(row_mapping_confidence),
                            "blank_confidence": float(row_blank_conf) if row_blank_conf is not None else None,
                            "row_blank_floor": float(row_blank_floor_safe),
                            "apply_mode": str(row_apply_mode),
                            "reason_code": str(row_apply_reason_code),
                            "reasons": list(row_apply.get("reasons") or []),
                            "routing_profile": str(routing_profile_code),
                            "pitch_zone": str(pitch_zone_for_file),
                        }
                    )
                    if row_apply_mode == "review_required":
                        if kr_mapping_debug_reason_logging:
                            log(
                                f"[WARN] {fname}: KR 행 보정 스킵 "
                                f"({row_apply_reason_code or row_abstain.get('reason')}, {alias})"
                            )
                        _record_unset(
                            str(row_apply_reason_code or row_abstain.get("reason") or "review_required"),
                            fname,
                            line,
                            meta={
                                "diag_hint": str(row_abstain.get("diag_hint", "") or ""),
                                "apply_mode": str(row_apply_mode),
                                "routing_profile": str(routing_profile_code),
                            },
                        )
                        if use_template:
                            # 奛懦伯・ｿ ・ｨ・懍乱・罹株 ・､﨑卓擽 ・逸剳・､﨑・嵂餓擽・ｼ・・・ｰ・ｴ alias・ｼ ・ｴ・ｴ﨑罹共.
                            final_lines.append(
                                apply_suffix_to_oto_line(line, alias_suffix)
                            )
                        continue
                    if row_apply_mode == "template_preserve" and use_template:
                        final_lines.append(apply_suffix_to_oto_line(line, alias_suffix))
                        continue
                    current_w_idx = max(current_w_idx, selected_w_idx)
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
                    onset_slice_hint = _select_kr_cv_onset_slice(curr_phones) if curr_phones else None
                    onset_idx_hint = int(onset_slice_hint[0]) if onset_slice_hint is not None else 0
                    glide_dur_hint_ms = float(onset_slice_hint[5]) if onset_slice_hint is not None and len(onset_slice_hint) >= 6 else 0.0
                    if onset_idx_hint < 0 or onset_idx_hint >= len(curr_phones or []):
                        onset_idx_hint = 0
                    c_hint = curr_phones[onset_idx_hint].mark if curr_phones else ""

                    if is_diph:
                        alias_onset = _extract_alias_onset(alias)
                        offset, consonant, cutoff, pre, ovl = _compute_kr_cv_timing(
                            c_start,
                            c_end,
                            cv_vowel_len,
                            c_hint,
                            alias_onset,
                            True,
                            False,
                            glide_dur_ms=glide_dur_hint_ms,
                            v_ref_baseline=kr_cv_v_ref_baseline,
                            cv_mode=kr_cv_timing_mode,
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
                        first_phone_plosive = bool(curr_phones) and is_plosive_ipa(c_hint)
                        alias_consonant = re.match(r'^([^aeiouyw]+)', alias.lower())
                        roman_plosive = alias_consonant and is_plosive_roman(alias_consonant.group(1)) if alias_consonant else False
                        alias_onset = alias_consonant.group(1) if alias_consonant else ""
                        is_plosive = first_phone_plosive or roman_plosive
                        offset, consonant, cutoff, pre, ovl = _compute_kr_cv_timing(
                            c_start,
                            c_end,
                            cv_vowel_len,
                            c_hint,
                            alias_onset,
                            False,
                            is_plosive,
                            v_ref_baseline=kr_cv_v_ref_baseline,
                            cv_mode=kr_cv_timing_mode,
                        )

                row_mel_voiced_onset_ms = None
                if mel_ctx_for_file and alias_type == "cv":
                    pre_abs = float(offset) + float(pre)
                    mel_weight = _resolve_mel_onset_weight(
                        alignment_weight,
                        textgrid_trust_tier,
                        trust_score=textgrid_trust_score,
                        mapping_confidence=row_mapping_confidence,
                        mel_reliability=mel_reliability_score,
                        mel_reliability_floor=mel_reliability_floor,
                    )
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
                if alias_type == "cv" and selected_w_idx is not None and 0 <= int(selected_w_idx) < len(syllables_info):
                    row_syl = syllables_info[int(selected_w_idx)] or {}
                    onset_dist = float(row_syl.get("mel_onset_distance_ms", 999.0) or 999.0)
                    onset_energy = float(row_syl.get("mel_onset_energy", 0.0) or 0.0)
                    if onset_dist < 40.0 and onset_energy > 0.20:
                        ovl_cap = max(0.0, float(pre) - onset_dist - 8.0)
                        ovl = max(0.0, min(float(ovl), ovl_cap))

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
                            f"[MAP] {fname}: CV 포커스 가드 적용 "
                            f"(offset -{cv_offset_pulled:.1f}ms, cutoff -{cv_cutoff_trimmed:.1f}ms) [{alias}]"
                        )
                if row_apply_mode != "template_preserve":
                    before_nat = (offset, pre, consonant, cutoff)
                    (
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        nat_info,
                    ) = _apply_kr_mel_naturalness_adjustments(
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        alias_type=alias_type,
                        c_start_ms=c_start,
                        c_end_ms=c_end,
                        vowel_start_ms=n_start,
                        vowel_end_ms=n_end,
                        mel_ctx=mel_ctx_for_file,
                    )
                    if nat_info:
                        detail = ", ".join(
                            f"{k}={float(v):.1f}"
                            for k, v in nat_info.items()
                            if isinstance(v, (int, float))
                        )
                        _debug_offset_trace(
                            log,
                            "mel_naturalness",
                            fname,
                            alias,
                            before_nat,
                            (offset, pre, consonant, cutoff),
                            extra=detail,
                        )
                if row_apply_mode in {"conservative_apply", "template_preserve"}:
                    offset, consonant, cutoff, pre, ovl = _apply_conservative_delta_clamp(
                        mode=row_apply_mode,
                        delta_clamp_scale=delta_clamp_scale,
                        pitch_zone=pitch_zone_for_file,
                        base_shape=base_shape,
                        offset=offset,
                        consonant=consonant,
                        cutoff=cutoff,
                        pre=pre,
                        ovl=ovl,
                    )
                    if kr_mapping_debug_reason_logging:
                        log(
                            f"[ROW] {fname}: mode={row_apply_mode} reason={row_apply_reason_code or '-'} "
                            f"delta_scale={delta_clamp_scale:.2f} alias={alias}"
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
            err_msg = f"파일 처리 실패 ({fname}): {e}{loc}"
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
        log("누락된 단모음 alias 자동 생성을 시작합니다...")
        vowels_list = ['a', 'e', 'i', 'o', 'u', 'eo', 'eu', 'ae', 'oe', 'wi', 'wa', 'we', 'weo', 'ya', 'ye', 'yo', 'yeo', 'yu', 'ui', 'eui']
        template_aliases = set()
        for g_lines in file_groups.values():
            for line in g_lines:
                parts = line.split('=', 1)
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
                v_span = single_vowel_span_by_tg_path.get(_norm_tg_path_key(tg_info.get("path")))
                if v_span is None:
                    try:
                        tg = textgrid.TextGrid.fromFile(tg_info['path'])
                        phone_tier = next((t for t in tg if isinstance(t, textgrid.IntervalTier) and t.name == 'phones'), None)
                        if not phone_tier:
                            continue
                        intervals = [i for i in phone_tier if i.mark.strip() not in ['', 'sil', 'spn', 'pau']]
                        if len(intervals) != 1:
                            continue
                        vowel = intervals[0]
                        v_span = (float(vowel.minTime) * 1000.0, float(vowel.maxTime) * 1000.0)
                    except Exception:
                        continue
                v_start, v_end = v_span
                alias = detected_vowel
                if alias not in template_aliases:
                    log(f"추가: 단모음 alias 생성 -> {tg_info['real_name']} [{alias}]")
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
        log(f"1차 생성 완료: OTO 저장 -> {out_path}")
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
                ml_route=os.environ.get("UTOA_ML_ROUTE", "legacy"),
            )
        )
        renamed = apply_output_wav_name_map(out_path, wav_name_map)
        if renamed:
            log(f"WAV 이름 매핑 적용: {renamed}개")
    except Exception as e:
        err = f"OTO 저장 실패: {e}"
        logger.error(err)
        errors.append(err)

    if review_queue_export:
        try:
            review_rows = [
                row
                for row in row_apply_decisions
                if str(row.get("apply_mode", "")).strip().lower() in {
                    "conservative_apply",
                    "template_preserve",
                    "review_required",
                }
            ]
            if review_rows:
                written = write_jsonl_records(review_queue_path, review_rows)
                log(f"[REVIEW] review queue exported: {written} rows -> {review_queue_path}")
        except Exception as e:
            log(f"[REVIEW] review queue export failed: {e}")

    if isinstance(runtime_report, dict):
        runtime_report["row_apply_mode_counts"] = {
            str(k): int(v)
            for k, v in dict(row_apply_mode_counts).items()
        }
        if review_queue_export and row_apply_decisions:
            runtime_report["review_queue_path"] = str(review_queue_path)

    finalize_generator_finish(finish_context)

    _log_unset_summary()
    return processed, total, errors



