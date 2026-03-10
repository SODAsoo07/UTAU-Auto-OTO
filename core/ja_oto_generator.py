"""
日本語 CVVC OTO.ini 自動生成モジュール
- TextGrid 基準の OTO パラメータ計算
- CV, VC, VV, V タイプの自動分類と処理
- リスト順優先マッピング (List-Order-First) アルゴリズム
"""

import os
import re
import math
import logging
import textgrid
from types import SimpleNamespace
from core.lab_generator import load_custom_phonemes
from core.ja_lab_generator import (
    parse_ja_filename,
    split_ja_romaji_syllable,
    KANA_COMBO_ROMAJI,
    KANA_SINGLE_ROMAJI,
)
from core.ja_auto_alias_setup import build_ja_auto_file_groups
from core.ja_alias_family_stage import (
    build_ja_alias_family_state,
    try_handle_ja_br_alias,
    try_handle_ja_tail_breath_alias,
)
from core.ja_file_stage import build_ja_postprocess_context, try_handle_ja_single_vowel_file
from core.ja_loop_prep import prepare_ja_loop_state
from core.ja_oto_finalize import (
    _convert_ja_internal_cutoff_to_oto_field,
    _sanitize_ja_internal_params_for_wav_duration,
    sanitize_ja_oto_for_wav_duration,
)
from core.ja_generator_setup import (
    build_ja_trace_preparation,
    prepare_ja_generation_setup,
)
from core.ja_oto_file_finalize import JaPostFilePipelineContext, run_ja_post_file_pipeline
from core.ja_oto_autotune import (
    JaAutotuneRefreshContext,
    _apply_profile_to_oto_file,
    _profile_path_for_out,
    _train_ja_autotune_profile,
    apply_ja_autotune_delta as _apply_ja_autotune_profile_impl,
    load_ja_autotune_profile as _load_ja_autotune_profile,
    save_ja_autotune_profile as _save_ja_autotune_profile,
    refresh_ja_autotune_profile_after_generation,
)
from core.oto_file_utils import (
    build_occurrence_map as _build_occurrence_map_common,
    extract_base_timing_shape as _extract_base_timing_shape_common,
    parse_oto_line as _parse_oto_line_common,
    read_oto_rows_for_profile as _read_oto_rows_for_profile_common,
    read_text_with_fallback as _read_text_with_fallback_common,
)
from core.generator_finish import (
    GeneratorFinishContext,
    finalize_generator_finish,
    write_jsonl_records,
    write_oto_lines,
)
from core.alignment_ingest import build_ja_alignment_ingest
from core.oto_candidate_selection import maybe_promote_alias_candidate, select_primary_mapping_candidate
from core.oto_diagnostics import MappingTraceCollector, SkippedEntryCollector
from core.oto_diagnostics_adapter_v2 import GeneratorDiagnosticsAdapter
from core.file_prepare import load_named_tiers, prepare_file_context
from core.oto_normalization import canonicalize_alias_for_matching, normalize_wav_key
from core.timing_anchor_profiles import is_anchor_lock_enabled
from core.timing_anchor_runtime import append_timing_anchor_log
from core.anchor_lock_adapter_v2 import apply_language_anchor_lock
from core.textio_utils import load_template_oto_lines
from core.oto_profile_presets import get_ja_profile_preset
from core.ja_mapping_v2 import (
    build_ja_cv_anchor_plan as _build_ja_cv_anchor_plan_v2,
    collect_ja_syllable_activity_metrics as _collect_ja_syllable_activity_metrics_v2,
    is_ja_cv_syllable_active as _is_ja_cv_syllable_active_v2,
    resolve_ja_planned_cv_index as _resolve_ja_planned_cv_index_v2,
)
from core.sinsy_label_ingest import build_sinsy_guided_anchor_plan, load_sinsy_label_entries
from core.ja_mapping_scoring_v2 import (
    clamp_ja_cv_index_to_order as _clamp_ja_cv_index_to_order_v2,
    find_ja_cv_vowel_match_index as _find_ja_cv_vowel_match_index_v2,
    find_ja_exact_target_index as _find_ja_exact_target_index_v2,
    prefer_vcv_candidate_index as _prefer_vcv_candidate_index_v2,
    should_allow_ja_soft_forward_shift as _should_allow_ja_soft_forward_shift_v2,
)
from core.ja_timing_v2 import (
    build_realized_cv_anchor as _build_realized_cv_anchor_v2,
    compute_ja_vc_from_adjacent_cv as _compute_ja_vc_from_adjacent_cv_v2,
    compute_vcv_params_from_virtual_split as _compute_vcv_params_from_virtual_split_v2,
    enforce_cv_pre_anchor_guard as _enforce_cv_pre_anchor_guard_v2,
    enforce_vcv_cv_entry_guard as _enforce_vcv_cv_entry_guard_v2,
    estimate_ja_cv_anchor as _estimate_ja_cv_anchor_v2,
)
from core.oto_anchor_graph import build_adjacent_anchor_graph, resolve_bridge_anchor_pair
from core.oto_mapping_policy import resolve_plan_policy
from core.oto_row_abstain import decide_cv_row_abstain
from core.oto_row_policy import build_mapping_trace_record, should_trace_mapping_decision
from core.oto_runtime_policy import resolve_runtime_mapping_policy
from core.ja_row_runtime_v2 import (
    build_ja_alias_output_rows as _build_ja_alias_output_rows_v2,
    build_ja_cv_guard_messages as _build_ja_cv_guard_messages_v2,
    maybe_build_ja_realized_cv_anchor_record as _maybe_build_ja_realized_cv_anchor_record_v2,
)
from core.ja_row_finalize_v2 import finalize_ja_row as _finalize_ja_row_v2
from core.ja_cv_head_row_v2 import run_ja_cv_head_row as _run_ja_cv_head_row_v2
from core.ja_vcv_row_v2 import run_ja_vcv_row as _run_ja_vcv_row_v2
from core.ja_general_row_v2 import run_ja_general_row as _run_ja_general_row_v2
from core.ja_mapping_select_v2 import (
    resolve_ja_forced_cv_index as _resolve_ja_forced_cv_index_v2,
    select_ja_vcv_mapping as _select_ja_vcv_mapping_v2,
)
from core.ja_anchor_lock_v2 import (
    build_ja_anchor_lock_log_record as _build_ja_anchor_lock_log_record_v2,
    build_ja_anchor_lock_stats_delta as _build_ja_anchor_lock_stats_delta_v2,
    retune_ja_anchor_profile as _retune_ja_anchor_profile_v2,
)

logger = logging.getLogger(__name__)

__all__ = [
    "generate_ja_oto",
    "train_ja_autotune_profile",
    "save_ja_autotune_profile",
    "load_ja_autotune_profile",
    "apply_ja_autotune_profile_to_oto",
    "_convert_ja_internal_cutoff_to_oto_field",
    "_sanitize_ja_internal_params_for_wav_duration",
    "sanitize_ja_oto_for_wav_duration",
]

# ==============================================================================
# 에일리어스 접미사 유틸
# ==============================================================================

def _normalize_alias_suffix(suffix):
    s = (suffix or "").strip()
    if not s:
        return ""
    return s[1:] if s.startswith("_") else s


def apply_alias_suffix(alias, suffix):
    """에일리어스 문자열 끝에 `_<suffix>`를 부여합니다."""
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


def _replace_alias_in_oto_line(line, new_alias):
    """`wav=alias,params...` line의 alias만 교체합니다."""
    if not line or "=" not in line:
        return line
    left, right = line.split("=", 1)
    if "," in right:
        _, rest = right.split(",", 1)
        return f"{left}={new_alias},{rest}"
    return f"{left}={new_alias}"


def _katakana_to_hiragana(text):
    out = []
    for ch in text or "":
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


_ROMAJI_TO_HIRA = {}
for _k, _v in KANA_COMBO_ROMAJI.items():
    _ROMAJI_TO_HIRA.setdefault(_v, _k)
for _k, _v in KANA_SINGLE_ROMAJI.items():
    _ROMAJI_TO_HIRA.setdefault(_v, _k)
_ROMAJI_TO_HIRA.update({
    "si": "し",
    "zi": "じ",
    "ti": "てぃ",
    "tu": "とぅ",
    "di": "でぃ",
    "du": "どぅ",
    "ji": "じ",
    "hu": "ふ",
    "chi": "ち",
    "tsu": "つ",
    "fu": "ふ",
    "wi": "うぃ",
    "we": "うぇ",
    "wo": "を",
})


def _looks_japanese_kana(text):
    return bool(re.search(r"[\u3041-\u3096\u30A1-\u30FA\u30FC]", text or ""))


def _romaji_token_to_hiragana(token):
    t = (token or "").strip().lower()
    if not t:
        return t
    # VC의 자음 단독 토큰(k, s...)은 그대로 유지
    if re.fullmatch(r"[bcdfghjklmnpqrstvwxyz]{1,3}", t):
        return t
    syls = parse_ja_filename(t)
    if not syls:
        return t
    out = []
    for syl in syls:
        out.append(_ROMAJI_TO_HIRA.get(syl, syl))
    return "".join(out)


def _token_to_romaji(token):
    t = (token or "").strip()
    if not t:
        return ""
    if _looks_japanese_kana(t):
        syls = parse_ja_filename(t)
        return "".join(syls).lower() if syls else t.lower()
    return t.lower()


def _bridge_prefix_vowel_romaji(parts):
    """
    VC/VCV 계열(`V C` / `V CV`)의 앞 V를 로마자로 고정합니다.
    """
    if not parts or len(parts) < 2:
        return ""
    left = _token_to_romaji(parts[0])
    if not left:
        return ""
    if left in {"n", "nn", "xn", "m"}:
        return "n"
    onset, vowel = split_ja_romaji_syllable(left)
    if not onset and vowel in {"a", "i", "u", "e", "o"}:
        return vowel
    return ""


def convert_ja_alias_style(alias, alias_style="original"):
    """
    일본어 에일리어스를 표기 스타일에 맞게 변환합니다.
    alias_style: original | hiragana | romaji
    """
    style = (alias_style or "original").strip().lower()
    if style in {"", "original", "원본", "원본 그대로"}:
        return alias

    parts = re.split(r"\s+", (alias or "").strip())
    if not parts or parts == [""]:
        return alias

    bridge_v_prefix = _bridge_prefix_vowel_romaji(parts)
    out_parts = []
    for i, tok in enumerate(parts):
        if tok in {"-", "R", "H", "r", "h"}:
            out_parts.append(tok)
            continue

        if style in {"romaji", "roma", "로마자"}:
            if _looks_japanese_kana(tok):
                syls = parse_ja_filename(tok)
                out_parts.append("".join(syls) if syls else tok)
            else:
                out_parts.append(tok.lower())
            continue

        if style in {"hiragana", "hira", "히라가나"}:
            # VC/VCV 계열의 앞 V는 표기 옵션과 무관하게 로마자 유지
            if i == 0 and bridge_v_prefix:
                out_parts.append(bridge_v_prefix)
                continue
            if _looks_japanese_kana(tok):
                out_parts.append(_katakana_to_hiragana(tok))
            else:
                out_parts.append(_romaji_token_to_hiragana(tok))
            continue

        out_parts.append(tok)

    return " ".join(out_parts)

# 기본 파라미터
JA_DEFAULT_PARAMS = {
    'VC_OFFSET_PADDING': 180,    # VC offset에서 모음 시작 이전 패딩 (ms)
    'CV_CONSONANT_EXTEND': 0.5,  # CV에서 모음 구간 확장 비율
    'CV_CUTOFF_RATIO': 0.25,     # CV에서 cutoff 비율
}

# 일본어 매핑 신뢰도 임계치 기본값(포맷별).
# - CV/CVVC: 음절 1칸 밀림 방지를 위해 보수적으로 운용
# - VCV: 기존 호환성을 유지
JA_MAPPING_CONF_THRESHOLD_BY_FORMAT = {
    "cv": 0.68,
    "cvvc": 0.68,
    "vcv": 0.60,
    "default": 0.62,
}


def _resolve_ja_mapping_conf_threshold(format_type, override_threshold=None):
    if override_threshold is not None:
        try:
            return float(override_threshold)
        except Exception:
            pass
    fmt = str(format_type or "").strip().lower()
    base = JA_MAPPING_CONF_THRESHOLD_BY_FORMAT.get(
        fmt,
        JA_MAPPING_CONF_THRESHOLD_BY_FORMAT["default"],
    )
    return float(base)

def is_glide(phone_mark):
    clean = re.sub(r"[0-9]", "", phone_mark).lower()
    return clean in ['j', 'w', 'ɰ']

# 일본어 자음 목록 (VC/CV 분류용)
JA_CONSONANTS = [
    'k', 'g', 's', 'z', 't', 'd', 'n', 'h', 'b', 'p', 'm', 'r', 'l',
    'w', 'y', 'f', 'v', 'j', 'q', 'c',
    'sh', 'ch', 'ts', 'dz', 'hy', 'ky', 'gy', 'ny', 'my', 'ry', 'by', 'py',
    'dy', 'ty', 'ss', 'kk', 'tt', 'pp', 'dd', 'gg', 'bb', 'zz', 'jj',
]

# 일본어 모음 (단독 또는 VV 판별용)
JA_VOWELS = ['a', 'i', 'u', 'e', 'o']
JA_KANA_VOWELS = {'あ', 'い', 'う', 'え', 'お', 'ん', 'ア', 'イ', 'ウ', 'エ', 'オ', 'ン'}

# 파열음/파찰음 (VC cutoff에서 자음 소음 삭제용)
JA_PLOSIVE_CONSONANTS = [
    'k', 'g', 't', 'd', 'b', 'p',
    'kk', 'tt', 'pp', 'dd', 'gg', 'bb',
    'ch', 'ts', 'q', 'c', 'j',
    'ky', 'gy', 'ty', 'dy', 'by', 'py',
]

JA_SIBILANT_ONSETS = {
    's', 'z', 'sh', 'j', 'ts', 'dz', 'ch',
}

JA_PLOSIVE_ONSETS = {
    'k', 'g', 't', 'd', 'b', 'p', 'q', 'c',
    'kk', 'tt', 'pp', 'dd', 'gg', 'bb',
    'ky', 'gy', 'ty', 'dy', 'by', 'py',
}

JA_VOICED_ONSETS = {
    'g', 'z', 'd', 'b', 'j', 'dz', 'v',
    'm', 'n', 'ny', 'r', 'l', 'ry', 'w', 'y',
}

JA_VOICELESS_ONSETS = {
    'k', 's', 't', 'p', 'h', 'f', 'sh', 'ch', 'ts',
    'q', 'c', 'ky', 'ty', 'py', 'hy', 'ss', 'kk', 'tt', 'pp',
}

JA_NASAL_ONSETS = {'m', 'n', 'ny', 'ng', 'ngy'}
JA_LIQUID_ONSETS = {'r', 'ry', 'l'}

from core.ja_oto_mapping import (
    _build_ja_cvvc_occurrence_map,
    classify_ja_alias,
    detect_ja_alias_format,
    get_vc_consonant,
    _build_ja_syllables_from_phone_nuclei,
    _build_syllables_from_filename,
    _extract_ja_cv_target_syllable,
    _extract_ja_cv_targets_from_lines,
    _extract_ja_onset_token,
    _extract_vcv_target_syllable,
    _ja_is_n_bridge_alias,
    _ja_special_mora_class,
    _ja_soft_cv_match_level,
    _ja_syllable_tail,
    _normalize_ja_syllable_token,
    _score_ja_syllable_mapping,
    _select_ja_cv_syllable_index,
    _select_vcv_syllable_index,
    _resolve_ja_cvvc_occurrence_index,
    _syllable_info_token,
    _vcv_syllable_match_score,
)
from core.ja_oto_bridge import (
    _adaptive_ja_overlap,
    _clamp_ja_bridge_overlap,
    _ja_cv_offset_and_pre,
    _ja_cv_onset_class,
    _ja_extract_cv_bounds,
    _ja_onset_class,
    _ja_pick_vowel_phone,
    _ja_target_vowel_from_alias,
    _recenter_ja_params_around_pre,
)
JA_GLIDE_ONSETS = {'y', 'w'}
JA_FRICATIVE_ONSETS = {'h', 'f', 'v', 'hy', 's', 'z', 'sh'}

# vcv2cvvc(setting.csv) 방식의 p1~p4 구조를 직접 복제하지 않고,
# 자음군별 상대 규칙으로 일반화한 CVVC 브릿지 보정 프리셋.
JA_CVVC_BRIDGE_TIMING = {
    'default': {
        'offset_pad': 86.0,
        'offset_pad_min': 42.0,
        'offset_pad_floor': 36.0,
        'offset_len_mul': 0.08,
        'pre_lead_mul': 0.35,
        'pre_lead_min': 8.0,
        'pre_lead_max': 30.0,
        'ovl_ratio': 0.50,
        'ovl_min_ratio': 0.40,
        'ovl_pre_margin': 6.0,
        'tail_margin_base': 10.0,
        'tail_margin_mul': 0.05,
        'cons_add_base': 36.0,
        'cons_add_mul': 0.12,
        'cons_add_min': 20.0,
        'cons_add_max': 68.0,
        'cons_floor': 18.0,
        'cons_to_next_margin': 8.0,
        'cut_add_base': 58.0,
        'cut_add_mul': 0.20,
        'cut_add_min': 34.0,
        'cut_add_max': 120.0,
        'cut_min_gap': 16.0,
        'cut_to_next_allow': 22.0,
    },
    'plosive': {
        'offset_pad': 92.0,
        'offset_len_mul': 0.07,
        'pre_lead_mul': 0.42,
        'ovl_ratio': 0.44,
        'ovl_min_ratio': 0.34,
        'tail_margin_base': 8.0,
        'tail_margin_mul': 0.03,
        'cons_add_base': 22.0,
        'cons_add_mul': 0.08,
        'cons_add_max': 46.0,
        'cons_floor': 14.0,
        'cut_add_base': 40.0,
        'cut_add_mul': 0.16,
        'cut_add_max': 82.0,
        'cut_to_next_allow': 16.0,
    },
    'sibilant': {
        'offset_pad': 114.0,
        'offset_len_mul': 0.10,
        'pre_lead_mul': 0.30,
        'ovl_ratio': 0.58,
        'ovl_min_ratio': 0.48,
        'tail_margin_base': 13.0,
        'tail_margin_mul': 0.06,
        'cons_add_base': 44.0,
        'cons_add_mul': 0.15,
        'cons_add_max': 82.0,
        'cut_add_base': 78.0,
        'cut_add_mul': 0.24,
        'cut_add_max': 148.0,
        'cut_to_next_allow': 26.0,
    },
    'nasal': {
        'offset_pad': 70.0,
        'offset_len_mul': 0.06,
        'pre_lead_mul': 0.22,
        'ovl_ratio': 0.62,
        'ovl_min_ratio': 0.50,
        'tail_margin_base': 11.0,
        'tail_margin_mul': 0.05,
        'cons_add_base': 56.0,
        'cons_add_mul': 0.16,
        'cons_add_max': 94.0,
        'cut_add_base': 88.0,
        'cut_add_mul': 0.22,
        'cut_add_max': 168.0,
        'cut_to_next_allow': 32.0,
    },
    'liquid': {
        'offset_pad': 64.0,
        'offset_len_mul': 0.06,
        'pre_lead_mul': 0.24,
        'ovl_ratio': 0.60,
        'ovl_min_ratio': 0.48,
        'tail_margin_base': 10.0,
        'tail_margin_mul': 0.05,
        'cons_add_base': 52.0,
        'cons_add_mul': 0.14,
        'cons_add_max': 88.0,
        'cut_add_base': 84.0,
        'cut_add_mul': 0.20,
        'cut_add_max': 152.0,
        'cut_to_next_allow': 28.0,
    },
    'glide': {
        'offset_pad': 66.0,
        'offset_len_mul': 0.06,
        'pre_lead_mul': 0.26,
        'ovl_ratio': 0.56,
        'ovl_min_ratio': 0.46,
        'tail_margin_base': 10.0,
        'tail_margin_mul': 0.05,
        'cons_add_base': 46.0,
        'cons_add_mul': 0.13,
        'cons_add_max': 84.0,
        'cut_add_base': 78.0,
        'cut_add_mul': 0.20,
        'cut_add_max': 148.0,
        'cut_to_next_allow': 26.0,
    },
    'fricative': {
        'offset_pad': 102.0,
        'offset_len_mul': 0.09,
        'pre_lead_mul': 0.30,
        'ovl_ratio': 0.55,
        'ovl_min_ratio': 0.46,
        'tail_margin_base': 12.0,
        'tail_margin_mul': 0.05,
        'cons_add_base': 42.0,
        'cons_add_mul': 0.14,
        'cons_add_max': 84.0,
        'cut_add_base': 72.0,
        'cut_add_mul': 0.22,
        'cut_add_max': 148.0,
        'cut_to_next_allow': 26.0,
    },
}


def validate_oto_params(offset, consonant, cutoff, pre, ovl, alias_type=""):
    """
    UTAU OTO 파라미터 순서 제약을 강제합니다.
    올바른 순서: offset → overlap → preutterance → consonant → cutoff
    """
    a_type = str(alias_type or "").strip().lower()
    min_cons_gap_by_type = {
        "cv": 20.0,
        "cv_head": 20.0,
        "vc": 6.0,
        "vv": 14.0,
        "vcv": 16.0,
        "mono": 12.0,
        "br": 8.0,
    }
    min_cut_gap_by_type = {
        "cv": 14.0,
        "cv_head": 14.0,
        "vc": 4.0,
        "vv": 10.0,
        "vcv": 12.0,
        "mono": 10.0,
        "br": 6.0,
    }
    min_cons_gap = min_cons_gap_by_type.get(a_type, 30.0 if not a_type else 14.0)
    min_cut_gap = min_cut_gap_by_type.get(a_type, 50.0 if not a_type else 12.0)

    if offset < 0: offset = 0
    if pre < 0: pre = 0
    if ovl < 0: ovl = 0
    if consonant < 0: consonant = 0
    
    if ovl > pre:
        ovl = pre * 0.75
    if consonant < pre + min_cons_gap:
        consonant = pre + min_cons_gap
    
    cutoff_abs = abs(cutoff)
    if cutoff_abs <= consonant + min_cut_gap:
        cutoff_abs = consonant + min_cut_gap
    cutoff = -cutoff_abs
    
    return offset, consonant, cutoff, pre, ovl




def normalize_key(name):
    """파일명을 정규화된 키로 변환 (일본어/한국어/영문 공용)."""
    return normalize_wav_key(name)


def is_breath(alias):
    """숨소리(breath) 판별"""
    clean = alias.strip().lower()
    return bool(re.match(r'^br\d*$', clean))


def _is_ja_vowel_token(token):
    t_raw = (token or '').strip()
    if not t_raw:
        return False
    if t_raw in JA_KANA_VOWELS:
        return True
    t = t_raw.lower()
    if t in JA_VOWELS or t in ['n', 'nn', 'xn']:
        return True
    onset, vowel = split_ja_romaji_syllable(t)
    return (not onset) and (vowel in JA_VOWELS)

def _clean_phone_mark(mark):
    return re.sub(r"[0-9]", "", (mark or "").strip().lower())


def _is_nucleus_phone(mark):
    """
    음절 핵(모음/모라 비음)으로 볼 수 있는 phone인지 판별.
    """
    m = _clean_phone_mark(mark)
    return m in {'a', 'i', 'ɯ', 'u', 'e', 'o', 'ɴ', 'n', 'ə', 'ɪ', 'ʊ', 'æ', 'ɑ', 'ɔ', 'ɐ'}


def _alias_to_ja_cv_target(alias, alias_type):
    a = (alias or "").strip()
    if not a:
        return ""

    if alias_type in {"cv", "mono"}:
        return _normalize_ja_syllable_token(a)
    if alias_type == "cv_head":
        parts = a.split()
        if len(parts) >= 2 and parts[0] == "-":
            return _normalize_ja_syllable_token(parts[1])
        return _normalize_ja_syllable_token(a.lstrip("-"))
    if alias_type == "vcv":
        parts = a.split()
        if len(parts) >= 2:
            return _normalize_ja_syllable_token(parts[1])
        return _normalize_ja_syllable_token(a)
    if alias_type == "vv":
        parts = a.split()
        if len(parts) >= 2:
            return _normalize_ja_syllable_token(parts[1])
    return ""


def _target_vowel_from_filename_syllable(syl):
    s = _normalize_ja_syllable_token(syl)
    if not s:
        return ""
    onset, vowel = split_ja_romaji_syllable(s)
    if vowel in JA_VOWELS:
        return vowel
    if s in {"n", "nn", "xn", "m"}:
        return "n"
    if s in JA_VOWELS and not onset:
        return s
    return ""


def _vowel_match_score(target_vowel, nucleus_vowel):
    tv = (target_vowel or "").strip().lower()
    nv = (nucleus_vowel or "").strip().lower()
    if not tv or not nv:
        return -1.2
    if tv == nv:
        return 4.0
    if {tv, nv} <= {"i", "u"}:
        return 1.0
    if {tv, nv} <= {"e", "i"}:
        return 0.6
    if {tv, nv} <= {"o", "u"}:
        return 0.6
    if tv == "n" or nv == "n":
        return -2.0
    return -1.5


def _align_nuclei_positions_to_targets(nuclei, nucleus_vowels, target_vowels):
    """
    nuclei(관측 모음핵)와 target_vowels(파일명 음절 모음열)를 단조 정렬합니다.
    균등 샘플링 대신 모음 일치도를 사용해 1음절 밀림을 줄입니다.
    """
    n = len(nuclei)
    m = len(target_vowels)
    if n < m or m <= 0:
        return None

    neg_inf = -10**9
    dp = [[neg_inf] * n for _ in range(m)]
    parent = [[-1] * n for _ in range(m)]

    for j in range(0, n - m + 1):
        start_penalty = j * 0.35
        dp[0][j] = _vowel_match_score(target_vowels[0], nucleus_vowels[j]) - start_penalty

    for i in range(1, m):
        j_min = i
        j_max = n - (m - i)
        ideal = (i * (n - 1) / float(max(m - 1, 1)))
        for j in range(j_min, j_max + 1):
            base = _vowel_match_score(target_vowels[i], nucleus_vowels[j])
            pos_pen = abs(j - ideal) * 0.08
            best_val = neg_inf
            best_k = -1
            for k in range(i - 1, j):
                prev = dp[i - 1][k]
                if prev <= neg_inf / 2:
                    continue
                gap = j - k - 1
                gap_pen = max(0, gap) * 0.55
                val = prev + base - gap_pen - pos_pen
                if val > best_val:
                    best_val = val
                    best_k = k
            dp[i][j] = best_val
            parent[i][j] = best_k

    best_j = -1
    best_final = neg_inf
    for j in range(m - 1, n):
        if dp[m - 1][j] <= neg_inf / 2:
            continue
        tail_pen = (n - 1 - j) * 0.22
        val = dp[m - 1][j] - tail_pen
        if val > best_final:
            best_final = val
            best_j = j
    if best_j < 0:
        return None

    pos = [0] * m
    cur_j = best_j
    for i in range(m - 1, -1, -1):
        pos[i] = cur_j
        cur_j = parent[i][cur_j]
        if i > 0 and cur_j < 0:
            return None
    return [nuclei[p] for p in pos]


def _is_vowel_chain_syllables(syllables):
    """
    파일명이 순수 모음 연속음(예: a a i a u e a)인지 판별.
    CVVC 강제 선택 시에도 VCV 방식으로 처리하기 위한 예외 규칙.
    """
    if not syllables or len(syllables) < 3:
        return False
    for syl in syllables:
        onset, vowel = split_ja_romaji_syllable(syl)
        if onset:
            return False
        if vowel not in JA_VOWELS:
            return False
    return True


def _build_words_synth_phones(wd_intervals, filename_syllables=None):
    synth_phones = []
    filename_syllables = filename_syllables or []
    for idx, w in enumerate(wd_intervals or []):
        w_mark = str(getattr(w, "mark", "") or "").strip()
        if w_mark.lower() in {"<unk>", "unk", "spn"} and idx < len(filename_syllables):
            w_mark = filename_syllables[idx]
        synth_phones.extend(
            _synthesize_word_phones(w_mark, float(w.minTime), float(w.maxTime))
        )
    return synth_phones


def _build_ja_linear_syllables_from_phones(ph_intervals, filename_syllables):
    """
    파일명 음절 수는 확실하지만 phone 기반 정밀 정렬이 실패한 경우,
    순서만 보장하는 선형 분할 fallback을 만든다.
    """
    if not ph_intervals or not filename_syllables:
        return None
    total = len(ph_intervals)
    n = len(filename_syllables)
    if total <= 0 or n <= 0:
        return None
    infos = []
    for i, syl in enumerate(filename_syllables):
        start_idx = int(round(i * total / float(n)))
        end_idx = int(round((i + 1) * total / float(n)))
        if end_idx <= start_idx:
            end_idx = min(start_idx + 1, total)
        start_idx = min(max(start_idx, 0), total - 1)
        end_idx = min(max(end_idx, start_idx + 1), total)
        phones = ph_intervals[start_idx:end_idx]
        if not phones:
            phones = [ph_intervals[start_idx]]
        s_t = float(phones[0].minTime)
        e_t = float(phones[-1].maxTime)
        infos.append({
            'word': syl,
            'roman': syl,
            'start_time': s_t,
            'end_time': e_t,
            'phones': list(phones),
        })
    return infos if infos else None


def _collect_phone_tier_quality(ph_tier, expected_syllables, min_vowel_phone_ratio=0.5):
    """
    phones tier 품질 지표를 계산합니다.
    - 비침묵 phone 수
    - spn 비율
    - 핵 모음 phone 수
    - 기대 음절 대비 phone 수 비율
    """
    silence_marks = {"", "sil", "pau", "sp"}
    phone_count_non_sil = 0
    spn_count = 0
    known_vowel_phone_count = 0
    for p in ph_tier or []:
        mark = str(getattr(p, "mark", "") or "").strip().lower()
        if mark in silence_marks:
            continue
        phone_count_non_sil += 1
        if mark == "spn":
            spn_count += 1
            continue
        if _is_nucleus_phone(mark):
            known_vowel_phone_count += 1

    expected = max(0, int(expected_syllables or 0))
    spn_ratio = (float(spn_count) / float(max(1, phone_count_non_sil)))
    phones_vs_expected_ratio = (
        float(phone_count_non_sil) / float(max(1, expected))
        if expected > 0 else 0.0
    )
    min_vowel_needed = max(2, int(math.ceil(expected * max(0.1, float(min_vowel_phone_ratio or 0.5)))))

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
        "phones_vs_expected_syllables_ratio": float(phones_vs_expected_ratio),
        "expected_syllables": int(expected),
        "low_confidence_reasons": reasons,
    }


def _clamp01(v):
    try:
        x = float(v)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x))


def _time_to_ms(v):
    try:
        x = float(v)
    except Exception:
        return None
    return x * 1000.0 if abs(x) < 60.0 else x


def _estimate_words_tier_confidence(wd_intervals, expected_syllables):
    if not wd_intervals:
        return 0.0
    total = len(wd_intervals)
    if total <= 0:
        return 0.0
    expected = max(1, int(expected_syllables or total))
    ratio = float(total) / float(expected)
    ratio_pen = min(abs(1.0 - ratio) * 0.68, 0.68)
    unknown_count = 0
    for w in wd_intervals:
        mark = str(getattr(w, "mark", "") or "").strip().lower()
        if mark in {"", "<unk>", "unk", "spn", "sil", "pau", "sp"}:
            unknown_count += 1
    unknown_ratio = float(unknown_count) / float(max(1, total))
    conf = 1.0 - ratio_pen - (unknown_ratio * 0.52)
    if total < 2:
        conf -= 0.10
    return _clamp01(conf)


def _score_words_boundary_alignment(candidate_infos, wd_intervals):
    if not candidate_infos or not wd_intervals:
        return 50.0
    cand_bounds = []
    for s in (candidate_infos[:-1] if len(candidate_infos) >= 2 else []):
        if not isinstance(s, dict):
            continue
        end_ms = _time_to_ms(s.get("end_time"))
        if end_ms is not None:
            cand_bounds.append(end_ms)
    word_bounds = [float(w.maxTime) * 1000.0 for w in (wd_intervals[:-1] if len(wd_intervals) >= 2 else [])]
    if not cand_bounds or not word_bounds:
        return 50.0
    diffs = []
    for wb in word_bounds:
        nearest = min(cand_bounds, key=lambda cb: abs(cb - wb))
        diffs.append(abs(nearest - wb))
    mean_diff = sum(diffs) / float(max(1, len(diffs)))
    size_gap = abs(len(cand_bounds) - len(word_bounds))
    # 0ms -> 100점, 180ms -> 약 0점에 근접.
    score = 100.0 - (mean_diff / 1.8) - (size_gap * 6.0)
    return max(0.0, min(100.0, score))


def _estimate_ja_syllable_confidences(candidate_infos, cv_targets):
    if not candidate_infos:
        return []

    targets = [_normalize_ja_syllable_token(t) for t in (cv_targets or []) if t]
    infos = list(candidate_infos)
    n = len(infos)
    confs = [0.55] * n

    target_n = len(targets)
    if target_n > 0:
        m = min(n, target_n)
        for i in range(m):
            cand_tok = _normalize_ja_syllable_token(_syllable_info_token(infos[i]))
            tgt_tok = targets[i]
            match_score = float(_vcv_syllable_match_score(tgt_tok, cand_tok))
            conf = _clamp01(match_score / 100.0)

            t_on, t_vow = split_ja_romaji_syllable(tgt_tok)
            c_on, c_vow = split_ja_romaji_syllable(cand_tok)
            if t_vow in JA_VOWELS and c_vow and c_vow != t_vow:
                conf -= 0.34
            if t_on and c_on and t_on != c_on:
                conf -= 0.14
            t_tail = _ja_syllable_tail(tgt_tok)
            c_tail = _ja_syllable_tail(cand_tok)
            if t_tail != c_tail:
                conf -= 0.08

            # 현재 위치보다 인접 위치가 더 잘 맞으면 1음절 밀림 가능성으로 감점.
            if i > 0:
                prev_tok = _normalize_ja_syllable_token(_syllable_info_token(infos[i - 1]))
                if _vcv_syllable_match_score(tgt_tok, prev_tok) - match_score > 12.0:
                    conf -= 0.12
            if (i + 1) < n:
                next_tok = _normalize_ja_syllable_token(_syllable_info_token(infos[i + 1]))
                if _vcv_syllable_match_score(tgt_tok, next_tok) - match_score > 12.0:
                    conf -= 0.12
            confs[i] = _clamp01(conf)

        if n != target_n:
            mismatch_pen = min(0.14, abs(n - target_n) * 0.03)
            confs = [_clamp01(c - mismatch_pen) for c in confs]

    # 시간 순서 이상치(역행/비정상 급점프)를 음절 단위로 감점.
    centers = []
    for s in infos:
        if not isinstance(s, dict):
            centers.append(None)
            continue
        st = _time_to_ms(s.get("start_time"))
        et = _time_to_ms(s.get("end_time"))
        if st is None and et is None:
            centers.append(None)
        elif st is None:
            centers.append(et)
        elif et is None:
            centers.append(st)
        else:
            centers.append((st + et) * 0.5)
    prev_t = None
    for i, t in enumerate(centers):
        if t is None:
            continue
        if prev_t is not None:
            delta = t - prev_t
            if delta <= 0.0:
                confs[i] = _clamp01(confs[i] - 0.32)
                confs[max(0, i - 1)] = _clamp01(confs[max(0, i - 1)] - 0.18)
            elif delta < 24.0:
                confs[i] = _clamp01(confs[i] - 0.10)
            elif delta > 1500.0:
                confs[i] = _clamp01(confs[i] - 0.08)
        prev_t = t
    return confs


def _summarize_ja_syllable_confidences(syllable_confidences, low_cut=0.58):
    vals = [float(c) for c in (syllable_confidences or [])]
    if not vals:
        return 0.5, 0.5, 0.0
    mean_conf = sum(vals) / float(len(vals))
    min_conf = min(vals)
    low_ratio = sum(1 for c in vals if c < float(low_cut)) / float(len(vals))
    return float(mean_conf), float(min_conf), float(low_ratio)


def _evaluate_ja_mapping_candidate(candidate_infos, cv_targets, wd_intervals=None, words_tier_confidence=0.0):
    if not candidate_infos:
        return None
    score = float(_score_ja_syllable_mapping(candidate_infos, cv_targets)) if cv_targets else 0.0
    syllable_conf = _estimate_ja_syllable_confidences(candidate_infos, cv_targets)
    mean_conf, min_conf, low_ratio = _summarize_ja_syllable_confidences(syllable_conf)
    words_align_score = _score_words_boundary_alignment(candidate_infos, wd_intervals)

    # words 경계 점수는 하드 제약이 아니라 soft weight로만 반영한다.
    words_conf = _clamp01(words_tier_confidence)
    words_weight = 4.0 + (18.0 * (words_conf ** 1.35))
    words_bonus = ((words_align_score - 70.0) / 30.0) * words_weight

    objective = score + (mean_conf * 17.0) - (low_ratio * 14.0) + words_bonus
    return {
        "score": float(score),
        "objective": float(objective),
        "syllable_confidences": syllable_conf,
        "mean_syll_conf": float(mean_conf),
        "min_syll_conf": float(min_conf),
        "low_syll_ratio": float(low_ratio),
        "words_align_score": float(words_align_score),
        "words_weight": float(words_weight),
    }


def _estimate_ja_mapping_confidence(
    phone_quality,
    words_score=0.0,
    alias_score=0.0,
    used_filename_based=False,
    used_alias_based=False,
    forced_words_mapping=False,
    syllable_confidences=None,
    words_align_score=None,
    words_tier_confidence=0.0,
):
    """
    일본어 음절 매핑 신뢰도를 0~1로 추정합니다.
    - phones tier 품질
    - 현재 매핑 점수(words_score)와 alias 대안 점수(alias_score)
    - filename 기반 잠금 여부
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
    if used_filename_based and not used_alias_based:
        conf += 0.07
    if used_alias_based and not used_filename_based:
        conf -= 0.04
    if forced_words_mapping:
        conf -= 0.03
    conf += max(-0.12, min(0.12, margin / 100.0))

    mean_syll_conf, min_syll_conf, low_syll_ratio = _summarize_ja_syllable_confidences(
        syllable_confidences,
        low_cut=0.58,
    )
    conf += (mean_syll_conf - 0.62) * 0.30
    conf += (min_syll_conf - 0.50) * 0.14
    conf -= low_syll_ratio * 0.26

    if words_align_score is not None:
        words_conf = _clamp01(words_tier_confidence)
        # words 신뢰도가 높을수록 경계 점수를 더 크게 반영하되, soft하게만 반영.
        align_norm = _clamp01((float(words_align_score) - 55.0) / 45.0)
        conf += (align_norm - 0.5) * (0.18 * words_conf)

    if "insufficient_phones" in low_reasons:
        conf -= 0.14
    if "insufficient_vowel_phones" in low_reasons:
        conf -= 0.10
    if "spn_heavy" in low_reasons:
        conf -= 0.20
    conf = max(0.0, min(1.0, conf))
    return float(conf), float(margin)


def _should_allow_ja_soft_forward_shift(target_tok, expected_tok, mapped_tok):
    return _should_allow_ja_soft_forward_shift_v2(target_tok, expected_tok, mapped_tok)


def _clamp_ja_cv_index_to_order(
    target_tok,
    expected_idx,
    mapped_idx,
    syllables_info,
    *,
    format_type,
    filename_order_locked,
    mapping_tier,
):
    return _clamp_ja_cv_index_to_order_v2(
        target_tok,
        expected_idx,
        mapped_idx,
        syllables_info,
        format_type=format_type,
        filename_order_locked=filename_order_locked,
        mapping_tier=mapping_tier,
    )


def _build_ja_mapping_trace_record(
    *,
    fname,
    alias,
    alias_type,
    format_type,
    target_tok,
    expected_idx,
    mapped_idx,
    expected_tok,
    mapped_tok,
    mapping_tier,
    mapping_reason_code,
    mapping_confidence,
    filename_order_locked,
    local_conf=None,
    note="",
):
    expected_norm = _normalize_ja_syllable_token(expected_tok)
    mapped_norm = _normalize_ja_syllable_token(mapped_tok)
    target_norm = _normalize_ja_syllable_token(target_tok)
    expected_match_level = int(_ja_soft_cv_match_level(target_norm, expected_norm) or 0) if target_norm else 0
    mapped_match_level = int(_ja_soft_cv_match_level(target_norm, mapped_norm) or 0) if target_norm else 0
    return build_mapping_trace_record(
        event="ja_mapping_decision",
        fname=fname,
        alias=alias,
        alias_type=alias_type,
        format_type=format_type,
        target_token=target_norm,
        expected_idx=expected_idx,
        mapped_idx=mapped_idx,
        expected_token=expected_norm,
        mapped_token=mapped_norm,
        mapping_tier=mapping_tier,
        mapping_reason_code=mapping_reason_code,
        mapping_confidence=mapping_confidence,
        extra={
            "expected_match_level": expected_match_level,
            "mapped_match_level": mapped_match_level,
            "filename_order_locked": bool(filename_order_locked),
            "local_confidence": None if local_conf is None else float(local_conf),
            "note": str(note or ""),
        },
    )


def _find_ja_cv_vowel_match_index(target_tok, expected_idx, syllables_info, search_back=1, search_fwd=2):
    return _find_ja_cv_vowel_match_index_v2(
        target_tok,
        expected_idx,
        syllables_info,
        search_back=search_back,
        search_fwd=search_fwd,
    )


def _find_ja_exact_target_index(target_tok, expected_idx, syllables_info, search_back=3, search_fwd=3):
    return _find_ja_exact_target_index_v2(
        target_tok,
        expected_idx,
        syllables_info,
        search_back=search_back,
        search_fwd=search_fwd,
    )


def _compute_ja_youon_mismatch_ratio(candidate_infos, target_tokens):
    """candidate 음절열과 target 음절열의 요음/삽입모라 불일치 비율을 반환합니다."""
    if not candidate_infos or not target_tokens:
        return 0.0

    cand = []
    for s in candidate_infos:
        tok = _normalize_ja_syllable_token(_syllable_info_token(s))
        if tok:
            cand.append(tok)
    targets = [_normalize_ja_syllable_token(t) for t in (target_tokens or []) if t]
    if not cand or not targets:
        return 0.0

    n = min(len(cand), len(targets))
    if n <= 0:
        return 0.0

    compared = 0
    mismatch = 0
    for i in range(n):
        c_cls = _ja_special_mora_class(cand[i])
        t_cls = _ja_special_mora_class(targets[i])
        c_youon = c_cls in {"youon", "inserted"}
        t_youon = t_cls in {"youon", "inserted"}
        compared += 1
        if c_youon != t_youon:
            mismatch += 1
    if compared <= 0:
        return 0.0
    return float(mismatch) / float(compared)


def _should_force_alias_for_ja_cvvc(
    *,
    selected_candidate,
    alias_candidate,
    cv_targets,
    low_phone_quality,
):
    """일본어 CVVC에서 words 기반 후보가 불안정할 때 alias 기반으로 강제 전환할지 판단합니다."""
    if not selected_candidate or not alias_candidate or alias_candidate is selected_candidate:
        return False, "", 0.0

    # filename 기반 후보는 CVVC 순서 안정성에 유리하므로 강제 전환하지 않는다.
    if bool(selected_candidate.get("use_filename")):
        return False, "", 0.0

    selected_infos = list(selected_candidate.get("infos") or [])
    target_tokens = list(cv_targets or [])
    if not selected_infos or not target_tokens:
        return False, "", 0.0

    selected_len = len(selected_infos)
    target_len = len(target_tokens)
    youon_mismatch_ratio = _compute_ja_youon_mismatch_ratio(selected_infos, target_tokens)

    if selected_len and target_len and selected_len != target_len:
        return True, "alias_cvvc_length_mismatch", youon_mismatch_ratio

    if max(selected_len, target_len) >= 3 and youon_mismatch_ratio >= 0.34:
        return True, "alias_cvvc_youon_mismatch", youon_mismatch_ratio

    if bool(low_phone_quality) and not bool(selected_candidate.get("order_preserving")):
        return True, "alias_cvvc_low_phone_quality", youon_mismatch_ratio

    return False, "", youon_mismatch_ratio


def _remap_ja_forced_cv_index(target_tok, expected_idx, syllables_info):
    """범위를 벗어난 CVVC 강제 인덱스를 기대 위치 근처의 안정 인덱스로 재매핑합니다."""
    if not target_tok or not syllables_info:
        return None

    n = len(syllables_info)
    if n <= 0:
        return None
    e = max(0, min(int(expected_idx), n - 1))

    exact_idx = _find_ja_exact_target_index(
        target_tok,
        e,
        syllables_info,
        search_back=2,
        search_fwd=6,
    )
    if exact_idx is not None:
        return int(exact_idx)

    vowel_idx = _find_ja_cv_vowel_match_index(
        target_tok,
        e,
        syllables_info,
        search_back=2,
        search_fwd=4,
    )
    if vowel_idx is None:
        return None
    vowel_idx = int(max(0, min(int(vowel_idx), n - 1)))
    if vowel_idx <= e:
        return vowel_idx

    expected_tok = _syllable_info_token(syllables_info[e])
    mapped_tok = _syllable_info_token(syllables_info[vowel_idx])
    if _should_allow_ja_soft_forward_shift(target_tok, expected_tok, mapped_tok):
        return vowel_idx
    return None


def _ja_syllable_activity_metrics(syl_info):
    return _collect_ja_syllable_activity_metrics_v2(syl_info)


def _is_ja_cv_syllable_active(syl_info, *, require_vowel=True, min_active_ms=16.0, min_vowel_ms=10.0):
    return _is_ja_cv_syllable_active_v2(
        syl_info,
        require_vowel=require_vowel,
        min_active_ms=min_active_ms,
        min_vowel_ms=min_vowel_ms,
    )


def _remap_ja_cvvc_inactive_cv_index(
    target_tok,
    expected_idx,
    mapped_idx,
    syllables_info,
    *,
    alias_type="cv",
):
    if not syllables_info:
        return int(mapped_idx)
    n = len(syllables_info)
    if n <= 0:
        return int(mapped_idx)

    e = max(0, min(int(expected_idx), n - 1))
    m = max(0, min(int(mapped_idx), n - 1))
    require_vowel = str(alias_type or "").strip().lower() in {"cv", "cv_head"}

    if _is_ja_cv_syllable_active(syllables_info[m], require_vowel=require_vowel):
        return m

    target_norm = _normalize_ja_syllable_token(target_tok)
    seen = set()
    scan_order = []

    def _push(i):
        ii = int(i)
        if 0 <= ii < n and ii not in seen:
            seen.add(ii)
            scan_order.append(ii)

    for i in range(e, min(n, e + 3)):
        _push(i)
    for i in range(max(0, e - 1), min(n, e + 5)):
        _push(i)
    for i in range(max(0, m - 2), min(n, m + 3)):
        _push(i)
    for i in range(n):
        _push(i)

    best_idx = None
    best_key = None
    for i in scan_order:
        syl = syllables_info[i]
        active_ms, vowel_ms, _cnt = _ja_syllable_activity_metrics(syl)
        if not _is_ja_cv_syllable_active(syl, require_vowel=require_vowel):
            continue
        cand_tok = _syllable_info_token(syl)
        soft_match = int(_ja_soft_cv_match_level(target_norm, cand_tok) or 0) if target_norm else 0
        key = (
            soft_match,
            1 if i >= e else 0,
            -abs(i - e),
            active_ms,
            vowel_ms,
            -abs(i - m),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_idx = i
    return int(best_idx) if best_idx is not None else m


def _find_ja_vv_pair_prev_index(alias, expected_prev_idx, syllables_info, search_back=2, search_fwd=3):
    parts = [p for p in (alias or "").strip().split() if p]
    if len(parts) < 2:
        return None
    left = _normalize_ja_syllable_token(parts[0])
    right = _normalize_ja_syllable_token(parts[1])
    if not left or not right or not syllables_info:
        return None
    n = len(syllables_info)
    if n < 2:
        return None
    e = max(0, min(int(expected_prev_idx), n - 2))
    lo = max(0, e - max(0, int(search_back)))
    hi = min(n - 2, e + max(0, int(search_fwd)))
    for i in range(lo, hi + 1):
        prev_tok = _normalize_ja_syllable_token(_syllable_info_token(syllables_info[i]))
        next_tok = _normalize_ja_syllable_token(_syllable_info_token(syllables_info[i + 1]))
        if prev_tok == left and next_tok == right:
            return i
    return None


def _build_ja_planned_cv_indices(expected_tokens, syllables_info):
    plan = _build_ja_cv_anchor_plan_v2(expected_tokens, syllables_info)
    return plan.get("indices")


def _resolve_ja_planned_cv_index(planned_indices, expected_seq_idx, target_tok, syllables_info, *, alias_type="cv"):
    return _resolve_ja_planned_cv_index_v2(
        planned_indices,
        expected_seq_idx,
        target_tok,
        syllables_info,
        alias_type=alias_type,
    )


def _prefer_vcv_candidate_index(expected_idx, mapped_idx, target_tok, syllables_info, max_delta=1):
    return _prefer_vcv_candidate_index_v2(
        expected_idx,
        mapped_idx,
        target_tok,
        syllables_info,
        max_delta=max_delta,
    )


def _expand_vcv_lines_for_cvvc(lines, custom_map=None, include_bridge=True):
    """
    VCV alias를 CVVC 처리용으로 전개합니다.
    - `a ka` -> `a k` + `ka` (bridge + CV)
    - `a a`  -> `a a` + `a`  (VV + CV)
    """
    expanded = []
    stats = {
        "converted": 0,
        "added_bridge": 0,
        "added_cv": 0,
    }
    alias_type_cache = {}

    def _classify_cached(alias_text):
        t = alias_type_cache.get(alias_text)
        if t is None:
            t = classify_ja_alias(alias_text, custom_map)
            alias_type_cache[alias_text] = t
        return t

    for raw in lines or []:
        if "=" not in raw:
            expanded.append(raw)
            continue

        rhs = raw.split("=", 1)[1]
        alias = rhs.split(",", 1)[0].strip()
        if not alias:
            expanded.append(raw)
            continue

        alias_type = _classify_cached(alias)
        if alias_type != "vcv":
            expanded.append(raw)
            continue

        parts = alias.split()
        if len(parts) < 2:
            expanded.append(raw)
            continue

        left_norm = _normalize_ja_syllable_token(parts[0])
        right_norm = _normalize_ja_syllable_token(parts[1])
        if not right_norm:
            expanded.append(raw)
            continue

        _, left_vowel = split_ja_romaji_syllable(left_norm)
        left_v = left_vowel if left_vowel in JA_VOWELS else left_norm
        onset, vowel = split_ja_romaji_syllable(right_norm)

        if include_bridge and left_v:
            bridge_alias = ""
            if onset:
                bridge_alias = f"{left_v} {onset}"
            elif vowel:
                bridge_alias = f"{left_v} {vowel}"
            if bridge_alias:
                expanded.append(_replace_alias_in_oto_line(raw, bridge_alias))
                stats["added_bridge"] += 1

        expanded.append(_replace_alias_in_oto_line(raw, right_norm))
        stats["added_cv"] += 1
        stats["converted"] += 1

    return expanded, stats


def _compute_vcv_params_from_virtual_split(alias, prev_v_start, prev_v_end, c_boundary, n_end, base_shape=None):
    return _compute_vcv_params_from_virtual_split_v2(
        alias,
        prev_v_start,
        prev_v_end,
        c_boundary,
        n_end,
        base_shape=base_shape,
        extract_target_syllable_fn=_extract_vcv_target_syllable,
        split_syllable_fn=split_ja_romaji_syllable,
        get_bridge_profile_fn=_get_ja_cvvc_bridge_profile,
        is_n_bridge_fn=_ja_is_n_bridge_alias,
        onset_class_fn=_ja_onset_class,
        validate_fn=validate_oto_params,
        apply_base_shape_blend_fn=_apply_base_shape_blend,
    )


def _enforce_vcv_cv_entry_guard(offset, consonant, cutoff, pre, ovl, c_boundary, n_end, alias_text=""):
    return _enforce_vcv_cv_entry_guard_v2(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        c_boundary=c_boundary,
        n_end=n_end,
        alias_text=alias_text,
        is_n_bridge_fn=_ja_is_n_bridge_alias,
        extract_target_syllable_fn=_extract_vcv_target_syllable,
        split_syllable_fn=split_ja_romaji_syllable,
        validate_fn=validate_oto_params,
    )


def _enforce_cv_pre_anchor_guard(offset, consonant, cutoff, pre, ovl, c_end_abs, alias_type="cv"):
    return _enforce_cv_pre_anchor_guard_v2(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        c_end_abs=c_end_abs,
        alias_type=alias_type,
        validate_fn=validate_oto_params,
    )


def _mk_phone(start_t: float, end_t: float, mark: str):
    return SimpleNamespace(minTime=float(start_t), maxTime=float(end_t), mark=str(mark))


def _synthesize_word_phones(word_mark: str, start_t: float, end_t: float):
    """
    phones tier가 희소할 때 words 구간만으로 최소 phone 경계를 합성한다.
    """
    token_raw = (word_mark or "").strip()
    if not token_raw:
        return [_mk_phone(start_t, end_t, "a")]

    token = token_raw
    if re.search(r'[\u3041-\u3096\u30A1-\u30FA\u30FC]', token_raw):
        syls = parse_ja_filename(token_raw)
        if syls:
            token = syls[0]
    token = token.lower().strip()

    onset, vowel = split_ja_romaji_syllable(token)
    dur = max(float(end_t - start_t), 0.02)

    if token in ['n', 'nn', 'xn', 'm'] or vowel == 'n':
        return [_mk_phone(start_t, end_t, 'ɴ')]

    if onset and vowel in JA_VOWELS:
        cons_dur = min(max(dur * 0.38, 0.018), dur * 0.70)
        mid = min(max(start_t + cons_dur, start_t + 0.005), end_t - 0.005)
        vowel_mark = {'a': 'a', 'i': 'i', 'u': 'ɯ', 'e': 'e', 'o': 'o'}.get(vowel, vowel)
        return [_mk_phone(start_t, mid, onset), _mk_phone(mid, end_t, vowel_mark)]

    if vowel in JA_VOWELS:
        vowel_mark = {'a': 'a', 'i': 'i', 'u': 'ɯ', 'e': 'e', 'o': 'o'}.get(vowel, vowel)
        return [_mk_phone(start_t, end_t, vowel_mark)]

    return [_mk_phone(start_t, end_t, token or 'a')]


def generate_ja_openutau_aliases(base_alias):
    """
    OpenUtau 일본어 CVVC 포네마이저 호환을 위한 에일리어스 변형 리스트를 생성합니다.
    """
    aliases = {base_alias}
    parts = base_alias.split()

    if len(parts) == 2:
        v, c = parts[0], parts[1]
        # 스페이스 있는/없는 버전
        aliases.add(f"{v} {c}")
        aliases.add(f"{v}{c}")
        
        # 모음 종료 표기
        if c == '-' or c == 'R':
            aliases.add(f"{v} -")
            aliases.add(f"{v}-")
            aliases.add(f"{v} R")
            aliases.add(f"{v}R")

    elif len(parts) == 1:
        cv = parts[0]
        # 시작형
        aliases.add(f"- {cv}")
        aliases.add(f"-{cv}")
        # 종료형
        aliases.add(f"{cv} -")
        aliases.add(f"{cv}-")

    return list(aliases)


def _parse_oto_line_profile(line):
    return _parse_oto_line_common(line)


def _extract_base_timing_shape(line):
    return _extract_base_timing_shape_common(line)


def _apply_base_shape_blend(offset, consonant, cutoff, pre, ovl, base_shape, alias_type="cv"):
    if os.environ.get("UTOA_DISABLE_BASE_SHAPE_BLEND", "").strip().lower() in {"1", "true", "yes", "on"}:
        return validate_oto_params(offset, consonant, cutoff, pre, ovl, alias_type=alias_type)
    if not base_shape:
        return validate_oto_params(offset, consonant, cutoff, pre, ovl, alias_type=alias_type)
    if alias_type == "cv_head":
        # 템플릿의 비정상 head 라인(cut_gap 과소/선행발음 과대)을 그대로 따라가면
        # 어두 CV 길이가 과도하게 짧아질 수 있어 블렌딩을 생략한다.
        src_cut_gap = float(base_shape.get("cut_gap", max(abs(cutoff) - consonant, 20.0)))
        src_pre = float(base_shape.get("pre", pre))
        if src_cut_gap < 90.0 or src_pre > 280.0:
            return validate_oto_params(offset, consonant, cutoff, pre, ovl, alias_type=alias_type)

    if alias_type in {"vc", "vv"}:
        w = 0.30
    elif alias_type == "vcv":
        w = 0.34
    elif alias_type == "cv_head":
        w = 0.22
    else:
        w = 0.24

    pre_t = _clamp_range(base_shape.get("pre", pre), 12.0, 420.0)
    cons_gap_t = _clamp_range(base_shape.get("cons_gap", max(consonant - pre, 10.0)), 8.0, 260.0)
    cut_gap_t = _clamp_range(base_shape.get("cut_gap", max(abs(cutoff) - consonant, 20.0)), 16.0, 300.0)
    ovl_ratio_t = _clamp_range(base_shape.get("ovl_ratio", (ovl / pre) if pre > 0 else 0.30), 0.04, 0.86)

    pre_new = _blend(pre, pre_t, w)
    cons_gap_now = max(consonant - pre, 10.0)
    cons_gap_new = _blend(cons_gap_now, cons_gap_t, min(0.42, w + 0.07))
    cons_new = pre_new + cons_gap_new

    cut_gap_now = max(abs(cutoff) - consonant, 20.0)
    cut_gap_new = _blend(cut_gap_now, cut_gap_t, min(0.38, w + 0.03))
    cutoff_new = -(cons_new + cut_gap_new)

    ovl_ratio_now = (ovl / pre) if pre > 1e-6 else 0.30
    ovl_ratio_new = _blend(ovl_ratio_now, ovl_ratio_t, min(0.38, w + 0.04))
    ovl_new = max(0.0, pre_new * _clamp_range(ovl_ratio_new, 0.04, 0.86))

    return validate_oto_params(offset, cons_new, cutoff_new, pre_new, ovl_new, alias_type=alias_type)


def _is_reliable_base_profile(profile):
    if not profile:
        return False
    pre = max(float(profile.get("pre", 0.0)), 0.0)
    cons = max(float(profile.get("cons", 0.0)), 0.0)
    cut_abs = abs(float(profile.get("cutoff", 0.0)))
    ovl = max(float(profile.get("ovl", 0.0)), 0.0)
    if pre < 6.0 and cons < 6.0 and cut_abs < 6.0 and ovl < 6.0:
        return False
    if cut_abs <= cons + 2.0:
        return False
    if cons + 6.0 < pre:
        return False
    return True


def _compute_vc_params_from_vcv_anchor(
    source_profile,
    prev_v_start,
    prev_v_end,
    next_c_start,
    next_c_end,
    bridge_profile,
):
    """
    vcv2cvvc의 p1/p2 상대 이동 원리를 TextGrid 앵커와 결합한 VC 파라미터 계산.
    - source_profile: 원본 VCV OTO 한 줄(또는 그에 준하는 베이스 라인)
    - p1 성격: VCV pre 기준점에서 VC pre 절대 위치를 앞당기는 이동량
    - p2 성격: VC pre 이후 cutoff까지 확보할 길이
    """
    if not _is_reliable_base_profile(source_profile):
        return None
    if not bridge_profile:
        return None

    src_offset = max(float(source_profile.get("offset", 0.0)), 0.0)
    src_pre = _clamp_range(float(source_profile.get("pre", 0.0)), 12.0, 420.0)
    src_cons = max(float(source_profile.get("cons", 0.0)), src_pre)
    src_cut_abs = max(abs(float(source_profile.get("cutoff", 0.0))), src_cons + 12.0)
    src_ovl = max(float(source_profile.get("ovl", 0.0)), 0.0)

    prev_v_start = float(prev_v_start)
    prev_v_end = float(prev_v_end)
    next_c_start = float(next_c_start)
    next_c_end = float(next_c_end)
    v_len = max(prev_v_end - prev_v_start, 40.0)
    c_len = max(next_c_end - next_c_start, 60.0)

    # p1: VCV pre 절대 위치에서 VC pre 절대 위치를 얼마나 당길지 결정
    p1_base = bridge_profile.get("offset_pad", 86.0)
    p1_dyn = p1_base + max(v_len - 140.0, 0.0) * bridge_profile.get("offset_len_mul", 0.08)
    p1_lo = bridge_profile.get("offset_pad_min", 42.0)
    p1_hi = min(240.0, max(v_len * 0.92, p1_base + 36.0))
    p1 = _clamp_range(p1_dyn, p1_lo, p1_hi)
    if v_len < p1:
        p1 = max(v_len * 0.78, bridge_profile.get("offset_pad_floor", 36.0))

    src_pre_abs = src_offset + src_pre
    target_pre_abs = src_pre_abs - p1
    pre_abs_min = prev_v_start + max(v_len * 0.28, 10.0)
    pre_abs_max = min(next_c_start - 4.0, prev_v_end + 16.0)
    if pre_abs_max <= pre_abs_min:
        pre_abs_max = pre_abs_min + 2.0
    target_pre_abs = _clamp_range(target_pre_abs, pre_abs_min, pre_abs_max)

    dyn_pre = _clamp_range(next_c_start - target_pre_abs, 16.0, 260.0)
    pre = _clamp_range(_blend(dyn_pre, src_pre, 0.74), 16.0, 420.0)
    offset = max(target_pre_abs - pre, 0.0)

    # overlap은 원본 pre-ovl 간격을 최대한 유지하면서, 앞 모음 tail 앵커에도 맞춘다.
    tail_margin = bridge_profile.get("tail_margin_base", 10.0) + v_len * bridge_profile.get("tail_margin_mul", 0.05)
    tail_margin = _clamp_range(tail_margin, 4.0, 24.0)
    target_ovl_abs = prev_v_end - tail_margin
    ovl_pre_margin = bridge_profile.get("ovl_pre_margin", 6.0)
    ovl_min_ratio = bridge_profile.get("ovl_min_ratio", 0.40)
    ovl_ratio = bridge_profile.get("ovl_ratio", 0.50)
    upper_ovl = max(pre - ovl_pre_margin, 0.0)
    lower_ovl = min(pre * ovl_min_ratio, upper_ovl)
    src_pre_ovl_gap = max(src_pre - src_ovl, 4.0)
    ovl_from_src = pre - src_pre_ovl_gap
    ovl_from_anchor = target_ovl_abs - offset
    ovl = _blend(ovl_from_src, ovl_from_anchor, 0.58)
    ovl = _blend(ovl, pre * ovl_ratio, 0.20)
    ovl = min(upper_ovl, max(lower_ovl, ovl))

    # consonant/cutoff는 source gap + 자음군 규칙을 혼합해 안정화
    src_cons_gap = max(src_cons - src_pre, 8.0)
    dyn_cons_gap = bridge_profile.get("cons_add_base", 36.0) + max(c_len - 70.0, 0.0) * bridge_profile.get("cons_add_mul", 0.12)
    dyn_cons_gap = _clamp_range(
        dyn_cons_gap,
        bridge_profile.get("cons_add_min", 20.0),
        bridge_profile.get("cons_add_max", 68.0),
    )
    cons_gap = _blend(dyn_cons_gap, src_cons_gap, 0.36)
    cons_gap = max(cons_gap, bridge_profile.get("cons_floor", 18.0))
    consonant = pre + cons_gap

    next_cv_pre_rel = max(next_c_end - offset, pre + 20.0)
    next_cv_pre_rel = min(next_cv_pre_rel, pre + 280.0)
    consonant = min(consonant, next_cv_pre_rel - bridge_profile.get("cons_to_next_margin", 8.0))
    consonant = max(consonant, pre + bridge_profile.get("cons_floor", 18.0))

    # p2에 해당하는 cutoff 여유 길이
    src_cut_gap = max(src_cut_abs - src_cons, 16.0)
    dyn_cut_gap = bridge_profile.get("cut_add_base", 58.0) + max(c_len - 70.0, 0.0) * bridge_profile.get("cut_add_mul", 0.20)
    dyn_cut_gap = _clamp_range(
        dyn_cut_gap,
        bridge_profile.get("cut_add_min", 34.0),
        bridge_profile.get("cut_add_max", 120.0),
    )
    cut_gap = _blend(dyn_cut_gap, src_cut_gap, 0.34)
    cutoff_abs = max(
        consonant + bridge_profile.get("cut_min_gap", 16.0),
        pre + cut_gap,
    )
    cutoff_abs = min(cutoff_abs, next_cv_pre_rel + bridge_profile.get("cut_to_next_allow", 22.0))
    if cutoff_abs <= consonant + 8.0:
        cutoff_abs = consonant + 10.0
    cutoff = -cutoff_abs

    return validate_oto_params(offset, consonant, cutoff, pre, ovl, alias_type="vc")


def _refine_ja_vc_with_adjacent_cv(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    *,
    c_char="",
    prev_cv_anchor=None,
    next_cv_anchor=None,
    prev_v_end_abs=None,
    next_c_start_abs=None,
    next_c_end_abs=None,
):
    """
    CV 기반 VC 재정렬(2차 가드).
    - pre(abs): 다음 CV 자음 시작 근처
    - ovl(abs): 이전 CV 모음 tail 근처
    - cutoff(abs): 다음 CV 모음 진입 전/근처에서 종료
    """
    if next_cv_anchor is None and next_c_start_abs is None:
        return validate_oto_params(offset, consonant, cutoff, pre, ovl, alias_type="vc")

    hard_cls = (c_char in JA_PLOSIVE_CONSONANTS) or (c_char in JA_SIBILANT_ONSETS) or (c_char in JA_FRICATIVE_ONSETS)
    son_cls = (c_char in JA_NASAL_ONSETS) or (c_char in JA_LIQUID_ONSETS) or (c_char in JA_GLIDE_ONSETS)

    onset_abs = (
        float(next_cv_anchor.get("onset_abs"))
        if next_cv_anchor and next_cv_anchor.get("onset_abs") is not None
        else float(next_c_start_abs or 0.0)
    )
    vowel_start_abs = (
        float(next_cv_anchor.get("vowel_start_abs"))
        if next_cv_anchor and next_cv_anchor.get("vowel_start_abs") is not None
        else float(next_c_end_abs if next_c_end_abs is not None else (onset_abs + 30.0))
    )
    if vowel_start_abs < onset_abs:
        vowel_start_abs = onset_abs

    prev_v_end = (
        float(prev_cv_anchor.get("vowel_end_abs"))
        if prev_cv_anchor and prev_cv_anchor.get("vowel_end_abs") is not None
        else float(prev_v_end_abs if prev_v_end_abs is not None else onset_abs)
    )

    cur_pre_abs = float(offset) + float(pre)
    cur_cons_abs = float(offset) + float(consonant)
    cur_cut_abs = float(offset) + abs(float(cutoff))

    # 1) pre(abs) 고정: 다음 C 시작 근처
    if hard_cls:
        pre_lead = 5.0
    elif son_cls:
        pre_lead = 8.0
    else:
        pre_lead = 6.0
    target_pre_abs = onset_abs - pre_lead
    target_pre_abs = _clamp_range(target_pre_abs, onset_abs - 18.0, onset_abs + 8.0)
    pre_abs_new = _blend(cur_pre_abs, target_pre_abs, 0.72)
    pre_abs_new = _clamp_range(pre_abs_new, onset_abs - 18.0, onset_abs + 8.0)

    pre_target = float(pre)
    if next_cv_anchor and next_cv_anchor.get("pre") is not None:
        pre_target = _blend(pre_target, float(next_cv_anchor["pre"]), 0.38)
    pre_target = _clamp_range(pre_target, 30.0, 220.0)

    offset_new = max(pre_abs_new - pre_target, 0.0)
    pre_new = max(pre_abs_new - offset_new, 0.0)

    # 2) ovl(abs): 앞 모음 tail 기준
    if hard_cls:
        tail_margin = 10.0
    elif son_cls:
        tail_margin = 6.0
    else:
        tail_margin = 8.0
    ovl_abs_target = prev_v_end - tail_margin
    ovl_new = ovl_abs_target - offset_new
    ovl_new = _clamp_ja_bridge_overlap(pre_new, ovl_new, c_char, mode="vc")
    ovl_abs_new = offset_new + ovl_new

    # 3) consonant/cutoff: 다음 CV 진입 기준으로 제한
    if hard_cls:
        cons_abs_target = onset_abs - 8.0
        cons_allow_after_onset = -2.0
        cut_gap_target = 8.0
        cut_allow_after_vowel = -8.0
        cut_allow_after_onset = -1.0
    elif son_cls:
        cons_abs_target = onset_abs + 6.0
        cons_allow_after_onset = 14.0
        cut_gap_target = 18.0
        cut_allow_after_vowel = 6.0
        cut_allow_after_onset = 18.0
    else:
        cons_abs_target = onset_abs + 2.0
        cons_allow_after_onset = 10.0
        cut_gap_target = 15.0
        cut_allow_after_vowel = 2.0
        cut_allow_after_onset = 14.0

    cons_abs_min = (offset_new + pre_new) + 12.0
    cons_abs_max = min(vowel_start_abs - 6.0, onset_abs + cons_allow_after_onset)
    if cons_abs_max <= cons_abs_min:
        cons_abs_max = cons_abs_min + 2.0
    cons_abs_new = _blend(cur_cons_abs, cons_abs_target, 0.66)
    cons_abs_new = _clamp_range(cons_abs_new, cons_abs_min, cons_abs_max)
    consonant_new = max(cons_abs_new - offset_new, pre_new + 10.0)

    cut_abs_target = cons_abs_new + cut_gap_target
    cut_abs_upper = min(vowel_start_abs + cut_allow_after_vowel, onset_abs + cut_allow_after_onset)
    cut_abs_min = cons_abs_new + 8.0
    if cut_abs_upper <= cut_abs_min:
        cut_abs_upper = cut_abs_min + 2.0
    cut_abs_new = _blend(cur_cut_abs, cut_abs_target, 0.72)
    cut_abs_new = _clamp_range(cut_abs_new, cut_abs_min, cut_abs_upper)

    if hard_cls:
        onset_rel = max(onset_abs - offset_new, pre_new + 8.0)
        cons_floor = pre_new + 6.0
        cons_cap = onset_rel - 6.0
        consonant_new = min(consonant_new, cons_cap)
        consonant_new = max(consonant_new, cons_floor)

        cut_cap = onset_rel - 1.0
        cut_floor = consonant_new + 4.0
        if cut_floor > cut_cap:
            consonant_new = max(cons_floor, cut_cap - 4.0)
            cut_floor = consonant_new + 4.0
        if cut_floor > cut_cap:
            cut_cap = cut_floor + 0.8
        cut_abs_new = min(cut_abs_new, cut_cap)
        cut_abs_new = max(cut_abs_new, cut_floor)

    cutoff_new = -(cut_abs_new - offset_new)

    return validate_oto_params(offset_new, consonant_new, cutoff_new, pre_new, ovl_abs_new - offset_new, alias_type="vc")


def _limit_pre_anchor_shift(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    *,
    pre_abs_before,
    max_shift_ms,
    alias_type="",
):
    """
    pre 절대 위치 이동량을 제한해 브리지 앵커 과보정을 방지한다.
    반환: (offset, consonant, cutoff, pre, ovl, pre_abs_after, clamped)
    """
    pre_abs_after = float(offset) + float(pre)
    delta = float(pre_abs_after - pre_abs_before)
    limit = max(0.0, float(max_shift_ms))
    if abs(delta) <= limit:
        out = validate_oto_params(offset, consonant, cutoff, pre, ovl, alias_type=alias_type)
        return (*out, float(out[0] + out[3]), False)

    limited_pre_abs = float(pre_abs_before) + (limit if delta > 0.0 else -limit)
    abs_shift = pre_abs_after - limited_pre_abs
    offset = float(offset) - float(abs_shift)
    out = validate_oto_params(offset, consonant, cutoff, pre, ovl, alias_type=alias_type)
    return (*out, float(out[0] + out[3]), True)


def _normalize_alias_for_profile(alias):
    return canonicalize_alias_for_matching("japanese", alias)


def _read_text_with_fallback(path):
    return _read_text_with_fallback_common(path)


def _read_oto_rows_for_profile(path):
    return _read_oto_rows_for_profile_common(
        path,
        wav_normalizer=normalize_wav_key,
        alias_normalizer=_normalize_alias_for_profile,
    )


def _occurrence_map(rows):
    return _build_occurrence_map_common(rows)


def _median(vals):
    if not vals:
        return 0.0
    s = sorted(float(v) for v in vals)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _clamp(v, lim):
    return max(-lim, min(lim, float(v)))


def _clamp_range(v, lo, hi):
    return max(float(lo), min(float(hi), float(v)))


def _blend(a, b, w):
    ww = max(0.0, min(1.0, float(w)))
    return (1.0 - ww) * float(a) + ww * float(b)


def _apply_ja_autotune_profile(alias_type, offset, consonant, cutoff, pre, ovl, profile):
    return _apply_ja_autotune_profile_impl(
        alias_type,
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        profile,
        validate_fn=validate_oto_params,
    )


def _ja_extract_onset_for_timing(alias_text, alias_type):
    alias = (alias_text or "").strip().lower()
    if not alias:
        return ""
    parts = alias.split()
    token = alias

    if len(parts) >= 2:
        if alias_type in {"vc", "vv", "vcv"}:
            token = parts[1]
        elif alias_type == "cv_head":
            token = parts[1]
        else:
            token = parts[0] if parts[0] != "-" else parts[1]
    elif alias.startswith("-"):
        token = alias[1:]

    token = token.strip("-_ ")
    if token in JA_CONSONANTS:
        return token
    onset, _ = split_ja_romaji_syllable(token)
    return (onset or token).lower()


def _get_ja_timing_traits(alias_text, alias_type):
    onset = _ja_extract_onset_for_timing(alias_text, alias_type)
    if not onset:
        return {"onset": "", "manner": "other", "voicing": "unknown"}

    manner = "other"
    if onset in JA_SIBILANT_ONSETS:
        manner = "sibilant"
    elif onset in JA_PLOSIVE_ONSETS:
        manner = "plosive"

    voicing = "unknown"
    if onset in JA_VOICED_ONSETS:
        voicing = "voiced"
    elif onset in JA_VOICELESS_ONSETS:
        voicing = "voiceless"
    return {"onset": onset, "manner": manner, "voicing": voicing}


def _get_ja_cvvc_bridge_profile(onset):
    o = (onset or "").strip().lower()
    key = "default"
    if o in JA_SIBILANT_ONSETS:
        key = "sibilant"
    elif o in JA_PLOSIVE_ONSETS:
        key = "plosive"
    elif o in JA_NASAL_ONSETS:
        key = "nasal"
    elif o in JA_LIQUID_ONSETS:
        key = "liquid"
    elif o in JA_GLIDE_ONSETS:
        key = "glide"
    elif o in JA_FRICATIVE_ONSETS:
        key = "fricative"

    merged = dict(JA_CVVC_BRIDGE_TIMING["default"])
    merged.update(JA_CVVC_BRIDGE_TIMING.get(key, {}))
    return merged


def _apply_ja_consonant_timing_shaping(alias_text, alias_type, pre, consonant, cutoff, ovl):
    traits = _get_ja_timing_traits(alias_text, alias_type)
    manner = traits["manner"]
    voicing = traits["voicing"]

    pre_mul = 1.0
    cons_gap_mul = 1.0
    cut_gap_mul = 1.0
    ovl_bias = 0.0

    if manner == "plosive":
        pre_mul *= 0.95
        cons_gap_mul *= 0.93
        cut_gap_mul *= 0.95
        ovl_bias -= 0.05
    elif manner == "sibilant":
        pre_mul *= 1.07
        cons_gap_mul *= 1.04
        cut_gap_mul *= 1.03
        ovl_bias += 0.06

    if voicing == "voiced":
        pre_mul *= 1.03
        cons_gap_mul *= 1.05
        cut_gap_mul *= 1.02
        ovl_bias += 0.02
    elif voicing == "voiceless":
        pre_mul *= 0.97
        cons_gap_mul *= 0.95
        cut_gap_mul *= 0.98
        ovl_bias -= 0.02

    pre_new = _clamp_range(pre * pre_mul, 18.0, 360.0)
    cons_gap_now = max(consonant - pre, 8.0)
    cons_gap_new = _clamp_range(cons_gap_now * cons_gap_mul, 8.0, 260.0)
    cons_new = pre_new + cons_gap_new

    ovl_ratio_now = (ovl / pre) if pre > 0 else 0.30
    ovl_ratio_new = _clamp_range(ovl_ratio_now + ovl_bias, 0.06, 0.82)
    ovl_new = max(0.0, pre_new * ovl_ratio_new)

    cut_gap_now = max(abs(cutoff) - consonant, 16.0)
    cut_gap_new = _clamp_range(cut_gap_now * cut_gap_mul, 16.0, 240.0)
    cutoff_new = -(cons_new + cut_gap_new)
    _, cons_new, cutoff_new, pre_new, ovl_new = validate_oto_params(
        0.0, cons_new, cutoff_new, pre_new, ovl_new, alias_type=alias_type
    )
    return pre_new, cons_new, cutoff_new, ovl_new


def _apply_ja_style_profile(alias_type, offset, consonant, cutoff, pre, ovl, profile, alias_text=""):
    if not profile:
        return offset, consonant, cutoff, pre, ovl
    buckets = profile.get("buckets") or {}
    if not isinstance(buckets, dict):
        return offset, consonant, cutoff, pre, ovl
    stat = buckets.get(alias_type)
    if not stat:
        return offset, consonant, cutoff, pre, ovl

    n = float(stat.get("n", 0))
    if n < 8:
        return offset, consonant, cutoff, pre, ovl

    # conservative style blending to avoid overriding per-file timing
    w = max(0.10, min(0.34, n / 2600.0))
    pre_t = _clamp_range(stat.get("pre", pre), 0.0, 360.0)
    cons_gap_t = _clamp_range(stat.get("cons_gap", max(consonant - pre, 10.0)), 8.0, 260.0)
    cut_gap_t = _clamp_range(stat.get("cut_gap", max(abs(cutoff) - consonant, 20.0)), 16.0, 260.0)
    ovl_ratio_t = _clamp_range(stat.get("ovl_ratio", (ovl / pre) if pre > 0 else 0.3), 0.0, 0.82)

    pre_new = _blend(pre, pre_t, w)
    cons_gap_now = max(consonant - pre, 10.0)
    cons_new = pre_new + _blend(cons_gap_now, cons_gap_t, w)
    cut_gap_now = max(abs(cutoff) - consonant, 20.0)
    cut_gap_new = _blend(cut_gap_now, cut_gap_t, min(0.14, w * 0.5))
    cutoff_new = -(cons_new + cut_gap_new)

    ovl_ratio_now = (ovl / pre) if pre > 0 else 0.3
    ovl_ratio_new = _blend(ovl_ratio_now, ovl_ratio_t, w)
    ovl_new = max(0.0, pre_new * _clamp_range(ovl_ratio_new, 0.0, 0.82))
    pre_new, cons_new, cutoff_new, ovl_new = _apply_ja_consonant_timing_shaping(
        alias_text, alias_type, pre_new, cons_new, cutoff_new, ovl_new
    )

    if alias_type == "cv_head" and stat.get("head_offset") is not None:
        # head_offset은 "어두 시작점"에 대한 참고치다.
        # 후반 음절까지 절대값으로 끌어당기면 한 음절 밀림/공백 배치를 유발하므로,
        # 파일 초반 근처에서만 약하게 보정한다.
        head_off_t = _clamp_range(stat.get("head_offset", offset), 0.0, 2200.0)
        if offset <= 1400.0 and abs(offset - head_off_t) <= 260.0:
            offset = _blend(offset, head_off_t, min(0.10, w * 0.35))

    return validate_oto_params(offset, cons_new, cutoff_new, pre_new, ovl_new, alias_type=alias_type)


def train_ja_autotune_profile(auto_oto_path, manual_oto_path, custom_phonemes_path=""):
    """부분 수동 OTO와 자동 OTO를 매칭해 일본어 델타 기반 프로파일을 학습합니다."""
    custom_map = load_custom_phonemes(custom_phonemes_path)
    return _train_ja_autotune_profile(auto_oto_path, manual_oto_path, custom_map=custom_map)


def save_ja_autotune_profile(path, profile):
    return _save_ja_autotune_profile(path, profile)


def load_ja_autotune_profile(path):
    return _load_ja_autotune_profile(path)


def apply_ja_autotune_profile_to_oto(oto_path, profile, custom_phonemes_path=""):
    """학습된 델타 프로파일을 OTO에 적용합니다. 반환값은 변경 라인 수."""
    if isinstance(profile, str):
        profile = _load_ja_autotune_profile(profile)
    if not profile:
        return 0
    custom_map = load_custom_phonemes(custom_phonemes_path)
    return _apply_profile_to_oto_file(
        oto_path,
        profile,
        custom_map=custom_map,
        validate_fn=validate_oto_params,
    )


def generate_ja_oto(
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
    alias_style='original',
    ja_mapping_words_fallback_enabled=True,
    ja_mapping_spn_ratio_threshold=0.35,
    ja_mapping_min_vowel_phone_ratio=0.5,
    ja_mapping_debug_reason_logging=True,
    ja_mapping_confidence_threshold=None,
    cleanup_timing_jsonl=True,
    auto_format=None,
    callback=None,
    ml_policy="",
    runtime_report=None,
):
    """
    TextGrid 기반 일본어 CVVC OTO.ini 자동 생성.
    
    Args:
        tg_folder: TextGrid 파일이 있는 폴더
        tpl_path: 템플릿 OTO.ini 파일 경로 (없으면 자동 생성)
        out_path: 출력 OTO.ini 파일 경로
        params: 튜닝 파라미터 딕셔너리
        generate_openutau: OpenUtau 호환 에일리어스 복제 여부
        gen_missing_vowels: 미지정 모음 자동 생성 여부
        fallback_format: 템플릿이 없을 때 사용할 포맷 ('cv', 'cvvc', 'vcv')
        custom_phonemes_path: 특수 발음 매핑 파일 경로
        alias_suffix: 생성되는 모든 에일리어스에 부여할 접미사 (예: 'C4' -> '_C4')
        alias_style: 일본어 에일리어스 표기 방식 ('original'|'hiragana'|'romaji')
        auto_format: GUI 드롭다운 포맷 문자열 (있으면 fallback_format을 자동 매핑)
        callback: 진행 상황 콜백 함수
    
    Returns:
        (처리된 파일 수, 전체 파일 수, 에러 목록)
    """
    try:
        # textgrid import is already handled at the top
        pass
    except ImportError:
        err = "❌ textgrid 모듈이 설치되지 않았습니다. pip install textgrid 를 실행해 주세요."
        logger.error(err)
        if callback:
            callback(err)
        return 0, 0, [err]

    if params is None:
        params = JA_DEFAULT_PARAMS.copy()
    use_sinsy_labels = bool(params.get("USE_SINSY_LABELS", False)) if params else False
    sinsy_label_path = str(params.get("SINSY_LABEL_PATH", "") or "").strip() if params else ""
    # KR 생성기와 동일한 신호 분석 헬퍼를 재사용해 멜 기반 가드 동작을 맞춘다.
    from core.oto_generator import (
        _read_wav_mono_np,
        _mel_envelope,
        _find_wav_path_for_name,
        _wav_duration_ms,
        _apply_soft_mel_offset_cutoff_guard,
    )

    # GUI 형식 지정:
    # - "자동 감지"면 강제 지정 없이 템플릿/에일리어스를 자동 판별
    # - 값이 지정되면 템플릿이 있어도 해당 형식을 우선 적용
    forced_format = None
    if auto_format:
        from core.format_type_utils import normalize_auto_format_value

        normalized_auto_format = normalize_auto_format_value("japanese", auto_format)
        if normalized_auto_format:
            forced_format = normalized_auto_format
            fallback_format = normalized_auto_format

    auto_gen_format = (fallback_format or "cvvc").strip().lower()
    if auto_gen_format not in {"cv", "cvvc", "vcv"}:
        log_msg = (
            f"⚠️ 자동 에일리어스 생성은 현재 CV/CVVC/VCV만 지원합니다. "
            f"{auto_gen_format.upper()} -> CVVC로 전환합니다."
        )
        if callback:
            callback(log_msg)
        auto_gen_format = "cvvc"

    # 매핑 옵션: 함수 인자 + 환경변수 병행 지원
    env_words_fallback = str(os.environ.get("UTOA_JA_MAPPING_WORDS_FALLBACK", "")).strip().lower()
    if env_words_fallback in {"0", "false", "off", "no"}:
        ja_mapping_words_fallback_enabled = False
    elif env_words_fallback in {"1", "true", "on", "yes"}:
        ja_mapping_words_fallback_enabled = True
    env_use_sinsy = str(os.environ.get("UTOA_USE_SINSY_LABELS", "")).strip().lower()
    if env_use_sinsy in {"0", "false", "off", "no"}:
        use_sinsy_labels = False
    elif env_use_sinsy in {"1", "true", "on", "yes"}:
        use_sinsy_labels = True
    env_sinsy_path = str(os.environ.get("UTOA_SINSY_LABEL_PATH", "")).strip()
    if env_sinsy_path:
        sinsy_label_path = env_sinsy_path
    env_spn_th = str(os.environ.get("UTOA_JA_MAPPING_SPN_RATIO_THRESHOLD", "")).strip()
    if env_spn_th:
        try:
            ja_mapping_spn_ratio_threshold = float(env_spn_th)
        except Exception:
            pass
    env_vowel_ratio = str(os.environ.get("UTOA_JA_MAPPING_MIN_VOWEL_PHONE_RATIO", "")).strip()
    if env_vowel_ratio:
        try:
            ja_mapping_min_vowel_phone_ratio = float(env_vowel_ratio)
        except Exception:
            pass
    env_debug_reason = str(os.environ.get("UTOA_JA_MAPPING_DEBUG_REASON", "")).strip().lower()
    if env_debug_reason in {"0", "false", "off", "no"}:
        ja_mapping_debug_reason_logging = False
    elif env_debug_reason in {"1", "true", "on", "yes"}:
        ja_mapping_debug_reason_logging = True
    ja_mapping_trace_logging = True
    env_mapping_trace = str(os.environ.get("UTOA_JA_MAPPING_TRACE", "")).strip().lower()
    if env_mapping_trace in {"0", "false", "off", "no"}:
        ja_mapping_trace_logging = False
    elif env_mapping_trace in {"1", "true", "on", "yes"}:
        ja_mapping_trace_logging = True
    env_conf_th = str(os.environ.get("UTOA_JA_MAPPING_CONF_THRESHOLD", "")).strip()
    if env_conf_th:
        try:
            ja_mapping_confidence_threshold = float(env_conf_th)
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

    def _alias_out(a):
        a_conv = convert_ja_alias_style(a, alias_style=alias_style)
        return apply_alias_suffix(a_conv, alias_suffix)

    alias_type_cache = {}

    def _classify_alias_cached(alias_text):
        t = alias_type_cache.get(alias_text)
        if t is None:
            t = classify_ja_alias(alias_text, custom_map)
            alias_type_cache[alias_text] = t
        return t

    errors = []
    skipped_entries = SkippedEntryCollector()
    anchor_stats = {
        "anchor_locked_count": 0,
        "cutoff_clamped_count": 0,
        "vc_cutoff_leak_guard_count": 0,
    }
    _core_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(_core_dir)
    trace_setup = build_ja_trace_preparation(_project_dir)
    _anchor_log_dir = trace_setup.anchor_log_dir
    anchor_log_path = trace_setup.anchor_log_path
    mapping_trace_path = trace_setup.mapping_trace_path
    mapping_trace_records = MappingTraceCollector(enabled=ja_mapping_trace_logging)
    diagnostics = GeneratorDiagnosticsAdapter(
        skipped_collector=skipped_entries,
        log_fn=log,
        trace_collector=mapping_trace_records,
    )

    def _apply_ja_anchor_lock(
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
        anchor_abs_ms: float,
        next_onset_abs_ms: float | None = None,
        next_vowel_abs_ms: float | None = None,
        profile_alias_type: str | None = None,
        lite: bool = False,
    ):
        alias_key = str(profile_alias_type or alias_type or "").strip().lower()

        def _get_profile(lang, fmt, alias_kind):
            from core.timing_anchor_profiles import get_anchor_profile

            return get_anchor_profile(lang, fmt, alias_kind, mode="rhythm_stable")

        def _apply_stats_delta(alias_kind, applied_rules):
            delta = _build_ja_anchor_lock_stats_delta_v2(alias_kind, applied_rules)
            for key, value in delta.items():
                if value:
                    anchor_stats[key] += int(value)

        def _build_log_record(fmt, alias_kind, before, result):
            return _build_ja_anchor_lock_log_record_v2(
                fname=fname,
                alias_text=alias_text,
                format_type=fmt,
                alias_type=alias_kind,
                lite=bool(lite),
                before=before,
                result=result,
            )

        out = apply_language_anchor_lock(
            language="japanese",
            format_type=format_type,
            alias_type=alias_type,
            before=(offset, consonant, cutoff, pre, ovl),
            file_duration_ms=wav_duration_ms,
            timeline_start_ms=timeline_start_ms,
            timeline_end_ms=effective_end_ms,
            anchor_abs_ms=anchor_abs_ms,
            next_onset_abs_ms=next_onset_abs_ms,
            next_vowel_abs_ms=next_vowel_abs_ms,
            mapping_confidence=1.0,
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
            get_profile_fn=lambda lang, fmt, alias_kind: _get_profile(lang, fmt, alias_key),
            retune_profile_fn=lambda profile: _retune_ja_anchor_profile_v2(
                profile,
                format_type=format_type,
                alias_key=alias_key,
            ),
            apply_stats_delta_fn=_apply_stats_delta,
            build_log_record_fn=_build_log_record,
            append_log_fn=append_timing_anchor_log,
            log_path=anchor_log_path,
        )
        if str(alias_type or "").strip().lower() == "vc" and next_onset_abs_ms is not None:
            c_hint = get_vc_consonant(alias_text)
            is_hard_vc = (
                c_hint in JA_PLOSIVE_CONSONANTS
                or c_hint in JA_SIBILANT_ONSETS
                or c_hint in JA_FRICATIVE_ONSETS
            )
            if is_hard_vc:
                o, c, cut, p, v = [float(x) for x in out]
                next_onset_rel = max(float(next_onset_abs_ms) - o, p + 8.0)
                c_cap = next_onset_rel - 6.0
                c = min(c, c_cap)
                c = max(c, p + 6.0)
                cut_abs = abs(cut)
                cut_cap = next_onset_rel - 1.0
                cut_floor = c + 4.0
                if cut_floor > cut_cap:
                    c = max(p + 6.0, cut_cap - 4.0)
                    cut_floor = c + 4.0
                if cut_floor > cut_cap:
                    cut_cap = cut_floor + 0.8
                cut_abs = min(cut_abs, cut_cap)
                cut_abs = max(cut_abs, cut_floor)
                out = validate_oto_params(o, c, -cut_abs, p, v, alias_type="vc")
        return out

    _append_mapping_trace = diagnostics.append_mapping_trace
    _record_unset = diagnostics.record_unset
    _record_unset_lines = diagnostics.record_unset_lines
    _log_unset_summary = diagnostics.log_unset_summary

    profile_path = _profile_path_for_out(out_path)
    autotune_profile = _load_ja_autotune_profile(profile_path)
    had_profile_on_start = bool(autotune_profile)
    ja_style_profile = get_ja_profile_preset(fallback_format)
    # Default behavior: apply abstract style preset unless explicitly disabled.
    ja_style_env = str(os.environ.get("UTOA_JA_USE_STYLE_PRESET", "")).strip().lower()
    if ja_style_env in {"0", "false", "no", "off"}:
        ja_style_enabled = False
    else:
        ja_style_enabled = True
    if autotune_profile:
        b_count = len((autotune_profile.get("buckets") or {}))
        log(f"[AutoTune] 내부 OTO 튜닝 프로파일 적용: {profile_path} (buckets={b_count})")
    if ja_style_profile and ja_style_enabled:
        log(f"[JA-Profile] 추상 프리셋 적용 준비(기본값): {ja_style_profile.get('preset_name', 'ja_profile')}")
    elif ja_style_profile and not ja_style_enabled:
        log("[JA-Profile] 추상 프리셋 적용 비활성화(환경변수 설정)")

    ja_setup = prepare_ja_generation_setup(
        tg_folder=tg_folder,
        tpl_path=tpl_path,
        auto_gen_format=auto_gen_format,
        custom_phonemes_path=custom_phonemes_path,
        log_fn=log,
        normalize_key_fn=normalize_key,
        load_template_lines_fn=lambda path: load_template_oto_lines(
            path,
            require_utf8=False,
            mode_label="일본어 모드",
        ),
    )
    final_lines = []
    custom_map = ja_setup.custom_map
    tg_entries = ja_setup.tg_index.tg_entries
    file_groups = dict(ja_setup.file_groups)
    use_template = bool(ja_setup.use_template)

    def _resolve_tg_info(fname):
        return ja_setup.tg_index.resolve_tg_info(fname, log_fn=log)

    def _textgrid_missing_diagnostics(fname):
        return ja_setup.tg_index.textgrid_missing_diagnostics(fname)

    if not use_template:
        # 템플릿 없는 자동 생성 모드 (Auto-Generation)
        log(f"⚡ 템플릿 없음/미적합 → OpenUtau 호환 {auto_gen_format.upper()} 포맷 자동 에일리어스 생성 시작")
        file_groups = build_ja_auto_file_groups(
            tg_entries=tg_entries,
            auto_gen_format=auto_gen_format,
            log_fn=log,
            load_textgrid_fn=textgrid.TextGrid.fromFile,
            parse_filename_fn=parse_ja_filename,
            split_syllable_fn=split_ja_romaji_syllable,
            is_vowel_chain_fn=_is_vowel_chain_syllables,
        )

    processed = 0
    total = len(file_groups)
    wav_root_for_signal = os.path.dirname(os.path.abspath(tg_folder.rstrip("\\/")))
    wav_index_for_signal = {}
    try:
        if os.path.isdir(wav_root_for_signal):
            for fn in os.listdir(wav_root_for_signal):
                if fn.lower().endswith(".wav"):
                    nkey = normalize_key(fn)
                    if nkey:
                        wav_index_for_signal[nkey] = os.path.join(wav_root_for_signal, fn)
    except Exception:
        pass
    mel_cache_for_signal = {}

    for fname, lines in file_groups.items():
        file_ctx = prepare_file_context(
            fname=fname,
            lines=lines,
            resolve_tg_info_fn=_resolve_tg_info,
            missing_diagnostics_fn=_textgrid_missing_diagnostics,
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
            miss_diag = file_ctx.diagnostics
            log(
                f"📝 {fname}: TextGrid 없음 → 원본 유지 "
                f"(norm={miss_diag['lookup_norm']}, 후보={len(miss_diag['norm_candidates'])})"
            )
            if ja_mapping_debug_reason_logging:
                for p in miss_diag.get("candidate_paths", [])[:3]:
                    log(f"   ㄴ 후보 경로: {p}")
            _record_unset_lines("textgrid_missing", fname, lines, meta=miss_diag)
            final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
            processed += 1
            continue

        file_ctx = load_named_tiers(
            file_ctx,
            load_textgrid_fn=textgrid.TextGrid.fromFile,
            tier_predicate=lambda _tier: True,
        )
        if file_ctx.status == "textgrid_load_failed":
            log(f"⚠️ {fname}: TextGrid 로드 실패 → 원본 유지 ({file_ctx.error_message})")
            _record_unset_lines("textgrid_load_failed", fname, lines)
            final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
            processed += 1
            continue
        if file_ctx.status == "tier_missing":
            log(f"⚠️ {fname}: phones 티어 없음 → 원본 유지")
            _record_unset_lines("tier_missing", fname, lines)
            final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
            processed += 1
            continue

        real_wav_name = file_ctx.real_wav_name
        mel_ctx_for_file = file_ctx.mel_ctx_for_file
        wav_duration_ms = float(file_ctx.wav_duration_ms or 0.0)
        tg = file_ctx.tg
        ph_tier = file_ctx.phone_tier
        word_tier = file_ctx.word_tier

        try:
            if not ph_tier:
                log(f"⚠️ {fname}: phones 티어 없음 → 원본 유지")
                _record_unset_lines("tier_missing", fname, lines)
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue
            
            loop_prep = prepare_ja_loop_state(
                fname=fname,
                lines=lines,
                real_wav_name=real_wav_name,
                ph_tier=ph_tier,
                word_tier=word_tier,
                custom_map=custom_map,
                forced_format=forced_format,
                ja_mapping_words_fallback_enabled=ja_mapping_words_fallback_enabled,
                ja_mapping_spn_ratio_threshold=ja_mapping_spn_ratio_threshold,
                ja_mapping_min_vowel_phone_ratio=ja_mapping_min_vowel_phone_ratio,
                ja_mapping_confidence_threshold=ja_mapping_confidence_threshold,
                debug_reason_logging=ja_mapping_debug_reason_logging,
                log_fn=log,
                parse_filename_fn=parse_ja_filename,
                is_vowel_chain_fn=_is_vowel_chain_syllables,
                extract_cv_targets_fn=_extract_ja_cv_targets_from_lines,
                detect_alias_format_fn=detect_ja_alias_format,
                get_profile_fn=get_ja_profile_preset,
                collect_phone_quality_fn=_collect_phone_tier_quality,
                build_words_synth_phones_fn=_build_words_synth_phones,
                resolve_conf_threshold_fn=_resolve_ja_mapping_conf_threshold,
            )
            if loop_prep.status == "no_valid_alias":
                log(f"⚠️ {fname}: 유효한 에일리어스가 없어 원본 유지")
                _record_unset_lines("no_valid_alias", fname, lines)
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue
            if loop_prep.status == "empty_intervals":
                low_quality_reasons = list(loop_prep.low_quality_reasons)
                reason = "mapping_failed_empty_intervals"
                if loop_prep.low_phone_quality and "spn_heavy" in low_quality_reasons:
                    reason = "mapping_failed_spn_heavy"
                elif loop_prep.low_phone_quality and any(r in {"insufficient_phones", "insufficient_vowel_phones"} for r in low_quality_reasons):
                    reason = "mapping_failed_insufficient_phones"
                if loop_prep.low_phone_quality and not loop_prep.wd_intervals:
                    reason = "mapping_failed_no_words_support"
                log(f"⚠️ {fname}: 음소 정보 없음 → 원본 유지")
                _record_unset_lines(reason, fname, lines, meta=loop_prep.meta)
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue

            alignment_ingest = build_ja_alignment_ingest(file_ctx, loop_prep)
            wd_intervals = alignment_ingest.words
            ph_intervals = alignment_ingest.phones
            filename_syllables = list(alignment_ingest.extra.get("filename_syllables") or [])
            cv_targets = list(alignment_ingest.extra.get("cv_targets") or [])
            sinsy_label_entries = list(alignment_ingest.extra.get("sinsy_label_entries") or [])
            detected_format = str(alignment_ingest.extra.get("detected_format") or "")
            format_type = str(alignment_ingest.extra.get("format_type") or "")
            ja_style_profile = alignment_ingest.extra.get("ja_style_profile")
            phone_quality = alignment_ingest.phone_quality
            low_quality_reasons = alignment_ingest.low_quality_reasons
            low_phone_quality = alignment_ingest.low_phone_quality
            forced_words_mapping = bool(alignment_ingest.extra.get("forced_words_mapping"))
            timeline_start_ms = float(alignment_ingest.timeline_meta.get("timeline_start_ms", 0.0) or 0.0)
            effective_end_ms = float(alignment_ingest.timeline_meta.get("effective_end_ms", 0.0) or 0.0)
            phone_spans_ms = list(alignment_ingest.timeline_meta.get("phone_spans_ms") or [])
            conf_th = float(alignment_ingest.extra.get("conf_th", 0.0) or 0.0)
            textgrid_trust_score = float(alignment_ingest.textgrid_trust_score or 0.0)
            textgrid_trust_tier = str(alignment_ingest.textgrid_trust_tier or "low")
            prefer_filename_sequence = bool(alignment_ingest.prefer_filename_sequence)
            spn_ratio = float(phone_quality.get("spn_ratio_in_phone_tier", 0.0) or 0.0)

            post_ctx = build_ja_postprocess_context(
                phone_spans_ms=phone_spans_ms,
                timeline_start_ms=timeline_start_ms,
                effective_end_ms=effective_end_ms,
                file_format=format_type,
                validate_fn=validate_oto_params,
                recenter_fn=_recenter_ja_params_around_pre,
                extract_cv_bounds_fn=_ja_extract_cv_bounds,
                cv_onset_class_fn=_ja_cv_onset_class,
                syllables_info=[],
                ja_style_enabled=ja_style_enabled,
                ja_style_profile=ja_style_profile,
                autotune_profile=autotune_profile,
                style_apply_fn=_apply_ja_style_profile,
                autotune_apply_fn=_apply_ja_autotune_profile,
            )

            if try_handle_ja_single_vowel_file(
                fname=fname,
                lines=lines,
                real_wav_name=real_wav_name,
                ph_intervals=ph_intervals,
                final_lines=final_lines,
                log_fn=log,
                record_unset_fn=_record_unset,
                post_ctx=post_ctx,
                alias_out_fn=_alias_out,
                apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line,
                generate_openutau=generate_openutau,
                generate_openutau_aliases_fn=generate_ja_openutau_aliases,
                alias_suffix=alias_suffix,
            ):
                processed += 1
                continue

            # === 다중 음절 처리: 1단계(순서 보존 후보) -> 2단계(세부 후보 보정) ===
            syllables_info = []
            used_filename_based = False
            used_alias_based = False
            filename_order_locked = False
            mapping_reason_code = "filename_token"
            base_score = -1.0
            alt_score = -1.0
            mapping_confidence_base = 1.0
            mapping_margin = 0.0
            mapping_tier = "high"
            syllable_confidence_by_idx = []

            sparse_phone_mode = bool(
                filename_syllables and len(ph_intervals) < max(4, len(filename_syllables) // 2)
            )
            words_tier_confidence = _estimate_words_tier_confidence(
                wd_intervals,
                expected_syllables=max(len(filename_syllables), len(cv_targets), len(wd_intervals)),
            )

            filename_based = _build_syllables_from_filename(ph_intervals, filename_syllables)
            alias_based = _build_ja_syllables_from_phone_nuclei(ph_intervals, cv_targets) if cv_targets else None
            linear_filename_based = None
            if (not filename_based) and filename_syllables:
                linear_filename_based = _build_ja_linear_syllables_from_phones(ph_intervals, filename_syllables)

            words_sparse_based = None
            words_tier_based = None
            if wd_intervals and sparse_phone_mode:
                words_sparse_based = []
                for w in wd_intervals:
                    w_start = float(w.minTime)
                    w_end = float(w.maxTime)
                    s_phones = _synthesize_word_phones(w.mark.strip(), w_start, w_end)
                    words_sparse_based.append({
                        'word': w.mark.strip().lower(),
                        'start_time': w_start,
                        'end_time': w_end,
                        'phones': s_phones
                    })
            elif wd_intervals:
                words_tier_based = []
                for w in wd_intervals:
                    w_start = float(w.minTime)
                    w_end = float(w.maxTime)
                    s_phones = [p for p in ph_intervals if p.minTime >= (w_start - 0.01) and p.maxTime <= (w_end + 0.01)]
                    words_tier_based.append({
                        'word': w.mark.strip().lower(),
                        'start_time': w_start,
                        'end_time': w_end,
                        'phones': s_phones
                    })

            mapping_candidates = []

            def _append_candidate(
                name,
                infos,
                *,
                use_filename=False,
                use_alias=False,
                lock_order=False,
                linear=False,
                order_preserving=False,
            ):
                if not infos:
                    return
                eval_info = _evaluate_ja_mapping_candidate(
                    infos,
                    cv_targets,
                    wd_intervals=wd_intervals,
                    words_tier_confidence=words_tier_confidence,
                )
                if eval_info is None:
                    return
                mapping_candidates.append({
                    "name": name,
                    "infos": infos,
                    "use_filename": bool(use_filename),
                    "use_alias": bool(use_alias),
                    "lock_order": bool(lock_order),
                    "linear_fallback": bool(linear),
                    "order_preserving": bool(order_preserving),
                    **eval_info,
                })

            _append_candidate(
                "filename_token",
                filename_based,
                use_filename=True,
                lock_order=bool(format_type in {"cvvc", "cv"}),
                order_preserving=True,
            )
            _append_candidate(
                "filename_linear_fallback",
                linear_filename_based if format_type in {"cvvc", "cv"} else None,
                use_filename=True,
                lock_order=bool(format_type in {"cvvc", "cv"}),
                linear=True,
                order_preserving=True,
            )
            _append_candidate(
                "alias_phone_fallback",
                alias_based,
                use_alias=True,
                lock_order=False,
                order_preserving=False,
            )
            _append_candidate(
                "words_sparse_synth",
                words_sparse_based,
                use_filename=False,
                lock_order=False,
                order_preserving=False,
            )
            _append_candidate(
                "words_tier",
                words_tier_based,
                use_filename=False,
                lock_order=False,
                order_preserving=False,
            )

            if not mapping_candidates:
                # words 티어가 없고 파일명 기반 분할도 실패한 경우 fallback
                phones_split_based = []
                current_phones = []
                for p in ph_intervals:
                    mark = p.mark.strip().lower()
                    if current_phones and mark in ['a', 'i', 'ɯ', 'e', 'o', 'ɴ', 'cl']:
                        if mark in ['ɴ', 'cl'] and len(current_phones) > 0:
                            current_phones.append(p)
                            phones_split_based.append({
                                'word': ''.join(cp.mark.strip() for cp in current_phones),
                                'start_time': current_phones[0].minTime,
                                'end_time': current_phones[-1].maxTime,
                                'phones': list(current_phones)
                            })
                            current_phones = []
                            continue
                        if current_phones:
                            phones_split_based.append({
                                'word': ''.join(cp.mark.strip() for cp in current_phones),
                                'start_time': current_phones[0].minTime,
                                'end_time': current_phones[-1].maxTime,
                                'phones': list(current_phones)
                            })
                        current_phones = [p]
                    else:
                        current_phones.append(p)
                if current_phones:
                    phones_split_based.append({
                        'word': ''.join(cp.mark.strip() for cp in current_phones),
                        'start_time': current_phones[0].minTime,
                        'end_time': current_phones[-1].maxTime,
                        'phones': list(current_phones)
                    })
                _append_candidate(
                    "phones_split_fallback",
                    phones_split_based,
                    use_filename=False,
                    lock_order=False,
                    order_preserving=False,
                )

            selected_candidate, mapping_reason_code, candidate_by_name = select_primary_mapping_candidate(
                mapping_candidates,
                format_type=format_type,
                forced_words_mapping=forced_words_mapping,
                alias_candidate_name="alias_phone_fallback",
            )
            alias_candidate = candidate_by_name.get("alias_phone_fallback")

            # 2단계: 세부 후보 보정(alias 전환)은 고신뢰 상황에서만 허용.
            if selected_candidate:
                provisional_conf, _ = _estimate_ja_mapping_confidence(
                    phone_quality,
                    words_score=selected_candidate.get("score", 0.0),
                    alias_score=(alias_candidate.get("score", -1.0) if alias_candidate else -1.0),
                    used_filename_based=selected_candidate.get("use_filename", False),
                    used_alias_based=selected_candidate.get("use_alias", False),
                    forced_words_mapping=forced_words_mapping,
                    syllable_confidences=selected_candidate.get("syllable_confidences", []),
                    words_align_score=selected_candidate.get("words_align_score"),
                    words_tier_confidence=words_tier_confidence,
                )
                if (
                    alias_candidate
                    and alias_candidate is not selected_candidate
                    and not (format_type in {"cvvc", "cv"} and selected_candidate.get("lock_order"))
                    and provisional_conf >= conf_th
                ):
                    selected_candidate, promoted = maybe_promote_alias_candidate(
                        selected_candidate=selected_candidate,
                        alias_candidate=alias_candidate,
                        provisional_conf=provisional_conf,
                        conf_threshold=conf_th,
                        format_type=format_type,
                    )
                    if promoted:
                        selected_candidate = alias_candidate
                        mapping_reason_code = "alias_recover"
                if (
                    format_type == "cvvc"
                    and alias_candidate
                    and alias_candidate is not selected_candidate
                ):
                    force_alias, force_reason, youon_mismatch_ratio = _should_force_alias_for_ja_cvvc(
                        selected_candidate=selected_candidate,
                        alias_candidate=alias_candidate,
                        cv_targets=cv_targets,
                        low_phone_quality=low_phone_quality,
                    )
                    if force_alias:
                        selected_candidate = alias_candidate
                        mapping_reason_code = force_reason
                        if ja_mapping_debug_reason_logging:
                            log(
                                f"🧭 {fname}: CVVC 보수 전환 "
                                f"(reason={force_reason}, youon_mismatch={youon_mismatch_ratio:.2f})"
                            )

            if selected_candidate:
                syllables_info = list(selected_candidate.get("infos") or [])
                used_filename_based = bool(selected_candidate.get("use_filename"))
                used_alias_based = bool(selected_candidate.get("use_alias"))
                filename_order_locked = bool(
                    selected_candidate.get("lock_order") and format_type in {"cvvc", "cv"}
                )
                base_score = float(selected_candidate.get("score", -1.0))
                syllable_confidence_by_idx = list(selected_candidate.get("syllable_confidences", []) or [])
            else:
                syllables_info = []
                base_score = -1.0
                syllable_confidence_by_idx = []

            planned_cv_source = list(filename_syllables or cv_targets or [])
            ja_cv_plan = {"indices": None, "meta": {}, "source": ""}
            if planned_cv_source and sinsy_label_entries:
                ja_cv_plan = build_sinsy_guided_anchor_plan(
                    expected_tokens=planned_cv_source,
                    syllables_info=syllables_info,
                    label_entries=sinsy_label_entries,
                    normalize_expected_fn=_normalize_ja_syllable_token,
                    normalize_label_fn=_normalize_ja_syllable_token,
                    label_match_score_fn=lambda target, label: float(int(_ja_soft_cv_match_level(target, label) or 0) * 42.0),
                )
            if not ja_cv_plan.get("indices"):
                ja_cv_plan = _build_ja_cv_anchor_plan_v2(
                    planned_cv_source,
                    syllables_info,
                ) if planned_cv_source else {"indices": None, "meta": {}}
            ja_planned_cv_indices = ja_cv_plan.get("indices")
            ja_anchor_graph = build_adjacent_anchor_graph(ja_planned_cv_indices)
            ja_plan_policy = resolve_plan_policy(
                alignment_trust=textgrid_trust_score,
                plan_meta=ja_cv_plan.get("meta"),
                expected_count=len(planned_cv_source),
                planned_count=len(ja_planned_cv_indices or []),
                format_type=format_type,
                prefer_sequence=prefer_filename_sequence,
            )

            alt_score = float(alias_candidate.get("score", -1.0)) if alias_candidate else (-1.0 if cv_targets else 0.0)
            if not cv_targets and base_score < 0.0:
                base_score = 0.0

            mapping_confidence_base, mapping_margin = _estimate_ja_mapping_confidence(
                phone_quality,
                words_score=base_score,
                alias_score=alt_score,
                used_filename_based=used_filename_based,
                used_alias_based=used_alias_based,
                forced_words_mapping=forced_words_mapping,
                syllable_confidences=syllable_confidence_by_idx,
                words_align_score=(selected_candidate.get("words_align_score") if selected_candidate else None),
                words_tier_confidence=words_tier_confidence,
            )
            runtime_policy = resolve_runtime_mapping_policy(
                ingest_snapshot=alignment_ingest,
                plan_policy=ja_plan_policy,
                mapping_confidence=mapping_confidence_base,
                mapping_margin=mapping_margin,
                conf_threshold=conf_th,
                format_type=format_type,
                score_a=base_score,
                score_b=alt_score,
                sequence_lock_formats={"cvvc", "cv"},
                abstain_formats={"cvvc", "vcv", "cv"},
            )
            if sinsy_label_entries:
                plan_source = str(ja_cv_plan.get("source") or "")
                if plan_source != "sinsy_labels":
                    log(
                        f"🛡️ {fname}: sinsy 라벨이 있지만 planner에 적용되지 않음 "
                        f"(source={plan_source or 'fallback'})"
                    )
                else:
                    plan_margin = float((ja_cv_plan.get("meta") or {}).get("margin", 0.0) or 0.0)
                    row_margin_floor = float(runtime_policy.get("row_margin_floor", 6.0))
                    if plan_margin < row_margin_floor:
                        log(
                            f"🛡️ {fname}: sinsy planner margin 낮음 "
                            f"(margin={plan_margin:.1f} < {row_margin_floor:.1f})"
                        )
            mapping_confidence_base = float(runtime_policy.get("mapping_confidence", mapping_confidence_base))
            mapping_tier = str(runtime_policy.get("mapping_tier", "low"))

            # 일본어 CVVC/CV는 저신뢰일 때 파일명 순서를 기준 축으로 고정.
            if (
                bool(runtime_policy.get("force_sequence_lock"))
                and (not filename_order_locked)
            ):
                fallback_candidate = candidate_by_name.get("filename_linear_fallback") or candidate_by_name.get("filename_token")
                if fallback_candidate is not None and fallback_candidate is not selected_candidate:
                    selected_candidate = fallback_candidate
                    syllables_info = list(selected_candidate.get("infos") or [])
                    used_filename_based = bool(selected_candidate.get("use_filename"))
                    used_alias_based = bool(selected_candidate.get("use_alias"))
                    filename_order_locked = True
                    base_score = float(selected_candidate.get("score", base_score))
                    syllable_confidence_by_idx = list(selected_candidate.get("syllable_confidences", []) or [])
                    mapping_reason_code = (
                        "filename_linear_low_conf"
                        if selected_candidate.get("name") == "filename_linear_fallback"
                        else "filename_low_conf"
                    )
                    mapping_confidence_base, mapping_margin = _estimate_ja_mapping_confidence(
                        phone_quality,
                        words_score=base_score,
                        alias_score=alt_score,
                        used_filename_based=used_filename_based,
                        used_alias_based=used_alias_based,
                        forced_words_mapping=forced_words_mapping,
                        syllable_confidences=syllable_confidence_by_idx,
                        words_align_score=selected_candidate.get("words_align_score"),
                        words_tier_confidence=words_tier_confidence,
                    )
                    ja_cv_plan = {"indices": None, "meta": {}, "source": ""}
                    if planned_cv_source and sinsy_label_entries:
                        ja_cv_plan = build_sinsy_guided_anchor_plan(
                            expected_tokens=planned_cv_source,
                            syllables_info=syllables_info,
                            label_entries=sinsy_label_entries,
                            normalize_expected_fn=_normalize_ja_syllable_token,
                            normalize_label_fn=_normalize_ja_syllable_token,
                            label_match_score_fn=lambda target, label: float(int(_ja_soft_cv_match_level(target, label) or 0) * 42.0),
                        )
                    if not ja_cv_plan.get("indices"):
                        ja_cv_plan = _build_ja_cv_anchor_plan_v2(
                            planned_cv_source,
                            syllables_info,
                        ) if planned_cv_source else {"indices": None, "meta": {}}
                    ja_planned_cv_indices = ja_cv_plan.get("indices")
                    ja_anchor_graph = build_adjacent_anchor_graph(ja_planned_cv_indices)
                    ja_plan_policy = resolve_plan_policy(
                        alignment_trust=textgrid_trust_score,
                        plan_meta=ja_cv_plan.get("meta"),
                        expected_count=len(planned_cv_source),
                        planned_count=len(ja_planned_cv_indices or []),
                        format_type=format_type,
                        prefer_sequence=True,
                    )
                    runtime_policy = resolve_runtime_mapping_policy(
                        ingest_snapshot=alignment_ingest,
                        plan_policy=ja_plan_policy,
                        mapping_confidence=mapping_confidence_base,
                        mapping_margin=mapping_margin,
                        conf_threshold=conf_th,
                        format_type=format_type,
                        score_a=base_score,
                        score_b=alt_score,
                        sequence_lock_formats={"cvvc", "cv"},
                        abstain_formats={"cvvc", "vcv", "cv"},
                        prefer_sequence=True,
                    )
                    mapping_confidence_base = float(runtime_policy.get("mapping_confidence", mapping_confidence_base))
                    mapping_tier = str(runtime_policy.get("mapping_tier", "low"))
                    log(
                        f"🧭 {fname}: 일본어 TextGrid 신뢰도 {textgrid_trust_tier.upper()} "
                        f"(conf={mapping_confidence_base:.2f}, trust={textgrid_trust_score:.2f}) "
                        f"→ 파일명 기준 매핑 고정"
                    )

            if selected_candidate:
                if mapping_reason_code in {"filename_words_lock", "filename_words_linear_lock"}:
                    log(
                        f"🧭 {fname}: words 합성 phone 기반 매핑 고정 "
                        f"({len(filename_syllables)}음절, spn_ratio={spn_ratio:.2f})"
                    )
                elif mapping_reason_code == "alias_words_fallback":
                    log(
                        f"🧭 {fname}: words 합성 phone 기반 filename 실패 → alias/phone 매핑 사용 "
                        f"({len(cv_targets)}음절)"
                    )
                elif mapping_reason_code == "filename_token":
                    log(f"🧭 {fname}: 파일명 우선 음절 매핑 사용 ({len(filename_syllables)}음절)")
                elif mapping_reason_code == "filename_linear_fallback":
                    log(f"🧭 {fname}: 파일명 선형 fallback 음절 매핑 사용 ({len(filename_syllables)}음절)")
                elif mapping_reason_code == "alias_phone_fallback":
                    log(f"🧭 {fname}: 파일명 매핑 실패 → alias/phone 기반 음절 매핑 사용 ({len(cv_targets)}음절)")
                elif mapping_reason_code == "words_sparse_synth":
                    log(f"🧭 {fname}: phones 희소 감지({len(ph_intervals)}개) → words 기반 합성 phone 매핑 사용")
                elif mapping_reason_code == "words_tier":
                    log(f"🧭 {fname}: words 티어 기반 음절 매핑 사용 ({len(wd_intervals)}구간)")
                elif mapping_reason_code == "phones_split_fallback":
                    log(f"🧭 {fname}: phones split fallback 음절 매핑 사용")
                elif mapping_reason_code == "alias_recover":
                    log(
                        f"🧭 {fname}: 매핑 이탈 보정 적용 "
                        f"(base={base_score:.1f}, corrected={alt_score:.1f})"
                    )
                elif mapping_reason_code in {
                    "alias_cvvc_length_mismatch",
                    "alias_cvvc_youon_mismatch",
                    "alias_cvvc_low_phone_quality",
                }:
                    log(
                        f"🧭 {fname}: CVVC 보수 가드로 alias/phone 매핑 전환 "
                        f"(base={base_score:.1f}, corrected={alt_score:.1f}, reason={mapping_reason_code})"
                    )

            if ja_mapping_debug_reason_logging and mapping_confidence_base < conf_th:
                log(
                    f"🧭 {fname}: JA 매핑 신뢰도 낮음(conf={mapping_confidence_base:.2f}, "
                    f"tier={mapping_tier}, margin={mapping_margin:+.1f}, reason={mapping_reason_code})"
                )

            if syllables_info and alias_candidate and cv_targets:
                if format_type in {"cvvc", "cv"} and filename_order_locked:
                    log(
                        f"🧭 {fname}: 파일명 순서 잠금 유지 "
                        f"(base={base_score:.1f}, corrected={alt_score:.1f}, tier={mapping_tier})"
                    )
                elif mapping_reason_code not in {
                    "alias_recover",
                    "alias_cvvc_length_mismatch",
                    "alias_cvvc_youon_mismatch",
                    "alias_cvvc_low_phone_quality",
                }:
                    log(
                        f"🧭 {fname}: TextGrid 매핑 유지 "
                        f"(base={base_score:.1f}, corrected={alt_score:.1f})"
                    )

            if selected_candidate:
                candidate_rows = []
                for cand in sorted(mapping_candidates, key=lambda c: c.get("objective", -10**9), reverse=True)[:5]:
                    candidate_rows.append({
                        "name": str(cand.get("name", "")),
                        "score": float(cand.get("score", 0.0) or 0.0),
                        "objective": float(cand.get("objective", 0.0) or 0.0),
                        "order_preserving": bool(cand.get("order_preserving")),
                        "lock_order": bool(cand.get("lock_order")),
                        "mean_syll_conf": float(cand.get("mean_syll_conf", 0.0) or 0.0),
                    })
                _append_mapping_trace(
                    {
                        "event": "ja_mapping_candidate",
                        "file": str(fname or ""),
                        "format_type": str(format_type or ""),
                        "mapping_reason_code": str(mapping_reason_code or ""),
                        "mapping_tier": str(mapping_tier or ""),
                        "mapping_confidence": float(mapping_confidence_base or 0.0),
                        "mapping_margin": float(mapping_margin or 0.0),
                        "filename_order_locked": bool(filename_order_locked),
                        "forced_words_mapping": bool(forced_words_mapping),
                        "selected_candidate": str(selected_candidate.get("name", "")),
                        "filename_syllables": list(filename_syllables or []),
                        "cv_targets": list(cv_targets or []),
                        "candidate_rows": candidate_rows,
                    }
                )

            if not syllables_info or any(len(s['phones']) == 0 for s in syllables_info):
                fail_reason = "mapping_failed"
                if "spn_heavy" in low_quality_reasons:
                    fail_reason = "mapping_failed_spn_heavy"
                elif "insufficient_phones" in low_quality_reasons or "insufficient_vowel_phones" in low_quality_reasons:
                    fail_reason = "mapping_failed_insufficient_phones"
                elif low_phone_quality and not wd_intervals:
                    fail_reason = "mapping_failed_no_words_support"
                elif not ph_intervals:
                    fail_reason = "mapping_failed_empty_intervals"
                log(f"⚠️ {fname}: 음소-음절 매핑 실패 → 원본 유지")
                _record_unset_lines(
                    fail_reason,
                    fname,
                    lines,
                    meta={
                        "diag_hint": (
                            f"spn_ratio={spn_ratio:.2f}; "
                            f"conf={mapping_confidence_base:.2f}; reason={mapping_reason_code}"
                        ),
                        "phone_quality": phone_quality,
                        "forced_words_mapping": forced_words_mapping,
                        "mapping_confidence": mapping_confidence_base,
                        "mapping_reason_code": mapping_reason_code,
                        "mapping_tier": mapping_tier,
                    },
                )
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue

            if bool(runtime_policy.get("should_abstain")):
                log(
                    f"⚠️ {fname}: JA v2 planner abstain "
                    f"(trust={textgrid_trust_score:.2f}, coverage={float(ja_plan_policy.get('coverage', 0.0)):.2f}, "
                    f"margin={float(ja_plan_policy.get('margin', 0.0)):.1f}) → 원본 유지"
                )
                _record_unset_lines(
                    "mapping_v2_abstain",
                    fname,
                    lines,
                    meta={
                        "mapping_confidence": mapping_confidence_base,
                        "mapping_reason_code": mapping_reason_code,
                        "mapping_tier": mapping_tier,
                        "plan_policy": dict(ja_plan_policy or {}),
                    },
                )
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue

            lines_for_mapping = lines
            if format_type in ('cvvc', 'cv'):
                lines_for_mapping, expand_stats = _expand_vcv_lines_for_cvvc(
                    lines,
                    custom_map=custom_map,
                    include_bridge=(format_type == 'cvvc'),
                )
                if expand_stats.get("converted", 0) > 0:
                    log(
                        f"🧩 {fname}: VCV→{format_type.upper()} 전개 "
                        f"(원본 {len(lines)}줄 → 전개 {len(lines_for_mapping)}줄, "
                        f"bridge {expand_stats.get('added_bridge', 0)}개)"
                    )

            def _estimate_ja_cv_anchor(idx):
                return _estimate_ja_cv_anchor_v2(
                    idx,
                    syllables_info,
                    extract_cv_bounds_fn=_ja_extract_cv_bounds,
                    cv_offset_and_pre_fn=_ja_cv_offset_and_pre,
                    adaptive_overlap_fn=_adaptive_ja_overlap,
                    validate_fn=lambda o, c, cut, p, v: validate_oto_params(
                        o,
                        c,
                        cut,
                        p,
                        v,
                        alias_type="cv",
                    ),
                )

            def _compute_ja_vc_from_adjacent_cv(prev_cv, next_cv, alias_type, c_char, bridge_profile):
                return _compute_ja_vc_from_adjacent_cv_v2(
                    prev_cv,
                    next_cv,
                    alias_type=alias_type,
                    c_char=c_char,
                    bridge_profile=(bridge_profile or JA_CVVC_BRIDGE_TIMING.get("default", {})),
                    plosive_consonants=JA_PLOSIVE_CONSONANTS,
                    validate_fn=lambda o, c, cut, p, v: validate_oto_params(
                        o,
                        c,
                        cut,
                        p,
                        v,
                        alias_type=alias_type,
                    ),
                )

            cv_anchor_by_idx = {
                i: _estimate_ja_cv_anchor(i)
                for i in range(len(syllables_info))
            }
            realized_cv_anchor_by_idx = {}

            post_ctx.syllables_info = syllables_info

            local_conf_lock_threshold = max(0.46, min(conf_th - 0.08, 0.62))
            local_low_conf_lock_logged = set()

            # CV 순서 카운터로 리스트 순서 우선 매핑
            current_w_idx = 0
            cv_seq_idx = 0
            vc_seq_idx = 0
            last_vcv_mapped_idx = -1
            stable_vcv_seq_idx = 0
            ja_order_locked_format = format_type in {"cvvc", "cv"}
            ja_cvvc_occurrence_source = (
                filename_syllables
                if (ja_order_locked_format and filename_syllables)
                else [_syllable_info_token(s) for s in (syllables_info or [])]
            )
            ja_cvvc_occurrence_map = (
                _build_ja_cvvc_occurrence_map(ja_cvvc_occurrence_source)
                if ja_order_locked_format
                else None
            )
            ja_cvvc_occurrence_state = {}
            for line_num, line in enumerate(lines_for_mapping):
                parts = line.split('=', 1)
                if len(parts) < 2:
                    _record_unset("malformed_line", fname, line)
                    final_lines.append(apply_suffix_to_oto_line(line, alias_suffix))
                    continue
                alias = parts[1].split(',')[0].strip()
                if not alias:
                    # 빈 alias 항목(=,offset,...)은 누락 없이 보존
                    _record_unset("empty_alias", fname, line)
                    preserved = f"{real_wav_name}={parts[1]}"
                    final_lines.append(apply_suffix_to_oto_line(preserved, alias_suffix))
                    continue
                base_shape = _extract_base_timing_shape(line)
                source_profile = _parse_oto_line_profile(line)

                alias_type = _classify_alias_cached(alias)
                onset_hint_local = ""

                family_state = build_ja_alias_family_state(
                    alias=alias,
                    alias_type=alias_type,
                    format_type=format_type,
                    is_vowel_token_fn=_is_ja_vowel_token,
                    ja_consonants=JA_CONSONANTS,
                )
                alias_type = family_state.alias_type
                is_vc = family_state.is_vc
                is_vcv = family_state.is_vcv
                is_cv_head = family_state.is_cv_head
                c_char = ""
                vc_prev_anchor = None
                vc_next_anchor = None

                if try_handle_ja_br_alias(
                    alias=alias,
                    alias_type=alias_type,
                    ph_intervals=ph_intervals,
                    real_wav_name=real_wav_name,
                    final_lines=final_lines,
                    post_ctx=post_ctx,
                    alias_out_fn=_alias_out,
                    generate_openutau=generate_openutau,
                    generate_openutau_aliases_fn=generate_ja_openutau_aliases,
                ):
                    continue

                handled_tail_breath, current_w_idx = try_handle_ja_tail_breath_alias(
                    alias=alias,
                    state=family_state,
                    current_w_idx=current_w_idx,
                    syllables_info=syllables_info,
                    real_wav_name=real_wav_name,
                    final_lines=final_lines,
                    post_ctx=post_ctx,
                    alias_out_fn=_alias_out,
                    generate_openutau=generate_openutau,
                    generate_openutau_aliases_fn=generate_ja_openutau_aliases,
                )
                if handled_tail_breath:
                    continue

                # === VCV 연속음 처리 ===
                if is_vcv:
                    vcv_mapping = _select_ja_vcv_mapping_v2(
                        alias=alias,
                        fname=fname,
                        format_type=format_type,
                        stable_vcv_seq_idx=stable_vcv_seq_idx,
                        cv_seq_idx=cv_seq_idx,
                        syllables_info=syllables_info,
                        ja_planned_cv_indices=ja_planned_cv_indices,
                        last_vcv_mapped_idx=last_vcv_mapped_idx,
                        mapping_tier=mapping_tier,
                        mapping_reason_code=mapping_reason_code,
                        mapping_confidence_base=mapping_confidence_base,
                        filename_order_locked=filename_order_locked,
                        syllable_confidence_by_idx=syllable_confidence_by_idx,
                        log_fn=log,
                        debug_logging=ja_mapping_debug_reason_logging,
                        resolve_planned_cv_index_fn=_resolve_ja_planned_cv_index,
                        select_vcv_syllable_index_fn=_select_vcv_syllable_index,
                        extract_target_syllable_fn=_extract_vcv_target_syllable,
                        find_exact_target_index_fn=_find_ja_exact_target_index,
                        normalize_syllable_token_fn=_normalize_ja_syllable_token,
                        syllable_info_token_fn=_syllable_info_token,
                        split_syllable_fn=split_ja_romaji_syllable,
                        ja_vowels=JA_VOWELS,
                        find_vowel_match_index_fn=_find_ja_cv_vowel_match_index,
                        prefer_vcv_candidate_index_fn=_prefer_vcv_candidate_index,
                        should_trace_mapping_decision_fn=should_trace_mapping_decision,
                        build_mapping_trace_record_fn=_build_ja_mapping_trace_record,
                        append_mapping_trace_fn=_append_mapping_trace,
                        decide_cv_row_abstain_fn=decide_cv_row_abstain,
                        is_cv_syllable_active_fn=_is_ja_cv_syllable_active,
                    )
                    expected_idx = int(vcv_mapping["expected_idx"])
                    mapped_idx = int(vcv_mapping["mapped_idx"])
                    row_abstain = dict(vcv_mapping["row_abstain"])
                    if row_abstain.get("should_skip"):
                        if ja_mapping_debug_reason_logging:
                            log(
                                f"🛡️ {fname}: JA 행 생성 스킵 "
                                f"({row_abstain.get('reason')}, {alias})"
                            )
                        _record_unset(
                            str(row_abstain.get("reason") or "row_abstain"),
                            fname,
                            line,
                            meta={"diag_hint": row_abstain.get("diag_hint", "")},
                        )
                        continue

                    (
                        current_w_idx,
                        cv_seq_idx,
                        stable_vcv_seq_idx,
                        last_vcv_mapped_idx,
                    ) = _run_ja_vcv_row_v2(
                        syllables_info=syllables_info,
                        current_w_idx=current_w_idx,
                        stable_vcv_seq_idx=stable_vcv_seq_idx,
                        cv_seq_idx=cv_seq_idx,
                        mapped_idx=mapped_idx,
                        alias=alias,
                        base_shape=base_shape,
                        mel_ctx_for_file=mel_ctx_for_file,
                        onset_hint_local=onset_hint_local,
                        post_ctx=post_ctx,
                        real_wav_name=real_wav_name,
                        final_lines=final_lines,
                        generate_openutau=generate_openutau,
                        fname=fname,
                        log_fn=log,
                        prepare_cv_bounds_fn=_ja_extract_cv_bounds,
                        compute_vcv_params_fn=_compute_vcv_params_from_virtual_split,
                        soft_guard_fn=_apply_soft_mel_offset_cutoff_guard,
                        enforce_vcv_cv_entry_guard_fn=_enforce_vcv_cv_entry_guard,
                        extract_target_syllable_fn=_extract_vcv_target_syllable,
                        split_syllable_fn=split_ja_romaji_syllable,
                        is_n_bridge_fn=_ja_is_n_bridge_alias,
                        apply_anchor_lock_fn=_apply_ja_anchor_lock,
                        finalize_row_fn=_finalize_ja_row_v2,
                        row_builder_fn=_build_ja_alias_output_rows_v2,
                        generate_openutau_aliases_fn=generate_ja_openutau_aliases,
                        alias_out_fn=_alias_out,
                    )
                    continue

                # === 어두 CV (- a) 처리 ===
                if is_cv_head:
                    expected_seq_idx = stable_vcv_seq_idx if format_type == "vcv" else cv_seq_idx
                    if expected_seq_idx < len(syllables_info):
                        expected_idx = expected_seq_idx
                    else:
                        expected_idx = len(syllables_info) - 1
                    cvvc_order_soft_align = bool(
                        filename_order_locked
                        and format_type in {"cvvc", "cv"}
                        and mapping_tier != "low"
                    )
                    skip_cv_align = bool(
                        filename_order_locked
                        and format_type in {"cvvc", "cv"}
                        and mapping_tier == "low"
                    )
                    if (not skip_cv_align) and format_type in {"cvvc", "cv"} and syllable_confidence_by_idx:
                        conf_idx = max(0, min(expected_idx, len(syllable_confidence_by_idx) - 1))
                        local_conf = float(syllable_confidence_by_idx[conf_idx])
                        if local_conf < local_conf_lock_threshold:
                            skip_cv_align = True
                            if (
                                ja_mapping_debug_reason_logging
                                and conf_idx not in local_low_conf_lock_logged
                            ):
                                local_low_conf_lock_logged.add(conf_idx)
                                log(
                                    f"🧭 {fname}: CV_HEAD 음절 저신뢰(conf={local_conf:.2f}) "
                                    f"→ 순서 잠금({conf_idx + 1}번)"
                                )
                    target_tok = _extract_ja_cv_target_syllable(alias, alias_type="cv_head")
                    resynced_cv_head_exact = False
                    forced_cvvc_idx = _resolve_ja_forced_cv_index_v2(
                        alias=alias,
                        alias_type="cv_head",
                        target_tok=target_tok,
                        expected_seq_idx=expected_seq_idx,
                        expected_idx=expected_idx,
                        syllables_info=syllables_info,
                        planned_indices=ja_planned_cv_indices,
                        occurrence_map=ja_cvvc_occurrence_map or {},
                        occurrence_state=ja_cvvc_occurrence_state,
                        resolve_planned_cv_index_fn=_resolve_ja_planned_cv_index,
                        resolve_cvvc_occurrence_index_fn=_resolve_ja_cvvc_occurrence_index,
                        remap_forced_cv_index_fn=_remap_ja_forced_cv_index,
                        log_fn=log,
                        debug_logging=ja_mapping_debug_reason_logging,
                        fname=fname,
                    )
                    if forced_cvvc_idx is not None:
                        mapped_idx = max(0, min(int(forced_cvvc_idx), len(syllables_info) - 1))
                        if ja_mapping_debug_reason_logging and mapped_idx != expected_idx:
                            log(
                                f"🧭 {fname}: CV_HEAD occurrence 고정 "
                                f"{expected_idx + 1}->{mapped_idx + 1} ({alias})"
                            )
                    elif skip_cv_align:
                        mapped_idx = expected_idx
                        # CVVC/CV 파일명 순서 잠금에서도 요음/삽입 모음으로 인한 1칸 밀림은 보정.
                        if cvvc_order_soft_align and target_tok:
                            resync_idx = _find_ja_exact_target_index(
                                target_tok,
                                expected_idx,
                                syllables_info,
                                search_back=0,
                                search_fwd=2,
                            )
                            if resync_idx is None:
                                resync_idx = _find_ja_cv_vowel_match_index(
                                    target_tok,
                                    expected_idx,
                                    syllables_info,
                                    search_back=0,
                                    search_fwd=1,
                                )
                            if (
                                resync_idx is not None
                                and expected_idx < resync_idx <= (expected_idx + 1)
                            ):
                                expected_tok_trace = _syllable_info_token(syllables_info[expected_idx])
                                mapped_tok_trace = _syllable_info_token(syllables_info[resync_idx])
                                if _should_allow_ja_soft_forward_shift(target_tok, expected_tok_trace, mapped_tok_trace):
                                    mapped_idx = resync_idx
                                    log(
                                        f"🧭 {fname}: CV_HEAD 순서 잠금 미세 보정 "
                                        f"{expected_idx + 1}->{mapped_idx + 1} ({alias})"
                                    )
                    else:
                        mapped_idx = _select_ja_cv_syllable_index(
                            alias, expected_idx, syllables_info, alias_type="cv_head"
                        )
                        if target_tok:
                            mapped_tok = _syllable_info_token(syllables_info[mapped_idx])
                            _mo, mapped_vowel = split_ja_romaji_syllable(mapped_tok)
                            _to, target_vowel = split_ja_romaji_syllable(target_tok)
                            if target_vowel in JA_VOWELS and mapped_vowel and mapped_vowel != target_vowel:
                                fixed_idx = _find_ja_cv_vowel_match_index(
                                    target_tok,
                                    expected_idx,
                                    syllables_info,
                                    search_back=(2 if format_type == 'vcv' else 1),
                                    search_fwd=(3 if format_type == 'vcv' else 2),
                                )
                                if fixed_idx is not None:
                                    mapped_idx = fixed_idx
                            if format_type == 'vcv':
                                resync_idx = _find_ja_exact_target_index(
                                    target_tok,
                                    expected_idx,
                                    syllables_info,
                                    search_back=(0 if format_type == "vcv" else 3),
                                    search_fwd=2,
                                )
                                if resync_idx is not None and resync_idx != mapped_idx:
                                    log(
                                        f"🧭 {fname}: CV_HEAD 순서 드리프트 복구 "
                                        f"{expected_idx + 1}->{resync_idx + 1} ({alias})"
                                    )
                                    mapped_idx = resync_idx
                                    resynced_cv_head_exact = True
                        if mapped_idx < expected_idx:
                            if format_type != 'vcv' or (expected_idx - mapped_idx) > 3:
                                mapped_idx = expected_idx
                        if format_type == 'vcv' and mapped_idx > (expected_idx + 1):
                            if not (resynced_cv_head_exact and mapped_idx <= (expected_idx + 2)):
                                log(
                                    f"🛡️ {fname}: CV_HEAD 과도 점프 차단 "
                                    f"({expected_idx + 1}->{mapped_idx + 1}, {alias})"
                                )
                                mapped_idx = expected_idx
                        if format_type == "vcv":
                            mapped_idx = _prefer_vcv_candidate_index(
                                expected_idx,
                                mapped_idx,
                                target_tok,
                                syllables_info,
                                max_delta=1,
                            )
                        if format_type in {"cvvc", "cv"} and filename_order_locked:
                            if mapped_idx < expected_idx:
                                mapped_idx = expected_idx
                            elif format_type == "cvvc" and mapped_idx > expected_idx:
                                # CVVC는 순서 안정성을 우선한다.
                                # 1칸 전진은 target token과 exact match일 때만 허용한다.
                                allow_forward = False
                                if (
                                    mapped_idx == (expected_idx + 1)
                                    and target_tok
                                    and expected_idx >= 0
                                    and mapped_idx < len(syllables_info)
                                ):
                                    target_norm = _normalize_ja_syllable_token(target_tok)
                                    mapped_tok_norm = _normalize_ja_syllable_token(
                                        _syllable_info_token(syllables_info[mapped_idx])
                                    )
                                    expected_tok_norm = _normalize_ja_syllable_token(
                                        _syllable_info_token(syllables_info[expected_idx])
                                    )
                                    allow_forward = _should_allow_ja_soft_forward_shift(
                                        target_norm,
                                        expected_tok_norm,
                                        mapped_tok_norm,
                                    )
                                if not allow_forward:
                                    mapped_idx = expected_idx
                            elif mapped_idx > (expected_idx + 1):
                                mapped_idx = expected_idx + 1
                        ordered_idx = _clamp_ja_cv_index_to_order(
                            target_tok,
                            expected_idx,
                            mapped_idx,
                            syllables_info,
                            format_type=format_type,
                            filename_order_locked=filename_order_locked,
                            mapping_tier=mapping_tier,
                        )
                        if ordered_idx != mapped_idx:
                            log(
                                f"🛡️ {fname}: CV_HEAD 순서 고정 "
                                f"({mapped_idx + 1}->{ordered_idx + 1}, {alias})"
                            )
                            mapped_idx = ordered_idx
                        if mapped_idx != expected_idx and abs(mapped_idx - expected_idx) <= 1:
                            log(f"🧭 {fname}: CV 음절 정렬 보정 {expected_idx + 1}->{mapped_idx + 1} ({alias})")
                    if format_type == "cvvc":
                        active_idx = _remap_ja_cvvc_inactive_cv_index(
                            target_tok,
                            expected_idx,
                            mapped_idx,
                            syllables_info,
                            alias_type="cv_head",
                        )
                        if active_idx != mapped_idx and ja_mapping_debug_reason_logging:
                            log(
                                f"🛡️ {fname}: CV_HEAD 무음 매핑 회피 "
                                f"({mapped_idx + 1}->{active_idx + 1}, {alias})"
                            )
                        mapped_idx = active_idx
                    expected_tok_trace = _syllable_info_token(syllables_info[expected_idx])
                    mapped_tok_trace = _syllable_info_token(syllables_info[mapped_idx])
                    local_trace_conf = None
                    if syllable_confidence_by_idx:
                        conf_idx = max(0, min(expected_idx, len(syllable_confidence_by_idx) - 1))
                        local_trace_conf = float(syllable_confidence_by_idx[conf_idx])
                    if should_trace_mapping_decision(
                        mapping_tier=mapping_tier,
                        expected_idx=expected_idx,
                        mapped_idx=mapped_idx,
                        target_token=_normalize_ja_syllable_token(target_tok),
                        mapped_token=_normalize_ja_syllable_token(mapped_tok_trace),
                    ):
                        _append_mapping_trace(
                            _build_ja_mapping_trace_record(
                                fname=fname,
                                alias=alias,
                                alias_type="cv_head",
                                format_type=format_type,
                                target_tok=target_tok,
                                expected_idx=expected_idx,
                                mapped_idx=mapped_idx,
                                expected_tok=expected_tok_trace,
                                mapped_tok=mapped_tok_trace,
                                mapping_tier=mapping_tier,
                                mapping_reason_code=mapping_reason_code,
                                mapping_confidence=mapping_confidence_base,
                                filename_order_locked=filename_order_locked,
                                local_conf=local_trace_conf,
                            )
                        )
                    row_abstain = decide_cv_row_abstain(
                        alias_type="cv_head",
                        format_type=format_type,
                        candidate_idx=mapped_idx,
                        candidate_count=len(syllables_info),
                        confidence_margin=mapping_margin,
                        min_confidence_margin=runtime_policy.get("row_margin_floor"),
                        candidate_active=(
                            _is_ja_cv_syllable_active(syllables_info[mapped_idx], require_vowel=True)
                            if 0 <= mapped_idx < len(syllables_info)
                            else False
                        ),
                        active_only_formats={"cvvc", "cv"},
                        margin_formats={"cvvc", "cv"},
                    )
                    if row_abstain.get("should_skip"):
                        if ja_mapping_debug_reason_logging:
                            log(
                                f"🛡️ {fname}: JA 행 생성 스킵 "
                                f"({row_abstain.get('reason')}, {alias})"
                            )
                        _record_unset(
                            str(row_abstain.get("reason") or "row_abstain"),
                            fname,
                            line,
                            meta={"diag_hint": row_abstain.get("diag_hint", "")},
                        )
                        continue
                    current_w_idx = mapped_idx
                    if format_type == "vcv":
                        stable_vcv_seq_idx = min(stable_vcv_seq_idx + 1, max(len(syllables_info) - 1, 0))
                        cv_seq_idx = stable_vcv_seq_idx
                    else:
                        cv_seq_idx = current_w_idx + 1
                    if current_w_idx >= len(syllables_info):
                        current_w_idx = len(syllables_info) - 1
                        
                    curr_syl = syllables_info[current_w_idx]
                    curr_phones = curr_syl['phones']
                    if format_type == "cvvc" and not _is_ja_cv_syllable_active(curr_syl, require_vowel=True):
                        if ja_mapping_debug_reason_logging:
                            log(
                                f"🛡️ {fname}: CV_HEAD 무음/저활성 구간 스킵 "
                                f"(idx={current_w_idx + 1}, {alias})"
                            )
                        continue
                    
                    _run_ja_cv_head_row_v2(
                        curr_phones=curr_phones,
                        current_w_idx=current_w_idx,
                        alias=alias,
                        format_type=format_type,
                        base_shape=base_shape,
                        mel_ctx_for_file=mel_ctx_for_file,
                        post_ctx=post_ctx,
                        real_wav_name=real_wav_name,
                        final_lines=final_lines,
                        generate_openutau=generate_openutau,
                        fname=fname,
                        log_fn=log,
                        validate_fn=validate_oto_params,
                        extract_cv_bounds_fn=_ja_extract_cv_bounds,
                        cv_offset_and_pre_fn=_ja_cv_offset_and_pre,
                        adaptive_overlap_fn=_adaptive_ja_overlap,
                        soft_guard_fn=_apply_soft_mel_offset_cutoff_guard,
                        apply_base_shape_blend_fn=_apply_base_shape_blend,
                        apply_anchor_lock_fn=_apply_ja_anchor_lock,
                        target_vowel_from_alias_fn=_ja_target_vowel_from_alias,
                        pick_vowel_phone_fn=_ja_pick_vowel_phone,
                        build_anchor_record_fn=lambda current_idx, **kwargs: _maybe_build_ja_realized_cv_anchor_record_v2(
                            current_idx,
                            build_anchor_fn=_build_realized_cv_anchor_v2,
                            **kwargs,
                        ),
                        finalize_row_fn=_finalize_ja_row_v2,
                        row_builder_fn=_build_ja_alias_output_rows_v2,
                        generate_openutau_aliases_fn=generate_ja_openutau_aliases,
                        alias_out_fn=_alias_out,
                        build_guard_messages_fn=_build_ja_cv_guard_messages_v2,
                        anchor_store=realized_cv_anchor_by_idx,
                    )
                    continue

                # === 기존 CVC 매핑 ===
                if not is_vc:
                    # CV 에일리어스: 순서대로 다음 음절에 매핑
                    expected_seq_idx = stable_vcv_seq_idx if format_type == "vcv" else cv_seq_idx
                    if expected_seq_idx < len(syllables_info):
                        expected_idx = expected_seq_idx
                    else:
                        expected_idx = len(syllables_info) - 1
                    cvvc_order_soft_align = bool(
                        filename_order_locked
                        and format_type in {"cvvc", "cv"}
                        and mapping_tier != "low"
                    )
                    skip_cv_align = bool(
                        filename_order_locked
                        and format_type in {"cvvc", "cv"}
                        and mapping_tier == "low"
                    )
                    if (not skip_cv_align) and format_type in {"cvvc", "cv"} and syllable_confidence_by_idx:
                        conf_idx = max(0, min(expected_idx, len(syllable_confidence_by_idx) - 1))
                        local_conf = float(syllable_confidence_by_idx[conf_idx])
                        if local_conf < local_conf_lock_threshold:
                            skip_cv_align = True
                            if (
                                ja_mapping_debug_reason_logging
                                and conf_idx not in local_low_conf_lock_logged
                            ):
                                local_low_conf_lock_logged.add(conf_idx)
                                log(
                                    f"🧭 {fname}: CV 음절 저신뢰(conf={local_conf:.2f}) "
                                    f"→ 순서 잠금({conf_idx + 1}번)"
                                )
                    target_tok = _extract_ja_cv_target_syllable(alias, alias_type="cv")
                    resynced_cv_exact = False
                    forced_cvvc_idx = _resolve_ja_forced_cv_index_v2(
                        alias=alias,
                        alias_type="cv",
                        target_tok=target_tok,
                        expected_seq_idx=expected_seq_idx,
                        expected_idx=expected_idx,
                        syllables_info=syllables_info,
                        planned_indices=ja_planned_cv_indices,
                        occurrence_map=ja_cvvc_occurrence_map or {},
                        occurrence_state=ja_cvvc_occurrence_state,
                        resolve_planned_cv_index_fn=_resolve_ja_planned_cv_index,
                        resolve_cvvc_occurrence_index_fn=_resolve_ja_cvvc_occurrence_index,
                        remap_forced_cv_index_fn=_remap_ja_forced_cv_index,
                        log_fn=log,
                        debug_logging=ja_mapping_debug_reason_logging,
                        fname=fname,
                    )
                    if forced_cvvc_idx is not None:
                        mapped_idx = max(0, min(int(forced_cvvc_idx), len(syllables_info) - 1))
                        if ja_mapping_debug_reason_logging and mapped_idx != expected_idx:
                            log(
                                f"🧭 {fname}: CV occurrence 고정 "
                                f"{expected_idx + 1}->{mapped_idx + 1} ({alias})"
                            )
                    elif skip_cv_align:
                        mapped_idx = expected_idx
                        # CVVC/CV 파일명 순서 잠금에서도 요음/삽입 모음으로 인한 1칸 밀림은 보정.
                        if cvvc_order_soft_align and target_tok:
                            resync_idx = _find_ja_exact_target_index(
                                target_tok,
                                expected_idx,
                                syllables_info,
                                search_back=0,
                                search_fwd=2,
                            )
                            if resync_idx is None:
                                resync_idx = _find_ja_cv_vowel_match_index(
                                    target_tok,
                                    expected_idx,
                                    syllables_info,
                                    search_back=0,
                                    search_fwd=1,
                                )
                            if (
                                resync_idx is not None
                                and expected_idx < resync_idx <= (expected_idx + 1)
                            ):
                                expected_tok_trace = _syllable_info_token(syllables_info[expected_idx])
                                mapped_tok_trace = _syllable_info_token(syllables_info[resync_idx])
                                if _should_allow_ja_soft_forward_shift(target_tok, expected_tok_trace, mapped_tok_trace):
                                    mapped_idx = resync_idx
                                    log(
                                        f"🧭 {fname}: CV 순서 잠금 미세 보정 "
                                        f"{expected_idx + 1}->{mapped_idx + 1} ({alias})"
                                    )
                    else:
                        mapped_idx = _select_ja_cv_syllable_index(
                            alias, expected_idx, syllables_info, alias_type="cv"
                        )
                        if target_tok:
                            mapped_tok = _syllable_info_token(syllables_info[mapped_idx])
                            _mo, mapped_vowel = split_ja_romaji_syllable(mapped_tok)
                            _to, target_vowel = split_ja_romaji_syllable(target_tok)
                            if target_vowel in JA_VOWELS and mapped_vowel and mapped_vowel != target_vowel:
                                fixed_idx = _find_ja_cv_vowel_match_index(
                                    target_tok,
                                    expected_idx,
                                    syllables_info,
                                    search_back=(2 if format_type == 'vcv' else 1),
                                    search_fwd=(3 if format_type == 'vcv' else 2),
                                )
                                if fixed_idx is not None:
                                    mapped_idx = fixed_idx
                            if format_type == 'vcv':
                                resync_idx = _find_ja_exact_target_index(
                                    target_tok,
                                    expected_idx,
                                    syllables_info,
                                    search_back=(0 if format_type == "vcv" else 3),
                                    search_fwd=2,
                                )
                                if resync_idx is not None and resync_idx != mapped_idx:
                                    log(
                                        f"🧭 {fname}: CV 순서 드리프트 복구 "
                                        f"{expected_idx + 1}->{resync_idx + 1} ({alias})"
                                    )
                                    mapped_idx = resync_idx
                                    resynced_cv_exact = True
                        if mapped_idx < expected_idx:
                            if format_type != 'vcv' or (expected_idx - mapped_idx) > 3:
                                mapped_idx = expected_idx
                        if format_type == 'vcv' and mapped_idx > (expected_idx + 1):
                            if not (resynced_cv_exact and mapped_idx <= (expected_idx + 2)):
                                log(
                                    f"🛡️ {fname}: CV 과도 점프 차단 "
                                    f"({expected_idx + 1}->{mapped_idx + 1}, {alias})"
                                )
                                mapped_idx = expected_idx
                        if format_type == "vcv":
                            mapped_idx = _prefer_vcv_candidate_index(
                                expected_idx,
                                mapped_idx,
                                target_tok,
                                syllables_info,
                                max_delta=1,
                            )
                        if format_type in {"cvvc", "cv"} and filename_order_locked:
                            if mapped_idx < expected_idx:
                                mapped_idx = expected_idx
                            elif format_type == "cvvc" and mapped_idx > expected_idx:
                                # CVVC는 순서 안정성을 우선한다.
                                # 1칸 전진은 target token과 exact match일 때만 허용한다.
                                allow_forward = False
                                if (
                                    mapped_idx == (expected_idx + 1)
                                    and target_tok
                                    and expected_idx >= 0
                                    and mapped_idx < len(syllables_info)
                                ):
                                    target_norm = _normalize_ja_syllable_token(target_tok)
                                    mapped_tok_norm = _normalize_ja_syllable_token(
                                        _syllable_info_token(syllables_info[mapped_idx])
                                    )
                                    expected_tok_norm = _normalize_ja_syllable_token(
                                        _syllable_info_token(syllables_info[expected_idx])
                                    )
                                    allow_forward = _should_allow_ja_soft_forward_shift(
                                        target_norm,
                                        expected_tok_norm,
                                        mapped_tok_norm,
                                    )
                                if not allow_forward:
                                    mapped_idx = expected_idx
                            elif mapped_idx > (expected_idx + 1):
                                mapped_idx = expected_idx + 1
                        ordered_idx = _clamp_ja_cv_index_to_order(
                            target_tok,
                            expected_idx,
                            mapped_idx,
                            syllables_info,
                            format_type=format_type,
                            filename_order_locked=filename_order_locked,
                            mapping_tier=mapping_tier,
                        )
                        if ordered_idx != mapped_idx:
                            log(
                                f"🛡️ {fname}: CV 순서 고정 "
                                f"({mapped_idx + 1}->{ordered_idx + 1}, {alias})"
                            )
                            mapped_idx = ordered_idx
                        if mapped_idx != expected_idx and abs(mapped_idx - expected_idx) <= 1:
                            log(f"🧭 {fname}: CV 음절 정렬 보정 {expected_idx + 1}->{mapped_idx + 1} ({alias})")
                    if format_type == "cvvc":
                        active_idx = _remap_ja_cvvc_inactive_cv_index(
                            target_tok,
                            expected_idx,
                            mapped_idx,
                            syllables_info,
                            alias_type="cv",
                        )
                        if active_idx != mapped_idx and ja_mapping_debug_reason_logging:
                            log(
                                f"🛡️ {fname}: CV 무음 매핑 회피 "
                                f"({mapped_idx + 1}->{active_idx + 1}, {alias})"
                            )
                        mapped_idx = active_idx
                    expected_tok_trace = _syllable_info_token(syllables_info[expected_idx])
                    mapped_tok_trace = _syllable_info_token(syllables_info[mapped_idx])
                    local_trace_conf = None
                    if syllable_confidence_by_idx:
                        conf_idx = max(0, min(expected_idx, len(syllable_confidence_by_idx) - 1))
                        local_trace_conf = float(syllable_confidence_by_idx[conf_idx])
                    if should_trace_mapping_decision(
                        mapping_tier=mapping_tier,
                        expected_idx=expected_idx,
                        mapped_idx=mapped_idx,
                        target_token=_normalize_ja_syllable_token(target_tok),
                        mapped_token=_normalize_ja_syllable_token(mapped_tok_trace),
                    ):
                        _append_mapping_trace(
                            _build_ja_mapping_trace_record(
                                fname=fname,
                                alias=alias,
                                alias_type="cv",
                                format_type=format_type,
                                target_tok=target_tok,
                                expected_idx=expected_idx,
                                mapped_idx=mapped_idx,
                                expected_tok=expected_tok_trace,
                                mapped_tok=mapped_tok_trace,
                                mapping_tier=mapping_tier,
                                mapping_reason_code=mapping_reason_code,
                                mapping_confidence=mapping_confidence_base,
                                filename_order_locked=filename_order_locked,
                                local_conf=local_trace_conf,
                            )
                        )
                    current_w_idx = mapped_idx
                    if format_type == "vcv":
                        stable_vcv_seq_idx = min(stable_vcv_seq_idx + 1, max(len(syllables_info) - 1, 0))
                        cv_seq_idx = stable_vcv_seq_idx
                    else:
                        cv_seq_idx = current_w_idx + 1

                    if current_w_idx >= len(syllables_info):
                        current_w_idx = len(syllables_info) - 1

                    curr_syl = syllables_info[current_w_idx]
                    curr_phones = curr_syl['phones']

                    c_start, c_end, n_start, n_end = _ja_extract_cv_bounds(
                        curr_phones, alias_text=alias, alias_type="cv"
                    )

                    cv_vowel_len = n_end - n_start

                    c_hint = curr_phones[0].mark if curr_phones else ""
                    offset, pre = _ja_cv_offset_and_pre(
                        c_start,
                        c_end,
                        alias,
                        c_hint=c_hint,
                        alias_type="cv",
                        vowel_start=n_start,
                        vowel_end=n_end,
                    )
                    if pre < 10: pre = 10
                    onset_hint_local = c_hint
                    ovl = _adaptive_ja_overlap(pre, c_hint, mode="cv")

                    v_ref = max(cv_vowel_len, 120)
                    added_cons = min(max(v_ref * 0.45, 70), 180)
                    consonant = pre + added_cons
                    cutoff = -(consonant + max(cv_vowel_len * 0.25, 45))

                else:
                    # VC/VV 에일리어스: 이전 음절의 모음과 다음 음절의 자음/모음을 브릿지
                    # CV가 적거나 없는 리스트에서도 VC/VV가 순서대로 전진하도록 별도 인덱스 사용
                    local_bridge_prev_idx = max(cv_seq_idx - 1, vc_seq_idx)
                    if local_bridge_prev_idx >= len(syllables_info):
                        local_bridge_prev_idx = len(syllables_info) - 1
                    local_bridge_next_idx = local_bridge_prev_idx + 1
                    bridge_pair = resolve_bridge_anchor_pair(
                        ja_anchor_graph,
                        vc_seq_idx,
                        realized_anchor_by_idx=realized_cv_anchor_by_idx,
                        estimated_anchor_by_idx=cv_anchor_by_idx,
                        local_prev_idx=local_bridge_prev_idx,
                        local_next_idx=local_bridge_next_idx,
                    )
                    bridge_prev_idx = (
                        int(bridge_pair["prev_idx"])
                        if bridge_pair.get("prev_idx") is not None
                        else local_bridge_prev_idx
                    )
                    bridge_next_idx = (
                        int(bridge_pair["next_idx"])
                        if bridge_pair.get("next_idx") is not None
                        else local_bridge_next_idx
                    )
                    if alias_type == "vv":
                        vv_back = 1 if str(mapping_tier or "").strip().lower() == "low" else 2
                        vv_fwd = 1 if str(mapping_tier or "").strip().lower() == "low" else 3
                        pair_prev_idx = _find_ja_vv_pair_prev_index(
                            alias,
                            bridge_prev_idx,
                            syllables_info,
                            search_back=vv_back,
                            search_fwd=vv_fwd,
                        )
                        if pair_prev_idx is not None and pair_prev_idx != bridge_prev_idx:
                            log(
                                f"🛡️ {fname}: VV pair 매칭 보정 "
                                f"{bridge_prev_idx + 1}->{pair_prev_idx + 1} ({alias})"
                            )
                            bridge_prev_idx = int(pair_prev_idx)
                            bridge_next_idx = int(pair_prev_idx + 1)
                    current_w_idx = bridge_prev_idx
                    if vc_seq_idx < len(syllables_info) - 1:
                        vc_seq_idx += 1

                    curr_syl = syllables_info[current_w_idx]
                    curr_phones = curr_syl['phones']

                    v_start = curr_phones[-1].minTime * 1000
                    v_end = curr_phones[-1].maxTime * 1000
                    c_start = v_start
                    c_end = v_end

                    if bridge_next_idx < len(syllables_info):
                        next_syl = syllables_info[bridge_next_idx]
                        n_start = next_syl['phones'][0].minTime * 1000
                        n_end = next_syl['phones'][0].maxTime * 1000
                    else:
                        n_start = v_end
                        n_end = v_end + 100

                    vc_target = n_start if alias_type == 'vc' else n_end
                    boundary = min(vc_target, c_end + 260)
                    v_len = c_end - c_start
                    n_len = n_end - n_start
                    transition_gap = max(n_start - c_end, 0.0)
                    c_char = get_vc_consonant(alias)
                    bridge_profile = _get_ja_cvvc_bridge_profile(c_char) if alias_type == 'vc' else None
                    n_bridge = _ja_is_n_bridge_alias(alias, alias_type)
                    onset_cls = _ja_onset_class(c_char)

                    # VC는 pre 기준점을 약간 앞당겨(다음 자음 시작 직전) 박자 밀림을 완화한다.
                    if alias_type == 'vc':
                        pre_lead_mul = bridge_profile.get('pre_lead_mul', 0.35) if bridge_profile else 0.35
                        pre_lead_min = bridge_profile.get('pre_lead_min', 8.0) if bridge_profile else 8.0
                        pre_lead_max = bridge_profile.get('pre_lead_max', 32.0) if bridge_profile else 32.0
                        if n_bridge:
                            pre_lead_mul = 0.16 if onset_cls in {"voiced", "nasal"} else 0.20
                            pre_lead_min = 4.0
                            pre_lead_max = 14.0 if onset_cls in {"voiced", "nasal"} else 18.0
                        pre_lead = min(max(transition_gap * pre_lead_mul, pre_lead_min), pre_lead_max)
                        boundary = max(c_end + (2.0 if n_bridge else 8.0), boundary - pre_lead)

                    if alias_type == 'vc' and bridge_profile:
                        base_pad = bridge_profile.get('offset_pad', 86.0)
                        dyn_pad = base_pad + max(v_len - 140.0, 0.0) * bridge_profile.get('offset_len_mul', 0.08)
                        pad_lo = bridge_profile.get('offset_pad_min', 42.0)
                        pad_hi = min(240.0, max(v_len * 0.92, base_pad + 36.0))
                        offset_padding = _clamp_range(dyn_pad, pad_lo, pad_hi)
                        if v_len < offset_padding:
                            offset_padding = max(v_len * 0.78, bridge_profile.get('offset_pad_floor', 36.0))
                        if n_bridge:
                            pad_cap = 70.0 if onset_cls in {"voiced", "nasal"} else 82.0
                            offset_padding = min(offset_padding, max(v_len * 0.60, pad_cap))
                    else:
                        offset_padding = 180
                        if v_len < offset_padding:
                            offset_padding = max(v_len * 0.8, 50)

                    offset = boundary - offset_padding
                    pre = boundary - offset

                    ovl_mode = 'vv' if alias_type == 'vv' else 'vc'
                    ovl = _adaptive_ja_overlap(pre, c_char, mode=ovl_mode)

                    # VC/VV overlap은 앞 모음의 끝(c_end) 근처에 오도록 보정한다.
                    # 절대 위치 기준으로 맞춘 뒤, pre보다 작게 유지한다.
                    if pre > 0:
                        if alias_type == 'vc':
                            if bridge_profile:
                                tail_margin = bridge_profile.get('tail_margin_base', 10.0) + v_len * bridge_profile.get('tail_margin_mul', 0.05)
                                tail_margin = _clamp_range(tail_margin, 4.0, 24.0)
                                ovl_pre_margin = bridge_profile.get('ovl_pre_margin', 6.0)
                                ovl_min_ratio = bridge_profile.get('ovl_min_ratio', 0.40)
                                ovl_ratio = bridge_profile.get('ovl_ratio', 0.50)
                            else:
                                tail_margin = min(max(v_len * 0.08, 4.0), 18.0)
                                ovl_pre_margin = 6.0
                                ovl_min_ratio = 0.40
                                ovl_ratio = 0.50
                            target_ovl_abs = c_end - tail_margin
                            upper_ovl = max(pre - ovl_pre_margin, 0.0)
                            lower_ovl = min(pre * ovl_min_ratio, upper_ovl)
                            ovl_anchored = min(upper_ovl, max(lower_ovl, target_ovl_abs - offset))
                            ovl = _blend(ovl_anchored, pre * ovl_ratio, 0.28)
                            ovl = min(upper_ovl, max(lower_ovl, ovl))
                        elif alias_type == 'vv':
                            tail_margin = min(max(v_len * 0.12, 6.0), 22.0)
                            target_ovl_abs = c_end - tail_margin
                            upper_ovl = max(pre - 8.0, 0.0)
                            lower_ovl = min(pre * 0.52, upper_ovl)
                            ovl = min(upper_ovl, max(lower_ovl, target_ovl_abs - offset))

                    is_plosive = c_char in JA_PLOSIVE_CONSONANTS
                    # 다음 CV의 선행발성 기준점(대체로 다음 자음의 끝) 근처를 참조
                    next_cv_pre_rel = max(n_end - offset, pre + 20)
                    # 음소 경계 이상치(긴 무음/정렬 흔들림)로 인한 과도 확장 방지
                    next_cv_pre_rel = min(next_cv_pre_rel, pre + 260)
                    # 다음 자음 onset 기준점(VC 파열음 보호용)
                    next_c_onset_rel = max(n_start - offset, pre + 12)
                    next_c_onset_rel = min(next_c_onset_rel, pre + 220)

                    use_vcv_anchor = (
                        alias_type == 'vc'
                        and bridge_profile is not None
                        and format_type in ('cvvc', 'cv')
                        and detected_format == 'vcv'
                        and _is_reliable_base_profile(source_profile)
                    )

                    if use_vcv_anchor:
                        anchored = _compute_vc_params_from_vcv_anchor(
                            source_profile=source_profile,
                            prev_v_start=c_start,
                            prev_v_end=c_end,
                            next_c_start=n_start,
                            next_c_end=n_end,
                            bridge_profile=bridge_profile,
                        )
                        if anchored is not None:
                            offset, consonant, cutoff, pre, ovl = anchored
                        else:
                            use_vcv_anchor = False

                    use_cv_anchor_bridge = False
                    if (not use_vcv_anchor) and alias_type in ('vc', 'vv') and bridge_next_idx < len(syllables_info):
                        prev_cv_anchor = bridge_pair.get("prev_anchor") or realized_cv_anchor_by_idx.get(current_w_idx) or cv_anchor_by_idx.get(current_w_idx)
                        next_cv_anchor = bridge_pair.get("next_anchor") or realized_cv_anchor_by_idx.get(bridge_next_idx) or cv_anchor_by_idx.get(bridge_next_idx)
                        vc_prev_anchor = prev_cv_anchor
                        vc_next_anchor = next_cv_anchor
                        anchor_params = _compute_ja_vc_from_adjacent_cv(
                            prev_cv_anchor, next_cv_anchor, alias_type, c_char, bridge_profile
                        )
                        if anchor_params is not None:
                            offset, consonant, cutoff, pre, ovl = anchor_params
                            use_cv_anchor_bridge = True

                    if not use_cv_anchor_bridge:
                        if alias_type == 'vc' and bridge_profile and not use_vcv_anchor:
                            n_ref = max(n_len, 60.0)
                            cons_add = bridge_profile.get('cons_add_base', 36.0) + max(n_ref - 70.0, 0.0) * bridge_profile.get('cons_add_mul', 0.12)
                            cons_add = _clamp_range(
                                cons_add,
                                bridge_profile.get('cons_add_min', 20.0),
                                bridge_profile.get('cons_add_max', 68.0),
                            )
                            consonant = min(pre + cons_add, next_cv_pre_rel - bridge_profile.get('cons_to_next_margin', 8.0))
                            consonant = max(consonant, pre + bridge_profile.get('cons_floor', 18.0))

                            cut_add = bridge_profile.get('cut_add_base', 58.0) + max(n_ref - 70.0, 0.0) * bridge_profile.get('cut_add_mul', 0.20)
                            cut_add = _clamp_range(
                                cut_add,
                                bridge_profile.get('cut_add_min', 34.0),
                                bridge_profile.get('cut_add_max', 120.0),
                            )
                            cutoff_abs = max(
                                consonant + bridge_profile.get('cut_min_gap', 16.0),
                                pre + cut_add,
                            )
                            cutoff_abs = min(cutoff_abs, next_cv_pre_rel + bridge_profile.get('cut_to_next_allow', 22.0))
                            if cutoff_abs <= consonant + 8:
                                cutoff_abs = consonant + 10
                            cutoff = -cutoff_abs
                        elif is_plosive:
                            # 파열/파찰음은 VC 쪽에서 자음을 과감히 절단해 중복 파열을 방지
                            n_ref = max(n_len, 60)
                            added_cons = min(max(n_ref * 0.25, 18), 40)
                            consonant = pre + added_cons
                            cutoff_abs = pre + min(max(n_ref * 0.40, 28), 62)
                            if cutoff_abs <= consonant + 10:
                                cutoff_abs = consonant + 14
                            cutoff = -cutoff_abs
                        elif alias_type == 'vv':
                            # 모음-모음 전환은 뒤 모음의 입구를 넉넉히 유지
                            n_ref = max(n_len, 80)
                            added_cons = min(max(n_ref * 0.55, 55), 160)
                            consonant = min(pre + added_cons, next_cv_pre_rel - 10)
                            if consonant < pre + 25:
                                consonant = pre + 25
                            cutoff_abs = max(consonant + 24, next_cv_pre_rel - 6)
                            cutoff_abs = min(cutoff_abs, consonant + 140)
                            cutoff = -cutoff_abs
                        else:
                            # 비음/유음/마찰음은 연결 자음을 유지하되 다음 CV 직전에 컷
                            n_ref = max(n_len, 80)
                            added_cons = min(max(n_ref * 0.50, 42), 120)
                            consonant = min(pre + added_cons, next_cv_pre_rel - 10)
                            if consonant < pre + 25:
                                consonant = pre + 25
                            cutoff_abs = max(consonant + 20, next_cv_pre_rel - 8)
                            cutoff_abs = min(cutoff_abs, consonant + 95)
                            cutoff = -cutoff_abs

                    # CVVC 원리: VC 파열/파찰/치찰음은 다음 자음 onset 직전에서 정리해 중복 자음을 줄인다.
                    if alias_type == 'vc' and (
                        c_char in JA_PLOSIVE_CONSONANTS
                        or c_char in JA_SIBILANT_ONSETS
                        or c_char in JA_FRICATIVE_ONSETS
                    ):
                        onset_guard = max(next_c_onset_rel, pre + 14.0)
                        consonant = min(consonant, onset_guard - 7.0)
                        consonant = max(consonant, pre + 6.0)
                        cutoff_abs = max(consonant + 4.0, onset_guard - 1.0)
                        cutoff_cap = onset_guard - 0.8
                        cutoff_floor = consonant + 4.0
                        if cutoff_floor > cutoff_cap:
                            consonant = max(pre + 6.0, cutoff_cap - 4.0)
                            cutoff_floor = consonant + 4.0
                        if cutoff_floor > cutoff_cap:
                            cutoff_cap = cutoff_floor + 0.8
                        cutoff_abs = min(cutoff_abs, cutoff_cap)
                        cutoff_abs = max(cutoff_abs, cutoff_floor)
                        cutoff = -cutoff_abs

                    if alias_type == 'vc':
                        if vc_prev_anchor is None:
                            vc_prev_anchor = realized_cv_anchor_by_idx.get(current_w_idx) or cv_anchor_by_idx.get(current_w_idx)
                        if vc_next_anchor is None:
                            vc_next_anchor = realized_cv_anchor_by_idx.get(current_w_idx + 1) or cv_anchor_by_idx.get(current_w_idx + 1)

                _run_ja_general_row_v2(
                    final_lines=final_lines,
                    real_wav_name=real_wav_name,
                    alias=alias,
                    alias_type=alias_type,
                    format_type=format_type,
                    offset=offset,
                    consonant=consonant,
                    cutoff=cutoff,
                    pre=pre,
                    ovl=ovl,
                    c_end=c_end,
                    n_start=n_start,
                    n_end=n_end,
                    c_start=c_start,
                    current_w_idx=current_w_idx,
                    generate_openutau=generate_openutau,
                    fname=fname,
                    log_fn=log,
                    base_shape=base_shape,
                    mel_ctx_for_file=mel_ctx_for_file,
                    onset_hint_local=onset_hint_local,
                    mapping_tier=mapping_tier,
                    c_char=c_char,
                    vc_prev_anchor=vc_prev_anchor,
                    vc_next_anchor=vc_next_anchor,
                    post_ctx=post_ctx,
                    validate_fn=lambda o, c, cut, p, v, _atype=alias_type: validate_oto_params(
                        o,
                        c,
                        cut,
                        p,
                        v,
                        alias_type=_atype,
                    ),
                    soft_guard_fn=_apply_soft_mel_offset_cutoff_guard,
                    base_shape_blend_fn=_apply_base_shape_blend,
                    refine_ja_vc_fn=_refine_ja_vc_with_adjacent_cv,
                    onset_class_fn=_ja_onset_class,
                    is_n_bridge_fn=_ja_is_n_bridge_alias,
                    limit_pre_anchor_shift_fn=_limit_pre_anchor_shift,
                    apply_anchor_lock_fn=_apply_ja_anchor_lock,
                    anchor_lock_enabled_fn=is_anchor_lock_enabled,
                    enforce_cv_pre_anchor_guard_fn=_enforce_cv_pre_anchor_guard,
                    build_anchor_record_fn=lambda current_idx, **kwargs: _maybe_build_ja_realized_cv_anchor_record_v2(
                        current_idx,
                        build_anchor_fn=_build_realized_cv_anchor_v2,
                        **kwargs,
                    ),
                    finalize_row_fn=_finalize_ja_row_v2,
                    row_builder_fn=_build_ja_alias_output_rows_v2,
                    generate_openutau_aliases_fn=generate_ja_openutau_aliases,
                    alias_out_fn=_alias_out,
                    build_guard_messages_fn=_build_ja_cv_guard_messages_v2,
                    anchor_store=realized_cv_anchor_by_idx,
                )

            processed += 1

        except Exception as e:
            err_msg = f"에러 ({fname}): {e}"
            logger.error(err_msg)
            errors.append(err_msg)
            _record_unset_lines("file_exception", fname, lines)
            final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
            processed += 1

        if callback and (processed % 5 == 0 or processed == total):
            callback(f"OTO 생성 중... ({processed}/{total})")

    # === 미지정 모음 자동 생성 ===
    if gen_missing_vowels:
        log("🔍 미지정 모음 파일 자동 탐색 중...")
        template_aliases = set()
        for lines in file_groups.values():
            for line in lines:
                parts = line.split('=')
                if len(parts) > 1:
                    alias = parts[1].split(',')[0].strip().lower()
                    template_aliases.add(alias)

        for tg_info in tg_entries:
            base_filename = os.path.splitext(tg_info['real_name'])[0].lower()
            base_filename = re.sub(r'long$', '', base_filename)

            # 단독 모음 파일 감지
            tokens = re.split(r'[-_ ]+', base_filename)
            detected_vowel = None
            for token in reversed(tokens):
                clean_token = re.sub(r'[^a-z]', '', token)
                if clean_token in JA_VOWELS:
                    detected_vowel = clean_token
                    break

            if detected_vowel and detected_vowel not in template_aliases:
                try:
                    tg = textgrid.TextGrid.fromFile(tg_info['path'])
                    phone_tier = next((t for t in tg if isinstance(t, textgrid.IntervalTier) and t.name == 'phones'), None)
                    if not phone_tier:
                        continue
                    intervals = [i for i in phone_tier if i.mark.strip() not in ['', 'sil', 'spn', 'pau', 'sp']]

                    if len(intervals) == 1:
                        vowel = intervals[0]
                        v_start = vowel.minTime * 1000
                        v_end = vowel.maxTime * 1000
                        v_len = v_end - v_start

                        log(f"➕ 모음 추가 생성: {tg_info['real_name']} -> [{detected_vowel}]")
                        offset = v_start
                        pre = 0
                        ovl = 0
                        consonant = min(v_len * 0.25, 120)
                        cutoff = -(v_len * 0.8)

                        offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)

                        aliases_to_write = generate_ja_openutau_aliases(detected_vowel) if generate_openutau else [detected_vowel]
                        for a in aliases_to_write:
                            a2 = _alias_out(a)
                            new_line = f"{tg_info['real_name']}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                            final_lines.append(new_line)
                except:
                    continue

    finish_context = GeneratorFinishContext(
        log_fn=log,
        anchor_stats=anchor_stats,
        anchor_log_path=anchor_log_path,
        anchor_log_dir=_anchor_log_dir,
        cleanup_timing_jsonl=cleanup_timing_jsonl,
        timing_jsonl_prefix="timing_anchor_ja_",
    )

    # 저장
    try:
        write_oto_lines(out_path, final_lines)
        log(f"✅ 일본어 OTO 1차 생성 완료! 저장 경로: {out_path}")
    except Exception as e:
        err = f"❌ OTO 저장 실패: {e}"
        logger.error(err)
        errors.append(err)

    try:
        refresh_ja_autotune_profile_after_generation(
            JaAutotuneRefreshContext(
                out_path=out_path,
                profile_path=profile_path,
                had_profile_on_start=had_profile_on_start,
                custom_map=custom_map,
                log_fn=log,
                validate_fn=validate_oto_params,
            )
        )
    except Exception as e:
        log(f"[AutoTune] 내부 튜닝 프로파일 갱신 실패: {e}")

    run_ja_post_file_pipeline(
        JaPostFilePipelineContext(
            out_path=out_path,
            tg_folder=tg_folder,
            custom_map=custom_map,
            custom_phonemes_path=custom_phonemes_path,
            enable_ml_correction=enable_ml_correction,
            forced_format=forced_format,
            fallback_format=fallback_format,
            ml_policy=ml_policy,
            runtime_report=runtime_report,
            log_fn=log,
            validate_fn=validate_oto_params,
            classify_alias_fn=classify_ja_alias,
        )
    )

    if anchor_stats["anchor_locked_count"] > 0:
        log(
            "[AnchorLock] 요약: "
            f"anchor_locked_count={anchor_stats['anchor_locked_count']}, "
            f"cutoff_clamped_count={anchor_stats['cutoff_clamped_count']}, "
            f"vc_cutoff_leak_guard_count={anchor_stats['vc_cutoff_leak_guard_count']}"
        )
        log(f"[AnchorLock] 상세 로그: {anchor_log_path}")

    if ja_mapping_trace_logging and mapping_trace_records.count():
        try:
            trace_count = write_jsonl_records(mapping_trace_path, mapping_trace_records.records)
            log(f"[JA-Mapping] trace 로그: {mapping_trace_path}")
            log(f"[JA-Mapping] trace 건수: {trace_count}")
        except Exception as e:
            log(f"[JA-Mapping] trace 저장 스킵: {e}")

    finalize_generator_finish(finish_context)

    _log_unset_summary()
    return processed, total, errors

