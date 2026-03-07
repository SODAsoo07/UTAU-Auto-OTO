"""
日本語 CVVC OTO.ini 自動生成モジュール
- TextGrid 基準の OTO パラメータ計算
- CV, VC, VV, V タイプの自動分類と処理
- リスト順優先マッピング (List-Order-First) アルゴリズム
"""

import os
import re
import json
import math
import datetime
import logging
import unicodedata
from dataclasses import replace
from functools import lru_cache
import textgrid
import copy
from types import SimpleNamespace
from core.lab_generator import load_custom_phonemes
from core.ja_lab_generator import (
    romaji_to_ipa,
    parse_ja_filename,
    split_ja_romaji_syllable,
    KANA_COMBO_ROMAJI,
    KANA_SINGLE_ROMAJI,
)
from core.ja_oto_finalize import (
    _convert_ja_internal_cutoff_to_oto_field,
    _sanitize_ja_internal_params_for_wav_duration,
    sanitize_ja_oto_for_wav_duration,
)
from core.ja_oto_postprocess import (
    JaPostprocessContext,
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
from core.oto_profile_presets import get_ja_profile_preset

logger = logging.getLogger(__name__)

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
    _ja_left_bridge_token,
    _ja_soft_cv_match_level,
    _ja_syllable_tail,
    _normalize_ja_syllable_token,
    _score_ja_syllable_mapping,
    _select_ja_cv_syllable_index,
    _select_vcv_syllable_index,
    _syllable_info_token,
    _vcv_syllable_match_score,
)
from core.ja_oto_bridge import (
    _adaptive_ja_overlap,
    _clamp_ja_bridge_overlap,
    _ja_bridge_overlap_window,
    _ja_cv_offset_and_pre,
    _ja_cv_onset_class,
    _ja_extract_cv_bounds,
    _ja_mark_to_vowel,
    _ja_overlap_consonant_hint,
    _ja_onset_class,
    _ja_pick_vowel_phone,
    _ja_precenter_gap_targets,
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


def validate_oto_params(offset, consonant, cutoff, pre, ovl):
    """
    UTAU OTO 파라미터 순서 제약을 강제합니다.
    올바른 순서: offset → overlap → preutterance → consonant → cutoff
    """
    if offset < 0: offset = 0
    if pre < 0: pre = 0
    if ovl < 0: ovl = 0
    if consonant < 0: consonant = 0
    
    if ovl > pre:
        ovl = pre * 0.75
    if consonant < pre:
        consonant = pre + 30
    
    cutoff_abs = abs(cutoff)
    if cutoff_abs <= consonant:
        cutoff_abs = consonant + 50
    cutoff = -cutoff_abs
    
    return offset, consonant, cutoff, pre, ovl




def normalize_key(name):
    """파일명을 정규화된 키로 변환 (일본어/한국어/영문 공용)."""
    base = os.path.splitext(os.path.basename(str(name or "")))[0]
    if not base:
        return ""
    txt = unicodedata.normalize("NFKC", base).strip().lower()
    txt = _katakana_to_hiragana(txt)
    # 구분자/문장부호 제거: 동일 발음 파일명의 표기 차이를 흡수
    txt = re.sub(r"[\s_\-]+", "", txt)
    txt = re.sub(r"[`~!@#$%^&*()+={}\[\]|\\:;\"'<>,.?/・｡､。、「」『』（）［］｛｝]", "", txt)
    # 라틴/숫자/일본어/한글 이외 문자는 제거
    txt = re.sub(r"[^0-9a-z\u3041-\u3096\u30fc\u31f0-\u31ff\u3400-\u9fff\uac00-\ud7a3]+", "", txt)
    return txt


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


def _should_allow_ja_soft_forward_shift(target_tok, expected_tok, mapped_tok):
    target_norm = _normalize_ja_syllable_token(target_tok)
    expected_norm = _normalize_ja_syllable_token(expected_tok)
    mapped_norm = _normalize_ja_syllable_token(mapped_tok)
    if not target_norm or not mapped_norm:
        return False
    mapped_level = int(_ja_soft_cv_match_level(target_norm, mapped_norm) or 0)
    expected_level = int(_ja_soft_cv_match_level(target_norm, expected_norm) or 0)
    if mapped_level >= 3 and expected_level < 3:
        return True
    if mapped_level >= 2 and expected_level <= 1:
        return True
    return False


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
    return {
        "event": "ja_mapping_decision",
        "file": str(fname or ""),
        "alias": str(alias or ""),
        "alias_type": str(alias_type or ""),
        "format_type": str(format_type or ""),
        "target_token": target_norm,
        "expected_index": int(expected_idx),
        "mapped_index": int(mapped_idx),
        "delta": int(mapped_idx) - int(expected_idx),
        "expected_token": expected_norm,
        "mapped_token": mapped_norm,
        "expected_match_level": expected_match_level,
        "mapped_match_level": mapped_match_level,
        "mapping_tier": str(mapping_tier or ""),
        "mapping_reason_code": str(mapping_reason_code or ""),
        "mapping_confidence": float(mapping_confidence or 0.0),
        "filename_order_locked": bool(filename_order_locked),
        "local_confidence": None if local_conf is None else float(local_conf),
        "note": str(note or ""),
    }


def _find_ja_cv_vowel_match_index(target_tok, expected_idx, syllables_info, search_back=1, search_fwd=2):
    """
    target CV와 모음이 일치하는 음절을 기대 인덱스 근처에서 재탐색합니다.
    CV 한 칸 밀림(예: よ -> や) 보정용 안전장치.
    """
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
    expected_soft_match = _ja_soft_cv_match_level(
        target_tok,
        _syllable_info_token(syllables_info[e]),
    )

    lo = max(0, e - max(0, int(search_back)))
    hi = min(n - 1, e + max(0, int(search_fwd)))
    best_idx = None
    best_score = -10**9
    for i in range(lo, hi + 1):
        cand = _syllable_info_token(syllables_info[i])
        if not cand:
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


def _find_ja_exact_target_index(target_tok, expected_idx, syllables_info, search_back=3, search_fwd=3):
    """
    expected 인덱스 주변에서 target 토큰 exact match를 찾는다.
    목적: 연속 처리 중 누적 드리프트가 생겼을 때 제한적으로 역방향 재동기화.
    """
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
    # expected가 이미 exact면 재동기화 불필요
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
        # 거리 우선, 동점이면 순방향(i>e) 우선:
        # 반복 음절(ma ma ...)에서 앞 음절로 잘못 되돌아가는 현상을 줄인다.
        key = (abs(i - e), 0 if i > e else 1, i)
        if best_key is None or key < best_key:
            best_key = key
            best_idx = i
    return best_idx


def _prefer_vcv_candidate_index(expected_idx, mapped_idx, target_tok, syllables_info, max_delta=1):
    """
    VCV에서 expected 인덱스를 기본으로 유지하되, 아래 경우에만 제한적으로 보정 인덱스를 채택.
    - 보정 인덱스가 expected와 최대 1칸 이내
    - 보정 토큰은 target과 exact 일치
    - expected 토큰은 target과 불일치 (즉, 실제 보정 필요가 명확)
    """
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
        # 동일 토큰 반복 구간에서는 순서 안정 우선
        return e
    return m


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
    """
    VCV를 가상 `VC + CV`로 나눠 계산한 뒤 하나의 VCV 파라미터로 재조합합니다.
    """
    prev_v_len = max(float(prev_v_end) - float(prev_v_start), 40.0)
    curr_v_len = max(float(n_end) - float(c_boundary), 40.0)
    transition_gap = max(float(c_boundary) - float(prev_v_end), 0.0)

    target = _extract_vcv_target_syllable(alias)
    onset, _ = split_ja_romaji_syllable(target)
    onset = (onset or "").strip().lower()
    profile = _get_ja_cvvc_bridge_profile(onset)
    n_bridge = _ja_is_n_bridge_alias(alias, "vcv")
    onset_cls = _ja_onset_class(onset)

    pre_lead = _clamp_range(
        transition_gap * profile.get("pre_lead_mul", 0.35),
        profile.get("pre_lead_min", 8.0),
        profile.get("pre_lead_max", 30.0),
    )
    if n_bridge:
        pre_lead = _clamp_range(
            transition_gap * (0.16 if onset_cls in {"voiced", "nasal"} else 0.20),
            4.0,
            14.0 if onset_cls in {"voiced", "nasal"} else 18.0,
        )
    boundary = max(float(prev_v_end) + 6.0, float(c_boundary) - pre_lead)
    if n_bridge:
        # n+CV 브리지는 n 구간에만 박히지 않게 기준점을 조금 늦춰 다음 CV 진입을 더 포함한다.
        boundary = max(float(prev_v_end) + 2.0, float(c_boundary) - pre_lead)
        late_shift = _clamp_range(curr_v_len * 0.18, 8.0, 24.0)
        boundary = min(float(n_end) - 8.0, boundary + late_shift)

    base_pad = profile.get("offset_pad", 86.0)
    dyn_pad = base_pad + max(prev_v_len - 120.0, 0.0) * profile.get("offset_len_mul", 0.08)
    pad_lo = profile.get("offset_pad_min", 42.0)
    pad_hi = min(260.0, max(prev_v_len * 0.94, base_pad + 40.0))
    offset_padding = _clamp_range(dyn_pad, pad_lo, pad_hi)
    if prev_v_len < offset_padding:
        offset_padding = max(prev_v_len * 0.76, profile.get("offset_pad_floor", 36.0))
    if n_bridge:
        pad_cap = 70.0 if onset_cls in {"voiced", "nasal"} else 82.0
        offset_padding = min(offset_padding, max(prev_v_len * 0.60, pad_cap))

    offset = max(boundary - offset_padding, 0.0)
    # VCV는 pre가 지나치게 짧으면 후행 CV 체감이 약해져 기본 하한을 둔다.
    pre_floor = 44.0 if n_bridge else 36.0
    pre = max(boundary - offset, pre_floor)

    tail_margin = profile.get("tail_margin_base", 10.0) + prev_v_len * profile.get("tail_margin_mul", 0.05)
    tail_margin = _clamp_range(tail_margin, 4.0, 24.0)
    target_ovl_abs = float(prev_v_end) - tail_margin
    upper_ovl = max(pre - profile.get("ovl_pre_margin", 6.0), 0.0)
    lower_ovl = min(pre * profile.get("ovl_min_ratio", 0.40), upper_ovl)
    ovl_anchored = min(upper_ovl, max(lower_ovl, target_ovl_abs - offset))
    ovl = _blend(ovl_anchored, pre * profile.get("ovl_ratio", 0.50), 0.34)
    ovl = min(upper_ovl, max(lower_ovl, ovl))

    n_ref = max(curr_v_len, 70.0 if n_bridge else 60.0)
    cons_add = profile.get("cons_add_base", 36.0) + max(n_ref - 70.0, 0.0) * profile.get("cons_add_mul", 0.12)
    cons_add = _clamp_range(
        cons_add,
        profile.get("cons_add_min", 20.0),
        profile.get("cons_add_max", 68.0),
    )
    if n_bridge:
        cons_add = min(cons_add + 14.0, profile.get("cons_add_max", 68.0) + 18.0)
    consonant = pre + cons_add
    consonant = max(consonant, pre + profile.get("cons_floor", 18.0))
    consonant = min(consonant, pre + max(curr_v_len * 0.72, 96.0))

    cut_add = profile.get("cut_add_base", 58.0) + max(n_ref - 70.0, 0.0) * profile.get("cut_add_mul", 0.20)
    cut_add = _clamp_range(
        cut_add,
        profile.get("cut_add_min", 34.0),
        profile.get("cut_add_max", 120.0),
    )
    if n_bridge:
        cut_add = min(cut_add + 20.0, profile.get("cut_add_max", 120.0) + 28.0)
    cutoff_abs = max(
        consonant + profile.get("cut_min_gap", 16.0),
        pre + cut_add,
    )
    end_rel = max(float(n_end) - offset, pre + 40.0)
    cutoff_abs = min(cutoff_abs, end_rel + profile.get("cut_to_next_allow", 22.0))
    if cutoff_abs <= consonant + 10.0:
        cutoff_abs = consonant + 12.0
    cutoff = -cutoff_abs

    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
    return _apply_base_shape_blend(offset, consonant, cutoff, pre, ovl, base_shape, alias_type="vcv")


def _enforce_vcv_cv_entry_guard(offset, consonant, cutoff, pre, ovl, c_boundary, n_end, alias_text=""):
    """
    VCV(V-CV) 청감 보장 가드.
    - pre(abs)는 다음 C 끝 근처에 위치
    - consonant는 다음 V 진입부를 포함
    """
    offset = float(max(offset, 0.0))
    consonant = float(max(consonant, 0.0))
    pre = float(max(pre, 0.0))
    ovl = float(max(ovl, 0.0))
    c_boundary = float(c_boundary)
    n_end = float(max(n_end, c_boundary))

    vv_like = False
    n_bridge = _ja_is_n_bridge_alias(alias_text, "vcv")
    tok = _extract_vcv_target_syllable(alias_text)
    onset, _vowel = split_ja_romaji_syllable(tok)
    if not (onset or "").strip():
        vv_like = True

    pre_abs = offset + pre
    if vv_like:
        pre_abs = _clamp_range(pre_abs, max(c_boundary - 4.0, 0.0), c_boundary + 10.0)
        pre_floor = 52.0
    elif n_bridge:
        # n+CV는 pre를 조금 늦춰 다음 CV 자음/모음 진입이 체감되도록 고정한다.
        pre_abs = _clamp_range(pre_abs, max(c_boundary - 2.0, 0.0), c_boundary + 14.0)
        pre_floor = 50.0
    else:
        pre_abs = _clamp_range(pre_abs, max(c_boundary - 8.0, 0.0), c_boundary + 5.0)
        pre_floor = 44.0
    if (pre_abs - offset) < pre_floor:
        offset = max(pre_abs - pre_floor, 0.0)
    pre = max(pre_abs - offset, 0.0)

    vowel_start_rel = max(c_boundary - offset, pre + 8.0)
    vowel_len = max(n_end - c_boundary, 40.0)
    if vv_like:
        min_vowel_keep = _clamp_range(vowel_len * 0.30, 34.0, 88.0)
        min_cons_rel = max(pre + 72.0, vowel_start_rel + min_vowel_keep)
        max_cons_rel = max(vowel_start_rel + min_vowel_keep + 86.0, pre + 142.0)
    elif n_bridge:
        min_vowel_keep = _clamp_range(vowel_len * 0.30, 34.0, 96.0)
        min_cons_rel = max(pre + 76.0, vowel_start_rel + min_vowel_keep)
        max_cons_rel = max(vowel_start_rel + min_vowel_keep + 96.0, pre + 156.0)
    else:
        min_vowel_keep = _clamp_range(vowel_len * 0.22, 24.0, 76.0)
        min_cons_rel = max(pre + 64.0, vowel_start_rel + min_vowel_keep)
        max_cons_rel = max(vowel_start_rel + min_vowel_keep + 70.0, pre + 110.0)
    consonant = _clamp_range(consonant, min_cons_rel, max_cons_rel)

    cutoff_abs = abs(float(cutoff))
    min_cut_abs = consonant + (34.0 if n_bridge else 28.0)
    max_cut_abs = max((n_end - offset) + 86.0, consonant + 36.0)
    cutoff_abs = _clamp_range(cutoff_abs, min_cut_abs, max_cut_abs)
    cutoff = -cutoff_abs

    if vv_like:
        vv_gap_target = _clamp_range(vowel_len * 0.06, 5.0, 10.0)
        ovl = max(ovl, pre - vv_gap_target)
        ovl = min(ovl, max(pre - 2.0, 0.0))
    if ovl > pre:
        ovl = pre * 0.72
    return validate_oto_params(offset, consonant, cutoff, pre, ovl)


def _enforce_cv_pre_anchor_guard(offset, consonant, cutoff, pre, ovl, c_end_abs, alias_type="cv"):
    """
    CV/CV_HEAD의 pre(abs)를 자음 끝(c_end) 근처로 제한해 박자 앵커 흔들림을 줄인다.
    """
    offset = float(max(offset, 0.0))
    pre = float(max(pre, 0.0))
    consonant = float(max(consonant, 0.0))
    ovl = float(max(ovl, 0.0))
    c_end_abs = float(max(c_end_abs, 0.0))

    pre_abs = offset + pre
    win = 10.0 if alias_type == "cv_head" else 7.0
    target_abs = _clamp_range(pre_abs, max(c_end_abs - win, 0.0), c_end_abs + win)
    if abs(target_abs - pre_abs) > 2.0:
        new_offset = target_abs - pre
        if new_offset < 0.0:
            offset = 0.0
            pre = target_abs
        else:
            offset = new_offset

    if ovl > pre:
        ovl = pre * 0.72
    return validate_oto_params(offset, consonant, cutoff, pre, ovl)


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
    s = (line or "").strip()
    if not s or "=" not in s:
        return None
    wav, rest = s.split("=", 1)
    parts = rest.split(",")
    if len(parts) < 6:
        return None
    try:
        offset = float(parts[1].strip())
        consonant = float(parts[2].strip())
        cutoff = float(parts[3].strip())
        pre = float(parts[4].strip())
        ovl = float(parts[5].strip())
    except ValueError:
        return None
    return {
        "wav": wav.strip(),
        "alias": parts[0].strip(),
        "offset": offset,
        "cons": consonant,
        "cutoff": cutoff,
        "pre": pre,
        "ovl": ovl,
    }


def _extract_base_timing_shape(line):
    """
    base oto 한 줄에서 상대 타이밍 shape를 추출합니다.
    절대값(offset)은 직접 복사하지 않고, gap/ratio 중심으로 사용합니다.
    """
    p = _parse_oto_line_profile(line)
    if not p:
        return None
    pre = max(float(p["pre"]), 0.0)
    cons = max(float(p["cons"]), 0.0)
    cut_abs = abs(float(p["cutoff"]))
    ovl = max(float(p["ovl"]), 0.0)
    off = max(float(p["offset"]), 0.0)

    # auto-generated 0줄(템플릿 없음 fallback)은 shape로 쓰지 않는다.
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
        "ovl_ratio": _clamp_range(ovl_ratio, 0.04, 0.86),
    }


def _apply_base_shape_blend(offset, consonant, cutoff, pre, ovl, base_shape, alias_type="cv"):
    if os.environ.get("UTOA_DISABLE_BASE_SHAPE_BLEND", "").strip().lower() in {"1", "true", "yes", "on"}:
        return validate_oto_params(offset, consonant, cutoff, pre, ovl)
    if not base_shape:
        return validate_oto_params(offset, consonant, cutoff, pre, ovl)
    if alias_type == "cv_head":
        # 템플릿의 비정상 head 라인(cut_gap 과소/선행발음 과대)을 그대로 따라가면
        # 어두 CV 길이가 과도하게 짧아질 수 있어 블렌딩을 생략한다.
        src_cut_gap = float(base_shape.get("cut_gap", max(abs(cutoff) - consonant, 20.0)))
        src_pre = float(base_shape.get("pre", pre))
        if src_cut_gap < 90.0 or src_pre > 280.0:
            return validate_oto_params(offset, consonant, cutoff, pre, ovl)

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

    return validate_oto_params(offset, cons_new, cutoff_new, pre_new, ovl_new)


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

    return validate_oto_params(offset, consonant, cutoff, pre, ovl)


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
        return validate_oto_params(offset, consonant, cutoff, pre, ovl)

    cls = _ja_onset_class(c_char)
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
    cur_ovl_abs = float(offset) + float(ovl)
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
        cons_abs_target = onset_abs - 4.0
        cons_allow_after_onset = 6.0
        cut_gap_target = 12.0
        cut_allow_after_vowel = -2.0
        cut_allow_after_onset = 8.0
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
    cutoff_new = -(cut_abs_new - offset_new)

    return validate_oto_params(offset_new, consonant_new, cutoff_new, pre_new, ovl_abs_new - offset_new)


def _limit_pre_anchor_shift(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    *,
    pre_abs_before,
    max_shift_ms,
):
    """
    pre 절대 위치 이동량을 제한해 브리지 앵커 과보정을 방지한다.
    반환: (offset, consonant, cutoff, pre, ovl, pre_abs_after, clamped)
    """
    pre_abs_after = float(offset) + float(pre)
    delta = float(pre_abs_after - pre_abs_before)
    limit = max(0.0, float(max_shift_ms))
    if abs(delta) <= limit:
        out = validate_oto_params(offset, consonant, cutoff, pre, ovl)
        return (*out, float(out[0] + out[3]), False)

    limited_pre_abs = float(pre_abs_before) + (limit if delta > 0.0 else -limit)
    abs_shift = pre_abs_after - limited_pre_abs
    offset = float(offset) - float(abs_shift)
    out = validate_oto_params(offset, consonant, cutoff, pre, ovl)
    return (*out, float(out[0] + out[3]), True)


def _normalize_alias_for_profile(alias):
    a = re.sub(r"\s+", " ", (alias or "").strip().lower())
    # treat trailing suffix like _C4 as metadata
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


def _read_oto_rows_for_profile(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    text = _read_text_with_fallback(path)
    for raw in text.splitlines():
        p = _parse_oto_line_profile(raw)
        if not p:
            continue
        p["wav_norm"] = p["wav"].strip().lower()
        p["alias_norm"] = _normalize_alias_for_profile(p["alias"])
        rows.append(p)
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


def _profile_path_for_out(out_path):
    out_dir = os.path.dirname(os.path.abspath(out_path or "")) or os.getcwd()
    return os.path.join(out_dir, ".ja_oto_autotune_profile.json")


def _find_reference_oto(out_path):
    env_ref = os.environ.get("UTOA_JA_REF_OTO", "").strip()
    if env_ref and os.path.exists(env_ref):
        if os.path.abspath(env_ref) != os.path.abspath(out_path):
            return env_ref

    out_dir = os.path.dirname(os.path.abspath(out_path or "")) or os.getcwd()
    candidates = [
        "base_oto.ini",
        "oto.base.ini",
        "oto.manual.ini",
        "oto.correct.ini",
        "oto.reference.ini",
        "oto.gold.ini",
        "oto.human.ini",
        "oto_old.ini",
    ]
    for name in candidates:
        p = os.path.join(out_dir, name)
        if os.path.exists(p) and os.path.abspath(p) != os.path.abspath(out_path):
            return p
    return ""


def _train_ja_autotune_profile(auto_oto_path, ref_oto_path, custom_map=None):
    auto_rows = _read_oto_rows_for_profile(auto_oto_path)
    ref_rows = _read_oto_rows_for_profile(ref_oto_path)
    if not auto_rows or not ref_rows:
        return None

    auto_map = _occurrence_map(auto_rows)
    ref_map = _occurrence_map(ref_rows)

    fields = ["offset", "cons", "cutoff", "pre", "ovl"]
    buckets = {}
    matched = 0
    alias_type_cache = {}

    def _classify_cached(alias_text):
        t = alias_type_cache.get(alias_text)
        if t is None:
            t = classify_ja_alias(alias_text, custom_map)
            alias_type_cache[alias_text] = t
        return t

    for k, a_row in auto_map.items():
        r_row = ref_map.get(k)
        if not r_row:
            continue
        alias_for_cls = re.sub(r"_[A-Za-z0-9]{1,8}$", "", a_row["alias"])
        alias_type = _classify_cached(alias_for_cls)
        b = buckets.setdefault(alias_type, {f: [] for f in fields})
        for f in fields:
            b[f].append(r_row[f] - a_row[f])
        matched += 1

    if matched < 8:
        return None

    profile = {
        "version": 1,
        "language": "japanese",
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "matched_pairs": matched,
        "source_auto_name": os.path.basename(auto_oto_path),
        "source_ref_name": os.path.basename(ref_oto_path),
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


def _load_ja_autotune_profile(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "buckets" not in data:
            return None
        return data
    except Exception:
        return None


def _save_ja_autotune_profile(path, profile):
    if not path or not profile:
        return False
    try:
        data = dict(profile)
        # Remove direct path fields if present from legacy profile objects.
        data.pop("source_auto", None)
        data.pop("source_ref", None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def _apply_ja_autotune_profile(alias_type, offset, consonant, cutoff, pre, ovl, profile):
    if not profile:
        return offset, consonant, cutoff, pre, ovl
    buckets = profile.get("buckets") or {}
    if not isinstance(buckets, dict):
        return offset, consonant, cutoff, pre, ovl
    stat = buckets.get(alias_type)
    if not stat:
        return offset, consonant, cutoff, pre, ovl
    n = float(stat.get("n", 0))
    if n < 3:
        return offset, consonant, cutoff, pre, ovl
    # Gradually trust buckets as examples accumulate.
    w = min(1.0, max(0.25, n / 24.0))
    offset += _clamp(stat.get("offset", 0.0), 140.0) * w
    consonant += _clamp(stat.get("cons", 0.0), 160.0) * w
    cutoff += _clamp(stat.get("cutoff", 0.0), 180.0) * w
    pre += _clamp(stat.get("pre", 0.0), 140.0) * w
    ovl += _clamp(stat.get("ovl", 0.0), 100.0) * w
    return validate_oto_params(offset, consonant, cutoff, pre, ovl)


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
        0.0, cons_new, cutoff_new, pre_new, ovl_new
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

    return validate_oto_params(offset, cons_new, cutoff_new, pre_new, ovl_new)


def _apply_profile_to_oto_file(oto_path, profile, custom_map=None):
    if not oto_path or not os.path.exists(oto_path):
        return 0
    changed = 0
    out_lines = []
    alias_type_cache = {}

    def _classify_cached(alias_text):
        t = alias_type_cache.get(alias_text)
        if t is None:
            t = classify_ja_alias(alias_text, custom_map)
            alias_type_cache[alias_text] = t
        return t

    with open(oto_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            p = _parse_oto_line_profile(raw)
            if not p:
                out_lines.append(raw.rstrip("\n"))
                continue
            alias_for_cls = re.sub(r"_[A-Za-z0-9]{1,8}$", "", p["alias"])
            alias_type = _classify_cached(alias_for_cls)
            o2, c2, ct2, pr2, ov2 = _apply_ja_autotune_profile(
                alias_type, p["offset"], p["cons"], p["cutoff"], p["pre"], p["ovl"], profile
            )
            if (
                abs(o2 - p["offset"]) > 1e-6
                or abs(c2 - p["cons"]) > 1e-6
                or abs(ct2 - p["cutoff"]) > 1e-6
                or abs(pr2 - p["pre"]) > 1e-6
                or abs(ov2 - p["ovl"]) > 1e-6
            ):
                changed += 1
            out_lines.append(f"{p['wav']}={p['alias']},{o2:.2f},{c2:.2f},{ct2:.2f},{pr2:.2f},{ov2:.2f}")
    with open(oto_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")
    return changed


def _apply_ja_mel_refine_to_oto_file(oto_path, wav_dir, custom_map=None):
    """
    멜 에너지 골짜기 기준으로 일본어 CV 계열 cutoff 과연장을 후처리 보정합니다.
    """
    if os.environ.get("UTOA_DISABLE_MEL_REFINER", "").strip().lower() in {"1", "true", "yes", "on"}:
        return 0
    if not oto_path or not os.path.exists(oto_path) or not os.path.isdir(wav_dir):
        return 0
    try:
        import numpy as np
    except Exception:
        return 0

    # Reuse tested mel helpers from KR generator to keep behavior aligned.
    from core.oto_generator import _read_wav_mono_np, _mel_envelope, _find_wav_path_for_name

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
                nkey = normalize_key(fn)
                if nkey:
                    wav_index[nkey] = os.path.join(wav_dir, fn)
    except Exception:
        pass

    by_wav = {}
    for line_idx, row in parsed:
        by_wav.setdefault(row["wav"], []).append((line_idx, row))
    alias_type_cache = {}

    def _classify_cached(alias_text):
        t = alias_type_cache.get(alias_text)
        if t is None:
            t = classify_ja_alias(alias_text, custom_map)
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
            if alias_type not in {"cv", "cv_head", "vcv"}:
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

            # dB+mel 기반 선택을 우선, F0는 낮은 비중으로만 반영.
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

            if contrast < 0.11 and db_drop < 2.2:
                continue
            if valley_e > 0.40 and valley_db > (db_sil_th + 6.0):
                continue
            if valley_f0v > 0.72 and contrast < 0.16:
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
    return _apply_profile_to_oto_file(oto_path, profile, custom_map=custom_map)


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
    callback=None
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
    # KR 생성기와 동일한 신호 분석 헬퍼를 재사용해 멜 기반 가드 동작을 맞춘다.
    from core.oto_generator import (
        _read_wav_mono_np,
        _mel_envelope,
        _find_wav_path_for_name,
        _apply_soft_mel_offset_cutoff_guard,
    )

    # GUI 형식 지정:
    # - "자동 감지"면 강제 지정 없이 템플릿/에일리어스를 자동 판별
    # - 값이 지정되면 템플릿이 있어도 해당 형식을 우선 적용
    forced_format = None
    if auto_format:
        af = str(auto_format).strip().lower()
        if af.startswith('cvc'):
            forced_format = 'cvvc'
            fallback_format = 'cvvc'
        elif af == 'cv' or af.startswith('cv '):
            forced_format = 'cv'
            fallback_format = 'cv'
        elif af.startswith('cvvc'):
            forced_format = 'cvvc'
            fallback_format = 'cvvc'
        elif af.startswith('vcv'):
            forced_format = 'vcv'
            fallback_format = 'vcv'

    auto_gen_format = (fallback_format or "cvvc").strip().lower()
    if auto_gen_format not in {"cv", "cvvc", "vcv"}:
        log_msg = (
            f"⚠️ 자동 에일리어스 생성은 현재 CV/CVVC/VCV만 지원합니다. "
            f"{auto_gen_format.upper()} -> CVVC로 전환합니다."
        )
        if callback:
            callback(log_msg)
        auto_gen_format = "cvvc"
    elif auto_gen_format == "cv":
        auto_gen_format = "cvvc"

    # 매핑 옵션: 함수 인자 + 환경변수 병행 지원
    env_words_fallback = str(os.environ.get("UTOA_JA_MAPPING_WORDS_FALLBACK", "")).strip().lower()
    if env_words_fallback in {"0", "false", "off", "no"}:
        ja_mapping_words_fallback_enabled = False
    elif env_words_fallback in {"1", "true", "on", "yes"}:
        ja_mapping_words_fallback_enabled = True
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
    skipped_entries = []
    anchor_stats = {
        "anchor_locked_count": 0,
        "cutoff_clamped_count": 0,
        "vc_cutoff_leak_guard_count": 0,
    }
    _core_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(_core_dir)
    _anchor_log_dir = os.path.join(_project_dir, "logs")
    _anchor_log_name = f"timing_anchor_ja_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    anchor_log_path = os.path.join(_anchor_log_dir, _anchor_log_name)
    mapping_trace_name = f"ja_mapping_trace_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    mapping_trace_path = os.path.join(_anchor_log_dir, mapping_trace_name)
    mapping_trace_records = []

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
        fmt = str(format_type or "").strip().lower()
        if not is_anchor_lock_enabled("japanese", fmt):
            return offset, consonant, cutoff, pre, ovl
        alias_key = str(profile_alias_type or alias_type or "").strip().lower()
        profile = get_anchor_profile("japanese", fmt, alias_key, mode="rhythm_stable")
        if profile is None:
            return offset, consonant, cutoff, pre, ovl
        if fmt == "vcv" and alias_key in {"vcv", "vcv_n_bridge", "cv", "cv_head"}:
            profile = replace(
                profile,
                pre_window_before_ms=max(2.0, float(profile.pre_window_before_ms) - 1.0),
                pre_window_after_ms=float(profile.pre_window_after_ms) + 2.0,
                pre_floor_ms=float(profile.pre_floor_ms) + 2.0,
                blend_weight=min(0.66, float(profile.blend_weight) + 0.04),
            )
        # VCV의 CV/CV_HEAD/VCV 본체에서는 cutoff 강클램프가 과개입되기 쉬워
        # 리듬 앵커(pre) 중심으로만 고정하고 cutoff 상한은 원 계산/가드에 맡긴다.
        if fmt == "vcv" and alias_key in {"vcv", "vcv_vv_like", "vcv_n_bridge", "cv", "cv_head"}:
            profile = replace(
                profile,
                cut_to_next_onset_allow_ms=None,
                cut_to_next_vowel_allow_ms=None,
            )

        before = (float(offset), float(consonant), float(cutoff), float(pre), float(ovl))
        ctx = AnchorTimingContext(
            file_duration_ms=float(wav_duration_ms or 0.0),
            timeline_start_ms=float(timeline_start_ms),
            timeline_end_ms=float(effective_end_ms),
            anchor_abs_ms=float(anchor_abs_ms) if anchor_abs_ms is not None else None,
            next_onset_abs_ms=float(next_onset_abs_ms) if next_onset_abs_ms is not None else None,
            next_vowel_abs_ms=float(next_vowel_abs_ms) if next_vowel_abs_ms is not None else None,
            alias_type=str(alias_type or ""),
            language="japanese",
            format_type=fmt,
            mapping_confidence=1.0,
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
                    "language": "japanese",
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

    def _append_mapping_trace(record):
        if not ja_mapping_trace_logging:
            return
        if not isinstance(record, dict):
            return
        mapping_trace_records.append(record)

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
                extra = ""
                meta = it.get("meta") or {}
                if meta.get("diag_hint"):
                    extra = f" | {meta.get('diag_hint')}"
                log(f"    예시: {it['file']} | alias={alias_txt}{extra}")
                shown += 1
                if shown >= 5:
                    break

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

    if tpl_path and not os.path.exists(tpl_path):
        log(f"⚠️ 템플릿 파일을 찾을 수 없습니다: {tpl_path}")
        log(f"⚡ OpenUtau 호환 {auto_gen_format.upper()} 자동 에일리어스 생성으로 전환합니다.")
        tpl_path = ""

    # 템플릿 읽기
    template_lines = []
    if tpl_path:
        lines, detected_enc, warning, err = load_template_oto_lines(
            tpl_path,
            require_utf8=False,
            mode_label="일본어 모드",
        )
        if err:
            log(f"{err}")
            log(f"⚡ 템플릿 로드 실패로 OpenUtau 호환 {auto_gen_format.upper()} 자동 에일리어스 생성으로 전환합니다.")
            lines = []
        if warning:
            log(warning)
        template_lines = lines or []

    # TextGrid 맵핑 (정확 매칭 + 정규화 후보 매칭)
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
            if info['norm_key']:
                tg_norm_map.setdefault(info['norm_key'], []).append(info)

    def _resolve_tg_info(fname):
        """템플릿의 wav 파일명을 TextGrid 엔트리에 안정적으로 매핑합니다."""
        wav_name = os.path.basename((fname or '').strip())
        base_lower = os.path.splitext(wav_name)[0].lower()
        if base_lower in tg_exact_map:
            return tg_exact_map[base_lower]

        norm_name = normalize_key(wav_name)
        if not norm_name:
            return None
        candidates = tg_norm_map.get(norm_name, [])
        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            exact_name = wav_name.lower()
            same_name = [c for c in candidates if c['real_name'].lower() == exact_name]
            if len(same_name) == 1:
                return same_name[0]
            log(f"⚠️ 파일명 매핑 충돌: {wav_name} (정규화 키 {norm_name}, 후보 {len(candidates)}개) → 원본 파일명 유지")
            return None
        return None

    def _textgrid_missing_diagnostics(fname):
        wav_name = os.path.basename((fname or "").strip())
        base = os.path.splitext(wav_name)[0]
        norm = normalize_key(wav_name)
        base_candidates = {base}
        if base.startswith("_"):
            base_candidates.add(base[1:])
        else:
            base_candidates.add("_" + base)
        candidate_paths = [os.path.join(tg_folder, c + ".TextGrid") for c in sorted(base_candidates)]
        norm_candidates = [c.get("path", "") for c in (tg_norm_map.get(norm, []) or [])][:5]
        return {
            "wav_basename": wav_name,
            "lookup_base": base,
            "lookup_norm": norm,
            "candidate_paths": candidate_paths,
            "norm_candidates": norm_candidates,
            "diag_hint": f"norm={norm}, candidates={len(norm_candidates)}",
        }

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

    # 커스텀 맵 로드
    custom_map = load_custom_phonemes(custom_phonemes_path)

    # 템플릿 사용 (일반 모드)
    use_template = bool(template_lines)
    if use_template:
        t_match, t_total, t_ratio = _template_match_stats(template_lines)
        if t_total == 0 or t_match == 0 or t_ratio < 0.25:
            log(
                f"⚠️ 템플릿-TextGrid 매칭률 낮음 ({t_match}/{t_total}, {t_ratio:.1%}) "
                f"→ OpenUtau 호환 {auto_gen_format.upper()} 자동 에일리어스 생성으로 전환"
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
        # 템플릿 없는 자동 생성 모드 (Auto-Generation)
        log(f"⚡ 템플릿 없음/미적합 → OpenUtau 호환 {auto_gen_format.upper()} 포맷 자동 에일리어스 생성 시작")
        file_groups = {}

        def _resolve_vowel_onset(syl):
            onset, vowel = split_ja_romaji_syllable(syl)
            if vowel in JA_VOWELS:
                return vowel, onset
            if syl in ['n', 'nn', 'xn', 'm']:
                return 'n', ''
            return 'a', onset
        
        for tg_info in tg_entries:
            tg_path = tg_info['path']
            real_name = tg_info['real_name']
            
            try:
                tg = textgrid.TextGrid.fromFile(tg_path)
                # 단모음/연단음 포맷 확인 (filename 기반)
                base_name = os.path.splitext(real_name)[0]
                syllables = parse_ja_filename(base_name)
                
                if not syllables:
                    continue
                    
                lines = []
                is_long = base_name.lower().endswith('long') or len(syllables) == 1
                local_format = auto_gen_format
                if _is_vowel_chain_syllables(syllables):
                    local_format = 'vcv'
                    log(f"🧭 {real_name}: 모음 연속음 파일 감지 → 자동 생성 포맷 VCV 적용")
                
                if is_long:
                    # 단모음/연단음 포맷
                    for syl in syllables:
                        lines.append(f"{real_name}={syl},0,0,0,0,0")
                else:
                    # 다중 음절
                    for idx, syl in enumerate(syllables):
                        # 자음 모음 분리 (단순 휴리스틱)
                        vowel, onset = _resolve_vowel_onset(syl)
                            
                        
                        if local_format == 'vcv':
                            if idx == 0:
                                lines.append(f"{real_name}=- {syl},0,0,0,0,0")
                            else:
                                prev_syl = syllables[idx-1]
                                prev_vowel, _ = _resolve_vowel_onset(prev_syl)
                                    
                                lines.append(f"{real_name}={prev_vowel} {syl},0,0,0,0,0")
                                
                        elif local_format == 'cvc' or local_format == 'cvvc':
                            # CVVC / CVC 공통: 기본 CV
                            lines.append(f"{real_name}={syl},0,0,0,0,0")
                            
                            if idx > 0 and local_format == 'cvvc':
                                # 이전 모음 -> 현재 자음 VC 브릿지
                                prev_syl = syllables[idx-1]
                                prev_vowel, _ = _resolve_vowel_onset(prev_syl)
                                
                                # 현재 초성 자음 추출
                                cur_start_cons = onset
                                
                                if cur_start_cons:
                                    lines.append(f"{real_name}={prev_vowel} {cur_start_cons},0,0,0,0,0")
                                else:
                                    # 앞 음절 모음 -> 뒤 음절 모음 (VV)
                                    lines.append(f"{real_name}={prev_vowel} {vowel},0,0,0,0,0")
                
                if lines:
                    file_groups[real_name] = lines

            except Exception as e:
                log(f"⚠️ {real_name}: 자동 템플릿 생성 실패 ({e})")

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
        tg_info = _resolve_tg_info(fname)

        if not tg_info:
            miss_diag = _textgrid_missing_diagnostics(fname)
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
                try:
                    audio_sig, sr_sig = _read_wav_mono_np(wav_path_for_signal)
                    if sr_sig:
                        wav_duration_ms = (len(audio_sig) / float(sr_sig)) * 1000.0
                except Exception:
                    wav_duration_ms = 0.0
        
        try:
            tg = textgrid.TextGrid.fromFile(tg_path)
            
            ph_tier = None
            word_tier = None
            for t in tg:
                if hasattr(t, 'name') and t.name == 'phones':
                    ph_tier = t
                elif hasattr(t, 'name') and t.name == 'words':
                    word_tier = t
            
            if not ph_tier:
                log(f"⚠️ {fname}: phones 티어 없음 → 원본 유지")
                _record_unset_lines("tier_missing", fname, lines)
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue
            
            silence_marks = {'', 'sil', 'spn', 'pau', 'sp'}
            phone_silence_marks = {'', 'sil', 'pau', 'sp'}
            wd_intervals = []
            if word_tier:
                wd_intervals = [i for i in word_tier if i.mark.strip().lower() not in silence_marks]

            ph_intervals_raw = [i for i in ph_tier if i.mark.strip().lower() not in phone_silence_marks]
            ph_intervals = [i for i in ph_intervals_raw if i.mark.strip().lower() != 'spn']
            words_synth_phones = []

            base_name = os.path.splitext(real_wav_name)[0]
            filename_syllables = parse_ja_filename(base_name)
            is_vowel_chain = _is_vowel_chain_syllables(filename_syllables)
            cv_targets = _extract_ja_cv_targets_from_lines(lines, custom_map)

            detected_format = detect_ja_alias_format([l.split('=')[1].split(',')[0].strip() for l in lines], custom_map)
            format_type = forced_format or detected_format
            ja_style_profile = get_ja_profile_preset(format_type)
            if is_vowel_chain:
                prev_format = format_type
                format_type = 'vcv'
                ja_style_profile = get_ja_profile_preset(format_type)
                log(f"🎵 {fname}: 모음 연속음 파일 감지 → VCV 강제 적용 (기존: {prev_format.upper()})")
            elif forced_format:
                log(f"🎵 {fname}: 포맷 수동 지정 → {format_type.upper()} (자동 감지: {detected_format.upper()})")
            else:
                log(f"🎵 {fname}: 포맷 감지 → {format_type.upper()}")

            expected_syllables = max(len(filename_syllables), len(cv_targets), len(wd_intervals))
            phone_quality = _collect_phone_tier_quality(
                ph_tier,
                expected_syllables=expected_syllables,
                min_vowel_phone_ratio=ja_mapping_min_vowel_phone_ratio,
            )
            low_quality_reasons = list(phone_quality.get("low_confidence_reasons", []))
            spn_ratio = float(phone_quality.get("spn_ratio_in_phone_tier", 0.0))
            if spn_ratio >= float(ja_mapping_spn_ratio_threshold):
                low_quality_reasons.append("spn_heavy")
            low_quality_reasons = sorted(set(low_quality_reasons))
            low_phone_quality = bool(low_quality_reasons)

            forced_words_mapping = False
            if (
                format_type in {"vcv", "cvvc", "cv"}
                and ja_mapping_words_fallback_enabled
                and low_phone_quality
            ):
                if wd_intervals:
                    words_synth_phones = _build_words_synth_phones(wd_intervals, filename_syllables)
                    if words_synth_phones:
                        ph_intervals = words_synth_phones
                        forced_words_mapping = True
                        if ja_mapping_debug_reason_logging:
                            log(
                                f"🧭 {fname}: phones 신뢰도 낮음({','.join(low_quality_reasons)}) "
                                f"→ words 기반 합성 phone 우선 사용"
                            )
                elif ja_mapping_debug_reason_logging:
                    log(f"⚠️ {fname}: phones 신뢰도 낮음({','.join(low_quality_reasons)}), words 티어 없음")

            if not ph_intervals and wd_intervals:
                words_synth_phones = _build_words_synth_phones(wd_intervals, filename_syllables)
                if words_synth_phones:
                    ph_intervals = words_synth_phones

            if not ph_intervals:
                reason = "mapping_failed_empty_intervals"
                if low_phone_quality and "spn_heavy" in low_quality_reasons:
                    reason = "mapping_failed_spn_heavy"
                elif low_phone_quality and any(r in {"insufficient_phones", "insufficient_vowel_phones"} for r in low_quality_reasons):
                    reason = "mapping_failed_insufficient_phones"
                if low_phone_quality and not wd_intervals:
                    reason = "mapping_failed_no_words_support"
                log(f"⚠️ {fname}: 음소 정보 없음 → 원본 유지")
                _record_unset_lines(
                    reason,
                    fname,
                    lines,
                    meta={
                        "diag_hint": f"spn_ratio={spn_ratio:.2f}",
                        "phone_quality": phone_quality,
                        "forced_words_mapping": forced_words_mapping,
                    },
                )
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue

            timeline_start_ms = float(ph_intervals[0].minTime * 1000.0)
            timeline_end_ms = float(ph_intervals[-1].maxTime * 1000.0)
            if wd_intervals:
                timeline_start_ms = min(timeline_start_ms, float(wd_intervals[0].minTime * 1000.0))
                timeline_end_ms = max(timeline_end_ms, float(wd_intervals[-1].maxTime * 1000.0))

            effective_end_ms = timeline_end_ms
            if len(ph_intervals) >= 2:
                prev_p = ph_intervals[-2]
                last_p = ph_intervals[-1]
                gap_ms = float((last_p.minTime - prev_p.maxTime) * 1000.0)
                last_len_ms = float((last_p.maxTime - last_p.minTime) * 1000.0)
                if gap_ms > 450.0 and last_len_ms < 100.0:
                    effective_end_ms = float(prev_p.maxTime * 1000.0)

            boundary_points_ms = set()
            for p in ph_intervals:
                boundary_points_ms.add(float(p.minTime * 1000.0))
                boundary_points_ms.add(float(p.maxTime * 1000.0))
            if len(boundary_points_ms) < 4 and wd_intervals:
                for w in wd_intervals:
                    boundary_points_ms.add(float(w.minTime * 1000.0))
                    boundary_points_ms.add(float(w.maxTime * 1000.0))
            boundary_points_ms = sorted(boundary_points_ms)
            phone_spans_ms = [
                (float(p.minTime * 1000.0), float(p.maxTime * 1000.0))
                for p in ph_intervals
            ]

            post_ctx = JaPostprocessContext(
                phone_spans_ms=phone_spans_ms,
                timeline_start_ms=timeline_start_ms,
                effective_end_ms=effective_end_ms,
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
            
            # === 단모음 처리 (음소 1개) ===
            if len(ph_intervals) == 1:
                log(f"🎵 {fname}: 단모음 파일")
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

                    offset, consonant, cutoff, pre, ovl = post_ctx.post_adjust(
                        offset, consonant, cutoff, pre, ovl, alias_type='br', alias_text=alias
                    )

                    aliases_to_write = generate_ja_openutau_aliases(alias) if generate_openutau else [alias]
                    for a in aliases_to_write:
                        a2 = _alias_out(a)
                        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                        final_lines.append(new_line)
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
            linear_fallback_used = False
            conf_th = _resolve_ja_mapping_conf_threshold(
                format_type,
                override_threshold=ja_mapping_confidence_threshold,
            )

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

            candidate_by_name = {c["name"]: c for c in mapping_candidates}
            alias_candidate = candidate_by_name.get("alias_phone_fallback")
            selected_candidate = None

            # 1단계: 순서 보존 후보를 우선 필터링.
            if mapping_candidates:
                if forced_words_mapping:
                    if candidate_by_name.get("filename_token"):
                        selected_candidate = candidate_by_name["filename_token"]
                        mapping_reason_code = "filename_words_lock"
                    elif candidate_by_name.get("filename_linear_fallback"):
                        selected_candidate = candidate_by_name["filename_linear_fallback"]
                        mapping_reason_code = "filename_words_linear_lock"
                    elif alias_candidate:
                        selected_candidate = alias_candidate
                        mapping_reason_code = "alias_words_fallback"
                    else:
                        selected_candidate = max(mapping_candidates, key=lambda c: c.get("objective", -10**9))
                        mapping_reason_code = selected_candidate["name"]
                else:
                    stage1_pool = list(mapping_candidates)
                    if format_type in {"cvvc", "cv"}:
                        order_pool = [c for c in mapping_candidates if c.get("order_preserving")]
                        if order_pool:
                            stage1_pool = order_pool
                    selected_candidate = max(
                        stage1_pool,
                        key=lambda c: (
                            c.get("objective", -10**9)
                            + (3.0 if (format_type in {"cvvc", "cv"} and c.get("order_preserving")) else 0.0)
                        ),
                    )
                    mapping_reason_code = selected_candidate["name"]

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
                    objective_gain = alias_candidate.get("objective", -10**9) - selected_candidate.get("objective", -10**9)
                    score_gain = alias_candidate.get("score", -10**9) - selected_candidate.get("score", -10**9)
                    conf_gain = alias_candidate.get("mean_syll_conf", 0.0) - selected_candidate.get("mean_syll_conf", 0.0)
                    gain_th = 7.0 if format_type in {"cvvc", "cv"} else 5.0
                    if (
                        alias_candidate.get("score", 0.0) >= 70.0
                        and objective_gain >= gain_th
                        and (score_gain >= 5.0 or conf_gain >= 0.06)
                    ):
                        selected_candidate = alias_candidate
                        mapping_reason_code = "alias_recover"

            if selected_candidate:
                syllables_info = list(selected_candidate.get("infos") or [])
                used_filename_based = bool(selected_candidate.get("use_filename"))
                used_alias_based = bool(selected_candidate.get("use_alias"))
                filename_order_locked = bool(
                    selected_candidate.get("lock_order") and format_type in {"cvvc", "cv"}
                )
                linear_fallback_used = bool(selected_candidate.get("linear_fallback"))
                base_score = float(selected_candidate.get("score", -1.0))
                syllable_confidence_by_idx = list(selected_candidate.get("syllable_confidences", []) or [])
            else:
                syllables_info = []
                base_score = -1.0
                syllable_confidence_by_idx = []

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
            if mapping_confidence_base >= (conf_th + 0.12):
                mapping_tier = "high"
            elif mapping_confidence_base >= conf_th:
                mapping_tier = "mid"
            else:
                mapping_tier = "low"

            # 일본어 CVVC/CV는 저신뢰일 때 파일명 순서를 기준 축으로 고정.
            if format_type in {"cvvc", "cv"} and mapping_tier == "low" and (not filename_order_locked):
                fallback_candidate = candidate_by_name.get("filename_linear_fallback") or candidate_by_name.get("filename_token")
                if fallback_candidate is not None and fallback_candidate is not selected_candidate:
                    selected_candidate = fallback_candidate
                    syllables_info = list(selected_candidate.get("infos") or [])
                    used_filename_based = bool(selected_candidate.get("use_filename"))
                    used_alias_based = bool(selected_candidate.get("use_alias"))
                    filename_order_locked = True
                    linear_fallback_used = bool(selected_candidate.get("linear_fallback"))
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
                    if mapping_confidence_base >= (conf_th + 0.12):
                        mapping_tier = "high"
                    elif mapping_confidence_base >= conf_th:
                        mapping_tier = "mid"
                    else:
                        mapping_tier = "low"
                    log(
                        f"🧭 {fname}: 일본어 매핑 저신뢰(conf={mapping_confidence_base:.2f}) "
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
                elif mapping_reason_code not in {"alias_recover"}:
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
                if idx < 0 or idx >= len(syllables_info):
                    return None
                syl = syllables_info[idx]
                curr_phones = syl.get('phones') or []
                if not curr_phones:
                    return None

                anchor_alias = syl.get('roman_cv') or syl.get('roman') or syl.get('word') or ""
                c_start, c_end, n_start, n_end = _ja_extract_cv_bounds(
                    curr_phones, alias_text=anchor_alias, alias_type="cv"
                )

                c_hint = curr_phones[0].mark if curr_phones else ""
                anchor_alias = syl.get('roman_cv') or syl.get('roman') or syl.get('word') or ""
                cv_vowel_len = max(n_end - n_start, 20.0)
                offset, pre = _ja_cv_offset_and_pre(
                    c_start,
                    c_end,
                    anchor_alias,
                    c_hint=c_hint,
                    alias_type="cv",
                    vowel_start=n_start,
                    vowel_end=n_end,
                )
                pre = max(pre, 10.0)
                ovl = _adaptive_ja_overlap(pre, c_hint, mode="cv")
                v_ref = max(cv_vowel_len, 120.0)
                added_cons = min(max(v_ref * 0.45, 70.0), 180.0)
                consonant = pre + added_cons
                cutoff = -(consonant + max(cv_vowel_len * 0.25, 45.0))

                offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
                return {
                    "offset": offset,
                    "pre": pre,
                    "ovl": ovl,
                    "cons": consonant,
                    "cutoff": cutoff,
                    "pre_abs": offset + pre,
                    "cons_abs": offset + consonant,
                    "onset_abs": c_start,
                    "vowel_start_abs": n_start,
                    "c_end_abs": c_end,
                    "vowel_end_abs": n_end,
                    "vowel_len": cv_vowel_len,
                    "cons_gap": max(consonant - pre, 10.0),
                    "cut_gap": max(abs(cutoff) - consonant, 16.0),
                }

            def _compute_ja_vc_from_adjacent_cv(prev_cv, next_cv, alias_type, c_char, bridge_profile):
                if not prev_cv or not next_cv:
                    return None

                profile = bridge_profile or JA_CVVC_BRIDGE_TIMING.get("default", {})
                boundary_abs = next_cv["onset_abs"] if alias_type == "vc" else next_cv["pre_abs"]
                pre_target = _clamp_range(_blend(prev_cv["pre"], next_cv["pre"], 0.34), 40.0, 220.0)
                offset = max(boundary_abs - pre_target, 0.0)
                pre = boundary_abs - offset
                if pre <= 0:
                    return None

                tail_margin = profile.get("tail_margin_base", 10.0) + prev_cv["vowel_len"] * profile.get("tail_margin_mul", 0.05)
                tail_margin = _clamp_range(tail_margin, 4.0, 24.0)
                target_ovl_abs = prev_cv["vowel_end_abs"] - tail_margin
                upper_ovl = max(pre - profile.get("ovl_pre_margin", 6.0), 0.0)
                lower_ovl = min(pre * profile.get("ovl_min_ratio", 0.40), upper_ovl)
                ovl = min(upper_ovl, max(lower_ovl, target_ovl_abs - offset))
                ovl = _blend(ovl, pre * profile.get("ovl_ratio", 0.50), 0.20)
                ovl = min(upper_ovl, max(lower_ovl, ovl))

                cons_gap = _clamp_range(_blend(prev_cv["cons_gap"], next_cv["cons_gap"], 0.45), 14.0, 120.0)
                consonant = pre + cons_gap
                next_onset_rel = max(next_cv["onset_abs"] - offset, pre + 10.0)
                next_pre_rel = max(next_cv["pre_abs"] - offset, pre + 16.0)
                next_cons_rel = max(next_cv["cons_abs"] - offset, next_pre_rel + 10.0)

                if alias_type == "vc":
                    if c_char in JA_PLOSIVE_CONSONANTS:
                        consonant = min(consonant, next_onset_rel - 6.0)
                        consonant = max(consonant, pre + 12.0)
                        cutoff_abs = max(consonant + 10.0, next_onset_rel - 2.0)
                        cutoff_abs = min(cutoff_abs, next_onset_rel + 6.0)
                    else:
                        consonant = min(consonant, next_onset_rel + 24.0)
                        consonant = max(consonant, pre + 16.0)
                        cutoff_abs = max(consonant + 12.0, min(next_cons_rel + 24.0, next_pre_rel + 40.0))
                else:
                    consonant = min(max(consonant, pre + 22.0), next_pre_rel + 44.0)
                    cutoff_abs = max(consonant + 20.0, next_pre_rel + 10.0)
                    cutoff_abs = min(cutoff_abs, next_cons_rel + 54.0)

                cutoff = -cutoff_abs
                return validate_oto_params(offset, consonant, cutoff, pre, ovl)

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

                # 포맷 강제 시 VCV 오인식을 보정
                if format_type in ('cvvc', 'cv') and alias_type == 'vcv':
                    parts = alias.strip().split()
                    if len(parts) >= 2 and _is_ja_vowel_token(parts[0]):
                        right = parts[1].strip().lower()
                        if _is_ja_vowel_token(right):
                            alias_type = 'vv'
                        else:
                            alias_type = 'vc'
                    else:
                        alias_type = 'cv'
                elif format_type == 'vcv' and alias_type in ('vc', 'vv'):
                    parts = alias.strip().split()
                    if alias_type == 'vv':
                        alias_type = 'vcv'
                    elif len(parts) >= 2 and _is_ja_vowel_token(parts[0]):
                        right = parts[1].strip().lower()
                        if right not in JA_CONSONANTS and not _is_ja_vowel_token(right):
                            alias_type = 'vcv'

                # VC 꼬리 호흡(R/H) 전용 처리: 과도한 컷오프 확장을 방지
                tail_breath = None
                a_parts = alias.strip().split()
                if len(a_parts) >= 2 and _is_ja_vowel_token(a_parts[0]) and a_parts[1].upper() in {'R', 'H'}:
                    tail_breath = a_parts[1].upper()
                
                # === 숨소리(br) 처리 ===
                if alias_type == 'br':
                    first_ph = ph_intervals[0] if ph_intervals else None
                    last_ph = ph_intervals[-1] if ph_intervals else None
                    if first_ph and last_ph:
                        br_start = first_ph.minTime * 1000
                        br_end = last_ph.maxTime * 1000
                        br_len = br_end - br_start
                    else:
                        br_start = 0
                        br_end = 500
                        br_len = 500
                    offset = max(br_start - 30, 0)
                    pre = 0
                    ovl = 0
                    consonant = min(br_len * 0.3, 100)
                    cutoff = -(br_len * 0.85)
                    offset, consonant, cutoff, pre, ovl = post_ctx.post_adjust(
                        offset, consonant, cutoff, pre, ovl,
                        alias_type='vc', alias_text=alias,
                        local_end_ms=br_end, local_cut_allow_ms=40.0
                    )
                    
                    aliases_to_write = generate_ja_openutau_aliases(alias) if generate_openutau else [alias]
                    for a in aliases_to_write:
                        a2 = _alias_out(a)
                        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                        final_lines.append(new_line)
                    continue

                is_vc = alias_type in ('vc', 'vv')
                is_vcv = alias_type == 'vcv'
                is_cv_head = alias_type == 'cv_head'
                c_char = ""
                vc_prev_anchor = None
                vc_next_anchor = None

                if tail_breath:
                    if current_w_idx >= len(syllables_info):
                        current_w_idx = len(syllables_info) - 1

                    curr_syl = syllables_info[current_w_idx]
                    curr_phones = curr_syl['phones']
                    onset_hint_local = curr_phones[0].mark if curr_phones else ""
                    v_phone = curr_phones[-1]
                    v_end = v_phone.maxTime * 1000
                    v_start = v_phone.minTime * 1000
                    v_len = max(v_end - v_start, 80)

                    offset = max(v_end - min(max(v_len * 0.8, 120), 260), 0)
                    pre = max(v_end - offset, 40)
                    ovl = min(pre * 0.3, max(pre - 16, 0))
                    consonant = pre + min(max(v_len * 0.2, 22), 55)
                    cutoff = -(consonant + 38)

                    offset, consonant, cutoff, pre, ovl = post_ctx.post_adjust(
                        offset, consonant, cutoff, pre, ovl,
                        alias_type='mono', alias_text=alias,
                        local_end_ms=v_end, local_cut_allow_ms=36.0
                    )

                    aliases_to_write = generate_ja_openutau_aliases(alias) if generate_openutau else [alias]
                    for a in aliases_to_write:
                        a2 = _alias_out(a)
                        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                        final_lines.append(new_line)
                    continue

                # === VCV 연속음 처리 ===
                if is_vcv:
                    expected_seq_idx = stable_vcv_seq_idx if format_type == "vcv" else cv_seq_idx
                    if expected_seq_idx < len(syllables_info):
                        expected_idx = expected_seq_idx
                    else:
                        expected_idx = len(syllables_info) - 1
                    mapped_idx = _select_vcv_syllable_index(alias, expected_idx, syllables_info)
                    target_tok_vcv_raw = _extract_vcv_target_syllable(alias)
                    resynced_vcv_exact = False
                    resync_idx_vcv = _find_ja_exact_target_index(
                        target_tok_vcv_raw,
                        expected_idx,
                        syllables_info,
                        search_back=(0 if format_type == "vcv" else 3),
                        search_fwd=3,
                    )
                    target_tok_vcv_norm_for_resync = _normalize_ja_syllable_token(target_tok_vcv_raw)
                    exp_tok_vcv_for_resync = _normalize_ja_syllable_token(
                        _syllable_info_token(syllables_info[expected_idx])
                    )
                    # 역방향 exact 재동기화는 매우 보수적으로 허용:
                    # expected가 이미 target이면 앞 음절로 되돌아가면 안 된다.
                    if (
                        resync_idx_vcv is not None
                        and resync_idx_vcv < expected_idx
                        and exp_tok_vcv_for_resync == target_tok_vcv_norm_for_resync
                    ):
                        resync_idx_vcv = None
                    if resync_idx_vcv is not None and resync_idx_vcv != mapped_idx:
                        log(
                            f"🧭 {fname}: VCV 순서 드리프트 복구 "
                            f"{expected_idx + 1}->{resync_idx_vcv + 1} ({alias})"
                        )
                        mapped_idx = resync_idx_vcv
                        resynced_vcv_exact = True
                    # 모음 불일치(예: ri -> ra) 시는 강제로 재탐색/되돌림
                    target_tok_vcv_norm = _normalize_ja_syllable_token(target_tok_vcv_raw)
                    mapped_tok_vcv_now = _normalize_ja_syllable_token(_syllable_info_token(syllables_info[mapped_idx]))
                    exp_tok_vcv_now = _normalize_ja_syllable_token(_syllable_info_token(syllables_info[expected_idx]))
                    _t_on, target_vowel_vcv = split_ja_romaji_syllable(target_tok_vcv_norm)
                    _m_on, mapped_vowel_vcv = split_ja_romaji_syllable(mapped_tok_vcv_now)
                    _e_on, expected_vowel_vcv = split_ja_romaji_syllable(exp_tok_vcv_now)
                    if target_vowel_vcv in JA_VOWELS and mapped_vowel_vcv and mapped_vowel_vcv != target_vowel_vcv:
                        fixed_idx_vcv = _find_ja_cv_vowel_match_index(
                            target_tok_vcv_norm,
                            expected_idx,
                            syllables_info,
                            search_back=2,
                            search_fwd=3,
                        )
                        if fixed_idx_vcv is not None and fixed_idx_vcv != mapped_idx:
                            log(
                                f"🧭 {fname}: VCV 모음 불일치 보정 "
                                f"{mapped_idx + 1}->{fixed_idx_vcv + 1} ({alias})"
                            )
                            mapped_idx = fixed_idx_vcv
                        elif expected_vowel_vcv == target_vowel_vcv:
                            log(
                                f"🛡️ {fname}: VCV 모음 불일치 차단 "
                                f"({mapped_idx + 1}->{expected_idx + 1}, {alias})"
                            )
                            mapped_idx = expected_idx
                    if mapped_idx > expected_idx:
                        target_tok_vcv = _normalize_ja_syllable_token(target_tok_vcv_raw)
                        exp_tok_vcv = _normalize_ja_syllable_token(_syllable_info_token(syllables_info[expected_idx]))
                        mapped_tok_vcv = _normalize_ja_syllable_token(_syllable_info_token(syllables_info[mapped_idx]))
                        if mapped_tok_vcv != target_tok_vcv or exp_tok_vcv == target_tok_vcv:
                            mapped_idx = expected_idx
                    if mapped_idx < expected_idx:
                        # VCV에서는 누적 선행 드리프트 복구를 위해 제한적 역방향 이동 허용
                        if (expected_idx - mapped_idx) > 3:
                            mapped_idx = expected_idx
                    if mapped_idx > (expected_idx + 1):
                        # exact 토큰 재동기화로 확인된 경우에만 +2까지 허용
                        if not (resynced_vcv_exact and mapped_idx <= (expected_idx + 2)):
                            mapped_idx = expected_idx
                    # 과도 점프/역행 가드: 한 줄에서 여러 음절을 건너뛰지 않도록 제한
                    if last_vcv_mapped_idx >= 0:
                        if mapped_idx < last_vcv_mapped_idx:
                            log(
                                f"🛡️ {fname}: VCV 역행 점프 차단 "
                                f"{mapped_idx + 1}->{last_vcv_mapped_idx + 1} ({alias})"
                            )
                            mapped_idx = last_vcv_mapped_idx
                        elif mapped_idx > (last_vcv_mapped_idx + 2):
                            clamped_idx = last_vcv_mapped_idx + 1
                            log(
                                f"🛡️ {fname}: VCV 과도 점프 차단 "
                                f"{mapped_idx + 1}->{clamped_idx + 1} ({alias})"
                            )
                            mapped_idx = clamped_idx
                    # 최종 안전장치: 목표 토큰과 매핑 토큰이 다르면 모음 기준 재탐색 후 되돌림
                    mapped_tok_final = _normalize_ja_syllable_token(
                        _syllable_info_token(syllables_info[mapped_idx])
                    )
                    if target_tok_vcv_norm and mapped_tok_final != target_tok_vcv_norm:
                        retry_idx_vcv = _find_ja_cv_vowel_match_index(
                            target_tok_vcv_norm,
                            expected_idx,
                            syllables_info,
                            search_back=4,
                            search_fwd=4,
                        )
                        if (
                            retry_idx_vcv is not None
                            and retry_idx_vcv != mapped_idx
                            and abs(retry_idx_vcv - expected_idx) <= 2
                        ):
                            log(
                                f"🧭 {fname}: VCV 최종 재탐색 "
                                f"{mapped_idx + 1}->{retry_idx_vcv + 1} ({alias})"
                            )
                            mapped_idx = retry_idx_vcv
                            mapped_tok_final = _normalize_ja_syllable_token(
                                _syllable_info_token(syllables_info[mapped_idx])
                            )
                        if mapped_tok_final != target_tok_vcv_norm:
                            log(
                                f"🛡️ {fname}: VCV 토큰 불일치 되돌림 "
                                f"({mapped_idx + 1}->{expected_idx + 1}, {alias})"
                            )
                            mapped_idx = expected_idx
                    # VCV는 MFA/phone 순서를 기본으로 유지하되,
                    # target exact match가 명확한 1칸 보정만 제한 허용한다.
                    if format_type == "vcv":
                        mapped_idx = _prefer_vcv_candidate_index(
                            expected_idx,
                            mapped_idx,
                            target_tok_vcv_norm,
                            syllables_info,
                            max_delta=1,
                        )
                    if mapped_idx != expected_idx and abs(mapped_idx - expected_idx) <= 1:
                        log(f"🧭 {fname}: VCV 음절 정렬 보정 {expected_idx + 1}->{mapped_idx + 1} ({alias})")
                    expected_tok_trace = _syllable_info_token(syllables_info[expected_idx])
                    mapped_tok_trace = _syllable_info_token(syllables_info[mapped_idx])
                    local_trace_conf = None
                    if syllable_confidence_by_idx:
                        conf_idx = max(0, min(expected_idx, len(syllable_confidence_by_idx) - 1))
                        local_trace_conf = float(syllable_confidence_by_idx[conf_idx])
                    if (
                        mapping_tier == "low"
                        or mapped_idx != expected_idx
                        or _normalize_ja_syllable_token(mapped_tok_trace) != _normalize_ja_syllable_token(target_tok_vcv_norm)
                    ):
                        _append_mapping_trace(
                            _build_ja_mapping_trace_record(
                                fname=fname,
                                alias=alias,
                                alias_type="vcv",
                                format_type=format_type,
                                target_tok=target_tok_vcv_norm,
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
                    last_vcv_mapped_idx = current_w_idx
                        
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
                        
                    _c_start, c_boundary, n_start, n_end = _ja_extract_cv_bounds(
                        curr_phones, alias_text=alias, alias_type="vcv"
                    )

                    offset, consonant, cutoff, pre, ovl = _compute_vcv_params_from_virtual_split(
                        alias, prev_v_start, prev_v_end, c_boundary, n_end, base_shape=base_shape
                    )
                    offset, consonant, cutoff, pre, ovl, soft_off_shift, soft_cut_shift = _apply_soft_mel_offset_cutoff_guard(
                        offset, consonant, cutoff, pre, ovl, "vcv", mel_ctx_for_file,
                        onset_hint=onset_hint_local, alias_text=alias
                    )
                    if abs(soft_off_shift) > 1.0 or abs(soft_cut_shift) > 1.0:
                        log(
                            f"🛡️ {fname}: 초기 멜 가드 적용 (offset {soft_off_shift:+.1f}ms, cutoff -{soft_cut_shift:.1f}ms) [{alias}]"
                        )
                    
                    offset, consonant, cutoff, pre, ovl = post_ctx.post_adjust(
                        offset, consonant, cutoff, pre, ovl,
                        alias_type='vcv', alias_text=alias,
                        local_end_ms=n_end, local_cut_allow_ms=26.0
                    )
                    offset, consonant, cutoff, pre, ovl = _enforce_vcv_cv_entry_guard(
                        offset, consonant, cutoff, pre, ovl,
                        c_boundary=c_boundary, n_end=n_end, alias_text=alias
                    )
                    vcv_tok_for_profile = _extract_vcv_target_syllable(alias)
                    vcv_onset_for_profile, _vcv_vowel_for_profile = split_ja_romaji_syllable(vcv_tok_for_profile)
                    n_bridge_vcv = _ja_is_n_bridge_alias(alias, "vcv")
                    vv_like_profile_key = "vcv_vv_like" if not (vcv_onset_for_profile or "").strip() else "vcv"
                    if n_bridge_vcv and vv_like_profile_key == "vcv":
                        vv_like_profile_key = "vcv_n_bridge"
                    offset, consonant, cutoff, pre, ovl = _apply_ja_anchor_lock(
                        fname=fname,
                        alias_text=alias,
                        format_type="vcv",
                        alias_type="vcv",
                        profile_alias_type=vv_like_profile_key,
                        offset=offset,
                        consonant=consonant,
                        cutoff=cutoff,
                        pre=pre,
                        ovl=ovl,
                        anchor_abs_ms=c_boundary,
                        next_onset_abs_ms=n_start,
                        next_vowel_abs_ms=n_end,
                    )
                    
                    aliases_to_write = generate_ja_openutau_aliases(alias) if generate_openutau else [alias]
                    for a in aliases_to_write:
                        a2 = _alias_out(a)
                        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                        final_lines.append(new_line)
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
                    if skip_cv_align:
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
                        if mapped_idx != expected_idx and abs(mapped_idx - expected_idx) <= 1:
                            log(f"🧭 {fname}: CV 음절 정렬 보정 {expected_idx + 1}->{mapped_idx + 1} ({alias})")
                    expected_tok_trace = _syllable_info_token(syllables_info[expected_idx])
                    mapped_tok_trace = _syllable_info_token(syllables_info[mapped_idx])
                    local_trace_conf = None
                    if syllable_confidence_by_idx:
                        conf_idx = max(0, min(expected_idx, len(syllable_confidence_by_idx) - 1))
                        local_trace_conf = float(syllable_confidence_by_idx[conf_idx])
                    if (
                        mapping_tier == "low"
                        or mapped_idx != expected_idx
                        or _normalize_ja_syllable_token(mapped_tok_trace) != _normalize_ja_syllable_token(target_tok)
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
                        curr_phones, alias_text=alias, alias_type="cv_head"
                    )
                        
                    c_hint = curr_phones[0].mark if curr_phones else ""
                    offset, pre = _ja_cv_offset_and_pre(
                        c_start,
                        c_end,
                        alias,
                        c_hint=c_hint,
                        alias_type="cv_head",
                        vowel_start=n_start,
                        vowel_end=n_end,
                    )
                    onset_hint_local = c_hint
                    ovl = _adaptive_ja_overlap(pre, c_hint, mode="cv_head")
                    
                    cv_vowel_len = n_end - n_start
                    added_cons = min(cv_vowel_len * 0.5, 150)
                    if added_cons < 80: added_cons = 80
                    consonant = pre + added_cons
                    min_cut_abs = post_ctx.cv_head_min_cutoff(
                        offset, consonant, pre, n_start, n_end
                    )
                    if min_cut_abs is None:
                        cutoff = -(consonant + cv_vowel_len * 0.25)
                    else:
                        cutoff = -max(consonant + 14.0, min_cut_abs)
                    offset, consonant, cutoff, pre, ovl, soft_off_shift, soft_cut_shift = _apply_soft_mel_offset_cutoff_guard(
                        offset, consonant, cutoff, pre, ovl, "cv_head", mel_ctx_for_file,
                        onset_hint=onset_hint_local, alias_text=alias
                    )
                    if abs(soft_off_shift) > 1.0 or abs(soft_cut_shift) > 1.0:
                        log(
                            f"🛡️ {fname}: 초기 멜 가드 적용 (offset {soft_off_shift:+.1f}ms, cutoff -{soft_cut_shift:.1f}ms) [{alias}]"
                        )

                    offset, consonant, cutoff, pre, ovl = _apply_base_shape_blend(
                        offset, consonant, cutoff, pre, ovl, base_shape, alias_type="cv_head"
                    )
                    
                    offset, consonant, cutoff, pre, ovl = post_ctx.post_adjust(
                        offset, consonant, cutoff, pre, ovl,
                        alias_type='cv_head', alias_text=alias,
                        local_end_ms=n_end, local_cut_allow_ms=28.0
                    )
                    offset, consonant, cutoff, pre, ovl = _apply_ja_anchor_lock(
                        fname=fname,
                        alias_text=alias,
                        format_type=format_type,
                        alias_type="cv_head",
                        offset=offset,
                        consonant=consonant,
                        cutoff=cutoff,
                        pre=pre,
                        ovl=ovl,
                        anchor_abs_ms=c_end,
                        next_onset_abs_ms=n_start,
                        next_vowel_abs_ms=n_end,
                    )
                    offset, consonant, cutoff, pre, offset_reduced = post_ctx.guard_cv_head_offset_to_onset(
                        offset, consonant, cutoff, pre, current_w_idx, alias_text=alias
                    )
                    v_cov_start = n_start
                    v_cov_end = n_end
                    try:
                        v_target = _ja_target_vowel_from_alias(alias, alias_type="cv_head")
                        v_phone, _v_idx = _ja_pick_vowel_phone(curr_phones, target_vowel=v_target)
                        if v_phone is not None:
                            v_cov_start = float(v_phone.minTime) * 1000.0
                            v_cov_end = float(v_phone.maxTime) * 1000.0
                    except Exception:
                        pass
                    offset, consonant, cutoff, pre, cutoff_extended = post_ctx.ensure_cv_head_min_vowel_coverage(
                        offset, consonant, cutoff, pre, v_cov_start, v_cov_end
                    )
                    if offset_reduced > 1.0:
                        log(f"🛡️ {fname}: CV_HEAD 오프셋 과선행 보정(+{offset_reduced:.1f}ms) [{alias}]")
                    if cutoff_extended > 1.0:
                        log(f"🛡️ {fname}: CV_HEAD 모음 길이 보정(+{cutoff_extended:.1f}ms) [{alias}]")
                    offset, consonant, cutoff, pre, cutoff_reduced = post_ctx.guard_cv_cutoff_to_next_onset(
                        offset, consonant, cutoff, pre, current_w_idx,
                        alias_type="cv_head",
                        format_type=format_type,
                        vowel_start_ms=v_cov_start,
                        vowel_end_ms=v_cov_end,
                    )
                    if cutoff_reduced > 0.5:
                        log(f"🛡️ {fname}: CV 컷오프 과연장 보정(-{cutoff_reduced:.1f}ms) [{alias}]")
                    offset, consonant, cutoff, pre, ovl = validate_oto_params(
                        offset, consonant, cutoff, pre, ovl
                    )
                    realized_cv_anchor_by_idx[current_w_idx] = {
                        "offset": offset,
                        "pre": pre,
                        "ovl": ovl,
                        "cons": consonant,
                        "cutoff": cutoff,
                        "pre_abs": offset + pre,
                        "cons_abs": offset + consonant,
                        "onset_abs": c_start,
                        "vowel_start_abs": n_start,
                        "c_end_abs": c_end,
                        "vowel_end_abs": n_end,
                        "vowel_len": max(n_end - n_start, 20.0),
                        "cons_gap": max(consonant - pre, 10.0),
                        "cut_gap": max(abs(cutoff) - consonant, 16.0),
                    }
                    
                    aliases_to_write = generate_ja_openutau_aliases(alias) if generate_openutau else [alias]
                    for a in aliases_to_write:
                        a2 = _alias_out(a)
                        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                        final_lines.append(new_line)
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
                    if skip_cv_align:
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
                        if mapped_idx != expected_idx and abs(mapped_idx - expected_idx) <= 1:
                            log(f"🧭 {fname}: CV 음절 정렬 보정 {expected_idx + 1}->{mapped_idx + 1} ({alias})")
                    expected_tok_trace = _syllable_info_token(syllables_info[expected_idx])
                    mapped_tok_trace = _syllable_info_token(syllables_info[mapped_idx])
                    local_trace_conf = None
                    if syllable_confidence_by_idx:
                        conf_idx = max(0, min(expected_idx, len(syllable_confidence_by_idx) - 1))
                        local_trace_conf = float(syllable_confidence_by_idx[conf_idx])
                    if (
                        mapping_tier == "low"
                        or mapped_idx != expected_idx
                        or _normalize_ja_syllable_token(mapped_tok_trace) != _normalize_ja_syllable_token(target_tok)
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
                    current_w_idx = max(cv_seq_idx - 1, vc_seq_idx)
                    if current_w_idx >= len(syllables_info):
                        current_w_idx = len(syllables_info) - 1
                    if vc_seq_idx < len(syllables_info) - 1:
                        vc_seq_idx += 1

                    curr_syl = syllables_info[current_w_idx]
                    curr_phones = curr_syl['phones']

                    v_start = curr_phones[-1].minTime * 1000
                    v_end = curr_phones[-1].maxTime * 1000
                    c_start = v_start
                    c_end = v_end

                    if current_w_idx + 1 < len(syllables_info):
                        next_syl = syllables_info[current_w_idx + 1]
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
                    if (not use_vcv_anchor) and alias_type in ('vc', 'vv') and (current_w_idx + 1) < len(syllables_info):
                        prev_cv_anchor = realized_cv_anchor_by_idx.get(current_w_idx) or cv_anchor_by_idx.get(current_w_idx)
                        next_cv_anchor = realized_cv_anchor_by_idx.get(current_w_idx + 1) or cv_anchor_by_idx.get(current_w_idx + 1)
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

                    # CVVC 원리: VC 파열/파찰음은 다음 자음 onset 직전에서 정리해 중복 파열을 줄인다.
                    if alias_type == 'vc' and c_char in JA_PLOSIVE_CONSONANTS:
                        onset_guard = max(next_c_onset_rel, pre + 14.0)
                        consonant = min(consonant, onset_guard - 6.0)
                        consonant = max(consonant, pre + 12.0)
                        cutoff_abs = max(consonant + 10.0, onset_guard - 2.0)
                        cutoff_abs = min(cutoff_abs, onset_guard + 6.0)
                        cutoff = -cutoff_abs

                    if alias_type == 'vc':
                        if vc_prev_anchor is None:
                            vc_prev_anchor = realized_cv_anchor_by_idx.get(current_w_idx) or cv_anchor_by_idx.get(current_w_idx)
                        if vc_next_anchor is None:
                            vc_next_anchor = realized_cv_anchor_by_idx.get(current_w_idx + 1) or cv_anchor_by_idx.get(current_w_idx + 1)

                if alias_type in {"cv", "cv_head", "vcv"}:
                    offset, consonant, cutoff, pre, ovl, soft_off_shift, soft_cut_shift = _apply_soft_mel_offset_cutoff_guard(
                        offset, consonant, cutoff, pre, ovl, alias_type, mel_ctx_for_file,
                        onset_hint=onset_hint_local, alias_text=alias
                    )
                    if abs(soft_off_shift) > 1.0 or abs(soft_cut_shift) > 1.0:
                        log(
                            f"🛡️ {fname}: 초기 멜 가드 적용 (offset {soft_off_shift:+.1f}ms, cutoff -{soft_cut_shift:.1f}ms) [{alias}]"
                        )

                offset, consonant, cutoff, pre, ovl = _apply_base_shape_blend(
                    offset, consonant, cutoff, pre, ovl, base_shape, alias_type=alias_type
                )
                offset, consonant, cutoff, pre, ovl = post_ctx.post_adjust(
                    offset, consonant, cutoff, pre, ovl,
                    alias_type=alias_type, alias_text=alias,
                    local_end_ms=n_end,
                    local_cut_allow_ms=(44.0 if alias_type == 'vc' else 40.0 if alias_type == 'vv' else 54.0),
                )
                if alias_type == "vc":
                    pre_abs_before = offset + pre
                    offset, consonant, cutoff, pre, ovl = _refine_ja_vc_with_adjacent_cv(
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        c_char=c_char,
                        prev_cv_anchor=vc_prev_anchor,
                        next_cv_anchor=vc_next_anchor,
                        prev_v_end_abs=c_end,
                        next_c_start_abs=n_start,
                        next_c_end_abs=n_end,
                    )
                    pre_abs_after = offset + pre
                    # CVVC에서 CV 앵커 드리프트가 있을 때 VC까지 과보정되는 것을 막기 위해
                    # pre 절대 이동량을 보수적으로 제한한다.
                    if format_type in {"cvvc", "cv"}:
                        max_shift = 26.0
                        if mapping_tier == "high":
                            max_shift = 34.0
                        onset_cls = _ja_onset_class(c_char)
                        if onset_cls in {"voiced", "nasal"}:
                            max_shift += 4.0
                        if _ja_is_n_bridge_alias(alias, "vc"):
                            max_shift = min(max_shift, 22.0)
                        (
                            offset,
                            consonant,
                            cutoff,
                            pre,
                            ovl,
                            pre_abs_after,
                            clamped_shift,
                        ) = _limit_pre_anchor_shift(
                            offset,
                            consonant,
                            cutoff,
                            pre,
                            ovl,
                            pre_abs_before=pre_abs_before,
                            max_shift_ms=max_shift,
                        )
                        if clamped_shift:
                            log(
                                f"🛡️ {fname}: VC-CV 앵커 이동 제한 "
                                f"({pre_abs_before:.1f}->{pre_abs_after:.1f}ms, {alias})"
                            )
                    if abs(pre_abs_after - pre_abs_before) >= 6.0:
                        log(
                            f"🧭 {fname}: VC-CV 앵커 재정렬 "
                            f"({pre_abs_before:.1f}->{pre_abs_after:.1f}ms, {alias})"
                        )
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
                offset, consonant, cutoff, pre, ovl = _apply_ja_anchor_lock(
                    fname=fname,
                    alias_text=alias,
                    format_type=format_type,
                    alias_type=alias_type,
                    offset=offset,
                    consonant=consonant,
                    cutoff=cutoff,
                    pre=pre,
                    ovl=ovl,
                    anchor_abs_ms=anchor_abs,
                    next_onset_abs_ms=next_onset_abs,
                    next_vowel_abs_ms=next_vowel_abs,
                )
                if alias_type in {"cv", "cv_head"} and not is_anchor_lock_enabled("japanese", format_type):
                    offset, consonant, cutoff, pre, ovl = _enforce_cv_pre_anchor_guard(
                        offset, consonant, cutoff, pre, ovl,
                        c_end_abs=c_end, alias_type=alias_type
                    )
                if alias_type in {"cv", "cv_head"}:
                    offset, consonant, cutoff, pre, cutoff_reduced = post_ctx.guard_cv_cutoff_to_next_onset(
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        current_w_idx,
                        alias_type=alias_type,
                        format_type=format_type,
                        vowel_start_ms=(n_start if alias_type == "cv_head" else None),
                        vowel_end_ms=(n_end if alias_type == "cv_head" else None),
                    )
                    if cutoff_reduced > 0.5:
                        log(f"🛡️ {fname}: CV 컷오프 과연장 보정(-{cutoff_reduced:.1f}ms) [{alias}]")
                    offset, consonant, cutoff, pre, ovl = validate_oto_params(
                        offset, consonant, cutoff, pre, ovl
                    )
                    realized_cv_anchor_by_idx[current_w_idx] = {
                        "offset": offset,
                        "pre": pre,
                        "ovl": ovl,
                        "cons": consonant,
                        "cutoff": cutoff,
                        "pre_abs": offset + pre,
                        "cons_abs": offset + consonant,
                        "onset_abs": c_start,
                        "vowel_start_abs": n_start,
                        "c_end_abs": c_end,
                        "vowel_end_abs": n_end,
                        "vowel_len": max(n_end - n_start, 20.0),
                        "cons_gap": max(consonant - pre, 10.0),
                        "cut_gap": max(abs(cutoff) - consonant, 16.0),
                    }

                aliases_to_write = generate_ja_openutau_aliases(alias) if generate_openutau else [alias]
                for a in aliases_to_write:
                    a2 = _alias_out(a)
                    new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                    final_lines.append(new_line)

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

    # 저장
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            for line in final_lines:
                f.write(line + "\n")
        log(f"✅ 일본어 OTO 1차 생성 완료! 저장 경로: {out_path}")
    except Exception as e:
        err = f"❌ OTO 저장 실패: {e}"
        logger.error(err)
        errors.append(err)

    # Hidden auto-calibration path:
    # if a manually aligned reference oto exists, keep learning internally
    # while maintaining the same user workflow.
    try:
        ref_oto = _find_reference_oto(out_path)
        if ref_oto:
            trained = _train_ja_autotune_profile(out_path, ref_oto, custom_map=custom_map)
            if trained and _save_ja_autotune_profile(profile_path, trained):
                b_count = len((trained.get("buckets") or {}))
                pair_count = int(trained.get("matched_pairs", 0))
                log(f"[AutoTune] 내부 튜닝 프로파일 갱신 완료: pairs={pair_count}, buckets={b_count}")
                log(f"[AutoTune] 프로파일 저장: {profile_path}")
                # Bootstrapping pass on first profile creation.
                if not had_profile_on_start:
                    changed = _apply_profile_to_oto_file(out_path, trained, custom_map=custom_map)
                    if changed > 0:
                        log(f"[AutoTune] 프로파일 1차 적용 완료: {changed} lines adjusted")
            else:
                log("[AutoTune] 참고 OTO를 찾았지만 프로파일 학습용 매칭 샘플이 충분하지 않습니다.")
    except Exception as e:
        log(f"[AutoTune] 내부 튜닝 프로파일 갱신 실패: {e}")

    # Default runtime path: apply mel-based refinement for end users as well.
    try:
        wav_dir_for_mel = os.path.dirname(os.path.abspath(tg_folder.rstrip("\\/")))
        try:
            from core.oto_ml_refiner import apply_oto_ml_to_oto_file

            ml_changed = apply_oto_ml_to_oto_file(
                "japanese",
                out_path,
                tg_dir=tg_folder,
                wav_dir=wav_dir_for_mel,
                custom_phonemes_path=custom_phonemes_path,
                callback=log,
                enabled=enable_ml_correction,
                format_override=(forced_format or fallback_format or "cvvc"),
            )
            if ml_changed > 0:
                log(f"[OTO-ML] 일본어 수치 보정 적용: {ml_changed} lines")
        except Exception as ml_e:
            log(f"[OTO-ML] 일본어 보정 스킵: {ml_e}")
        mel_changed = _apply_ja_mel_refine_to_oto_file(
            out_path, wav_dir_for_mel, custom_map=custom_map
        )
        if mel_changed > 0:
            log(f"[JA-Mel] 멜 에너지 기반 cutoff 보정 적용: {mel_changed} lines")
    except Exception as e:
        log(f"[JA-Mel] 보정 스킵: {e}")

    try:
        wav_dir_for_export = os.path.dirname(os.path.abspath(tg_folder.rstrip("\\/")))
        final_changed = _convert_ja_internal_cutoff_to_oto_field(out_path, wav_dir_for_export)
        if final_changed > 0:
            log(f"[JA-Finalize] cutoff 좌표계 변환 적용: {final_changed} lines")
    except Exception as e:
        log(f"[JA-Finalize] cutoff 변환 스킵: {e}")

    if anchor_stats["anchor_locked_count"] > 0:
        log(
            "[AnchorLock] 요약: "
            f"anchor_locked_count={anchor_stats['anchor_locked_count']}, "
            f"cutoff_clamped_count={anchor_stats['cutoff_clamped_count']}, "
            f"vc_cutoff_leak_guard_count={anchor_stats['vc_cutoff_leak_guard_count']}"
        )
        log(f"[AnchorLock] 상세 로그: {anchor_log_path}")

    if ja_mapping_trace_logging and mapping_trace_records:
        try:
            os.makedirs(_anchor_log_dir, exist_ok=True)
            with open(mapping_trace_path, "w", encoding="utf-8") as f:
                for row in mapping_trace_records:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log(f"[JA-Mapping] trace 로그: {mapping_trace_path}")
            log(f"[JA-Mapping] trace 건수: {len(mapping_trace_records)}")
        except Exception as e:
            log(f"[JA-Mapping] trace 저장 스킵: {e}")

    if cleanup_timing_jsonl:
        removed_count, failed_count = _cleanup_timing_anchor_jsonl_files(
            _anchor_log_dir,
            "timing_anchor_ja_",
        )
        if removed_count > 0:
            log(f"[AnchorLock] timing jsonl 자동 정리: {removed_count} files")
        if failed_count > 0:
            log(f"[AnchorLock] timing jsonl 정리 실패: {failed_count} files")

    _log_unset_summary()
    return processed, total, errors

