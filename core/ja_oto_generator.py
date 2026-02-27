"""
日本語 CVVC OTO.ini 自動生成モジュール
- TextGrid 基準の OTO パラメータ計算
- CV, VC, VV, V タイプの自動分類と処理
- リスト順優先マッピング (List-Order-First) アルゴリズム
"""

import os
import re
import json
import datetime
import logging
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
    """파일명을 정규화된 키로 변환"""
    base = os.path.splitext(name)[0]
    clean = re.sub(r"[^a-zA-Z0-9]", "", base)
    return clean.lower()


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

def classify_ja_alias(alias, custom_map=None):
    """
    일본어 에일리어스 유형을 분류합니다.
    
    Returns:
        'br'   - 숨소리 (br)
        'cv_head' - 어두 CV (- か)
        'vcv'  - VCV 연속음 (a か)
        'vc'   - VC 연단음 (a k)
        'vv'   - VV 모음전환 (a i)
        'cv'   - CV 단독음 (か)
        'mono' - 단모음 (あ)
    """
    clean = alias.strip()
    
    # 숨소리
    if is_breath(clean):
        return 'br'
        
    # 커스텀 맵 강제 검사 (특수발음 단독)
    if custom_map and clean in custom_map:
        mapped_val = custom_map[clean].lower()
        if mapped_val in ['r', 'h', 'sil', 'br', 'pau', 'sp']:
            return 'br'
        
        # 로마자 변환 후 모음 여부 체크
        ipa = romaji_to_ipa(mapped_val)
        if ipa and (ipa in ['a', 'i', '?', 'e', 'o'] or _is_ja_vowel_token(mapped_val)):
            return 'mono'
        return 'cv'
    
    # 띄어쓰기 있는 경우 (2토큰)
    if ' ' in clean:
        parts = clean.split()
        left = parts[0]
        right = ' '.join(parts[1:])
        
        # 어두 CV
        if left == '-':
            return 'cv_head'
            
        # VCV 판별 로직
        left_is_vowel = _is_ja_vowel_token(left)
        
        if left_is_vowel:
            # "o -" 같은 종결형은 VCV가 아니라 VV 성격으로 처리
            if right.strip() == '-':
                return 'vv'
            if re.fullmatch(r"^[^0-9A-Za-z\u3040-\u30ff\u31f0-\u31ff]+$", right.strip()):
                return 'vc'
            # 오른쪽이 알파벳 자음만 있는 경우 -> VC
            if right.lower() in JA_CONSONANTS:
                return 'vc'
            # 오른쪽이 로마자 모음이거나 히라가나 모음인 경우 -> VV
            if _is_ja_vowel_token(right):
                return 'vv'
            _, right_vowel = split_ja_romaji_syllable(right.lower())
            if right_vowel not in JA_VOWELS:
                return 'vc'
            # 그 외 (명확한 CV 음절) -> VCV
            return 'vcv'
            
        return 'vc'
        
    # 단모음 판별
    if _is_ja_vowel_token(clean) or clean == '-':
        return 'mono'

    clean_lower = clean.lower()
    for v in JA_VOWELS:
        if clean_lower.startswith(v) and len(clean_lower) > len(v):
            tail = clean_lower[len(v):]
            if tail in JA_CONSONANTS:
                return 'vc'
            if _is_ja_vowel_token(tail):
                return 'vv'

    onset, vowel = split_ja_romaji_syllable(clean_lower)
    if vowel in JA_VOWELS:
        return 'mono' if not onset else 'cv'
    return 'cv'


def detect_ja_alias_format(alias_list, custom_map=None):
    """
    템플릿 에일리어스 목록으로 파일의 녹음 포맷을 판별합니다.
    """
    if not alias_list:
        return 'cvc'
    
    types = [classify_ja_alias(a, custom_map) for a in alias_list]
    type_set = set(types)
    
    if type_set == {'br'}: return 'br'
    if type_set <= {'mono', 'cv_head', 'cv'}: return 'mono'
    if 'vcv' in type_set: return 'vcv'
    
    non_br = type_set - {'br'}
    if non_br <= {'vc', 'vv'}: return 'vc_only'
    if 'vv' in type_set: return 'cvvc'
    
    return 'cvc'

def get_vc_consonant(alias):
    """VC 에일리어스에서 자음 파트를 추출합니다."""
    parts = alias.strip().split()
    right = parts[1].lower() if len(parts) >= 2 else alias.strip().lower()
    if not right:
        return ''
    if right in JA_CONSONANTS:
        return right
    onset, _ = split_ja_romaji_syllable(right)
    if onset:
        return onset
    return '' if _is_ja_vowel_token(right) else right


def _clean_phone_mark(mark):
    return re.sub(r"[0-9]", "", (mark or "").strip().lower())


def _adaptive_ja_overlap(pre, consonant_hint="", mode="cv"):
    """
    子音タイプとエイリアスタイプに応じて overlap を動的調整する.
    """
    p = max(float(pre), 0.0)
    if p <= 0:
        return 0.0

    c = _clean_phone_mark(consonant_hint)
    hard = {"k", "g", "t", "d", "p", "b", "q", "c", "ch", "ts", "dz", "ky", "gy", "ty", "dy", "py", "by"}
    fric = {"s", "z", "sh", "h", "f", "v", "hy"}
    sonorant = {"m", "n", "ny", "r", "l", "ry", "w", "y"}

    base_by_mode = {
        "cv": 0.46,
        "cv_head": 0.40,
        "vcv": 0.50,
        "vc": 0.44,
        "vv": 0.55,
    }
    ratio = base_by_mode.get(mode, 0.46)

    if c in hard:
        ratio -= 0.16
    elif c in fric:
        ratio += 0.05
    elif c in sonorant:
        ratio += 0.08

    ratio = max(0.20, min(0.72, ratio))
    ovl = p * ratio
    if mode in ("vc", "vv", "vcv"):
        ovl = min(ovl, max(p - 14.0, 0.0))
    return max(0.0, min(ovl, p))


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


def _extract_ja_cv_targets_from_lines(lines, custom_map=None):
    targets = []
    for line in lines or []:
        if "=" not in line:
            continue
        alias = line.split("=", 1)[1].split(",", 1)[0].strip()
        if not alias:
            continue
        a_type = classify_ja_alias(alias, custom_map)
        tok = _alias_to_ja_cv_target(alias, a_type)
        if tok:
            targets.append(tok)
    if not targets:
        return []
    collapsed = []
    for t in targets:
        if not collapsed or collapsed[-1] != t:
            collapsed.append(t)
    return collapsed


def _build_ja_syllables_from_phone_nuclei(ph_intervals, cv_targets):
    if not ph_intervals or not cv_targets:
        return None

    target_n = len(cv_targets)
    nuclei = [i for i, p in enumerate(ph_intervals) if _is_nucleus_phone(getattr(p, "mark", ""))]
    if len(nuclei) < target_n:
        return None

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

    infos = []
    global_start = float(ph_intervals[0].minTime)
    global_end = float(ph_intervals[-1].maxTime)
    for i, n_idx in enumerate(selected):
        if i == 0:
            s_t = global_start
        else:
            prev_n = selected[i - 1]
            s_t = (float(ph_intervals[prev_n].maxTime) + float(ph_intervals[n_idx].minTime)) * 0.5

        if i + 1 < len(selected):
            next_n = selected[i + 1]
            e_t = (float(ph_intervals[n_idx].maxTime) + float(ph_intervals[next_n].minTime)) * 0.5
        else:
            e_t = global_end

        phones = [
            p for p in ph_intervals
            if float(p.minTime) >= s_t - 1e-6 and float(p.maxTime) <= e_t + 1e-6
        ]
        if not phones:
            phones = [ph_intervals[n_idx]]

        tok = cv_targets[i] if i < len(cv_targets) else ""
        infos.append({
            'word': tok,
            'roman': tok,
            'start_time': s_t,
            'end_time': e_t,
            'phones': list(phones),
        })

    return infos if infos else None


def _score_ja_syllable_mapping(candidate_infos, cv_targets):
    """candidate 음절열이 alias 기반 cv_targets와 얼마나 일치하는지 점수화."""
    if not candidate_infos or not cv_targets:
        return -1.0

    cand = []
    for s in candidate_infos:
        tok = _syllable_info_token(s)
        if tok:
            cand.append(tok)
    if not cand:
        return -1.0

    targets = [_normalize_ja_syllable_token(t) for t in cv_targets if t]
    if not targets:
        return -1.0

    def _avg_with_shift(shift):
        vals = []
        for i, tgt in enumerate(targets):
            j = i + shift
            if 0 <= j < len(cand):
                vals.append(_vcv_syllable_match_score(tgt, cand[j]))
        if not vals:
            return -1.0
        coverage = len(vals) / float(max(len(targets), 1))
        length_penalty = abs(len(cand) - len(targets)) * 4.0
        return (sum(vals) / float(len(vals))) * coverage - length_penalty

    return max(_avg_with_shift(0), _avg_with_shift(1), _avg_with_shift(-1))


def _build_syllables_from_filename(ph_intervals, filename_syllables):
    """
    파일명 음절 순서를 기준으로 phones를 음절 단위로 재분할합니다.
    List/filename-first 매핑의 핵심.
    """
    if not ph_intervals or not filename_syllables:
        return None

    target_n = len(filename_syllables)
    nuclei = [i for i, p in enumerate(ph_intervals) if _is_nucleus_phone(p.mark)]
    if len(nuclei) < target_n:
        return None

    if target_n == 1:
        selected = [nuclei[0]]
    elif len(nuclei) == target_n:
        selected = list(nuclei)
    else:
        # nucleus가 더 많을 때는 파일명 음절 수에 맞춰 단조 증가 샘플링
        selected = []
        prev = -1
        last_idx = len(nuclei) - 1
        for i in range(target_n):
            pos = round(i * last_idx / (target_n - 1))
            cand = nuclei[pos]
            if cand <= prev:
                cand = prev + 1
            if cand >= len(ph_intervals):
                cand = len(ph_intervals) - 1
            selected.append(cand)
            prev = cand

    infos = []
    start_idx = 0
    for i, n_idx in enumerate(selected):
        if n_idx < start_idx:
            return None
        phones = ph_intervals[start_idx:n_idx + 1]
        if not phones:
            return None
        infos.append({
            'word': filename_syllables[i],
            'start_time': phones[0].minTime,
            'end_time': phones[-1].maxTime,
            'phones': list(phones),
        })
        start_idx = n_idx + 1

    # 남은 phone 꼬리는 마지막 음절에 합침
    if infos and start_idx < len(ph_intervals):
        infos[-1]['phones'].extend(ph_intervals[start_idx:])
        infos[-1]['end_time'] = infos[-1]['phones'][-1].maxTime

    return infos if infos else None


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


def _normalize_ja_syllable_token(token):
    t_raw = (token or "").strip()
    if not t_raw:
        return ""
    t = t_raw.lower()
    if re.search(r'[\u3041-\u3096\u30A1-\u30FA\u30FC]', t_raw):
        syls = parse_ja_filename(t_raw)
        if syls:
            t = syls[0].lower()
    t = t.replace("'", "").strip("-_ ")
    if not t:
        return ""
    if t in {"n", "nn", "xn", "m"}:
        return "n"
    onset, vowel = split_ja_romaji_syllable(t)
    if vowel in JA_VOWELS:
        return f"{onset}{vowel}" if onset else vowel
    return t


def _extract_vcv_target_syllable(alias):
    parts = (alias or "").strip().split()
    if len(parts) >= 2:
        return _normalize_ja_syllable_token(parts[1])
    return _normalize_ja_syllable_token(alias)


def _syllable_info_token(syl_info):
    if not isinstance(syl_info, dict):
        return ""
    for k in ("roman", "word"):
        if k in syl_info:
            tok = _normalize_ja_syllable_token(syl_info.get(k, ""))
            if tok:
                return tok
    phones = syl_info.get("phones") or []
    marks = [_clean_phone_mark(getattr(p, "mark", "")) for p in phones]
    marks = [m for m in marks if m and m not in {"sil", "sp", "spn", "pau"}]
    if not marks:
        return ""
    onset = ""
    vowel = ""
    for m in marks:
        if m in {"a", "i", "u", "ɯ", "e", "o", "ɴ", "n"}:
            vowel = "u" if m == "ɯ" else ("n" if m in {"ɴ", "n"} else m)
            break
        if not onset:
            onset = m
    if vowel:
        return f"{onset}{vowel}" if onset and vowel != "n" else vowel
    return onset


def _vcv_syllable_match_score(target, candidate):
    t = _normalize_ja_syllable_token(target)
    c = _normalize_ja_syllable_token(candidate)
    if not t or not c:
        return 0
    if t == c:
        return 100

    to, tv = split_ja_romaji_syllable(t)
    co, cv = split_ja_romaji_syllable(c)
    score = 0
    if tv and cv and tv == cv:
        score += 60
    if to and co:
        if to == co:
            score += 35
        elif to.startswith(co) or co.startswith(to):
            score += 22
        elif to[0] == co[0]:
            score += 10
    if t in c or c in t:
        score += 8
    return score


def _select_vcv_syllable_index(alias, expected_idx, syllables_info):
    if not syllables_info:
        return 0
    n = len(syllables_info)
    e = max(0, min(int(expected_idx), n - 1))
    target = _extract_vcv_target_syllable(alias)
    if not target:
        return e

    start = max(0, e - 2)
    end = min(n, e + 4)
    best_idx = e
    best_score = -10**9
    for i in range(start, end):
        cand = _syllable_info_token(syllables_info[i])
        score = _vcv_syllable_match_score(target, cand) - abs(i - e) * 7
        if score > best_score:
            best_score = score
            best_idx = i
    if best_score >= 56:
        return best_idx
    return e


def _extract_ja_cv_target_syllable(alias, alias_type="cv"):
    parts = (alias or "").strip().split()
    if not parts:
        return ""
    if alias_type in {"cv_head", "vcv"} and len(parts) >= 2:
        return _normalize_ja_syllable_token(parts[1])
    if len(parts) >= 2 and parts[0] == "-":
        return _normalize_ja_syllable_token(parts[1])
    return _normalize_ja_syllable_token(parts[-1])


def _extract_ja_onset_token(token):
    t = _normalize_ja_syllable_token(token)
    if not t:
        return ""
    onset, vowel = split_ja_romaji_syllable(t)
    if vowel in JA_VOWELS:
        return (onset or "").lower()
    if t in JA_VOWELS:
        return ""
    return t


def _ja_onset_class(onset):
    o = (onset or "").strip().lower()
    if not o:
        return "other"
    if o in JA_NASAL_ONSETS or o.startswith("m"):
        return "nasal"
    if o in JA_VOICED_ONSETS:
        return "voiced"
    if o in JA_VOICELESS_ONSETS:
        return "voiceless"
    return "other"


def _ja_cv_onset_class(alias_text, c_hint="", alias_type="cv"):
    onset = _ja_extract_onset_for_timing(alias_text, alias_type=alias_type)
    if not onset:
        hint = re.sub(r"[^a-z]", "", (c_hint or "").lower())
        onset = hint
    return _ja_onset_class(onset), onset


def _ja_cv_offset_and_pre(c_start, c_end, alias_text, c_hint="", alias_type="cv"):
    """
    JA CV/CV_HEAD의 과도한 선행 리드를 줄이기 위해 onset 계열별 offset/pre를 계산합니다.
    """
    cls, onset = _ja_cv_onset_class(alias_text, c_hint=c_hint, alias_type=alias_type)
    is_head = alias_type == "cv_head"

    lead = 30.0
    if cls == "nasal":
        lead = 18.0
    elif cls == "voiced":
        lead = 24.0
    elif cls == "voiceless":
        lead = 34.0
    if onset in JA_SIBILANT_ONSETS or onset in JA_FRICATIVE_ONSETS:
        lead += 6.0
    if onset in {"m", "n", "ny", "r", "l", "ry"}:
        lead -= 4.0
    if is_head:
        lead -= 4.0
    lead = _clamp_range(lead, 12.0, 50.0)

    offset = max(float(c_start) - lead, 0.0)
    pre = float(c_end) - offset if c_end > c_start else (28.0 if is_head else 24.0)

    if c_end > c_start:
        c_len = max(float(c_end) - float(c_start), 8.0)
        min_pre = max(20.0, c_len + 8.0)
        if cls in {"nasal", "voiced"}:
            max_pre = c_len + 64.0
        elif cls == "voiceless":
            max_pre = c_len + 54.0
        else:
            max_pre = c_len + 58.0
        if is_head:
            max_pre += 10.0
        pre = _clamp_range(pre, min_pre, max_pre)
        offset = max(float(c_end) - pre, 0.0)

    return offset, pre


def _select_ja_cv_syllable_index(alias, expected_idx, syllables_info, alias_type="cv"):
    if not syllables_info:
        return 0
    n = len(syllables_info)
    e = max(0, min(int(expected_idx), n - 1))
    target = _extract_ja_cv_target_syllable(alias, alias_type=alias_type)
    if not target:
        return e

    start = max(0, e - 1)
    end = min(n, e + 5)
    best_idx = e
    best_score = -10**9
    target_onset = _extract_ja_onset_token(target)
    target_cls = _ja_onset_class(target_onset)
    dist_penalty = 7 if target_cls in {"nasal", "voiced"} else 6

    def _score_idx(i):
        cand = _syllable_info_token(syllables_info[i])
        cand_onset = _extract_ja_onset_token(cand)
        cand_cls = _ja_onset_class(cand_onset)
        score = _vcv_syllable_match_score(target, cand) - abs(i - e) * dist_penalty
        # m/n/ny 계열은 한 음절 밀림이 체감이 커서 비호환 onset에 강한 페널티를 준다.
        if target_cls == "nasal" and cand_cls != "nasal":
            score -= 18
        elif target_cls == "voiced" and cand_cls == "voiceless":
            score -= 12
        elif target_cls == "voiceless" and cand_cls == "voiced":
            score -= 8
        if target_onset and cand_onset and target_onset[:1] == cand_onset[:1]:
            score += 4
        return score

    expected_score = _score_idx(e)
    for i in range(start, end):
        score = _score_idx(i)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_score >= 54:
        # Diphthong/glide rows can over-jump; keep expected index when nearly tied.
        hold_margin = 14 if target_cls in {"nasal", "voiced"} else 16
        best_gain = best_score - expected_score
        expected_tok = _syllable_info_token(syllables_info[e])
        best_tok = _syllable_info_token(syllables_info[best_idx])
        _to, target_vowel = split_ja_romaji_syllable(target)
        _eo, expected_vowel = split_ja_romaji_syllable(expected_tok)
        _bo, best_vowel = split_ja_romaji_syllable(best_tok)
        expected_onset = _extract_ja_onset_token(expected_tok)
        best_onset = _extract_ja_onset_token(best_tok)
        same_vowel_expected = bool(target_vowel and expected_vowel and target_vowel == expected_vowel)
        best_vowel_match = bool(target_vowel and best_vowel and target_vowel == best_vowel)
        same_onset_expected = bool(
            target_onset and expected_onset and (target_onset == expected_onset or target_onset[:1] == expected_onset[:1])
        )
        best_onset_match = bool(
            target_onset and best_onset and (target_onset == best_onset or target_onset[:1] == best_onset[:1])
        )

        if best_idx > e and expected_score >= max(50, best_score - hold_margin):
            return e
        # 한 음절 점프는 충분한 점수 이득이 없으면 유지(일/한 공통 안전장치).
        if abs(best_idx - e) == 1:
            min_gain = 22
            if target_cls in {"nasal", "voiced"}:
                min_gain = 20
            if same_vowel_expected:
                min_gain = min(min_gain, 18)
            if best_gain < min_gain:
                return e
            if same_vowel_expected and (not best_vowel_match) and best_gain < 20:
                return e
            if same_onset_expected and (not best_onset_match) and best_gain < 24:
                return e
        if best_idx < e and best_gain < 24:
            return e
        return best_idx
    return e


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
    for raw in lines or []:
        if "=" not in raw:
            expanded.append(raw)
            continue

        rhs = raw.split("=", 1)[1]
        alias = rhs.split(",", 1)[0].strip()
        if not alias:
            expanded.append(raw)
            continue

        alias_type = classify_ja_alias(alias, custom_map)
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

    pre_lead = _clamp_range(
        transition_gap * profile.get("pre_lead_mul", 0.35),
        profile.get("pre_lead_min", 8.0),
        profile.get("pre_lead_max", 30.0),
    )
    boundary = max(float(prev_v_end) + 6.0, float(c_boundary) - pre_lead)

    base_pad = profile.get("offset_pad", 86.0)
    dyn_pad = base_pad + max(prev_v_len - 120.0, 0.0) * profile.get("offset_len_mul", 0.08)
    pad_lo = profile.get("offset_pad_min", 42.0)
    pad_hi = min(260.0, max(prev_v_len * 0.94, base_pad + 40.0))
    offset_padding = _clamp_range(dyn_pad, pad_lo, pad_hi)
    if prev_v_len < offset_padding:
        offset_padding = max(prev_v_len * 0.76, profile.get("offset_pad_floor", 36.0))

    offset = max(boundary - offset_padding, 0.0)
    pre = max(boundary - offset, 8.0)

    tail_margin = profile.get("tail_margin_base", 10.0) + prev_v_len * profile.get("tail_margin_mul", 0.05)
    tail_margin = _clamp_range(tail_margin, 4.0, 24.0)
    target_ovl_abs = float(prev_v_end) - tail_margin
    upper_ovl = max(pre - profile.get("ovl_pre_margin", 6.0), 0.0)
    lower_ovl = min(pre * profile.get("ovl_min_ratio", 0.40), upper_ovl)
    ovl_anchored = min(upper_ovl, max(lower_ovl, target_ovl_abs - offset))
    ovl = _blend(ovl_anchored, pre * profile.get("ovl_ratio", 0.50), 0.34)
    ovl = min(upper_ovl, max(lower_ovl, ovl))

    n_ref = max(curr_v_len, 60.0)
    cons_add = profile.get("cons_add_base", 36.0) + max(n_ref - 70.0, 0.0) * profile.get("cons_add_mul", 0.12)
    cons_add = _clamp_range(
        cons_add,
        profile.get("cons_add_min", 20.0),
        profile.get("cons_add_max", 68.0),
    )
    consonant = pre + cons_add
    consonant = max(consonant, pre + profile.get("cons_floor", 18.0))
    consonant = min(consonant, pre + max(curr_v_len * 0.72, 96.0))

    cut_add = profile.get("cut_add_base", 58.0) + max(n_ref - 70.0, 0.0) * profile.get("cut_add_mul", 0.20)
    cut_add = _clamp_range(
        cut_add,
        profile.get("cut_add_min", 34.0),
        profile.get("cut_add_max", 120.0),
    )
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
    for k, a_row in auto_map.items():
        r_row = ref_map.get(k)
        if not r_row:
            continue
        alias_for_cls = re.sub(r"_[A-Za-z0-9]{1,8}$", "", a_row["alias"])
        alias_type = classify_ja_alias(alias_for_cls, custom_map)
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
        head_off_t = _clamp_range(stat.get("head_offset", offset), 0.0, 2200.0)
        offset = _blend(offset, head_off_t, min(0.20, w * 0.7))

    return validate_oto_params(offset, cons_new, cutoff_new, pre_new, ovl_new)


def _apply_profile_to_oto_file(oto_path, profile, custom_map=None):
    if not oto_path or not os.path.exists(oto_path):
        return 0
    changed = 0
    out_lines = []
    with open(oto_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            p = _parse_oto_line_profile(raw)
            if not p:
                out_lines.append(raw.rstrip("\n"))
                continue
            alias_for_cls = re.sub(r"_[A-Za-z0-9]{1,8}$", "", p["alias"])
            alias_type = classify_ja_alias(alias_for_cls, custom_map)
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
                wav_index[normalize_key(fn)] = os.path.join(wav_dir, fn)
    except Exception:
        pass

    by_wav = {}
    for line_idx, row in parsed:
        by_wav.setdefault(row["wav"], []).append((line_idx, row))

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
            alias_type = classify_ja_alias(alias, custom_map)
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
                t2 = classify_ja_alias(a2, custom_map)
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
    fallback_format='cvvc',
    custom_phonemes_path='',
    alias_suffix='',
    alias_style='original',
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
        fallback_format: 템플릿이 없을 때 사용할 포맷 ('cvc', 'cvvc', 'vcv')
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
            forced_format = 'cvc'
            fallback_format = 'cvc'
        elif af.startswith('cvvc'):
            forced_format = 'cvvc'
            fallback_format = 'cvvc'
        elif af.startswith('vcv'):
            forced_format = 'vcv'
            fallback_format = 'vcv'

    auto_gen_format = (fallback_format or "cvvc").strip().lower()
    if auto_gen_format not in {"cvvc", "vcv"}:
        log_msg = (
            f"⚠️ 자동 에일리어스 생성은 현재 CVVC/VCV만 지원합니다. "
            f"{auto_gen_format.upper()} -> CVVC로 전환합니다."
        )
        if callback:
            callback(log_msg)
        auto_gen_format = "cvvc"

    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    def _alias_out(a):
        a_conv = convert_ja_alias_style(a, alias_style=alias_style)
        return apply_alias_suffix(a_conv, alias_suffix)

    errors = []
    skipped_entries = []

    def _record_unset(reason, fname, line):
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
        })

    def _record_unset_lines(reason, fname, src_lines):
        for raw in (src_lines or []):
            _record_unset(reason, fname, raw)

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
            for it in skipped_entries:
                if it["reason"] != reason:
                    continue
                alias_txt = it["alias"] if it["alias"] else "<empty>"
                log(f"    예시: {it['file']} | alias={alias_txt}")
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
            tg_norm_map.setdefault(info['norm_key'], []).append(info)

    def _resolve_tg_info(fname):
        """템플릿의 wav 파일명을 TextGrid 엔트리에 안정적으로 매핑합니다."""
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
            log(f"⚠️ 파일명 매핑 충돌: {wav_name} (정규화 키 {norm_name}, 후보 {len(candidates)}개) → 원본 파일명 유지")
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
                    wav_index_for_signal[normalize_key(fn)] = os.path.join(wav_root_for_signal, fn)
    except Exception:
        pass
    mel_cache_for_signal = {}

    for fname, lines in file_groups.items():
        tg_info = _resolve_tg_info(fname)

        if not tg_info:
            log(f"📝 {fname}: TextGrid 없음 → 원본 유지")
            _record_unset_lines("textgrid_missing", fname, lines)
            final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
            processed += 1
            continue
            
        tg_path = tg_info['path']
        real_wav_name = tg_info['real_name']
        wav_path_for_signal = _find_wav_path_for_name(real_wav_name, wav_root_for_signal, wav_index_for_signal)
        mel_ctx_for_file = None
        if wav_path_for_signal:
            mel_ctx_for_file = mel_cache_for_signal.get(wav_path_for_signal)
            if mel_ctx_for_file is None:
                audio_sig, sr_sig = _read_wav_mono_np(wav_path_for_signal)
                mel_ctx_for_file = _mel_envelope(audio_sig, sr_sig)
                mel_cache_for_signal[wav_path_for_signal] = mel_ctx_for_file
        
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
            wd_intervals = []
            if word_tier:
                wd_intervals = [i for i in word_tier if i.mark.strip().lower() not in silence_marks]

            ph_intervals = [i for i in ph_tier if i.mark.strip().lower() not in silence_marks]
            if not ph_intervals and wd_intervals:
                base_name_for_synth = os.path.splitext(real_wav_name)[0]
                filename_syls_for_synth = parse_ja_filename(base_name_for_synth)
                synth_phones = []
                for idx, w in enumerate(wd_intervals):
                    w_mark = w.mark.strip()
                    if w_mark.lower() in {'<unk>', 'unk', 'spn'} and idx < len(filename_syls_for_synth):
                        w_mark = filename_syls_for_synth[idx]
                    synth_phones.extend(
                        _synthesize_word_phones(w_mark, float(w.minTime), float(w.maxTime))
                    )
                ph_intervals = synth_phones
            if not ph_intervals:
                log(f"⚠️ {fname}: 음소 정보 없음 → 원본 유지")
                _record_unset_lines("empty_intervals", fname, lines)
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

            def _nearest_phone_edge_ms(anchor_ms):
                nearest = None
                nearest_dist = float("inf")
                for s_ms, e_ms in phone_spans_ms:
                    if s_ms <= anchor_ms <= e_ms:
                        return anchor_ms, 0.0
                    ds = abs(anchor_ms - s_ms)
                    de = abs(anchor_ms - e_ms)
                    if ds < nearest_dist:
                        nearest_dist = ds
                        nearest = s_ms
                    if de < nearest_dist:
                        nearest_dist = de
                        nearest = e_ms
                if nearest is None:
                    return anchor_ms, 0.0
                return nearest, nearest_dist

            def _surrounding_gap_ms(anchor_ms):
                prev_end = None
                next_start = None
                for s_ms, e_ms in phone_spans_ms:
                    if s_ms <= anchor_ms <= e_ms:
                        return None, None, 0.0
                    if e_ms < anchor_ms:
                        prev_end = e_ms
                    elif s_ms > anchor_ms and next_start is None:
                        next_start = s_ms
                        break
                if prev_end is None or next_start is None or next_start <= prev_end:
                    return None, None, 0.0
                return prev_end, next_start, (next_start - prev_end)

            def _post_adjust_params(offset, consonant, cutoff, pre, ovl, alias_type='cv', alias_text=''):
                offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)

                pre_abs = offset + pre
                nearest_edge, nearest_dist = _nearest_phone_edge_ms(pre_abs)
                prev_end_ms, next_start_ms, gap_len_ms = _surrounding_gap_ms(pre_abs)
                # 정렬 오차로 pre가 무음 gap에 걸리면 alias 타입별로 안전한 경계로 스냅한다.
                if gap_len_ms >= 55.0 or nearest_dist > 34.0:
                    target = nearest_edge
                    if prev_end_ms is not None and next_start_ms is not None:
                        if alias_type in ('vc', 'vv'):
                            target = max(prev_end_ms + 4.0, next_start_ms - 6.0)
                        elif alias_type in ('cv', 'cv_head'):
                            target = max(prev_end_ms + 3.0, next_start_ms - 4.0)
                        else:
                            target = prev_end_ms
                    if abs(target - pre_abs) >= 2.0:
                        pre_abs = target

                min_pre_abs = max(timeline_start_ms - (20.0 if alias_type in ('vc', 'vv', 'vcv') else 10.0), 0.0)
                max_pre_abs = effective_end_ms + (30.0 if alias_type in ('vc', 'vv', 'vcv') else 80.0)
                pre_abs = max(min_pre_abs, min(pre_abs, max_pre_abs))

                offset_floor = max(timeline_start_ms - (70.0 if alias_type in ('vc', 'vv', 'vcv') else 40.0), 0.0)
                if offset < offset_floor:
                    offset = offset_floor

                if pre_abs - offset > 340.0:
                    offset = max(pre_abs - 340.0, 0.0)
                if pre_abs < offset:
                    pre_abs = offset + 10.0

                pre = max(pre_abs - offset, 0.0)
                if ovl > pre:
                    ovl = pre * 0.72
                if consonant < pre + 25.0:
                    consonant = pre + 25.0

                max_cons_abs = max((effective_end_ms + 80.0) - offset, pre + 40.0)
                consonant = min(consonant, max_cons_abs)
                max_cut_abs = max((effective_end_ms + 120.0) - offset, consonant + 35.0)
                cutoff = -min(abs(cutoff), max_cut_abs)
                offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
                if ja_style_enabled and ja_style_profile and not autotune_profile:
                    offset, consonant, cutoff, pre, ovl = _apply_ja_style_profile(
                        alias_type, offset, consonant, cutoff, pre, ovl, ja_style_profile, alias_text=alias_text
                    )
                if autotune_profile:
                    offset, consonant, cutoff, pre, ovl = _apply_ja_autotune_profile(
                        alias_type, offset, consonant, cutoff, pre, ovl, autotune_profile
                    )
                return validate_oto_params(offset, consonant, cutoff, pre, ovl)
            
            base_name = os.path.splitext(real_wav_name)[0]
            filename_syllables = parse_ja_filename(base_name)
            is_vowel_chain = _is_vowel_chain_syllables(filename_syllables)

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
            
            # 발음 처리 로직 시작
            aliases_found = []
            
            for line in lines:
                parts = line.split('=')
                if len(parts) < 2:
                    continue
                
                alias_def = parts[1].split(',')
                alias = alias_def[0].strip()
                
                if not alias:
                    continue
                
                a_type = classify_ja_alias(alias, custom_map)

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

                    offset, consonant, cutoff, pre, ovl = _post_adjust_params(
                        offset, consonant, cutoff, pre, ovl, alias_type='br', alias_text=alias
                    )

                    aliases_to_write = generate_ja_openutau_aliases(alias) if generate_openutau else [alias]
                    for a in aliases_to_write:
                        a2 = _alias_out(a)
                        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                        final_lines.append(new_line)
                processed += 1
                continue

            # === 다중 음절 처리: 파일명/리스트 우선 매핑 ===
            syllables_info = []
            sparse_phone_mode = bool(
                filename_syllables and len(ph_intervals) < max(4, len(filename_syllables) // 2)
            )
            cv_targets = _extract_ja_cv_targets_from_lines(lines, custom_map)
            filename_based = _build_syllables_from_filename(ph_intervals, filename_syllables)
            alias_based = _build_ja_syllables_from_phone_nuclei(ph_intervals, cv_targets) if cv_targets else None
            if filename_based and len(filename_based) >= 1:
                syllables_info = filename_based
                log(f"🧭 {fname}: 파일명 우선 음절 매핑 사용 ({len(filename_syllables)}음절)")
            elif alias_based:
                syllables_info = alias_based
                log(f"🧭 {fname}: 파일명 매핑 실패 → alias/phone 기반 음절 매핑 사용 ({len(cv_targets)}음절)")
            elif wd_intervals and sparse_phone_mode:
                log(f"🧭 {fname}: phones 희소 감지({len(ph_intervals)}개) → words 기반 합성 phone 매핑 사용")
                for w in wd_intervals:
                    w_start = float(w.minTime)
                    w_end = float(w.maxTime)
                    s_phones = _synthesize_word_phones(w.mark.strip(), w_start, w_end)
                    syllables_info.append({
                        'word': w.mark.strip().lower(),
                        'start_time': w_start,
                        'end_time': w_end,
                        'phones': s_phones
                    })
            elif wd_intervals:
                for w in wd_intervals:
                    w_start = w.minTime
                    w_end = w.maxTime
                    s_phones = [p for p in ph_intervals if p.minTime >= w_start - 0.01 and p.maxTime <= w_end + 0.01]
                    syllables_info.append({
                        'word': w.mark.strip().lower(),
                        'start_time': w_start,
                        'end_time': w_end,
                        'phones': s_phones
                    })
            else:
                # words 티어가 없고 파일명 기반 분할도 실패한 경우 fallback
                current_phones = []
                for p in ph_intervals:
                    mark = p.mark.strip().lower()
                    if current_phones and mark in ['a', 'i', 'ɯ', 'e', 'o', 'ɴ', 'cl']:
                        if mark in ['ɴ', 'cl'] and len(current_phones) > 0:
                            current_phones.append(p)
                            syllables_info.append({
                                'word': ''.join(cp.mark.strip() for cp in current_phones),
                                'start_time': current_phones[0].minTime,
                                'end_time': current_phones[-1].maxTime,
                                'phones': list(current_phones)
                            })
                            current_phones = []
                            continue
                        if current_phones:
                            syllables_info.append({
                                'word': ''.join(cp.mark.strip() for cp in current_phones),
                                'start_time': current_phones[0].minTime,
                                'end_time': current_phones[-1].maxTime,
                                'phones': list(current_phones)
                            })
                        current_phones = [p]
                    else:
                        current_phones.append(p)

                if current_phones:
                    syllables_info.append({
                        'word': ''.join(cp.mark.strip() for cp in current_phones),
                        'start_time': current_phones[0].minTime,
                        'end_time': current_phones[-1].maxTime,
                        'phones': list(current_phones)
                    })

            if syllables_info and alias_based and cv_targets:
                base_score = _score_ja_syllable_mapping(syllables_info, cv_targets)
                alt_score = _score_ja_syllable_mapping(alias_based, cv_targets)
                # TextGrid 정렬 결과를 우선하되, alias/filename 기준과 크게 어긋난 경우에만 보정한다.
                if base_score < 66.0 and alt_score >= 70.0 and alt_score >= (base_score + 8.0):
                    syllables_info = alias_based
                    log(
                        f"🧭 {fname}: 매핑 이탈 보정 적용 "
                        f"(base={base_score:.1f}, corrected={alt_score:.1f})"
                    )
                else:
                    log(
                        f"🧭 {fname}: TextGrid 매핑 유지 "
                        f"(base={base_score:.1f}, corrected={alt_score:.1f})"
                    )

            if not syllables_info or any(len(s['phones']) == 0 for s in syllables_info):
                log(f"⚠️ {fname}: 음소-음절 매핑 실패 → 원본 유지")
                _record_unset_lines("mapping_failed", fname, lines)
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue

            lines_for_mapping = lines
            if format_type in ('cvvc', 'cvc'):
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

                if len(curr_phones) >= 2:
                    c_start = curr_phones[0].minTime * 1000
                    c_end = curr_phones[-1].minTime * 1000
                    n_start = curr_phones[-1].minTime * 1000
                    n_end = curr_phones[-1].maxTime * 1000
                else:
                    c_start = curr_phones[0].minTime * 1000
                    c_end = c_start
                    n_start = c_start
                    n_end = curr_phones[0].maxTime * 1000

                c_hint = curr_phones[0].mark if curr_phones else ""
                anchor_alias = syl.get('roman_cv') or syl.get('roman') or syl.get('word') or ""
                cv_vowel_len = max(n_end - n_start, 20.0)
                offset, pre = _ja_cv_offset_and_pre(
                    c_start, c_end, anchor_alias, c_hint=c_hint, alias_type="cv"
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

            def _guard_ja_cv_cutoff_to_next_onset(offset, consonant, cutoff, pre, syll_idx):
                """CV/CV_HEAD cutoff이 다음 음절 onset을 넘어가지 않도록 제한."""
                if syll_idx is None or syll_idx < 0:
                    return offset, consonant, cutoff, pre, 0.0
                if (syll_idx + 1) >= len(syllables_info):
                    return offset, consonant, cutoff, pre, 0.0

                next_syl = syllables_info[syll_idx + 1]
                next_phones = next_syl.get("phones") or []
                if not next_phones:
                    return offset, consonant, cutoff, pre, 0.0

                next_mark = _clean_phone_mark(getattr(next_phones[0], "mark", ""))
                hard_next = (
                    next_mark in JA_PLOSIVE_CONSONANTS
                    or next_mark in JA_SIBILANT_ONSETS
                    or next_mark in {"ts", "ch", "j", "sh", "s", "z", "h"}
                )
                safety = 14.0 if hard_next else 9.0
                next_onset_rel = (next_phones[0].minTime * 1000.0) - offset
                max_cutoff_abs = next_onset_rel - safety
                if max_cutoff_abs <= (pre + 18.0):
                    return offset, consonant, cutoff, pre, 0.0

                original_cutoff_abs = abs(cutoff)
                consonant = min(consonant, max_cutoff_abs - 14.0)
                consonant = max(consonant, pre + 10.0)

                cutoff_abs = min(original_cutoff_abs, max_cutoff_abs)
                if cutoff_abs <= (consonant + 8.0):
                    cutoff_abs = min(max_cutoff_abs, consonant + 10.0)
                    if cutoff_abs <= (consonant + 6.0):
                        consonant = max(pre + 8.0, cutoff_abs - 10.0)
                cutoff = -cutoff_abs

                offset, consonant, cutoff, pre, _ovl = validate_oto_params(
                    offset, consonant, cutoff, pre, 0.0
                )
                reduction = max(0.0, original_cutoff_abs - abs(cutoff))
                return offset, consonant, cutoff, pre, reduction

            # CV 순서 카운터로 리스트 순서 우선 매핑
            current_w_idx = 0
            cv_seq_idx = 0
            vc_seq_idx = 0

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

                alias_type = classify_ja_alias(alias, custom_map)
                onset_hint_local = ""

                # 포맷 강제 시 VCV 오인식을 보정
                if format_type in ('cvvc', 'cvc') and alias_type == 'vcv':
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
                        br_len = 500
                    offset = max(br_start - 30, 0)
                    pre = 0
                    ovl = 0
                    consonant = min(br_len * 0.3, 100)
                    cutoff = -(br_len * 0.85)
                    offset, consonant, cutoff, pre, ovl = _post_adjust_params(
                        offset, consonant, cutoff, pre, ovl, alias_type='vc', alias_text=alias
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

                    offset, consonant, cutoff, pre, ovl = _post_adjust_params(
                        offset, consonant, cutoff, pre, ovl, alias_type='mono', alias_text=alias
                    )

                    aliases_to_write = generate_ja_openutau_aliases(alias) if generate_openutau else [alias]
                    for a in aliases_to_write:
                        a2 = _alias_out(a)
                        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                        final_lines.append(new_line)
                    continue

                # === VCV 연속음 처리 ===
                if is_vcv:
                    if cv_seq_idx < len(syllables_info):
                        expected_idx = cv_seq_idx
                    else:
                        expected_idx = len(syllables_info) - 1
                    mapped_idx = _select_vcv_syllable_index(alias, expected_idx, syllables_info)
                    if mapped_idx != expected_idx and abs(mapped_idx - expected_idx) <= 1:
                        log(f"🧭 {fname}: VCV 음절 정렬 보정 {expected_idx + 1}->{mapped_idx + 1} ({alias})")
                    current_w_idx = mapped_idx
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
                    
                    offset, consonant, cutoff, pre, ovl = _post_adjust_params(
                        offset, consonant, cutoff, pre, ovl, alias_type='vcv', alias_text=alias
                    )
                    
                    aliases_to_write = generate_ja_openutau_aliases(alias) if generate_openutau else [alias]
                    for a in aliases_to_write:
                        a2 = _alias_out(a)
                        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                        final_lines.append(new_line)
                    continue

                # === 어두 CV (- a) 처리 ===
                if is_cv_head:
                    if cv_seq_idx < len(syllables_info):
                        expected_idx = cv_seq_idx
                    else:
                        expected_idx = len(syllables_info) - 1
                    mapped_idx = _select_ja_cv_syllable_index(
                        alias, expected_idx, syllables_info, alias_type="cv_head"
                    )
                    if mapped_idx != expected_idx and abs(mapped_idx - expected_idx) <= 1:
                        log(f"🧭 {fname}: CV 음절 정렬 보정 {expected_idx + 1}->{mapped_idx + 1} ({alias})")
                    current_w_idx = mapped_idx
                    cv_seq_idx = current_w_idx + 1
                    if current_w_idx >= len(syllables_info):
                        current_w_idx = len(syllables_info) - 1
                        
                    curr_syl = syllables_info[current_w_idx]
                    curr_phones = curr_syl['phones']
                    
                    if len(curr_phones) >= 2:
                        c_start = curr_phones[0].minTime * 1000
                        c_end = curr_phones[-1].minTime * 1000
                        n_start = c_end
                        n_end = curr_phones[-1].maxTime * 1000
                    else:
                        c_start = curr_phones[0].minTime * 1000
                        c_end = c_start
                        n_start = c_start
                        n_end = curr_phones[0].maxTime * 1000
                        
                    c_hint = curr_phones[0].mark if curr_phones else ""
                    offset, pre = _ja_cv_offset_and_pre(
                        c_start, c_end, alias, c_hint=c_hint, alias_type="cv_head"
                    )
                    onset_hint_local = c_hint
                    ovl = _adaptive_ja_overlap(pre, c_hint, mode="cv_head")
                    
                    cv_vowel_len = n_end - n_start
                    added_cons = min(cv_vowel_len * 0.5, 150)
                    if added_cons < 80: added_cons = 80
                    consonant = pre + added_cons
                    cutoff = -(consonant + cv_vowel_len * 0.25)
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
                    
                    offset, consonant, cutoff, pre, ovl = _post_adjust_params(
                        offset, consonant, cutoff, pre, ovl, alias_type='cv_head', alias_text=alias
                    )
                    offset, consonant, cutoff, pre, cutoff_reduced = _guard_ja_cv_cutoff_to_next_onset(
                        offset, consonant, cutoff, pre, current_w_idx
                    )
                    if cutoff_reduced > 0.5:
                        log(f"🛡️ {fname}: CV 컷오프 과연장 보정(-{cutoff_reduced:.1f}ms) [{alias}]")
                    offset, consonant, cutoff, pre, ovl = validate_oto_params(
                        offset, consonant, cutoff, pre, ovl
                    )
                    
                    aliases_to_write = generate_ja_openutau_aliases(alias) if generate_openutau else [alias]
                    for a in aliases_to_write:
                        a2 = _alias_out(a)
                        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                        final_lines.append(new_line)
                    continue

                # === 기존 CVC 매핑 ===
                if not is_vc:
                    # CV 에일리어스: 순서대로 다음 음절에 매핑
                    if cv_seq_idx < len(syllables_info):
                        expected_idx = cv_seq_idx
                    else:
                        expected_idx = len(syllables_info) - 1
                    mapped_idx = _select_ja_cv_syllable_index(
                        alias, expected_idx, syllables_info, alias_type="cv"
                    )
                    if mapped_idx != expected_idx and abs(mapped_idx - expected_idx) <= 1:
                        log(f"🧭 {fname}: CV 음절 정렬 보정 {expected_idx + 1}->{mapped_idx + 1} ({alias})")
                    current_w_idx = mapped_idx
                    cv_seq_idx = current_w_idx + 1

                    if current_w_idx >= len(syllables_info):
                        current_w_idx = len(syllables_info) - 1

                    curr_syl = syllables_info[current_w_idx]
                    curr_phones = curr_syl['phones']

                    if len(curr_phones) >= 2:
                        # 자음 + 모음
                        c_start = curr_phones[0].minTime * 1000
                        c_end = curr_phones[-1].minTime * 1000
                        n_start = curr_phones[-1].minTime * 1000
                        n_end = curr_phones[-1].maxTime * 1000
                    else:
                        # 단독 모음 또는 특수 음소
                        c_start = curr_phones[0].minTime * 1000
                        c_end = c_start
                        n_start = c_start
                        n_end = curr_phones[0].maxTime * 1000

                    cv_vowel_len = n_end - n_start

                    c_hint = curr_phones[0].mark if curr_phones else ""
                    offset, pre = _ja_cv_offset_and_pre(
                        c_start, c_end, alias, c_hint=c_hint, alias_type="cv"
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

                    # VC는 pre 기준점을 약간 앞당겨(다음 자음 시작 직전) 박자 밀림을 완화한다.
                    if alias_type == 'vc':
                        pre_lead_mul = bridge_profile.get('pre_lead_mul', 0.35) if bridge_profile else 0.35
                        pre_lead_min = bridge_profile.get('pre_lead_min', 8.0) if bridge_profile else 8.0
                        pre_lead_max = bridge_profile.get('pre_lead_max', 32.0) if bridge_profile else 32.0
                        pre_lead = min(max(transition_gap * pre_lead_mul, pre_lead_min), pre_lead_max)
                        boundary = max(c_end + 8.0, boundary - pre_lead)

                    if alias_type == 'vc' and bridge_profile:
                        base_pad = bridge_profile.get('offset_pad', 86.0)
                        dyn_pad = base_pad + max(v_len - 140.0, 0.0) * bridge_profile.get('offset_len_mul', 0.08)
                        pad_lo = bridge_profile.get('offset_pad_min', 42.0)
                        pad_hi = min(240.0, max(v_len * 0.92, base_pad + 36.0))
                        offset_padding = _clamp_range(dyn_pad, pad_lo, pad_hi)
                        if v_len < offset_padding:
                            offset_padding = max(v_len * 0.78, bridge_profile.get('offset_pad_floor', 36.0))
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
                        and format_type in ('cvvc', 'cvc')
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
                        prev_cv_anchor = cv_anchor_by_idx.get(current_w_idx)
                        next_cv_anchor = cv_anchor_by_idx.get(current_w_idx + 1)
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
                offset, consonant, cutoff, pre, ovl = _post_adjust_params(
                    offset, consonant, cutoff, pre, ovl, alias_type=alias_type, alias_text=alias
                )
                if alias_type in {"cv", "cv_head"}:
                    offset, consonant, cutoff, pre, cutoff_reduced = _guard_ja_cv_cutoff_to_next_onset(
                        offset, consonant, cutoff, pre, current_w_idx
                    )
                    if cutoff_reduced > 0.5:
                        log(f"🛡️ {fname}: CV 컷오프 과연장 보정(-{cutoff_reduced:.1f}ms) [{alias}]")
                    offset, consonant, cutoff, pre, ovl = validate_oto_params(
                        offset, consonant, cutoff, pre, ovl
                    )

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
        log(f"✅ 일본어 CVVC OTO 생성 완료! 저장 경로: {out_path}")
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
        mel_changed = _apply_ja_mel_refine_to_oto_file(
            out_path, wav_dir_for_mel, custom_map=custom_map
        )
        if mel_changed > 0:
            log(f"[JA-Mel] 멜 에너지 기반 cutoff 보정 적용: {mel_changed} lines")
    except Exception as e:
        log(f"[JA-Mel] 보정 스킵: {e}")

    _log_unset_summary()
    return processed, total, errors
