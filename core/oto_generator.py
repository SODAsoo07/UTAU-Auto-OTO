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
import logging
import unicodedata
import textgrid
import copy
from core.lab_generator import load_custom_phonemes
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


def normalize_key(name):
    base = os.path.splitext(name)[0]
    clean = re.sub(r"[^a-zA-Z0-9・-德｣]", "", base)
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


def is_diphthong(alias):
    """에일리어스 문자열에 이중모음 패턴이 있는지 확인합니다."""
    clean = alias.replace(' ', '').lower()
    diphthongs = ['ya','yeo','yo','yu','ye','wa','wo','wi','we','weo','eui','ui']
    for diph in diphthongs:
        if diph in clean:
            return True
    return False


def clean_phone_mark(mark):
    return re.sub(r"[0-9]", "", mark).lower()


def normalize_ipa_mark(mark):
    """IPA 마크를 정규화해 비교 가능한 형태로 만듭니다."""
    base = clean_phone_mark(mark)
    nfd = unicodedata.normalize("NFD", base)
    stripped = "".join(ch for ch in nfd if not unicodedata.combining(ch))
    stripped = stripped.replace("?", "").replace("?", "").replace("?", "").replace("?", "")
    return stripped


def is_glide(phone_mark):
    clean = normalize_ipa_mark(phone_mark)
    return clean in ['j', 'w', 'y', '?', '?', '?']


# MFA phone tier에서 자주 보이는 모음/파열음 표기를 넓게 커버
IPA_VOWELS = {
    'a', 'e', 'i', 'o', 'u', 'y',
    'eo', 'eu', 'ae', 'oe', 'wa', 'we', 'wi', 'wo', 'ui',
}

IPA_PLOSIVES = {
    'k', 'g', 't', 'd', 'p', 'b', 'c', 'q', 'ch', 'j',
}


def find_vowel_phone(phones):
    """음소 시퀀스에서 대표 모음을 찾아 (index, phone)으로 반환합니다."""

    for idx, ph in enumerate(phones):
        mark = normalize_ipa_mark(ph.mark)
        if mark in IPA_VOWELS and not is_glide(mark):
            return idx, ph
    
    phones_list = list(phones)

    for idx, ph in enumerate(phones_list):
        if is_glide(ph.mark) and idx + 1 < len(phones_list):
            return idx + 1, phones_list[idx + 1]
    

    return len(phones_list) - 1, phones_list[-1]


def is_plosive_ipa(phone_mark):
    """IPA 표기가 파열음 계열인지 판별합니다."""
    clean = normalize_ipa_mark(phone_mark)
    if clean in IPA_PLOSIVES:
        return True
    # ???/??? ????? ??? ??? ?? (?: t? -> t)
    return clean in {'k', 't', 'p', 'c', 'g', 'd', 'b'}


def is_plosive_roman(consonant_str):
    """로마자 자음 문자열이 파열음 계열인지 판별합니다."""
    return consonant_str.lower() in [
        'g', 'gg', 'd', 'dd', 'b', 'bb', 'j', 'jj',
        'k', 'kk', 't', 'tt', 'p', 'pp', 'ch'
    ]


KR_TENSE_CONSONANTS = {
    "kk", "tt", "pp", "ss", "jj", "gg", "dd", "bb",
}

KR_SONORANT_CONSONANTS = {
    "n", "m", "r", "l", "ng", "y", "w", "ry", "ly", "ny", "my",
}


def _extract_alias_onset(alias):
    """CV 에일리어스에서 초성(로마자 자음군)을 추출합니다."""
    a = re.sub(r"[^a-z]", "", (alias or "").lower())
    if not a:
        return ""
    m = re.match(r"^([^aeiouyw]+)", a)
    return m.group(1) if m else ""


def _is_tense_consonant(ipa_hint="", roman_hint=""):
    i = normalize_ipa_mark(ipa_hint or "")
    r = re.sub(r"[^a-z]", "", (roman_hint or "").lower())
    return i in KR_TENSE_CONSONANTS or r in KR_TENSE_CONSONANTS


def _is_sonorant_consonant(ipa_hint="", roman_hint=""):
    i = normalize_ipa_mark(ipa_hint or "")
    r = re.sub(r"[^a-z]", "", (roman_hint or "").lower())
    return i in KR_SONORANT_CONSONANTS or r in KR_SONORANT_CONSONANTS

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


# ==============================================================================
# 에일리어스 분류용 상수
# ==============================================================================


KR_VOWELS = {
    'a', 'e', 'i', 'o', 'u', 'eo', 'eu', 'ae', 'oe', 'wi',
    'ya', 'yeo', 'yo', 'yu', 'ye', 'wa', 'wo', 'we', 'weo', 'eui', 'ui', 'wae'
}


KR_CONSONANTS = {
    'g', 'n', 'd', 'r', 'l', 'm', 'b', 's', 'j', 'ch', 'k', 't', 'p', 'h',
    'gg', 'dd', 'bb', 'ss', 'jj', 'kk', 'tt', 'pp',
    'ng', 'sh', 'f', 'z', 'q', 'c'
}


KR_BATCHIM_MARKERS = {'N', 'L', 'M', 'NG', 'K', 'T', 'P', 'H'}


GLOTTAL_MARKS = {'繝ｻ', '.'}


def _detect_glottal_kind(alias):
    """에일리어스의 성문 파열 표식 위치(head/tail)를 판별합니다."""
    a = (alias or "").strip()
    if not a:
        return None
    parts = a.split()
    if len(parts) >= 2:
        if parts[0] in GLOTTAL_MARKS:
            return 'head'
        if parts[-1] in GLOTTAL_MARKS:
            return 'tail'

    if a[0] in GLOTTAL_MARKS:
        return 'head'
    if a[-1] in GLOTTAL_MARKS:
        return 'tail'
    return None


def is_breath(alias):
    """숨소리(br, br1...) 에일리어스인지 판별합니다."""
    clean = alias.strip().lower()
    return bool(re.match(r'^br\d*$', clean))


def _extract_vowel_consonant(text):
    """로마자 음절에서 모음부와 자음부를 분리해 반환합니다."""
    text = text.lower().strip()

    for v in sorted(KR_VOWELS, key=len, reverse=True):
        if text.endswith(v):
            remainder = text[:-len(v)] if len(text) > len(v) else ""
            return v, remainder
    return None, text


def _normalize_cv_match_token(token):
    """CV 음절 매핑 비교용 정규화(r/l 혼용 등)를 적용합니다."""
    t = re.sub(r"[^a-z]", "", (token or "").lower())
    if not t:
        return t
    # ㄹ 초성 표기가 r/l로 혼용되는 경우를 흡수
    if t.startswith("l"):
        t = "r" + t[1:]
    t = t.replace("ly", "ry")
    return t


def _cv_match_score(alias_token, syllable_token):
    """에일리어스 CV와 음절 로마자 간 매핑 점수를 계산합니다."""
    a = _normalize_cv_match_token(alias_token)
    s = _normalize_cv_match_token(syllable_token)
    if not a or not s:
        return 0
    if a == s:
        return 100

    av, ac = _extract_vowel_consonant(a)
    sv, sc = _extract_vowel_consonant(s)
    score = 0

    if av and sv and av == sv:
        score += 65
    if ac and sc:
        if ac == sc:
            score += 30
        elif {ac, sc} <= {"r", "l"}:
            score += 26
    if a in s or s in a:
        score += 10
    return score


def classify_alias(alias, custom_map=None):
    """에일리어스 문자열을 br/cv/cv_head/vc/vv/vcv/mono 타입으로 분류합니다."""
    clean = alias.strip()
    

    if is_breath(clean):
        return 'br'
        

    if custom_map and clean in custom_map:
        mapped_val = custom_map[clean].lower()
        if mapped_val in ['r', 'h', 'sil', 'br']:
            return 'br'
        if mapped_val in KR_VOWELS:
            return 'mono'
        return 'cv'
    

    if ' ' in clean:
        parts = clean.split()
        left = parts[0].strip()
        right = ' '.join(parts[1:]).strip()
        

        if left == '-':
            return 'cv_head'
        

        if right == '-':
            return 'cv'


        left_lower = left.lower()
        right_lower = right.lower().rstrip('-')
        if left in GLOTTAL_MARKS and right_lower in KR_VOWELS:
            return 'cv_head'
        if left_lower in KR_VOWELS and right in GLOTTAL_MARKS:
            return 'vc'
        

        left_upper = left.upper()
        if left_upper in KR_BATCHIM_MARKERS:

            return 'vcv'
        

        left_lower = left.lower()
        right_lower = right.lower().rstrip('-')
        

        if left_lower in KR_VOWELS:

            if right_lower in KR_CONSONANTS or right.upper() in KR_BATCHIM_MARKERS:
                return 'vc'

            if right_lower in KR_VOWELS:
                return 'vv'

            return 'vcv'
        

        return 'vc'
    

    clean_lower = clean.lower().rstrip('-')


    gk = _detect_glottal_kind(clean)
    if gk == 'head':
        return 'cv_head'
    if gk == 'tail':
        return 'vc'
    

    if custom_map and clean_lower in custom_map:
        mapped_val = custom_map[clean_lower].lower()
        if mapped_val in ['sil', 'br', 'r', 'h']: return 'br'
        if mapped_val in KR_VOWELS: return 'mono'
        return 'cv'
    

    if clean_lower in KR_VOWELS:
        return 'mono'
    

    m = re.match(r'^([aoueiwy]+|eo|eu|ae|oe|wa|wo|we|ye|ya|yo|yu|wae|weo|eui|ui)([gknmdrlbsjtph]+|ng|kk|ss|pp|tt|jj|ch)$', clean_lower)
    if m:
        return 'vc'
    

    return 'cv'


def detect_alias_format(alias_list, custom_map=None):
    """파일 단위 에일리어스 목록의 전체 포맷(CVC/CVVC/VCV 등)을 추정합니다."""
    if not alias_list:
        return 'cvc'
    
    types = [classify_alias(a, custom_map) for a in alias_list]
    type_set = set(types)
    

    if type_set == {'br'}:
        return 'br'
    

    if type_set <= {'mono', 'cv_head', 'cv'}:
        return 'mono'
    

    if 'vcv' in type_set:
        return 'vcv'
    

    non_br = type_set - {'br'}
    if non_br <= {'vc', 'vv'}:
        return 'vc_only'
    

    if non_br <= {'cv', 'mono', 'cv_head'}:
        return 'cv_simple'
    

    if 'vv' in type_set:
        return 'cvvc'
    
    return 'cvc'


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
                alias_type = classify_alias(row["alias"], custom_map)
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


def _read_kr_oto_rows_for_profile(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
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
        alias_type = classify_alias(alias_for_cls, custom_map)
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
    with open(oto_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            row = _parse_oto_line_profile(raw)
            if not row:
                out_lines.append(raw.rstrip("\n"))
                continue

            alias_for_cls = re.sub(r"_[A-Za-z0-9]{1,8}$", "", row["alias"])
            alias_type = classify_alias(alias_for_cls, custom_map)
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
    for wav_name, rows in rows_by_wav.items():
        dur_ms = _wav_duration_ms(os.path.join(wav_dir, wav_name))
        for idx, row in enumerate(rows):
            alias_type = classify_alias(row["alias"], custom_map)
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
    fallback_format='cvvc',
    custom_phonemes_path='',
    alias_suffix='',
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

    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

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

    kr_profile = None
    profile_path = _default_kr_profile_cache_path()
    try:
        env_profile = os.environ.get("UTOA_KR_PROFILE_JSON", "").strip()
        if env_profile:
            kr_profile = _load_kr_reference_profile(env_profile)
            if kr_profile:
                log(f"[KR-Profile] 외부 프로파일 로드: {env_profile}")

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
        msg = f"템플릿 파일을 찾을 수 없습니다: {tpl_path}"
        log(msg)
        return 0, 0, [msg]


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
            return 0, 0, [err]
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

    final_lines = []


    custom_map = load_custom_phonemes(custom_phonemes_path)


    if template_lines:
        file_groups = {}
        for line in template_lines:
            fname = line.split('=')[0]
            if fname not in file_groups:
                file_groups[fname] = []
            file_groups[fname].append(line)
    else:

        log(f"템플릿이 없어 {fallback_format.upper()} 형식으로 에일리어스를 자동 생성합니다.")
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

                    for idx, w in enumerate(wd_intervals):
                        roman_parts = []
                        for ch in w.mark:
                            roman_parts.extend(decompose_hangul_to_roman(ch))
                        roman = "".join(roman_parts).lower()
                        
                        vowel_part, const_part = _extract_vowel_consonant(roman)
                        
                        if fallback_format == 'vcv':
                            if idx == 0:
                                lines.append(f"{real_name}=- {roman},0,0,0,0,0")
                            else:
                                prev_w = wd_intervals[idx-1]
                                prev_parts = []
                                for ch in prev_w.mark:
                                    prev_parts.extend(decompose_hangul_to_roman(ch))
                                prev_roman = "".join(prev_parts).lower()
                                prev_vowel, prev_cons = _extract_vowel_consonant(prev_roman)
                                
                                if prev_cons in KR_BATCHIM_MARKERS.union([c.lower() for c in KR_BATCHIM_MARKERS]):
                                    lines.append(f"{real_name}={prev_cons.upper()} {roman},0,0,0,0,0")
                                else:
                                    lines.append(f"{real_name}={prev_vowel} {roman},0,0,0,0,0")
                                
                        elif fallback_format == 'cvc' or fallback_format == 'cvvc':

                            lines.append(f"{real_name}={roman},0,0,0,0,0")
                            
                            if idx > 0 and fallback_format == 'cvvc':

                                prev_w = wd_intervals[idx-1]
                                prev_parts = []
                                for ch in prev_w.mark:
                                    prev_parts.extend(decompose_hangul_to_roman(ch))
                                prev_roman = "".join(prev_parts).lower()
                                prev_vowel, _ = _extract_vowel_consonant(prev_roman)
                                

                                cur_start_cons = ""
                                for char in roman:
                                    if char in KR_CONSONANTS:
                                        cur_start_cons += char
                                    else:
                                        break
                                
                                if cur_start_cons:
                                    lines.append(f"{real_name}={prev_vowel} {cur_start_cons},0,0,0,0,0")
                                else:

                                    cur_vowel, _ = _extract_vowel_consonant(roman)
                                    lines.append(f"{real_name}={prev_vowel} {cur_vowel},0,0,0,0,0")
                
                if lines:
                    file_groups[real_name] = lines

            except Exception as e:
                log(f"경고: {real_name} 자동 템플릿 생성 실패 ({e})")
                
    processed = 0
    total = len(file_groups)

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

            if not phone_tier or not word_tier:
                log(f"경고: {fname}: phones 또는 words tier가 없어 원본 라인을 유지합니다.")
                _record_unset_lines("tier_missing", fname, lines)
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue


            ph_intervals_all = [i for i in phone_tier if i.mark.strip() not in ['', 'spn', 'pau']]
            ph_intervals = [i for i in ph_intervals_all if i.mark.strip() not in ['sil']]
            wd_intervals = [i for i in word_tier if i.mark.strip() not in ['', 'sil', 'spn', 'pau']]

            if len(ph_intervals) == 0 or len(wd_intervals) == 0:
                log(f"경고: {fname}: 유효한 음소/단어 구간이 없어 원본 라인을 유지합니다.")
                _record_unset_lines("empty_intervals", fname, lines)
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue


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
                    
                    aliases_to_write = generate_openutau_aliases(alias) if generate_openutau else [alias]
                    for a in aliases_to_write:
                        a2 = apply_alias_suffix(a, alias_suffix)
                        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                        final_lines.append(new_line)
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
            file_format = detect_alias_format(alias_names)
            log(f"처리: {fname}: 형식 감지 -> {file_format.upper()}")
            
            is_vc_only = (file_format == 'vc_only')
            is_vcv_file = (file_format == 'vcv')


            try:
                from core.lab_generator import decompose_hangul_to_roman
            except ImportError:
                pass
            
            syllables_info = []
            for w in wd_intervals:
                w_start = w.minTime
                w_end = w.maxTime

                s_phones = [p for p in ph_intervals if p.minTime >= w_start - 0.01 and p.maxTime <= w_end + 0.01]
                roman_parts = []
                for ch in w.mark:
                    roman_parts.extend(decompose_hangul_to_roman(ch))
                
                syllables_info.append({
                    'word': w.mark,
                    'roman': "".join(roman_parts).lower(),
                    'start_time': w_start,
                    'end_time': w_end,
                    'phones': s_phones
                })
                

            if any(len(s['phones']) == 0 for s in syllables_info):
                log(f"경고: {fname}: 음절-음소 매핑 실패로 원본 라인을 유지합니다.")
                _record_unset_lines("mapping_failed", fname, lines)
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue
                
            romaji_syllables = [s['roman'] for s in syllables_info]
            current_w_idx = 0
            cv_seq_idx = 0
            
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
                

                alias_type = classify_alias(alias)
                

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
                    
                    aliases_to_write = generate_openutau_aliases(alias) if generate_openutau else [alias]
                    for a in aliases_to_write:
                        a2 = apply_alias_suffix(a, alias_suffix)
                        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                        final_lines.append(new_line)
                    continue
                

                is_vc = alias_type in ('vc', 'vv')
                is_vcv = alias_type == 'vcv'
                is_cv_head = alias_type == 'cv_head'
                is_diph = is_diphthong(alias)
                
                target_clean = re.sub(r'[^a-zA-Z]', '', alias.lower())




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
                    if offset < 0: offset = 0
                    

                    pre = c_boundary - offset
                    c_hint = curr_phones[0].mark if curr_phones else ""
                    ovl = adaptive_overlap(pre, c_hint, mode='vcv')
                    

                    vowel_len = n_end - c_boundary
                    added_cons = min(vowel_len * DIPHTHONG_CV_CONSONANT_RATIO, 150)
                    if added_cons < 50: added_cons = 50
                    consonant = pre + added_cons
                    cutoff = -(consonant + vowel_len * 0.25)
                    
                    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
                    
                    aliases_to_write = generate_openutau_aliases(alias) if generate_openutau else [alias]
                    for a in aliases_to_write:
                        a2 = apply_alias_suffix(a, alias_suffix)
                        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                        final_lines.append(new_line)
                    continue
                

                if is_cv_head:

                    if cv_seq_idx < len(syllables_info):
                        current_w_idx = cv_seq_idx
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
                    alias_onset = _extract_alias_onset(alias)
                    is_tense_cv = _is_tense_consonant(c_hint, alias_onset)
                    is_sonorant_cv = _is_sonorant_consonant(c_hint, alias_onset)

                    if is_tense_cv:
                        offset = max(c_start - 40, 0)
                    elif is_sonorant_cv:
                        offset = max(c_start - 90, 0)
                    else:
                        offset = max(c_start - 50, 0)

                    pre = c_end - offset if c_end > c_start else 30
                    ovl = adaptive_overlap(pre, c_hint, mode='cv_head')
                    if is_tense_cv:
                        ovl = min(ovl, max(pre * 0.34, 10.0))
                    elif is_sonorant_cv:
                        ovl = max(ovl, min(pre * 0.48, max(pre - 10.0, 0.0)))
                    
                    cv_vowel_len = n_end - n_start
                    if is_tense_cv:
                        added_cons = min(max(cv_vowel_len * 0.42, 68), 140)
                    elif is_sonorant_cv:
                        added_cons = min(max(cv_vowel_len * 0.58, 88), 190)
                    else:
                        added_cons = min(cv_vowel_len * 0.5, 150)
                        if added_cons < 80:
                            added_cons = 80
                    consonant = pre + added_cons
                    cutoff = -(consonant + cv_vowel_len * 0.25)
                    
                    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
                    
                    aliases_to_write = generate_openutau_aliases(alias) if generate_openutau else [alias]
                    for a in aliases_to_write:
                        a2 = apply_alias_suffix(a, alias_suffix)
                        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                        final_lines.append(new_line)
                    continue
                

                if not is_vc:

                    if cv_seq_idx < len(syllables_info):

                        name_match_idx = None
                        best_score = -1
                        # diphthong/r-l 혼용 케이스를 위해 탐색 범위를 넓혀 안정적으로 음절 정렬
                        scan_start = max(cv_seq_idx - 1, 0)
                        scan_end = min(cv_seq_idx + 5, len(romaji_syllables))
                        for i in range(scan_start, scan_end):
                            score = _cv_match_score(target_clean, romaji_syllables[i])
                            score -= abs(i - cv_seq_idx) * 4
                            if score > best_score:
                                best_score = score
                                name_match_idx = i
                            if score >= 98:
                                break
                        
                        if name_match_idx is not None and best_score >= 62:

                            current_w_idx = name_match_idx
                        else:


                            current_w_idx = cv_seq_idx
                        
                        cv_seq_idx = current_w_idx + 1

                            

                    if current_w_idx >= len(syllables_info):
                        current_w_idx = len(syllables_info) - 1
                    
                    curr_syl = syllables_info[current_w_idx]
                    curr_phones = curr_syl['phones']
                    

                    if len(curr_phones) >= 2:
                        v_idx, v_phone = find_vowel_phone(curr_phones)
                        c_start = curr_phones[0].minTime * 1000
                        c_end = v_phone.minTime * 1000
                        n_start = v_phone.minTime * 1000
                        n_end = v_phone.maxTime * 1000
                    else:
                        c_start = curr_phones[0].minTime * 1000
                        c_end = c_start
                        n_start = c_start
                        n_end = curr_phones[0].maxTime * 1000
                        
                else:


                    

                    if current_w_idx >= len(syllables_info):
                        current_w_idx = len(syllables_info) - 1
                            
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
                        
                cv_vowel_len = n_end - n_start
                if is_vc:

                    # VC should attach to the next consonant onset.
                    # Consonant-end anchoring often causes awkward late VC timing.
                    vc_target = n_start if alias_type == 'vc' else n_end
                    boundary = min(vc_target, c_end + 260)
                    v_len = c_end - c_start
                    n_len = n_end - n_start
                    
                    offset_padding = 180
                    if v_len < offset_padding:
                        offset_padding = max(v_len * 0.8, 50)
                        
                    offset = boundary - offset_padding
                    pre = boundary - offset
                    

                    c_char = ""
                    if ' ' in alias:
                        parts = alias.split()
                        if len(parts) >= 2: c_char = parts[1]
                    else:
                        m = re.match(r'^([aoueiwy]+|eo|eu|ae|oe|wa|wo|we|ye|ya|yo|yu|wae|weo|eui|ui)([gknmdrlbsjtph]+|ng|kk|ss|pp|tt|jj|ch)$', alias.lower())
                        if m: c_char = m.group(2)
                    
                    is_plosive_sibilant = c_char in ['g','k','kk','gg','d','t','tt','dd','b','p','bb','pp','s','ss','h','j','jj','ch']
                    ovl = adaptive_overlap(pre, c_char, mode='vv' if alias_type == 'vv' else 'vc')
                    
                    if is_plosive_sibilant:

                        n_ref = max(n_len, 100)
                        added_cons = min(max(n_ref * 0.35, 45), 95)
                        consonant = pre + added_cons
                        cutoff = -(consonant + max(n_len * 0.35, 45))
                    else:

                        n_ref = max(n_len, 80)
                        added_cons = min(max(n_ref * 0.55, 55), 160)
                        consonant = pre + added_cons
                        cutoff = -(consonant + max(n_len * 0.45, 50))

                else:
                    cv_vowel_len = n_end - n_start
                    
                    if is_diph:



                        boundary = c_end
                        # 이중모음 CV는 pre가 과도하게 길어지지 않도록 자음 길이 기반으로 제한
                        c_len = max(c_end - c_start, 10.0)
                        target_pre = max(72.0, min(148.0, c_len + 18.0))
                        offset = max(boundary - target_pre, 0)
                        
                        pre = boundary - offset
                        c_hint = curr_phones[0].mark if curr_phones else ""
                        alias_onset = _extract_alias_onset(alias)
                        is_tense_cv = _is_tense_consonant(c_hint, alias_onset)
                        is_sonorant_cv = _is_sonorant_consonant(c_hint, alias_onset)
                        if is_tense_cv:
                            target_pre = max(60.0, target_pre - 10.0)
                            offset = max(boundary - target_pre, 0)
                            pre = boundary - offset
                        elif is_sonorant_cv:
                            target_pre = min(164.0, target_pre + 12.0)
                            offset = max(boundary - target_pre, 0)
                            pre = boundary - offset
                        ovl = adaptive_overlap(pre, c_hint, mode='cv')
                        if is_tense_cv:
                            ovl = min(ovl, max(pre * 0.34, 10.0))
                        elif is_sonorant_cv:
                            ovl = max(ovl, min(pre * 0.50, max(pre - 8.0, 0.0)))

                        v_ref = max(cv_vowel_len, 140)
                        if is_tense_cv:
                            added_cons = min(max(v_ref * 0.48, 82), 190)
                        elif is_sonorant_cv:
                            added_cons = min(max(v_ref * 0.62, 98), 240)
                        else:
                            added_cons = min(max(v_ref * 0.55, 90), 230)
                        consonant = pre + added_cons
                        cutoff = -(consonant + max(cv_vowel_len * 0.2, 45))
                        
                    elif ' ' not in alias and len(alias) >= 2 and alias[0] in ['a','e','i','o','u','y','w'] and alias[-1] in ['a','e','i','o','u','y','w'] and alias.lower() not in ['eo', 'eu', 'ae', 'oe', 'wi']:
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


                        boundary = c_end
                        c_len = c_end - c_start
                        

                        first_phone_plosive = len(curr_phones) >= 2 and is_plosive_ipa(curr_phones[0].mark)
                        alias_consonant = re.match(r'^([^aeiouyw]+)', alias.lower())
                        roman_plosive = alias_consonant and is_plosive_roman(alias_consonant.group(1)) if alias_consonant else False
                        alias_onset = alias_consonant.group(1) if alias_consonant else ""
                        c_hint = curr_phones[0].mark if curr_phones else ""
                        is_tense_cv = _is_tense_consonant(c_hint, alias_onset)
                        is_sonorant_cv = _is_sonorant_consonant(c_hint, alias_onset)
                        is_plosive = first_phone_plosive or roman_plosive
                        
                        if is_tense_cv:
                            offset = max(c_start - 40, 0)
                            pre = boundary - offset
                            ovl = adaptive_overlap(pre, c_hint, mode='cv')
                            ovl = min(ovl, max(pre * 0.34, 10.0))
                        elif is_sonorant_cv:
                            offset = max(c_start - 92, 0)
                            pre = boundary - offset
                            ovl = adaptive_overlap(pre, c_hint, mode='cv')
                            ovl = max(ovl, min(pre * 0.50, max(pre - 8.0, 0.0)))
                        elif is_plosive:


                            offset = max(c_start - 50, 0)
                            pre = boundary - offset
                            ovl = adaptive_overlap(pre, c_hint, mode='cv')
                        else:


                            offset = max(c_start - 80, 0)
                            pre = boundary - offset         # C-V boundary
                            ovl = adaptive_overlap(pre, c_hint, mode='cv')
                        

                        v_ref = max(cv_vowel_len, 130)
                        if is_tense_cv:
                            added_cons = min(max(v_ref * 0.38, 62), 150)
                        elif is_sonorant_cv:
                            added_cons = min(max(v_ref * 0.52, 86), 210)
                        else:
                            added_cons = min(max(v_ref * 0.45, 70), 180)
                        consonant = pre + added_cons
                        cutoff = -(consonant + max(cv_vowel_len * 0.25, 45))



                offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)

                aliases_to_write = generate_openutau_aliases(alias) if generate_openutau else [alias]
                for a in aliases_to_write:
                    a2 = apply_alias_suffix(a, alias_suffix)
                    new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                    final_lines.append(new_line)

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
                            
                            aliases_to_write = generate_openutau_aliases(alias) if generate_openutau else [alias]
                            for a in aliases_to_write:
                                a2 = apply_alias_suffix(a, alias_suffix)
                                new_line = f"{tg_info['real_name']}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                                final_lines.append(new_line)
                except:
                    continue


    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            for line in final_lines:
                f.write(line + "\n")
        log(f"완료: OTO 파일 저장 -> {out_path}")
        if kr_profile:
            wav_dir_for_profile = os.path.dirname(os.path.abspath(tg_folder.rstrip("\\/")))
            changed = _apply_kr_profile_to_oto_file(
                out_path, wav_dir_for_profile, kr_profile, custom_map=custom_map
            )
            if changed > 0:
                log(f"[KR-Profile] 기준 프로파일 보정 적용: {changed} lines")
    except Exception as e:
        err = f"OTO 파일 저장 실패: {e}"
        logger.error(err)
        errors.append(err)

    _log_unset_summary()
    return processed, total, errors




