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
from core.ja_lab_generator import romaji_to_ipa, parse_ja_filename, split_ja_romaji_syllable
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
    return m in {'a', 'i', 'ɯ', 'u', 'e', 'o', 'ɴ', 'n'}


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


def _normalize_alias_for_profile(alias):
    a = re.sub(r"\s+", " ", (alias or "").strip().lower())
    # treat trailing suffix like _C4 as metadata
    a = re.sub(r"_[a-z0-9]{1,8}$", "", a)
    return a


def _read_oto_rows_for_profile(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
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
        "oto.manual.ini",
        "oto.correct.ini",
        "oto.reference.ini",
        "oto.gold.ini",
        "oto.human.ini",
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
        msg = f"❌ 템플릿 파일을 찾을 수 없습니다: {tpl_path}"
        log(msg)
        return 0, 0, [msg]

    # 템플릿 읽기
    template_lines = []
    if tpl_path:
        lines, detected_enc, warning, err = load_template_oto_lines(
            tpl_path,
            require_utf8=False,
            mode_label="일본어 모드",
        )
        if err:
            log(err)
            return 0, 0, [err]
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

    final_lines = []

    # 커스텀 맵 로드
    custom_map = load_custom_phonemes(custom_phonemes_path)

    # 템플릿 사용 (일반 모드)
    if template_lines:
        file_groups = {}
        for line in template_lines:
            fname = line.split('=')[0]
            if fname not in file_groups:
                file_groups[fname] = []
            file_groups[fname].append(line)
    else:
        # 템플릿 없는 자동 생성 모드 (Auto-Generation)
        log(f"⚡ 템플릿 없음 → {fallback_format.upper()} 포맷으로 자동 에일리어스 생성 시작")
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
                local_format = fallback_format
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

            def _post_adjust_params(offset, consonant, cutoff, pre, ovl, alias_type='cv', alias_text=''):
                offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)

                pre_abs = offset + pre
                if boundary_points_ms:
                    nearest = min(boundary_points_ms, key=lambda x: abs(x - pre_abs))
                    snap_limit = 320.0 if alias_type in ('vc', 'vv', 'vcv') else 260.0
                    if abs(pre_abs - nearest) > snap_limit:
                        pre_abs = nearest

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
                        a2 = apply_alias_suffix(a, alias_suffix)
                        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                        final_lines.append(new_line)
                processed += 1
                continue

            # === 다중 음절 처리: 파일명/리스트 우선 매핑 ===
            syllables_info = []
            sparse_phone_mode = bool(
                filename_syllables and len(ph_intervals) < max(4, len(filename_syllables) // 2)
            )
            filename_based = _build_syllables_from_filename(ph_intervals, filename_syllables)
            if filename_based and len(filename_based) >= 1:
                syllables_info = filename_based
                log(f"🧭 {fname}: 파일명 우선 음절 매핑 사용 ({len(filename_syllables)}음절)")
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

            if not syllables_info or any(len(s['phones']) == 0 for s in syllables_info):
                log(f"⚠️ {fname}: 음소-음절 매핑 실패 → 원본 유지")
                _record_unset_lines("mapping_failed", fname, lines)
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue

            # CV 순서 카운터로 리스트 순서 우선 매핑
            current_w_idx = 0
            cv_seq_idx = 0
            vc_seq_idx = 0

            for line_num, line in enumerate(lines):
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

                alias_type = classify_ja_alias(alias, custom_map)

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
                        a2 = apply_alias_suffix(a, alias_suffix)
                        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                        final_lines.append(new_line)
                    continue

                is_vc = alias_type in ('vc', 'vv')
                is_vcv = alias_type == 'vcv'
                is_cv_head = alias_type == 'cv_head'

                if tail_breath:
                    if current_w_idx >= len(syllables_info):
                        current_w_idx = len(syllables_info) - 1

                    curr_syl = syllables_info[current_w_idx]
                    curr_phones = curr_syl['phones']
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
                        a2 = apply_alias_suffix(a, alias_suffix)
                        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                        final_lines.append(new_line)
                    continue

                # === VCV 연속음 처리 ===
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
                    ovl = _adaptive_ja_overlap(pre, c_hint, mode="vcv")
                    
                    vowel_len = n_end - c_boundary
                    added_cons = min(vowel_len * 0.5, 200)
                    if added_cons < 80: added_cons = 80
                    consonant = pre + added_cons
                    cutoff = -(consonant + vowel_len * 0.25)
                    
                    offset, consonant, cutoff, pre, ovl = _post_adjust_params(
                        offset, consonant, cutoff, pre, ovl, alias_type='vcv', alias_text=alias
                    )
                    
                    aliases_to_write = generate_ja_openutau_aliases(alias) if generate_openutau else [alias]
                    for a in aliases_to_write:
                        a2 = apply_alias_suffix(a, alias_suffix)
                        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                        final_lines.append(new_line)
                    continue

                # === 어두 CV (- a) 처리 ===
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
                        
                    offset = max(c_start - 50, 0)
                    pre = c_end - offset if c_end > c_start else 30
                    c_hint = curr_phones[0].mark if curr_phones else ""
                    ovl = _adaptive_ja_overlap(pre, c_hint, mode="cv_head")
                    
                    cv_vowel_len = n_end - n_start
                    added_cons = min(cv_vowel_len * 0.5, 150)
                    if added_cons < 80: added_cons = 80
                    consonant = pre + added_cons
                    cutoff = -(consonant + cv_vowel_len * 0.25)
                    
                    offset, consonant, cutoff, pre, ovl = _post_adjust_params(
                        offset, consonant, cutoff, pre, ovl, alias_type='cv_head', alias_text=alias
                    )
                    
                    aliases_to_write = generate_ja_openutau_aliases(alias) if generate_openutau else [alias]
                    for a in aliases_to_write:
                        a2 = apply_alias_suffix(a, alias_suffix)
                        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                        final_lines.append(new_line)
                    continue

                # === 기존 CVC 매핑 ===
                if not is_vc:
                    # CV 에일리어스: 순서대로 다음 음절에 매핑
                    if cv_seq_idx < len(syllables_info):
                        current_w_idx = cv_seq_idx
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
                    c_len = c_end - c_start
                    boundary = c_end

                    offset = c_start - 50
                    if offset < 0:
                        offset = 0

                    pre = boundary - offset
                    if pre < 10: pre = 10
                    c_hint = curr_phones[0].mark if curr_phones else ""
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

                    # VC는 pre 기준점을 약간 앞당겨(다음 자음 시작 직전) 박자 밀림을 완화한다.
                    if alias_type == 'vc':
                        pre_lead = min(max(transition_gap * 0.45, 10.0), 32.0)
                        boundary = max(c_end + 8.0, boundary - pre_lead)

                    offset_padding = 180
                    if v_len < offset_padding:
                        offset_padding = max(v_len * 0.8, 50)

                    offset = boundary - offset_padding
                    pre = boundary - offset

                    c_char = get_vc_consonant(alias)
                    ovl_mode = 'vv' if alias_type == 'vv' else 'vc'
                    ovl = _adaptive_ja_overlap(pre, c_char, mode=ovl_mode)

                    # VC/VV overlap은 앞 모음의 끝(c_end) 근처에 오도록 보정한다.
                    # 절대 위치 기준으로 맞춘 뒤, pre보다 작게 유지한다.
                    if pre > 0:
                        if alias_type == 'vc':
                            tail_margin = min(max(v_len * 0.08, 4.0), 18.0)
                            target_ovl_abs = c_end - tail_margin
                            upper_ovl = max(pre - 6.0, 0.0)
                            lower_ovl = min(pre * 0.58, upper_ovl)
                            ovl = min(upper_ovl, max(lower_ovl, target_ovl_abs - offset))
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

                    if is_plosive:
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

                offset, consonant, cutoff, pre, ovl = _post_adjust_params(
                    offset, consonant, cutoff, pre, ovl, alias_type=alias_type, alias_text=alias
                )

                aliases_to_write = generate_ja_openutau_aliases(alias) if generate_openutau else [alias]
                for a in aliases_to_write:
                    a2 = apply_alias_suffix(a, alias_suffix)
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
                            a2 = apply_alias_suffix(a, alias_suffix)
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

    _log_unset_summary()
    return processed, total, errors
