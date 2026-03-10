"""
日本語 Lab/辞書(Dictionary) ファイル自動生成モジュール
- ローマ字 -> IPA 変換
- Lab ファイル生成
- MFA用辞書ファイル生成

外国人アクセントにも対応するため、非標準ローマ字も広く受け入れます。
"""

import os
import re
import logging
import unicodedata
from typing import Any
from core.lab_generator import load_custom_phonemes, apply_custom_phonemes
from core.textio_utils import read_text_auto
from core.conversion_tables import load_japanese_conversion_table

logger = logging.getLogger(__name__)

_JA_KANA_RE = re.compile(r"[\u3041-\u3096\u30A1-\u30FA\u30FC]")
_HANGUL_RE = re.compile(r"[\uAC00-\uD7A3]")

# ==============================================================================
# 일본어 로마자 → IPA 매핑 (MFA japanese_mfa 음향 모델 기준)
# ==============================================================================

# 일본어 자음 IPA 매핑 (표준 + 비표준/외국인 표기 모두 포함)
JA_CONSONANT_IPA = {
    'k': 'k', 'g': 'ɡ', 's': 's', 'z': 'z',
    't': 't', 'd': 'd', 'n': 'n', 'h': 'h',
    'b': 'b', 'p': 'p', 'm': 'm', 'r': 'ɾ',
    'l': 'ɾ',  # 외국인 발음 대응: l → r
    'w': 'w', 'y': 'j',
    'f': 'ɸ', 'v': 'v',
}

# 일본어 모음 IPA 매핑
JA_VOWEL_IPA = {
    'a': 'a', 'i': 'i', 'u': 'ɯ', 'e': 'e', 'o': 'o',
}

# 특수 음절 직접 매핑 (자음+모음 결합이 비표준적인 경우)
JA_SPECIAL_SYLLABLE_IPA = {
    # し행
    'shi': 'ɕ i', 'si': 'ɕ i',
    'sha': 'ɕ a', 'shu': 'ɕ ɯ', 'she': 'ɕ e', 'sho': 'ɕ o',
    # ち행
    'chi': 'tɕ i', 'ti': 'tɕ i',
    'cha': 'tɕ a', 'chu': 'tɕ ɯ', 'che': 'tɕ e', 'cho': 'tɕ o',
    # つ행
    'tsu': 'ts ɯ', 'tu': 'ts ɯ',
    'tsa': 'ts a', 'tsi': 'ts i', 'tse': 'ts e', 'tso': 'ts o',
    # ふ행
    'fu': 'ɸ ɯ', 'hu': 'ɸ ɯ',
    # じ행
    'ji': 'dʑ i', 'zi': 'dʑ i',
    'ja': 'dʑ a', 'ju': 'dʑ ɯ', 'je': 'dʑ e', 'jo': 'dʑ o',
    # ぢ행 (발음은 じ와 동일하지만 표기가 다름)
    'di': 'dʑ i',
    # ん
    'n': 'ɴ', 'nn': 'ɴ', 'xn': 'ɴ',
    # っ (촉음) — MFA에서는 cl(closure)로 표현
    'q': 'cl', 'xtu': 'cl', 'xtsu': 'cl', 'ltsu': 'cl', 'ltu': 'cl',
    # 장모음 기호 (무시하거나 앞 모음 연장)
    '-': '',

    # 拗音 (요음) — 결합 자음+반모음+모음
    'kya': 'kʲ a', 'kyu': 'kʲ ɯ', 'kyo': 'kʲ o', 'kye': 'kʲ e', 'kyi': 'kʲ i',
    'gya': 'ɡʲ a', 'gyu': 'ɡʲ ɯ', 'gyo': 'ɡʲ o', 'gye': 'ɡʲ e', 'gyi': 'ɡʲ i',
    'sha': 'ɕ a', 'shu': 'ɕ ɯ', 'sho': 'ɕ o',
    'nya': 'ɲ a', 'nyu': 'ɲ ɯ', 'nyo': 'ɲ o', 'nye': 'ɲ e', 'nyi': 'ɲ i',
    'hya': 'ç a', 'hyu': 'ç ɯ', 'hyo': 'ç o', 'hye': 'ç e', 'hyi': 'ç i',
    'hi': 'ç i',
    'mya': 'mʲ a', 'myu': 'mʲ ɯ', 'myo': 'mʲ o', 'mye': 'mʲ e', 'myi': 'mʲ i',
    'rya': 'ɾʲ a', 'ryu': 'ɾʲ ɯ', 'ryo': 'ɾʲ o', 'rye': 'ɾʲ e', 'ryi': 'ɾʲ i',
    'bya': 'bʲ a', 'byu': 'bʲ ɯ', 'byo': 'bʲ o', 'bye': 'bʲ e', 'byi': 'bʲ i',
    'pya': 'pʲ a', 'pyu': 'pʲ ɯ', 'pyo': 'pʲ o', 'pye': 'pʲ e', 'pyi': 'pʲ i',
    'dya': 'dʲ a', 'dyu': 'dʲ ɯ', 'dyo': 'dʲ o', 'dye': 'dʲ e',
    'tya': 'tɕ a', 'tyu': 'tɕ ɯ', 'tyo': 'tɕ o', 'tye': 'tɕ e',

    # 외래어 표기
    'fa': 'ɸ a', 'fi': 'ɸ i', 'fe': 'ɸ e', 'fo': 'ɸ o',
    'wi': 'w i', 'we': 'w e', 'wo': 'w o',
    'va': 'v a', 'vi': 'v i', 'vu': 'v ɯ', 've': 'v e', 'vo': 'v o',
    'thi': 'θ i', 'tha': 'θ a', 'thu': 'θ ɯ', 'the': 'θ e', 'tho': 'θ o',
    'dhi': 'ð i', 'dha': 'ð a', 'dhu': 'ð ɯ', 'dhe': 'ð e', 'dho': 'ð o',

    # 외국인 발음 허용: 혼동되기 쉬운 표기
    'lya': 'ɾʲ a', 'lyu': 'ɾʲ ɯ', 'lyo': 'ɾʲ o',
    'la': 'ɾ a', 'li': 'ɾ i', 'lu': 'ɾ ɯ', 'le': 'ɾ e', 'lo': 'ɾ o',
}


def romaji_to_ipa(syllable):
    """
    일본어 로마자 음절 하나를 IPA 음소 리스트로 변환합니다.
    
    Args:
        syllable: 로마자 음절 (예: 'ka', 'shi', 'n')
    
    Returns:
        IPA 음소 문자열 (예: 'k a', 'ɕ i', 'ɴ')
    """
    raw = str(syllable or "").strip()
    s = raw.lower()
    if not s:
        return ''

    # 0차: 외부 변환 테이블 직접 매핑(대소문자 구분/무시 둘 다 지원)
    if raw in JA_ROMAJI_IPA_TABLE:
        return str(JA_ROMAJI_IPA_TABLE[raw] or "")
    if s in JA_ROMAJI_IPA_TABLE:
        return str(JA_ROMAJI_IPA_TABLE[s] or "")

    # 1차: 특수 음절 직접 매핑
    if s in JA_SPECIAL_SYLLABLE_IPA:
        return JA_SPECIAL_SYLLABLE_IPA[s]

    # 촉음이 붙은 로마자(예: kko, sshi) 보정: cl + 기본 음절
    if len(s) >= 3 and s[0] == s[1] and s[0] not in JA_VOWEL_IPA:
        tail_ipa = romaji_to_ipa(s[1:])
        if tail_ipa:
            return f"cl {tail_ipa}"

    # 2차: 단독 모음
    if s in JA_VOWEL_IPA:
        return JA_VOWEL_IPA[s]

    # 3차: 자음+모음 조합 (일반 규칙)
    # 긴 자음부터 시도 (sh, ch, ts 등)
    for clen in range(3, 0, -1):
        c_part = s[:clen]
        v_part = s[clen:]
        if c_part in JA_CONSONANT_IPA and v_part in JA_VOWEL_IPA:
            return f"{JA_CONSONANT_IPA[c_part]} {JA_VOWEL_IPA[v_part]}"

    # 4차: 그대로 반환 (알 수 없는 음절)
    logger.warning(f"알 수 없는 일본어 로마자: '{syllable}' → 그대로 사용")
    return s


JA_VOWELS_ROMAJI = {'a', 'i', 'u', 'e', 'o'}

# 2글자 우선 매칭 (요음/확장음)
KANA_COMBO_ROMAJI = {
    'きゃ': 'kya', 'きゅ': 'kyu', 'きょ': 'kyo', 'きぇ': 'kye',
    'ぎゃ': 'gya', 'ぎゅ': 'gyu', 'ぎょ': 'gyo', 'ぎぇ': 'gye',
    'しゃ': 'sha', 'しゅ': 'shu', 'しょ': 'sho', 'しぇ': 'she',
    'じゃ': 'ja', 'じゅ': 'ju', 'じょ': 'jo', 'じぇ': 'je',
    'ちゃ': 'cha', 'ちゅ': 'chu', 'ちょ': 'cho', 'ちぇ': 'che',
    'にゃ': 'nya', 'にゅ': 'nyu', 'にょ': 'nyo', 'にぇ': 'nye',
    'ひゃ': 'hya', 'ひゅ': 'hyu', 'ひょ': 'hyo', 'ひぇ': 'hye',
    'みゃ': 'mya', 'みゅ': 'myu', 'みょ': 'myo', 'みぇ': 'mye',
    'りゃ': 'rya', 'りゅ': 'ryu', 'りょ': 'ryo', 'りぇ': 'rye',
    'びゃ': 'bya', 'びゅ': 'byu', 'びょ': 'byo', 'びぇ': 'bye',
    'ぴゃ': 'pya', 'ぴゅ': 'pyu', 'ぴょ': 'pyo', 'ぴぇ': 'pye',
    'てぃ': 'ti', 'でぃ': 'di', 'とぅ': 'tu', 'どぅ': 'du',
    'つぁ': 'tsa', 'つぃ': 'tsi', 'つぇ': 'tse', 'つぉ': 'tso',
    'ふぁ': 'fa', 'ふぃ': 'fi', 'ふぇ': 'fe', 'ふぉ': 'fo',
    'うぁ': 'wa', 'うぃ': 'wi', 'うぇ': 'we', 'うぉ': 'wo',
    'ゔぁ': 'va', 'ゔぃ': 'vi', 'ゔ': 'vu', 'ゔぇ': 've', 'ゔぉ': 'vo',
    # 확장 소형모음 조합: 두 글자지만 한 모라로 취급해야 하는 경우.
    'すぁ': 'sa', 'すぃ': 'si', 'すぇ': 'se', 'すぉ': 'so',
    'ずぁ': 'za', 'ずぃ': 'zi', 'ずぇ': 'ze', 'ずぉ': 'zo',
    'しぃ': 'shi', 'じぃ': 'ji',
    'ちぃ': 'chi', 'ぢぃ': 'ji',
    'てぇ': 'te', 'でぇ': 'de',
    'とぉ': 'to', 'どぉ': 'do',
}

KANA_SINGLE_ROMAJI = {
    'あ': 'a', 'い': 'i', 'う': 'u', 'え': 'e', 'お': 'o',
    'か': 'ka', 'き': 'ki', 'く': 'ku', 'け': 'ke', 'こ': 'ko',
    'が': 'ga', 'ぎ': 'gi', 'ぐ': 'gu', 'げ': 'ge', 'ご': 'go',
    'さ': 'sa', 'し': 'shi', 'す': 'su', 'せ': 'se', 'そ': 'so',
    'ざ': 'za', 'じ': 'ji', 'ず': 'zu', 'ぜ': 'ze', 'ぞ': 'zo',
    'た': 'ta', 'ち': 'chi', 'つ': 'tsu', 'て': 'te', 'と': 'to',
    'だ': 'da', 'ぢ': 'ji', 'づ': 'zu', 'で': 'de', 'ど': 'do',
    'な': 'na', 'に': 'ni', 'ぬ': 'nu', 'ね': 'ne', 'の': 'no',
    'は': 'ha', 'ひ': 'hi', 'ふ': 'fu', 'へ': 'he', 'ほ': 'ho',
    'ば': 'ba', 'び': 'bi', 'ぶ': 'bu', 'べ': 'be', 'ぼ': 'bo',
    'ぱ': 'pa', 'ぴ': 'pi', 'ぷ': 'pu', 'ぺ': 'pe', 'ぽ': 'po',
    'ま': 'ma', 'み': 'mi', 'む': 'mu', 'め': 'me', 'も': 'mo',
    'や': 'ya', 'ゆ': 'yu', 'よ': 'yo',
    'ら': 'ra', 'り': 'ri', 'る': 'ru', 'れ': 're', 'ろ': 'ro',
    'わ': 'wa', 'ゐ': 'wi', 'ゑ': 'we', 'を': 'o',
    'ん': 'n',
    'ぁ': 'a', 'ぃ': 'i', 'ぅ': 'u', 'ぇ': 'e', 'ぉ': 'o',
}

ROMAJI_META_TOKENS = {
    'long', 'vcv', 'cvvc', 'cvc', 'mono', 'oto', 'wav'
}

JA_ROMAJI_IPA_TABLE = {}
JA_ALIAS_TO_KANA_TABLE = {}
JA_KANA_TO_ALIAS_TABLE = {}


def _pick_preferred_romaji(value: Any) -> str:
    candidates = value if isinstance(value, list) else [value]
    norm = []
    for cand in candidates:
        tok = str(cand or "").strip()
        if tok:
            norm.append(tok)
    if not norm:
        return ""
    norm.sort(key=lambda t: (not re.fullmatch(r"[a-z]+", t), len(t), t))
    return norm[0]


def _build_romaji_syllable_patterns():
    tokens = set()
    tokens.update({'a', 'i', 'u', 'e', 'o'})
    tokens.update(c + v for c in JA_CONSONANT_IPA.keys() for v in JA_VOWEL_IPA.keys())
    tokens.update(k for k in JA_SPECIAL_SYLLABLE_IPA.keys() if re.fullmatch(r'[a-z]+', k or ''))
    for k in JA_ROMAJI_IPA_TABLE.keys():
        key = str(k or "")
        if re.fullmatch(r"[A-Za-z]+", key):
            tokens.add(key.lower())
    for k in JA_ALIAS_TO_KANA_TABLE.keys():
        key = str(k or "")
        if re.fullmatch(r"[A-Za-z]+", key):
            tokens.add(key.lower())
    return sorted(tokens, key=len, reverse=True)


def _apply_external_japanese_conversion_table():
    global JA_ROMAJI_IPA_TABLE
    global JA_ALIAS_TO_KANA_TABLE
    global JA_KANA_TO_ALIAS_TABLE

    table = load_japanese_conversion_table() or {}
    alias_to_kana = dict(table.get("alias_to_kana", {}) or {})
    kana_to_alias = dict(table.get("kana_to_alias", {}) or {})
    romaji_to_ipa = dict(table.get("romaji_to_ipa", {}) or {})

    JA_ALIAS_TO_KANA_TABLE = alias_to_kana
    JA_KANA_TO_ALIAS_TABLE = kana_to_alias
    JA_ROMAJI_IPA_TABLE = romaji_to_ipa

    # 테이블이 있으면 kana->romaji를 우선 갱신해 리클리스트와 동일한 표기를 따른다.
    for kana, alias_value in kana_to_alias.items():
        alias = _pick_preferred_romaji(alias_value)
        if not alias:
            continue
        k = str(kana or "")
        if not k:
            continue
        if len(k) == 2:
            KANA_COMBO_ROMAJI[k] = alias
        elif len(k) == 1:
            KANA_SINGLE_ROMAJI[k] = alias

    # MFA japanese_mfa에서는 breath/end-breath를 sil 대신 spn으로 두는 편이 안정적이다.
    # (sil 매핑 시 그래프 컴파일 단계에서 비정상 종료가 관측됨)
    # 특수 토큰(br/R)이 테이블에 있는 경우 IPA로 안전하게 고정한다.
    special_tokens = dict(table.get("special_tokens", {}) or {})
    breath = dict(special_tokens.get("breath_standalone", {}) or {})
    breath_alias = str(breath.get("alias", "") or "").strip()
    if breath_alias:
        current = str(JA_ROMAJI_IPA_TABLE.get(breath_alias, "") or "").strip().lower()
        if current in {"", "sil"}:
            JA_ROMAJI_IPA_TABLE[breath_alias] = "spn"
    end_breath = dict(special_tokens.get("end_breath", {}) or {})
    for token in end_breath.get("inputs", []) or []:
        t = str(token or "").strip()
        if t:
            current = str(JA_ROMAJI_IPA_TABLE.get(t, "") or "").strip().lower()
            if current in {"", "sil"}:
                JA_ROMAJI_IPA_TABLE[t] = "spn"


_apply_external_japanese_conversion_table()

# 로마자 음절 분할 시 우선 매칭할 패턴(긴 패턴 우선)
ROMAJI_SYLLABLE_PATTERNS = _build_romaji_syllable_patterns()


def _katakana_to_hiragana(text):
    out = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return ''.join(out)


def _jp_char_score(text):
    s = str(text or "")
    kana = len(_JA_KANA_RE.findall(s))
    hangul = len(_HANGUL_RE.findall(s))
    ascii_word = len(re.findall(r"[0-9A-Za-z]", s))
    return (kana * 12) + ascii_word - (hangul * 8)


def repair_japanese_mojibake_text(text):
    """
    Shift-JIS(cp932) 기반 일본어 파일명이 cp949/UHC로 잘못 풀린 경우를 복구한다.
    예: "あい" -> cp932 bytes -> cp949 decode -> "궇궋"
    """
    src = unicodedata.normalize("NFKC", str(text or ""))
    if not src or not _HANGUL_RE.search(src):
        return src

    best = src
    best_score = _jp_char_score(src)
    for source_enc in ("cp949", "uhc"):
        for target_dec in ("cp932", "shift_jis"):
            try:
                cand = src.encode(source_enc).decode(target_dec)
            except Exception:
                continue
            cand = unicodedata.normalize("NFKC", cand)
            if not _JA_KANA_RE.search(cand):
                continue
            cand_score = _jp_char_score(cand)
            if cand_score > best_score:
                best = cand
                best_score = cand_score
    return best


def _apply_sokuon(romaji):
    if not romaji:
        return romaji
    if romaji.startswith('ch'):
        return 't' + romaji
    if romaji.startswith('sh') or romaji.startswith('j'):
        return romaji[0] + romaji
    if romaji[0] in 'aeioun':
        return romaji
    return romaji[0] + romaji


def split_romaji_token_to_syllables(token):
    """
    붙여쓴 로마자 토큰을 일본어 음절 단위로 분해합니다.
    예: gakkou -> ['ga', 'kko', 'u']
    """
    raw = str(token or '').strip()
    if not raw:
        return []
    s = raw.lower()
    if s in ROMAJI_META_TOKENS:
        return []

    # End-breath compact forms: aR, iR, uR, eR, oR, NR...
    m_end_breath = re.fullmatch(r"([A-Za-z]+)R", raw)
    if m_end_breath:
        head = m_end_breath.group(1)
        head_syllables = split_romaji_token_to_syllables(head)
        if head_syllables:
            return [*head_syllables, "R"]
        return [head.lower(), "R"]

    if raw in JA_ROMAJI_IPA_TABLE:
        return [raw]
    if s in JA_ROMAJI_IPA_TABLE:
        return [s]

    result = []
    i = 0
    geminate = False

    while i < len(s):
        ch = s[i]
        if ch in "_- '":
            i += 1
            continue

        # 촉음(자음 겹침): 다음 음절 초성을 겹쳐서 표기
        if i + 1 < len(s) and s[i] == s[i + 1] and s[i] not in 'aeioun':
            geminate = True
            i += 1
            continue

        matched = None
        for pat in ROMAJI_SYLLABLE_PATTERNS:
            if s.startswith(pat, i):
                matched = pat
                break

        if matched:
            syl = matched
            if geminate:
                syl = _apply_sokuon(syl)
                geminate = False
            result.append(syl)
            i += len(matched)
            continue

        # 단독 n은 비음으로 처리
        if ch == 'n':
            result.append('n')
            i += 1
            continue

        # 불명 문자는 스킵(파일명 노이즈)
        i += 1

    return result


def kana_to_romaji_syllables(text):
    """
    가나 문자열을 로마자 음절 시퀀스로 변환합니다.
    """
    hira = _katakana_to_hiragana(text or '')
    result = []
    geminate = False
    i = 0

    while i < len(hira):
        ch = hira[i]

        if ch in ['っ', 'ッ']:
            geminate = True
            i += 1
            continue

        if ch == 'ー':
            if result and result[-1] and result[-1][-1] in JA_VOWELS_ROMAJI:
                result.append(result[-1][-1])
            i += 1
            continue

        syl = None
        if i + 1 < len(hira):
            two = hira[i:i+2]
            if two in KANA_COMBO_ROMAJI:
                syl = KANA_COMBO_ROMAJI[two]
                i += 2

        if syl is None:
            syl = KANA_SINGLE_ROMAJI.get(ch)
            i += 1

        if not syl:
            continue

        if geminate:
            syl = _apply_sokuon(syl)
            geminate = False

        result.append(syl)

    return result


def split_ja_romaji_syllable(syllable):
    """
    로마자 음절을 (초성, 모음)으로 분해합니다.
    실패 시 (원문, '')를 반환합니다.
    """
    s = (syllable or '').strip().lower()
    if not s:
        return '', ''

    if s in ['n', 'nn', 'xn', 'm']:
        return '', 'n'
    if s in ['q', 'cl', 'xtu', 'xtsu', 'ltu', 'ltsu']:
        return '', 'cl'

    for v in ['a', 'i', 'u', 'e', 'o']:
        if s.endswith(v):
            return s[:-1], v

    return s, ''


def parse_ja_filename(filename):
    """
    일본어 UTAU WAV 파일명을 로마자 음절 리스트로 분리합니다.
    
    구분자: _, ', -, 스페이스
    예: 'ka_ki_ku_ke_ko.wav' → ['ka', 'ki', 'ku', 'ke', 'ko']
    예: "a'ka'sa'ta'na.wav" → ['a', 'ka', 'sa', 'ta', 'na']
    """
    repaired = repair_japanese_mojibake_text(os.path.basename(filename or ""))
    base = os.path.splitext(repaired)[0]
    # 음역/피치 태그 제거 (예: C4_, _D#3)
    base = re.sub(r'(^|[_\-\s])([A-Ga-g][#b]?\d+)($|[_\-\s])', ' ', base)
    tokens = [t for t in re.split(r"[_'\-\s]+", base) if t.strip()]

    syllables = []
    for token in tokens:
        t = token.strip()
        if not t:
            continue

        # Breath tokens: 息 (standalone) / <token>息 (end-breath).
        if t == "息":
            syllables.append("br")
            continue
        m_end_breath_romaji = re.fullmatch(r"([A-Za-z]+)息", t)
        if m_end_breath_romaji:
            syllables.extend(split_romaji_token_to_syllables(m_end_breath_romaji.group(1)))
            syllables.append("R")
            continue
        m_end_breath_kana = re.fullmatch(r"([\u3041-\u3096\u30A1-\u30FA\u30FC]+)息", t)
        if m_end_breath_kana:
            syllables.extend(kana_to_romaji_syllables(m_end_breath_kana.group(1)))
            syllables.append("R")
            continue

        # 순수 로마자 토큰
        if re.fullmatch(r'[A-Za-z]+', t):
            syllables.extend(split_romaji_token_to_syllables(t))
            continue

        # 가나 포함 토큰
        if re.search(r'[\u3041-\u3096\u30A1-\u30FA\u30FC]', t):
            syllables.extend(kana_to_romaji_syllables(t))
            continue

        # 혼합 토큰: 가나/로마자를 분리하여 각각 처리
        for part in re.findall(r'[\u3041-\u3096\u30A1-\u30FA\u30FC]+|[A-Za-z]+', t):
            if re.fullmatch(r'[A-Za-z]+', part):
                syllables.extend(split_romaji_token_to_syllables(part))
            else:
                syllables.extend(kana_to_romaji_syllables(part))

    return [s for s in syllables if s]


def generate_ja_labs(wav_dir, custom_phonemes_path='', callback=None):
    """
    일본어 WAV 파일이 있는 폴더를 기반으로 Lab 파일을 생성합니다.
    
    Args:
        wav_dir: WAV 파일이 있는 폴더 경로
        custom_phonemes_path: 특수 발음 매핑 파일 경로
        callback: 진행 상황 콜백 함수
    
    Returns:
        (성공 개수, 전체 개수, 에러 목록)
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    errors = []
    
    if not os.path.exists(wav_dir):
        msg = f"❌ 폴더를 찾을 수 없습니다: {wav_dir}"
        log(msg)
        return 0, 0, [msg]

    wav_files = [f for f in os.listdir(wav_dir) if f.lower().endswith('.wav')]
    if not wav_files:
        msg = "❌ WAV 파일이 없습니다."
        log(msg)
        return 0, 0, [msg]

    total = len(wav_files)
    success = 0
    
    custom_map = load_custom_phonemes(custom_phonemes_path)
    if custom_map:
        log(f"🔧 커스텀 음소 매핑 사용됨: {len(custom_map)}개 항목")

    for wf in wav_files:
        try:
            syllables = parse_ja_filename(wf)
            syllables = apply_custom_phonemes(syllables, custom_map)
            
            if not syllables:
                errors.append(f"파싱 실패: {wf}")
                continue

            # Lab 내용 = 파일명에서 추출한 텍스트 (스페이스 구분)
            lab_text = " ".join(syllables)
            
            lab_path = os.path.join(wav_dir, os.path.splitext(wf)[0] + '.lab')
            with open(lab_path, 'w', encoding='utf-8') as f:
                f.write(lab_text)
            
            success += 1
        except Exception as e:
            errors.append(f"Lab 생성 실패 ({wf}): {e}")

    log(f"✅ 일본어 Lab 파일 생성 완료: {success}/{total}")
    return success, total, errors


def generate_ja_dictionary(target_folder, dict_save_path, custom_phonemes_path='', callback=None):
    """
    Lab 파일로부터 MFA용 일본어 사전 파일을 자동 생성합니다.
    
    Args:
        target_folder: Lab 파일이 있는 폴더
        dict_save_path: 사전 파일 저장 경로
        callback: 진행 상황 콜백 함수
    
    Returns:
        (성공 파일 수, 등록 항목 수, 에러 목록)
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    errors = []

    if not os.path.exists(target_folder):
        msg = f"❌ 폴더를 찾을 수 없습니다: {target_folder}"
        log(msg)
        return 0, 0, [msg]

    lab_files = [f for f in os.listdir(target_folder) if f.lower().endswith('.lab')]
    wav_files = [f for f in os.listdir(target_folder) if f.lower().endswith('.wav')]
    if not lab_files and not wav_files:
        msg = "❌ Lab/WAV 파일이 없습니다."
        log(msg)
        return 0, 0, [msg]
        
    custom_map = load_custom_phonemes(custom_phonemes_path)

    # 모든 Lab 파일에서 고유 단어를 수집
    word_set = set()
    success = 0
    
    for lf in lab_files:
        try:
            lab_path = os.path.join(target_folder, lf)
            content, enc, read_err = read_text_auto(lab_path)
            if read_err:
                errors.append(f"Lab 읽기 실패 ({lf}): {read_err}")
                continue
            if enc not in ('utf-8', 'utf-8-sig'):
                log(f"⚠ Lab 인코딩 감지: {lf} -> {enc} (자동 변환 읽기)")
            content = (content or '').strip()
            
            words = content.split()
            for w in words:
                wr = str(w or "").strip()
                if not wr:
                    continue
                wl = wr.lower()

                # 특수 토큰은 대소문자 의미를 보존한다. (예: R=end-breath)
                if wr in JA_ROMAJI_IPA_TABLE:
                    word_set.add(wr)
                    continue

                if re.search(r'[\u3041-\u3096\u30A1-\u30FA\u30FC]', wr):
                    word_set.update(kana_to_romaji_syllables(wr))
                elif re.fullmatch(r'[A-Za-z]+', wr):
                    split_words = split_romaji_token_to_syllables(wr)
                    if split_words:
                        word_set.update(split_words)
                    else:
                        word_set.add(wl)
                else:
                    # 혼합 토큰도 음절 단위로 최대한 복구
                    word_set.update(parse_ja_filename(wr))
            success += 1
        except Exception as e:
            errors.append(f"Lab 읽기 실패 ({lf}): {e}")

    if not word_set and wav_files:
        log("⚠ Lab에서 유효 음절을 찾지 못해 WAV 파일명 기반으로 사전을 보완합니다.")
        for wf in wav_files:
            word_set.update(parse_ja_filename(wf))

    # 사전 생성: 각 단어(로마자 음절)에 대해 IPA 변환
    dict_entries = {}
    for word in sorted(word_set):
        ipa = romaji_to_ipa(word)
        if ipa:
            dict_entries[word] = ipa
            
    # 커스텀 맵 강제 등록
    for raw_char, mapped_pho in custom_map.items():
        key_variants = [raw_char.lower()]
        if re.search(r'[\u3041-\u3096\u30A1-\u30FA\u30FC]', raw_char):
            key_variants.extend(kana_to_romaji_syllables(raw_char))

        for key in key_variants:
            if key in dict_entries:
                continue
            c_ipa = romaji_to_ipa(mapped_pho)
            if not c_ipa or c_ipa == 'sil':
                c_ipa = mapped_pho  # 변환 없으면 원본값
            dict_entries[key] = c_ipa

    # 사전 파일 저장
    try:
        with open(dict_save_path, 'w', encoding='utf-8') as f:
            for word, ipa in sorted(dict_entries.items()):
                f.write(f"{word}\t{ipa}\n")
        log(f"✅ 일본어 사전 생성 완료: {len(dict_entries)}개 항목 ({success}개 Lab 파일)")
    except Exception as e:
        err = f"❌ 사전 저장 실패: {e}"
        logger.error(err)
        errors.append(err)

    return success, len(dict_entries), errors


def decompose_romaji_to_phones(syllable):
    """
    로마자 음절을 IPA 음소 개별 리스트로 분해합니다.
    OTO 생성 시 TextGrid 음소와 대조하는 데 사용됩니다.
    
    Args:
        syllable: 로마자 음절 (예: 'ka')
    
    Returns:
        IPA 음소 리스트 (예: ['k', 'a'])
    """
    ipa = romaji_to_ipa(syllable)
    if not ipa:
        return []
    return ipa.split()
