"""
Lab/사전(Dictionary) 파일 자동 생성 모듈
- 로마자 -> 한글 변환
- 로마자 -> IPA 변환
- Lab 파일 생성
- MFA용 사전 파일 생성
"""

import os
import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# 로마자 <-> 한글 자모 매핑 테이블
# ==============================================================================

CHOSUNG_LIST_KR = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
JUNGSUNG_LIST_KR = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
JONGSUNG_LIST_KR = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

INITIAL_MAP = {
    'g': 0, 'k': 0, 'kk': 1, 'gg': 1, 'n': 2, 'd': 3, 't': 3, 'tt': 4, 'dd': 4,
    'r': 5, 'l': 5, 'm': 6, 'b': 7, 'p': 7, 'pp': 8, 'bb': 8, 's': 9, 'ss': 10,
    '': 11, 'ng': 11,
    'j': 12, 'z': 12, 'jj': 13, 'zz': 13, 'ch': 14, 'c': 14, 'q': 15, 'kh': 15,
    'tx': 16, 'th': 16, 'ph': 17, 'f': 17, 'h': 18,
    'sh': 9  # 'sh' is mapped to 'ㅅ'
}

VOWEL_MAP = {
    'a': 0, 'ae': 1, 'ya': 2, 'yae': 3, 'eo': 4, 'e': 5, 'yeo': 6, 'ye': 7,
    'o': 8, 'wa': 9, 'wae': 10, 'oe': 11, 'yo': 12, 'u': 13, 'wo': 14, 'we': 15, 'wi': 16,
    'yu': 17, 'eu': 18, 'ui': 19, 'yi': 19, 'eui': 19, 'i': 20,
    'weo': 14  # 'weo' is often used for 'wo' (ㅝ)
}

FINAL_MAP = {
    '': 0, 'g': 1, 'k': 1, 'kk': 2, 'gs': 3, 'n': 4, 'nj': 5, 'nh': 6, 'd': 7, 't': 7,
    'l': 8, 'r': 8, 'lg': 9, 'lm': 10, 'lb': 11, 'ls': 12, 'lt': 13, 'lp': 14, 'lh': 15,
    'm': 16, 'b': 17, 'p': 17, 'bs': 18, 's': 19, 'ss': 20, 'ng': 21, 'j': 22, 'ch': 23,
}

# 로마자 -> IPA
KO_IPA_MAP = {
    'g': 'k', 'n': 'n', 'd': 't', 'r': 'ɾ', 'l': 'ɭ', 'm': 'm', 'b': 'p', 's': 's',
    'j': 'tɕ', 'h': 'h',
    'k': 'kʰ', 't': 'tʰ', 'p': 'pʰ', 'ch': 'tɕʰ',
    'gg': 'k͈', 'kk': 'k͈', 'dd': 't͈', 'tt': 't͈', 'bb': 'p͈', 'pp': 'p͈', 'ss': 's͈', 'jj': 'tɕ͈',
    'ng': 'ŋ',
    'a': 'ɐ', 'i': 'i', 'u': 'u', 'e': 'e', 'o': 'o', 'eu': 'ɨ', 'eo': 'ʌ', 'ae': 'ɛ',
    'y': 'j', 'w': 'w',
    'R': 'sil', 'H': 'sil', 'br': 'sil', 'pau': 'sil', 'sil': 'sil', 'bre': 'sil',
    'ui': 'ɰ i', 'wa': 'w ɐ', 'wo': 'w o', 'wi': 'w i', 'we': 'w e',
    'yeo': 'j ʌ', 'wae': 'w ɛ', 'oe': 'w e', 'vi': 'v i'
}

# 한글 자모 -> 로마자
CHOSUNG_LIST_ROMAN = ['g','kk','n','d','tt','r','m','b','pp','s','ss','','j','jj','ch','k','t','p','h']
JUNGSUNG_LIST_ROMAN = ['a','ae','ya','yae','eo','e','yeo','ye','o','wa','wae','oe','yo','u','wo','we','wi','yu','eu','ui','i']
JONGSUNG_LIST_ROMAN = ['','k','kk','gs','n','nj','nh','d','l','lg','lm','lb','ls','lt','lp','lh','m','b','bs','s','ss','ng','j','ch','k','t','p','h']


def compose_hangul(initial, vowel, final=''):
    """초/중/종성 로마자로 한글 유니코드 문자 생성"""
    try:
        cho = INITIAL_MAP.get(initial.lower(), -1)
        jung = VOWEL_MAP.get(vowel.lower(), -1)
        jong = FINAL_MAP.get(final.lower(), 0)
        if cho == -1 or jung == -1:
            return initial + vowel + final
        code = 0xAC00 + (cho * 21 * 28) + (jung * 28) + jong
        return chr(code)
    except Exception:
        return initial + vowel + final


def parse_romaji_syllable(text):
    """로마자 음절 하나를 초/중/종성으로 분리하여 한글 변환"""
    # 0. 'Long' 접미사 제거
    clean_text = re.sub(r'long$', '', text, flags=re.IGNORECASE)
    
    # 1. 중성(Vowel) 찾기 (긴 순서대로 매칭하여 욕심쟁이 매칭 방지)
    vowels = sorted(VOWEL_MAP.keys(), key=len, reverse=True)
    vowel = ""
    v_idx = -1
    for v in vowels:
        idx = clean_text.find(v)
        if idx != -1:
            vowel = v
            v_idx = idx
            break
            
    if v_idx == -1:
        return text
        
    initial = clean_text[:v_idx]
    final = clean_text[v_idx + len(vowel):]
    return compose_hangul(initial, vowel, final)


def decompose_hangul_to_roman(char):
    """한글 글자를 로마자 토큰으로 분해 (예: '가' -> ['g', 'a'])"""
    if not (0xAC00 <= ord(char) <= 0xD7A3):
        return [char]
    code = ord(char) - 0xAC00
    jong = code % 28
    jung = (code // 28) % 21
    cho = code // 588
    result = []
    if CHOSUNG_LIST_ROMAN[cho]:
        result.append(CHOSUNG_LIST_ROMAN[cho])
    result.append(JUNGSUNG_LIST_ROMAN[jung])
    if JONGSUNG_LIST_ROMAN[jong]:
        result.append(JONGSUNG_LIST_ROMAN[jong])
    return result


def get_ipa_from_roman(token):
    """로마자 토큰을 IPA 음소 리스트로 변환"""
    token = token.lower()
    token = re.sub(r'[0-9]+', '', token)
    token = token.replace('long', '')
    if not token or token in ['br', 'pau', 'sil', 'r', 'h', 'bre']:
        return ['sil']
    phonemes = []
    remainder = token
    limit = 50
    count = 0
    while remainder and count < limit:
        count += 1
        matched = False
        for length in [3, 2, 1]:
            if len(remainder) < length:
                continue
            chunk = remainder[:length]
            if chunk in KO_IPA_MAP:
                val = KO_IPA_MAP[chunk]
                is_coda = False
                if len(phonemes) > 0 and chunk in ['k','p','t','l','m','n','ng']:
                    next_is_vowel = False
                    if len(remainder) > length:
                        next_char = remainder[length]
                        if next_char in ['a','e','i','o','u','y','w']:
                            next_is_vowel = True
                    if not next_is_vowel:
                        is_coda = True
                if is_coda:
                    base = chunk
                    if base in ['k','g','gg','kk']: coda = 'k̚'
                    elif base in ['p','b','bb','pp']: coda = 'p̚'
                    elif base in ['t','d','s','ss','j','ch','tt','jj']: coda = 't̚'
                    elif base == 'l': coda = 'ɭ'
                    elif base == 'm': coda = 'm'
                    elif base == 'n': coda = 'n'
                    elif base == 'ng': coda = 'ŋ'
                    else: coda = KO_IPA_MAP.get(base, base)
                    phonemes.append(coda)
                else:
                    if ' ' in val:
                        phonemes.extend(val.split())
                    else:
                        phonemes.append(val)
                remainder = remainder[length:]
                matched = True
                break
            if chunk == 'yeo':
                phonemes.extend(['j', 'ʌ']); remainder = remainder[length:]; matched = True; break
            elif chunk == 'wae':
                phonemes.extend(['w', 'ɛ']); remainder = remainder[length:]; matched = True; break
        if matched:
            continue
        remainder = remainder[1:]
    return phonemes


# ==============================================================================
# 커스텀 폰 파싱 유틸리티
# ==============================================================================

def load_custom_phonemes(file_path):
    """
    커스텀 음소 매핑 파일을 읽어서 딕셔너리로 반환합니다.
    형식: 원어=변환어 (예: 엣지=ed)
    """
    custom_map = {}
    if not file_path or not os.path.exists(file_path):
        return custom_map
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    custom_map[k.strip()] = v.strip()
    except Exception as e:
        logger.error(f"커스텀 음소 파일 로드 실패: {file_path}, 에러: {e}")
        
    return custom_map

def apply_custom_phonemes(text_list, custom_map):
    """
    텍스트 리스트에 커스텀 음소 매핑을 적용합니다.
    """
    if not custom_map:
        return text_list
        
    result = []
    for token in text_list:
        # 단어 전체가 매칭되는 경우 우선 치환
        if token in custom_map:
            result.append(custom_map[token])
        else:
            # 부분 문자열 매칭 시도 (가장 긴 것부터)
            # 한국어 "아엣지" 같은 붙어있는 문자열 해결을 위함
            mod_token = token
            for k in sorted(custom_map.keys(), key=len, reverse=True):
                if k in mod_token:
                    # 임시 구분자 삽입하여 분리되게 함
                    mod_token = mod_token.replace(k, f" {custom_map[k]} ")
            
            if mod_token != token:
                result.extend([p for p in mod_token.split() if p])
            else:
                result.append(token)
                
    return result


# ==============================================================================
# Lab 파일 생성
# ==============================================================================

def generate_labs(wav_dir, reclist_file='', convert_to_hangul=True, custom_phonemes_path='', callback=None):
    """
    WAV 파일이 있는 폴더를 기반으로 Lab 파일을 생성합니다.
    
    Args:
        wav_dir: WAV 파일이 있는 폴더 경로
        reclist_file: 녹음 리스트 파일 경로 (비어있으면 WAV 파일명 사용)
        convert_to_hangul: 로마자를 한글로 변환할지 여부
        callback: 진행 상황 콜백 함수 (message: str) -> None
    
    Returns:
        (성공 개수, 전체 개수, 에러 목록)
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    errors = []
    targets = []

    if reclist_file and os.path.exists(reclist_file):
        log(f"📄 리스트 파일 읽는 중: {reclist_file}")
        with open(reclist_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    targets.append(line)
    elif os.path.exists(wav_dir):
        log(f"📂 WAV 폴더 스캔 중: {wav_dir}")
        files = [f for f in os.listdir(wav_dir) if f.lower().endswith('.wav')]
        targets = files
    else:
        msg = f"❌ 오류: WAV 폴더를 찾을 수 없습니다: {wav_dir}"
        log(msg)
        return 0, 0, [msg]

    count = 0
    total = len(targets)
    custom_map = load_custom_phonemes(custom_phonemes_path)
    if custom_map:
        log(f"🔧 커스텀 음소 매핑 사용됨: {len(custom_map)}개 항목")

    for i, filename in enumerate(targets):
        filename = os.path.basename(filename)
        try:
            phonemes = _parse_filename(filename, convert_to_hangul)
            phonemes = apply_custom_phonemes(phonemes, custom_map)
            
            if not phonemes:
                continue
            lab_content = " ".join(phonemes)
            base_name = os.path.splitext(filename)[0]
            lab_filename = base_name + ".lab"
            save_path = os.path.join(wav_dir, lab_filename)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(lab_content)
            count += 1
        except Exception as e:
            err = f"Lab 생성 실패 ({filename}): {e}"
            logger.error(err)
            errors.append(err)

        if callback and ((i + 1) % 10 == 0 or (i + 1) == total):
            callback(f"진행 중... ({i + 1}/{total})")

    log(f"✅ 완료! 총 {count}/{total}개의 Lab 파일 생성됨")
    return count, total, errors


def _parse_filename(filename, convert_to_hangul):
    """파일명에서 음소 리스트 추출"""
    base = os.path.splitext(filename)[0]
    base = re.sub(r'\d+$', '', base)

    if re.search(r'[가-힣]', base):
        if '_' in base or ' ' in base:
            parts = re.split(r'[_\s]+', base)
            return [p for p in parts if p]
        else:
            return list(base)
    else:
        parts = re.split(r'[_\s]+', base)
        parts = [p for p in parts if p]
        if convert_to_hangul:
            return [parse_romaji_syllable(t) for t in parts]
        return parts


# ==============================================================================
# 사전(Dictionary) 생성
# ==============================================================================

def generate_dictionary(target_folder, dict_save_path, custom_phonemes_path='', callback=None):
    """
    Lab 파일로부터 MFA용 사전 파일을 자동 생성합니다.
    
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
    count_files = 0
    dictionary_entries = {}
    
    custom_map = load_custom_phonemes(custom_phonemes_path)

    log(f"📄 총 {len(lab_files)}개의 Lab 파일 처리 시작")

    for idx, lab_file in enumerate(lab_files):
        full_path = os.path.join(target_folder, lab_file)
        try:
            clean_filename_str = unicodedata.normalize('NFC', lab_file)
            base_name = os.path.splitext(clean_filename_str)[0]

            raw_tokens = re.split(r"[_'\s]+", base_name)
            expanded_tokens = []
            for t in raw_tokens:
                if not t or t == '~':
                    continue
                if re.match(r'^[가-힣]+$', t):
                    expanded_tokens.extend(list(t))
                else:
                    expanded_tokens.append(t)
            final_tokens = expanded_tokens

            with open(full_path, 'r', encoding='utf-8') as f:
                content_raw = f.read().strip()
                content_normalized = unicodedata.normalize('NFC', content_raw)

            chars_only = list(re.sub(r"[\s_]+", "", content_normalized))
            new_content = " ".join(chars_only)
            if content_raw != new_content:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

            if len(final_tokens) != len(chars_only):
                log(f"⚠️ 개수 불일치 ({lab_file}): 파일명 {len(final_tokens)} vs 내용 {len(chars_only)}")

            full_sentence_ipa = []
            for token, char in zip(final_tokens, chars_only):
                ipa_list = []
                if re.search(r'[가-힣]', token):
                    roman_parts = []
                    for ch in token:
                        roman_parts.extend(decompose_hangul_to_roman(ch))
                    for part in roman_parts:
                        ipa_list.extend(get_ipa_from_roman(part))
                else:
                    ipa_list = get_ipa_from_roman(token)

                ipa_str = " ".join(ipa_list)
                if ipa_str and ipa_str != 'sil':
                    dictionary_entries[char] = ipa_str
                    full_sentence_ipa.append(ipa_str)

            # 커스텀 맵에 있는 기호들도 렉시콘에 등록 (MFA가 인식할 수 있도록 강제)
            for raw_char, mapped_pho in custom_map.items():
                if raw_char not in dictionary_entries:
                    # 매핑된 음소(mapped_pho)가 한국어 로마자라면 ipa로 변환, 아니면 그대로 사용
                    c_ipa = " ".join(get_ipa_from_roman(mapped_pho))
                    # 변환 결과가 없거나 sil이면 매핑어 원본을 그대로 음소로 (영어 등 대응)
                    if not c_ipa or c_ipa == 'sil':
                        c_ipa = mapped_pho
                    dictionary_entries[raw_char] = c_ipa

            if full_sentence_ipa:
                full_key = new_content
                full_value = " ".join(full_sentence_ipa)
                dictionary_entries[full_key] = full_value
                count_files += 1

        except Exception as e:
            err = f"사전 생성 에러 ({lab_file}): {e}"
            logger.error(err)
            errors.append(err)

        if callback and ((idx + 1) % 10 == 0 or (idx + 1) == len(lab_files)):
            callback(f"사전 생성 중... ({idx + 1}/{len(lab_files)})")

    try:
        with open(dict_save_path, 'w', encoding='utf-8') as f:
            for key, ipa in sorted(dictionary_entries.items()):
                f.write(f"{key}\t{ipa}\n")
        log(f"📘 사전 파일 생성 완료: {dict_save_path} ({len(dictionary_entries)}개 항목)")
    except Exception as e:
        err = f"❌ 사전 저장 실패: {e}"
        logger.error(err)
        errors.append(err)

    return count_files, len(dictionary_entries), errors


def verify_dict_lab_match(dict_path, lab_folder, callback=None):
    """사전과 Lab 파일 간의 매칭 검사"""
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    dict_keys = set()
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                dict_keys.add(parts[0])

    lab_files = [f for f in os.listdir(lab_folder) if f.endswith('.lab')]
    mismatches = []

    for lab_file in lab_files:
        lab_path = os.path.join(lab_folder, lab_file)
        with open(lab_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        words = content.split()
        for w in words:
            if w not in dict_keys:
                mismatches.append((lab_file, w))

    if not mismatches:
        log("✅ 완벽합니다! 모든 Lab 파일의 내용이 사전에 등록되어 있습니다.")
    else:
        log(f"❌ 총 {len(mismatches)}개의 불일치가 발견되었습니다.")

    return mismatches
