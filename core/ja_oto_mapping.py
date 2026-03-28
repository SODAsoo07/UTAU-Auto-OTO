"""
Japanese OTO mapping / format detection helpers.

This module isolates alias classification, syllable token normalization,
and filename/phone-nuclei based syllable mapping from the main generator.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from core.alias_annotation_utils import strip_alias_annotation_suffixes
from core.ja_lab_generator import parse_ja_filename, romaji_to_ipa, split_ja_romaji_syllable

JA_CONSONANTS = [
    'k', 'g', 's', 'z', 't', 'd', 'n', 'h', 'b', 'p', 'm', 'r', 'l',
    'w', 'y', 'f', 'v', 'j', 'q', 'c',
    'sh', 'ch', 'ts', 'dz', 'hy', 'ky', 'gy', 'ny', 'my', 'ry', 'by', 'py',
    'dy', 'ty', 'ss', 'kk', 'tt', 'pp', 'dd', 'gg', 'bb', 'zz', 'jj',
]
JA_VOWELS = ['a', 'i', 'u', 'e', 'o']
JA_KANA_VOWELS = {'?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?'}
JA_NASAL_ONSETS = {'m', 'n', 'ny', 'ng', 'ngy'}
JA_VOICED_ONSETS = {
    'g', 'z', 'd', 'b', 'j', 'dz', 'v',
    'm', 'n', 'ny', 'r', 'l', 'ry', 'w', 'y',
}
JA_VOICELESS_ONSETS = {
    'k', 's', 't', 'p', 'h', 'f', 'sh', 'ch', 'ts',
    'q', 'c', 'ky', 'ty', 'py', 'hy', 'ss', 'kk', 'tt', 'pp',
}
_JA_KANA_RE = re.compile(r"[\u3041-\u3096\u30A1-\u30FA\u30FC]")


def _normalize_custom_alias_lookup(alias):
    text = unicodedata.normalize("NFKC", str(alias or "")).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return []
    variants = [text, text.lower()]
    stripped = re.sub(r"(?:[_\-\s]+)(?:[a-g](?:#|b)?[0-8])$", "", text, flags=re.IGNORECASE)
    stripped = re.sub(r"(?:[_\-\s]+)(?:[a-g](?:sharp|flat)?[0-8])$", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*[\(\[\{（【].*?[\)\]\}）】]\s*$", "", stripped).strip()
    if stripped:
        variants.extend([stripped, stripped.lower()])
    out = []
    seen = set()
    for item in variants:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _lookup_custom_alias_value(custom_map, alias):
    if not custom_map:
        return None
    for candidate in _normalize_custom_alias_lookup(alias):
        if candidate in custom_map:
            return custom_map[candidate]
    return None


def _clean_phone_mark(mark):
    return re.sub(r"[0-9]", "", (mark or "").strip().lower())


def is_breath(alias):
    clean = unicodedata.normalize("NFKC", str(alias or "")).strip().lower()
    if not clean:
        return False
    if re.match(r'^br\d*$', clean):
        return True
    if clean in {"bre", "breath", "息", "吸", "吐", "吸い", "吐き", "息継ぎ"}:
        return True
    # Some banks annotate inhale/exhale variants like 吸こ吐, 吸き吐.
    if any(ch in clean for ch in ("吸", "吐", "息")):
        return True
    return False


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


def _classify_ja_alias_core(clean, custom_map=None):
    if is_breath(clean):
        return 'br'
    mapped_val = _lookup_custom_alias_value(custom_map, clean)
    if mapped_val is not None:
        mapped_val = str(mapped_val).lower()
        if mapped_val in ['r', 'h', 'sil', 'br', 'pau', 'sp']:
            return 'br'
        ipa = romaji_to_ipa(mapped_val)
        if ipa and (ipa in ['a', 'i', '?', 'e', 'o'] or _is_ja_vowel_token(mapped_val)):
            return 'mono'
        return 'cv'
    if ' ' in clean:
        parts = clean.split()
        left = parts[0]
        right = ' '.join(parts[1:])
        if left == '-':
            return 'cv_head'
        left_is_vowel = _is_ja_vowel_token(left)
        if left_is_vowel:
            if right.strip() == '-':
                return 'vv'
            if re.fullmatch(r"^[^0-9A-Za-z぀-ヿㇰ-ㇿ]+$", right.strip()):
                return 'vc'
            if right.lower() in JA_CONSONANTS:
                return 'vc'
            right_norm = _normalize_ja_syllable_token(right)
            if _is_ja_vowel_token(right) or _is_ja_vowel_token(right_norm):
                return 'vv'
            _, right_vowel = split_ja_romaji_syllable(right_norm.lower())
            if right_vowel not in JA_VOWELS:
                return 'vc'
            return 'vcv'
        return 'vc'
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


def classify_ja_alias(alias, custom_map=None):
    clean = unicodedata.normalize("NFKC", str(alias or "")).strip()
    clean = strip_alias_annotation_suffixes(
        clean,
        classifier=lambda candidate: _classify_ja_alias_core(candidate, custom_map),
    )
    return _classify_ja_alias_core(clean, custom_map)


def detect_ja_alias_format(alias_list, custom_map=None):
    if not alias_list:
        return 'cv'
    type_cache = {}
    types = []
    for a in alias_list:
        t = type_cache.get(a)
        if t is None:
            t = classify_ja_alias(a, custom_map)
            type_cache[a] = t
        types.append(t)
    type_set = set(types)
    type_counts = {}
    for t in types:
        type_counts[t] = int(type_counts.get(t, 0)) + 1
    non_br_total = int(sum(v for k, v in type_counts.items() if k != "br"))
    vcv_count = int(type_counts.get("vcv", 0))
    vc_count = int(type_counts.get("vc", 0))
    vv_count = int(type_counts.get("vv", 0))
    cv_like_count = int(
        type_counts.get("cv", 0)
        + type_counts.get("cv_head", 0)
        + type_counts.get("mono", 0)
    )
    if type_set == {'br'}:
        return 'br'
    if type_set <= {'mono', 'cv_head', 'cv'}:
        return 'cv'
    non_br = type_set - {'br'}
    if non_br <= {'vc', 'vv'}:
        return 'vc_only'
    if vc_count > 0 or vv_count > 0:
        # CVVC 계열은 VC/VV가 존재하는 순간 우선한다.
        return 'cvvc'
    if vcv_count > 0:
        # 과거에는 vcv가 1개만 있어도 전체를 vcv로 잡아 CVVC가 흔들렸다.
        # vcv가 충분히 지배적인 경우에만 vcv로 분류한다.
        if non_br_total <= 0:
            return 'vcv'
        vcv_ratio = float(vcv_count) / float(non_br_total)
        if vcv_count >= 2 and (vcv_ratio >= 0.55 or vcv_count > cv_like_count):
            return 'vcv'
        if cv_like_count > 0:
            return 'cv'
        return 'vcv'
    return 'cv'


def get_vc_consonant(alias):
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


@lru_cache(maxsize=65536)
def _normalize_ja_syllable_token(token):
    t_raw = (token or '').strip()
    if not t_raw:
        return ''
    t = t_raw.lower()
    if re.search(r'[ぁ-ゖァ-ヺー]', t_raw):
        syls = parse_ja_filename(t_raw)
        if syls:
            t = syls[0].lower()
    t = t.replace("'", '').strip('-_ ')
    if not t:
        return ''
    if t in {'n', 'nn', 'xn', 'm'}:
        return 'n'
    onset, vowel = split_ja_romaji_syllable(t)
    if vowel in JA_VOWELS:
        return f"{onset}{vowel}" if onset else vowel
    return t


@lru_cache(maxsize=65536)
def _normalize_ja_syllable_token_strict(token):
    raw = str(token or "").strip()
    if not raw:
        return ""
    return raw.lower().replace("'", "").strip("-_ ")


def _is_kana_token(token):
    return bool(_JA_KANA_RE.search(str(token or "")))


@lru_cache(maxsize=65536)
def _extract_vcv_target_syllable(alias):
    parts = (alias or '').strip().split()
    if len(parts) >= 2:
        return _normalize_ja_syllable_token(parts[1])
    return _normalize_ja_syllable_token(alias)


@lru_cache(maxsize=65536)
def _ja_left_bridge_token(alias_text):
    parts = (alias_text or '').strip().split()
    if len(parts) >= 2:
        return _normalize_ja_syllable_token(parts[0])
    return ''


@lru_cache(maxsize=65536)
def _ja_is_n_bridge_alias(alias_text, alias_type='vc'):
    alias_type = str(alias_type or '').strip().lower()
    if alias_type not in {'vc', 'vcv'}:
        return False
    left = _ja_left_bridge_token(alias_text)
    return left in {'n', 'nn', 'xn', 'm'}


def _syllable_info_token(syl_info):
    if not isinstance(syl_info, dict):
        return ''
    for k in ('roman', 'word'):
        if k in syl_info:
            tok = _normalize_ja_syllable_token(syl_info.get(k, ''))
            if tok:
                return tok
    phones = syl_info.get('phones') or []
    marks = [_clean_phone_mark(getattr(p, 'mark', '')) for p in phones]
    marks = [m for m in marks if m and m not in {'sil', 'sp', 'spn', 'pau'}]
    if not marks:
        return ''
    onset = ''
    vowel = ''
    for m in marks:
        if m in {'a', 'i', 'u', '?', 'e', 'o', '?', 'n'}:
            vowel = 'u' if m == '?' else ('n' if m in {'?', 'n'} else m)
            break
        if not onset:
            onset = m
    if vowel:
        return f"{onset}{vowel}" if onset and vowel != 'n' else vowel
    return onset


def _syllable_info_raw_token(syl_info):
    if not isinstance(syl_info, dict):
        return ""
    for k in ("roman", "word", "kana"):
        raw = syl_info.get(k, "")
        if raw:
            return str(raw).strip()
    return ""


@lru_cache(maxsize=131072)
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


def _alias_to_ja_cv_target(alias, alias_type):
    a = (alias or '').strip()
    if not a:
        return ''
    if alias_type in {'cv', 'mono'}:
        return _normalize_ja_syllable_token(a)
    if alias_type == 'cv_head':
        parts = a.split()
        if len(parts) >= 2 and parts[0] == '-':
            return _normalize_ja_syllable_token(parts[1])
        return _normalize_ja_syllable_token(a.lstrip('-'))
    if alias_type == 'vcv':
        parts = a.split()
        if len(parts) >= 2:
            return _normalize_ja_syllable_token(parts[1])
        return _normalize_ja_syllable_token(a)
    if alias_type == 'vv':
        parts = a.split()
        if len(parts) >= 2:
            return _normalize_ja_syllable_token(parts[1])
    return ''


def _extract_ja_cv_targets_from_lines(lines, custom_map=None):
    targets = []
    type_cache = {}
    for line in lines or []:
        if '=' not in line:
            continue
        alias = line.split('=', 1)[1].split(',', 1)[0].strip()
        if not alias:
            continue
        a_type = type_cache.get(alias)
        if a_type is None:
            a_type = classify_ja_alias(alias, custom_map)
            type_cache[alias] = a_type
        tok = _alias_to_ja_cv_target(alias, a_type)
        if tok:
            targets.append(tok)
    # IMPORTANT:
    # VCV/CVVC에서는 연속 중복 음절(예: ma ma mi..., ra ra ri...) 자체가
    # 실제 매핑 순서를 결정하므로, 여기서 중복을 제거하면 1칸 밀림이 발생한다.
    return targets if targets else []


def _is_nucleus_phone(mark):
    m = _clean_phone_mark(mark)
    return m in {'a', 'i', '?', 'u', 'e', 'o', '?', 'n', '?', '?', '?', '?', '?', '?', '?'}


def _build_ja_syllables_from_phone_nuclei(ph_intervals, cv_targets):
    if not ph_intervals or not cv_targets:
        return None
    target_n = len(cv_targets)
    nuclei = [i for i, p in enumerate(ph_intervals) if _is_nucleus_phone(getattr(p, 'mark', ''))]
    if len(nuclei) < target_n:
        return None
    target_vowels = [_target_vowel_from_filename_syllable(s) for s in cv_targets]
    nucleus_vowels = [_ja_mark_to_vowel(getattr(ph_intervals[i], 'mark', '')) for i in nuclei]
    aligned = _align_nuclei_positions_to_targets(nuclei, nucleus_vowels, target_vowels)
    if aligned and len(aligned) == target_n:
        selected = aligned
    elif target_n == 1:
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
        phones = [p for p in ph_intervals if float(p.minTime) >= s_t - 1e-6 and float(p.maxTime) <= e_t + 1e-6]
        if not phones:
            phones = [ph_intervals[n_idx]]
        tok = cv_targets[i] if i < len(cv_targets) else ''
        infos.append({'word': tok, 'roman': tok, 'start_time': s_t, 'end_time': e_t, 'phones': list(phones)})
    return infos if infos else None


def _score_ja_syllable_mapping(candidate_infos, cv_targets):
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

    def _to_ms(v):
        try:
            x = float(v)
        except Exception:
            return None
        # TextGrid sec 기반이면 ms로 변환
        return x * 1000.0 if abs(x) < 60.0 else x

    anchor_ms = []
    for s in candidate_infos:
        st = _to_ms(s.get('start_time')) if isinstance(s, dict) else None
        et = _to_ms(s.get('end_time')) if isinstance(s, dict) else None
        if st is None and et is None:
            anchor_ms.append(None)
        elif st is None:
            anchor_ms.append(et)
        elif et is None:
            anchor_ms.append(st)
        else:
            anchor_ms.append((st + et) * 0.5)

    def _avg_with_shift(shift):
        vals = []
        idxs = []
        local_drift_pen = 0.0
        for i, tgt in enumerate(targets):
            j = i + shift
            if 0 <= j < len(cand):
                cur = _vcv_syllable_match_score(tgt, cand[j])
                vals.append(cur)
                idxs.append(j)
                # 이웃보다 현재 점수가 많이 낮으면 한 칸 밀림 가능성이 높다.
                neighbor_best = cur
                if j > 0:
                    neighbor_best = max(neighbor_best, _vcv_syllable_match_score(tgt, cand[j - 1]))
                if (j + 1) < len(cand):
                    neighbor_best = max(neighbor_best, _vcv_syllable_match_score(tgt, cand[j + 1]))
                drift_gain = neighbor_best - cur
                if drift_gain > 10.0:
                    local_drift_pen += min((drift_gain - 10.0) * 0.35, 12.0)
        if not vals:
            return -1.0
        coverage = len(vals) / float(max(len(targets), 1))
        length_penalty = abs(len(cand) - len(targets)) * 4.0
        shift_penalty = abs(int(shift)) * 7.0

        # 순서 일관성 패널티: 시간축 역행/급점프를 감점.
        order_penalty = 0.0
        if idxs:
            prev_t = None
            for j in idxs:
                t = anchor_ms[j] if 0 <= j < len(anchor_ms) else None
                if t is None:
                    continue
                if prev_t is not None:
                    delta = t - prev_t
                    if delta <= 0.0:
                        order_penalty += 22.0
                    elif delta < 24.0:
                        order_penalty += 4.5
                    elif delta > 1500.0:
                        order_penalty += 6.0
                prev_t = t
        return (
            (sum(vals) / float(len(vals))) * coverage
            - length_penalty
            - shift_penalty
            - order_penalty
            - local_drift_pen
        )
    return max(_avg_with_shift(0), _avg_with_shift(1), _avg_with_shift(-1))


def _target_vowel_from_filename_syllable(syl):
    s = _normalize_ja_syllable_token(syl)
    if not s:
        return ''
    onset, vowel = split_ja_romaji_syllable(s)
    if vowel in JA_VOWELS:
        return vowel
    if s in {'n', 'nn', 'xn', 'm'}:
        return 'n'
    if s in JA_VOWELS and not onset:
        return s
    return ''


def _vowel_match_score(target_vowel, nucleus_vowel):
    tv = (target_vowel or '').strip().lower()
    nv = (nucleus_vowel or '').strip().lower()
    if not tv or not nv:
        return -1.2
    if tv == nv:
        return 4.0
    if {tv, nv} <= {'i', 'u'}:
        return 1.0
    if {tv, nv} <= {'e', 'i'}:
        return 0.6
    if {tv, nv} <= {'o', 'u'}:
        return 0.6
    if tv == 'n' or nv == 'n':
        return -2.0
    return -1.5


def _align_nuclei_positions_to_targets(nuclei, nucleus_vowels, target_vowels):
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


def _ja_mark_to_vowel(mark):
    clean = re.sub(r"[0-9]", "", (mark or "").strip().lower())
    if not clean:
        return ""
    if clean in {"n", "ɴ", "m", "nn", "xn", "ng", "ngy"}:
        return "n"
    if clean in {"a", "i", "e", "o"}:
        return clean
    if clean in {"u", "?", "?"}:
        return "u"
    if _is_ja_vowel_token(clean):
        onset, vowel = split_ja_romaji_syllable(clean)
        if vowel in JA_VOWELS:
            return vowel
        if clean in JA_VOWELS:
            return clean
    return ""


def _build_syllables_from_filename(ph_intervals, filename_syllables):
    if not ph_intervals or not filename_syllables:
        return None
    target_n = len(filename_syllables)
    nuclei = [i for i, p in enumerate(ph_intervals) if _is_nucleus_phone(p.mark)]
    if len(nuclei) < target_n:
        return None
    target_vowels = [_target_vowel_from_filename_syllable(s) for s in filename_syllables]
    nucleus_vowels = [_ja_mark_to_vowel(getattr(ph_intervals[i], 'mark', '')) for i in nuclei]
    aligned = _align_nuclei_positions_to_targets(nuclei, nucleus_vowels, target_vowels)
    if aligned and len(aligned) == target_n:
        selected = aligned
    elif target_n == 1:
        selected = [nuclei[0]]
    elif len(nuclei) == target_n:
        selected = list(nuclei)
    else:
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
        infos.append({'word': filename_syllables[i], 'start_time': phones[0].minTime, 'end_time': phones[-1].maxTime, 'phones': list(phones)})
        start_idx = n_idx + 1
    if infos and start_idx < len(ph_intervals):
        infos[-1]['phones'].extend(ph_intervals[start_idx:])
        infos[-1]['end_time'] = infos[-1]['phones'][-1].maxTime
    return infos if infos else None


@lru_cache(maxsize=131072)
def _extract_ja_cv_target_syllable(alias, alias_type='cv'):
    parts = (alias or '').strip().split()
    if not parts:
        return ''
    if alias_type in {'cv_head', 'vcv'} and len(parts) >= 2:
        return _normalize_ja_syllable_token(parts[1])
    if len(parts) >= 2 and parts[0] == '-':
        return _normalize_ja_syllable_token(parts[1])
    return _normalize_ja_syllable_token(parts[-1])


@lru_cache(maxsize=131072)
def _extract_ja_cv_target_raw(alias, alias_type="cv"):
    alias_type = str(alias_type or "").strip().lower()
    parts = (alias or "").strip().split()
    if not parts:
        return ""
    if alias_type in {"cv_head", "vcv"} and len(parts) >= 2:
        return parts[1]
    if len(parts) >= 2 and parts[0] == "-":
        return parts[1]
    return parts[-1]


def _build_ja_cvvc_occurrence_map(token_source):
    """
    Japanese CV/CV_HEAD aliases in CV/CVVC formats are mapped by token occurrence.

    This keeps repeated mora tokens (e.g. `do ... da ... do ... da`) stable even
    when nearby candidates have similar vowel/onset scores.
    """
    occ = {}
    for idx, raw in enumerate(token_source or []):
        tok = _normalize_ja_syllable_token(raw)
        if not tok:
            continue
        onset, vowel = split_ja_romaji_syllable(tok)
        if vowel not in JA_VOWELS and tok not in {"n", "nn", "xn", "m"}:
            continue
        occ.setdefault(tok, []).append(idx)
    return occ


def _resolve_ja_cvvc_occurrence_index(alias, alias_type, occurrence_map, occurrence_state):
    """
    Resolve deterministic CV/CV_HEAD index by token occurrence for Japanese CV/CVVC.
    """
    if alias_type not in {"cv", "cv_head"}:
        return None
    tok = _extract_ja_cv_target_syllable(alias, alias_type=alias_type)
    if not tok:
        return None
    idxs = (occurrence_map or {}).get(tok) or []
    if not idxs:
        return None
    if occurrence_state is None:
        return idxs[0]

    state_key = (str(alias_type), tok)
    used = int(occurrence_state.get(state_key, 0))
    if used >= len(idxs):
        used = len(idxs) - 1
    occurrence_state[state_key] = used + 1
    return idxs[used]


@lru_cache(maxsize=65536)
def _extract_ja_onset_token(token):
    t = _normalize_ja_syllable_token(token)
    if not t:
        return ''
    onset, vowel = split_ja_romaji_syllable(t)
    if vowel in JA_VOWELS:
        return (onset or '').lower()
    if t in JA_VOWELS:
        return ''
    return t


@lru_cache(maxsize=65536)
def _ja_syllable_tail(token):
    t = _normalize_ja_syllable_token(token)
    if not t:
        return ''
    onset, vowel = split_ja_romaji_syllable(t)
    if vowel in JA_VOWELS:
        core = f"{onset}{vowel}"
        if t.startswith(core):
            return t[len(core):]
    return ''


@lru_cache(maxsize=65536)
def _ja_onset_signature(onset):
    o = (onset or '').strip().lower()
    if not o:
        return ''
    out = []
    for idx, ch in enumerate(o):
        if ch in JA_VOWELS:
            continue
        if idx > 0 and ch in {'y', 'w'}:
            continue
        if out and out[-1] == ch:
            continue
        out.append(ch)
    return ''.join(out)


@lru_cache(maxsize=131072)
def _ja_soft_cv_match_level(target_token, candidate_token):
    t = _normalize_ja_syllable_token(target_token)
    c = _normalize_ja_syllable_token(candidate_token)
    if not t or not c:
        return 0
    if t == c:
        return 3

    to, tv = split_ja_romaji_syllable(t)
    co, cv = split_ja_romaji_syllable(c)
    if tv not in JA_VOWELS or cv != tv:
        return 0

    t_sig = _ja_onset_signature(to)
    c_sig = _ja_onset_signature(co)
    if t_sig and c_sig and t_sig == c_sig:
        t_has_glide = len(to) > 1 and any(ch in {'y', 'w'} for ch in to[1:])
        c_has_glide = len(co) > 1 and any(ch in {'y', 'w'} for ch in co[1:])
        t_tail = _ja_syllable_tail(t)
        c_tail = _ja_syllable_tail(c)
        if t_has_glide != c_has_glide:
            return 1
        if t_tail == c_tail:
            return 2
        if (not t_tail) or (not c_tail):
            return 1
    if to and co and to[:1] == co[:1]:
        return 1
    return 0


def _ja_onset_class(onset):
    o = (onset or '').strip().lower()
    if not o:
        return 'other'
    if o in JA_NASAL_ONSETS or o.startswith('m'):
        return 'nasal'
    if o in JA_VOICED_ONSETS:
        return 'voiced'
    if o in JA_VOICELESS_ONSETS:
        return 'voiceless'
    return 'other'


@lru_cache(maxsize=65536)
def _ja_special_mora_class(token):
    raw = str(token or "").strip().lower()
    norm = _normalize_ja_syllable_token(token)
    if not raw and not norm:
        return "plain"
    if norm in {"n", "nn", "xn", "m"} or raw in {"n", "nn", "xn", "m"}:
        return "nasal"
    if raw in {"q", "cl", "xtu", "xtsu", "ltsu", "ltu"} or norm in {"q", "cl", "xtu", "xtsu", "ltsu", "ltu"}:
        return "geminate"
    if raw in {"-", "ー", "long"}:
        return "long"
    onset, vowel = split_ja_romaji_syllable(norm)
    if onset and any(ch in {"y", "w"} for ch in onset[1:]):
        return "youon"
    if norm != raw and _ja_soft_cv_match_level(norm, raw) >= 2:
        return "inserted"
    if (not onset) and vowel in JA_VOWELS and raw.endswith(vowel * 2):
        return "long"
    return "plain"


def _select_vcv_syllable_index(alias, expected_idx, syllables_info):
    if not syllables_info:
        return 0
    n = len(syllables_info)
    e = max(0, min(int(expected_idx), n - 1))
    target = _extract_vcv_target_syllable(alias)
    if not target:
        return e
    start = e
    end = min(n, e + 3)
    best_idx = e
    best_score = -10**9
    best_key = (-10**9, -10**9, -10**9, -10**9)
    target_onset = _extract_ja_onset_token(target)
    _to, target_vowel = split_ja_romaji_syllable(target)
    target_tail = _ja_syllable_tail(target)
    target_cls = _ja_onset_class(target_onset)
    dist_penalty = 8 if target_cls in {'nasal', 'voiced'} else 7
    def _score_idx(i):
        cand = _syllable_info_token(syllables_info[i])
        cand_onset = _extract_ja_onset_token(cand)
        _co, cand_vowel = split_ja_romaji_syllable(cand)
        cand_tail = _ja_syllable_tail(cand)
        soft_match = _ja_soft_cv_match_level(target, cand)
        score = _vcv_syllable_match_score(target, cand) - abs(i - e) * dist_penalty
        if target_vowel and cand_vowel and target_vowel != cand_vowel:
            score -= 26
        if target_onset and cand_onset:
            if target_onset == cand_onset:
                score += 8
            elif target_onset[:1] == cand_onset[:1]:
                score += 3
            else:
                score -= 12
        if soft_match >= 2:
            score += 8
        elif soft_match == 1:
            score += 3
        if target_tail and cand_tail and target_tail != cand_tail:
            score -= 10
        elif (not target_tail) and cand_tail:
            score -= 10
        return score
    expected_score = _score_idx(e)
    for i in range(start, end):
        cand = _syllable_info_token(syllables_info[i])
        score = _score_idx(i)
        cand_norm = _normalize_ja_syllable_token(cand)
        key = (score, 1 if cand_norm == target else 0, -abs(i - e), 1 if i >= e else 0)
        if key > best_key:
            best_key = key
            best_score = score
            best_idx = i
    if best_score < 58:
        return e
    if best_idx <= e:
        return e
    if best_idx > (e + 1):
        return e
    best_gain = best_score - expected_score
    if best_gain < 20:
        return e
    # VCV는 한 칸 전진을 매우 보수적으로 허용한다.
    # 다음 칸이 "정확히 target 토큰"이고 현재 expected가 target이 아닐 때만 전진.
    cand_tok = _normalize_ja_syllable_token(_syllable_info_token(syllables_info[best_idx]))
    exp_tok = _normalize_ja_syllable_token(_syllable_info_token(syllables_info[e]))
    tgt_tok = _normalize_ja_syllable_token(target)
    if cand_tok != tgt_tok:
        return e
    if exp_tok == tgt_tok:
        return e
    if best_gain < 28:
        return e
    return best_idx


def _select_ja_cv_syllable_index(alias, expected_idx, syllables_info, alias_type='cv'):
    if not syllables_info:
        return 0
    n = len(syllables_info)
    e = max(0, min(int(expected_idx), n - 1))
    target = _extract_ja_cv_target_syllable(alias, alias_type=alias_type)
    target_raw = _extract_ja_cv_target_raw(alias, alias_type=alias_type)
    if not target:
        return e
    start = max(0, e - 1)
    end = min(n, e + 4)
    best_idx = e
    best_score = -10**9
    target_onset = _extract_ja_onset_token(target)
    _t_onset, target_vowel = split_ja_romaji_syllable(target)
    target_tail = _ja_syllable_tail(target)
    target_cls = _ja_onset_class(target_onset)
    target_special = _ja_special_mora_class(target)
    dist_penalty = 7 if target_cls in {'nasal', 'voiced'} else 6
    expected_soft_match = _ja_soft_cv_match_level(target, _syllable_info_token(syllables_info[e]))
    def _score_idx(i):
        cand = _syllable_info_token(syllables_info[i])
        cand_onset = _extract_ja_onset_token(cand)
        _co, cand_vowel = split_ja_romaji_syllable(cand)
        cand_tail = _ja_syllable_tail(cand)
        cand_cls = _ja_onset_class(cand_onset)
        cand_special = _ja_special_mora_class(cand)
        soft_match = _ja_soft_cv_match_level(target, cand)
        score = _vcv_syllable_match_score(target, cand) - abs(i - e) * dist_penalty
        if target_cls == 'nasal' and cand_cls != 'nasal':
            score -= 18
        elif target_cls == 'voiced' and cand_cls == 'voiceless':
            score -= 12
        elif target_cls == 'voiceless' and cand_cls == 'voiced':
            score -= 8
        if target_onset and cand_onset and target_onset[:1] == cand_onset[:1]:
            score += 4
        elif target_onset and cand_onset:
            score -= 9
        if target_vowel and cand_vowel and target_vowel != cand_vowel:
            score -= 24
        if soft_match >= 2:
            score += 18
        elif soft_match == 1:
            score += 6
        if soft_match > expected_soft_match:
            score += 10
        if not target_tail and cand_tail:
            if cand_tail[0] in {'i', 'u', 'e', 'o', 'a', 'y', 'w'}:
                score -= 20
            else:
                score -= 10
        elif target_tail and not cand_tail:
            score -= 8
        elif target_tail and cand_tail and target_tail != cand_tail:
            score -= 12
        if target_special in {"nasal", "geminate", "long"} and cand_special != target_special:
            score -= 28
        elif target_special == "plain" and cand_special in {"nasal", "geminate", "long"}:
            score -= 18
        elif target_special in {"youon", "inserted"}:
            if cand_special in {"youon", "inserted"}:
                score += 8
            elif soft_match < 2:
                score -= 16
        return score
    expected_score = _score_idx(e)
    best_key = (-10**9, -10**9, -10**9, -10**9)
    for i in range(start, end):
        score = _score_idx(i)
        cand_norm = _normalize_ja_syllable_token(_syllable_info_token(syllables_info[i]))
        key = (score, 1 if cand_norm == target else 0, -abs(i - e), 1 if i >= e else 0)
        if key > best_key:
            best_key = key
            best_score = score
            best_idx = i
    if best_score >= 54:
        hold_margin = 14 if target_cls in {'nasal', 'voiced'} else 16
        best_gain = best_score - expected_score
        expected_tok = _syllable_info_token(syllables_info[e])
        best_tok = _syllable_info_token(syllables_info[best_idx])
        _eo, expected_vowel = split_ja_romaji_syllable(expected_tok)
        _bo, best_vowel = split_ja_romaji_syllable(best_tok)
        expected_onset = _extract_ja_onset_token(expected_tok)
        best_onset = _extract_ja_onset_token(best_tok)
        expected_tail = _ja_syllable_tail(expected_tok)
        best_tail = _ja_syllable_tail(best_tok)
        expected_soft = _ja_soft_cv_match_level(target, expected_tok)
        best_soft = _ja_soft_cv_match_level(target, best_tok)
        expected_special = _ja_special_mora_class(expected_tok)
        best_special = _ja_special_mora_class(best_tok)
        if best_idx > e and best_gain < hold_margin:
            if not (best_idx == (e + 1) and best_soft >= 2 and best_soft > expected_soft):
                return e
        if target_vowel and best_vowel != target_vowel and expected_vowel == target_vowel:
            return e
        if target_onset and best_onset != target_onset and expected_onset == target_onset and best_gain < 24:
            if best_soft < 2:
                return e
        if (not target_tail) and best_tail and not expected_tail and best_gain < 22:
            return e
        if target_special in {"nasal", "geminate", "long"}:
            if best_special != target_special:
                return e
        elif target_special == "plain" and best_special in {"nasal", "geminate", "long"}:
            return e
        elif target_special in {"youon", "inserted"}:
            if expected_special in {"youon", "inserted"} and best_special not in {"youon", "inserted"}:
                return e
            target_strict = _normalize_ja_syllable_token_strict(target_raw or target)
            best_raw = _syllable_info_raw_token(syllables_info[best_idx]) or best_tok
            best_strict = _normalize_ja_syllable_token_strict(best_raw or best_tok)
            expected_raw = _syllable_info_raw_token(syllables_info[e]) or expected_tok
            expected_strict = _normalize_ja_syllable_token_strict(expected_raw or expected_tok)
            if (
                target_strict
                and best_strict
                and target_strict != best_strict
                and not (_is_kana_token(target_raw) or _is_kana_token(best_raw))
            ):
                return e
            if expected_strict == target_strict and best_strict != target_strict:
                return e
        return best_idx
    return e
