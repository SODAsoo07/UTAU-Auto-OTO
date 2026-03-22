from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


_KR_KANA_ONLY_RE = re.compile(r"^[\u3041-\u3096\u30A1-\u30FA\u30FC]+$")
_KR_TAIL_BREATH_HINTS = ("숨", "흡", "호", "호기", "흡기", "breath", "inhale", "exhale", "asp", "air", "息", "吸", "吐")
_KR_TAIL_BREATH_WORDS = {
    "r",
    "h",
    "숨",
    "흡",
    "호",
    "호기",
    "흡기",
    "bre",
    "breath",
    "breathy",
    "inhale",
    "exhale",
    "asp",
    "aspirate",
    "air",
    "息",
    "吸",
    "吐",
    "吸い",
    "吐き",
    "息継ぎ",
}


def _is_kr_tail_breath_marker(token: str) -> bool:
    raw = unicodedata.normalize("NFKC", str(token or "")).strip()
    if not raw:
        return False
    if raw.upper() in {"R", "H"}:
        return True
    low = raw.lower()
    if low in _KR_TAIL_BREATH_WORDS:
        return True
    return any(hint in low for hint in _KR_TAIL_BREATH_HINTS)


@dataclass(frozen=True)
class KrAliasFamilyState:
    alias_type: str
    is_vc: bool
    is_vcv: bool
    is_cv_head: bool
    is_diph: bool
    target_clean: str
    glottal_kind: str
    breath_tail: str


def build_kr_alias_family_state(
    *,
    alias: str,
    alias_type: str,
    uses_vc_context_fn,
    is_diphthong_fn,
    extract_cv_alias_token_fn,
    detect_glottal_kind_fn,
    kr_vowels,
) -> KrAliasFamilyState:
    text = str(alias or "")
    alias_type_norm = str(alias_type or "").strip().lower()
    breath_tail = ""
    parts = text.split()
    if len(parts) >= 2:
        left = parts[0].strip()
        left_low = left.lower()
        right = parts[-1].strip()
        left_is_vowel_like = (
            left_low in kr_vowels
            or left_low in {"a", "e", "i", "o", "u"}
            or bool(_KR_KANA_ONLY_RE.fullmatch(left))
        )
        if left_is_vowel_like and _is_kr_tail_breath_marker(right):
            breath_tail = right
    if not breath_tail and text and text[-1].upper() in {"R", "H"} and text[:-1].lower() in kr_vowels:
        breath_tail = text[-1].upper()
    if not breath_tail and re.fullmatch(r"[\u3041-\u3096\u30A1-\u30FA\u30FC]+(?:息|吸|吐|吸い|吐き|息継ぎ)", text):
        breath_tail = "BREATH"

    return KrAliasFamilyState(
        alias_type=alias_type_norm,
        is_vc=bool(uses_vc_context_fn(alias_type_norm)),
        is_vcv=alias_type_norm == "vcv",
        is_cv_head=alias_type_norm == "cv_head",
        is_diph=bool(is_diphthong_fn(text)),
        target_clean=str(extract_cv_alias_token_fn(text) or ""),
        glottal_kind=str(detect_glottal_kind_fn(text) or ""),
        breath_tail=breath_tail,
    )


def try_handle_kr_glottal_alias(
    *,
    alias: str,
    state: KrAliasFamilyState,
    current_w_idx: int,
    cv_seq_idx: int,
    syllables_info,
    ph_intervals,
    real_wav_name: str,
    alias_suffix: str,
    final_lines,
    validate_fn,
    apply_alias_suffix_fn,
    find_vowel_phone_fn,
    fit_to_wav_fn=None,
    wav_duration_ms: float = 0.0,
) -> tuple[bool, int, int]:
    if state.glottal_kind not in {"head", "tail"}:
        return False, current_w_idx, cv_seq_idx

    if state.glottal_kind == "head":
        if cv_seq_idx < len(syllables_info):
            current_w_idx = cv_seq_idx
            cv_seq_idx = current_w_idx + 1

    if current_w_idx >= len(syllables_info):
        current_w_idx = len(syllables_info) - 1

    curr_syl = syllables_info[current_w_idx]
    curr_phones = curr_syl["phones"]
    _, v_phone = find_vowel_phone_fn(curr_phones)
    vowel_start = v_phone.minTime * 1000
    vowel_end = v_phone.maxTime * 1000

    g_idx = None
    for idx, phone in enumerate(ph_intervals):
        if abs(phone.minTime - v_phone.minTime) < 1e-6 and abs(phone.maxTime - v_phone.maxTime) < 1e-6:
            g_idx = idx
            break

    if state.glottal_kind == "tail":
        next_ph = ph_intervals[g_idx + 1] if g_idx is not None and g_idx + 1 < len(ph_intervals) else None
        boundary_end = (next_ph.maxTime * 1000) if next_ph else (curr_syl["end_time"] * 1000)
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

    try:
        offset, consonant, cutoff, pre, ovl = validate_fn(
            offset, consonant, cutoff, pre, ovl, alias_type=getattr(state, "alias_type", "cv")
        )
    except TypeError:
        offset, consonant, cutoff, pre, ovl = validate_fn(offset, consonant, cutoff, pre, ovl)
    if callable(fit_to_wav_fn):
        offset, consonant, cutoff, pre, ovl, _ = fit_to_wav_fn(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            wav_duration_ms,
            alias_type=getattr(state, "alias_type", "cv"),
            validate_fn=validate_fn,
        )
    alias_out = apply_alias_suffix_fn(alias, alias_suffix)
    final_lines.append(
        f"{real_wav_name}={alias_out},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
    )
    return True, current_w_idx, cv_seq_idx


def try_handle_kr_breath_tail_alias(
    *,
    alias: str,
    state: KrAliasFamilyState,
    current_w_idx: int,
    syllables_info,
    ph_intervals_all,
    real_wav_name: str,
    alias_suffix: str,
    final_lines,
    validate_fn,
    apply_alias_suffix_fn,
    find_vowel_phone_fn,
    fit_to_wav_fn=None,
    wav_duration_ms: float = 0.0,
) -> tuple[bool, int]:
    if not state.breath_tail:
        return False, current_w_idx

    if current_w_idx >= len(syllables_info):
        current_w_idx = len(syllables_info) - 1
    curr_syl = syllables_info[current_w_idx]
    curr_phones = list(curr_syl.get("phones") or [])
    if not curr_phones:
        return False, current_w_idx
    _, v_phone = find_vowel_phone_fn(curr_phones)
    vowel_start = v_phone.minTime * 1000
    vowel_end = v_phone.maxTime * 1000
    vowel_len = max(vowel_end - vowel_start, 60)

    all_phones = list(ph_intervals_all or [])
    last_end = (all_phones[-1].maxTime * 1000) if all_phones else max(curr_syl.get("end_time", 0.0) * 1000, curr_phones[-1].maxTime * 1000)
    breath_start = vowel_end
    for ph in all_phones:
        st = float(getattr(ph, "minTime", 0.0)) * 1000.0
        en = float(getattr(ph, "maxTime", 0.0)) * 1000.0
        if st >= (vowel_end - 1.0) and en > (vowel_end + 2.0):
            breath_start = max(vowel_end, st)
            break
    if last_end <= breath_start + 8.0:
        fallback_tail = max(min(vowel_len * 0.42, 120.0), 45.0)
        breath_start = max(vowel_end - min(fallback_tail * 0.42, 26.0), 0.0)
        last_end = max(vowel_end + fallback_tail, breath_start + 42.0)

    breath_len = max(last_end - breath_start, 40.0)
    lead = min(max(breath_len * 0.34, 18.0), 74.0)
    offset = max(breath_start - lead, 0.0)
    pre = max(breath_start - offset, 12.0)
    ovl = min(pre * 0.45, max(pre - 10.0, 0.0))
    consonant = pre + min(max(breath_len * 0.64, 40.0), 200.0)
    cutoff = -(consonant + min(max(breath_len * 0.82, 58.0), 280.0))
    try:
        offset, consonant, cutoff, pre, ovl = validate_fn(
            offset, consonant, cutoff, pre, ovl, alias_type=state.alias_type
        )
    except TypeError:
        offset, consonant, cutoff, pre, ovl = validate_fn(offset, consonant, cutoff, pre, ovl)
    if callable(fit_to_wav_fn):
        offset, consonant, cutoff, pre, ovl, _ = fit_to_wav_fn(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            wav_duration_ms,
            alias_type=state.alias_type,
            validate_fn=validate_fn,
        )
    alias_out = apply_alias_suffix_fn(alias, alias_suffix)
    final_lines.append(
        f"{real_wav_name}={alias_out},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
    )
    return True, current_w_idx


__all__ = [
    "KrAliasFamilyState",
    "build_kr_alias_family_state",
    "try_handle_kr_breath_tail_alias",
    "try_handle_kr_glottal_alias",
]
