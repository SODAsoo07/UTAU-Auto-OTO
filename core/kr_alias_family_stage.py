from __future__ import annotations

from dataclasses import dataclass


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
    if len(parts) >= 2 and parts[0].lower() in kr_vowels and parts[1].upper() in {"R", "H"}:
        breath_tail = parts[1].upper()
    elif text and text[-1].upper() in {"R", "H"} and text[:-1].lower() in kr_vowels:
        breath_tail = text[-1].upper()

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
    curr_phones = curr_syl["phones"]
    _, v_phone = find_vowel_phone_fn(curr_phones)
    vowel_start = v_phone.minTime * 1000
    vowel_end = v_phone.maxTime * 1000
    vowel_len = max(vowel_end - vowel_start, 80)
    last_end = (ph_intervals_all[-1].maxTime * 1000) if ph_intervals_all else (curr_syl["end_time"] * 1000)

    offset_padding = min(max(vowel_len * 0.7, 180), 320)
    offset = max(vowel_end - offset_padding, 0)
    pre_abs = max(vowel_end - 20, offset)
    pre = pre_abs - offset
    ovl = pre * 0.85
    consonant = max(vowel_end - offset, pre + 10)
    cutoff = -(max(last_end - offset, consonant + 80))
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
