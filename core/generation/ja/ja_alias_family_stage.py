from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JaAliasFamilyState:
    alias_type: str
    is_vc: bool
    is_vcv: bool
    is_cv_head: bool
    tail_breath: str


def build_ja_alias_family_state(
    *,
    alias: str,
    alias_type: str,
    format_type: str,
    is_vowel_token_fn,
    ja_consonants,
) -> JaAliasFamilyState:
    alias_type_norm = str(alias_type or "").strip().lower()
    format_norm = str(format_type or "").strip().lower()
    parts = str(alias or "").strip().split()

    if format_norm in {"cvvc", "cv"} and alias_type_norm == "vcv":
        if len(parts) >= 2:
            right = parts[1].strip().lower()
            alias_type_norm = "vv" if is_vowel_token_fn(right) else "vc"
        else:
            alias_type_norm = "cv"
    elif format_norm == "vcv" and alias_type_norm in {"vc", "vv"}:
        if alias_type_norm == "vv":
            alias_type_norm = "vcv"
        elif len(parts) >= 2 and is_vowel_token_fn(parts[0]):
            right = parts[1].strip().lower()
            if right not in ja_consonants and not is_vowel_token_fn(right):
                alias_type_norm = "vcv"

    tail_breath = ""
    if len(parts) >= 2 and is_vowel_token_fn(parts[0]) and parts[1].upper() in {"R", "H"}:
        tail_breath = parts[1].upper()

    return JaAliasFamilyState(
        alias_type=alias_type_norm,
        is_vc=alias_type_norm in {"vc", "vv"},
        is_vcv=alias_type_norm == "vcv",
        is_cv_head=alias_type_norm == "cv_head",
        tail_breath=tail_breath,
    )


def try_handle_ja_br_alias(
    *,
    alias: str,
    alias_type: str,
    ph_intervals,
    real_wav_name: str,
    final_lines,
    post_ctx,
    alias_out_fn,
    generate_openutau: bool,
    generate_openutau_aliases_fn,
) -> bool:
    if alias_type != "br":
        return False
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
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        alias_type="vc",
        alias_text=alias,
        local_end_ms=br_end,
        local_cut_allow_ms=40.0,
    )
    aliases_to_write = generate_openutau_aliases_fn(alias) if generate_openutau else [alias]
    for item in aliases_to_write:
        alias_out = alias_out_fn(item)
        final_lines.append(
            f"{real_wav_name}={alias_out},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
        )
    return True


def try_handle_ja_tail_breath_alias(
    *,
    alias: str,
    state: JaAliasFamilyState,
    current_w_idx: int,
    syllables_info,
    real_wav_name: str,
    final_lines,
    post_ctx,
    alias_out_fn,
    generate_openutau: bool,
    generate_openutau_aliases_fn,
) -> tuple[bool, int]:
    if not state.tail_breath:
        return False, current_w_idx
    if current_w_idx >= len(syllables_info):
        current_w_idx = len(syllables_info) - 1

    curr_syl = syllables_info[current_w_idx]
    curr_phones = curr_syl["phones"]
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
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        alias_type="mono",
        alias_text=alias,
        local_end_ms=v_end,
        local_cut_allow_ms=36.0,
    )
    aliases_to_write = generate_openutau_aliases_fn(alias) if generate_openutau else [alias]
    for item in aliases_to_write:
        alias_out = alias_out_fn(item)
        final_lines.append(
            f"{real_wav_name}={alias_out},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
        )
    return True, current_w_idx


__all__ = [
    "JaAliasFamilyState",
    "build_ja_alias_family_state",
    "try_handle_ja_br_alias",
    "try_handle_ja_tail_breath_alias",
]
