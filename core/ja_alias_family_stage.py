from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


_JA_KANA_ONLY_RE = re.compile(r"^[\u3041-\u3096\u30A1-\u30FA\u30FC]+$")
_JA_TAIL_BREATH_HINTS = ("息", "吸", "吐", "breath", "inhale", "exhale", "asp", "air")
_JA_TAIL_BREATH_WORDS = {
    "r",
    "h",
    "息",
    "吸",
    "吐",
    "吸い",
    "吐き",
    "息継ぎ",
    "bre",
    "breath",
    "breathy",
    "inhale",
    "exhale",
    "asp",
    "aspirate",
    "air",
}


def _is_ja_tail_breath_marker(token: str) -> bool:
    raw = unicodedata.normalize("NFKC", str(token or "")).strip()
    if not raw:
        return False
    if raw.upper() in {"R", "H"}:
        return True
    low = raw.lower()
    if low in _JA_TAIL_BREATH_WORDS:
        return True
    return any(hint in low for hint in _JA_TAIL_BREATH_HINTS)


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
    if len(parts) >= 2:
        left = parts[0].strip()
        right = parts[-1].strip()
        if (is_vowel_token_fn(left) or bool(_JA_KANA_ONLY_RE.fullmatch(left))) and _is_ja_tail_breath_marker(right):
            tail_breath = right
    if not tail_breath:
        compact = re.sub(r"[\s_\-]+", "", str(alias or ""))
        if len(compact) >= 2 and compact[-1].upper() in {"R", "H"} and compact[:-1].lower() in {"a", "i", "u", "e", "o"}:
            tail_breath = compact[-1].upper()
    if not tail_breath and re.fullmatch(r"[\u3041-\u3096\u30A1-\u30FA\u30FC]+(?:息|吸|吐|吸い|吐き|息継ぎ)", str(alias or "")):
        tail_breath = "BREATH"

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
    curr_phones = list(curr_syl.get("phones") or [])
    if not curr_phones:
        return False, current_w_idx

    def _clean_mark(mark: str) -> str:
        return re.sub(r"[0-9]", "", str(mark or "").strip().lower()).replace(":", "")

    def _is_vowel_mark(mark: str) -> bool:
        m = _clean_mark(mark)
        return m in {"a", "i", "u", "e", "o", "ɯ", "ə", "ɨ", "ɴ", "n"}

    vowel_idx = -1
    for idx, ph in enumerate(curr_phones):
        if _is_vowel_mark(getattr(ph, "mark", "")):
            vowel_idx = idx
    if vowel_idx < 0:
        vowel_idx = max(len(curr_phones) - 1, 0)

    v_phone = curr_phones[vowel_idx]
    v_start = float(getattr(v_phone, "minTime", 0.0)) * 1000.0
    v_end = float(getattr(v_phone, "maxTime", 0.0)) * 1000.0
    v_len = max(v_end - v_start, 60.0)
    syllable_end = max(float(curr_syl.get("end_time", 0.0)) * 1000.0, float(getattr(curr_phones[-1], "maxTime", 0.0)) * 1000.0)

    breath_start = v_end
    if vowel_idx + 1 < len(curr_phones):
        next_phone = curr_phones[vowel_idx + 1]
        breath_start = max(v_end, float(getattr(next_phone, "minTime", 0.0)) * 1000.0)
    breath_end = max(syllable_end, float(getattr(curr_phones[-1], "maxTime", 0.0)) * 1000.0)

    if breath_end <= breath_start + 8.0:
        fallback_tail = max(min(v_len * 0.40, 100.0), 40.0)
        breath_start = max(v_end - min(fallback_tail * 0.45, 26.0), 0.0)
        breath_end = max(v_end + fallback_tail, breath_start + 40.0)

    breath_len = max(breath_end - breath_start, 35.0)
    lead = min(max(breath_len * 0.35, 18.0), 70.0)
    offset = max(breath_start - lead, 0.0)
    pre = max(breath_start - offset, 12.0)
    ovl = min(pre * 0.42, max(pre - 10.0, 0.0))
    consonant = pre + min(max(breath_len * 0.62, 36.0), 170.0)
    cutoff = -(consonant + min(max(breath_len * 0.78, 48.0), 240.0))
    offset, consonant, cutoff, pre, ovl = post_ctx.post_adjust(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        alias_type="vc",
        alias_text=alias,
        local_end_ms=breath_end,
        local_cut_allow_ms=92.0,
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
