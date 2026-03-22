from __future__ import annotations

from core.kr_oto_postprocess import KrPostprocessContext
from core.kr_oto_rules import classify_alias
from core.kr_oto_vv import _compute_kr_noninitial_vowel_timing


def build_kr_postprocess_context(
    *,
    file_format: str,
    mel_ctx_for_file,
    ph_intervals,
    syllables_info,
    validate_fn,
    soft_mel_guard_fn,
    base_shape_blend_fn,
    stabilize_fn,
    recenter_fn,
    cv_cutoff_guard_fn,
):
    return KrPostprocessContext(
        file_format=file_format,
        mel_ctx_for_file=mel_ctx_for_file,
        ph_intervals=ph_intervals,
        syllables_info=syllables_info,
        validate_fn=validate_fn,
        soft_mel_guard_fn=soft_mel_guard_fn,
        base_shape_blend_fn=base_shape_blend_fn,
        stabilize_fn=stabilize_fn,
        recenter_fn=recenter_fn,
        cv_cutoff_guard_fn=cv_cutoff_guard_fn,
    )


def try_handle_kr_single_vowel_file(
    *,
    fname: str,
    lines,
    real_wav_name: str,
    ph_intervals,
    wd_intervals,
    final_lines,
    log_fn,
    record_unset_fn,
    validate_fn,
    append_alias_rows_fn,
    apply_suffix_to_oto_line_fn,
    generate_openutau: bool,
    alias_suffix: str,
) -> bool:
    if not (len(ph_intervals) == 1 and len(wd_intervals) == 1):
        return False

    log_fn(f"처리: {fname}: 단모음 파일")
    vowel = ph_intervals[0]
    v_start = vowel.minTime * 1000
    v_end = vowel.maxTime * 1000
    v_len = v_end - v_start
    for line in lines:
        parts = line.split("=", 1)
        if len(parts) < 2:
            record_unset_fn("malformed_line", fname, line)
            final_lines.append(apply_suffix_to_oto_line_fn(line, alias_suffix))
            continue
        alias = parts[1].split(",", 1)[0].strip()
        if not alias:
            record_unset_fn("empty_alias", fname, line)
            preserved = f"{real_wav_name}={parts[1]}"
            final_lines.append(apply_suffix_to_oto_line_fn(preserved, alias_suffix))
            continue

        alias_type = classify_alias(alias)
        if alias_type == "mono":
            offset, consonant, cutoff, pre, ovl = _compute_kr_noninitial_vowel_timing(v_start, v_end)
        else:
            offset = v_start
            pre = 0
            ovl = 0
            consonant = min(v_len * 0.25, 120)
            cutoff = -(v_len * 0.8)
        offset, consonant, cutoff, pre, ovl = validate_fn(offset, consonant, cutoff, pre, ovl)
        append_alias_rows_fn(
            final_lines,
            real_wav_name,
            alias,
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            generate_openutau=generate_openutau,
            alias_suffix=alias_suffix,
        )
    return True


__all__ = [
    "build_kr_postprocess_context",
    "try_handle_kr_single_vowel_file",
]
