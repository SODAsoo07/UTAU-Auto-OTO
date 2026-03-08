from __future__ import annotations

from core.ja_oto_postprocess import JaPostprocessContext


def build_ja_postprocess_context(
    *,
    phone_spans_ms,
    timeline_start_ms: float,
    effective_end_ms: float,
    validate_fn,
    recenter_fn,
    extract_cv_bounds_fn,
    cv_onset_class_fn,
    syllables_info,
    ja_style_enabled: bool,
    ja_style_profile,
    autotune_profile,
    style_apply_fn,
    autotune_apply_fn,
):
    return JaPostprocessContext(
        phone_spans_ms=phone_spans_ms,
        timeline_start_ms=timeline_start_ms,
        effective_end_ms=effective_end_ms,
        validate_fn=validate_fn,
        recenter_fn=recenter_fn,
        extract_cv_bounds_fn=extract_cv_bounds_fn,
        cv_onset_class_fn=cv_onset_class_fn,
        syllables_info=syllables_info,
        ja_style_enabled=ja_style_enabled,
        ja_style_profile=ja_style_profile,
        autotune_profile=autotune_profile,
        style_apply_fn=style_apply_fn,
        autotune_apply_fn=autotune_apply_fn,
    )


def try_handle_ja_single_vowel_file(
    *,
    fname: str,
    lines,
    real_wav_name: str,
    ph_intervals,
    final_lines,
    log_fn,
    record_unset_fn,
    post_ctx,
    alias_out_fn,
    apply_suffix_to_oto_line_fn,
    generate_openutau: bool,
    generate_openutau_aliases_fn,
    alias_suffix: str,
) -> bool:
    if len(ph_intervals) != 1:
        return False

    log_fn(f"🎵 {fname}: 단모음 파일")
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
        offset = v_start
        pre = 0
        ovl = 0
        consonant = min(v_len * 0.25, 120)
        cutoff = -(v_len * 0.8)
        offset, consonant, cutoff, pre, ovl = post_ctx.post_adjust(
            offset, consonant, cutoff, pre, ovl, alias_type="br", alias_text=alias
        )
        aliases_to_write = generate_openutau_aliases_fn(alias) if generate_openutau else [alias]
        for item in aliases_to_write:
            alias_text = alias_out_fn(item)
            final_lines.append(
                f"{real_wav_name}={alias_text},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
            )
    return True


__all__ = [
    "build_ja_postprocess_context",
    "try_handle_ja_single_vowel_file",
]
