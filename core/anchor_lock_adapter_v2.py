from __future__ import annotations

from core.timing_anchor_runtime import AnchorTimingContext, apply_anchor_lock


def apply_language_anchor_lock(
    *,
    language,
    format_type,
    alias_type,
    before,
    file_duration_ms,
    timeline_start_ms,
    timeline_end_ms,
    anchor_abs_ms,
    next_onset_abs_ms=None,
    next_vowel_abs_ms=None,
    mapping_confidence=1.0,
    validate_fn,
    lite=False,
    is_enabled_fn,
    get_profile_fn,
    retune_profile_fn=None,
    apply_stats_delta_fn=None,
    build_log_record_fn=None,
    append_log_fn=None,
    log_path="",
):
    lang = str(language or "").strip().lower()
    fmt = str(format_type or "").strip().lower()
    alias_kind = str(alias_type or "").strip().lower()
    params = tuple(float(v) for v in before)

    if not is_enabled_fn(lang, fmt):
        return params

    profile = get_profile_fn(lang, fmt, alias_kind)
    if profile is None:
        return params
    if retune_profile_fn is not None:
        profile = retune_profile_fn(profile)
        if profile is None:
            return params

    ctx = AnchorTimingContext(
        file_duration_ms=float(file_duration_ms or 0.0),
        timeline_start_ms=float(timeline_start_ms or 0.0),
        timeline_end_ms=float(timeline_end_ms or 0.0),
        anchor_abs_ms=float(anchor_abs_ms) if anchor_abs_ms is not None else None,
        next_onset_abs_ms=float(next_onset_abs_ms) if next_onset_abs_ms is not None else None,
        next_vowel_abs_ms=float(next_vowel_abs_ms) if next_vowel_abs_ms is not None else None,
        alias_type=alias_kind,
        language=lang,
        format_type=fmt,
        mapping_confidence=float(mapping_confidence or 1.0),
    )
    result = apply_anchor_lock(
        params,
        ctx,
        profile,
        validate_fn=validate_fn,
        lite=bool(lite),
    )

    if apply_stats_delta_fn is not None:
        apply_stats_delta_fn(alias_kind, getattr(result, "applied_rules", ()))

    if build_log_record_fn is not None and append_log_fn is not None:
        log_record = build_log_record_fn(fmt, alias_kind, params, result)
        if log_record is not None:
            append_log_fn(log_path, log_record)

    return (
        float(result.offset),
        float(result.consonant),
        float(result.cutoff),
        float(result.pre),
        float(result.ovl),
    )


__all__ = ["apply_language_anchor_lock"]
