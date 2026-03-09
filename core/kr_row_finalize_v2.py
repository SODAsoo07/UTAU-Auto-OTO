from __future__ import annotations

from core.oto_row_finalize_v2 import commit_row_artifacts


def finalize_kr_row(
    *,
    final_lines,
    row_builder_fn,
    real_wav_name,
    alias,
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    generate_openutau,
    alias_suffix,
    alias_type,
    wav_duration_ms,
    validate_fn,
    log_post_timing_events_fn,
    log_fn,
    fname,
    soft_off_shift,
    soft_cut_shift,
    cutoff_reduced,
    anchor_store=None,
    anchor_record=None,
    messages=None,
):
    log_post_timing_events_fn(log_fn, fname, alias, soft_off_shift, soft_cut_shift, cutoff_reduced)
    rows = row_builder_fn(
        real_wav_name,
        alias,
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        generate_openutau=generate_openutau,
        alias_suffix=alias_suffix,
        alias_type=alias_type,
        wav_duration_ms=wav_duration_ms,
        validate_fn=validate_fn,
    )
    commit_row_artifacts(
        final_lines,
        rows,
        anchor_store=anchor_store,
        anchor_record=anchor_record,
        log_fn=log_fn,
        messages=messages,
    )


__all__ = ["finalize_kr_row"]
