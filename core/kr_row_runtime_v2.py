from __future__ import annotations


def prepare_kr_cv_head_anchor_context(
    syllables_info,
    selected_w_idx,
    *,
    fallback_anchor_abs=None,
    prepare_cv_bounds_fn,
):
    context = {
        "anchor_abs": None if fallback_anchor_abs is None else float(fallback_anchor_abs),
        "onset_abs": None,
        "c_end_abs": None,
        "vowel_start_abs": None,
        "vowel_end_abs": None,
    }
    if selected_w_idx is None:
        return context
    try:
        _sw_idx, _sw_phones, c_start, c_end, n_start, n_end = prepare_cv_bounds_fn(
            syllables_info,
            selected_w_idx,
        )
    except Exception:
        return context
    context.update(
        {
            "anchor_abs": float(c_end),
            "onset_abs": float(c_start),
            "c_end_abs": float(c_end),
            "vowel_start_abs": float(n_start),
            "vowel_end_abs": float(n_end),
        }
    )
    return context


def maybe_build_kr_realized_cv_anchor_record(
    selected_w_idx,
    *,
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    onset_abs,
    vowel_start_abs,
    vowel_end_abs,
    c_end_abs=None,
    mel_voiced_onset_abs=None,
    build_anchor_fn,
):
    if selected_w_idx is None or onset_abs is None:
        return None
    return (
        int(selected_w_idx),
        build_anchor_fn(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            onset_abs=onset_abs,
            vowel_start_abs=vowel_start_abs,
            vowel_end_abs=vowel_end_abs,
            c_end_abs=c_end_abs,
            mel_voiced_onset_abs=mel_voiced_onset_abs,
        ),
    )


__all__ = [
    "maybe_build_kr_realized_cv_anchor_record",
    "prepare_kr_cv_head_anchor_context",
]
