from __future__ import annotations

from core.oto_row_output_v2 import prepare_oto_alias_rows


def maybe_build_ja_realized_cv_anchor_record(
    current_w_idx,
    *,
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    onset_abs,
    vowel_start_abs,
    c_end_abs,
    vowel_end_abs,
    mel_voiced_onset_abs=None,
    build_anchor_fn,
):
    if current_w_idx is None:
        return None
    return (
        int(current_w_idx),
        build_anchor_fn(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            onset_abs=onset_abs,
            vowel_start_abs=vowel_start_abs,
            c_end_abs=c_end_abs,
            vowel_end_abs=vowel_end_abs,
            mel_voiced_onset_abs=mel_voiced_onset_abs,
        ),
    )


def build_ja_cv_guard_messages(
    fname,
    alias,
    *,
    cutoff_reduced=0.0,
    offset_reduced=0.0,
    cutoff_extended=0.0,
):
    messages = []
    if float(offset_reduced or 0.0) > 1.0:
        messages.append(f"🛡️ {fname}: CV_HEAD 오프셋 과선행 보정(+{float(offset_reduced):.1f}ms) [{alias}]")
    if float(cutoff_extended or 0.0) > 1.0:
        messages.append(f"🛡️ {fname}: CV_HEAD 모음 길이 보정(+{float(cutoff_extended):.1f}ms) [{alias}]")
    if float(cutoff_reduced or 0.0) > 0.5:
        messages.append(f"🛡️ {fname}: CV 컷오프 과연장 보정(-{float(cutoff_reduced):.1f}ms) [{alias}]")
    return messages


def build_ja_alias_output_rows(
    real_wav_name,
    alias,
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    *,
    generate_openutau,
    generate_openutau_aliases_fn,
    alias_out_fn,
):
    _params, rows = prepare_oto_alias_rows(
        real_wav_name,
        alias,
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        generate_openutau=generate_openutau,
        generate_aliases_fn=generate_openutau_aliases_fn,
        alias_transform_fn=alias_out_fn,
    )
    return rows


__all__ = [
    "build_ja_alias_output_rows",
    "build_ja_cv_guard_messages",
    "maybe_build_ja_realized_cv_anchor_record",
]
