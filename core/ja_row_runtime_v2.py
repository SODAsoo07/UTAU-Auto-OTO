from __future__ import annotations


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
    aliases_to_write = generate_openutau_aliases_fn(alias) if generate_openutau else [alias]
    rows = []
    for a in aliases_to_write:
        a2 = alias_out_fn(a)
        rows.append(f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}")
    return rows


__all__ = [
    "build_ja_alias_output_rows",
    "build_ja_cv_guard_messages",
    "maybe_build_ja_realized_cv_anchor_record",
]
