from __future__ import annotations

from core.oto_row_finalize_v2 import commit_row_artifacts


def finalize_ja_row(
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
    generate_openutau_aliases_fn,
    alias_out_fn,
    anchor_store=None,
    anchor_record=None,
    log_fn=None,
    messages=None,
):
    rows = row_builder_fn(
        real_wav_name,
        alias,
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        generate_openutau=generate_openutau,
        generate_openutau_aliases_fn=generate_openutau_aliases_fn,
        alias_out_fn=alias_out_fn,
    )
    commit_row_artifacts(
        final_lines,
        rows,
        anchor_store=anchor_store,
        anchor_record=anchor_record,
        log_fn=log_fn,
        messages=messages,
    )


__all__ = ["finalize_ja_row"]
