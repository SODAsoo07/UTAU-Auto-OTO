from __future__ import annotations


def prepare_oto_alias_rows(
    real_wav_name,
    alias,
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    *,
    generate_aliases_fn,
    alias_transform_fn,
    generate_openutau=False,
    pre_write_adjust_fn=None,
):
    if pre_write_adjust_fn is not None:
        offset, consonant, cutoff, pre, ovl = pre_write_adjust_fn(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
        )

    aliases_to_write = generate_aliases_fn(alias) if generate_openutau else [alias]
    rows = []
    for alias_item in aliases_to_write:
        alias_out = alias_transform_fn(alias_item)
        rows.append(
            f"{real_wav_name}={alias_out},{float(offset):.2f},{float(consonant):.2f},{float(cutoff):.2f},{float(pre):.2f},{float(ovl):.2f}"
        )
    return (float(offset), float(consonant), float(cutoff), float(pre), float(ovl)), rows


__all__ = ["prepare_oto_alias_rows"]
