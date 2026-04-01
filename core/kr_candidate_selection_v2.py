from __future__ import annotations


def select_kr_syllable_source(
    *,
    file_format,
    prefer_filename_sequence,
    low_phone_quality,
    used_words_based,
    words_syllables,
    alias_syllables,
    targets_for_build,
    score_mapping_fn,
    should_prefer_alias_fn,
    compute_glide_mismatch_fn,
    is_order_locked_format_fn,
):
    words_infos = list(words_syllables or [])
    alias_infos = list(alias_syllables or []) if alias_syllables else None
    fmt = str(file_format or "").strip().lower()

    result = {
        "syllables_info": words_infos,
        "used_words_based": bool(used_words_based and bool(words_infos)),
        "used_alias_based": False,
        "mapping_reason_code": "words_keep",
        "base_score": 0.0,
        "alt_score": 0.0,
        "words_glide_mismatch_ratio": 0.0,
    }

    if prefer_filename_sequence and alias_infos:
        result.update(
            {
                "syllables_info": alias_infos,
                "used_words_based": False,
                "used_alias_based": True,
                "mapping_reason_code": "filename_sequence_lock",
            }
        )
        return result

    if words_infos and any(len(s.get("phones") or []) == 0 for s in words_infos) and alias_infos:
        result.update(
            {
                "syllables_info": alias_infos,
                "used_words_based": False,
                "used_alias_based": True,
                "mapping_reason_code": "alias_based_empty_words",
            }
        )
        return result

    if not words_infos and alias_infos:
        result.update(
            {
                "syllables_info": alias_infos,
                "used_words_based": False,
                "used_alias_based": True,
                "mapping_reason_code": "alias_phone_minimal",
            }
        )
        return result

    if not (words_infos and alias_infos and targets_for_build):
        return result

    base_score = float(score_mapping_fn(words_infos, targets_for_build))
    alt_score = float(score_mapping_fn(alias_infos, targets_for_build))
    words_glide_mismatch_ratio = 0.0
    force_alias_based_reason = ""
    if is_order_locked_format_fn(fmt):
        words_glide_mismatch_ratio = float(compute_glide_mismatch_fn(words_infos, targets_for_build))
        words_len = len(words_infos or [])
        target_len = len(targets_for_build or [])
        if words_len and target_len and words_len != target_len:
            force_alias_based_reason = "order_locked_length_mismatch"
        elif words_glide_mismatch_ratio >= 0.34 and max(words_len, target_len) >= 3:
            force_alias_based_reason = "order_locked_glide_mismatch"
        elif low_phone_quality and used_words_based:
            force_alias_based_reason = "order_locked_low_phone_quality"

    result["base_score"] = base_score
    result["alt_score"] = alt_score
    result["words_glide_mismatch_ratio"] = words_glide_mismatch_ratio

    if force_alias_based_reason:
        result.update(
            {
                "syllables_info": alias_infos,
                "used_words_based": False,
                "used_alias_based": True,
                "mapping_reason_code": force_alias_based_reason,
            }
        )
        return result

    if low_phone_quality and used_words_based:
        result["mapping_reason_code"] = "words_low_phone_quality"
        return result

    if should_prefer_alias_fn(fmt, used_words_based, base_score, alt_score):
        result.update(
            {
                "syllables_info": alias_infos,
                "used_alias_based": True,
                "mapping_reason_code": ("alias_based_cvvc" if fmt == "cvvc" else "alias_based_recover"),
            }
        )
        return result

    if fmt != "cvvc" and used_words_based and base_score >= 58.0 and alt_score >= (base_score + 8.0):
        result["mapping_reason_code"] = "words_keep_high_conf"
        return result

    result["mapping_reason_code"] = "words_keep"
    return result


__all__ = ["select_kr_syllable_source"]
