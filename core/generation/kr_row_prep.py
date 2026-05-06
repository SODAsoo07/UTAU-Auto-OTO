from __future__ import annotations

from collections.abc import Callable, MutableSequence, Sequence
from typing import Any, Dict


def prepare_kr_alias_row_start(
    *,
    line: str,
    fname: str,
    real_wav_name: str,
    alias_suffix: str,
    final_lines: MutableSequence[str],
    mapping_confidence_base: float,
    record_unset_fn: Callable[..., None],
    apply_suffix_to_oto_line_fn: Callable[[str, str], str],
    extract_base_timing_shape_fn: Callable[[str], Any],
    classify_alias_fn: Callable[[str], str],
) -> Dict[str, Any]:
    parts = line.split("=", 1)
    if len(parts) < 2:
        record_unset_fn("malformed_line", fname, line)
        final_lines.append(apply_suffix_to_oto_line_fn(line, alias_suffix))
        return {"should_continue": True}

    alias = parts[1].split(",", 1)[0].strip()
    if not alias:
        record_unset_fn("empty_alias", fname, line)
        preserved = f"{real_wav_name}={parts[1]}"
        final_lines.append(apply_suffix_to_oto_line_fn(preserved, alias_suffix))
        return {"should_continue": True}

    return {
        "should_continue": False,
        "alias": alias,
        "base_shape": extract_base_timing_shape_fn(line),
        "row_mapping_confidence": float(mapping_confidence_base),
        "row_jump_blocked": 0,
        "row_apply_mode": "full_apply",
        "row_apply_reason_code": "",
        "alias_type": classify_alias_fn(alias),
    }


def resolve_kr_row_jump_limits(
    *,
    alias_type: str,
    mapping_only_enabled: bool,
    order_locked_format: bool,
    textgrid_trust_tier: str,
    file_mapping_low_conf: bool,
    max_index_jump_default: int,
    max_index_jump_high_conf: int,
) -> Dict[str, int]:
    row_jump_default = int(max(0, max_index_jump_default))
    row_jump_high_conf = int(max(row_jump_default, max_index_jump_high_conf))

    if mapping_only_enabled and alias_type in {"cv", "cv_head", "vcv", "vv", "mono"}:
        row_jump_default = 0
        row_jump_high_conf = 0

    if order_locked_format and textgrid_trust_tier == "low":
        row_jump_default = 0
        row_jump_high_conf = max(0, min(row_jump_high_conf, 1))
    elif order_locked_format and textgrid_trust_tier == "mid":
        row_jump_high_conf = max(row_jump_default, min(row_jump_high_conf, 1))

    if file_mapping_low_conf:
        row_jump_high_conf = int(max(0, min(row_jump_high_conf, row_jump_default)))
        if order_locked_format:
            row_jump_default = 0
        else:
            row_jump_default = int(max(0, row_jump_default - 1))

    if alias_type == "cv_head":
        row_jump_default = int(max(0, row_jump_default - 1))
        row_jump_high_conf = int(max(row_jump_default, row_jump_high_conf - 1))
    elif alias_type == "vv":
        row_jump_default = int(max(row_jump_default, max_index_jump_default + 1))
        row_jump_high_conf = int(max(row_jump_high_conf, row_jump_default + 1))

    return {
        "row_jump_default": int(row_jump_default),
        "row_jump_high_conf": int(row_jump_high_conf),
    }


def try_append_kr_br_alias_row(
    *,
    alias_type: str,
    ph_intervals: Sequence[Any],
    final_lines: MutableSequence[str],
    real_wav_name: str,
    alias: str,
    generate_openutau: bool,
    alias_suffix: str,
    wav_duration_ms: float,
    validate_fn: Callable[..., tuple[Any, Any, Any, Any, Any]],
    append_alias_rows_fn: Callable[..., Any],
) -> bool:
    if alias_type != "br":
        return False

    first_ph = ph_intervals[0] if ph_intervals else None
    last_ph = ph_intervals[-1] if ph_intervals else None
    if first_ph and last_ph:
        br_start = float(first_ph.minTime) * 1000.0
        br_end = float(last_ph.maxTime) * 1000.0
        br_len = br_end - br_start
    else:
        br_start = 0.0
        br_len = 500.0

    offset = max(br_start - 30.0, 0.0)
    pre = 0.0
    ovl = 0.0
    consonant = min(br_len * 0.3, 100.0)
    cutoff = -(br_len * 0.85)
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
        alias_type=alias_type,
        wav_duration_ms=wav_duration_ms,
        validate_fn=validate_fn,
    )
    return True


__all__ = [
    "prepare_kr_alias_row_start",
    "resolve_kr_row_jump_limits",
    "try_append_kr_br_alias_row",
]
